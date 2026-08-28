"""Full pipeline format benchmark.

Simulates the call_endpoint post-cache path:
  sort → fields projection → paginate → format switch → envelope → compact_json

Measures final wire tokens (tiktoken cl100k_base) for representative
biomedical endpoint shapes across format × fields × size combinations.

Outputs: benchmarks/formats-YYYY-MM-DD.md
"""

import json
import random
import sys
from datetime import date
from pathlib import Path

import tiktoken

# Load modules directly by file path to avoid server __init__.py (requires env vars)
import importlib.util

_src = Path(__file__).resolve().parent.parent / "src" / "indra_agent" / "mcp_server"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


formats = _load("formats", _src / "formats.py")
stable_key_union = formats.stable_key_union
to_toon = formats.to_toon
is_heterogeneous = formats.is_heterogeneous

pagination = _load("pagination", _src / "pagination.py")
paginate_response = pagination.paginate_response
estimate_tokens = pagination.estimate_tokens
estimate_tokens_str = pagination.estimate_tokens_str
TOON_CHARS_PER_TOKEN = pagination.TOON_CHARS_PER_TOKEN

enc = tiktoken.get_encoding("cl100k_base")


def compact_json(obj):
    return json.dumps(obj, separators=(",", ":"), default=str)


def count_tokens(s: str) -> int:
    return len(enc.encode(s))


# --- Payload generators (same as pre-flight but more variety) ---

random.seed(42)

GENE_NAMES = [
    "LRRK2", "TP53", "BRCA1", "EGFR", "BRAF", "KRAS", "MYC", "AKT1",
    "PIK3CA", "PTEN", "CDH1", "RB1", "ERBB2", "JAK2", "FGFR3", "ALK",
    "IDH1", "NOTCH1", "SMAD4", "VHL", "NPM1", "FLT3", "KIT", "NRAS",
    "ABL1", "PDGFRA", "MET", "RET", "CTNNB1", "ATM",
]

DISEASE_NAMES = [
    "Parkinson disease", "breast cancer", "lung adenocarcinoma",
    "glioblastoma", "acute myeloid leukemia", "colorectal cancer",
    "hepatocellular carcinoma", "melanoma", "prostate cancer",
    "ovarian cancer", "pancreatic cancer", "non-small cell lung cancer",
]

DRUG_NAMES = [
    "imatinib", "trastuzumab", "pembrolizumab", "osimertinib",
    "venetoclax", "olaparib", "ruxolitinib", "dabrafenib",
    "nivolumab", "erlotinib", "sorafenib", "lenvatinib",
]

STMT_TYPES = [
    "Activation", "Inhibition", "IncreaseAmount", "DecreaseAmount",
    "Phosphorylation", "Complex", "Dephosphorylation",
]


def make_gene(i):
    name = GENE_NAMES[i % len(GENE_NAMES)]
    if i >= len(GENE_NAMES):
        name += f"L{i}"
    return {
        "db_ns": "HGNC", "db_id": str(6000 + i), "name": name,
        "evidence_count": random.randint(1, 500),
        "source_counts": {"reach": random.randint(0, 200), "bel": random.randint(0, 100),
                          "sparser": random.randint(0, 50), "signor": random.randint(0, 30)},
    }


def make_disease(i):
    return {
        "db_ns": "MESH", "db_id": f"D{50000 + i:06d}",
        "name": DISEASE_NAMES[i % len(DISEASE_NAMES)],
        "evidence_count": random.randint(1, 300),
        "source_counts": {"reach": random.randint(0, 150), "bel": random.randint(0, 80)},
    }


def make_drug(i):
    return {
        "db_ns": "CHEBI", "db_id": str(40000 + i),
        "name": DRUG_NAMES[i % len(DRUG_NAMES)],
        "evidence_count": random.randint(1, 200),
        "source_counts": {"reach": random.randint(0, 100), "bel": random.randint(0, 60),
                          "drugbank": random.randint(0, 20)},
    }


def make_statement(i):
    return {
        "stmt_hash": random.randint(10**17, 10**18),
        "stmt_type": random.choice(STMT_TYPES),
        "subj_ns": "HGNC", "subj_id": str(6000 + i), "subj_name": f"GENE{i}",
        "obj_ns": "HGNC", "obj_id": str(7000 + i), "obj_name": f"GENE{i + 500}",
        "evidence_count": random.randint(1, 100),
        "source_counts": {"reach": random.randint(0, 50), "bel": random.randint(0, 30),
                          "signor": random.randint(0, 10)},
        "belief": round(random.uniform(0.5, 1.0), 4),
    }


def make_pathway(i):
    pathways = [
        "MAPK signaling pathway", "PI3K-Akt signaling pathway",
        "Wnt signaling pathway", "Notch signaling pathway",
        "JAK-STAT signaling pathway", "NF-kappa B signaling pathway",
        "p53 signaling pathway", "mTOR signaling pathway",
    ]
    return {
        "db_ns": "FPLX", "db_id": f"PW{i:04d}",
        "name": pathways[i % len(pathways)],
        "evidence_count": random.randint(1, 150),
    }


SHAPES = {
    "gene (6 keys)": (make_gene, ["db_ns", "db_id", "name"]),
    "disease (5 keys)": (make_disease, ["db_ns", "db_id", "name"]),
    "drug (6 keys)": (make_drug, ["db_ns", "db_id", "name"]),
    "statement (11 keys)": (make_statement, ["stmt_type", "subj_name", "obj_name", "evidence_count", "belief"]),
    "pathway (4 keys)": (make_pathway, ["db_ns", "db_id", "name"]),
}

SIZES = [10, 50, 200, 500]

NAV_HINTS = [
    {"from": "Gene", "to": "Disease", "functions": ["get_diseases_for_gene", "get_shared_pathways_for_genes"]},
    {"from": "Gene", "to": "Pathway", "functions": ["get_pathways_for_gene"]},
]


def build_envelope(final_results, total, fmt, columns=None, include_nav=True):
    """Simulate the response envelope from call_endpoint."""
    response = {}

    if fmt == "toon" and isinstance(final_results, list) and final_results:
        cols = columns or stable_key_union(final_results)
        toon_str = to_toon(final_results, cols)
        response["results_toon"] = toon_str
        response["results_columns"] = cols
    else:
        response["results"] = final_results

    if total > 50:
        response["pagination"] = {
            "total": total, "offset": 0, "limit": 50,
            "returned": min(50, len(final_results)),
            "has_more": True, "next_offset": 50,
        }
    else:
        response["total"] = total

    if include_nav:
        response["suggested_next"] = NAV_HINTS

    return response


def run():
    rows = []

    for shape_name, (gen, aggressive_fields) in SHAPES.items():
        for n in SIZES:
            items = [gen(i) for i in range(n)]
            all_cols = stable_key_union(items)

            # Simulate paginate (first page of 50)
            page = items[:min(50, n)]

            # 4 variants: JSON, JSON+fields, TOON, TOON+fields
            variants = {}

            # JSON (current default)
            env = build_envelope(page, n, "json")
            wire = compact_json(env)
            variants["json"] = count_tokens(wire)

            # JSON + fields
            proj = [{k: v for k, v in item.items() if k in set(aggressive_fields)} for item in page]
            env_f = build_envelope(proj, n, "json")
            wire_f = compact_json(env_f)
            variants["json+f"] = count_tokens(wire_f)

            # TOON (all columns)
            env_t = build_envelope(page, n, "toon", all_cols)
            wire_t = compact_json(env_t)
            variants["toon"] = count_tokens(wire_t)

            # TOON + fields
            env_tf = build_envelope(proj, n, "toon", aggressive_fields)
            wire_tf = compact_json(env_tf)
            variants["toon+f"] = count_tokens(wire_tf)

            # Estimator accuracy check (TOON escaped)
            toon_str = env_t.get("results_toon", "")
            if toon_str:
                escaped_len = len(json.dumps(toon_str))
                est = int(escaped_len / TOON_CHARS_PER_TOKEN)
                real = count_tokens(json.dumps(toon_str))
                est_err = (est - real) / real * 100 if real else 0
            else:
                est_err = 0

            rows.append({
                "shape": shape_name,
                "n": n,
                **variants,
                "toon_vs_json": f"{(1 - variants['toon'] / variants['json']) * 100:+.1f}%",
                "toon_f_vs_json_f": f"{(1 - variants['toon+f'] / variants['json+f']) * 100:+.1f}%",
                "est_err": f"{est_err:+.1f}%",
            })

    # --- Print table ---
    print("=" * 130)
    print("FULL PIPELINE FORMAT BENCHMARK")
    print("Wire = compact_json(envelope) → tiktoken cl100k_base")
    print("Page size capped at 50 items (pagination default)")
    print("=" * 130)
    print()
    hdr = f"{'Shape':<22} {'N':>4}  {'JSON':>6} {'JSON+F':>6} {'TOON':>6} {'TOON+F':>6}  {'TOON/JSON':>10} {'TOON+F/JSON+F':>14} {'EstErr':>7}"
    print(hdr)
    print("-" * 130)

    for r in rows:
        print(f"{r['shape']:<22} {r['n']:>4}  {r['json']:>6} {r['json+f']:>6} {r['toon']:>6} {r['toon+f']:>6}  "
              f"{r['toon_vs_json']:>10} {r['toon_f_vs_json_f']:>14} {r['est_err']:>7}")

    print()
    print("Legend:")
    print("  JSON/JSON+F = compact JSON with/without fields projection")
    print("  TOON/TOON+F = TOON format with/without fields projection")
    print("  TOON/JSON   = % token savings TOON vs full JSON")
    print("  TOON+F/JSON+F = % token savings TOON+fields vs JSON+fields (key metric)")
    print("  EstErr = TOON token estimator error vs real (positive = overestimate = safe)")
    print()

    # --- Write markdown artifact ---
    today = date.today().isoformat()
    md_lines = [
        f"# Full Pipeline Format Benchmark — {today}",
        "",
        "## Method",
        "- Synthetic payloads through actual `formats.py` serializers + `pagination.py`",
        "- Full envelope: pagination, suggested_next, all envelope fields",
        "- Wire format: `compact_json(envelope)` — actual MCP wire representation",
        "- Tokenizer: `cl100k_base` (tiktoken)",
        f"- {len(SHAPES)} shapes x {len(SIZES)} sizes x 4 format variants",
        "- Page size capped at 50 (pagination default)",
        "",
        "## Results",
        "",
        "| Shape | N | JSON | JSON+F | TOON | TOON+F | TOON/JSON | TOON+F/JSON+F | EstErr |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['shape']} | {r['n']} | {r['json']} | {r['json+f']} | "
            f"{r['toon']} | {r['toon+f']} | {r['toon_vs_json']} | "
            f"{r['toon_f_vs_json_f']} | {r['est_err']} |"
        )

    md_lines += [
        "",
        "## Key Findings",
        "",
        "1. **TOON+fields vs JSON+fields** is the key comparison (both use fields projection).",
        "2. **Estimator accuracy** shows the calibrated TOON_CHARS_PER_TOKEN error vs real tiktoken.",
        "3. All measurements include the full JSON envelope overhead (pagination, nav hints).",
        "4. Page size capped at 50 — for N>50, results are the first page only.",
        "",
        "## Decision",
        "",
        "Default format selection is based on these results.",
    ]

    out_path = Path("benchmarks") / f"formats-{today}.md"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {out_path}")

    return rows


if __name__ == "__main__":
    run()
