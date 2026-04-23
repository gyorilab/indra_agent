"""Pre-flight format benchmark: JSON vs TOON vs fields-aggressive JSON.

Generates realistic synthetic payloads matching actual call_endpoint response
shapes, wraps in full envelope (pagination, nav hints), serializes via
compact_json (actual MCP wire), and counts tokens with tiktoken cl100k_base.

Pass threshold: TOON must beat fields-aggressive JSON by >=15% at 50/500 rows
to justify the added serializer complexity.
"""

import json
import random
import string
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")


def compact_json(obj):
    return json.dumps(obj, separators=(",", ":"), default=str)


def count_tokens(s: str) -> int:
    return len(enc.encode(s))


# --- Synthetic payload generators ---


def make_gene_entity(i: int) -> dict:
    """Typical gene entity from get_genes_for_disease / get_targets_for_drug."""
    names = [
        "LRRK2", "TP53", "BRCA1", "EGFR", "BRAF", "KRAS", "MYC", "AKT1",
        "PIK3CA", "PTEN", "CDH1", "RB1", "ERBB2", "JAK2", "FGFR3", "ALK",
        "IDH1", "NOTCH1", "SMAD4", "VHL", "NPM1", "FLT3", "KIT", "NRAS",
        "ABL1", "PDGFRA", "MET", "RET", "CTNNB1", "ATM",
    ]
    name = names[i % len(names)] + (f"L{i}" if i >= len(names) else "")
    return {
        "db_ns": "HGNC",
        "db_id": str(6000 + i),
        "name": name,
        "evidence_count": random.randint(1, 500),
        "source_counts": {
            "reach": random.randint(0, 200),
            "bel": random.randint(0, 100),
            "sparser": random.randint(0, 50),
            "signor": random.randint(0, 30),
        },
    }


def make_disease_entity(i: int) -> dict:
    """Typical disease from get_diseases_for_gene."""
    diseases = [
        "Parkinson disease", "breast cancer", "lung adenocarcinoma",
        "glioblastoma", "acute myeloid leukemia", "colorectal cancer",
        "hepatocellular carcinoma", "melanoma", "prostate cancer",
        "ovarian cancer", "pancreatic cancer", "renal cell carcinoma",
        "non-small cell lung cancer", "chronic lymphocytic leukemia",
    ]
    name = diseases[i % len(diseases)]
    return {
        "db_ns": "MESH",
        "db_id": f"D{50000 + i:06d}",
        "name": name,
        "evidence_count": random.randint(1, 300),
        "source_counts": {
            "reach": random.randint(0, 150),
            "bel": random.randint(0, 80),
        },
    }


def make_drug_entity(i: int) -> dict:
    """Typical drug from get_drugs_for_target."""
    drugs = [
        "imatinib", "trastuzumab", "pembrolizumab", "osimertinib",
        "venetoclax", "olaparib", "ruxolitinib", "dabrafenib",
        "nivolumab", "erlotinib", "sorafenib", "lenvatinib",
    ]
    name = drugs[i % len(drugs)]
    return {
        "db_ns": "CHEBI",
        "db_id": str(40000 + i),
        "name": name,
        "evidence_count": random.randint(1, 200),
        "source_counts": {
            "reach": random.randint(0, 100),
            "bel": random.randint(0, 60),
            "drugbank": random.randint(0, 20),
        },
    }


def make_statement(i: int) -> dict:
    """Heavier shape: statement-like results with more fields."""
    return {
        "stmt_hash": random.randint(10**17, 10**18),
        "stmt_type": random.choice([
            "Activation", "Inhibition", "IncreaseAmount", "DecreaseAmount",
            "Phosphorylation", "Complex",
        ]),
        "subj_ns": "HGNC",
        "subj_id": str(6000 + i),
        "subj_name": f"GENE{i}",
        "obj_ns": "HGNC",
        "obj_id": str(7000 + i),
        "obj_name": f"GENE{i + 500}",
        "evidence_count": random.randint(1, 100),
        "source_counts": {
            "reach": random.randint(0, 50),
            "bel": random.randint(0, 30),
            "signor": random.randint(0, 10),
        },
        "belief": round(random.uniform(0.5, 1.0), 4),
    }


# --- Format renderers ---


def to_toon(rows: list[dict], columns: list[str]) -> str:
    """Render rows as TOON (tab-separated, header row, spec-compliant escaping)."""
    lines = ["\t".join(columns)]
    for row in rows:
        cells = []
        for col in columns:
            val = row.get(col)
            if val is None:
                cells.append("")
            elif isinstance(val, (dict, list)):
                # Nested → JSON substring, always quoted
                cells.append(json.dumps(val, separators=(",", ":")))
            else:
                s = str(val)
                # Escape if contains tab, newline, or leading/trailing whitespace
                if "\t" in s or "\n" in s or s != s.strip() or '"' in s or "\\" in s:
                    cells.append(json.dumps(s))
                else:
                    cells.append(s)
        lines.append("\t".join(cells))
    return "\n".join(lines)


def stable_key_union(rows: list[dict]) -> list[str]:
    """First-seen-order union of all keys."""
    seen = {}
    for row in rows:
        for k in row:
            if k not in seen:
                seen[k] = len(seen)
    return sorted(seen, key=seen.get)


def fields_project(rows: list[dict], fields: list[str]) -> list[dict]:
    return [{k: v for k, v in row.items() if k in field_set} for row in rows for field_set in [set(fields)]]


def make_envelope(results_payload, total: int, format_name: str,
                  columns: list[str] | None = None,
                  include_nav: bool = True) -> dict:
    """Build a realistic response envelope matching call_endpoint output."""
    env = {}

    if format_name == "toon":
        env["results_toon"] = results_payload
        env["results_columns"] = columns
    else:
        env["results"] = results_payload

    if total > 50:
        env["pagination"] = {
            "total": total,
            "offset": 0,
            "limit": 50,
            "returned": min(50, total),
            "has_more": total > 50,
            "next_offset": 50,
            "token_estimate": 0,  # not measured in synthetic payloads
        }
    else:
        env["total"] = total

    if include_nav:
        env["suggested_next"] = [
            {"from": "Gene", "to": "Disease", "functions": ["get_diseases_for_gene", "get_shared_pathways_for_genes"]},
            {"from": "Gene", "to": "Pathway", "functions": ["get_pathways_for_gene"]},
        ]

    return env


# --- Benchmark runner ---


ENTITY_GENERATORS = {
    "gene_entity": make_gene_entity,
    "disease_entity": make_disease_entity,
    "drug_entity": make_drug_entity,
    "statement": make_statement,
}

SIZES = [10, 50, 200, 500]

# Fields-aggressive projections per shape
FIELDS_AGGRESSIVE = {
    "gene_entity": ["db_ns", "db_id", "name"],
    "disease_entity": ["db_ns", "db_id", "name"],
    "drug_entity": ["db_ns", "db_id", "name"],
    "statement": ["stmt_type", "subj_name", "obj_name", "evidence_count", "belief"],
}


def run_benchmark():
    random.seed(42)

    rows_data = []

    for shape_name, gen in ENTITY_GENERATORS.items():
        for n in SIZES:
            items = [gen(i) for i in range(n)]
            columns = stable_key_union(items)
            aggressive_fields = FIELDS_AGGRESSIVE[shape_name]

            # --- Variant 1: JSON (current default) ---
            env_json = make_envelope(items, n, "json")
            wire_json = compact_json(env_json)
            tok_json = count_tokens(wire_json)

            # --- Variant 2: JSON + fields-aggressive ---
            projected = fields_project(items, aggressive_fields)
            env_json_fields = make_envelope(projected, n, "json")
            wire_json_fields = compact_json(env_json_fields)
            tok_json_fields = count_tokens(wire_json_fields)

            # --- Variant 3: TOON (full columns) ---
            toon_str = to_toon(items, columns)
            env_toon = make_envelope(toon_str, n, "toon", columns)
            wire_toon = compact_json(env_toon)
            tok_toon = count_tokens(wire_toon)

            # --- Variant 4: TOON + fields-aggressive ---
            toon_agg_str = to_toon(projected, aggressive_fields)
            env_toon_agg = make_envelope(toon_agg_str, n, "toon", aggressive_fields)
            wire_toon_agg = compact_json(env_toon_agg)
            tok_toon_agg = count_tokens(wire_toon_agg)

            # Compute savings
            toon_vs_json_pct = (1 - tok_toon / tok_json) * 100 if tok_json else 0
            toon_vs_fields_pct = (1 - tok_toon / tok_json_fields) * 100 if tok_json_fields else 0
            toon_agg_vs_fields_pct = (1 - tok_toon_agg / tok_json_fields) * 100 if tok_json_fields else 0

            rows_data.append({
                "shape": shape_name,
                "n": n,
                "tok_json": tok_json,
                "tok_json_fields": tok_json_fields,
                "tok_toon": tok_toon,
                "tok_toon_agg": tok_toon_agg,
                "bytes_json": len(wire_json),
                "bytes_toon": len(wire_toon),
                "toon_vs_json": f"{toon_vs_json_pct:+.1f}%",
                "toon_vs_fields": f"{toon_vs_fields_pct:+.1f}%",
                "toon_agg_vs_fields": f"{toon_agg_vs_fields_pct:+.1f}%",
            })

    # Print results
    print("=" * 120)
    print("PRE-FLIGHT FORMAT BENCHMARK")
    print("Pass threshold: TOON must beat fields-aggressive JSON by >15% at 50/500 rows")
    print("Tokenizer: cl100k_base (GPT-4/Claude proxy)")
    print("=" * 120)
    print()
    print(f"{'Shape':<18} {'N':>4}  {'JSON':>6}  {'JSON+F':>6}  {'TOON':>6}  {'TOON+F':>6}  "
          f"{'TOON/JSON':>10}  {'TOON/JSON+F':>12}  {'TOON+F/JSON+F':>14}")
    print("-" * 120)

    for r in rows_data:
        print(f"{r['shape']:<18} {r['n']:>4}  {r['tok_json']:>6}  {r['tok_json_fields']:>6}  "
              f"{r['tok_toon']:>6}  {r['tok_toon_agg']:>6}  "
              f"{r['toon_vs_json']:>10}  {r['toon_vs_fields']:>12}  {r['toon_agg_vs_fields']:>14}")

    print()
    print("Legend:")
    print("  JSON       = current compact JSON (all keys)")
    print("  JSON+F     = compact JSON with fields-aggressive projection")
    print("  TOON       = TOON format (all columns)")
    print("  TOON+F     = TOON format with fields-aggressive projection")
    print("  TOON/JSON  = % token savings of TOON vs full JSON (positive = TOON cheaper)")
    print("  TOON/JSON+F = % token savings of TOON (full) vs JSON+fields")
    print("  TOON+F/JSON+F = % token savings of TOON+fields vs JSON+fields (pass threshold)")
    print()

    # Envelope overhead analysis
    print("--- Envelope overhead (fixed cost regardless of format) ---")
    empty_json = make_envelope([], 0, "json", include_nav=True)
    empty_wire = compact_json(empty_json)
    empty_tok = count_tokens(empty_wire)
    print(f"  Empty envelope with nav hints: {empty_tok} tokens ({len(empty_wire)} bytes)")
    print()

    # Double-escape analysis
    print("--- Double-escape tax: TOON string inside outer JSON ---")
    sample_items = [make_gene_entity(i) for i in range(5)]
    sample_cols = stable_key_union(sample_items)
    raw_toon = to_toon(sample_items, sample_cols)
    raw_toon_tokens = count_tokens(raw_toon)
    # After compact_json escaping (tabs → \t, newlines → \n in JSON string)
    escaped_toon = json.dumps(raw_toon)  # what happens inside compact_json
    escaped_toon_tokens = count_tokens(escaped_toon)
    tax_pct = (escaped_toon_tokens / raw_toon_tokens - 1) * 100 if raw_toon_tokens else 0
    print(f"  5-row gene TOON raw:     {raw_toon_tokens} tokens")
    print(f"  5-row gene TOON escaped: {escaped_toon_tokens} tokens (inside JSON string)")
    print(f"  Double-escape tax:       {tax_pct:+.1f}%")

    return rows_data


if __name__ == "__main__":
    results = run_benchmark()
