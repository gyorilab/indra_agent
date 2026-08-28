"""Calibrate token estimation heuristics against tiktoken cl100k_base.

Tests both JSON and TOON paths to verify the heuristics in pagination.py
are within ±15% of real token counts.
"""

import json
import random
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")


def real_tokens(s: str) -> int:
    return len(enc.encode(s))


def estimate_json(obj) -> int:
    """Replicate pagination.py:estimate_tokens logic."""
    json_str = json.dumps(obj, separators=(",", ":"), default=str)
    est = int(len(json_str) / 4.0)
    if isinstance(obj, list):
        est += len(obj) * 2
    elif isinstance(obj, dict):
        est += len(obj) * 3
    return est


def estimate_str(s: str) -> int:
    """Replicate estimate_tokens_str logic."""
    return int(len(s) / 4.0)


# Generate realistic payloads
random.seed(42)


def make_genes(n):
    names = ["LRRK2", "TP53", "BRCA1", "EGFR", "BRAF", "KRAS", "MYC", "AKT1",
             "PIK3CA", "PTEN", "CDH1", "RB1", "ERBB2", "JAK2", "FGFR3"]
    return [
        {
            "db_ns": "HGNC",
            "db_id": str(6000 + i),
            "name": names[i % len(names)] + (f"L{i}" if i >= len(names) else ""),
            "evidence_count": random.randint(1, 500),
            "source_counts": {
                "reach": random.randint(0, 200),
                "bel": random.randint(0, 100),
                "sparser": random.randint(0, 50),
            },
        }
        for i in range(n)
    ]


def to_toon(rows, columns):
    lines = ["\t".join(columns)]
    for row in rows:
        cells = []
        for col in columns:
            val = row.get(col)
            if val is None:
                cells.append("")
            elif isinstance(val, (dict, list)):
                cells.append(json.dumps(val, separators=(",", ":"), default=str))
            else:
                cells.append(str(val))
        lines.append("\t".join(cells))
    return "\n".join(lines)


print(f"{'Payload':<30} {'Real':>6} {'Est':>6} {'Error':>8}")
print("-" * 60)

for n in [10, 50, 200, 500]:
    genes = make_genes(n)

    # JSON path
    json_str = json.dumps(genes, separators=(",", ":"))
    real_j = real_tokens(json_str)
    est_j = estimate_json(genes)
    err_j = (est_j - real_j) / real_j * 100

    # TOON path
    cols = ["db_ns", "db_id", "name", "evidence_count", "source_counts"]
    toon_str = to_toon(genes, cols)
    real_t = real_tokens(toon_str)
    est_t = estimate_str(toon_str)
    err_t = (est_t - real_t) / real_t * 100

    # TOON inside outer JSON (double-escape)
    escaped = json.dumps(toon_str)
    real_e = real_tokens(escaped)
    est_e = estimate_str(escaped)
    err_e = (est_e - real_e) / real_e * 100

    print(f"JSON ({n:>3} genes)              {real_j:>6} {est_j:>6} {err_j:>+7.1f}%")
    print(f"TOON ({n:>3} genes)              {real_t:>6} {est_t:>6} {err_t:>+7.1f}%")
    print(f"TOON-escaped ({n:>3} genes)      {real_e:>6} {est_e:>6} {err_e:>+7.1f}%")
    print()

print("Target: all estimates within ±15% of real")
