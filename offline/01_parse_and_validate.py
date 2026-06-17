import gzip
import json
import pandas as pd
from tqdm import tqdm
from collections import Counter, defaultdict

def load_candidates(filepath: str) -> list[dict]:
    """Stream-parse JSONL (handles uncompressed per user request). Memory-safe for 100K records."""
    import os
    if not os.path.exists(filepath) and filepath.endswith(".gz"):
        uncompressed = filepath[:-3]
        if os.path.exists(uncompressed):
            filepath = uncompressed

    candidates = []
    
    if filepath.endswith(".gz"):
        f = gzip.open(filepath, "rt", encoding="utf-8")
    else:
        f = open(filepath, "rt", encoding="utf-8")
        
    with f:
        for line in tqdm(f, desc="Loading candidates"):
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[WARN] Skipping malformed line: {e}")
    print(f"[INFO] Loaded {len(candidates):,} candidates")
    assert len(candidates) == 100_000, f"Expected 100K, got {len(candidates)}"
    return candidates

def audit_schema(candidates: list[dict], sample_n: int = 200):
    """Inventory all fields, types, and null rates."""
    field_types = defaultdict(Counter)
    field_nulls = Counter()
    
    for c in candidates[:sample_n]:
        for key, val in c.items():
            field_types[key][type(val).__name__] += 1
            if val is None or val == "" or val == []:
                field_nulls[key] += 1
    
    print("=== FIELD AUDIT ===")
    for field in sorted(field_types.keys()):
        null_rate = field_nulls[field] / sample_n * 100
        print(f"  {field:40s} | types={dict(field_types[field])} | null={null_rate:.1f}%")
    
    # Specifically check redrob_signals structure
    sample = candidates[0].get("redrob_signals", {})
    print("\n=== REDROB_SIGNALS KEYS ===")
    for k, v in sample.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    candidates = load_candidates("../data/candidates.jsonl.gz")
    audit_schema(candidates)
