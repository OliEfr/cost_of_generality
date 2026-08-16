"""Merge ops/eval_sets_raw/*.json into the frozen per-level eval sets (D11)."""
import json
from pathlib import Path

RAW = Path("ops/eval_sets_raw")
OUT = Path("configs/eval_sets")

def load(key):
    d = json.loads((RAW / f"{key}.json").read_text())
    assert d["num_envs"] == 20 and len(d["batches"]) == 10, key
    return d

for lvl in ("L0", "L1", "L2"):
    d = load(lvl)
    d["standard_eval"] = "batches 0-4 (100 eps)"
    d["headline_rerun"] = "batches 0-9 (200 eps)"
    (OUT / f"{lvl}.json").write_text(json.dumps(d, indent=1))
    print(f"{lvl}: 10 batches frozen")

variants = {f"L3v{v:02d}": load(f"L3v{v:02d}") for v in range(10)}
l3 = {
    "standard_eval": "batch 0 on each of the 10 sub-envs, pooled (200 eps)",
    "headline_rerun": "batches 0-1 on each sub-env, pooled (400 eps)",
    "variants": variants,
}
(OUT / "L3.json").write_text(json.dumps(l3, indent=1))
print("L3: 10 variants x 10 batches frozen")
print("MERGE_OK")
