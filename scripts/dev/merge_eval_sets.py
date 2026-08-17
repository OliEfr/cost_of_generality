"""Merge per-sub-level eval-set snapshots into the frozen per-level sets (D11).

  python scripts/dev/merge_eval_sets.py                                   # Task 1
  python scripts/dev/merge_eval_sets.py --raw ops/eval_sets_raw_t2 --prefix T2_

Writes configs/eval_sets/<prefix>{L0,L1,L2,L3}.json. Existing files of other
prefixes are never touched (rule 8: frozen benchmarks are not regenerated).
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="ops/eval_sets_raw")
ap.add_argument("--prefix", default="")
args = ap.parse_args()

RAW = Path(args.raw)
OUT = Path("configs/eval_sets")


def load(key):
    d = json.loads((RAW / f"{key}.json").read_text())
    assert d["num_envs"] == 20 and len(d["batches"]) == 10, key
    return d


for lvl in ("L0", "L1", "L2"):
    d = load(lvl)
    d["standard_eval"] = "batches 0-4 (100 eps)"
    d["headline_rerun"] = "batches 0-9 (200 eps)"
    (OUT / f"{args.prefix}{lvl}.json").write_text(json.dumps(d, indent=1))
    print(f"{args.prefix}{lvl}: 10 batches frozen")

variants = {f"L3v{v:02d}": load(f"L3v{v:02d}") for v in range(10)}
l3 = {
    "standard_eval": ("variant v uses batch v (diagonal), pooled: 200 eps, 200 DISTINCT poses, all 10 variants. NB batch 0 on every variant would also be 200 eps but only 20 distinct poses -- variants share the pose RNG stream (see D18)."),
    "headline_rerun": ("same as standard_eval: the frozen snapshots contain 10 batches x 20 envs = 200 distinct poses in total, so 200 eps is the maximum spatial coverage available at L3."),
    "variants": variants,
}
(OUT / f"{args.prefix}L3.json").write_text(json.dumps(l3, indent=1))
print(f"{args.prefix}L3: 10 variants x 10 batches frozen")
print("MERGE_OK")
