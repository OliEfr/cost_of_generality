"""Merge per-sub-level eval-set snapshots into the frozen per-level sets (D11).

  python scripts/dev/merge_eval_sets.py                                     # Task 1 ladder
  python scripts/dev/merge_eval_sets.py --raw ops/eval_sets_raw_t2 --prefix T2_
  python scripts/dev/merge_eval_sets.py --raw results/_evalsets_raw/T2 --prefix T2_ \
         --flat --variant-arm AC --variant-arm BC                           # reverse arms (D31)

Writes configs/eval_sets/<prefix><level>.json.

**Rule 8 is enforced here, not assumed.** An existing frozen set is never overwritten: the
script refuses and exits non-zero, because a silently regenerated benchmark invalidates every
number ever measured against it. `--force` exists only for the case where a merge was
interrupted midway and the half-written file has to be replaced.
"""
import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="ops/eval_sets_raw")
ap.add_argument("--prefix", default="")
ap.add_argument("--out", default="configs/eval_sets")
ap.add_argument("--flat", nargs="*", default=["L0", "L1", "L2"],
                help="flat levels (one sub-level each). Pass --flat with no values for none.")
ap.add_argument("--variant-arm", action="append", default=None, dest="variant_arms",
                help="arm whose sub-levels are <arm>v00..<arm>v09 (repeatable). "
                     "Default: L3. Pass --variant-arm '' for none.")
ap.add_argument("--force", action="store_true",
                help="replace an existing frozen set (see the rule-8 note in the docstring)")
args = ap.parse_args()

RAW = Path(args.raw)
OUT = Path(args.out)
ARMS = ["L3"] if args.variant_arms is None else [a for a in args.variant_arms if a]

DIAGONAL = (
    "variant v uses batch v (diagonal), pooled: 200 eps, 200 DISTINCT poses, all 10 variants. "
    "NB batch 0 on every variant would also be 200 eps but only 20 distinct poses -- variants "
    "share the pose RNG stream (see D18)."
)
DIAGONAL_MAX = (
    "same as standard_eval: the frozen snapshots contain 10 batches x 20 envs = 200 distinct "
    "poses in total, so 200 eps is the maximum spatial coverage available at this arm."
)


def load(key):
    d = json.loads((RAW / f"{key}.json").read_text())
    assert d["num_envs"] == 20 and len(d["batches"]) == 10, key
    return d


def write(level: str, payload: dict) -> None:
    path = OUT / f"{args.prefix}{level}.json"
    if path.exists() and not args.force:
        print(f"MERGE_REFUSED {path} already exists -- frozen benchmarks are never "
              f"regenerated (CLAUDE.md rule 8). Use --force only to replace a half-written file.")
        sys.exit(2)
    path.write_text(json.dumps(payload, indent=1))


for lvl in args.flat:
    d = load(lvl)
    d["standard_eval"] = "batches 0-4 (100 eps)"
    d["headline_rerun"] = "batches 0-9 (200 eps)"
    write(lvl, d)
    print(f"{args.prefix}{lvl}: 10 batches frozen")

for arm in ARMS:
    variants = {f"{arm}v{v:02d}": load(f"{arm}v{v:02d}") for v in range(10)}
    write(arm, {"standard_eval": DIAGONAL, "headline_rerun": DIAGONAL_MAX, "variants": variants})
    print(f"{args.prefix}{arm}: 10 variants x 10 batches frozen")

print("MERGE_OK")
