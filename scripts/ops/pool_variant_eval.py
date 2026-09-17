#!/usr/bin/env python3
"""Pool the per-slice eval JSONs of one cell into the single result the analysis reads.

A cell under the D31 uniform-slice protocol is TEN independent processes: slice s scores batch s
(reset seed 5000+s, 20 envs) after an unscored warm-up. For a variant arm slice s also selects
sub-env <ARM>v0s, which is D18's diagonal; for a flat arm every slice uses the same env. Either
way the pooled result is 200 episodes over 200 distinct poses.

Two invariants, both learned the hard way:

  * **A partial pool is never written.** Nine slices would silently produce a 180-episode number
    that looks like a cell and would be compared against 200-episode cells. The script exits
    non-zero with INCOMPLETE instead. (Same rule as run_local_eval_l3.py, which this replaces as
    the single implementation.)
  * **Slices measured under different settings are never pooled.** num_inference_steps and the
    warm-up block must agree across all ten, because two protocols averaged together is a number
    that describes nothing.

`scripts/ops/run_local_eval_l3.py` keeps its own inline pooling and is deliberately NOT refactored
to call this. The plan said to share the code so the two schemas stay identical -- but they must
NOT: that script pools the OLD diagonal protocol (no warm-up, 100-episode flat cells alongside), and
this one pools the D31 uniform slices. Two protocols that produce byte-identical results are two
protocols nobody can tell apart afterwards, which is the failure this file's MIXED_PROTOCOL check
exists to prevent. The local script stays the reader of the old surface; this one owns the new.

usage:
  pool_variant_eval.py --partials $WORK/cog/results/_partials --results $WORK/cog/results \\
                       --task T1 --arm BC --n 100 [--step 080000] [--suffix u200]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VARIANT_ARMS = {"AC", "BC", "L3", "L3b"}
GYM_PREFIX = {"T1": "Cog-CupPlace", "T2": "Cog-DrawerStow", "T3": "Cog-PushTarget"}
# Per-episode stage flags recorded by --stages (drawer_stow only). Pooled by recounting the
# episodes rather than averaging the per-slice rates, so an unequal slice can never skew them.
STAGE_KEYS = ("drawer_opened", "object_lifted", "object_over_drawer")


def build_payload(parts: list[dict], *, task_tag: str, arm: str, step: str, checkpoint: str,
                  slice_keys: list[str], scheme: str, comment: str) -> dict:
    """One pooled result from the per-slice results, in the order given."""
    per_slice = {}
    outcomes: list = []
    for key, d, seed in zip(slice_keys, parts, range(5000, 5000 + len(parts))):
        per_slice[key] = {
            "successes": d.get("successes"),
            "episodes": d.get("episodes"),
            "success_rate": d.get("success_rate"),
            "seed": seed,
        }
        outcomes.extend(d.get("outcomes", []))

    successes = sum(v["successes"] for v in per_slice.values())
    episodes = sum(v["episodes"] for v in per_slice.values())
    proto0 = parts[0].get("protocol", {})
    payload = {
        "task": f"{GYM_PREFIX[task_tag]}-{arm}-IK-Rel-Visuomotor-v0 "
                f"(pooled over {len(parts)} slices)",
        "checkpoint": checkpoint,
        "num_inference_steps": parts[0].get("num_inference_steps"),
        "success_rate": round(successes / episodes, 4),
        "successes": successes,
        "episodes": episodes,
        "outcomes": outcomes,
        "protocol": {
            "num_envs": proto0.get("num_envs"),
            "slices": len(parts),
            "base_seed": 5000,
            "scheme": scheme,
            "warmup": proto0.get("warmup"),
            "comment": comment,
        },
        "per_variant" if arm in VARIANT_ARMS else "per_batch": per_slice,
    }
    # Stage rates come back only when the env had stage definitions (drawer_stow). Recount from the
    # pooled episodes; a missing key in any episode means the slices disagree, which is a bug.
    if outcomes and all(k in outcomes[0] for k in STAGE_KEYS):
        payload["stages"] = {
            k: round(sum(bool(o[k]) for o in outcomes) / episodes, 4) for k in STAGE_KEYS
        }
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partials", required=True, help="directory holding the per-slice JSONs")
    ap.add_argument("--results", required=True, help="directory the pooled result is written to")
    ap.add_argument("--task", required=True, choices=sorted(GYM_PREFIX))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--n", required=True, type=int, help="demo count of the cell")
    ap.add_argument("--step", default="080000")
    ap.add_argument("--suffix", default="u200", help="protocol suffix of both inputs and output")
    ap.add_argument("--slices", type=int, default=10)
    args = ap.parse_args()

    stem = f"eval_{args.task}_{args.arm}_n{args.n}_{args.step}_{args.suffix}"
    out = Path(args.results) / f"{stem}.json"
    if out.exists() and out.stat().st_size > 0:
        print(f"POOL_SKIP {out.name} already exists")
        return 0

    paths = [Path(args.partials) / f"{stem}_s{s}.json" for s in range(args.slices)]
    missing = [p.name for p in paths if not (p.exists() and p.stat().st_size > 0)]
    if missing:
        print(f"INCOMPLETE {stem}: {len(missing)} of {args.slices} slice(s) missing: {missing}")
        print("refusing to pool a partial cell -- the result would understate coverage silently")
        return 5

    parts = [json.loads(p.read_text()) for p in paths]

    # Every slice must have been measured the same way.
    settings = {(d.get("num_inference_steps"),
                 json.dumps(d.get("protocol", {}).get("warmup"), sort_keys=True)) for d in parts}
    if len(settings) != 1:
        print(f"MIXED_PROTOCOL {stem}: slices disagree on (num_inference_steps, warmup): {settings}")
        return 6
    sizes = {d.get("episodes") for d in parts}
    if len(sizes) != 1:
        print(f"MIXED_EPISODES {stem}: slices disagree on episode count: {sizes}")
        return 6

    if args.arm in VARIANT_ARMS:
        slice_keys = [f"{args.arm}v{s:02d}" for s in range(args.slices)]
        scheme = "diagonal: variant v evaluated on batch v, one process each (D18 + D31)"
    else:
        slice_keys = [f"b{s:02d}" for s in range(args.slices)]
        scheme = "uniform slices: batch b evaluated in its own process (D31)"
    comment = (
        f"D31 uniform-slice protocol: {args.slices} processes x {sizes.pop()} episodes = "
        f"{sum(d['episodes'] for d in parts)} episodes over batches 0-{args.slices - 1}, so every "
        "arm -- flat or per-variant -- has the same spatial coverage AND the same process warm-up "
        "state. Supersedes the mixed 100-episode/diagonal protocol for the cells it covers."
    )

    payload = build_payload(
        parts, task_tag=args.task, arm=args.arm, step=args.step,
        checkpoint=parts[0].get("checkpoint", ""), slice_keys=slice_keys,
        scheme=scheme, comment=comment,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    stages = f" stages={payload['stages']}" if "stages" in payload else ""
    print(f"POOLED_OK {out.name}: SR={payload['success_rate']} "
          f"({payload['successes']}/{payload['episodes']}){stages}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
