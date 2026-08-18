#!/usr/bin/env python
"""Submit a study wave to Slurm and record every cell in experiments/registry.csv.

  python scripts/ops/launch_matrix.py --task T1 --dry-run          # print sbatch lines
  python scripts/ops/launch_matrix.py --task T1                    # submit 24 cells
  python scripts/ops/launch_matrix.py --task T1 --levels L0 --n 25  # one cell

Design notes:
- The full grid is submitted as ONE parallel wave of independent 1-GPU jobs. With a fixed
  step budget a cell costs the same regardless of N, so serial skip-on-saturation would
  save nothing (plan: "no adaptive ladder").
- Every submitted cell gets a registry row immediately, with its job id. A cell that is
  already present and not 'failed' is skipped, so re-running after a partial submission is
  safe and idempotent.
- Runs on the LOGIN node (sbatch lives there); it does not need Isaac or the training env.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[2]
REGISTRY = REPO / "experiments" / "registry.csv"
LEVELS = ["L0", "L1", "L2", "L3"]
NDEMOS = [10, 25, 50, 100, 200, 400]


def read_registry() -> tuple[list[str], list[dict]]:
    with REGISTRY.open() as fh:
        rows = list(csv.DictReader(fh))
    with REGISTRY.open() as fh:
        header = next(csv.reader(fh))
    return header, rows


def run_id(task: str, level: str, n: int) -> str:
    return f"{task.lower()}_{level}_n{n}_s0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["T1", "T2", "T3"])
    ap.add_argument("--levels", nargs="*", default=LEVELS)
    ap.add_argument("--n", nargs="*", type=int, default=NDEMOS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sbatch", default=str(REPO / "slurm" / "train.sbatch"))
    args = ap.parse_args()

    header, rows = read_registry()
    existing = {r["run_id"]: r for r in rows}

    planned = [(lvl, n) for lvl in args.levels for n in args.n]
    print(f"{len(planned)} cells planned for {args.task}")

    new_rows = []
    for lvl, n in planned:
        rid = run_id(args.task, lvl, n)
        prev = existing.get(rid)
        if prev and prev.get("status") not in ("failed", "", None):
            print(f"  skip {rid} (already {prev.get('status')})")
            continue
        cmd = ["sbatch", args.sbatch, args.task, lvl, str(n)]
        if args.dry_run:
            print("  DRY " + " ".join(cmd))
            continue
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            print(f"  FAILED to submit {rid}: {out.stderr.strip()}")
            continue
        # "Submitted batch job 1234567"
        jobid = out.stdout.strip().split()[-1]
        print(f"  submitted {rid} as {jobid}")
        row = {k: "" for k in header}
        row.update({
            "run_id": rid,
            "task": {"T1": "cup_place", "T2": "drawer_stow", "T3": "push_target"}[args.task],
            "level": lvl,
            "variant": "-",
            "n_demos": str(n),
            "dataset_id": f"local/{lvl if args.task == 'T1' else args.task + '_' + lvl}",
            "slurm_jobid": jobid,
            "status": "submitted",
            "eval_set": lvl,
            "notes": "launched by launch_matrix.py",
        })
        new_rows.append(row)

    if new_rows:
        with REGISTRY.open("a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=header).writerows(new_rows)
        print(f"appended {len(new_rows)} rows to {REGISTRY.relative_to(REPO)}")
    elif not args.dry_run:
        print("nothing submitted")


if __name__ == "__main__":
    main()
