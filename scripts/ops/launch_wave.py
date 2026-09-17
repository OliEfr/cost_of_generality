#!/usr/bin/env python
"""Submit a datagen / freeze / convert / eval wave to Slurm and log every job.

  python scripts/ops/launch_wave.py --stage datagen --task T1 --arms AC BC --dry-run
  python scripts/ops/launch_wave.py --stage freeze  --task T2 --arms AC BC
  python scripts/ops/launch_wave.py --stage convert --task T3 --arms AC BC
  python scripts/ops/launch_wave.py --stage eval    --task T1 --arms AC BC --n 10 25 50 100 200 400
  python scripts/ops/launch_wave.py --stage eval    --task T1 --arms L0 L1 L2 L3b   # baseline re-measure

Why this is NOT part of launch_matrix.py: that script's idempotency key and its output are one
`experiments/registry.csv` row per STUDY CELL, and the registry is a paper artifact. A datagen leg
is not a cell, and an eval slice is a tenth of one; folding them in would make both jobs worse.
The submission conventions are copied verbatim, though:

- **Runs on the WORKSTATION, submitting over ssh.** The ledger resolves relative to this file, so
  running on the cluster would write `$WORK/cog/repo/experiments/` -- the rsync mirror that the next
  `sync_up.sh code --delete` erases.
- `$WORK` is left unexpanded: the remote shell expands it (true for ssh command mode, NOT for an
  rsync destination).
- Slurm logs go to `$WORK/cog/logs/` via an explicit `-o`, or sbatch litters the code mirror.

Idempotency is ARTIFACT-based, not ledger-based: every sbatch here skips its own completed output
(GEN_SKIP / CONVERT_SKIP / FREEZE_SKIP / EVAL_SKIP). That survives a lost ledger and a partially
synced tree, which a CSV check does not. The ledger is for provenance, not for control flow.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import pathlib
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[2]
LEDGER = REPO / "experiments" / "cluster_jobs.csv"
LEDGER_FIELDS = ["submitted_at", "stage", "task", "arm", "key", "stem", "n_demos", "slice",
                 "seed", "trials", "sbatch", "walltime", "slurm_jobid", "notes"]

NDEMOS = [10, 25, 50, 100, 200, 400]
SLICES = 10          # D31: a cell is ten one-batch processes, pooled to 200 episodes
TRIALS_PER_VARIANT = 40

# THE study's shape, in one place. Seed blocks are disjoint from the additive ladder's
# (T1 1000+v / T2 2000+v / T3 3000+v), from the eval seeds 5000-5009 and from the warm-up seeds
# 4900+w -- so training poses, eval poses and warm-up poses can never coincide (D27, D31).
SEED_BASE = {"T1": 1000, "T2": 2000, "T3": 3000}
ARM_SEED_OFFSET = {"AC": 300, "BC": 400}
VARIANT_ARMS = {"AC", "BC", "L3", "L3b"}

# Datagen walltime per 40-demo variant leg, from docs/timings.md x ~3.3 (the A100 has no RT cores).
# These are GUESSES until gate G3 measures them -- update this table, not the call sites.
GEN_WALLTIME = {"T1": "00:30:00", "T2": "03:00:00", "T3": "00:30:00"}
# Eval walltime per SLICE (20 scored episodes + one warm-up batch), from the measured 15.9 s/episode
# at max_steps 600, scaled by each task's step cap.
EVAL_WALLTIME = {"T1": "01:00:00", "T2": "02:00:00", "T3": "01:30:00"}

# Superseded artifacts. A wave must never read or write one of these: L3 is the pose-redundant arm
# (43-48 unique poses against a nominal 400, D27/D29), _sharedenc is the pre-D26 architecture, and
# L3b_401ep_unbalanced is a rejected conversion. Checked as a substring of every job argument,
# because the failure mode is silent -- the job runs perfectly and measures the wrong thing.
SUPERSEDED = ("_sharedenc", "L3b_401ep_unbalanced")


def superseded_check(arm: str, args_list: list[str]) -> None:
    blob = " ".join(args_list)
    for bad in SUPERSEDED:
        if bad in blob:
            raise SystemExit(f"refusing: job arguments name the superseded artifact {bad!r}: {blob}")
    if arm == "L3":
        raise SystemExit(
            "refusing arm 'L3': that is the deprecated pose-redundant arm (D27/D29). The full "
            "disturbance set is 'L3b', which is what the study reports as L3.")


def gen_jobs(task: str, arm: str) -> list[dict]:
    if arm not in VARIANT_ARMS:
        raise SystemExit(f"--stage datagen supports the per-variant arms {sorted(VARIANT_ARMS)}; "
                         f"a flat arm needs its own trials/sharding decision")
    base = SEED_BASE[task] + ARM_SEED_OFFSET[arm]
    prefix = "" if task == "T1" else f"{task}_"
    out = []
    for v in range(10):
        key = f"{arm}v{v:02d}"
        out.append({"key": key, "stem": f"{prefix}{key}", "seed": base + v,
                    "trials": TRIALS_PER_VARIANT,
                    "args": [task, key, f"{prefix}{key}", str(base + v), str(TRIALS_PER_VARIANT), "8"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["datagen", "freeze", "convert", "eval"])
    ap.add_argument("--task", required=True, choices=["T1", "T2", "T3"])
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--n", nargs="*", type=int, default=NDEMOS, help="eval stage only")
    ap.add_argument("--step", default="080000", help="eval stage only")
    ap.add_argument("--slices", type=int, default=SLICES, help="eval stage only")
    ap.add_argument("--warmup-batches", type=int, default=1, help="eval stage only; see gate G6")
    ap.add_argument("--warmup-max-steps", type=int, default=0,
                    help="eval stage only; 0 = the task's own --max_steps")
    ap.add_argument("--suffix", default="u200", help="eval stage only: protocol suffix")
    ap.add_argument("--time", default=None, help="override the per-stage walltime table")
    ap.add_argument("--qos", default=None, help="e.g. boost_qos_dbg for a gate")
    ap.add_argument("--partition", default=None, help="e.g. boost_usr_prod for a convert fallback")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--remote", default="leonardo")
    ap.add_argument("--no-remote", dest="remote", action="store_const", const=None)
    args = ap.parse_args()

    jobs: list[dict] = []
    for arm in args.arms:
        superseded_check(arm, [])
        if args.stage == "datagen":
            sbatch, walltime = "slurm/datagen.sbatch", args.time or GEN_WALLTIME[args.task]
            for j in gen_jobs(args.task, arm):
                jobs.append({**j, "arm": arm, "sbatch": sbatch, "walltime": walltime})
        elif args.stage == "freeze":
            jobs.append({"arm": arm, "key": arm, "stem": "", "seed": "", "trials": "",
                         "args": [args.task, arm], "sbatch": "slurm/freeze_eval_sets.sbatch",
                         "walltime": args.time or "00:30:00"})
        elif args.stage == "convert":
            jobs.append({"arm": arm, "key": arm, "stem": "", "seed": "", "trials": "",
                         "args": [args.task, arm], "sbatch": "slurm/convert.sbatch",
                         "walltime": args.time or "04:00:00"})
        else:
            sbatch, walltime = "slurm/eval.sbatch", args.time or EVAL_WALLTIME[args.task]
            for n in args.n:
                for s in range(args.slices):
                    jobs.append({"arm": arm, "key": arm, "stem": "", "seed": "", "trials": "",
                                 "n_demos": n, "slice": s, "sbatch": sbatch, "walltime": walltime,
                                 "args": [args.task, arm, str(n), str(s), args.step]})
    superseded_check("", [str(x) for j in jobs for x in j["args"]])

    print(f"{len(jobs)} job(s) planned: stage={args.stage} task={args.task} arms={args.arms}")

    env = ""
    if args.stage == "eval":
        env = (f"COG_WARMUP_BATCHES={args.warmup_batches} "
               f"COG_WARMUP_MAX_STEPS={args.warmup_max_steps} COG_EVAL_SUFFIX={args.suffix} ")
    extra = ""
    if args.qos:
        extra += f"-q {args.qos} "
    if args.partition:
        extra += f"-p {args.partition} "

    rows = []
    for j in jobs:
        argstr = " ".join(str(a) for a in j["args"])
        submit = (f"cd $WORK/cog/repo && {env}sbatch --parsable -t {j['walltime']} {extra}"
                  f"-o $WORK/cog/logs/%x-%j.out {j['sbatch']} {argstr}")
        cmd = ["ssh", args.remote, submit] if args.remote else ["bash", "-c", submit]
        if args.dry_run:
            print(f"  DRY {submit}")
            continue
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            print(f"  FAILED to submit {argstr}: {out.stderr.strip()}")
            continue
        jobid = out.stdout.strip().split(";")[0].split()[-1]
        print(f"  submitted {args.stage} {argstr} as {jobid}")
        rows.append({
            "submitted_at": dt.datetime.now().isoformat(timespec="seconds"),
            "stage": args.stage, "task": args.task, "arm": j["arm"], "key": j.get("key", ""),
            "stem": j.get("stem", ""), "n_demos": j.get("n_demos", ""), "slice": j.get("slice", ""),
            "seed": j.get("seed", ""), "trials": j.get("trials", ""), "sbatch": j["sbatch"],
            "walltime": j["walltime"], "slurm_jobid": jobid, "notes": "launch_wave.py",
        })

    if rows:
        new = not LEDGER.exists()
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS)
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"appended {len(rows)} rows to {LEDGER.relative_to(REPO)}")
    elif not args.dry_run:
        print("nothing submitted")


if __name__ == "__main__":
    main()
