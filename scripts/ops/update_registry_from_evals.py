#!/usr/bin/env python3
"""Fill registry rows from eval result JSONs (and optionally Slurm elapsed -> gpu_h).

Why a script and not hand-edits: 24 cells x (SR, eval_n, status, gpu_h) is 96 fields, and the
registry is the single source the report's numbers must be traceable to (CLAUDE.md rule 6). Hand
editing a CSV that many times is how a transposed row happens.

D24: full-scale runs evaluate the LAST checkpoint only, so sr_40k/sr_60k stay empty and
sr_best == sr_80k. That is not a placeholder -- it is the protocol.

Idempotent: re-running only rewrites fields it can source from an artifact, and never downgrades a
populated field to empty.

usage:
  update_registry_from_evals.py [--gpu-h]     # --gpu-h queries sacct over ssh for elapsed times
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "experiments" / "registry.csv"
RESULTS = REPO / "results"

RUN_RE = re.compile(r"^t(?P<t>[123])_(?P<lvl>L\d)_n(?P<n>\d+)_s0$")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- the plan's chosen CI, valid near p=0 and p=1 unlike normal approx."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - m) / d, (c + m) / d)


def load_results() -> dict[str, dict]:
    """run_id -> parsed eval JSON for the 080000 checkpoint."""
    out = {}
    for f in sorted(RESULTS.glob("eval_T*_L*_n*_080000.json")):
        if "sharedenc" in f.name:          # superseded architecture; never mix into the matrix
            continue
        m = re.match(r"eval_(T[123])_(L\d)_n(\d+)_080000\.json$", f.name)
        if not m:
            continue
        try:
            rid = f"{m.group(1).lower()}_{m.group(2)}_n{m.group(3)}_s0"
            out[rid] = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            print(f"  WARN unreadable {f.name}: {e}", file=sys.stderr)
    return out


def elapsed_to_hours(s: str) -> float | None:
    """Slurm Elapsed is [DD-]HH:MM:SS."""
    s = s.strip()
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(x) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return days * 24 + h + m / 60 + sec / 3600


def fetch_gpu_h() -> dict[str, float]:
    """jobid -> GPU-h. On Leonardo a 1-GPU cell allocates billing=8 cores, and 8 billing-h = 1
    A100-h, so GPU-h == elapsed hours exactly. Verified against the calibration run (02:00:04)."""
    cmd = ["ssh", "leonardo",
           "sacct -X -u $USER --starttime 2026-08-19 -n -P -o JobID,JobName,State,Elapsed"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"  WARN sacct failed ({e}); skipping gpu_h", file=sys.stderr)
        return {}
    got = {}
    for line in res.stdout.splitlines():
        f = line.split("|")
        if len(f) < 4 or f[1] != "cog_train":
            continue
        hrs = elapsed_to_hours(f[3])
        if hrs:
            got[f[0].strip()] = round(hrs, 2)
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-h", action="store_true", help="also fill gpu_h from sacct over ssh")
    args = ap.parse_args()

    results = load_results()
    print(f"found {len(results)} new-architecture eval results")
    gpu_h = fetch_gpu_h() if args.gpu_h else {}
    if args.gpu_h:
        print(f"fetched elapsed for {len(gpu_h)} training jobs")

    with REGISTRY.open() as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())

    changed = 0
    for row in rows:
        rid = row["run_id"]
        if not RUN_RE.match(rid):
            continue

        jid = (row.get("slurm_jobid") or "").strip()
        if args.gpu_h and jid in gpu_h:
            # sacct is authoritative, and Elapsed only ever GROWS -- a row written while the job was
            # still RUNNING holds a partial value that must be refreshed once it completes. So take
            # the sacct number whenever it is larger than what is recorded (never smaller, which
            # would mean we are reading a different job).
            new_h = gpu_h[jid]
            cur = (row.get("gpu_h") or "").strip()
            try:
                cur_h = float(cur) if cur else -1.0
            except ValueError:
                cur_h = -1.0
            if new_h > cur_h:
                row["gpu_h"] = f"{new_h}"
                changed += 1

        d = results.get(rid)
        if not d:
            continue
        sr = d.get("success_rate")
        n = d.get("episodes")
        k = d.get("successes")
        if sr is None:
            continue
        lo, hi = wilson(int(k), int(n)) if (k is not None and n) else (None, None)

        new = {
            "sr_80k": f"{sr}",
            "sr_best": f"{sr}",          # D24: last checkpoint only, so best == last by definition
            "eval_n": f"{n}",
            "status": "done",
        }
        note = (f"eval 080000 only (D24); {k}/{n} successes; "
                f"Wilson95 [{lo:.3f},{hi:.3f}]; num_inference_steps=10; "
                f"sep_rgb_encoder_per_camera=true (D26)")
        if note not in (row.get("notes") or ""):
            row["notes"] = ((row.get("notes") or "").strip() + " | " + note).lstrip(" |")
        for kk, vv in new.items():
            if row.get(kk) != vv:
                row[kk] = vv
                changed += 1
        print(f"  {rid}: SR={sr} ({k}/{n}) Wilson95 [{lo:.3f},{hi:.3f}]")

    with REGISTRY.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(rows)

    print(f"registry updated: {changed} field(s) changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
