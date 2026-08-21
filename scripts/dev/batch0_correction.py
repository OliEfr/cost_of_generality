"""Batch-boundary staleness correction: batch-0-only SR for every published cell.

The success latch reads a stale termination buffer on the first step of batches 1..4
(t2_t0_artifact_check.py: zeros never in batch 0; artifact episodes never lift the
object). Batch 0 has no previous batch -> unbiased, n=20. L3 cells ran one batch per
variant in fresh processes -> fully clean, keep n=200. Validation: in the five
stage-instrumented runs, batch-0 SR should match the stage-corrected SR (successes
with a raw object_over_drawer latch).
"""
import json
import re
from pathlib import Path

from scipy.stats import binomtest

R = Path("results")

print("== validation on stage runs: batch0 vs stage-corrected ==")
for fn in ["eval_T2_L1_n400_080000_stages.json", "eval_T2_xeval_L2n400_onL1_080000.json",
           "eval_T2_xeval_L1n400_onL2_080000.json", "eval_T2_L2_n400_080000_stages.json",
           "eval_T2_L0_n400_080000_stages.json"]:
    out = json.load(open(R / fn))["outcomes"]
    b0 = [o for o in out if o["batch"] == 0]
    sr_b0 = sum(o["success"] for o in b0) / len(b0)
    corr = sum(o["success"] and o["object_over_drawer"] for o in out) / len(out)
    per_batch_corr = [
        sum(o["success"] and o["object_over_drawer"] for o in out if o["batch"] == b) / 20
        for b in range(5)
    ]
    print(f"{fn}: batch0 SR={sr_b0:.2f} (n=20)  stage-corrected SR={corr:.2f} (n=100) "
          f"per-batch corrected={per_batch_corr}")

print("\n== batch-0-only SR for all published flat cells (n=20 each) ==")
rows = []
for fn in sorted(R.glob("eval_T*_n*_080000.json")):
    m = re.match(r"eval_(T\d)_(L\d+b?)_n(\d+)_080000\.json", fn.name)
    if not m:
        continue
    task, level, n_demos = m.group(1), m.group(2), int(m.group(3))
    d = json.load(open(fn))
    out = d["outcomes"]
    pub = d["success_rate"]
    if level in ("L3", "L3b"):
        # one batch per variant, fresh process each -> clean as published
        rows.append((task, level, n_demos, pub, pub, len(out), "clean (1 batch/variant)"))
        continue
    b0 = [o for o in out if o["batch"] == 0]
    sr0 = sum(o["success"] for o in b0) / len(b0)
    rows.append((task, level, n_demos, pub, sr0, len(b0), ""))

rows.sort(key=lambda r: (r[0], r[1], r[2]))
print(f"{'cell':<16}{'published':>10}{'batch0':>8}{'n':>5}  note")
for task, level, nd, pub, sr0, n, note in rows:
    flag = " <-- drop >0.10" if pub - sr0 > 0.10 else ""
    print(f"{task}_{level}_n{nd:<6}{pub:>10.2f}{sr0:>8.2f}{n:>5}  {note}{flag}")

import csv
with open("experiments/batch0_corrected_sr.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "level", "n_demos", "sr_published", "sr_batch0", "n_batch0", "note"])
    w.writerows(rows)
print("\nwrote experiments/batch0_corrected_sr.csv")
