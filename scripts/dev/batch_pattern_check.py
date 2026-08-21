"""Mechanism checks for the batch-boundary success-carryover bug.

(1) staleness confined to t==0? -> min nonzero t_success across stage runs
(2) published flat cells: is batch-0 SR systematically below batches 1-4, and does the
    (batch_b - batch0) gap track the previous batch's SR? Pooled sign test across cells.
(3) carryover rate in stage runs: stale zeros in batch b vs TRUE successes in batch b-1
    (true = success & raw object_over_drawer).
"""
import json
import re
from pathlib import Path

from scipy.stats import binomtest, wilcoxon

R = Path("results")

print("== (1) min nonzero t_success / t staleness window ==")
for fn in R.glob("eval_T2_*stages*.json"):
    out = json.load(open(fn))["outcomes"]
    nz = sorted(o["t_success"] for o in out if o["success"] and o["t_success"] > 0)
    print(f"{fn.name}: min nonzero t_success = {nz[0] if nz else None}")
for fn in R.glob("eval_T2_xeval_*.json"):
    out = json.load(open(fn))["outcomes"]
    nz = sorted(o["t_success"] for o in out if o["success"] and o["t_success"] > 0)
    print(f"{fn.name}: min nonzero t_success = {nz[0] if nz else None}")

print("\n== (3) carryover: stale zeros in batch b vs true successes in batch b-1 ==")
for fn in list(R.glob("eval_T2_*stages*.json")) + list(R.glob("eval_T2_xeval_*.json")):
    out = json.load(open(fn))["outcomes"]
    for b in range(1, 5):
        prev_true = sum(o["success"] and o["object_over_drawer"] for o in out if o["batch"] == b - 1)
        zeros = sum(o["success"] and o["t_success"] == 0 for o in out if o["batch"] == b)
        print(f"  {fn.name} b{b}: zeros={zeros} prev_true={prev_true}")

print("\n== (2) published flat cells: batch0 vs batches 1-4 ==")
gaps = []
for fn in sorted(R.glob("eval_T*_n*_080000.json")):
    m = re.match(r"eval_(T\d)_(L\d)_n(\d+)_080000\.json", fn.name)
    if not m:
        continue
    d = json.load(open(fn))
    out = d["outcomes"]
    if len({o["batch"] for o in out}) != 5:
        continue
    srb = [sum(o["success"] for o in out if o["batch"] == b) / 20 for b in range(5)]
    rest = sum(srb[1:]) / 4
    if 0.05 <= d["success_rate"] <= 0.97:  # cells with room to show the effect
        gaps.append(rest - srb[0])
    print(f"{fn.name}: per-batch {srb} b0={srb[0]:.2f} rest={rest:.2f}")
pos = sum(g > 0 for g in gaps)
neg = sum(g < 0 for g in gaps)
print(f"\nnon-saturated cells: rest>b0 in {pos}, rest<b0 in {neg}, ties {len(gaps)-pos-neg}; "
      f"sign-test p={binomtest(pos, pos+neg, 0.5).pvalue if pos+neg else 1}")
print(f"mean gap (rest - batch0) = {sum(gaps)/len(gaps):.3f}")
try:
    print("wilcoxon:", wilcoxon(gaps))
except Exception as e:
    print("wilcoxon failed:", e)
