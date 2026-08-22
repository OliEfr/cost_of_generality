"""Warm-up-matched finding-4 comparison: flat levels restricted to their FIRST batch
(batch 0, n=20/cell) against L3 cells (every episode is a process's first batch, n=200).
Also the warm counterpart: flat batches 1-4 only. Clean data throughout."""
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

R = Path("results")
NS = [10, 25, 50, 100, 200, 400]

b0 = defaultdict(dict)
warm = defaultdict(dict)
for fp in glob.glob(str(R / "eval_T*_080000_fixed.json")):
    m = re.match(r"eval_(T\d)_(L\d)_n(\d+)_080000_fixed\.json", Path(fp).name)
    if not m:
        continue
    out = json.load(open(fp))["outcomes"]
    key = (m.group(1), m.group(2))
    n = int(m.group(3))
    b0[key][n] = sum(o["success"] for o in out if o["batch"] == 0) / 20
    warm[key][n] = sum(o["success"] for o in out if o["batch"] > 0) / 80

l3 = defaultdict(dict)
for task, src in [("T1", R), ("T3", R)]:
    for n in NS:
        l3[task][n] = json.load(open(src / f"eval_{task}_L3b_n{n}_080000.json"))["success_rate"]
for n in NS:
    l3["T2"][n] = json.load(open(R / "diagnostics" / f"eval_T2_L3b_n{n}_080000_fixed.json"))["success_rate"]

print("first-batch-matched comparison (flat = batch-0 only, n=20/cell; L3 = as measured, n=200):")
for task in ("T1", "T2", "T3"):
    print(f"  {task}  N:      " + "  ".join(f"{n:>4}" for n in NS))
    for lvl in ("L1", "L2"):
        print(f"    {lvl} b0-only: " + "  ".join(f"{b0[(task,lvl)].get(n, float('nan')):.2f}" for n in NS))
    print(f"    L3 (all-b0): " + "  ".join(f"{l3[task][n]:.2f}" for n in NS))
    for lvl in ("L1", "L2"):
        print(f"    {lvl} warm:    " + "  ".join(f"{warm[(task,lvl)].get(n, float('nan')):.2f}" for n in NS))

rows = []
for task in ("T1", "T2", "T3"):
    for n in NS:
        rows.append(dict(task=task, n_demos=n,
                         L1_b0=b0[(task, "L1")].get(n), L2_b0=b0[(task, "L2")].get(n),
                         L3=l3[task][n],
                         L1_warm=warm[(task, "L1")].get(n), L2_warm=warm[(task, "L2")].get(n)))
with open("experiments/warmup_matched_f4.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote experiments/warmup_matched_f4.csv")
