"""T2 with 'drawer opened' (joint >= 0.15, the success criterion's own threshold) as the task."""
import json
from pathlib import Path

from scipy.stats import beta


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return (c - h) / d, (c + h) / d


R = Path("results")
RUNS = [
    ("L0 policy -> L0 set", "eval_T2_L0_n400_080000_stages.json"),
    ("L1 policy -> L1 set", "eval_T2_L1_n400_080000_stages.json"),
    ("L2 policy -> L2 set", "eval_T2_L2_n400_080000_stages.json"),
    ("L2 policy -> L1 set", "diagnostics/eval_T2_xeval_L2n400_onL1_080000.json"),
    ("L1 policy -> L2 set", "diagnostics/eval_T2_xeval_L1n400_onL2_080000.json"),
]
for name, fn in RUNS:
    out = json.load(open(R / fn))["outcomes"]
    n = len(out)
    k = sum(o["drawer_opened"] for o in out)
    stow = sum(o["success"] and o["object_over_drawer"] for o in out)
    t = sorted(o["t_open"] for o in out if o["t_open"] >= 0)
    lo, hi = wilson(k, n)
    print(f"{name}: opened {k}/{n} = {k/n:.2f}  Wilson[{lo:.2f},{hi:.2f}]  "
          f"median t_open {t[len(t)//2]}  (full-task corrected SR {stow/n:.2f})")
