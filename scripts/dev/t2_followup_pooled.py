import json
from pathlib import Path

from scipy.stats import binomtest

R = Path("results")
d = {k: json.load(open(R / v))["outcomes"] for k, v in {
    "L1onL1": "eval_T2_L1_n400_080000_stages.json",
    "L2onL1": "diagnostics/eval_T2_xeval_L2n400_onL1_080000.json",
    "L1onL2": "diagnostics/eval_T2_xeval_L1n400_onL2_080000.json",
    "L2onL2": "eval_T2_L2_n400_080000_stages.json"}.items()}
key = lambda o: (o["batch"], o["env"])
S = {k: {key(o): o["success"] for o in v} for k, v in d.items()}

pol_a = sum(S["L2onL1"][k] and not S["L1onL1"][k] for k in S["L1onL1"]) \
    + sum(S["L2onL2"][k] and not S["L1onL2"][k] for k in S["L1onL2"])
pol_b = sum(S["L1onL1"][k] and not S["L2onL1"][k] for k in S["L1onL1"]) \
    + sum(S["L1onL2"][k] and not S["L2onL2"][k] for k in S["L1onL2"])
set_a = sum(S["L1onL2"][k] and not S["L1onL1"][k] for k in S["L1onL1"]) \
    + sum(S["L2onL2"][k] and not S["L2onL1"][k] for k in S["L2onL1"])
set_b = sum(S["L1onL1"][k] and not S["L1onL2"][k] for k in S["L1onL1"]) \
    + sum(S["L2onL1"][k] and not S["L2onL2"][k] for k in S["L2onL1"])
print("pooled policy effect: only-L2pol", pol_a, "only-L1pol", pol_b,
      "p=", binomtest(pol_a, pol_a + pol_b, 0.5).pvalue)
print("pooled set effect: only-L2set", set_a, "only-L1set", set_b,
      "p=", binomtest(set_a, set_a + set_b, 0.5).pvalue)

for k, v in d.items():
    ts = [o["t_success"] for o in v if o["success"]]
    nz = sorted(t for t in ts if t > 0)
    print(k, "t_success: zeros", sum(t == 0 for t in ts), "of", len(ts),
          "nonzero median", nz[len(nz) // 2] if nz else None)
