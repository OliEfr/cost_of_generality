"""Is t_success==0 a stale termination-buffer read at batch boundaries?

Discriminators:
  (a) batch 0 has nothing to be stale from -> zeros should only appear in batches 1-4;
  (b) a genuine success passes through the cavity for many pre-terminal steps -> its
      raw object_over_drawer latch (scene-buffer read, fresh) should be True; a pure
      artifact episode should have raw over == False (and raw lifted == False);
  (c) L0 (SR 0.94, policy near-memorised) shows what genuine successes look like.
"""
import json
from pathlib import Path

R = Path("results")
RUNS = {
    "L1onL1": "eval_T2_L1_n400_080000_stages.json",
    "L2onL1": "diagnostics/eval_T2_xeval_L2n400_onL1_080000.json",
    "L1onL2": "diagnostics/eval_T2_xeval_L1n400_onL2_080000.json",
    "L2onL2": "eval_T2_L2_n400_080000_stages.json",
    "L0onL0": "eval_T2_L0_n400_080000_stages.json",
}
for name, fn in RUNS.items():
    out = json.load(open(R / fn))["outcomes"]
    succ = [o for o in out if o["success"]]
    zeros = [o for o in succ if o["t_success"] == 0]
    per_batch = [sum(o["batch"] == b for o in zeros) for b in range(5)]
    z_no_over = sum(not o["object_over_drawer"] for o in zeros)
    z_no_lift = sum(not o["object_lifted"] for o in zeros)
    nz = [o for o in succ if o["t_success"] > 0]
    nz_over = sum(o["object_over_drawer"] for o in nz)
    print(f"{name}: succ={len(succ)} zeros={len(zeros)} per-batch={per_batch} "
          f"zeros-without-raw-over={z_no_over} zeros-without-raw-lift={z_no_lift} "
          f"| nonzero succ={len(nz)} with-raw-over={nz_over}")
    corrected = len(nz) + (len(zeros) - z_no_over)
    print(f"   corrected SR if over-less zeros are artifacts: {corrected}/{len(out)} = {corrected/len(out):.2f}")
