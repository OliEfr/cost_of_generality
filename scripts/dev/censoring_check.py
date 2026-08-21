"""Is `terminated` ALSO stale at t=0 (censoring carried envs as phantom failures)?

Under censoring, in the guard-fixed clean runs:
  (a) an env that succeeded in batch b-1 can never be recorded success in batch b
      -> zero consecutive-success pairs across env indices;
  (b) such envs' episodes end at t=0 -> max_drawer_open / max_object_lift stuck at the
      single post-reset read (~0), and all stage latches false.
Also check the pre-guard t2fu stage runs for comparison (there, carried envs are phantom
successes; consecutive recorded successes CAN occur).
"""
import json
from pathlib import Path

R = Path("results")
FILES = {
    "clean n400": "eval_T2_L1_n400_080000_fixed.json",
    "clean n200": "eval_T2_L1_n200_080000_fixed.json",
    "clean n100": "eval_T2_L1_n100_080000_fixed.json",
    "clean n50": "eval_T2_L1_n50_080000_fixed.json",
    "pre-guard L1n400 (t2fu)": "eval_T2_L1_n400_080000_stages.json",
}
for label, fn in FILES.items():
    out = json.load(open(R / fn))["outcomes"]
    S = {(o["batch"], o["env"]): o for o in out}
    consec = sum(
        1 for b in range(1, 5) for e in range(20)
        if S[(b - 1, e)]["success"] and S[(b, e)]["success"]
    )
    prev_succ_total = sum(1 for b in range(1, 5) for e in range(20) if S[(b - 1, e)]["success"])
    # carried candidates: env failed-in-b after success-in-b-1 -> inspect their maxima
    carried = [S[(b, e)] for b in range(1, 5) for e in range(20)
               if S[(b - 1, e)]["success"] and not S[(b, e)]["success"]]
    frozen = sum(1 for o in carried
                 if o.get("max_drawer_open", 1) < 0.01 and abs(o.get("max_object_lift", 1)) < 0.005
                 and not o.get("drawer_opened", True))
    # baseline: fraction of ordinary failures (no prev success) that look frozen
    ordinary = [S[(b, e)] for b in range(1, 5) for e in range(20)
                if not S[(b - 1, e)]["success"] and not S[(b, e)]["success"]]
    frozen_ord = sum(1 for o in ordinary
                     if o.get("max_drawer_open", 1) < 0.01 and abs(o.get("max_object_lift", 1)) < 0.005
                     and not o.get("drawer_opened", True))
    print(f"{label}: consecutive-success pairs={consec}/{prev_succ_total} prev-successes; "
          f"post-success failures frozen-at-t0: {frozen}/{len(carried)}; "
          f"ordinary failures frozen: {frozen_ord}/{len(ordinary)}")
