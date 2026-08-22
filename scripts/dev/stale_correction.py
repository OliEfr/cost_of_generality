"""Invert the batch-boundary success-carryover bug on published eval JSONs.

Mechanism (established in t2_t0_artifact_check.py / batch_pattern_check.py):
  recorded(b, env) = true(b, env) OR true(b-1, env)      [b >= 1; recorded(0)=true(0)]
i.e. every env that GENUINELY succeeded in batch b-1 gets a phantom success latched on
the first step of batch b (carryover rate 20/20 transitions, exact).

Constraint propagation per env chain:
  recorded(b)=0  ->  true(b)=0 AND true(b-1)=0   (propagates backwards)
  recorded(b)=1 and true(b-1)=0 known  ->  true(b)=1
Unresolved: runs of recorded=1 whose predecessor was a genuine success. Report
  lower bound  = all unknowns false
  upper bound  = all unknowns true (= published)
  point est    = known-true / known-count (unknowns excluded; unbiased if success is
                 independent of env index within a cell)

Validation: on the five stage-instrumented runs, ground truth = success AND raw
object_over_drawer (artifact episodes never reach the drawer).
"""
import json
import re
from pathlib import Path

R = Path("results")
UNKNOWN, TRUE, FALSE = -1, 1, 0


def solve(outcomes):
    envs = sorted({o["env"] for o in outcomes})
    batches = sorted({o["batch"] for o in outcomes})
    rec = {(o["batch"], o["env"]): o["success"] for o in outcomes}
    true = {}
    for e in envs:
        # forward init + backward zero-propagation, then forward resolution
        for b in batches:
            true[(b, e)] = UNKNOWN
        true[(0, e)] = TRUE if rec[(0, e)] else FALSE
        for b in batches:
            if not rec[(b, e)]:
                true[(b, e)] = FALSE
                if b - 1 in batches:
                    true[(b - 1, e)] = FALSE
        for b in batches[1:]:
            if rec[(b, e)] and true[(b - 1, e)] == FALSE:
                true[(b, e)] = TRUE
    vals = list(true.values())
    n = len(vals)
    k_true = sum(v == TRUE for v in vals)
    k_unk = sum(v == UNKNOWN for v in vals)
    known = n - k_unk
    return dict(n=n, lower=k_true / n, upper=(k_true + k_unk) / n,
                point=k_true / known if known else float("nan"),
                n_known=known, n_unknown=k_unk)


print("== validation on stage runs (truth = success & raw over) ==")
for fn in ["eval_T2_L1_n400_080000_stages.json", "diagnostics/eval_T2_xeval_L2n400_onL1_080000.json",
           "diagnostics/eval_T2_xeval_L1n400_onL2_080000.json", "eval_T2_L2_n400_080000_stages.json",
           "eval_T2_L0_n400_080000_stages.json"]:
    out = json.load(open(R / fn))["outcomes"]
    truth = sum(o["success"] and o["object_over_drawer"] for o in out) / len(out)
    s = solve(out)
    ok = s["lower"] - 1e-9 <= truth <= s["upper"] + 1e-9
    print(f"{fn}: truth={truth:.3f} point={s['point']:.3f} "
          f"bounds=[{s['lower']:.2f},{s['upper']:.2f}] known={s['n_known']} {'OK' if ok else 'VIOLATED'}")

print("\n== corrected SR for all published cells ==")
rows = []
for fn in sorted(R.glob("eval_T*_n*_080000.json")):
    m = re.match(r"eval_(T\d)_(L\d+b?)_n(\d+)_080000\.json", fn.name)
    if not m:
        continue
    task, level, nd = m.group(1), m.group(2), int(m.group(3))
    d = json.load(open(fn))
    out = d["outcomes"]
    if level in ("L3", "L3b"):
        rows.append((task, level, nd, d["success_rate"], d["success_rate"],
                     d["success_rate"], d["success_rate"], len(out), 0, "clean"))
        continue
    s = solve(out)
    rows.append((task, level, nd, d["success_rate"], s["point"], s["lower"], s["upper"],
                 s["n_known"], s["n_unknown"], ""))

rows.sort(key=lambda r: (r[0], r[1], r[2]))
print(f"{'cell':<16}{'published':>10}{'corrected':>10}{'lower':>7}{'upper':>7}{'known':>6}")
for task, level, nd, pub, pt, lo, hi, nk, nu, note in rows:
    print(f"{task}_{level}_n{nd:<6}{pub:>10.2f}{pt:>10.3f}{lo:>7.2f}{hi:>7.2f}{nk:>6} {note}")

import csv
with open("experiments/stale_corrected_sr.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "level", "n_demos", "sr_published", "sr_corrected_point",
                "sr_lower", "sr_upper", "n_identified", "n_unknown", "note"])
    w.writerows(rows)
print("\nwrote experiments/stale_corrected_sr.csv")
