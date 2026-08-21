"""Analyse the T2 follow-up evals: 2x2 policy-x-evalset decomposition + stage funnels.

Stage-latch caveat: Isaac auto-resets a terminated env inside step(), so the stage reads
miss the terminal step and undercount. success implies drawer-open and object-in-cavity-xy
by the success criterion itself, and physically implies a lift; report the per-episode
union (stage | success) as the corrected rate.
"""
import json
from pathlib import Path

from scipy.stats import binomtest

R = Path("results")
RUNS = {
    "L1pol_on_L1": "eval_T2_L1_n400_080000_stages.json",
    "L2pol_on_L1": "eval_T2_xeval_L2n400_onL1_080000.json",
    "L1pol_on_L2": "eval_T2_xeval_L1n400_onL2_080000.json",
    "L2pol_on_L2": "eval_T2_L2_n400_080000_stages.json",
    "L0pol_on_L0": "eval_T2_L0_n400_080000_stages.json",
}
PUBLISHED = {"L1pol_on_L1": "eval_T2_L1_n400_080000.json", "L2pol_on_L2": "eval_T2_L2_n400_080000.json"}

data = {k: json.load(open(R / v)) for k, v in RUNS.items()}
key = lambda o: (o["batch"], o["env"])

rows = []
for name, d in data.items():
    out = d["outcomes"]
    n = len(out)
    sr = sum(o["success"] for o in out) / n
    opened = sum(o["drawer_opened"] or o["success"] for o in out) / n
    lifted = sum(o["object_lifted"] or o["success"] for o in out) / n
    over = sum(o["object_over_drawer"] or o["success"] for o in out) / n
    n_open = [o for o in out if o["drawer_opened"] or o["success"]]
    n_lift = [o for o in out if o["object_lifted"] or o["success"]]
    n_over = [o for o in out if o["object_over_drawer"] or o["success"]]
    p_lift_g_open = (sum(o["object_lifted"] or o["success"] for o in n_open) / len(n_open)) if n_open else float("nan")
    p_succ_g_over = (sum(o["success"] for o in n_over) / len(n_over)) if n_over else float("nan")
    t_open = sorted(o["t_open"] for o in out if o["t_open"] >= 0)
    t_succ = sorted(o["t_success"] for o in out if o["t_success"] >= 0)
    med = lambda xs: xs[len(xs) // 2] if xs else -1
    rows.append(
        dict(run=name, n=n, sr=sr, opened=opened, lifted=lifted, over=over,
             p_lift_given_open=round(p_lift_g_open, 3), p_succ_given_over=round(p_succ_g_over, 3),
             med_t_open=med(t_open), med_t_success=med(t_succ),
             raw_opened=sum(o["drawer_opened"] for o in out) / n,
             raw_lifted=sum(o["object_lifted"] for o in out) / n,
             raw_over=sum(o["object_over_drawer"] for o in out) / n)
    )
    print(f"{name}: SR={sr:.2f} funnel opened={opened:.2f} lifted={lifted:.2f} over={over:.2f} "
          f"P(lift|open)={p_lift_g_open:.2f} P(succ|over)={p_succ_g_over:.2f} "
          f"med_t_open={med(t_open)} med_t_succ={med(t_succ)}")

def mcnemar(a, b, la, lb):
    sa = {key(o): o["success"] for o in a}
    sb = {key(o): o["success"] for o in b}
    ks = sorted(set(sa) & set(sb))
    only_a = sum(sa[k] and not sb[k] for k in ks)
    only_b = sum(sb[k] and not sa[k] for k in ks)
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    print(f"McNemar {la} vs {lb}: only-{la}={only_a} only-{lb}={only_b} p={p:.4f}")
    return only_a, only_b, p

print("\n-- policy effect at fixed eval set (episode-paired: identical initial conditions) --")
mcnemar(data["L2pol_on_L1"]["outcomes"], data["L1pol_on_L1"]["outcomes"], "L2pol", "L1pol")
mcnemar(data["L2pol_on_L2"]["outcomes"], data["L1pol_on_L2"]["outcomes"], "L2pol", "L1pol")
print("-- eval-set effect at fixed policy (paired by identical OBJECT pose; cabinet differs) --")
mcnemar(data["L1pol_on_L2"]["outcomes"], data["L1pol_on_L1"]["outcomes"], "L2set", "L1set")
mcnemar(data["L2pol_on_L2"]["outcomes"], data["L2pol_on_L1"]["outcomes"], "L2set", "L1set")

print("\n-- rerun vs published (same policy, same set, same seeds: determinism drift) --")
for name, pub in PUBLISHED.items():
    d0 = json.load(open(R / pub))
    s0 = {key(o): o["success"] for o in d0["outcomes"]}
    s1 = {key(o): o["success"] for o in data[name]["outcomes"]}
    agree = sum(s0[k] == s1[k] for k in s0) / len(s0)
    print(f"{name}: published {sum(s0.values())}/100 rerun {sum(s1.values())}/100 episode agreement {agree:.2f}")

import csv
with open("experiments/t2_followup.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("\nwrote experiments/t2_followup.csv")
