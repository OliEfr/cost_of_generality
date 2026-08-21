"""Follow-up: cell-level significance, paired overlap, per-batch sanity, N=25 oddity."""

import json
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

WT = pathlib.Path("/home/admin_07/cost_of_generality/.claude/worktrees/results-analysis")
NS = [10, 25, 50, 100, 200, 400]


def load(level):
    rows = []
    for n in NS:
        d = json.loads((WT / f"results/eval_T2_{level}_n{n}_080000.json").read_text())
        for o in d["outcomes"]:
            rows.append((n, o["batch"], o["env"], bool(o["success"])))
    return pd.DataFrame(rows, columns=["n", "batch", "env", "success"])


l1, l2 = load("L1"), load("L2")
print("batches used:", sorted(l1.batch.unique()), sorted(l2.batch.unique()))

print("\nper-batch SR (rows=N), L1 | L2 -- looking for a dead batch:")
for lvl, df in (("L1", l1), ("L2", l2)):
    piv = df.pivot_table(index="n", columns="batch", values="success", aggfunc="mean")
    print(lvl); print(piv.round(2).to_string())

# cell-level test: 6 paired (L2-L1) SR differences, unit = training run
d1 = l1.groupby("n").success.mean()
d2 = l2.groupby("n").success.mean()
diff = (d2 - d1).reindex(NS)
print("\ncell-level paired differences (L2-L1):", (diff * 100).round(1).tolist())
t, p = stats.ttest_rel(d2.reindex(NS), d1.reindex(NS))
w = stats.wilcoxon(diff)
print(f"paired t-test over 6 cells: t={t:.2f} p={p:.4f}; Wilcoxon p={w.pvalue:.4f}; "
      f"sign 5/6 binom p={stats.binomtest(5, 6).pvalue:.3f}")

# within-level adjacent-N jumps: how big is single-training-run noise?
print("\nadjacent-N Fisher tests WITHIN each level (same data distribution, nested subsets):")
for lvl, df in (("L1", l1), ("L2", l2)):
    ks = df.groupby("n").success.sum().reindex(NS)
    for a, b in zip(NS[:-1], NS[1:]):
        _, pf = stats.fisher_exact([[ks[a], 100 - ks[a]], [ks[b], 100 - ks[b]]])
        flag = " <-- " if pf < 0.05 else ""
        print(f"  {lvl} n{a}({ks[a]}) vs n{b}({ks[b]}): Fisher p={pf:.4f}{flag}")

# L2 n10 vs n25
_, pf = stats.fisher_exact([[17, 83], [11, 89]])
print(f"\nL2 n10 (17) vs n25 (11): Fisher p={pf:.4f}")

# paired overlap on identical object poses, pooled N>=50
big = [50, 100, 200, 400]
for n in big + ["pooled"]:
    if n == "pooled":
        a = l1[l1.n.isin(big)].set_index(["n", "batch", "env"]).success
        b = l2[l2.n.isin(big)].set_index(["n", "batch", "env"]).success
    else:
        a = l1[l1.n == n].set_index(["batch", "env"]).success
        b = l2[l2.n == n].set_index(["batch", "env"]).success
    both = int((a & b).sum()); only2 = int((~a & b).sum())
    only1 = int((a & ~b).sum()); neither = int((~a & ~b).sum())
    mcn = stats.binomtest(only2, only1 + only2).pvalue if (only1 + only2) else np.nan
    print(f"N={n}: both {both}, only-L2 {only2}, only-L1 {only1}, neither {neither}, "
          f"McNemar-exact p={mcn:.4f}")
