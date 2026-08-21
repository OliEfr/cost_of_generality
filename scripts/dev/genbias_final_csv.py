"""Quantify the T3_L1 reject strip + T2 pose-blind baseline; stamp verdicts into the CSV."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/admin_07/cost_of_generality/.claude/worktrees/results-analysis/scripts/dev')
from genbias_link_stats import gen_attempts, OUT_EXP

# --- T3 L1: how deep is the coverage thinning where the 22 rejects sit?
att = gen_attempts('T3', 'L1').drop_duplicates(subset=['obj_x', 'obj_y', 'retained'])
strip = att[att.obj_y > -0.088]          # min reject y
n_rej = int((~strip.retained).sum())
n_ret = int(strip.retained.sum())
print(f"T3 L1 strip y>-0.088 (top {((-0.04) - (-0.088)) / 0.12:.0%} of y-range): "
      f"{n_ret} retained, {n_rej} rejected -> local rejection {n_rej/(n_ret+n_rej):.1%} "
      f"(vs 0% below the strip). Retained demos still present: {n_ret}")
ep = pd.read_csv(OUT_EXP / 'genbias_link_episodes.csv')
j = ep[(ep.task == 'T3') & (ep.level == 'L1') & (ep.n_demos == 400)]
print(f"T3 L1 eval: max kNN local_rej seen by any eval episode = {j.local_rej.max():.2f}; "
      f"episodes in strip = {(j.obj_y > -0.088).sum()}/100, "
      f"SR in strip = {j[j.obj_y > -0.088].success.mean():.3f} "
      f"vs outside = {j[j.obj_y <= -0.088].success.mean():.3f}")
f4 = j[~j.success.astype(bool)]
print(f"the {len(f4)} failures sit at y = {np.sort(f4.obj_y.round(3).to_numpy())}")

# --- T2: pose-blind baseline decomposition (L0 rejects with a single fixed pose)
print("\npose-selective rejection mass (level rejection minus L0 pose-blind baseline):")
for task, base in (('T1', 0.1361), ('T2', 0.4505), ('T3', 0.0148)):
    for lv, rej in [(r['level'], r['rejection_rate']) for _, r in
                    pd.read_csv(OUT_EXP / 'genbias_link.csv')
                    .query(f"task=='{task}'").groupby('level').first().reset_index().iterrows()]:
        if lv == 'L0':
            continue
        print(f"  {task} {lv}: total {rej:.1%} - baseline {base:.1%} = "
              f"pose-attributable <= {max(rej-base,0):.1%}")

# --- stamp verdicts
VERDICTS = {
    ('T1', 'L0'): 'exempt (fixed pose)',
    ('T1', 'L1'): 'absent',
    ('T1', 'L2'): 'absent',
    ('T1', 'L3b'): 'absent',
    ('T2', 'L0'): 'exempt (fixed pose)',
    ('T2', 'L1'): 'present-but-immaterial',
    ('T2', 'L2'): 'present-but-immaterial',
    ('T2', 'L3b'): 'present-but-immaterial',
    ('T3', 'L0'): 'exempt (fixed pose)',
    ('T3', 'L1'): 'present-but-immaterial (suggestive link p=0.09, bounded <=5pts)',
    ('T3', 'L2'): 'present-but-immaterial',
    ('T3', 'L3b'): 'present-but-immaterial (shared difficulty, not coverage: NND null)',
}
df = pd.read_csv(OUT_EXP / 'genbias_link.csv')
df['verdict'] = [VERDICTS[(t, l)] for t, l in zip(df.task, df.level)]
df.to_csv(OUT_EXP / 'genbias_link.csv', index=False)
print(f"\nverdicts stamped into {OUT_EXP/'genbias_link.csv'}")
