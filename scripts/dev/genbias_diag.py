"""Diagnostics: directions of the generator filter and of the success gradient."""
import json
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, '/home/admin_07/cost_of_generality/.claude/worktrees/results-analysis/scripts/dev')
from genbias_link_stats import (TASKS, gen_attempts, env_origins, eval_states,
                                outcomes, N_PRIMARY, N_SECONDARY)

EP = pd.read_csv('/home/admin_07/cost_of_generality/.claude/worktrees/results-analysis/experiments/genbias_link_episodes.csv')

print("=== retained vs rejected means (unique poses) ===")
for task in ('T1', 'T2', 'T3'):
    for level in ('L1', 'L2', 'L3b'):
        att = gen_attempts(task, level)
        dd = [d for d in TASKS[task]['demo_dims'] if d in att.columns
              and d in TASKS[task]['dims'][level]]
        u = att.drop_duplicates(subset=dd + ['retained'])
        r, f = u[u.retained], u[~u.retained]
        line = f"{task} {level}: n_rej={len(f)} "
        for d in dd:
            line += (f"| {d}: ret {r[d].mean():+.3f}±{r[d].std():.3f} "
                     f"rej {f[d].mean():+.3f}±{f[d].std():.3f} ")
        print(line)
        if 'obj_yaw' in dd:
            line = "        |yaw|: "
            line += (f"ret {r.obj_yaw.abs().mean():.3f} rej {f.obj_yaw.abs().mean():.3f} "
                     f"MW_p={stats.mannwhitneyu(r.obj_yaw.abs(), f.obj_yaw.abs()).pvalue:.2e}")
            # signed asymmetry
            line += f" | signed MW_p={stats.mannwhitneyu(r.obj_yaw, f.obj_yaw).pvalue:.2e}"
            print(line)
        if 'bearing' in dd and len(f) > 1:
            db = (r.bearing - np.pi / 2).abs(), (f.bearing - np.pi / 2).abs()
            print(f"        |bearing-90deg|: ret {db[0].mean():.3f} rej {db[1].mean():.3f} "
                  f"MW_p={stats.mannwhitneyu(db[0], db[1]).pvalue:.2e}")

print("\n=== T3 L1: where the 22 rejects sit ===")
att = gen_attempts('T3', 'L1')
u = att.drop_duplicates(subset=['obj_x', 'obj_y', 'retained'])
f = u[~u.retained]
print(f.groupby(f.obj_y > -0.10).size().rename('rej by y>-0.10'))
print('reject obj_x:', np.sort(f.obj_x.round(3).to_numpy()))
print('reject obj_y:', np.sort(f.obj_y.round(3).to_numpy()))
print('range decl: x (0.36,0.48) y (-0.16,-0.04)')

print("\n=== success vs |yaw| (T2) and |bearing-90| (T3), N=400 ===")
for task, level, col, ref in (('T2', 'L1', 'obj_yaw', 0.0), ('T2', 'L2', 'obj_yaw', 0.0),
                              ('T2', 'L3b', 'obj_yaw', 0.0), ('T3', 'L2', 'bearing', np.pi/2),
                              ('T3', 'L3b', 'bearing', np.pi/2)):
    j = EP[(EP.task == task) & (EP.level == level) & (EP.n_demos == 400)]
    v = (j[col] - ref).abs()
    s = j.success.astype(bool)
    if s.sum() < 2 or (~s).sum() < 2:
        continue
    mw = stats.mannwhitneyu(v[s], v[~s])
    rb = 2 * mw.statistic / (s.sum() * (~s).sum()) - 1
    print(f"{task} {level} |{col}|: succ {v[s].mean():.3f} fail {v[~s].mean():.3f} "
          f"rb={rb:+.3f} p={mw.pvalue:.4f}")

print("\n=== variant-level: gen SR vs eval SR (L3b), and debug ===")
for task in ('T1', 'T2', 'T3'):
    att = gen_attempts(task, 'L3b')
    vg = att.groupby('variant')['retained'].mean()
    for n in (400, 100):
        j = EP[(EP.task == task) & (EP.level == 'L3b') & (EP.n_demos == n)]
        ve = j.groupby('variant')['success'].mean()
        both = pd.concat([vg.rename('gen'), ve.rename('ev')], axis=1).dropna()
        print(f"{task} L3b n{n}: k={len(both)} gen_sr std={both.gen.std():.4f} "
              f"eval_sr std={both.ev.std():.4f}")
        if len(both) >= 5 and both.gen.std() > 0 and both.ev.std() > 0:
            sp = stats.spearmanr(both.gen, both.ev)
            pr = stats.pearsonr(both.gen, both.ev)
            print(f"   spearman={sp.correlation:+.3f} p={sp.pvalue:.3f}  "
                  f"pearson={pr[0]:+.3f} p={pr[1]:.3f}")
            print('   gen :', both.gen.round(3).to_dict())
            print('   eval:', both.ev.round(3).to_dict())

print("\n=== T2 L3b: is local_rej link confounded by yaw? partial check ===")
j = EP[(EP.task == 'T2') & (EP.level == 'L3b') & (EP.n_demos == 400)].copy()
s = j.success.astype(bool)
print('corr(local_rej, obj_yaw):', stats.spearmanr(j.local_rej, j.obj_yaw).correlation.round(3))
print('corr(local_rej, |obj_yaw|):', stats.spearmanr(j.local_rej, j.obj_yaw.abs()).correlation.round(3))
print('corr(local_rej, obj_y):', stats.spearmanr(j.local_rej, j.obj_y).correlation.round(3))
# within yaw terciles, does local_rej still separate succ/fail?
j['yaw_bin'] = pd.qcut(j.obj_yaw, 3, labels=False)
for b in range(3):
    sub = j[j.yaw_bin == b]
    ss = sub.success.astype(bool)
    if ss.sum() >= 2 and (~ss).sum() >= 2:
        mw = stats.mannwhitneyu(sub.local_rej[ss], sub.local_rej[~ss])
        rb = 2 * mw.statistic / (ss.sum() * (~ss).sum()) - 1
        print(f' yaw tercile {b}: n={len(sub)} SR={ss.mean():.2f} local_rej rb={rb:+.3f} p={mw.pvalue:.3f}')

print("\n=== T2 L3b variant composition of local_rej (pose-pooled kNN) ===")
print(j.groupby('variant')[['local_rej', 'success']].mean().round(3))
