"""Are L2's retained demos 'cleaner' than L1's? Length conditional on geometry,
eef path efficiency, and action jerk, from the hdf5 sources (read-only)."""

import pathlib

import h5py
import numpy as np
import pandas as pd
from scipy import stats

MAIN = pathlib.Path("/home/admin_07/cost_of_generality")
OUT = pathlib.Path(
    "/home/admin_07/cost_of_generality/.claude/worktrees/results-analysis/experiments")


def yaw_of(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


rows = []
for lvl in ("L1", "L2"):
    with h5py.File(MAIN / f"data/hdf5/T2_{lvl}.hdf5", "r") as f:
        for demo in f["data"]:
            g = f[f"data/{demo}"]
            n = int(g.attrs["num_samples"])
            eef = np.asarray(g["obs/eef_pos"])
            path = np.linalg.norm(np.diff(eef, axis=0), axis=1).sum()
            act = np.asarray(g["actions"])
            jerk = np.abs(np.diff(act[:, :6], axis=0)).mean()
            cab = np.asarray(g["initial_state/articulation/cabinet/root_pose"])[0]
            obj = np.asarray(g["initial_state/rigid_object/object/root_pose"])[0]
            rows.append(dict(level=lvl, demo=demo, length=n, path=path, jerk=jerk,
                             cab_x=cab[0], cab_y=cab[1], obj_x=obj[0], obj_y=obj[1],
                             obj_yaw=yaw_of(obj[3:7])))
df = pd.DataFrame(rows)
df.to_csv(OUT / "t2_demo_quality.csv", index=False)

for lvl in ("L1", "L2"):
    d = df[df.level == lvl]
    print(f"T2_{lvl}: n={len(d)} length mean {d.length.mean():.1f} sd {d.length.std():.1f} "
          f"| eef path mean {d.path.mean():.3f} m sd {d.path.std():.3f} "
          f"| mean |d action| {d.jerk.mean():.5f}")

# geometry-matched: L2 demos whose cabinet is within 2 cm of L1's fixed (0.9, 0.0)
d2 = df[df.level == "L2"]
near = d2[(np.abs(d2.cab_x - 0.9) < 0.02) & (np.abs(d2.cab_y) < 0.02)]
d1 = df[df.level == "L1"]
print(f"\ngeometry-matched subset (|cab_x-0.9|<2cm & |cab_y|<2cm): n={len(near)}")
print(f"  L2 matched length mean {near.length.mean():.1f} vs L1 all {d1.length.mean():.1f}")
t, p = stats.ttest_ind(near.length, d1.length, equal_var=False)
print(f"  Welch t={t:.2f} p={p:.5f}")
print(f"  L2 matched path {near.path.mean():.3f} vs L1 {d1.path.mean():.3f} "
      f"(t={stats.ttest_ind(near.path, d1.path, equal_var=False).statistic:.2f}, "
      f"p={stats.ttest_ind(near.path, d1.path, equal_var=False).pvalue:.4f})")

# does L2 length track cabinet distance (sanity: geometry explains some of it)?
r = stats.pearsonr(d2.cab_x, d2.length)
print(f"\nL2: corr(cab_x, length) r={r.statistic:.3f} p={r.pvalue:.2e}")
r2 = stats.pearsonr(np.hypot(d2.cab_x - d2.obj_x, d2.cab_y - d2.obj_y), d2.length)
print(f"L2: corr(carry_dist, length) r={r2.statistic:.3f} p={r2.pvalue:.2e}")
