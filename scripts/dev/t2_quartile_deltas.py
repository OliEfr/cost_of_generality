import numpy as np
import pandas as pd

OUT = "/home/admin_07/cost_of_generality/.claude/worktrees/results-analysis/experiments/"
j = pd.read_csv(OUT + "t2_episode_join.csv")
if "level_x" in j.columns:
    j = j.rename(columns={"level_x": "level"}).drop(columns=["level_y"])
    j.to_csv(OUT + "t2_episode_join.csv", index=False)
j = j[j.n >= 50]
rows = []
for col, ab in (("obj_yaw", True), ("obj_x", False), ("obj_y", False)):
    v = j[col].abs() if ab else j[col]
    edges = np.quantile(v[j.level == "L1"], np.linspace(0, 1, 5))
    edges[-1] += 1e-9
    for k in range(4):
        m = (v >= edges[k]) & (v < edges[k + 1])
        s1 = j[(j.level == "L1") & m].success.mean()
        n1 = int(((j.level == "L1") & m).sum())
        s2 = j[(j.level == "L2") & m].success.mean()
        rows.append(dict(dim=("|" + col + "|") if ab else col, q=k + 1,
                         lo=round(edges[k], 3), hi=round(edges[k + 1], 3),
                         n_per_level=n1, sr_L1=round(s1, 3), sr_L2=round(s2, 3),
                         delta=round(s2 - s1, 3)))
d = pd.DataFrame(rows)
d.to_csv(OUT + "t2_pose_quartile_deltas.csv", index=False)
print(d.to_string(index=False))
print("\nL2>=L1 in", int((d.delta >= 0).sum()), "of", len(d), "quartile cells")
