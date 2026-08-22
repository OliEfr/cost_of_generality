"""T2 stage-funnel figure: per level, the four stages across N (clean sweep, own-set
evals only — matched policy/eval-set throughout). Data: experiments/t2_stage_funnel_full.csv.
Palette: dataviz categorical slots 1-4 (validated); 'stowed' is the outcome and gets the
heaviest line; direct labels relieve the contrast WARN."""
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

STAGES = [
    ("opened", "drawer opened", "#2a78d6", 1.6),
    ("lifted", "object lifted", "#eb6834", 1.6),
    ("over", "over drawer", "#eda100", 1.6),
    ("sr", "stowed (SR)", "#1baf7a", 2.6),
]
NS = [10, 25, 50, 100, 200, 400]

data = defaultdict(dict)
for r in csv.DictReader(open("experiments/t2_stage_funnel_full.csv")):
    data[r["level"]][int(r["n_demos"])] = r

fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
fig.patch.set_facecolor("white")
titles = {"L0": "L0 — fixed scene", "L1": "L1 — + object pose", "L2": "L2 — + cabinet pose",
          "L3": "L3 — + object variation"}
for ax, level in zip(axes, ("L0", "L1", "L2", "L3")):
    ax.set_facecolor("white")
    ns = [n for n in NS if n in data[level]]
    ends = []
    for key, label, col, lw in STAGES:
        ys = [float(data[level][n][key]) for n in ns]
        ax.plot(ns, ys, "-", color=col, lw=lw, marker="o",
                ms=5 if key == "sr" else 4, zorder=4 if key == "sr" else 3)
        ends.append((label, col, ys[-1]))
    if level == "L3":  # direct labels on the last panel, spread to avoid collisions
        ends.sort(key=lambda e: e[2])
        ys_lab = [y for *_, y in ends]
        for i in range(1, len(ys_lab)):
            ys_lab[i] = max(ys_lab[i], ys_lab[i - 1] + 0.07)
        for (label, col, _), y in zip(ends, ys_lab):
            ax.annotate(label, (ns[-1], y), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=9, color=col, fontweight="bold")
    # emphasize the open->lift collapse: shade between the two lines
    op = [float(data[level][n]["opened"]) for n in ns]
    li = [float(data[level][n]["lifted"]) for n in ns]
    ax.fill_between(ns, li, op, color="#eb6834", alpha=0.08, lw=0, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.minorticks_off()
    ax.set_title(titles[level], fontsize=11, loc="left", fontweight="bold", color="#1F2523")
    ax.set_xlabel("demonstrations N", fontsize=10, color="#55605C")
    ax.grid(axis="y", color="#E3E7E4", lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#C9CFCC")
    ax.tick_params(colors="#55605C", labelsize=9)
    ax.set_ylim(0, 1.04)
    ax.set_xlim(8.5, 750 if level == "L3" else 520)
axes[0].set_ylabel("fraction of episodes reaching stage", fontsize=10, color="#55605C")
handles = [Line2D([], [], color=c, lw=w, marker="o", ms=5, label=l) for _, l, c, w in STAGES]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 1.03), ncol=4,
           frameon=False, fontsize=9)
fig.suptitle("T2 drawer_stow — stage funnel per level (clean sweep, 100–200 episodes/cell)",
             x=0.005, ha="left", fontsize=13, fontweight="bold", color="#1F2523")
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig("paper/figures/t2_stage_funnel.png", dpi=170, bbox_inches="tight", facecolor="white")
print("wrote paper/figures/t2_stage_funnel.png")
