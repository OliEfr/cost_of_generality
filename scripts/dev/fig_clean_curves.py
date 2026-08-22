"""Final figure: clean re-eval surface (fixed harness) vs originally published curves.

Clean = experiments/clean_surface.csv (54 flat re-runs + 6 T2 L3 re-runs + T1/T3 L3 published,
which were single-batch and therefore already clean). Published = the original eval JSONs.
Palette: dataviz categorical slots 1-4 (validated); direct labels relieve the contrast WARN.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

C = {"L0": "#2a78d6", "L1": "#eb6834", "L2": "#1baf7a", "L3": "#eda100"}
TASKS = {"T1": "T1 cup_place", "T2": "T2 drawer_stow", "T3": "T3 push_target"}
NS = [10, 25, 50, 100, 200, 400]
R = Path("results")

clean = defaultdict(dict)
for r in csv.DictReader(open("experiments/clean_surface.csv")):
    clean[(r["task"], r["level"])][int(r["n_demos"])] = (
        float(r["sr"]), float(r["ci_lo"]), float(r["ci_hi"]))

pub = defaultdict(dict)
for task in TASKS:
    for level, fname_level in [("L0", "L0"), ("L1", "L1"), ("L2", "L2"), ("L3", "L3b")]:
        for n in NS:
            fp = R / f"eval_{task}_{fname_level}_n{n}_080000.json"
            if fp.exists():
                pub[(task, level)][n] = json.load(open(fp))["success_rate"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
fig.patch.set_facecolor("white")
for ax, task in zip(axes, TASKS):
    ax.set_facecolor("white")
    label_pts = []
    for lvl in ("L0", "L1", "L2", "L3"):
        cc = clean[(task, lvl)]
        ns = [n for n in NS if n in cc]
        sr = [cc[n][0] for n in ns]
        lo = [cc[n][1] for n in ns]
        hi = [cc[n][2] for n in ns]
        pp = pub[(task, lvl)]
        same = lvl == "L3" and task in ("T1", "T3")  # published IS the clean run there
        if not same and pp:
            ax.plot(ns, [pp[n] for n in ns if n in pp], "--", color=C[lvl], lw=1.2,
                    alpha=0.4, zorder=2)
        ax.fill_between(ns, lo, hi, color=C[lvl], alpha=0.12, lw=0, zorder=1)
        ax.plot(ns, sr, "-", color=C[lvl], lw=2, marker="o", ms=5, zorder=3)
        label_pts.append((lvl, C[lvl], ns[-1], sr[-1]))
    label_pts.sort(key=lambda x: x[3])
    ys = [y for *_, y in label_pts]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 0.055)
    over = max(0, ys[-1] - 1.01) if ys else 0
    for (label, col, x, _), y in zip(label_pts, ys):
        ax.annotate(label, (x, y - over), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=col, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xticks(NS)
    ax.set_xticklabels(NS)
    ax.minorticks_off()
    ax.set_title(TASKS[task], fontsize=11, loc="left", fontweight="bold", color="#1F2523")
    ax.set_xlabel("demonstrations N", fontsize=10, color="#55605C")
    ax.grid(axis="y", color="#E3E7E4", lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#C9CFCC")
    ax.tick_params(colors="#55605C", labelsize=9)
    ax.set_xlim(8.5, 700)
    ax.set_ylim(0, 1.02)
axes[0].set_ylabel("success rate (100–200 episodes)", fontsize=10, color="#55605C")
handles = [
    Line2D([], [], color="#55605C", lw=2, marker="o", ms=5, label="clean re-eval (fixed harness)"),
    plt.Rectangle((0, 0), 1, 1, fc="#55605C", alpha=0.12, label="Wilson 95% CI"),
    Line2D([], [], color="#55605C", lw=1.2, ls="--", alpha=0.5, label="originally published"),
]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 1.02), ncol=3,
           frameon=False, fontsize=9)
fig.suptitle("The clean surface — full re-eval with the fixed harness",
             x=0.005, ha="left", fontsize=13, fontweight="bold", color="#1F2523")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig("paper/figures/corrected_vs_published_sr.png", dpi=170, bbox_inches="tight",
            facecolor="white")
print("wrote paper/figures/corrected_vs_published_sr.png")
