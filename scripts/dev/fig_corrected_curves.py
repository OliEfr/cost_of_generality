"""Figure: published (bug-inflated) vs corrected SR curves for all three tasks.

Corrected = constraint-propagation point estimate from stale_corrected_sr.csv, with
[lower, upper] hard bounds as a band. L3 cells are clean (published == corrected).
Palette: dataviz reference categorical slots 1-4, validated (validate_palette.js PASS;
contrast WARN relieved by direct labels).
"""
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"L0": "#2a78d6", "L1": "#eb6834", "L2": "#1baf7a", "L3": "#eda100"}
TASKS = {"T1": "T1 cup_place", "T2": "T2 drawer_stow", "T3": "T3 push_target"}
NS = [10, 25, 50, 100, 200, 400]

rows = list(csv.DictReader(open("experiments/stale_corrected_sr.csv")))
data = defaultdict(dict)
for r in rows:
    lvl = "L3" if r["level"] in ("L3b",) else r["level"]
    if r["task"] == "T1" and r["level"] == "L3":
        continue  # deprecated pose-redundant arm; L3b is reported as L3
    data[(r["task"], lvl)][int(r["n_demos"])] = (
        float(r["sr_published"]), float(r["sr_corrected_point"]),
        float(r["sr_lower"]), float(r["sr_upper"]), r["note"],
        int(r["n_identified"]))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
fig.patch.set_facecolor("white")
for ax, task in zip(axes, TASKS):
    ax.set_facecolor("white")
    label_pts = []
    for lvl in ["L0", "L1", "L2", "L3"]:
        cells = data.get((task, lvl))
        if not cells:
            continue
        ns = [n for n in NS if n in cells]
        pub = [cells[n][0] for n in ns]
        pt = [cells[n][1] for n in ns]
        lo = [cells[n][2] for n in ns]
        hi = [cells[n][3] for n in ns]
        clean = cells[ns[0]][4] == "clean"
        if clean:
            ax.plot(ns, pub, "-", color=C[lvl], lw=2, marker="o", ms=5, zorder=3)
        else:
            ax.plot(ns, pub, "--", color=C[lvl], lw=1.2, alpha=0.45, zorder=2)
            # the hard bounds are vacuous for near-saturated cells (few identifiable
            # episodes); draw the band only where it is informative, and mark
            # low-information points with an open face
            band_lo = [l if h - l <= 0.35 else p for l, h, p in zip(lo, hi, pt)]
            band_hi = [h if h - l <= 0.35 else p for l, h, p in zip(lo, hi, pt)]
            ax.fill_between(ns, band_lo, band_hi, color=C[lvl], alpha=0.15, lw=0, zorder=1)
            ax.plot(ns, pt, "-", color=C[lvl], lw=2, zorder=3)
            for n, p in zip(ns, pt):
                solid = cells[n][5] >= 40
                ax.plot([n], [p], "o", ms=5.5, zorder=4, color=C[lvl],
                        mfc=C[lvl] if solid else "white", mew=1.6)
        label = f"{lvl} (clean)" if clean else lvl
        label_pts.append((label, C[lvl], ns[-1], (pt if not clean else pub)[-1]))
    # spread end labels so they never collide
    label_pts.sort(key=lambda x: x[3])
    ys = [y for *_, y in label_pts]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 0.055)
    over = max(0, ys[-1] - 1.01) if ys else 0
    for (label, col, x, _), y in zip(label_pts, ys):
        ax.annotate(label, (x, y - over), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=9, color=col, fontweight="bold")
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
axes[0].set_ylabel("success rate (100-200 episodes)", fontsize=10, color="#55605C")

from matplotlib.lines import Line2D

handles = [
    Line2D([], [], color="#55605C", lw=2, marker="o", ms=5, label="corrected (point estimate)"),
    plt.Rectangle((0, 0), 1, 1, fc="#55605C", alpha=0.13, label="corrected hard bounds"),
    Line2D([], [], color="#55605C", lw=1.2, ls="--", alpha=0.5, label="published (batch-carryover bug)"),
]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 1.02),
           ncol=3, frameon=False, fontsize=9)
fig.suptitle("Published vs corrected success rates -- batch-boundary carryover removed",
             x=0.005, ha="left", fontsize=13, fontweight="bold", color="#1F2523")
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig("paper/figures/corrected_vs_published_sr.png", dpi=170,
            bbox_inches="tight", facecolor="white")
print("wrote paper/figures/corrected_vs_published_sr.png")
