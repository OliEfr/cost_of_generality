"""Figures and a statistics table for demo-generation bias, per task.

The question these answer: **is the training distribution the same as the eval distribution?**
Isaac Lab Mimic keeps only the attempts it judges successful, so the demos a policy trains on are
drawn from p(pose | generator succeeded) while the frozen eval sets sample p(pose). Where those
differ, a low success rate is partly a data-pipeline artifact rather than a policy limit -- and the
three tasks here sit in genuinely different regimes:

  * T1 cup_place  -- generation SR ~85-88% at every level, retained and rejected poses
                     indistinguishable. Unbiased thinning; not a confound.
  * T2 drawer_stow -- generation SR falls 55 -> 31% and the retained poses ARE significantly
                     skewed (yaw especially). This is the case where the concern is real, so it
                     gets the detailed panels.
  * T3 push_target -- SR 88-98%, some statistically significant but tiny skew; bounded by the
                     rejected fraction to a few points.

Every number comes from `gen_bias.level_stats`, the same function the CLI table prints, so figures
and text cannot drift apart.

Reads the rejected attempts from the parallel `<level>_failed.hdf5` files that generation writes.

usage:
  python -m cog.analysis.gen_bias_plots                      # all tasks, default level sets
  python -m cog.analysis.gen_bias_plots --outdir paper/figures
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib

import matplotlib
matplotlib.use("Agg")                      # headless: no display on this box, and none needed
import matplotlib.pyplot as plt            # noqa: E402
import numpy as np                         # noqa: E402

from .gen_bias import DIMS, level_stats    # noqa: E402

# (task label, {reported level label: hdf5 stem}). The reported "L3" is the L3b arm on disk -- the
# one regenerated with per-variant seeds and the corrected palette (D27/D28/D29). The original L3
# datasets are still on disk and still auditable, but they are deprecated as a generality level, so
# they appear only under --include-deprecated.
TASKS = {
    "T1 cup_place": {"L0": "L0", "L1": "L1", "L2": "L2", "L3": "L3b"},
    "T2 drawer_stow": {"L0": "T2_L0", "L1": "T2_L1", "L2": "T2_L2", "L3": "T2_L3b"},
    "T3 push_target": {"L0": "T3_L0", "L1": "T3_L1", "L2": "T3_L2", "L3": "T3_L3b"},
}
DEPRECATED = {
    "T1 cup_place": {"L3 (deprecated)": "L3"},
    "T2 drawer_stow": {"L3 (deprecated)": "T2_L3"},
    "T3 push_target": {"L3 (deprecated)": "T3_L3"},
}
RETAINED_C, REJECTED_C = "#2b6cb0", "#c05621"


def ecdf(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(a)
    return x, np.arange(1, len(x) + 1) / len(x)


def task_figure(task: str, stats: dict[str, dict], out: pathlib.Path) -> None:
    """Six panels: the three summary bars, then the detail that shows HOW a level is skewed."""
    labels = list(stats)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    fig.suptitle(f"Demo-generation bias -- {task}", fontsize=15, fontweight="bold")

    # (1) generation SR: the demonstrator's own success rate on each distribution. This is the
    # ACHIEVABLE CEILING for any policy imitating these demos, which is why it belongs first.
    ax = axes[0][0]
    sr = [stats[l]["gen_sr"] for l in labels]
    bars = ax.bar(labels, sr, color=[RETAINED_C if s >= 50 else REJECTED_C for s in sr])
    for b, l in zip(bars, labels):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                f"{b.get_height():.1f}%\n{stats[l]['attempts']} att.",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_ylabel("generation success rate (%)")
    ax.set_title("Generator SR per level\n(= achievable ceiling for the policy)", fontsize=10)
    ax.axhline(50, ls=":", c="grey", lw=1)

    # (2) unique initial poses -- the D27 check. A demo axis is only real if this grows with N.
    ax = axes[0][1]
    uq = [stats[l]["unique_succ"] for l in labels]
    cols = [REJECTED_C if (stats[l]["redundancy"] > 1.11 and stats[l]["unique_succ"] > 1)
            else RETAINED_C for l in labels]
    bars = ax.bar(labels, uq, color=cols)
    for b, l in zip(bars, labels):
        r = stats[l]["redundancy"]
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8,
                f"{int(b.get_height())}" + (f"\n{r:.1f}x dup" if r > 1.11 else ""),
                ha="center", va="bottom", fontsize=8)
    ax.axhline(400, ls="--", c="green", lw=1.2, label="400 demos = 400 poses")
    ax.set_ylabel("unique initial poses among retained demos")
    ax.set_title("Pose diversity of the 400-demo set\n(L0 is 1 by design)", fontsize=10)
    ax.set_ylim(0, 470)
    ax.legend(fontsize=8)

    # (3) the bound: bias can only remove what it rejects.
    ax = axes[0][2]
    att = [stats[l]["max_attributable_pts"] for l in labels]
    # At a level with a SINGLE initial pose (L0 by design) a filter cannot shift the pose
    # distribution at all: those rejections are demonstrator stochasticity, not selection. Show the
    # bar hollow so it is not read as a bias bound.
    single = [stats[l]["unique_succ"] <= 1 for l in labels]
    ax.bar(labels, att,
           color=["none" if s else REJECTED_C for s in single],
           edgecolor=REJECTED_C, hatch=["//" if s else "" for s in single])
    for i, (v, s) in enumerate(zip(att, single)):
        ax.text(i, v + 1.5, f"{v:.1f}" + ("\n(n/a: 1 pose)" if s else ""),
                ha="center", fontsize=8)
    ax.set_ylabel("percentage points")
    ax.set_ylim(0, max(max(att) * 1.25, 20))
    ax.set_title("Max eval deficit attributable to generation bias\n"
                 "(= rejected fraction; a hard upper bound)", fontsize=10)

    # (4) KS D per level x dimension, filled where significant. Shows WHICH axis is filtered.
    ax = axes[1][0]
    w = 0.26
    for j, dim in enumerate(DIMS):
        ds = [stats[l]["ks"][dim]["d"] for l in labels]
        ds = [0 if math.isnan(d) else d for d in ds]
        sig = [dim in stats[l]["skewed"] for l in labels]
        xs = np.arange(len(labels)) + (j - 1) * w
        ax.bar(xs, ds, w, label=dim, color=["#2f855a", "#b7791f", "#822727"][j],
               edgecolor="black", linewidth=0.5)
        for x, d, s in zip(xs, ds, sig):
            if s:
                ax.text(x, d + 0.008, "*", ha="center", fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("KS D (retained vs rejected)")
    ax.set_title("Distribution shift per axis\n* = p < 0.05 (over unique poses)", fontsize=10)
    ax.legend(fontsize=8)

    # (5) + (6) detail for the most-skewed level: where the filter actually bites.
    worst = max(labels, key=lambda l: stats[l]["worst_ks"] if stats[l]["n_fail"] else -1)
    st = stats[worst]
    dim = max(DIMS, key=lambda d: (0 if math.isnan(st["ks"][d]["d"]) else st["ks"][d]["d"]))
    ax = axes[1][1]
    if st["n_fail"] and st["unique_fail"] > 1:
        i = DIMS.index(dim)
        for arr, lab, c in ((st["succ_poses"], "retained (trained on)", RETAINED_C),
                            (st["fail_poses"], "rejected (never seen)", REJECTED_C)):
            x, y = ecdf(arr[:, i])
            ax.step(x, y, where="post", label=lab, color=c, lw=2)
        k = st["ks"][dim]
        ax.set_title(f"{worst}: {dim} distribution\nKS D={k['d']:.3f}, p={k['p']:.4f}", fontsize=10)
        ax.set_xlabel(f"initial {dim}")
        ax.set_ylabel("cumulative fraction")
        ax.legend(fontsize=8)
    else:
        ax.axis("off")
        ax.set_title(f"{worst}: no rejected attempts to compare", fontsize=10)

    ax = axes[1][2]
    if st["n_fail"]:
        ax.scatter(st["succ_poses"][:, 0], st["succ_poses"][:, 1], s=14, alpha=0.6,
                   c=RETAINED_C, label=f"retained ({st['unique_succ']})")
        ax.scatter(st["fail_poses"][:, 0], st["fail_poses"][:, 1], s=22, alpha=0.75,
                   c=REJECTED_C, marker="x", label=f"rejected ({st['unique_fail']})")
        ax.set_xlabel("initial object x (m)")
        ax.set_ylabel("initial object y (m)")
        ax.set_title(f"{worst}: where in the workspace\ngeneration succeeded vs failed", fontsize=10)
        ax.legend(fontsize=8)
    else:
        ax.axis("off")

    for row in axes:
        for a in row:
            a.grid(alpha=0.25, ls=":")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def summary_figure(all_stats: dict[str, dict[str, dict]], out: pathlib.Path) -> None:
    """The cross-task figure: the demonstrator's SR is itself a function of the generality level,
    so part of any measured 'cost of generality' belongs to the data collector, not the policy."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    order = ["L0", "L1", "L2", "L3"]
    for task, stats in all_stats.items():
        xs = [l for l in order if l in stats]
        axes[0].plot(xs, [stats[l]["gen_sr"] for l in xs], "o-", lw=2, label=task)
        axes[1].plot(xs, [stats[l]["max_attributable_pts"] for l in xs], "o-", lw=2, label=task)
    axes[0].set_ylabel("generation success rate (%)")
    axes[0].set_title("Demonstrator SR falls with generality\n(T2 steeply; T1/T3 nearly flat)",
                      fontsize=11)
    axes[0].set_ylim(0, 105)
    axes[1].set_ylabel("percentage points")
    axes[1].set_title("Upper bound on eval deficit explainable\nby generation bias", fontsize=11)
    for a in axes:
        a.grid(alpha=0.3, ls=":")
        a.legend(fontsize=9)
        a.set_xlabel("generality level")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="paper/figures")
    ap.add_argument("--csv", default="experiments/gen_bias.csv")
    ap.add_argument("--include-deprecated", action="store_true",
                    help="also audit the superseded pose-redundant L3 datasets (D27), labelled "
                         "'L3 (deprecated)'. Off by default: they are not a reported level.")
    args = ap.parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_stats: dict[str, dict[str, dict]] = {}
    rows = []
    for task, levels in TASKS.items():
        if args.include_deprecated:
            levels = {**levels, **DEPRECATED[task]}
        stats = {}
        for label, stem in levels.items():
            st = level_stats(stem)
            if st is None:
                continue                    # dataset absent, or being written right now (file lock)
            stats[label] = st
            rows.append({
                "task": task.split()[0], "level": label, "stem": stem,
                "retained": st["n_succ"], "rejected": st["n_fail"], "attempts": st["attempts"],
                "gen_sr_pct": round(st["gen_sr"], 2),
                "unique_poses": st["unique_succ"], "redundancy": round(st["redundancy"], 2),
                "max_attributable_pts": round(st["max_attributable_pts"], 2),
                **{f"ks_d_{d}": round(st["ks"][d]["d"], 4) for d in DIMS},
                **{f"ks_p_{d}": round(st["ks"][d]["p"], 4) for d in DIMS},
                "skewed_dims": "|".join(st["skewed"]) or "none",
            })
        if not stats:
            continue
        all_stats[task] = stats
        task_figure(task, stats, outdir / f"gen_bias_{task.split()[0]}.png")

    summary_figure(all_stats, outdir / "gen_bias_summary.png")

    csv_path = pathlib.Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {csv_path} ({len(rows)} level rows)")


if __name__ == "__main__":
    main()
