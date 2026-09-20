"""Figures for the leave-one-out report. Writes PNGs to paper/loo_report/figures/.

Every number plotted here comes from cog.analysis.loo; nothing is recomputed locally.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cog.analysis.loo import (ARMS, AXIS_LABEL, NDEMOS, REMOVE, TASK_LABEL, TASKS,
                              build, wilson)

OUT = pathlib.Path(__file__).resolve().parents[3] / "paper" / "loo_report" / "figures"

ARM_COLOR = {"L0": "#4c78a8", "L1": "#72b7b2", "L2": "#54a24b",
             "L3b": "#333333", "AC": "#e45756", "BC": "#f58518"}
ARM_LABEL = {"L0": "L0  (none)", "L1": "L1  {A}", "L2": "L2  {A,B}  = L3b\\C",
             "L3b": "L3b  {A,B,C}  (full)", "AC": "AC  {A,C}  = L3b\\B",
             "BC": "BC  {B,C}  = L3b\\A"}
AXIS_COLOR = {"A": "#e45756", "B": "#f58518", "C": "#54a24b"}
STAGE_COLOR = {"p_opened": "#4c78a8", "p_lifted": "#72b7b2",
               "p_over": "#f58518", "p_success": "#333333"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 170, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p)
    return p


def _idx(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


# ------------------------------------------------------------------ figures


def fig_surface_curves(d):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), sharey=True)
    for ax, task in zip(axes, TASKS):
        for arm in ARMS:
            rows = sorted(_idx(d["surface"], task=task, arm=arm), key=lambda r: r["n"])
            x = [r["n"] for r in rows]
            y = [r["sr"] for r in rows]
            lo = [r["sr"] - r["wilson_lo"] for r in rows]
            hi = [r["wilson_hi"] - r["sr"] for r in rows]
            ax.errorbar(x, y, yerr=[lo, hi], marker="o", ms=4, lw=1.6, capsize=2.5,
                        color=ARM_COLOR[arm], label=ARM_LABEL[arm],
                        elinewidth=0.9)
        ax.set_xscale("log")
        ax.set_xticks(NDEMOS)
        ax.set_xticklabels([str(n) for n in NDEMOS])
        ax.set_xlabel("demonstrations")
        ax.set_title(TASK_LABEL[task])
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("success rate  (200 episodes, Wilson 95 %)")
    axes[2].legend(loc="lower right", fontsize=7.6)
    fig.suptitle("Measured surface: success rate vs demonstration budget, six arms per task",
                 y=1.03, fontsize=11, fontweight="bold")
    return _save(fig, "fig01_surface_curves.png")


def fig_surface_heatmap(d):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4))
    for ax, task in zip(axes, TASKS):
        M = np.array([[_idx(d["surface"], task=task, arm=a, n=n)[0]["sr"] for n in NDEMOS]
                      for a in ARMS])
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(NDEMOS)), [str(n) for n in NDEMOS])
        ax.set_yticks(range(len(ARMS)), ARMS)
        ax.set_xlabel("demonstrations")
        ax.set_title(TASK_LABEL[task])
        ax.grid(False)
        for i in range(len(ARMS)):
            for j in range(len(NDEMOS)):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=7.4,
                        color="white" if M[i, j] < 0.62 else "black")
    fig.colorbar(im, ax=axes, shrink=0.82, label="success rate", pad=0.015)
    fig.suptitle("Measured surface: 108 cells, 200 scored episodes each",
                 y=1.04, fontsize=11, fontweight="bold")
    return _save(fig, "fig02_surface_heatmap.png")


def fig_loo_delta_vs_n(d):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), sharey=True)
    for ax, task in zip(axes, TASKS):
        for axis in REMOVE:
            rows = sorted(_idx(d["loo"], task=task, axis=axis), key=lambda r: r["n"])
            x = [r["n"] for r in rows]
            y = [r["delta"] for r in rows]
            lo = [r["ci_lo"] for r in rows]
            hi = [r["ci_hi"] for r in rows]
            ax.plot(x, y, marker="o", ms=4.5, lw=1.7, color=AXIS_COLOR[axis],
                    label=f"remove {AXIS_LABEL[axis]}  (= {REMOVE[axis]})")
            ax.fill_between(x, lo, hi, color=AXIS_COLOR[axis], alpha=0.14, lw=0)
        ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.7)
        ax.set_xscale("log")
        ax.set_xticks(NDEMOS)
        ax.set_xticklabels([str(n) for n in NDEMOS])
        ax.set_xlabel("demonstrations")
        ax.set_title(TASK_LABEL[task])
    axes[0].set_ylabel("Δ success rate  (arm − L3b)")
    axes[0].legend(loc="upper right", fontsize=7.6)
    fig.suptitle("Leave-one-out: change in success rate when one axis is removed from the full set",
                 y=1.03, fontsize=11, fontweight="bold")
    return _save(fig, "fig03_loo_delta_vs_n.png")


def fig_loo_forest(d):
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5.6), sharex=True)
    for ax, task in zip(axes, TASKS):
        ys, labels, colors = [], [], []
        y = 0
        for axis in ("A", "B", "C"):
            for n in reversed(NDEMOS):
                r = _idx(d["loo"], task=task, axis=axis, n=n)[0]
                ax.plot([r["ci_lo"], r["ci_hi"]], [y, y], lw=2.0,
                        color=AXIS_COLOR[axis], alpha=0.85 if r["significant"] else 0.35,
                        solid_capstyle="butt")
                ax.plot([r["delta"]], [y], marker="D", ms=4.6, color=AXIS_COLOR[axis],
                        mec="white", mew=0.6)
                labels.append(f"{axis}  N={n}")
                ys.append(y)
                y += 1
            y += 0.8
        ax.axvline(0, color="black", lw=1.0, ls="--", alpha=0.7)
        ax.set_yticks(ys, labels, fontsize=7.4)
        ax.set_xlabel("Δ success rate  (arm − L3b),  Newcombe 95 %")
        ax.set_title(TASK_LABEL[task])
        ax.set_ylim(-1, y)
        ax.grid(axis="y", alpha=0.12)
    fig.suptitle("Leave-one-out deltas, every task × axis × demonstration budget "
                 "(faded = interval covers zero)", y=0.97, fontsize=11, fontweight="bold")
    return _save(fig, "fig04_loo_forest.png")


def fig_slice_dispersion(d):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), sharey=True)
    raw = d["_raw"]
    for ax, task in zip(axes, TASKS):
        pos = 0
        ticks, labels = [], []
        for arm in ARMS:
            for n in NDEMOS:
                c = raw[(task, arm, n)]
                vals = [k / 20 for k in c["slice_k"]]
                jitter = (np.random.default_rng(int(pos * 10)).random(len(vals)) - 0.5) * 0.45
                ax.scatter(np.full(len(vals), pos) + jitter, vals, s=7,
                           color=ARM_COLOR[arm], alpha=0.6, lw=0)
                ax.plot([pos - 0.42, pos + 0.42], [c["sr"], c["sr"]], color="black", lw=1.3)
                pos += 1
            ticks.append(pos - 3.5)
            labels.append(arm)
            pos += 1.6
        ax.set_xticks(ticks, labels)
        ax.set_title(TASK_LABEL[task])
        ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel("arm  (six budgets left→right: 10, 25, 50, 100, 200, 400)")
    axes[0].set_ylabel("per-slice success rate  (20 episodes each)")
    fig.suptitle("Within-cell dispersion: the ten independent slices behind every pooled cell",
                 y=1.02, fontsize=11, fontweight="bold")
    return _save(fig, "fig05_slice_dispersion.png")


def fig_t2_funnel(d):
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.4), sharey=True, sharex=True)
    for ax, arm in zip(axes.ravel(), ARMS):
        rows = sorted(_idx(d["stages"], arm=arm), key=lambda r: r["n"])
        x = [r["n"] for r in rows]
        for key, lab in (("p_opened", "drawer opened"), ("p_lifted", "object lifted"),
                         ("p_over", "object over drawer"), ("p_success", "success")):
            ax.plot(x, [r[key] for r in rows], marker="o", ms=4, lw=1.7,
                    color=STAGE_COLOR[key], label=lab)
        ax.set_xscale("log")
        ax.set_xticks(NDEMOS)
        ax.set_xticklabels([str(n) for n in NDEMOS])
        ax.set_title(f"T2  {ARM_LABEL[arm]}")
        ax.set_ylim(-0.03, 1.03)
    for ax in axes[1]:
        ax.set_xlabel("demonstrations")
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of 200 episodes")
    axes[0, 0].legend(loc="center left", fontsize=7.4, bbox_to_anchor=(0.02, 0.36))
    fig.suptitle("T2 drawer_stow stage funnel: absolute rate of each milestone, per arm",
                 y=0.99, fontsize=11, fontweight="bold")
    return _save(fig, "fig06_t2_stage_funnel.png")


def fig_t2_conditional(d):
    keys = [("lift_given_open", "lift | opened"), ("over_given_lift", "over | lifted"),
            ("success_given_over", "success | over")]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6), sharey=True)
    for ax, (key, lab) in zip(axes, keys):
        for arm in ARMS:
            rows = sorted(_idx(d["stages"], arm=arm), key=lambda r: r["n"])
            ax.plot([r["n"] for r in rows], [r[key] for r in rows], marker="o", ms=4,
                    lw=1.6, color=ARM_COLOR[arm], label=arm)
        ax.set_xscale("log")
        ax.set_xticks(NDEMOS)
        ax.set_xticklabels([str(n) for n in NDEMOS])
        ax.set_xlabel("demonstrations")
        ax.set_title(lab)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("conditional rate")
    axes[2].legend(loc="lower right", ncol=2, fontsize=7.6)
    fig.suptitle("T2 drawer_stow: conditional transition rates between consecutive milestones",
                 y=1.03, fontsize=11, fontweight="bold")
    return _save(fig, "fig07_t2_conditional.png")


def fig_time_to_success(d):
    raw = d["_raw"]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    for ax, task in zip(axes, TASKS):
        data, labels, colors = [], [], []
        for arm in ARMS:
            ts = []
            for n in NDEMOS:
                ts += [o["t_first_success"] for o in raw[(task, arm, n)]["outcomes"]
                       if o["success"] and o["t_first_success"] >= 0]
            data.append(ts if ts else [0])
            labels.append(f"{arm}\nn={len(ts)}")
            colors.append(ARM_COLOR[arm])
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                        medianprops=dict(color="black", lw=1.4))
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.55)
            patch.set_linewidth(0.8)
        ax.set_xticks(range(1, len(ARMS) + 1), labels, fontsize=7.2)
        ax.set_title(TASK_LABEL[task])
        ax.set_xlabel("arm")
    axes[0].set_ylabel("step of first success")
    fig.suptitle("Time to success: step index at which a successful episode first succeeded "
                 "(all budgets pooled)", y=1.03, fontsize=11, fontweight="bold")
    return _save(fig, "fig08_time_to_success.png")


def fig_t2_progress_magnitudes(d):
    raw = d["_raw"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.8))
    for ax, key, lab, thr in ((axes[0], "max_drawer_open", "max drawer opening (m)", 0.15),
                              (axes[1], "max_object_lift", "max object lift (m)", 0.05)):
        data, labels, colors = [], [], []
        for arm in ARMS:
            vals = []
            for n in NDEMOS:
                vals += [o[key] for o in raw[("T2", arm, n)]["outcomes"]]
            data.append(vals)
            labels.append(arm)
            colors.append(ARM_COLOR[arm])
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                        medianprops=dict(color="black", lw=1.4))
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.55)
            patch.set_linewidth(0.8)
        ax.axhline(thr, color="#e45756", lw=1.1, ls="--")
        ax.text(0.99, thr, f"  milestone threshold {thr}", color="#e45756", fontsize=7.4,
                va="bottom", ha="right", transform=ax.get_yaxis_transform())
        ax.set_xticks(range(1, len(ARMS) + 1), labels)
        ax.set_ylabel(lab)
        ax.set_xlabel("arm")
    fig.suptitle("T2 drawer_stow: continuous progress measures over all 1,200 episodes per arm",
                 y=1.02, fontsize=11, fontweight="bold")
    return _save(fig, "fig09_t2_progress_magnitudes.png")


def fig_generation(d):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.6))
    w = 0.13
    for i, arm in enumerate(ARMS):
        rows = [r for r in d["gen"] if r["arm"] == arm]
        by = {r["task"]: r for r in rows}
        xs = np.arange(len(TASKS))
        axes[0].bar(xs + (i - 2.5) * w, [by[t]["gen_sr"] for t in TASKS], width=w,
                    color=ARM_COLOR[arm], label=arm)
        axes[1].bar(xs + (i - 2.5) * w, [by[t]["attempts"] for t in TASKS], width=w,
                    color=ARM_COLOR[arm], label=arm)
    for ax, lab in ((axes[0], "generation success rate"), (axes[1], "generation attempts")):
        ax.set_xticks(range(len(TASKS)), [TASK_LABEL[t] for t in TASKS])
        ax.set_ylabel(lab)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(ncol=6, fontsize=7.6, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Demonstration generation: yield per arm (400 demonstrations produced in every cell)",
                 y=1.03, fontsize=11, fontweight="bold")
    return _save(fig, "fig10_generation.png")


def build_all():
    d = build()
    paths = [fig_surface_curves(d), fig_surface_heatmap(d), fig_loo_delta_vs_n(d),
             fig_loo_forest(d), fig_slice_dispersion(d), fig_t2_funnel(d),
             fig_t2_conditional(d), fig_time_to_success(d), fig_t2_progress_magnitudes(d),
             fig_generation(d)]
    return d, paths


if __name__ == "__main__":
    build_all()


# ------------------------------------------------- plain-language edition (4 setups only)

SETUPS = ["L3b", "BC", "AC", "L2"]
SETUP_NAME = {"L3b": "all on", "BC": "minus object start", "AC": "minus goal",
              "L2": "minus object type"}
SETUP_COLOR = {"L3b": "#333333", "BC": "#e45756", "AC": "#f58518", "L2": "#54a24b"}
AXIS_PLAIN = {"A": "object start", "B": "goal", "C": "object type"}


def fig_simple_curves(d):
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    for ax, task in zip(axes, TASKS):
        for arm in SETUPS:
            rows = sorted(_idx(d["surface"], task=task, arm=arm), key=lambda r: r["n"])
            x = [r["n"] for r in rows]
            y = [r["sr"] for r in rows]
            lo = [r["sr"] - r["wilson_lo"] for r in rows]
            hi = [r["wilson_hi"] - r["sr"] for r in rows]
            ax.errorbar(x, y, yerr=[lo, hi], marker="o", ms=4.5, lw=1.9, capsize=2.5,
                        color=SETUP_COLOR[arm], label=SETUP_NAME[arm], elinewidth=0.9)
        ax.set_xscale("log")
        ax.set_xticks(NDEMOS)
        ax.set_xticklabels([str(n) for n in NDEMOS])
        ax.set_xlabel("demonstrations")
        ax.set_title(TASK_LABEL[task])
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("success rate")
    axes[0].legend(loc="lower right", fontsize=8.4)
    fig.suptitle("Success rate of each setup", y=1.03, fontsize=11.5, fontweight="bold")
    return _save(fig, "simple01_success.png")


def fig_simple_delta(d):
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    for ax, task in zip(axes, TASKS):
        for axis in ("A", "B", "C"):
            rows = sorted(_idx(d["loo"], task=task, axis=axis), key=lambda r: r["n"])
            x = [r["n"] for r in rows]
            ax.plot(x, [r["delta"] for r in rows], marker="o", ms=4.5, lw=1.9,
                    color=SETUP_COLOR[REMOVE[axis]], label=f"minus {AXIS_PLAIN[axis]}")
            ax.fill_between(x, [r["ci_lo"] for r in rows], [r["ci_hi"] for r in rows],
                            color=SETUP_COLOR[REMOVE[axis]], alpha=0.14, lw=0)
        ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.7)
        ax.set_xscale("log")
        ax.set_xticks(NDEMOS)
        ax.set_xticklabels([str(n) for n in NDEMOS])
        ax.set_xlabel("demonstrations")
        ax.set_title(TASK_LABEL[task])
    axes[0].set_ylabel("change in success rate")
    axes[0].legend(loc="upper right", fontsize=8.4)
    fig.suptitle("What dropping one disturbance does, against keeping all three",
                 y=1.03, fontsize=11.5, fontweight="bold")
    return _save(fig, "simple02_delta.png")


def fig_simple_stages(d):
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.4), sharey=True)
    for ax, arm in zip(axes, SETUPS):
        rows = sorted(_idx(d["stages"], arm=arm), key=lambda r: r["n"])
        x = [r["n"] for r in rows]
        for key, lab in (("p_opened", "opened drawer"), ("p_lifted", "lifted object"),
                         ("p_over", "held it over drawer"), ("p_success", "succeeded")):
            ax.plot(x, [r[key] for r in rows], marker="o", ms=4, lw=1.8,
                    color=STAGE_COLOR[key], label=lab)
        ax.set_xscale("log")
        ax.set_xticks(NDEMOS)
        ax.set_xticklabels([str(n) for n in NDEMOS])
        ax.set_xlabel("demonstrations")
        ax.set_title(SETUP_NAME[arm])
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("fraction of episodes")
    axes[0].legend(loc="center left", fontsize=7.8, bbox_to_anchor=(0.02, 0.42))
    fig.suptitle("T2 drawer_stow: how far each setup got", y=1.03, fontsize=11.5,
                 fontweight="bold")
    return _save(fig, "simple03_stages.png")


def build_simple():
    d = build()
    return d, [fig_simple_curves(d), fig_simple_delta(d), fig_simple_stages(d)]
