"""Figures for the additive-ladder report. Writes PNGs to paper/ladder_report/figures/."""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cog.analysis.loo import NDEMOS, TASK_LABEL, TASKS
from cog.analysis.loo_figures import STOP_COLOR, STOP_LABEL, STOP_ORDER
from cog.analysis.ladder import ADD, AXIS_PLAIN, RUNGS, SETUP_NAME, SETUPS, build

OUT = pathlib.Path(__file__).resolve().parents[3] / "paper" / "ladder_report" / "figures"

# A sequential ramp, because the setups are a ladder rather than four unrelated conditions.
SETUP_COLOR = {"L0": "#a9c0d6", "L1": "#6d9cc4", "L2": "#3f6f9e", "L3b": "#1d3f5e"}
AXIS_COLOR = {"A": "#c2453f", "B": "#c96a11", "C": "#3f7f39"}

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


def fig_curves(d):
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
    fig.suptitle("Success rate at each rung of the ladder", y=1.03, fontsize=11.5,
                 fontweight="bold")
    return _save(fig, "ladder01_success.png")


def fig_delta(d):
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    for ax, task in zip(axes, TASKS):
        for axis in RUNGS:
            rows = sorted(_idx(d["ladder"], task=task, axis=axis), key=lambda r: r["n"])
            x = [r["n"] for r in rows]
            ax.plot(x, [r["delta"] for r in rows], marker="o", ms=4.5, lw=1.9,
                    color=AXIS_COLOR[axis], label=f"switch on {AXIS_PLAIN[axis]}")
            ax.fill_between(x, [r["ci_lo"] for r in rows], [r["ci_hi"] for r in rows],
                            color=AXIS_COLOR[axis], alpha=0.14, lw=0)
        ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.7)
        ax.set_xscale("log")
        ax.set_xticks(NDEMOS)
        ax.set_xticklabels([str(n) for n in NDEMOS])
        ax.set_xlabel("demonstrations")
        ax.set_title(TASK_LABEL[task])
    axes[0].set_ylabel("change in success rate")
    axes[0].legend(loc="lower right", fontsize=8.4)
    fig.suptitle("What switching each disturbance on costs, given the rungs below it",
                 y=1.03, fontsize=11.5, fontweight="bold")
    return _save(fig, "ladder02_delta.png")


def fig_funnel(d):
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.9), sharey=True)
    for ax, arm in zip(axes, SETUPS):
        rows = sorted([r for r in d["furthest"] if r["arm"] == arm], key=lambda r: r["n"])
        y = np.arange(len(rows))
        left = np.zeros(len(rows))
        for key in STOP_ORDER:
            vals = np.array([r[key] / r["episodes"] for r in rows])
            ax.barh(y, vals, left=left, height=0.72, color=STOP_COLOR[key],
                    label=STOP_LABEL[key], edgecolor="white", linewidth=0.7)
            for yi, (v, l) in enumerate(zip(vals, left)):
                if v > 0.085:
                    ax.text(l + v / 2, yi, f"{v*100:.0f}", ha="center", va="center",
                            fontsize=7.4, color="white" if key != "got nowhere" else "#444")
            left += vals
        ax.set_yticks(y, [str(r["n"]) for r in rows])
        ax.set_xlim(0, 1)
        ax.set_xticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100 %"])
        ax.invert_yaxis()
        ax.set_title(SETUP_NAME[arm])
        ax.grid(False)
        ax.set_xlabel("share of the 200 episodes")
    axes[0].set_ylabel("demonstrations")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8.2, bbox_to_anchor=(0.5, -0.09))
    fig.suptitle("T2 drawer_stow: how far each episode got before it stopped", y=1.03,
                 fontsize=11.5, fontweight="bold")
    return _save(fig, "ladder03_funnel.png")


def build_all():
    d = build()
    return d, [fig_curves(d), fig_delta(d), fig_funnel(d)]


if __name__ == "__main__":
    build_all()
