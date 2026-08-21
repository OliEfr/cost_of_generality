"""Publication figures for the generation-bias -> eval-SR link analysis.

One figure per task (rows = randomized levels, cols = the randomized state planes):
generation attempts (retained gray / rejected dark-yellow) with eval episodes
overlaid (success blue circle / failure red X) -- the single picture that shows
"demos only exist here / policy only fails there", or refutes it.
Plus one cross-cell summary: pose-selective rejection mass vs the success<->gen-hardness
effect, and the SR split between gen-easy and gen-hard half-spaces.

Palette (dataviz skill, validated with scripts/validate_palette.js, light surface
#fcfcfb): blue #2a78d6 / dark-yellow #c98500 / red #e34948 -- CVD worst pair 6.2
(legal with secondary encoding: failure uses an X marker, rejected attempts are
small context dots; a legend is always present; experiments/genbias_link.csv is
the table twin). Task colors in the summary: slots 1-3 (#2a78d6/#eb6834/#1baf7a),
validated all-pairs.

Run: /home/admin_07/miniconda3/envs/cog_isaac/bin/python scripts/dev/genbias_link_figs.py
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from genbias_link_stats import TASKS, gen_attempts, OUT_EXP

WT = pathlib.Path("/home/admin_07/cost_of_generality/.claude/worktrees/results-analysis")
FIGDIR = WT / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ---- tokens (light mode; papers print on white)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
C_RET = "#c3c2b7"      # retained attempts: recessive context layer
C_REJ = "#c98500"      # rejected attempts
C_SUCC = "#2a78d6"     # eval success
C_FAIL = "#e34948"     # eval failure (X marker = secondary encoding)
TASK_C = {"T1": "#2a78d6", "T2": "#eb6834", "T3": "#1baf7a"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9,
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "grid.linestyle": "-", "axes.axisbelow": True,
    "legend.frameon": False,
})

EP = pd.read_csv(OUT_EXP / "genbias_link_episodes.csv")
CELLS = json.load(open(OUT_EXP / "genbias_link_cells.json"))
LINK = pd.read_csv(OUT_EXP / "genbias_link.csv")

TASK_NAMES = {"T1": "T1 cup_place", "T2": "T2 drawer_stow", "T3": "T3 push_target"}
DIM_LABEL = {"obj_x": "object x (m)", "obj_y": "object y (m)", "obj_yaw": "object yaw (rad)",
             "goal_x": "goal x (m)", "goal_y": "goal y (m)",
             "cab_x": "cabinet x (m)", "cab_y": "cabinet y (m)",
             "cab_dyaw": "cabinet yaw dev. (rad)", "bearing": "push bearing (rad)"}


def cell_info(task, level, n=400):
    for c in CELLS:
        if c["task"] == task and c["level"] == level and c.get("n") == n:
            return c
    return None


def despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def scatter_plane(ax, att, ev, dx, dy, xr, yr, show_attempts=True):
    """Retained/rejected attempts + eval success/failure marks in one plane."""
    if show_attempts:
        r = att[att.retained]
        f = att[~att.retained]
        ax.scatter(r[dx], r[dy], s=5, c=C_RET, alpha=0.55, lw=0, zorder=2)
        # keep the reject layer readable when rejects outnumber demos (T2: 900+)
        rej_alpha = 0.75 if len(f) < 300 else 0.45
        ax.scatter(f[dx], f[dy], s=7, c=C_REJ, alpha=rej_alpha, lw=0, zorder=3)
    s = ev[ev.success.astype(bool)]
    x = ev[~ev.success.astype(bool)]
    ax.scatter(s[dx], s[dy], s=26, facecolor=C_SUCC, edgecolor=SURFACE,
               lw=1.0, zorder=4)
    ax.scatter(x[dx], x[dy], s=64, marker="X", facecolor=C_FAIL,
               edgecolor=SURFACE, lw=1.0, zorder=5)
    pad_x = (xr[1] - xr[0]) * 0.06
    pad_y = (yr[1] - yr[0]) * 0.06
    ax.set_xlim(xr[0] - pad_x, xr[1] + pad_x)
    ax.set_ylim(yr[0] - pad_y, yr[1] + pad_y)
    despine(ax)


def hist_panel(ax, att, ev, dim, rng):
    """Retained vs rejected densities + eval success/failure rug in one dim."""
    bins = np.linspace(rng[0], rng[1], 17)
    r = att.loc[att.retained, dim]
    f = att.loc[~att.retained, dim]
    ax.hist(r, bins=bins, density=True, color=C_RET, alpha=0.75, lw=0,
            label="retained", zorder=2)
    if len(f):
        ax.hist(f, bins=bins, density=True, histtype="step", lw=1.8,
                edgecolor=C_REJ, label="rejected", zorder=3)
    ymax = ax.get_ylim()[1]
    s = ev.loc[ev.success.astype(bool), dim]
    x = ev.loc[~ev.success.astype(bool), dim]
    ax.plot(s, np.full(len(s), ymax * 1.05), "|", ms=7, mew=1.1, c=C_SUCC,
            zorder=4)
    ax.plot(x, np.full(len(x), ymax * 1.13), "|", ms=8, mew=2.0, c=C_FAIL,
            zorder=5)
    ax.set_ylim(0, ymax * 1.19)
    pad = (rng[1] - rng[0]) * 0.06
    ax.set_xlim(rng[0] - pad, rng[1] + pad)
    ax.set_yticks([])
    ax.grid(False, axis="y")
    despine(ax)
    ax.spines["left"].set_visible(False)


def task_figure(task: str):
    cfg = TASKS[task]
    levels = ["L1", "L2", "L3b"]
    # column plan per level row: (kind, args)
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 10.2), dpi=200,
                             gridspec_kw=dict(hspace=0.42, wspace=0.26))
    for i, lv in enumerate(levels):
        att = gen_attempts(task, lv)
        dd = [d for d in cfg["demo_dims"] if d in cfg["dims"][lv]]
        att = att.drop_duplicates(subset=dd + ["retained"])
        ev = EP[(EP.task == task) & (EP.level == lv) & (EP.n_demos == 400)]
        ci = cell_info(task, lv)
        rej = ci["rejection_rate"]
        n_f = int((~ev.success.astype(bool)).sum())

        # -- col 1: object x-y plane
        ax = axes[i, 0]
        scatter_plane(ax, att, ev, "obj_x", "obj_y",
                      cfg["ranges"]["obj_x"], cfg["ranges"]["obj_y"])
        ax.set_xlabel(DIM_LABEL["obj_x"])
        ax.set_ylabel(DIM_LABEL["obj_y"])
        ax.set_title(f"{lv} -- object plane\ngen rej {rej:.0%} · "
                     f"eval SR {ev.success.mean():.2f} ({n_f} fail)",
                     loc="left", fontsize=9, color=INK, pad=6)

        # -- col 2: the 1-D dim with the strongest generator story
        dim1 = ("obj_yaw" if task in ("T1", "T2")
                else ("obj_y" if lv == "L1" else "bearing"))
        ax = axes[i, 1]
        hist_panel(ax, att, ev, dim1, cfg["ranges"][dim1])
        ax.set_xlabel(DIM_LABEL[dim1])
        gk = LINK[(LINK.task == task) & (LINK.level == lv) & (LINK.dim == dim1)]
        if len(gk) and np.isfinite(gk.ks_retained_rejected_D.iloc[0]):
            d0, p0 = gk.ks_retained_rejected_D.iloc[0], gk.ks_retained_rejected_p.iloc[0]
            note = f"gen filter: KS D={d0:.2f}, p={p0:.1g}"
        else:
            note = "gen filter: n/a"
        rb, pp = gk.succ_vs_fail_effect.iloc[0], gk.succ_vs_fail_p.iloc[0]
        note2 = (f"success vs state: r={rb:+.2f}, p={pp:.2f}"
                 if len(gk) and np.isfinite(rb) else "success vs state: saturated")
        ax.set_title(f"{note}\n{note2}", loc="left", fontsize=8.2, color=INK2, pad=5)

        # -- col 3: second randomized plane (eval-side only for T1/T2)
        ax = axes[i, 2]
        if task == "T1" and lv in ("L2", "L3b"):
            scatter_plane(ax, att, ev, "goal_x", "goal_y",
                          cfg["ranges"]["goal_x"], cfg["ranges"]["goal_y"],
                          show_attempts=False)
            ax.set_xlabel(DIM_LABEL["goal_x"])
            ax.set_ylabel(DIM_LABEL["goal_y"])
            ax.set_title("goal plane -- eval only\n(demos record no goal pose)",
                         loc="left", fontsize=8.2, color=INK2, pad=5)
        elif task == "T2" and lv in ("L2", "L3b"):
            scatter_plane(ax, att, ev, "cab_x", "cab_dyaw",
                          cfg["ranges"]["cab_x"], cfg["ranges"]["cab_dyaw"],
                          show_attempts=False)
            ax.set_xlabel(DIM_LABEL["cab_x"])
            ax.set_ylabel(DIM_LABEL["cab_dyaw"])
            ax.set_title("cabinet pose -- eval only\n(demos record no cabinet pose)",
                         loc="left", fontsize=8.2, color=INK2, pad=5)
        elif task == "T3" and lv in ("L2", "L3b"):
            scatter_plane(ax, att, ev, "obj_x", "bearing",
                          cfg["ranges"]["obj_x"], cfg["ranges"]["bearing"])
            ax.set_xlabel(DIM_LABEL["obj_x"])
            ax.set_ylabel(DIM_LABEL["bearing"])
            ax.set_title("object x vs bearing\n(bearing recorded in demos)",
                         loc="left", fontsize=8.2, color=INK2, pad=5)
        else:
            ax.axis("off")
            ax.text(0.02, 0.92, "no second randomized\naxis at this level",
                    transform=ax.transAxes, fontsize=8.5, color=MUTED, va="top")

    fig.subplots_adjust(top=0.865, bottom=0.055, left=0.07, right=0.985)
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=5, c=C_RET, label="gen attempt: retained (demo)"),
        plt.Line2D([], [], marker="o", ls="", ms=5, c=C_REJ, label="gen attempt: rejected"),
        plt.Line2D([], [], marker="o", ls="", ms=7, mfc=C_SUCC, mec=SURFACE, label="eval episode: success"),
        plt.Line2D([], [], marker="X", ls="", ms=9, mfc=C_FAIL, mec=SURFACE, label="eval episode: failure"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.995),
               ncol=2, fontsize=8.8, handletextpad=0.4, columnspacing=1.2)
    fig.text(0.01, 0.985, f"{TASK_NAMES[task]} -- does the generator's pose filter "
             "shape eval success?", ha="left", va="top", fontsize=12.5,
             fontweight="bold", color=INK)
    fig.text(0.01, 0.955,
             "generation attempts (initial states, retained vs rejected) with the frozen\n"
             "eval set's episodes (N=400 policy, 80k steps) overlaid.  L0 omitted:\n"
             "single fixed pose, nothing to filter spatially.",
             fontsize=8.8, color=INK2, va="top")
    out = FIGDIR / f"genbias_link_{task}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def summary_figure():
    rows = []
    base = {"T1": 0.1361, "T2": 0.4505, "T3": 0.0148}
    for c in CELLS:
        if c["level"] == "L0" or c.get("n") != 400:
            continue
        rows.append({
            "task": c["task"], "level": c["level"],
            "cell": f"{c['task']} {c['level']}",
            "rej": c["rejection_rate"],
            "rej_sel": max(c["rejection_rate"] - base[c["task"]], 0.0),
            "rb": c.get("local_rej_rb"), "p": c.get("local_rej_p"),
            "sr_easy": c.get("sr_gen_easy"), "sr_hard": c.get("sr_gen_hard"),
            "sr": c.get("sr"), "power": c.get("power_limited"),
        })
    df = pd.DataFrame(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 5.0), dpi=200,
                                   gridspec_kw=dict(wspace=0.32))

    # ---- panel 1: pose-selective rejection mass vs success<->gen-hardness effect
    ax1.axhspan(-0.68, 0, color="#e34948", alpha=0.05, zorder=0, lw=0)
    ax1.axhline(0, color=BASE, lw=0.8, zorder=1)
    sat = []
    for _, r in df.iterrows():
        if r.rb is None or not np.isfinite(r.rb):
            sat.append(f"{r.cell} (SR {r.sr:.2f})")
            continue
        sig = r.p is not None and np.isfinite(r.p) and r.p < 0.05
        ax1.plot(r.rej_sel * 100, r.rb, marker="o", ms=9 if sig else 8,
                 mfc=TASK_C[r.task] if sig else SURFACE, mec=TASK_C[r.task],
                 mew=1.6, zorder=4)
        dx, dy, ha = 0, 8, "center"
        if r.cell in ("T3 L1", "T3 L3b"):
            dy = -15
        elif r.cell == "T2 L2":
            dx, dy, ha = -8, 0, "right"
        ax1.annotate(r.cell + (f"  p={r.p:.2f}" if sig or (r.p or 1) < 0.1 else ""),
                     (r.rej_sel * 100, r.rb), xytext=(dx, dy),
                     textcoords="offset points", ha=ha, va="center",
                     fontsize=7.8, color=INK2)
    ax1.set_xlabel("pose-selective rejection mass (level rejection $-$ L0 baseline, % of attempts)")
    ax1.set_ylabel("success vs local gen-rejection  (rank-biserial $r$)")
    ax1.set_ylim(-0.68, 0.45)
    ax1.set_xlim(-1.5, 28)
    ax1.text(27.3, -0.47, "below 0 = failures sit where generation was rejected\n"
             "(the contamination signature)", ha="right", va="bottom",
             fontsize=8.0, color=INK2)
    if sat:
        ax1.text(-0.5, -0.665, "no estimate (saturated): " + ", ".join(sat),
                 ha="left", va="bottom", fontsize=7.6, color=MUTED)
    ax1.set_title("Is eval success lower where the generator rejected?",
                  loc="left", fontsize=10.5, color=INK, pad=8)
    despine(ax1)
    hs = [plt.Line2D([], [], marker="o", ls="", ms=7, mfc=TASK_C[t], mec=TASK_C[t],
                     label=TASK_NAMES[t]) for t in ("T1", "T2", "T3")]
    hs += [plt.Line2D([], [], marker="o", ls="", ms=7, mfc=SURFACE, mec=INK2,
                      label="p >= 0.05"),
           plt.Line2D([], [], marker="o", ls="", ms=7, mfc=INK2, mec=INK2,
                      label="p < 0.05")]
    ax1.legend(handles=hs, loc="upper right", fontsize=7.6, ncol=2,
               handletextpad=0.3, columnspacing=0.9)

    # ---- panel 2: SR in gen-easy vs gen-hard half-spaces (median local_rej split)
    d2 = df[df.sr_easy.notna()].reset_index(drop=True)
    order = [f"{t} {l}" for t in ("T1", "T2", "T3") for l in ("L1", "L2", "L3b")]
    d2["ord"] = d2.cell.map({c: i for i, c in enumerate(order)})
    d2 = d2.sort_values("ord", ascending=False).reset_index(drop=True)
    y = np.arange(len(d2))
    for i, r in d2.iterrows():
        ax2.plot([r.sr_easy, r.sr_hard], [i, i], "-", c=BASE, lw=1.6, zorder=2)
    ax2.plot(d2.sr_easy, y, "o", ms=8, mfc=C_SUCC, mec=SURFACE, mew=1.0,
             ls="", zorder=3, label="gen-easy half (low local rejection)")
    ax2.plot(d2.sr_hard, y, "o", ms=8, mfc=C_REJ, mec=SURFACE, mew=1.0,
             ls="", zorder=4, label="gen-hard half (high local rejection)")
    ax2.set_yticks(y, d2.cell)
    ax2.tick_params(axis="y", labelsize=8.4, labelcolor=INK)
    ax2.set_xlim(0, 1.03)
    ax2.set_xlabel("eval success rate (N=400 policy, 80k steps)")
    ax2.set_title("SR in gen-easy vs gen-hard halves of the eval set",
                  loc="left", fontsize=10.5, color=INK, pad=8)
    for i, r in d2.iterrows():
        if r.p is not None and np.isfinite(r.p) and r.p < 0.1:
            ax2.annotate(f"p={r.p:.2f}", (max(r.sr_easy, r.sr_hard) + 0.03, i),
                         fontsize=7.6, color=INK2, va="center")
    ax2.grid(False, axis="y")
    despine(ax2)
    ax2.legend(loc="lower left", fontsize=8.0, handletextpad=0.3,
               bbox_to_anchor=(0.02, 0.02))

    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.075, right=0.985)
    fig.text(0.01, 0.975, "Generator filtering vs downstream eval success -- "
             "all randomized cells (L0 cells exempt: single fixed pose)",
             ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.915,
             "local gen-rejection = fraction rejected among an eval state's 25 nearest "
             "generation attempts (z-scored demo-observable dims).\n"
             "T1 L1/L2 have <=1 eval failure at N=400, so no effect is estimable there "
             "(shown as saturated).",
             fontsize=8.4, color=INK2, va="top")
    out = FIGDIR / "genbias_link_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    for t in ("T1", "T2", "T3"):
        task_figure(t)
    summary_figure()
