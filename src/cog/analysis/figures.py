"""Paper figures. Each function is independent so a figure can be regenerated alone.

  python -m cog.analysis.figures --which gen_sr       # generation SR vs level, one line per task
  python -m cog.analysis.figures --which sr_vs_n      # SR vs demos, one FIGURE per task, 4 lines
  python -m cog.analysis.figures --which all

`gen_sr` reads experiments/gen_stats.csv. `sr_vs_n` reads results/eval_*.json directly through
cog.analysis.curves -- not a curves CSV -- so a figure can never disagree with a table that was
regenerated at a different time; both apply cog.analysis.curves.canonical, which reports the
seed-corrected L3b arm as "L3" and drops the deprecated pose-redundant one (D27/D29). A figure
whose inputs are missing is skipped with a message rather than faked.
"""

from __future__ import annotations

import argparse
import csv
import collections
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .curves import DEPRECATED_LEVELS, REPORT_AS

REPO = pathlib.Path(__file__).resolve().parents[3]
OUT = REPO / "paper" / "figures"
LEVELS = ["L0", "L1", "L2", "L3"]
TASK_LABEL = {
    "cup_place": "T1 cup-place (prehensile, short)",
    "drawer_stow": "T2 drawer-stow (articulated, long)",
    "push_target": "T3 push-target (non-prehensile)",
}
TASK_COLOR = {"cup_place": "#1f77b4", "drawer_stow": "#d62728", "push_target": "#2ca02c"}
N_GRID = [10, 25, 50, 100, 200, 400]
# Sequential, dark -> light-warm: the levels are ordered (each contains the previous), so a
# categorical palette would hide that. Markers differ too, so the figure survives greyscale print.
LEVEL_COLOR = {"L0": "#08306b", "L1": "#2171b5", "L2": "#d94801", "L3": "#7f2704"}
LEVEL_MARKER = {"L0": "o", "L1": "s", "L2": "^", "L3": "D"}
# What each level ADDS to the one before it (the ladder is cumulative). Per task, because L2 means
# "goal pose" on T1, "cabinet pose" on T2 and "target bearing" on T3.
LEVEL_ADDS = {
    "cup_place": {"L0": "fixed", "L1": "+cup pose", "L2": "+goal pose", "L3": "+object variation"},
    "drawer_stow": {"L0": "fixed", "L1": "+object pose", "L2": "+cabinet pose",
                    "L3": "+object variation"},
    "push_target": {"L0": "fixed", "L1": "+puck pose", "L2": "+target bearing",
                    "L3": "+object geometry"},
}
# T1's labels go BELOW its line: T1 and T3 run within ~6 points of each other at L1/L2 (85.8 vs
# 94.8, 85.1 vs 95.0) and both sets of labels above would sit in the same strip.
TASK_LABEL_DY = {"cup_place": -13, "drawer_stow": 8, "push_target": 8}


def _canon_gen_level(level: str) -> str | None:
    """Reporting name for a gen_stats.csv level, or None if the level is not reported.

    gen_stats.csv keys the two L3 arms differently: the original arm has level="L3" with the variant
    in its own column, while the regenerated arm encodes the variant in the level ("L3bv07"). So the
    variant suffix is stripped first, and only then are the study's reporting names applied. Without
    the None case, the ten deprecated "L3" rows would pool into the reported L3 point and the ten
    L3b rows would be silently dropped for not matching any level name -- which is exactly what this
    figure did until 2026-08-21.
    """
    base = re.sub(r"v\d\d$", "", level)
    if base in DEPRECATED_LEVELS:
        return None
    return REPORT_AS.get(base, base)


def _gen_stats() -> dict[str, dict[str, tuple[int, int]]]:
    """task -> level -> (successes, attempts), pooling L3 variants."""
    path = REPO / "experiments" / "gen_stats.csv"
    agg: dict[str, dict[str, list[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0, 0]))
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if not r["attempts"]:
                continue
            lvl = _canon_gen_level(r["level"])
            if lvl is None:
                continue
            cell = agg[r["task"]][lvl]
            cell[0] += int(r["successes"])
            cell[1] += int(r["attempts"])
    return {t: {l: tuple(v) for l, v in d.items()} for t, d in agg.items()}


def fig_gen_sr() -> pathlib.Path:
    """Generation SR vs generality level, one line per task.

    The study's first completed result: the three tasks have three distinct signatures, and
    the ordering is NOT by task difficulty -- T3 is hardest to control and generates best.
    """
    data = _gen_stats()
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for task in ("cup_place", "drawer_stow", "push_target"):
        if task not in data:
            continue
        xs, ys = [], []
        for i, lvl in enumerate(LEVELS):
            if lvl in data[task]:
                k, n = data[task][lvl]
                xs.append(i)
                ys.append(100 * k / n)
        ax.plot(xs, ys, "o-", color=TASK_COLOR[task], label=TASK_LABEL[task], lw=2, ms=6)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                        xytext=(0, TASK_LABEL_DY[task]), ha="center", fontsize=8,
                        color=TASK_COLOR[task])
    ax.axhline(30, ls="--", c="grey", lw=1)
    ax.text(0.02, 31, "G3 floor (30%)", fontsize=8, color="grey")
    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels(["L0\nfixed", "L1\n+object pose", "L2\n+fixture pose",
                        "L3\n+object variation"], fontsize=8)
    ax.set_ylabel("Mimic generation success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Generation cost of generality differs by task structure,\nnot by task difficulty",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "fig_gen_sr_vs_level.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def _star_label(star: str) -> str:
    """Render an N* string for a legend. n_star() returns "185", ">400" or "<=10"; splicing those
    straight after an "=" gives "N*=>400", which reads as a typo."""
    if star.startswith(">"):
        return f"$N^*_{{90}}$ > {star[1:]}"
    if star.startswith("<="):
        return f"$N^*_{{90}}$ $\\leq$ {star[2:]}"
    return f"$N^*_{{90}}$ = {star}"


def fig_sr_vs_n() -> list[pathlib.Path]:
    """The study's headline figure, one per task: success rate vs demo count, one line per level.

    Read straight from results/eval_*.json through cog.analysis.curves rather than from a curves
    CSV, so the figure cannot disagree with the tables -- both go through the same load ->
    canonical -> best_of_last3 path, including the L3b -> L3 rename and the deprecation of the
    pose-redundant L3 arm.

    Deliberate choices:
      * log x -- the demo grid doubles, so linear x would crush everything below N=100;
      * Wilson intervals, asymmetric near the ceiling, drawn as error bars. L3 cells have 200
        episodes (variant diagonal, D18) and the others 100, so L3's bars are visibly tighter;
        that is real and is stated in the caption rather than hidden by hiding the bars;
      * N*(90 %) printed in the legend, since the crossing is the quantity the paper reports and
        reading it off the curve by eye invites error;
      * a fixed y range 0-100 on every task, so the three figures can be compared side by side.
    """
    from .curves import best_of_last3, canonical, load, logistic_fit, n_star

    records = canonical(load(REPO / "results"))
    if not records:
        print("  skip fig_sr_vs_n: no eval_*.json under results/")
        return []
    out_paths = []
    for task_id, task in (("T1", "cup_place"), ("T2", "drawer_stow"), ("T3", "push_target")):
        cells = best_of_last3([r for r in records if r["task_id"] == task_id])
        if not cells:
            print(f"  skip fig_sr_vs_n {task_id}: no cells")
            continue
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for lvl in LEVELS:
            pts = sorted((c["n_demos"], c) for k, c in cells.items() if k[0] == lvl)
            if not pts:
                continue
            ns = [n for n, _ in pts]
            srs = [100 * c["sr_best"] for _, c in pts]
            # The Wilson interval is centred on a shrunk estimate, not on p, so at p=1.0 its upper
            # limit sits a hair BELOW the point (0.99987 at n=100) and the raw offset goes negative.
            # Clamp at zero: the bar then ends at the marker, which is what a 100 % cell should look
            # like anyway.
            lo = [max(0.0, 100 * (c["sr_best"] - c["ci_lo"])) for _, c in pts]
            hi = [max(0.0, 100 * (c["ci_hi"] - c["sr_best"])) for _, c in pts]
            star = n_star([(n, c["sr_best"]) for n, c in pts], 0.90,
                          logistic_fit([(n, c["sr_best"]) for n, c in pts]))
            ax.errorbar(ns, srs, yerr=[lo, hi], marker=LEVEL_MARKER[lvl], capsize=3, lw=1.9,
                        ms=5.5, color=LEVEL_COLOR[lvl],
                        label=f"{lvl} {LEVEL_ADDS[task][lvl]}   {_star_label(star)}")
        ax.axhline(90, ls=":", c="grey", lw=1)
        ax.text(9.6, 91, "90 % target", fontsize=7.5, color="grey")
        ax.set_xscale("log")
        ax.set_xticks(N_GRID)
        ax.set_xticklabels([str(n) for n in N_GRID])
        ax.minorticks_off()
        ax.set_xlim(9, 460)
        ax.set_ylim(0, 100)
        ax.set_xlabel("demonstrations per policy (log scale)")
        ax.set_ylabel("rollout success rate (%)")
        ax.set_title(f"{TASK_LABEL[task]}\nsuccess vs demonstrations, per generality level",
                     fontsize=10)
        # Legend BELOW the axes, two columns. Inside the axes there is no corner that is free on all
        # three tasks -- T1 empties its lower right as the curves rise, T2 keeps three curves under
        # 45 % and "best" then covers its L0 point at N=10 -- and a legend that hides a data point on
        # one task only is worse than one that costs 15 % of the figure height on every task.
        ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
                  frameon=False, handletextpad=0.5, columnspacing=1.6)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        OUT.mkdir(parents=True, exist_ok=True)
        p = OUT / f"fig_sr_vs_n_{task_id}.png"
        fig.savefig(p, dpi=180, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(p)
    return out_paths


FIGS = {"gen_sr": fig_gen_sr, "sr_vs_n": fig_sr_vs_n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all", choices=["all", *FIGS])
    args = ap.parse_args()
    names = list(FIGS) if args.which == "all" else [args.which]
    for name in names:
        out = FIGS[name]()
        for p in ([out] if isinstance(out, pathlib.Path) else out or []):
            print(f"  wrote {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
