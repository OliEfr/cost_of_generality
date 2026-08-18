"""Paper figures. Each function is independent so a figure can be regenerated alone.

  python -m cog.analysis.figures --which gen_sr        # ready NOW (real data)
  python -m cog.analysis.figures --which all           # skips figures whose inputs are absent

`gen_sr` reads experiments/gen_stats.csv (complete for all three tasks). The policy figures
read experiments/curves.csv, written by cog.analysis.curves once evals exist; they are
skipped with a message rather than faked when that file is missing.
"""

from __future__ import annotations

import argparse
import csv
import collections
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = pathlib.Path(__file__).resolve().parents[3]
OUT = REPO / "paper" / "figures"
LEVELS = ["L0", "L1", "L2", "L3"]
TASK_LABEL = {
    "cup_place": "T1 cup-place (prehensile, short)",
    "drawer_stow": "T2 drawer-stow (articulated, long)",
    "push_target": "T3 push-target (non-prehensile)",
}
TASK_COLOR = {"cup_place": "#1f77b4", "drawer_stow": "#d62728", "push_target": "#2ca02c"}
# T1 and T3 converge at L3 (87.9 vs 88.5), so their value labels collide unless pushed apart.
TASK_LABEL_DY = {"cup_place": -13, "drawer_stow": 8, "push_target": 8}


def _gen_stats() -> dict[str, dict[str, tuple[int, int]]]:
    """task -> level -> (successes, attempts), pooling L3 variants."""
    path = REPO / "experiments" / "gen_stats.csv"
    agg: dict[str, dict[str, list[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0, 0]))
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if not r["attempts"]:
                continue
            cell = agg[r["task"]][r["level"]]
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


def _curves() -> list[dict] | None:
    path = REPO / "experiments" / "curves.csv"
    if not path.exists():
        return None
    with path.open() as fh:
        return list(csv.DictReader(fh))


def fig_sr_vs_n() -> pathlib.Path | None:
    rows = _curves()
    if not rows:
        print("  skip fig_sr_vs_n: experiments/curves.csv not present yet (needs evals)")
        return None
    by_level: dict[str, list[tuple[int, float, float, float]]] = collections.defaultdict(list)
    for r in rows:
        by_level[r["level"]].append((int(r["n_demos"]), float(r["sr_best"]),
                                     float(r["ci_lo"]), float(r["ci_hi"])))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for lvl in sorted(by_level):
        pts = sorted(by_level[lvl])
        ns = [p[0] for p in pts]
        srs = [100 * p[1] for p in pts]
        lo = [100 * (p[1] - p[2]) for p in pts]
        hi = [100 * (p[3] - p[1]) for p in pts]
        ax.errorbar(ns, srs, yerr=[lo, hi], marker="o", capsize=3, lw=1.8, label=lvl)
    ax.set_xscale("log")
    ax.set_xlabel("demonstrations (log scale)")
    ax.set_ylabel("rollout success rate (%)")
    ax.set_title("Success vs demonstrations, per generality level", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "fig_sr_vs_n.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


FIGS = {"gen_sr": fig_gen_sr, "sr_vs_n": fig_sr_vs_n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all", choices=["all", *FIGS])
    args = ap.parse_args()
    names = list(FIGS) if args.which == "all" else [args.which]
    for name in names:
        out = FIGS[name]()
        if out:
            print(f"  wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
