"""The study's cross-task summary table: one row per (task, level), N* and cost ratios.

Every number here is derived, never typed in: the eval JSONs are the only input, and they go
through the same load -> canonical -> best_of_last3 path as the per-task tables and figures, so
the three cannot drift apart. `canonical` is what turns the L3b arm into the reported "L3" and
drops the deprecated pose-redundant L3 (D27/D29).

Tasks are never pooled -- each has its own L0 baseline, so a cost ratio only means something
within a task.

  python -m cog.analysis.summary                       # writes experiments/cost_of_generality_summary.csv
  python -m cog.analysis.summary --include-deprecated   # ablation view: L3 and L3b as separate rows
"""

from __future__ import annotations

import argparse
import csv
import pathlib

from .curves import (
    best_of_last3, canonical, cost_ratio, load, logistic_fit, n_star, TARGETS,
)

REPO = pathlib.Path(__file__).resolve().parents[3]
TASKS = {"T1": "cup_place", "T2": "drawer_stow", "T3": "push_target"}
LEVELS = ["L0", "L1", "L2", "L3"]


def build(results: pathlib.Path, keep_deprecated: bool = False) -> list[dict]:
    rows = []
    all_records = load(results)
    for task, name in TASKS.items():
        recs = canonical([r for r in all_records if r["task_id"] == task],
                         keep_deprecated=keep_deprecated)
        cells = best_of_last3(recs)
        by_level: dict[str, list[tuple[int, float]]] = {}
        for (lvl, n), c in cells.items():
            by_level.setdefault(lvl, []).append((n, c["sr_best"]))
        stars = {lvl: {t: n_star(sorted(pts), t, logistic_fit(sorted(pts))) for t in TARGETS}
                 for lvl, pts in by_level.items()}
        levels = LEVELS + [l for l in sorted(by_level) if l not in LEVELS]
        for lvl in levels:
            if lvl not in stars:
                continue
            pts = dict(sorted(by_level[lvl]))
            rows.append({
                "task": task, "task_name": name, "level": lvl,
                "sr_n10": pts.get(10), "sr_n100": pts.get(100), "sr_n400": pts.get(400),
                **{f"nstar_{int(t * 100)}": stars[lvl][t] for t in TARGETS},
                # Ratios are against this task's own L0. A deprecated level keeps its ratio for the
                # ablation view; it just never appears in the default table.
                **{f"ratio_{int(t * 100)}": cost_ratio(stars[lvl][t], stars["L0"][t])
                   for t in TARGETS},
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--out", default=str(REPO / "experiments" / "cost_of_generality_summary.csv"))
    ap.add_argument("--include-deprecated", action="store_true")
    args = ap.parse_args()

    rows = build(pathlib.Path(args.results), keep_deprecated=args.include_deprecated)
    if not rows:
        print("no cells found; refusing to write an empty summary")
        return
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)\n")

    hdr = (f"{'task':14} {'lvl':4} {'SR@10':>6} {'SR@100':>7} {'SR@400':>7} "
           f"{'N*(50)':>8} {'N*(80)':>8} {'N*(90)':>8} {'ratio@90':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def f(v):
            return f"{v:.2f}" if isinstance(v, float) else "-"
        print(f"{r['task_name']:14} {r['level']:4} {f(r['sr_n10']):>6} {f(r['sr_n100']):>7} "
              f"{f(r['sr_n400']):>7} {r['nstar_50']:>8} {r['nstar_80']:>8} {r['nstar_90']:>8} "
              f"{r['ratio_90']:>10}")


if __name__ == "__main__":
    main()
