"""Turn eval JSONs into the study's headline numbers.

Reads the `results/eval_<level>_n<N>_<step>.json` files written by
`cog.eval.rollout_eval` (schema: task, checkpoint, episodes, successes, success_rate,
outcomes) and produces, per (task, level):

  * SR(N) with Wilson 95 % intervals -- Wilson, not normal-approximation, because cells
    at N=10 can sit near 0 or 1 where the normal interval runs outside [0,1];
  * a logistic fit in log N;
  * N*(s) = demonstrations needed to reach success rate s, reported as "> N_max" when the
    curve never crosses s inside the measured range rather than extrapolated;
  * cost ratios N*(level) / N*(L0).

The primary metric per cell is best-of-last-three checkpoints (40k/60k/80k), with the
last-checkpoint SR also carried through, per the frozen protocol.

  python -m cog.analysis.curves --results results/ --out experiments/curves.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re

# Result files are task-tagged: eval_T2_L1_n50_080000.json. The task group is optional so the
# pre-migration T1 names still parse, defaulting to T1. Without the tag the greedy level group
# would swallow "T1_L0" as a single level name and quietly mix tasks into one table.
# The level may carry a single-letter suffix: "L3b" is the L3 arm regenerated with per-variant seeds
# and the corrected object palette (D27/D28). Parsing keeps it a SEPARATE key from the original "L3"
# -- the two differ in pose diversity, which is the whole point -- and the reporting rename happens
# later, in canonical(), so the file layer never loses which dataset a number came from.
FNAME = re.compile(
    r"eval_(?:(?P<task>T\d)_)?(?P<level>L\d[a-z]?)_n(?P<n>\d+)_(?P<step>\d+)\.json$")
TARGETS = (0.50, 0.80, 0.90)

# Reporting names (D29, 2026-08-21). "L3b" is the L3 arm regenerated with per-variant seeds and the
# corrected object palette (D27/D28). It is THE L3 of the study, so it is REPORTED as "L3".
# The original "L3" files keep their own name on disk -- they are the pose-redundancy ablation and
# their provenance has to stay traceable -- but as a generality LEVEL they are deprecated: those
# datasets hold only 43-48 of a nominal 400 unique initial poses, so their curve measures a data bug,
# not a breadth of distribution. No reported table, CSV or figure may contain both.
REPORT_AS = {"L3b": "L3"}
DEPRECATED_LEVELS = {"L3"}


def canonical(records: list[dict], keep_deprecated: bool = False) -> list[dict]:
    """Apply the study's reporting names.

    Default: drop the deprecated levels, then rename what survives (L3b -> L3). The drop is decided
    on the RAW name and happens FIRST, so the rename cannot collide with the name it takes over.

    keep_deprecated=True returns every record under its raw name instead -- the ablation view, where
    "L3" and "L3b" must stay distinguishable. It deliberately does NOT rename, because renaming
    while keeping both would merge two different datasets into one cell.
    """
    if keep_deprecated:
        return [dict(r, level_raw=r["level"]) for r in records]
    out = []
    for r in records:
        raw = r["level"]
        if raw in DEPRECATED_LEVELS:
            continue
        out.append(dict(r, level=REPORT_AS.get(raw, raw), level_raw=raw))
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Degenerate cells (n=0) return (0,1)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def load(results_dir: pathlib.Path) -> list[dict]:
    """One record per eval file; level/N/step come from the FILENAME, and the episode
    counts from the file contents."""
    out = []
    for path in sorted(results_dir.glob("eval_*.json")):
        m = FNAME.search(path.name)
        if not m:
            print(f"  skip unparseable name: {path.name}")
            continue
        d = json.loads(path.read_text())
        out.append({
            "task_id": m.group("task") or "T1",
            "level": m.group("level"),
            "n_demos": int(m.group("n")),
            "step": int(m.group("step")),
            "episodes": int(d["episodes"]),
            "successes": int(d["successes"]),
            "success_rate": float(d["success_rate"]),
            "task": d.get("task", ""),
            "file": path.name,
        })
    return out


def best_of_last3(records: list[dict]) -> dict[tuple[str, int], dict]:
    """Collapse checkpoints to one row per (level, N): best-of-last-three by SR, with the
    last checkpoint's SR retained alongside."""
    cells: dict[tuple[str, int], list[dict]] = {}
    for r in records:
        cells.setdefault((r["level"], r["n_demos"]), []).append(r)
    collapsed = {}
    for key, rs in cells.items():
        rs = sorted(rs, key=lambda r: r["step"])
        last3 = rs[-3:]
        best = max(last3, key=lambda r: r["success_rate"])
        lo, hi = wilson(best["successes"], best["episodes"])
        collapsed[key] = {
            "level": key[0],
            "n_demos": key[1],
            "sr_best": best["success_rate"],
            "sr_best_step": best["step"],
            "sr_last": rs[-1]["success_rate"],
            "episodes": best["episodes"],
            "successes": best["successes"],
            "ci_lo": lo,
            "ci_hi": hi,
            "n_checkpoints": len(rs),
        }
    return collapsed


def logistic_fit(points: list[tuple[int, float]]) -> tuple[float, float] | None:
    """Least-squares logistic in log10 N: logit(SR) = a + b*log10(N).

    Cells at exactly 0 or 1 have no finite logit, so they are clamped inward by half a
    success -- the standard continuity correction. Needs >= 3 usable points.
    """
    xs, ys = [], []
    for n, sr in points:
        p = min(max(sr, 1e-3), 1 - 1e-3)
        xs.append(math.log10(n))
        ys.append(math.log(p / (1 - p)))
    if len(xs) < 3:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    var = sum((x - mx) ** 2 for x in xs)
    if var == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
    return my - b * mx, b


def n_star_fit(fit: tuple[float, float] | None, target: float, n_max: int) -> str:
    """N* read off the logistic fit. Pools all cells, so one noisy cell moves it far less
    than it moves an interpolated crossing -- which matters here because a cell measured at
    n=100 carries +-5 points of binomial noise while the demo grid doubles between cells.
    Validated on synthetic data with known truth: see main()."""
    if not fit:
        return "n/a"
    a, b = fit
    if b <= 0:
        return "n/a"
    n = 10 ** ((math.log(target / (1 - target)) - a) / b)
    return f"{round(n)}" if n <= n_max else f">{n_max}"


def n_star(points: list[tuple[int, float]], target: float,
           fit: tuple[float, float] | None) -> str:
    """Demos needed to reach `target`. Prefers the measured crossing (linear
    interpolation between the bracketing cells); falls back to the logistic fit; reports
    '>N_max' when neither the data nor the fit crosses inside the measured range.
    """
    pts = sorted(points)
    for (n0, s0), (n1, s1) in zip(pts, pts[1:]):
        if s0 < target <= s1:
            frac = (target - s0) / (s1 - s0)
            return f"{round(n0 + frac * (n1 - n0))}"
    if pts and pts[0][1] >= target:
        return f"<={pts[0][0]}"
    if fit:
        a, b = fit
        if b > 0:
            logit = math.log(target / (1 - target))
            n = 10 ** ((logit - a) / b)
            if n <= pts[-1][0]:
                return f"{round(n)}"
    return f">{pts[-1][0]}" if pts else "n/a"


def cost_ratio(a: str, b: str) -> str:
    """Ratio of two N* strings, propagating their bounds instead of giving up.

    N* comes back as "57" (measured crossing), "<=10" (already above target at the smallest N on the
    grid) or ">400" (never reached inside it). Requiring both to be plain integers threw away real
    information: T3's L0 saturates below N=10, so every T3 ratio printed n/a even though the
    numerators are measured. The bounds compose in one direction only --
      * numerator ">A" or denominator "<=B" both make the ratio a LOWER bound,
      * numerator "<=A" or denominator ">B" would make it an upper bound, which is not informative
        here (it would read "the cost is at most X" for a level whose cost we could not measure),
    so those are reported as n/a rather than dressed up.
    """
    def parse(s: str) -> tuple[str, float] | None:
        s = s.strip()
        if s.isdigit():
            return ("=", float(s))
        if s.startswith("<=") and s[2:].isdigit():
            return ("<=", float(s[2:]))
        if s.startswith(">") and s[1:].isdigit():
            return (">", float(s[1:]))
        return None

    pa, pb = parse(a), parse(b)
    if pa is None or pb is None or pb[1] <= 0:
        return "n/a"
    ka, va = pa
    kb, vb = pb
    if ka == "<=" or kb == ">":
        return "n/a"                     # would be an upper bound; not the quantity of interest
    lower = ka == ">" or kb == "<="
    return f"{'>=' if lower else ''}{va / vb:.2f}x"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    # Default derived from --task below: a fixed "curves.csv" default meant three tasks overwrote
    # one file, and the file that survived carried no clue which task it held.
    ap.add_argument("--out", default=None)
    ap.add_argument("--task", default="T1",
                    help="which task's cells to analyse (T1/T2/T3). Tasks are NEVER pooled: each "
                         "has its own levels and its own L0 baseline, so a shared table would "
                         "compute cost ratios across unrelated tasks.")
    ap.add_argument("--include-deprecated", action="store_true",
                    help="ablation view: keep the deprecated pose-redundant L3 as its own row under "
                         "its raw name, instead of reporting L3b as L3 (see DEPRECATED_LEVELS).")
    args = ap.parse_args()
    out_path = args.out or f"experiments/curves_{args.task}.csv"

    records = canonical([r for r in load(pathlib.Path(args.results))
                         if r["task_id"] == args.task],
                        keep_deprecated=args.include_deprecated)
    if not records:
        print(f"no eval_*.json for task {args.task} under {args.results}")
        return
    print(f"task {args.task}")
    cells = best_of_last3(records)
    print(f"{len(records)} eval files -> {len(cells)} cells")

    rows = sorted(cells.values(), key=lambda r: (r["level"], r["n_demos"]))
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}")

    by_level: dict[str, list[tuple[int, float]]] = {}
    by_level_last: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        by_level.setdefault(r["level"], []).append((r["n_demos"], r["sr_best"]))
        by_level_last.setdefault(r["level"], []).append((r["n_demos"], r["sr_last"]))

    print(f"\n{'level':6} {'fit b':>7}  " + "  ".join(f"N*({int(t*100)}%)" for t in TARGETS)
          + "   [interpolated crossing]")
    stars: dict[str, dict[float, str]] = {}
    fits: dict[str, tuple[float, float] | None] = {}
    for level in sorted(by_level):
        pts = sorted(by_level[level])
        fit = logistic_fit(pts)
        fits[level] = fit
        stars[level] = {t: n_star(pts, t, fit) for t in TARGETS}
        bs = f"{fit[1]:+.2f}" if fit else "  n/a"
        print(f"{level:6} {bs:>7}  " + "  ".join(f"{stars[level][t]:>9}" for t in TARGETS))

    print(f"\n{'level':6} " + "  ".join(f"N*({int(t*100)}%)" for t in TARGETS)
          + "   [logistic fit -- preferred, pools all cells]")
    for level in sorted(by_level):
        n_max = max(n for n, _ in by_level[level])
        print(f"{level:6} " + "  ".join(
            f"{n_star_fit(fits[level], t, n_max):>9}" for t in TARGETS))

    # Best-of-last-three takes the max of three noisy draws, so it is an OPTIMISTIC
    # estimator of a cell's true SR, which pushes N* LEFT. Validated on synthetic data with
    # a known logistic (truth N*(50%) = 30/60/120/240): best-of-3 recovered 22/62/94/236
    # while the last checkpoint recovered values closer to truth. The protocol fixes
    # best-of-3 as primary, so the honest move is to report BOTH and let the gap show.
    print(f"\n{'level':6} N*(50%) best-of-3 vs last-checkpoint")
    for level in sorted(by_level):
        b = n_star(sorted(by_level[level]), 0.50, logistic_fit(sorted(by_level[level])))
        l = n_star(sorted(by_level_last[level]), 0.50,
                   logistic_fit(sorted(by_level_last[level])))
        print(f"  {level:6} best-of-3 {b:>8}   last {l:>8}")

    base = "L0" if "L0" in stars else sorted(stars)[0]
    print(f"\ncost ratios vs {base} (>= means a bound, not a point estimate):")
    for level in sorted(stars):
        parts = []
        for t in TARGETS:
            parts.append(f"{int(t * 100)}%: {cost_ratio(stars[level][t], stars[base][t])}")
        print(f"  {level:6} " + "  ".join(parts))


if __name__ == "__main__":
    main()
