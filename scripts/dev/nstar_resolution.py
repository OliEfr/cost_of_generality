#!/usr/bin/env python
"""How precisely can this study's design resolve N*(s) and the cost ratios?

The demo grid is 6 log-spaced cells (10..400) evaluated on 100 frozen episodes, so each
cell's SR carries about +-5 points of binomial noise. This asks what that noise does to the
derived quantities the paper reports, by simulating cells from a KNOWN logistic and
re-estimating N* the same way cog.analysis.curves does.

  python scripts/dev/nstar_resolution.py

Run before the matrix, not after: if the design cannot resolve the effect being claimed,
that is worth knowing while the eval budget is still a choice.
"""

from __future__ import annotations

import math
import random
import statistics

import sys
sys.path.insert(0, "src")
from cog.analysis.curves import logistic_fit, n_star, n_star_fit  # noqa: E402

GRID = (10, 25, 50, 100, 200, 400)
SLOPE = 2.2          # logit per decade; matches the synthetic study in curves.py's test
TRIALS = 400


def simulate(mid: float, eval_n: int, rng: random.Random) -> list[tuple[int, float]]:
    pts = []
    for n in GRID:
        p = 1 / (1 + math.exp(-SLOPE * (math.log10(n) - math.log10(mid))))
        k = sum(1 for _ in range(eval_n) if rng.random() < p)
        pts.append((n, k / eval_n))
    return pts


def rel_errors(mid: float, eval_n: int, target: float = 0.50):
    rng = random.Random(12345)
    interp, fit = [], []
    for _ in range(TRIALS):
        pts = simulate(mid, eval_n, rng)
        f = logistic_fit(pts)
        for est, bucket in ((n_star(pts, target, f), interp),
                            (n_star_fit(f, target, max(GRID)), fit)):
            if est.isdigit():
                bucket.append((int(est) - mid) / mid)
    return interp, fit


def summarise(name: str, errs: list[float]) -> str:
    if not errs:
        return f"{name}: no numeric crossings"
    med = statistics.median(errs)
    p90 = sorted(abs(e) for e in errs)[int(0.9 * len(errs)) - 1]
    return (f"{name}: bias {med*100:+5.1f}%  |err| p90 {p90*100:4.1f}%  "
            f"(n={len(errs)}/{TRIALS} resolved)")


def main() -> None:
    print(f"Monte Carlo: {TRIALS} trials per condition, true logistic slope {SLOPE}/decade,")
    print(f"grid {GRID}, target s=50%\n")
    for eval_n in (100, 200):
        print(f"--- {eval_n} eval episodes per cell ---")
        for mid in (30.0, 60.0, 120.0, 240.0):
            interp, fit = rel_errors(mid, eval_n)
            print(f"  true N*={mid:5.0f}  " + summarise("interp", interp))
            print(f"  {'':16}" + summarise("fit   ", fit))
        print()

    # What the noise does to a cost RATIO, which is what the paper actually claims.
    print("--- cost ratio N*(level)/N*(L0), 100 eval episodes, true ratio 2.0x ---")
    rng = random.Random(999)
    ratios = []
    for _ in range(TRIALS):
        a = simulate(30.0, 100, rng)
        b = simulate(60.0, 100, rng)
        fa, fb = logistic_fit(a), logistic_fit(b)
        sa, sb = n_star_fit(fa, 0.50, max(GRID)), n_star_fit(fb, 0.50, max(GRID))
        if sa.isdigit() and sb.isdigit() and int(sa) > 0:
            ratios.append(int(sb) / int(sa))
    if ratios:
        ratios.sort()
        lo, hi = ratios[int(0.05 * len(ratios))], ratios[int(0.95 * len(ratios)) - 1]
        print(f"  median {statistics.median(ratios):.2f}x, 90% of estimates in "
              f"[{lo:.2f}x, {hi:.2f}x]  (n={len(ratios)}/{TRIALS})")
        print(f"  -> a true 2x effect is resolved; an effect below ~{hi/2:.1f}x would not be"
              " distinguishable from 1x at this eval budget.")


if __name__ == "__main__":
    main()
