"""Leave-one-out ablation: all quantities reported in paper/loo_report.

This module is the single source of truth for the study's numbers. It was written after an
audit found the headline table was being produced by a throwaway scratch script, which meant
nothing in the repo could reproduce it (journal 2026-09-20, M5).

Design, in one paragraph. Every cell of this study is 200 episodes scored under the D31
uniform-slice protocol (ten independent processes, each running one unscored warm-up batch and
then scoring 20 episodes at seed 5000+s) with the D32 success signal. The full disturbance set
is L3b = {A pose, B goal, C variant}; the leave-one-out arms remove exactly one axis from it:
BC = L3b\\A, AC = L3b\\B, and L2 = L3b\\C (the variant axis collapsed onto its default member,
which is bit-identical to L2 -- verified at runtime, journal 2026-09-20). A delta is therefore
always SR(arm with the axis removed) - SR(L3b) at a matched demo budget, so a positive delta
means the simplified arm scored higher.

No pooling across demo budgets is done in the primary tables: the six budgets are six different
policies, not six samples of one. A pooled row is provided in the appendix WITH its homogeneity
test so the dispersion it hides is visible beside it.
"""
from __future__ import annotations

import json
import math
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[3]
RESULTS = REPO / "results"
SUFFIX = "u200d32"
STEP = "080000"

TASKS = {"T1": "cup_place", "T2": "drawer_stow", "T3": "push_target"}
TASK_LABEL = {"T1": "T1 cup_place", "T2": "T2 drawer_stow", "T3": "T3 push_target"}
ARMS = ["L0", "L1", "L2", "L3b", "AC", "BC"]
NDEMOS = [10, 25, 50, 100, 200, 400]

# The leave-one-out map: axis removed -> the arm that is L3b without it.
REMOVE = {"A": "BC", "B": "AC", "C": "L2"}
AXIS_LABEL = {"A": "A  manipulandum pose", "B": "B  goal / fixture pose", "C": "C  object variant"}
# Only the C pair is episode-matched: L2 and the L3b diagonal draw identical poses at every seed
# and differ solely in which variant is spawned. A and B change the pose distribution itself.
PAIRED = {"C"}

STAGE_KEYS = ("drawer_opened", "object_lifted", "object_over_drawer")

Z = 1.959963984540054


# ---------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for one proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe(k1: int, n1: int, k2: int, n2: int, z: float = Z) -> tuple[float, float]:
    """Newcombe method 10 interval for p1 - p2, two INDEPENDENT samples.

    Preferred over the Wald/normal-approximation interval because several cells here sit near
    0 or 1, where the normal approximation's interval leaves the parameter space.
    """
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (max(-1.0, lo), min(1.0, hi))


def mcnemar(b: int, c: int, n: int, z: float = Z) -> dict:
    """Paired comparison. b = only-first successes, c = only-second, over n matched episodes."""
    d = (b - c) / n
    var = (b + c - (b - c) ** 2 / n) / (n * n) if n else float("nan")
    se = math.sqrt(max(var, 0.0))
    chi2 = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) > 0 else 0.0
    return {"b": b, "c": c, "n": n, "diff": d, "lo": d - z * se, "hi": d + z * se,
            "chi2_cc": chi2, "significant": chi2 > 3.841459}


def chi2_homogeneity(ks: list[int], ns: list[int]) -> dict:
    """Pearson chi-square that a set of binomial cells share one proportion."""
    K, N = sum(ks), sum(ns)
    if N == 0 or K in (0, N):
        return {"chi2": 0.0, "df": max(len(ks) - 1, 0), "overdispersed": False}
    p = K / N
    chi2 = sum((k - n * p) ** 2 / (n * p * (1 - p)) for k, n in zip(ks, ns) if n)
    df = len(ks) - 1
    crit = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070, 6: 12.592,
            7: 14.067, 8: 15.507, 9: 16.919}.get(df, 1e9)
    return {"chi2": chi2, "df": df, "crit95": crit, "overdispersed": chi2 > crit}


# ---------------------------------------------------------------- loading


def cell_path(task: str, arm: str, n: int) -> pathlib.Path:
    return RESULTS / f"eval_{task}_{arm}_n{n}_{STEP}_{SUFFIX}.json"


def load_surface() -> dict:
    """{(task, arm, n): cell dict} for all 108 cells, with the per-slice block normalised."""
    surface = {}
    for task in TASKS:
        for arm in ARMS:
            for n in NDEMOS:
                p = cell_path(task, arm, n)
                if not p.exists():
                    raise FileNotFoundError(p)
                d = json.loads(p.read_text())
                block = d.get("per_variant") or d.get("per_batch")
                slices = [block[k] for k in sorted(block)]
                assert len(slices) == 10, (p, len(slices))
                assert d["episodes"] == 200, (p, d["episodes"])
                assert d["protocol"]["warmup"]["batches"] == 1, p
                surface[(task, arm, n)] = {
                    "task": task, "arm": arm, "n": n,
                    "k": d["successes"], "episodes": d["episodes"], "sr": d["success_rate"],
                    "slice_k": [s["successes"] for s in slices],
                    "slice_n": [s["episodes"] for s in slices],
                    "slice_keys": sorted(block),
                    "outcomes": d["outcomes"],
                    "stages": d.get("stages"),
                    "env": d["task"],
                }
    return surface


# ---------------------------------------------------------------- derived tables


def surface_rows(surface: dict) -> list[dict]:
    rows = []
    for (task, arm, n), c in sorted(surface.items()):
        lo, hi = wilson(c["k"], c["episodes"])
        hom = chi2_homogeneity(c["slice_k"], c["slice_n"])
        rows.append({"task": task, "arm": arm, "n": n, "k": c["k"], "episodes": c["episodes"],
                     "sr": c["sr"], "wilson_lo": lo, "wilson_hi": hi,
                     "slice_min": min(c["slice_k"]) / 20, "slice_max": max(c["slice_k"]) / 20,
                     "slice_chi2": hom["chi2"], "slice_df": hom["df"],
                     "slice_overdispersed": hom["overdispersed"]})
    return rows


def loo_rows(surface: dict) -> list[dict]:
    """One row per (task, axis removed, demo budget)."""
    rows = []
    for task in TASKS:
        for axis, arm in REMOVE.items():
            for n in NDEMOS:
                full = surface[(task, "L3b", n)]
                red = surface[(task, arm, n)]
                d = red["sr"] - full["sr"]
                lo, hi = newcombe(red["k"], red["episodes"], full["k"], full["episodes"])
                row = {"task": task, "axis": axis, "removed_arm": arm, "n": n,
                       "sr_full": full["sr"], "k_full": full["k"],
                       "sr_reduced": red["sr"], "k_reduced": red["k"],
                       "episodes": full["episodes"], "delta": d,
                       "ci_lo": lo, "ci_hi": hi, "significant": not (lo <= 0.0 <= hi),
                       "paired": axis in PAIRED}
                if axis in PAIRED:
                    ro = [o["success"] for o in red["outcomes"]]
                    fo = [o["success"] for o in full["outcomes"]]
                    assert len(ro) == len(fo) == 200
                    b = sum(1 for r, f in zip(ro, fo) if r and not f)
                    c = sum(1 for r, f in zip(ro, fo) if f and not r)
                    row["mcnemar"] = mcnemar(b, c, len(ro))
                rows.append(row)
    return rows


def pooled_rows(surface: dict) -> list[dict]:
    """Appendix only: the same deltas with all six budgets summed, beside their homogeneity test."""
    rows = []
    for task in TASKS:
        for axis, arm in REMOVE.items():
            kf = sum(surface[(task, "L3b", n)]["k"] for n in NDEMOS)
            kr = sum(surface[(task, arm, n)]["k"] for n in NDEMOS)
            nn = 200 * len(NDEMOS)
            lo, hi = newcombe(kr, nn, kf, nn)
            hom_f = chi2_homogeneity([surface[(task, "L3b", n)]["k"] for n in NDEMOS],
                                     [200] * len(NDEMOS))
            hom_r = chi2_homogeneity([surface[(task, arm, n)]["k"] for n in NDEMOS],
                                     [200] * len(NDEMOS))
            rows.append({"task": task, "axis": axis, "removed_arm": arm,
                         "k_full": kf, "k_reduced": kr, "episodes": nn,
                         "delta": kr / nn - kf / nn, "ci_lo": lo, "ci_hi": hi,
                         "chi2_full": hom_f["chi2"], "chi2_reduced": hom_r["chi2"],
                         "df": hom_f["df"], "crit95": hom_f["crit95"],
                         "overdispersed": hom_f["overdispersed"] or hom_r["overdispersed"]})
    return rows


def stage_rows(surface: dict) -> list[dict]:
    """T2 only. Absolute stage rates plus the conditional rate of each transition."""
    rows = []
    for arm in ARMS:
        for n in NDEMOS:
            c = surface[("T2", arm, n)]
            o = c["outcomes"]
            tot = len(o)
            opened = sum(1 for x in o if x["drawer_opened"])
            lifted = sum(1 for x in o if x["object_lifted"])
            over = sum(1 for x in o if x["object_over_drawer"])
            succ = sum(1 for x in o if x["success"])
            # The milestones are NOT nested -- an episode can lift the object without ever
            # opening the drawer past threshold -- so a conditional rate has to come from the
            # joint count, not from the ratio of two marginals (which can exceed 1).
            lift_and_open = sum(1 for x in o if x["object_lifted"] and x["drawer_opened"])
            over_and_lift = sum(1 for x in o if x["object_over_drawer"] and x["object_lifted"])
            succ_and_over = sum(1 for x in o if x["success"] and x["object_over_drawer"])
            rows.append({
                "arm": arm, "n": n, "episodes": tot,
                "opened": opened, "lifted": lifted, "over": over, "success": succ,
                "lift_and_open": lift_and_open, "over_and_lift": over_and_lift,
                "success_and_over": succ_and_over,
                "p_opened": opened / tot, "p_lifted": lifted / tot,
                "p_over": over / tot, "p_success": succ / tot,
                "lift_given_open": (lift_and_open / opened) if opened else float("nan"),
                "over_given_lift": (over_and_lift / lifted) if lifted else float("nan"),
                "success_given_over": (succ_and_over / over) if over else float("nan"),
                "max_open_median": _median([x["max_drawer_open"] for x in o]),
                "max_lift_median": _median([x["max_object_lift"] for x in o]),
            })
    return rows


def timing_rows(surface: dict) -> list[dict]:
    """Step index of the first success, over successful episodes only."""
    rows = []
    for task in TASKS:
        for arm in ARMS:
            ts = []
            for n in NDEMOS:
                ts += [o["t_first_success"] for o in surface[(task, arm, n)]["outcomes"]
                       if o["success"] and o["t_first_success"] >= 0]
            if not ts:
                rows.append({"task": task, "arm": arm, "n_success": 0})
                continue
            ts.sort()
            rows.append({"task": task, "arm": arm, "n_success": len(ts),
                         "min": ts[0], "p25": _quantile(ts, .25), "median": _median(ts),
                         "p75": _quantile(ts, .75), "max": ts[-1]})
    return rows


def _median(xs):
    return _quantile(sorted(xs), 0.5)


def _quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = q * (len(xs) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def gen_rows() -> list[dict]:
    """Generation statistics for the six arms, from the rule-9 authority file."""
    import csv
    path = REPO / "experiments" / "gen_stats.csv"
    out = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            task = {v: k for k, v in TASKS.items()}[r["task"]]
            lvl = r["level"]
            m = re.match(r"^(L3b|L3)v(\d+)$", lvl)
            arm = "L3b" if (m and m.group(1) == "L3b") else lvl
            if arm not in ARMS:
                continue
            key = (task, arm)
            e = out.setdefault(key, {"task": task, "arm": arm, "successes": 0,
                                     "failures": 0, "attempts": 0, "legs": 0, "ep_len": []})
            e["successes"] += int(r["successes"])
            e["failures"] += int(r["failures"])
            e["attempts"] += int(r["attempts"])
            e["legs"] += 1
            e["ep_len"].append(float(r["mean_ep_len"]))
    rows = []
    for (task, arm), e in sorted(out.items()):
        rows.append({"task": task, "arm": arm, "legs": e["legs"],
                     "demos": e["successes"], "failures": e["failures"],
                     "attempts": e["attempts"],
                     "gen_sr": e["successes"] / e["attempts"] if e["attempts"] else float("nan"),
                     "mean_ep_len": sum(e["ep_len"]) / len(e["ep_len"])})
    return rows


def build() -> dict:
    surface = load_surface()
    return {"surface": surface_rows(surface), "loo": loo_rows(surface),
            "pooled": pooled_rows(surface), "stages": stage_rows(surface),
            "timing": timing_rows(surface), "gen": gen_rows(), "_raw": surface}


if __name__ == "__main__":
    import csv as _csv
    out_dir = REPO / "paper" / "loo_report" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = build()
    for name in ("surface", "loo", "pooled", "stages", "timing", "gen"):
        rows = data[name]
        flat = []
        for r in rows:
            r = dict(r)
            mc = r.pop("mcnemar", None)
            if mc:
                r.update({f"mcnemar_{k}": v for k, v in mc.items()})
            flat.append(r)
        cols = sorted({k for r in flat for k in r})
        with (out_dir / f"{name}.csv").open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(flat)
        print(f"wrote {out_dir / (name + '.csv')}  ({len(flat)} rows)")
