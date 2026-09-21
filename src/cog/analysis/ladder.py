"""Additive ladder: what each disturbance costs when it is switched on, one at a time.

Companion module to cog.analysis.loo, which asks the opposite question. This one walks the ladder
L0 {} -> L1 {A} -> L2 {A,B} -> L3b {A,B,C}, so a delta is the cost of ADDING one axis on top of
the ones already on: SR(with it) - SR(without it) at a matched demonstration budget. A negative
delta means switching the disturbance on made the policy worse.

Statistics, protocol and the 108 result files are shared with cog.analysis.loo; only the pairings
differ. All four ladder arms were generated on the same GPU, so none of these comparisons crosses
a hardware boundary.
"""
from __future__ import annotations

import pathlib

from cog.analysis.loo import (NDEMOS, TASKS, chi2_homogeneity, furthest_rows, load_surface,
                              newcombe, stage_rows, surface_rows, wilson)

REPO = pathlib.Path(__file__).resolve().parents[3]

# rung -> (arm with the axis on, arm without it). The ladder order is fixed by the study design.
ADD = {"A": ("L1", "L0"), "B": ("L2", "L1"), "C": ("L3b", "L2")}
RUNGS = ["A", "B", "C"]
SETUPS = ["L0", "L1", "L2", "L3b"]
SETUP_NAME = {"L0": "nothing varies", "L1": "+ object start", "L2": "+ goal",
              "L3b": "+ object type"}
SETUP_LONG = {"L0": "nothing varies", "L1": "object start varies",
              "L2": "object start + goal vary", "L3b": "all three vary"}
AXIS_PLAIN = {"A": "object start", "B": "goal", "C": "object type"}


def ladder_rows(surface: dict) -> list[dict]:
    """One row per (task, axis switched on, demo budget)."""
    rows = []
    for task in TASKS:
        for axis, (on, off) in ADD.items():
            for n in NDEMOS:
                a = surface[(task, on, n)]
                b = surface[(task, off, n)]
                d = a["sr"] - b["sr"]
                lo, hi = newcombe(a["k"], a["episodes"], b["k"], b["episodes"])
                rows.append({"task": task, "axis": axis, "on_arm": on, "off_arm": off, "n": n,
                             "sr_on": a["sr"], "k_on": a["k"], "sr_off": b["sr"], "k_off": b["k"],
                             "episodes": a["episodes"], "delta": d, "ci_lo": lo, "ci_hi": hi,
                             "significant": not (lo <= 0.0 <= hi)})
    return rows


def pooled_rows(surface: dict) -> list[dict]:
    """Appendix only -- budgets summed, printed beside the homogeneity test that invalidates it."""
    rows = []
    for task in TASKS:
        for axis, (on, off) in ADD.items():
            ka = sum(surface[(task, on, n)]["k"] for n in NDEMOS)
            kb = sum(surface[(task, off, n)]["k"] for n in NDEMOS)
            nn = 200 * len(NDEMOS)
            lo, hi = newcombe(ka, nn, kb, nn)
            hom_a = chi2_homogeneity([surface[(task, on, n)]["k"] for n in NDEMOS], [200] * 6)
            hom_b = chi2_homogeneity([surface[(task, off, n)]["k"] for n in NDEMOS], [200] * 6)
            rows.append({"task": task, "axis": axis, "on_arm": on, "off_arm": off,
                         "k_on": ka, "k_off": kb, "episodes": nn, "delta": ka / nn - kb / nn,
                         "ci_lo": lo, "ci_hi": hi, "chi2_on": hom_a["chi2"],
                         "chi2_off": hom_b["chi2"], "df": hom_a["df"], "crit95": hom_a["crit95"],
                         "overdispersed": hom_a["overdispersed"] or hom_b["overdispersed"]})
    return rows


def build() -> dict:
    surface = load_surface()
    return {"surface": surface_rows(surface), "ladder": ladder_rows(surface),
            "pooled": pooled_rows(surface), "stages": stage_rows(surface),
            "furthest": furthest_rows(surface), "_raw": surface}


if __name__ == "__main__":
    import csv
    out = REPO / "paper" / "ladder_report" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    data = build()
    for name in ("surface", "ladder", "pooled", "stages", "furthest"):
        rows = data[name]
        cols = sorted({k for r in rows for k in r})
        with (out / f"{name}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out / (name + '.csv')}  ({len(rows)} rows)")
