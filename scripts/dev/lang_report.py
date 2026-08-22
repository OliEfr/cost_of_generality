"""Summarize language-conditioning investigation evals (results/diagnostics/).

Reads eval_lang_*.json and probe_*.json, prints per-run SR with Wilson 95% CI,
per-instruction min/median/max SR (n=5 per instruction -- SPREAD INDICATOR ONLY,
a Wilson CI at n=5 is roughly +/-0.4), and probe match-vs-swap deltas with the
PASS / LANGUAGE IGNORED / INCONCLUSIVE verdict from the approved protocol.

Usage: python scripts/dev/lang_report.py [--dir results/diagnostics] [--baseline 0.86]
"""

import argparse
import json
import math
import statistics
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def describe(path: Path) -> dict:
    d = json.loads(path.read_text())
    k, n = d["successes"], d["episodes"]
    lo, hi = wilson(k, n)
    row = {"file": path.name, "sr": k / n, "k": k, "n": n, "ci": (lo, hi)}
    if "per_instruction" in d:
        srs = [v["successes"] / v["episodes"] for v in d["per_instruction"].values()]
        row["instr_min"], row["instr_med"], row["instr_max"] = (
            min(srs), statistics.median(srs), max(srs))
    if "instructions" in d:
        row["swap"] = d["instructions"].get("swap_instructions_from")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/diagnostics")
    ap.add_argument("--baseline", type=float, default=0.86,
                    help="language-free reference SR for the verify cell (t1_L1_n100_s0)")
    ap.add_argument("--baseline_n", type=int, default=100)
    args = ap.parse_args()

    root = Path(args.dir)
    files = sorted(root.glob("eval_lang_*.json")) + sorted(root.glob("probe_*.json"))
    if not files:
        raise SystemExit(f"no eval_lang_*/probe_* files in {root}")

    blo, bhi = wilson(round(args.baseline * args.baseline_n), args.baseline_n)
    print(f"baseline t1_L1_n100_s0: SR={args.baseline:.3f} Wilson95 [{blo:.3f},{bhi:.3f}]\n")
    print(f"{'file':58s} {'SR':>6s} {'Wilson95':>15s} {'instr min/med/max':>18s}")
    rows = {}
    for f in files:
        r = describe(f)
        rows[f.name] = r
        spread = (f"{r['instr_min']:.2f}/{r['instr_med']:.2f}/{r['instr_max']:.2f}"
                  if "instr_min" in r else "-")
        print(f"{r['file']:58s} {r['sr']:6.3f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}]"
              f" {spread:>18s}")

    # probe verdicts: pair probe_<cand>_<env>env_match_* with ..._swap_*
    pairs = {}
    for name, r in rows.items():
        if not name.startswith("probe_"):
            continue
        parts = name.split("_")
        key = (parts[1], parts[2])  # (cand, T1env/T3env)
        pairs.setdefault(key, {})["match" if "match" in name else "swap"] = r
    for (cand, envtag), pr in sorted(pairs.items()):
        if "match" not in pr or "swap" not in pr:
            continue
        m, s = pr["match"], pr["swap"]
        delta = m["sr"] - s["sr"]
        disjoint = m["ci"][0] > s["ci"][1]
        if m["sr"] >= 0.50 and delta >= 0.30 and disjoint:
            verdict = "PASS (language steers behavior)"
        elif m["sr"] >= 0.50 and s["ci"][1] >= m["ci"][0]:
            verdict = "LANGUAGE IGNORED (vision dispatch suffices)"
        elif m["sr"] < 0.50:
            verdict = "INCONCLUSIVE (multi-task training weak)"
        else:
            verdict = "BORDERLINE (delta or CI criterion missed)"
        print(f"\nprobe {cand} {envtag}: match SR={m['sr']:.3f} {m['ci']}, "
              f"swap SR={s['sr']:.3f} {s['ci']}, delta={delta:+.3f} -> {verdict}")


if __name__ == "__main__":
    main()
