"""Completion analysis of the clean re-eval sweep (fixed harness, 2026-08-22).

Inputs: results/eval_*_080000_fixed.json (54 flat), results/diagnostics/eval_T2_L3b_n*_fixed.json
(6, stage-instrumented), published L3 files for T1/T3 (clean by construction: one batch/process).
Outputs: experiments/clean_surface.csv, clean_nstar.csv, t2_stage_funnel_full.csv, and a printed
summary covering: N*/cost ratios, batch-0 warm-up quantification (incl. the batches5to9 probe),
finding-4 re-judgment inputs, env-index uniformity, and old-vs-clean discrepancy flags.
"""
import csv
import glob
import json
import math
import re
from pathlib import Path

from scipy.stats import chi2_contingency, norm

R = Path("results")
NS = [10, 25, 50, 100, 200, 400]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def load(fp):
    d = json.load(open(fp))
    return d


surface = {}  # (task, level, n) -> dict
for fp in glob.glob(str(R / "eval_T*_080000_fixed.json")):
    m = re.match(r"eval_(T\d)_(L\d)_n(\d+)_080000_fixed\.json", Path(fp).name)
    if not m:
        continue
    d = load(fp)
    surface[(m.group(1), m.group(2), int(m.group(3)))] = d
for fp in glob.glob(str(R / "diagnostics" / "eval_T2_L3b_n*_080000_fixed.json")):
    m = re.match(r"eval_T2_L3b_n(\d+)_080000_fixed\.json", Path(fp).name)
    d = load(fp)
    surface[("T2", "L3", int(m.group(1)))] = d
for task in ("T1", "T3"):
    for n in NS:
        d = load(R / f"eval_{task}_L3b_n{n}_080000.json")  # published == clean for L3
        surface[(task, "L3", n)] = d

rows = []
for (task, level, n), d in sorted(surface.items()):
    k, N = d["successes"], d["episodes"]
    lo, hi = wilson(k, N)
    rows.append(dict(task=task, level=level, n_demos=n, sr=round(k / N, 4), successes=k,
                     episodes=N, ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
with open("experiments/clean_surface.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

print("== clean surface ==")
for task in ("T1", "T2", "T3"):
    for level in ("L0", "L1", "L2", "L3"):
        vals = [f"{surface[(task, level, n)]['success_rate']:.2f}" for n in NS
                if (task, level, n) in surface]
        print(f"{task} {level}: " + " ".join(vals))


def nstar(task, level, thr):
    pts = [(n, surface[(task, level, n)]["success_rate"]) for n in NS if (task, level, n) in surface]
    for i, (n, sr) in enumerate(pts):
        if sr >= thr:
            if i == 0:
                return f"<={n}", n
            n0, s0 = pts[i - 1]
            if sr == s0:
                return str(n), n
            x = math.log(n0) + (thr - s0) * (math.log(n) - math.log(n0)) / (sr - s0)
            return str(int(round(math.exp(x)))), math.exp(x)
    return ">400", None


print("\n== N* and cost ratios (clean) ==")
nstar_rows = []
for task in ("T1", "T2", "T3"):
    base = {}
    for level in ("L0", "L1", "L2", "L3"):
        e = {"task": task, "level": level}
        for thr in (0.5, 0.8, 0.9):
            s, v = nstar(task, level, thr)
            e[f"nstar_{int(thr*100)}"] = s
            e[f"_v{int(thr*100)}"] = v
        if level == "L0":
            base = e
        r = (e["_v90"] / base["_v90"]) if (e.get("_v90") and base.get("_v90")) else None
        e["cost_ratio_90_vs_L0"] = round(r, 2) if r else (">= " + f"{400/base['_v90']:.1f}x" if base.get("_v90") and not e.get("_v90") else "n/a")
        nstar_rows.append({k: v for k, v in e.items() if not k.startswith("_")})
        print(e["task"], e["level"], "N*50/80/90:", e["nstar_50"], e["nstar_80"], e["nstar_90"],
              "ratio90:", e["cost_ratio_90_vs_L0"])
with open("experiments/clean_nstar.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(nstar_rows[0]))
    w.writeheader()
    w.writerows(nstar_rows)

print("\n== batch-0 warm-up (clean flat cells, guard active) ==")
gaps = []
for (task, level, n), d in sorted(surface.items()):
    if level == "L3" or d["episodes"] != 100:
        continue
    out = d["outcomes"]
    b0 = sum(o["success"] for o in out if o["batch"] == 0) / 20
    rest = sum(o["success"] for o in out if o["batch"] > 0) / 80
    if 0.05 <= d["success_rate"] <= 0.97:
        gaps.append((task, level, n, b0, rest, rest - b0))
pos = sum(1 for g in gaps if g[5] > 0)
print(f"non-saturated flat cells: rest>batch0 in {pos}/{len(gaps)}; "
      f"mean gap {sum(g[5] for g in gaps)/len(gaps):.3f}")
for task in ("T1", "T2", "T3"):
    tg = [g for g in gaps if g[0] == task]
    if tg:
        print(f"  {task}: mean batch0 {sum(g[3] for g in tg)/len(tg):.2f} vs rest "
              f"{sum(g[4] for g in tg)/len(tg):.2f} ({len(tg)} cells)")

print("\n== batches5to9 probe (T1_L2_n400, separate process, batch5 = its first batch) ==")
try:
    d = load(R / "diagnostics" / "eval_T1_L2_n400_080000_batches5to9.json")
    pb = {}
    for o in d["outcomes"]:
        pb.setdefault(o["batch"], []).append(o["success"])
    for b in sorted(pb):
        print(f"  batch {b}: {sum(pb[b])}/{len(pb[b])}")
except Exception as e:
    print("  probe unavailable:", e)

print("\n== T2 stage funnel, full 24 cells (stage | success union not needed: guard active, raw latches artifact-free; success still implies stages at terminal step -> report stage|success) ==")
funnel = []
for level in ("L0", "L1", "L2", "L3"):
    for n in NS:
        d = surface.get(("T2", level, n))
        if not d or "drawer_opened" not in d["outcomes"][0]:
            continue
        out = d["outcomes"]
        N = len(out)
        opened = sum(o["drawer_opened"] or o["success"] for o in out) / N
        lifted = sum(o["object_lifted"] or o["success"] for o in out) / N
        over = sum(o["object_over_drawer"] or o["success"] for o in out) / N
        funnel.append(dict(level=level, n_demos=n, opened=round(opened, 3), lifted=round(lifted, 3),
                           over=round(over, 3), sr=d["success_rate"], episodes=N))
        print(f"  {level} n{n}: opened={opened:.2f} lifted={lifted:.2f} over={over:.2f} SR={d['success_rate']:.2f}")
with open("experiments/t2_stage_funnel_full.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(funnel[0]))
    w.writeheader()
    w.writerows(funnel)

print("\n== env-index uniformity (pooled clean flat cells per task, non-saturated) ==")
for task in ("T1", "T2", "T3"):
    tbl = []
    for (t, level, n), d in surface.items():
        if t != task or level == "L3" or not (0.05 <= d["success_rate"] <= 0.95):
            continue
        for o in d["outcomes"]:
            tbl.append((o["env"], o["success"]))
    if not tbl:
        print(f"  {task}: no non-saturated cells")
        continue
    succ = [sum(1 for e, s in tbl if e == i and s) for i in range(20)]
    tot = [sum(1 for e, s in tbl if e == i) for i in range(20)]
    chi2, p, *_ = chi2_contingency([succ, [t - s for t, s in zip(tot, succ)]])
    print(f"  {task}: env success counts min={min(succ)} max={max(succ)} (per {tot[0]} eps); "
          f"chi2={chi2:.1f} p={p:.2e}")

print("\n== old-vs-clean discrepancies (flat cells, |clean-published| in sigma) ==")
flagged = []
for (task, level, n), d in sorted(surface.items()):
    if level == "L3":
        continue
    try:
        old = load(R / f"eval_{task}_{level}_n{n}_080000.json")
    except FileNotFoundError:
        continue
    p1, p0 = d["success_rate"], old["success_rate"]
    se = math.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 100 + 1e-9)
    z = (p1 - p0) / se if se else 0
    if abs(z) > 2:
        flagged.append((task, level, n, p0, p1, round(z, 1)))
for f_ in flagged:
    print("  ", f_)
print(f"  {len(flagged)} flat cells differ >2 sigma from published (expected direction: negative for phantom-inflated mid-SR cells)")
