"""T2 L1-vs-L2 inversion analysis (Q1) + horizon sanity check (Q2).

Reads frozen eval sets, per-episode outcomes, lerobot metadata, and hdf5 initial
states. Writes t2_-prefixed CSVs into the worktree experiments/ dir and prints a
structured report. Read-only w.r.t. the main repo.
"""

from __future__ import annotations

import glob
import json
import pathlib

import h5py
import numpy as np
import pandas as pd
from scipy import stats

MAIN = pathlib.Path("/home/admin_07/cost_of_generality")
WT = MAIN / ".claude/worktrees/results-analysis"
EVAL_SETS = MAIN / "configs/eval_sets"
RESULTS = WT / "results"  # identical committed copies exist in the worktree
OUT = WT / "experiments"
NS = [10, 25, 50, 100, 200, 400]


def yaw_of(q):
    q = np.asarray(q, dtype=float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def env_origins():
    """Per-env origins from L1's FIXED cabinet: world cab = origin + (0.9, 0.0).

    The grid is 5x4 at 3 m spacing with y offsets at half-grid (+-1.5, +-4.5), so
    naive rounding of object coords fails; the fixed asset pins the origin exactly.
    Verified constant across batches before use.
    """
    d = json.loads((EVAL_SETS / "T2_L1.json").read_text())
    orgs = None
    for batch in d["batches"]:
        cp = np.asarray(batch["cabinet_pos"])[:, :2] - np.array([0.9, 0.0])
        if orgs is None:
            orgs = cp
        else:
            assert np.abs(orgs - cp).max() < 1e-5, "env origins differ across batches"
    return orgs  # (num_envs, 2)


def load_eval_set(level, orgs):
    d = json.loads((EVAL_SETS / f"T2_{level}.json").read_text())
    rows = []
    for b, batch in enumerate(d["batches"]):
        op = np.asarray(batch["object_pos"])
        oy = yaw_of(np.asarray(batch["object_quat"]))
        cp = np.asarray(batch["cabinet_pos"])
        cy = yaw_of(np.asarray(batch["cabinet_quat"]))
        cj = np.asarray(batch["cabinet_joint_pos"])
        ox_org, oy_org = orgs[:, 0], orgs[:, 1]
        for i in range(op.shape[0]):
            rows.append(dict(
                level=level, batch=b, env=i,
                obj_x=op[i, 0] - ox_org[i], obj_y=op[i, 1] - oy_org[i],
                obj_yaw=oy[i],
                cab_x=cp[i, 0] - ox_org[i], cab_y=cp[i, 1] - oy_org[i],
                cab_yaw=cy[i], cab_joint_max=np.abs(cj[i]).max(),
            ))
    df = pd.DataFrame(rows)
    # sanity: residuals must be inside declared ranges (+tiny slack)
    assert df.obj_x.between(0.15, 0.27).all() and df.obj_y.between(0.35, 0.55).all(), level
    assert df.cab_x.between(0.84, 0.96).all() and df.cab_y.between(-0.07, 0.07).all(), level
    return df


def load_outcomes(level):
    rows = []
    for n in NS:
        p = RESULTS / f"eval_T2_{level}_n{n}_080000.json"
        d = json.loads(p.read_text())
        assert d["episodes"] == 100
        for o in d["outcomes"]:
            rows.append(dict(level=level, n=n, batch=o["batch"], env=o["env"],
                             success=bool(o["success"])))
    return pd.DataFrame(rows)


def two_prop(k1, n1, k2, n2):
    """z-test (pooled) + Fisher exact. Returns z, p_z, p_fisher, odds ratio."""
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else np.nan
    pz = 2 * stats.norm.sf(abs(z))
    orr, pf = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    return z, pz, pf, orr


def sr_by_bins(df, col, nbins=4, use_abs=False, center=0.0):
    v = df[col].to_numpy(dtype=float)
    if use_abs:
        v = np.abs(v - center)
    s = df.success.to_numpy()
    edges = np.quantile(v, np.linspace(0, 1, nbins + 1))
    edges[-1] += 1e-9
    out = []
    for k in range(nbins):
        m = (v >= edges[k]) & (v < edges[k + 1])
        if m.sum():
            out.append((f"[{edges[k]:.3f},{edges[k+1]:.3f})", int(m.sum()),
                        float(s[m].mean())))
    # Mann-Whitney U on the covariate between successes and failures
    if s.any() and (~s).any():
        u, pu = stats.mannwhitneyu(v[s], v[~s])
        d_succ, d_fail = v[s].mean(), v[~s].mean()
    else:
        pu, d_succ, d_fail = np.nan, np.nan, np.nan
    return out, pu, d_succ, d_fail


def main():
    orgs = env_origins()
    es = {lvl: load_eval_set(lvl, orgs) for lvl in ("L1", "L2")}
    oc = {lvl: load_outcomes(lvl) for lvl in ("L1", "L2")}

    print("=" * 88)
    print("SECTION 1: eval-set config reality (frozen T2_L1.json vs T2_L2.json)")
    e1, e2 = es["L1"], es["L2"]
    m = e1.merge(e2, on=["batch", "env"], suffixes=("_1", "_2"))
    same_obj = (np.abs(m.obj_x_1 - m.obj_x_2).max(),
                np.abs(m.obj_y_1 - m.obj_y_2).max(),
                np.abs(m.obj_yaw_1 - m.obj_yaw_2).max())
    print(f"object pose identity across levels: max |dx|,|dy|,|dyaw| = "
          f"{same_obj[0]:.2e}, {same_obj[1]:.2e}, {same_obj[2]:.2e}")
    print(f"L1 cabinet: x unique={sorted(e1.cab_x.round(6).unique())}, "
          f"y unique={sorted(e1.cab_y.round(6).unique())}, "
          f"yaw dev from pi max={np.abs(np.abs(e1.cab_yaw)-np.pi).max():.4f}")
    print(f"L2 cabinet: x in [{e2.cab_x.min():.4f},{e2.cab_x.max():.4f}] mean {e2.cab_x.mean():.4f}; "
          f"y in [{e2.cab_y.min():.4f},{e2.cab_y.max():.4f}]; "
          f"|yaw-pi| max {np.abs(np.abs(e2.cab_yaw)-np.pi).max():.4f}")
    print(f"drawer initial joint (max |q|): L1={e1.cab_joint_max.max():.2e}  "
          f"L2={e2.cab_joint_max.max():.2e}  (0 = closed in every episode)")
    for lvl, e in es.items():
        print(f"{lvl} eval object pose: x[{e.obj_x.min():.3f},{e.obj_x.max():.3f}] "
              f"y[{e.obj_y.min():.3f},{e.obj_y.max():.3f}] "
              f"|yaw| mean {np.abs(e.obj_yaw).mean():.3f} max {np.abs(e.obj_yaw).max():.3f}")

    print()
    print("=" * 88)
    print("SECTION 2: statistics of the L2>L1 inversion")
    hdr = f"{'N':>4} {'L1':>7} {'L2':>7} {'z':>6} {'p(z)':>8} {'p(Fisher)':>10} {'OR':>6}"
    print(hdr)
    tab = []
    for n in NS:
        k1 = int(oc["L1"].query("n==@n").success.sum())
        k2 = int(oc["L2"].query("n==@n").success.sum())
        z, pz, pf, orr = two_prop(k2, 100, k1, 100)  # L2 vs L1
        tab.append(dict(n=n, L1=k1, L2=k2, z=z, p_z=pz, p_fisher=pf, odds_ratio=orr))
        print(f"{n:>4} {k1:>4}/100 {k2:>4}/100 {z:>6.2f} {pz:>8.4f} {pf:>10.4f} {orr:>6.2f}")
    pd.DataFrame(tab).to_csv(OUT / "t2_l1_vs_l2_tests.csv", index=False)

    for label, ns in (("pooled all N", NS), ("pooled N>=50", [50, 100, 200, 400])):
        k1 = int(oc["L1"][oc["L1"].n.isin(ns)].success.sum())
        k2 = int(oc["L2"][oc["L2"].n.isin(ns)].success.sum())
        n1 = 100 * len(ns)
        z, pz, pf, orr = two_prop(k2, n1, k1, n1)
        print(f"{label}: L1 {k1}/{n1} vs L2 {k2}/{n1}: z={z:.2f} p_z={pz:.5f} "
              f"p_fisher={pf:.5f} OR={orr:.2f}")

    # CMH across N strata (L2 vs L1)
    num = den = var = 0.0
    for n in NS:
        k2 = int(oc["L2"].query("n==@n").success.sum())
        k1 = int(oc["L1"].query("n==@n").success.sum())
        a, b_, c, d = k2, 100 - k2, k1, 100 - k1
        t = 200.0
        num += a - (a + b_) * (a + c) / t
        var += (a + b_) * (c + d) * (a + c) * (b_ + d) / (t * t * (t - 1))
    cmh = num ** 2 / var
    p_cmh = stats.chi2.sf(cmh, 1)
    print(f"CMH across the 6 N-strata: chi2={cmh:.2f} p={p_cmh:.5f}")

    # paired-by-episode (object poses identical): per (b,i), #successes over N>=50
    big = [50, 100, 200, 400]
    p1 = (oc["L1"][oc["L1"].n.isin(big)].groupby(["batch", "env"]).success.sum())
    p2 = (oc["L2"][oc["L2"].n.isin(big)].groupby(["batch", "env"]).success.sum())
    diff = (p2 - p1).to_numpy()
    w = stats.wilcoxon(diff) if np.any(diff) else None
    npos, nneg = int((diff > 0).sum()), int((diff < 0).sum())
    sign_p = stats.binomtest(npos, npos + nneg).pvalue
    print(f"paired by episode (same object pose, N>=50 pooled): "
          f"L2 better on {npos} eps, L1 better on {nneg}, tied {int((diff==0).sum())}; "
          f"sign test p={sign_p:.5f}; Wilcoxon p={w.pvalue:.5f}" if w else "no diffs")

    print()
    print("=" * 88)
    print("SECTION 3: success vs pose (joined eval sets x outcomes)")
    joined = {}
    for lvl in ("L1", "L2"):
        j = oc[lvl].merge(es[lvl], on=["batch", "env"])
        j["cab_yaw_dev"] = np.abs(np.abs(j.cab_yaw) - np.pi)
        j["carry_dist"] = np.hypot(j.cab_x - j.obj_x, j.cab_y - j.obj_y)
        joined[lvl] = j
    all_j = pd.concat(joined.values())
    all_j.to_csv(OUT / "t2_episode_join.csv", index=False)

    for lvl in ("L1", "L2"):
        j = joined[lvl][joined[lvl].n >= 50]
        print(f"\n-- {lvl}, pooled N>=50 ({len(j)} episode-evals, SR={j.success.mean():.3f})")
        for col, ab, ctr in (("obj_yaw", True, 0.0), ("obj_x", False, 0.0),
                             ("obj_y", False, 0.0)):
            bins, pu, ms, mf = sr_by_bins(j, col, use_abs=ab, center=ctr)
            nm = f"|{col}|" if ab else col
            print(f"  {nm:>10}: " + "  ".join(f"{b} n={n} SR={s:.2f}" for b, n, s in bins))
            print(f"  {nm:>10}: MWU succ-vs-fail p={pu:.4f} (mean succ {ms:.3f} vs fail {mf:.3f})")
        if lvl == "L2":
            for col, ab, ctr in (("cab_x", False, 0.0), ("cab_y", False, 0.0),
                                 ("cab_yaw_dev", False, 0.0), ("carry_dist", False, 0.0)):
                bins, pu, ms, mf = sr_by_bins(j, col, use_abs=ab, center=ctr)
                print(f"  {col:>10}: " + "  ".join(f"{b} n={n} SR={s:.2f}" for b, n, s in bins))
                print(f"  {col:>10}: MWU succ-vs-fail p={pu:.4f} (mean succ {ms:.3f} vs fail {mf:.3f})")

    # is L1's success set narrower/wider than L2's in object-yaw?
    for lvl in ("L1", "L2"):
        j = joined[lvl][joined[lvl].n >= 50]
        sy = np.abs(j[j.success].obj_yaw)
        fy = np.abs(j[~j.success].obj_yaw)
        print(f"{lvl} N>=50: |obj_yaw| of successes mean {sy.mean():.3f} sd {sy.std():.3f} "
              f"vs failures {fy.mean():.3f} sd {fy.std():.3f}")

    print()
    print("=" * 88)
    print("SECTION 4: training-data differences (lerobot metadata, nested subsets)")
    sub_rows = []
    for lvl in ("L1", "L2"):
        files = sorted(glob.glob(str(MAIN / f"data/lerobot/T2_{lvl}/meta/episodes/**/*.parquet"),
                                 recursive=True))
        eps = pd.concat([pd.read_parquet(f) for f in files]).sort_values("episode_index")
        lencol = "length" if "length" in eps.columns else [c for c in eps.columns if "length" in c][0]
        lengths = eps[lencol].to_numpy()
        for n in NS:
            fr = int(lengths[:n].sum())
            sub_rows.append(dict(level=lvl, n=n, frames=fr,
                                 epochs_at_80k_b64=80000 * 64 / fr,
                                 mean_len=lengths[:n].mean(), max_len=int(lengths[:n].max()),
                                 min_len=int(lengths[:n].min())))
        print(f"T2_{lvl}: 400-ep total {int(lengths.sum())} frames, mean {lengths.mean():.1f}, "
              f"max {int(lengths.max())}, min {int(lengths.min())}")
    sub = pd.DataFrame(sub_rows)
    sub.to_csv(OUT / "t2_subset_epochs.csv", index=False)
    piv = sub.pivot(index="n", columns="level", values=["epochs_at_80k_b64", "mean_len", "max_len"])
    print(piv.round(2).to_string())

    print()
    print("=" * 88)
    print("SECTION 5: demonstrator filter bias in the CABINET dimension (L2 only)")
    def cab_init(path):
        rows = []
        with h5py.File(path, "r") as f:
            if "data" not in f:
                return None
            for demo in f["data"]:
                g = f[f"data/{demo}/initial_state"]
                cab = np.asarray(g["articulation/cabinet/root_pose"])[0]
                obj = np.asarray(g["rigid_object/object/root_pose"])[0]
                rows.append(dict(demo=demo, cab_x=cab[0], cab_y=cab[1],
                                 cab_yaw=yaw_of(cab[None, 3:7])[0],
                                 obj_x=obj[0], obj_y=obj[1],
                                 obj_yaw=yaw_of(obj[None, 3:7])[0]))
        return pd.DataFrame(rows)

    ret = cab_init(MAIN / "data/hdf5/T2_L2.hdf5")
    rej = cab_init(MAIN / "data/hdf5/T2_L2_failed.hdf5")
    ret["cab_yaw_dev"] = np.abs(np.abs(ret.cab_yaw) - np.pi)
    rej["cab_yaw_dev"] = np.abs(np.abs(rej.cab_yaw) - np.pi)
    print(f"L2 retained demos n={len(ret)}, rejected attempts n={len(rej)}")
    for col in ("cab_x", "cab_y", "cab_yaw_dev", "obj_x", "obj_y"):
        ks = stats.ks_2samp(ret[col], rej[col])
        print(f"  {col}: retained mean {ret[col].mean():.4f} vs rejected {rej[col].mean():.4f} "
              f"KS D={ks.statistic:.3f} p={ks.pvalue:.4f}")
    oyr = np.abs(ret.obj_yaw); oyj = np.abs(rej.obj_yaw)
    ks = stats.ks_2samp(oyr, oyj)
    print(f"  |obj_yaw|: retained mean {oyr.mean():.4f} vs rejected {oyj.mean():.4f} "
          f"KS D={ks.statistic:.3f} p={ks.pvalue:.4f}")
    ret.to_csv(OUT / "t2_l2_demo_cabinet_init.csv", index=False)

    # nested-subset pose coverage per N, in lerobot shuffle order, for both levels
    print("\n  nested-subset demo pose coverage (lerobot shuffle order):")
    for lvl in ("L1", "L2"):
        man = json.loads((MAIN / f"data/lerobot/T2_{lvl}/conversion_manifest.json").read_text())
        order = [e["demo"] for e in sorted(man["episode_order"], key=lambda e: e["episode_index"])]
        pos = cab_init(MAIN / f"data/hdf5/T2_{lvl}.hdf5").set_index("demo").loc[order].reset_index()
        for n in (10, 25, 50):
            p = pos.iloc[:n]
            print(f"  T2_{lvl} first {n:>3}: |obj_yaw| mean {np.abs(p.obj_yaw).mean():.3f} "
                  f"max {np.abs(p.obj_yaw).max():.3f}; obj_y [{p.obj_y.min():.3f},{p.obj_y.max():.3f}]"
                  + (f"; cab_x [{p.cab_x.min():.3f},{p.cab_x.max():.3f}]" if lvl == "L2" else ""))

    print()
    print("=" * 88)
    print("SECTION 6 (Q2): horizon headroom")
    for lvl in ("L0", "L1", "L2"):
        h = h5py.File(MAIN / f"data/hdf5/T2_{lvl}.hdf5", "r")
        lens = np.array([h[f"data/{d}"].attrs["num_samples"] for d in h["data"]])
        h.close()
        print(f"T2_{lvl} demos: mean {lens.mean():.1f} max {int(lens.max())} p95 "
              f"{np.quantile(lens,0.95):.0f} -> headroom vs 1200-step eval horizon: "
              f"min {1200-int(lens.max())} steps ({(1200-lens.max())/lens.max()*100:.0f}% of max demo), "
              f"mean {1200-lens.mean():.0f}")

    print("\nepochs arithmetic check (80k steps x batch 64 = 5.12M samples):")
    for lvl in ("L0", "L1", "L2"):
        info = json.loads((MAIN / f"data/lerobot/T2_{lvl}/meta/info.json").read_text())
        print(f"  T2_{lvl}: total_frames={info['total_frames']} -> "
              f"epochs at N=400: {80000*64/info['total_frames']:.2f}")
    info1 = json.loads((MAIN / "data/lerobot/L1/meta/info.json").read_text())
    print(f"  T1_L1: total_frames={info1['total_frames']} -> {80000*64/info1['total_frames']:.2f}")


if __name__ == "__main__":
    main()
