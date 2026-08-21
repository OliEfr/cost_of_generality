"""Close the generation-bias loop: does the generator's pose filter SHAPE eval SR?

Chain under test, per cell (task x level):
  generator rejects attempts in region R  ->  demos under-cover R  ->  policy fails in R
  ->  measured SR partly reflects the generator's filter, not the task/policy.

gen_bias.py established step 1 (retained-vs-rejected KS per dim). This script adds the
missing steps: it joins the frozen eval-set initial states (configs/eval_sets/*.json,
world coords -> env-local via per-env origins) to per-episode outcomes
(results/eval_*_080000.json) and tests whether success depends on position in
initial-state space, and specifically whether it is higher where generation succeeded.

Measures per cell (N=400 primary, N=100 secondary):
  * per randomized dim: Mann-Whitney success-vs-failure states (rank-biserial r), KS D;
  * kNN local rejection rate: for each eval episode, fraction of rejected attempts among
    its k=25 nearest generation attempts (retained+rejected pooled, z-scored
    demo-observable dims) -- the direct, geometry-free "gen-hard here" score;
    tested against success (MW + rank-biserial), plus a median-split SR contrast;
  * nearest-demo distance (to the ACTUAL N-demo training subset from the conversion
    manifest) vs success;
  * logistic regression of success on z-scored randomized dims (IRLS, Wald p);
  * L3b only: per-variant gen SR vs per-variant eval SR (Spearman over 10 variants).

Frames: demos/rejects are env-local; eval sets are world. Env origins recovered from
the level with a fixed anchor (T1: L1 goal @ (0.50,0.25); T2: L1 cabinet @ (0.9,0.0);
T3: L0 puck @ (0.42,-0.10)) and validated against the declared randomization ranges.

Data gaps (stated, not papered over): T1 demos record no goal pose and T2 demos no
cabinet pose, so those dims are eval-side-only (success-vs-state testable, retained-vs-
rejected not). T3 demos DO record target_pos, so bearing is auditable end-to-end.

Run with the cog_isaac env python (needs h5py + scipy + pandas):
  /home/admin_07/miniconda3/envs/cog_isaac/bin/python scripts/dev/genbias_link_stats.py
"""

from __future__ import annotations

import json
import math
import pathlib

import h5py
import numpy as np
import pandas as pd
from scipy import stats

MAIN = pathlib.Path("/home/admin_07/cost_of_generality")
WT = MAIN / ".claude/worktrees/results-analysis"
HDF5 = MAIN / "data/hdf5"
EVAL_SETS = MAIN / "configs/eval_sets"
LEROBOT = MAIN / "data/lerobot"
RESULTS = WT / "results"          # full local eval results live in this worktree
OUT_EXP = WT / "experiments"

K_NN = 25          # neighbours for the local rejection-rate score
N_PRIMARY = 400
N_SECONDARY = 100

# ---------------------------------------------------------------- geometry helpers


def yaw_of(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- task definitions

TASKS = {
    "T1": {
        "kind": "cup_place",
        "eval_stem": lambda lv: lv,                       # L0.json etc.
        "hdf5_stem": lambda lv: lv,                       # L0.hdf5, L3bv##.hdf5
        "lerobot": lambda lv: lv,
        "pos_key": "cup_pos", "quat_key": "cup_quat",
        "goal_key": "goal_pos",
        "anchor": ("L1", "goal_pos", (0.50, 0.25)),       # fixed goal in L0/L1
        "dims": {
            "L1": ["obj_x", "obj_y", "obj_yaw"],
            "L2": ["obj_x", "obj_y", "obj_yaw", "goal_x", "goal_y"],
            "L3b": ["obj_x", "obj_y", "obj_yaw", "goal_x", "goal_y"],
        },
        "demo_dims": ["obj_x", "obj_y", "obj_yaw"],
        "ranges": {"obj_x": (0.35, 0.65), "obj_y": (-0.25, 0.15),
                   "obj_yaw": (-1.57, 1.57), "goal_x": (0.40, 0.60),
                   "goal_y": (0.10, 0.30)},
    },
    "T2": {
        "kind": "drawer_stow",
        "eval_stem": lambda lv: f"T2_{lv}",
        "hdf5_stem": lambda lv: f"T2_{lv}",
        "lerobot": lambda lv: f"T2_{lv}",
        "pos_key": "object_pos", "quat_key": "object_quat",
        "goal_key": None,
        "anchor": ("L1", "cabinet_pos", (0.9, 0.0)),
        "dims": {
            "L1": ["obj_x", "obj_y", "obj_yaw"],
            "L2": ["obj_x", "obj_y", "obj_yaw", "cab_x", "cab_y", "cab_dyaw"],
            "L3b": ["obj_x", "obj_y", "obj_yaw", "cab_x", "cab_y", "cab_dyaw"],
        },
        "demo_dims": ["obj_x", "obj_y", "obj_yaw"],
        "ranges": {"obj_x": (0.16, 0.26), "obj_y": (0.36, 0.54),
                   "obj_yaw": (-0.785, 0.785), "cab_x": (0.85, 0.95),
                   "cab_y": (-0.06, 0.06), "cab_dyaw": (-0.13, 0.13)},
    },
    "T3": {
        "kind": "push_target",
        "eval_stem": lambda lv: f"T3_{lv}",
        "hdf5_stem": lambda lv: f"T3_{lv}",
        "lerobot": lambda lv: f"T3_{lv}",
        "pos_key": "object_pos", "quat_key": "object_quat",
        "goal_key": "target_pos",
        "anchor": ("L0", "object_pos", (0.42, -0.10)),
        "dims": {
            "L1": ["obj_x", "obj_y"],
            "L2": ["obj_x", "obj_y", "bearing"],
            "L3b": ["obj_x", "obj_y", "bearing"],
        },
        # bearing is frame-independent and recorded in demos AND rejected attempts
        "demo_dims": ["obj_x", "obj_y", "bearing"],
        "ranges": {"obj_x": (0.36, 0.48), "obj_y": (-0.16, -0.04),
                   "bearing": (math.pi / 2 - 0.44, math.pi / 2 + 0.44)},
    },
}
LEVELS = ["L0", "L1", "L2", "L3b"]

# ---------------------------------------------------------------- demo/reject loading


def demo_states(task: str, path: pathlib.Path) -> pd.DataFrame | None:
    """Frame-0 initial state per demo: obj x/y/yaw (+ bearing for T3). Env-local."""
    cfg = TASKS[task]
    if not path.exists():
        return None
    rows = []
    with h5py.File(path, "r") as f:
        if "data" not in f:
            return None
        for demo in f["data"].keys():
            obs = f[f"data/{demo}/obs"]
            if cfg["pos_key"] not in obs:
                continue
            p = np.asarray(obs[cfg["pos_key"]][0])
            q = np.asarray(obs[cfg["quat_key"]][0])
            row = {"demo": demo, "obj_x": p[0], "obj_y": p[1],
                   "obj_yaw": float(yaw_of(q[None])[0])}
            if task == "T3" and "target_pos" in obs:
                t = np.asarray(obs["target_pos"][0])
                row["bearing"] = math.atan2(t[1] - p[1], t[0] - p[0])
            rows.append(row)
    return pd.DataFrame(rows) if rows else None


def gen_attempts(task: str, level: str) -> pd.DataFrame:
    """All generation attempts (retained + rejected) for one cell, env-local dims.

    L3b pools the ten per-variant files; `variant` records which one.
    """
    cfg = TASKS[task]
    stem = cfg["hdf5_stem"](level)
    parts = []
    if level == "L3b":
        for v in range(10):
            for kept, suffix in ((True, ""), (False, "_failed")):
                df = demo_states(task, HDF5 / f"{stem}v{v:02d}{suffix}.hdf5")
                if df is not None:
                    df["retained"], df["variant"] = kept, v
                    parts.append(df)
    else:
        for kept, suffix in ((True, ""), (False, "_failed")):
            df = demo_states(task, HDF5 / f"{stem}{suffix}.hdf5")
            if df is not None:
                df["retained"], df["variant"] = kept, -1
                parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    return out


# ---------------------------------------------------------------- eval-set loading


def env_origins(task: str) -> np.ndarray:
    """(20, 2) per-env world origin, from the level whose anchor entity is fixed."""
    lv, key, local = TASKS[task]["anchor"]
    d = json.loads((EVAL_SETS / f"{TASKS[task]['eval_stem'](lv)}.json").read_text())
    a = np.array([b[key] for b in d["batches"]])[:, :, :2] - np.asarray(local)
    assert np.allclose(a, a[0], atol=1e-6), f"{task}: origins differ across batches"
    return a[0]


def eval_states(task: str, level: str, origins: np.ndarray) -> pd.DataFrame:
    """One row per (batch, env) in the frozen eval set, env-local dims."""
    cfg = TASKS[task]
    stem = cfg["eval_stem"]("L3" if level == "L3b" else level)
    d = json.loads((EVAL_SETS / f"{stem}.json").read_text())
    rows = []

    def add_batch(batch: dict, b: int, variant: int) -> None:
        pos = np.asarray(batch[cfg["pos_key"]])[:, :2] - origins
        q = np.asarray(batch[cfg["quat_key"]])
        yaw = yaw_of(q)
        for e in range(len(pos)):
            row = {"batch": b, "env": e, "variant": variant,
                   "obj_x": pos[e, 0], "obj_y": pos[e, 1], "obj_yaw": yaw[e]}
            if task == "T1":
                g = np.asarray(batch["goal_pos"])[:, :2] - origins
                row["goal_x"], row["goal_y"] = g[e, 0], g[e, 1]
            elif task == "T2":
                c = np.asarray(batch["cabinet_pos"])[:, :2] - origins
                cy = yaw_of(np.asarray(batch["cabinet_quat"]))
                row["cab_x"], row["cab_y"] = c[e, 0], c[e, 1]
                row["cab_dyaw"] = float(wrap(cy[e] - math.pi))
            elif task == "T3":
                t = np.asarray(batch[cfg["goal_key"]])[:, :2]
                pw = np.asarray(batch[cfg["pos_key"]])[:, :2]
                row["bearing"] = math.atan2(t[e, 1] - pw[e, 1], t[e, 0] - pw[e, 0])
            rows.append(row)

    if level == "L3b":
        for v in range(10):                       # diagonal: variant v uses batch v
            add_batch(d["variants"][f"L3v{v:02d}"]["batches"][v], v, v)
    else:
        for b in range(5):                        # standard eval = batches 0-4
            add_batch(d["batches"][b], b, -1)
    df = pd.DataFrame(rows)
    # frame sanity: recovered local coords must sit inside the declared ranges
    for dim in TASKS[task]["dims"].get(level, []):
        lo, hi = TASKS[task]["ranges"][dim]
        v = df[dim].to_numpy()
        assert v.min() > lo - 0.02 and v.max() < hi + 0.02, \
            f"{task} {level} {dim}: [{v.min():.3f},{v.max():.3f}] outside [{lo},{hi}]"
    return df


def outcomes(task: str, level: str, n: int) -> pd.DataFrame | None:
    p = RESULTS / f"eval_{task}_{level}_n{n}_080000.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    df = pd.DataFrame(d["outcomes"])
    if level == "L3b":
        # The eval driver logs batch=0 for every episode of a diagonal L3 eval; the
        # list is variant-major (10 blocks of 20, env 0..19 per block), verified
        # against the per_variant success counts and seeds. Recover batch = variant.
        assert len(df) == 200 and (df["batch"] == 0).all()
        df["batch"] = np.arange(len(df)) // 20
        blocks = df.groupby("batch")["success"].sum()
        pv = [d["per_variant"][f"L3v{v:02d}"]["successes"] for v in range(10)]
        assert blocks.tolist() == pv, f"{task} L3b n{n}: variant-major order violated"
    return df


def training_subset(task: str, level: str, n: int) -> pd.DataFrame | None:
    """Initial states of the N demos this cell's policy actually trained on."""
    man_p = LEROBOT / TASKS[task]["lerobot"](level) / "conversion_manifest.json"
    if not man_p.exists():
        return None
    man = json.loads(man_p.read_text())
    order = man["episode_order"][:n]
    # group by file, read each once
    cache: dict[str, pd.DataFrame] = {}
    rows = []
    for ep in order:
        f = ep["file"]
        if f not in cache:
            path = pathlib.Path(f)
            if not path.is_absolute():
                path = MAIN / f
            cache[f] = demo_states(task, path).set_index("demo")
        rows.append(cache[f].loc[ep["demo"]])
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------- statistics


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """r = 2*U/(n1*n2) - 1: P(a > b) - P(b > a). a = success states, b = failures."""
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def logit_irls(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6,
               iters: int = 100) -> tuple[np.ndarray, np.ndarray, bool]:
    """Logistic regression with intercept; returns (beta, wald_p, converged)."""
    Xd = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(Xd.shape[1])
    conv = False
    for _ in range(iters):
        eta = np.clip(Xd @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-10, None)
        H = Xd.T @ (Xd * w[:, None]) + ridge * np.eye(Xd.shape[1])
        g = Xd.T @ (y - mu) - ridge * beta
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            conv = True
            break
    eta = np.clip(Xd @ beta, -30, 30)
    mu = 1 / (1 + np.exp(-eta))
    w = np.clip(mu * (1 - mu), 1e-10, None)
    H = Xd.T @ (Xd * w[:, None]) + ridge * np.eye(Xd.shape[1])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    zs = beta / se
    p = 2 * stats.norm.sf(np.abs(zs))
    return beta[1:], p[1:], conv


def knn_local_reject(eval_pts: np.ndarray, att_pts: np.ndarray,
                     att_retained: np.ndarray, k: int = K_NN) -> np.ndarray:
    """Fraction of rejected attempts among each eval point's k nearest attempts."""
    d2 = ((eval_pts[:, None, :] - att_pts[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d2, axis=1)[:, :k]
    return 1.0 - att_retained[idx].mean(axis=1)


def nearest_dist(eval_pts: np.ndarray, demo_pts: np.ndarray) -> np.ndarray:
    d2 = ((eval_pts[:, None, :] - demo_pts[None, :, :]) ** 2).sum(-1)
    return np.sqrt(d2.min(axis=1))


# ---------------------------------------------------------------- per-cell analysis


def analyse_cell(task: str, level: str) -> tuple[list[dict], list[dict], pd.DataFrame | None]:
    """Returns (tidy CSV rows, cell summary dicts per N, joined episode df)."""
    cfg = TASKS[task]
    att = gen_attempts(task, level)
    n_ret = int(att["retained"].sum())
    n_rej = int((~att["retained"]).sum())
    rej_rate = n_rej / max(n_ret + n_rej, 1)

    # unique poses (dedup: D27 -- duplicated pose streams carry no new information)
    demo_dims = [d for d in cfg["demo_dims"] if d in att.columns
                 and d in cfg["dims"].get(level, [])]
    csv_rows: list[dict] = []
    summaries: list[dict] = []

    if level == "L0":
        for n in (N_PRIMARY, N_SECONDARY):
            oc = outcomes(task, level, n)
            sr = float(oc["success"].mean()) if oc is not None else float("nan")
            summaries.append({
                "task": task, "level": level, "n": n, "sr": sr,
                "episodes": len(oc) if oc is not None else 0,
                "rejection_rate": rej_rate, "n_ret": n_ret, "n_rej": n_rej,
                "verdict": "exempt",
                "note": "single fixed initial state: rejection is pose-independent "
                        "controller/sim stochasticity; no state space to filter",
            })
        csv_rows.append({
            "task": task, "level": level, "dim": "(none)", "n_eval": N_PRIMARY,
            "rejection_rate": round(rej_rate, 4),
            "ks_retained_rejected_D": np.nan, "ks_retained_rejected_p": np.nan,
            "succ_vs_fail_effect": np.nan, "succ_vs_fail_p": np.nan,
            "verdict": "exempt (fixed pose)",
        })
        return csv_rows, summaries, None

    uatt = att.drop_duplicates(subset=demo_dims + ["retained"])
    ur = uatt[uatt["retained"]]
    uf = uatt[~uatt["retained"]]

    # ---- step 1: generator filter per demo-observable dim
    ks_gen: dict[str, tuple[float, float]] = {}
    for dim in demo_dims:
        if len(uf) >= 2:
            r = stats.ks_2samp(ur[dim], uf[dim])
            ks_gen[dim] = (float(r.statistic), float(r.pvalue))
        else:
            ks_gen[dim] = (float("nan"), float("nan"))

    origins = env_origins(task)
    ev = eval_states(task, level, origins)
    dims = cfg["dims"][level]

    joined_all = None
    for n in (N_PRIMARY, N_SECONDARY):
        oc = outcomes(task, level, n)
        if oc is None:
            continue
        j = oc.merge(ev, on=["batch", "env"], how="inner")
        assert len(j) == len(oc), f"{task} {level} n{n}: join lost episodes"
        succ = j["success"].to_numpy(bool)
        n_s, n_f = int(succ.sum()), int((~succ).sum())
        sr = n_s / len(j)

        # ---- kNN local rejection rate (demo-observable dims only)
        att_pts = uatt[demo_dims].to_numpy(float)
        mu, sd = att_pts.mean(0), att_pts.std(0)
        sd[sd < 1e-9] = 1.0
        z_att = (att_pts - mu) / sd
        z_ev = (j[demo_dims].to_numpy(float) - mu) / sd
        j["local_rej"] = knn_local_reject(z_ev, z_att, uatt["retained"].to_numpy(float))

        # ---- nearest-demo distance (the N demos actually trained on)
        sub = training_subset(task, level, n)
        if sub is not None:
            sub_pts = (sub[demo_dims].to_numpy(float) - mu) / sd
            j["nn_demo_dist"] = nearest_dist(z_ev, sub_pts)
        else:
            j["nn_demo_dist"] = np.nan

        power_limited = min(n_s, n_f) < 10
        cell = {
            "task": task, "level": level, "n": n, "sr": round(sr, 3),
            "episodes": len(j), "n_fail": n_f,
            "rejection_rate": round(rej_rate, 4), "n_ret": n_ret, "n_rej": n_rej,
            "power_limited": power_limited,
        }

        # ---- per-dim success-vs-failure tests
        per_dim = {}
        for dim in dims:
            a, b = j.loc[succ, dim].to_numpy(), j.loc[~succ, dim].to_numpy()
            if n_s >= 2 and n_f >= 2:
                mw_p = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
                rb = rank_biserial(a, b)
                ks = stats.ks_2samp(a, b)
                ks_d, ks_p = float(ks.statistic), float(ks.pvalue)
            else:
                mw_p = rb = ks_d = ks_p = float("nan")
            per_dim[dim] = {"rb": rb, "mw_p": mw_p, "ks_d": ks_d, "ks_p": ks_p}
            if n == N_PRIMARY:
                gd, gp = ks_gen.get(dim, (float("nan"), float("nan")))
                csv_rows.append({
                    "task": task, "level": level, "dim": dim, "n_eval": n,
                    "rejection_rate": round(rej_rate, 4),
                    "ks_retained_rejected_D": round(gd, 4) if gd == gd else np.nan,
                    "ks_retained_rejected_p": round(gp, 5) if gp == gp else np.nan,
                    "succ_vs_fail_effect": round(rb, 4) if rb == rb else np.nan,
                    "succ_vs_fail_p": round(mw_p, 5) if mw_p == mw_p else np.nan,
                })
        cell["per_dim"] = per_dim

        # ---- composite: local rejection rate vs success
        lr = j["local_rej"].to_numpy()
        if n_s >= 2 and n_f >= 2 and np.ptp(lr) > 0:
            cell["local_rej_rb"] = rank_biserial(lr[succ], lr[~succ])
            cell["local_rej_p"] = float(
                stats.mannwhitneyu(lr[succ], lr[~succ], alternative="two-sided").pvalue)
            med = np.median(lr)
            hard = lr > med
            if 0.2 < hard.mean() < 0.8:
                cell["sr_gen_easy"] = round(float(succ[~hard].mean()), 3)
                cell["sr_gen_hard"] = round(float(succ[hard].mean()), 3)
                cell["split_fisher_p"] = float(stats.fisher_exact(
                    [[int((hard & succ).sum()), int((hard & ~succ).sum())],
                     [int((~hard & succ).sum()), int((~hard & ~succ).sum())]])[1])
            cell["local_rej_spearman"] = float(stats.spearmanr(lr, succ).statistic)
        else:
            cell["local_rej_rb"] = cell["local_rej_p"] = float("nan")

        # ---- composite: nearest-demo distance vs success
        nd = j["nn_demo_dist"].to_numpy()
        if n_s >= 2 and n_f >= 2 and np.isfinite(nd).all():
            cell["nnd_rb"] = rank_biserial(nd[succ], nd[~succ])
            cell["nnd_p"] = float(
                stats.mannwhitneyu(nd[succ], nd[~succ], alternative="two-sided").pvalue)
        else:
            cell["nnd_rb"] = cell["nnd_p"] = float("nan")

        # ---- logistic regression on z-scored randomized dims
        if n_s >= 5 and n_f >= 5:
            X = j[dims].to_numpy(float)
            Xz = (X - X.mean(0)) / np.where(X.std(0) < 1e-9, 1.0, X.std(0))
            beta, pvals, conv = logit_irls(Xz, succ.astype(float))
            cell["logit"] = {d: {"beta": round(float(b), 3), "p": round(float(pv), 4)}
                             for d, b, pv in zip(dims, beta, pvals)}
            cell["logit_converged"] = conv
        # ---- L3b: variant-level gen SR vs eval SR
        if level == "L3b":
            vg = att.groupby("variant")["retained"].mean()
            ve = j.groupby("variant")["success"].mean()
            both = pd.concat([vg.rename("gen_sr"), ve.rename("eval_sr")],
                             axis=1).dropna()
            if len(both) >= 5 and both["gen_sr"].std() > 0 and both["eval_sr"].std() > 0:
                sp = stats.spearmanr(both["gen_sr"], both["eval_sr"])
                cell["variant_gen_eval_spearman"] = round(float(sp.statistic), 3)
                cell["variant_gen_eval_p"] = round(float(sp.pvalue), 4)
                cell["variant_gen_sr"] = {int(k): round(float(v), 3)
                                          for k, v in vg.items()}
                cell["variant_eval_sr"] = {int(k): round(float(v), 3)
                                           for k, v in ve.items()}
        summaries.append(cell)
        j.insert(0, "task", task)
        j.insert(1, "level", level)
        j.insert(2, "n_demos", n)
        joined_all = j if joined_all is None else pd.concat([joined_all, j])

        if n == N_PRIMARY:
            csv_rows.append({
                "task": task, "level": level, "dim": "local_rej_knn", "n_eval": n,
                "rejection_rate": round(rej_rate, 4),
                "ks_retained_rejected_D": np.nan, "ks_retained_rejected_p": np.nan,
                "succ_vs_fail_effect": round(cell["local_rej_rb"], 4)
                    if cell["local_rej_rb"] == cell["local_rej_rb"] else np.nan,
                "succ_vs_fail_p": round(cell["local_rej_p"], 5)
                    if cell["local_rej_p"] == cell["local_rej_p"] else np.nan,
            })
            csv_rows.append({
                "task": task, "level": level, "dim": "nearest_demo_dist", "n_eval": n,
                "rejection_rate": round(rej_rate, 4),
                "ks_retained_rejected_D": np.nan, "ks_retained_rejected_p": np.nan,
                "succ_vs_fail_effect": round(cell["nnd_rb"], 4)
                    if cell["nnd_rb"] == cell["nnd_rb"] else np.nan,
                "succ_vs_fail_p": round(cell["nnd_p"], 5)
                    if cell["nnd_p"] == cell["nnd_p"] else np.nan,
            })
    return csv_rows, summaries, joined_all


def main() -> None:
    all_rows, all_cells, all_eps = [], [], []
    for task in ("T1", "T2", "T3"):
        for level in LEVELS:
            rows, cells, eps = analyse_cell(task, level)
            all_rows += rows
            all_cells += cells
            if eps is not None:
                all_eps.append(eps)
            for c in cells:
                if c.get("n") in (N_PRIMARY,) or c.get("level") == "L0":
                    print(f"{task} {level} n{c['n']}: SR={c.get('sr')} "
                          f"rej={c['rejection_rate']:.2%} "
                          f"local_rej_rb={c.get('local_rej_rb', float('nan'))} "
                          f"p={c.get('local_rej_p', float('nan'))}")
    OUT_EXP.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(OUT_EXP / "genbias_link.csv", index=False)
    ep = pd.concat(all_eps, ignore_index=True)
    ep.to_csv(OUT_EXP / "genbias_link_episodes.csv", index=False)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o
    (OUT_EXP / "genbias_link_cells.json").write_text(
        json.dumps(clean(all_cells), indent=1))
    print(f"\nwrote {OUT_EXP/'genbias_link.csv'} ({len(all_rows)} rows), "
          f"episodes csv ({len(ep)} rows), cells json ({len(all_cells)} cells)")


if __name__ == "__main__":
    main()
