"""Is the demo-generation pipeline a SELECTION FILTER on the training distribution?

The concern (raised 2026-08-20): Isaac Lab Mimic keeps only the attempts it judges successful, so
the retained demos are a *conditioned* sample -- p(pose | generator succeeded) -- while the frozen
eval sets sample p(pose) unconditionally. If the generator fails preferentially in some region of
pose space, the policy is trained on a narrower distribution than it is tested on, and the resulting
low success rate would look like a policy/generality limit when it is really a data-pipeline
artifact. That confound has to be measured, not assumed away.

It is measurable here only because `generate_level.py` writes the rejected attempts to a parallel
`<level>_failed.hdf5` instead of discarding them. This script compares the two populations:

  * per-dimension summary of the initial object pose (x, y, yaw) for retained vs rejected attempts;
  * a two-sample Kolmogorov-Smirnov test per dimension (implemented here -- no scipy dependency);
  * the span of the declared randomization range that each population actually covers;
  * a 3x3 spatial occupancy table, because a KS test on marginals can miss a corner-shaped hole.

Reading the output: a large KS D with a small p-value on any dimension means generation success is
pose-dependent, so the training distribution is skewed relative to eval. Small D everywhere means
the generator's failures are spread across the space and the retained demos are an unbiased
subsample -- in which case a low eval SR cannot be blamed on the generator.

IMPORTANT interpretive bound: even a perfectly biased filter can only remove the fraction it
rejects. At 88% generation SR the missing mass is 12%, so generation bias cannot account for an eval
deficit larger than ~12 points no matter how it is distributed. Check gen_sr before reaching for
this explanation at all.

usage:
  python -m cog.analysis.gen_bias                    # every level of every task found on disk
  python -m cog.analysis.gen_bias --levels T2_L2     # one level
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re

import h5py
import numpy as np

HDF5_DIR = pathlib.Path("/home/admin_07/cost_of_generality/data/hdf5")

# The object-pose key differs by task: cup_place records cup_pos/cup_quat, the other two
# object_pos/object_quat. Note NEITHER cup_place nor drawer_stow records the GOAL pose, so goal
# coverage is not auditable from these files -- only the object pose is.
POS_KEYS = ("cup_pos", "object_pos")
QUAT_KEYS = ("cup_quat", "object_quat")


def _yaw(quat: np.ndarray) -> np.ndarray:
    """Yaw from (w,x,y,z), the Isaac Lab convention."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class FileLockedError(RuntimeError):
    """The HDF5 file is held by a writer -- i.e. generation is still running on this level."""


def initial_poses(path: pathlib.Path) -> np.ndarray | None:
    """(N, 3) array of initial [x, y, yaw], one row per demo in the file.

    Raises FileLockedError if a generation job currently holds the file. Reading a half-written
    HDF5 could return torn data, and silently skipping the file instead would understate the
    level's demo count -- so the caller drops the whole level rather than reporting a partial one.
    """
    rows = []
    try:
        h5py.File(path, "r").close()
    except BlockingIOError as e:
        raise FileLockedError(f"{path.name} is locked by a writer") from e
    with h5py.File(path, "r") as f:
        if "data" not in f:
            return None
        for demo in f["data"].keys():
            obs = f[f"data/{demo}/obs"]
            pk = next((k for k in POS_KEYS if k in obs), None)
            qk = next((k for k in QUAT_KEYS if k in obs), None)
            if pk is None or qk is None:
                continue
            pos = np.asarray(obs[pk][0:1])          # frame 0 = the sampled initial state
            quat = np.asarray(obs[qk][0:1])
            rows.append([float(pos[0, 0]), float(pos[0, 1]), float(_yaw(quat)[0])])
    return np.asarray(rows) if rows else None


def ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample KS statistic and asymptotic p-value.

    Hand-rolled to keep this runnable in the Isaac env (no scipy). The asymptotic p is adequate
    here: we only need to distinguish "obviously skewed" from "indistinguishable", and both samples
    are >= ~40 wherever we apply it.
    """
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    grid = np.sort(np.concatenate([a, b]))
    fa = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    fb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    d = float(np.max(np.abs(fa - fb)))
    # D == 0 means the two ECDFs are identical -- p must be 1, but the asymptotic series evaluates
    # to ~0 there (it is an alternating series that only converges for lam > 0). Without this guard a
    # dimension that is CONSTANT in both populations (push_target never randomizes yaw) is reported
    # as the most significant skew in the table. Caught on T3_L2, where yaw is 0.000 either way.
    if d == 0.0:
        return 0.0, 1.0
    en = math.sqrt(len(a) * len(b) / (len(a) + len(b)))
    lam = (en + 0.12 + 0.11 / en) * d
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return d, min(1.0, max(0.0, p))


def occupancy(succ: np.ndarray, fail: np.ndarray) -> str:
    """3x3 occupancy over the (x, y) plane: % of each population per cell.

    A KS test on marginals can pass while a whole corner of the workspace is missing, so compare
    the joint distribution too, coarsely enough to stay legible at n=40.
    """
    lo = np.minimum(succ[:, :2].min(0), fail[:, :2].min(0))
    hi = np.maximum(succ[:, :2].max(0), fail[:, :2].max(0))
    span = np.where(hi - lo > 0, hi - lo, 1.0)

    def grid(a: np.ndarray) -> np.ndarray:
        idx = np.clip(((a[:, :2] - lo) / span * 3).astype(int), 0, 2)
        g = np.zeros((3, 3))
        for i, j in idx:
            g[j, i] += 1                      # rows = y bands (high y first below), cols = x bands
        return g / max(len(a), 1) * 100

    gs, gf = grid(succ), grid(fail)
    out = ["    x-> low    mid    high        (cell = % of that population)",
           "         retained            rejected"]
    for r in (2, 1, 0):
        out.append("  " + " ".join(f"{gs[r, c]:5.1f}" for c in range(3))
                   + "     " + " ".join(f"{gf[r, c]:5.1f}" for c in range(3)))
    return "\n".join(out)


def _pool(paths: list[pathlib.Path]) -> np.ndarray | None:
    parts = [p for path in paths if (p := initial_poses(path)) is not None]
    return np.concatenate(parts) if parts else None


DIMS = ("x", "y", "yaw")


def _uniq(a: np.ndarray) -> np.ndarray:
    return np.unique(np.round(a, 5), axis=0)


def level_stats(level: str) -> dict | None:
    """Every number for one level, computed once and returned as data -- no printing.

    Both the CLI table and gen_bias_plots.py consume this, so the figures in the paper cannot drift
    from the numbers on the terminal.

    POSE REDUNDANCY (D27) is handled here. An L3 level is generated by one run per variant, and if
    those runs share a seed they replay one pose stream, so N demos can carry far fewer than N
    distinct initial states. Duplicated rows carry no extra information about the pose distribution,
    so the KS test runs on DEDUPLICATED poses -- using raw rows inflates n ~9x and turns a difference
    of no consequence into p<0.001, which this tool did before the fix.
    """
    try:
        return _level_stats_inner(level)
    except FileLockedError as e:
        print(f"  SKIP {level}: {e} (generation in progress) -- reporting no level rather than a "
              f"partial one")
        return None


def _level_stats_inner(level: str) -> dict | None:
    succ_p = HDF5_DIR / f"{level}.hdf5"
    pooled = 0
    if not succ_p.exists():
        # An L3 level has no single file: it is realized as 10 variant sub-levels whose datasets are
        # merged, and the study's distribution semantics are "uniform over variants x poses" (see
        # levels.py). So pool the variants -- per variant there are only ~5 rejected attempts, far
        # too few for a KS test, while pooled there are ~50-830.
        variants = sorted(HDF5_DIR.glob(f"{level}v[0-9][0-9].hdf5"))
        if not variants:
            return None
        succ = _pool(variants)
        fail = _pool(sorted(HDF5_DIR.glob(f"{level}v[0-9][0-9]_failed.hdf5")))
        pooled = len(variants)
    else:
        succ = initial_poses(succ_p)
        fp = HDF5_DIR / f"{level}_failed.hdf5"
        fail = initial_poses(fp) if fp.exists() else None
    if succ is None:
        return None

    n_s = len(succ)
    n_f = len(fail) if fail is not None else 0
    u_s = _uniq(succ)
    u_f = _uniq(fail) if n_f else np.empty((0, 3))
    st = {
        "level": level, "pooled_files": pooled,
        "n_succ": n_s, "n_fail": n_f, "attempts": n_s + n_f,
        "gen_sr": 100 * n_s / max(n_s + n_f, 1),
        "unique_succ": len(u_s), "unique_fail": len(u_f),
        "redundancy": n_s / max(len(u_s), 1),
        # The bound that matters: a selection filter can only remove what it rejects, so this caps
        # how much of any eval deficit generation bias could possibly explain.
        "max_attributable_pts": 100 * n_f / max(n_s + n_f, 1),
        "succ_poses": u_s, "fail_poses": u_f,
        "ks": {}, "skewed": [],
    }
    for i, name in enumerate(DIMS):
        if n_f:
            d, p = ks_2samp(u_s[:, i], u_f[:, i])
        else:
            d, p = float("nan"), float("nan")
        st["ks"][name] = {
            "d": d, "p": p,
            "succ_mean": float(u_s[:, i].mean()), "succ_sd": float(u_s[:, i].std()),
            "fail_mean": float(u_f[:, i].mean()) if n_f else float("nan"),
            "fail_sd": float(u_f[:, i].std()) if n_f else float("nan"),
        }
        if n_f and not math.isnan(p) and p < 0.05:
            st["skewed"].append(name)
    st["worst_ks"] = max((v["d"] for v in st["ks"].values()
                          if not math.isnan(v["d"])), default=0.0)
    return st


def analyse(level: str) -> dict | None:
    st = level_stats(level)
    if st is None:
        return None
    print(f"\n=== {level} ===")
    if st["pooled_files"]:
        print(f"  (pooled {st['pooled_files']} variant files)")
    print(f"retained {st['n_succ']}   rejected {st['n_fail']}   attempts {st['attempts']}   "
          f"gen_SR {st['gen_sr']:.1f}%")
    print(f"  UNIQUE initial poses: retained {st['unique_succ']}/{st['n_succ']}"
          + (f"   rejected {st['unique_fail']}/{st['n_fail']}" if st["n_fail"] else ""))
    if st["n_succ"] > 1 and st["redundancy"] > 1.11:
        print(f"  !! POSE REDUNDANCY {st['redundancy']:.1f}x -- this level's demo-count axis "
              f"is inflated: adding demos adds appearances, not new initial states. Expected only "
              f"for L0 (fixed by design); anywhere else it is the seeding bug of D27.")
    if not st["n_fail"]:
        print("  no rejected attempts on disk -> selection bias not measurable (and, at this "
              "generation SR, not material)")
        return st

    succ, fail = st["succ_poses"], st["fail_poses"]
    print(f"  {'dim':5} {'retained mean+-sd':>22} {'rejected mean+-sd':>22} "
          f"{'KS D':>7} {'p':>8}  verdict     (stats over UNIQUE poses)")
    for name in DIMS:
        k = st["ks"][name]
        flag = "SKEWED" if name in st["skewed"] else "same"
        print(f"  {name:5} {k['succ_mean']:>10.3f}+-{k['succ_sd']:<10.3f} "
              f"{k['fail_mean']:>10.3f}+-{k['fail_sd']:<10.3f} {k['d']:>7.3f} {k['p']:>8.3f}  {flag}")
    print(f"  range covered: retained x[{succ[:, 0].min():.3f},{succ[:, 0].max():.3f}] "
          f"y[{succ[:, 1].min():.3f},{succ[:, 1].max():.3f}] "
          f"yaw[{succ[:, 2].min():.2f},{succ[:, 2].max():.2f}]")
    print(f"                 rejected x[{fail[:, 0].min():.3f},{fail[:, 0].max():.3f}] "
          f"y[{fail[:, 1].min():.3f},{fail[:, 1].max():.3f}] "
          f"yaw[{fail[:, 2].min():.2f},{fail[:, 2].max():.2f}]")
    print(occupancy(succ, fail))
    print(f"  => max eval deficit attributable to generation bias: "
          f"{st['max_attributable_pts']:.1f} points (the rejected fraction)")
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="*", default=None,
                    help="level stems as they appear in data/hdf5 (e.g. L2, T2_L3v00). "
                         "Default: every level with a *_failed.hdf5 sibling.")
    args = ap.parse_args()

    if args.levels:
        levels = args.levels
    else:
        pat = re.compile(r"^((?:T[23]_)?L[0-3](?:v\d\d)?)\.hdf5$")
        levels = sorted(m.group(1) for p in HDF5_DIR.glob("*.hdf5")
                        if (m := pat.match(p.name)))
    rows = [r for lv in levels if (r := analyse(lv))]
    print(f"\n{len(rows)} level(s) analysed")


if __name__ == "__main__":
    main()
