#!/usr/bin/env python3
"""Compare two Mimic-generated HDF5 datasets demo-for-demo: gate G2, cluster vs workstation.

Why this exists. The reverse-ablation arms are generated on Leonardo's A100s but are compared
against L2 and L3b, whose demos were generated on the workstation's RTX 4090. If the two machines
produce different demonstrations -- different trajectories, or merely different pixels -- then a
success-rate difference between arms confounds "which disturbance was removed" with "which GPU
rendered the training images". That is structurally the same class of artefact as D27's
pose-redundancy bug, and it is worth one 30-minute job to rule out.

Run the SAME leg on both machines (same env key, same seed, same --num_envs; --num_envs is part of
the experiment because attempts are distributed across the async env pool, so changing it changes
which pose each attempt gets) and point this at the two files.

The checks are ordered by how diagnostic they are, and the first three are hard stops: if the
attempt counts or the sample counts differ, the physics did not reproduce and the pixels do not
matter yet.

usage: parity_check.py LOCAL.hdf5 CLUSTER.hdf5 [--demos 3] [--frames 3]
"""

from __future__ import annotations

import argparse
import sys

import h5py
import numpy as np

# Tolerances, loosest-justifiable rather than tightest-passing:
ATOL_INIT = 1e-6     # initial state: same RNG stream or nothing
ATOL_ACTION = 1e-4   # Mimic's transformed plan
ATOL_STATE = 1e-3    # 1 mm: PhysX GPU reduction order may differ by architecture
# Pixel verdict bands, in 0-255 units.
MAE_AGREE, P99_AGREE = 1.0, 8.0
MAE_TOLERABLE, CHANNEL_SHIFT_MAX = 3.0, 2.0


def leaves(group, prefix=""):
    for name, item in group.items():
        path = f"{prefix}{name}"
        if isinstance(item, h5py.Group):
            yield from leaves(item, path + "/")
        else:
            yield path, item


def compare_block(a_demo, b_demo, block: str, atol: float, label: str, problems: list) -> float:
    """Max |delta| over every leaf under `block`, or -1.0 when the block is absent."""
    if block not in a_demo or block not in b_demo:
        return -1.0
    a_leaves = dict(leaves(a_demo[block]))
    b_leaves = dict(leaves(b_demo[block]))
    if a_leaves.keys() != b_leaves.keys():
        problems.append(f"{label}: different leaf sets "
                        f"({sorted(set(a_leaves) ^ set(b_leaves))})")
        return -1.0
    worst = 0.0
    for key in a_leaves:
        x, y = np.asarray(a_leaves[key]), np.asarray(b_leaves[key])
        if x.shape != y.shape:
            problems.append(f"{label}/{key}: shape {x.shape} vs {y.shape}")
            continue
        worst = max(worst, float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()))
    if worst > atol:
        problems.append(f"{label}: max|delta| {worst:.3e} exceeds atol {atol:.0e}")
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("local")
    ap.add_argument("cluster")
    ap.add_argument("--demos", type=int, default=3, help="demos sampled for the pixel comparison")
    ap.add_argument("--frames", type=int, default=3, help="frames per sampled demo")
    args = ap.parse_args()

    fa, fb = h5py.File(args.local, "r"), h5py.File(args.cluster, "r")
    da, db = fa["data"], fb["data"]
    hard: list[str] = []
    soft: list[str] = []

    # 1. total recorded timesteps. Verified against L3bv00: data.attrs["total"] == sum of every
    # demo's num_samples (7511), NOT the attempt count -- the generator's accept/reject tally only
    # reaches the log. One scalar that changes if any trajectory anywhere changed length.
    ta, tb = da.attrs.get("total"), db.attrs.get("total")
    print(f"1. total timesteps local={ta}  cluster={tb}")
    if ta != tb:
        hard.append(f"total timesteps differ ({ta} vs {tb}): the trajectories are not reproducing")

    # 2. demo count and per-demo length -- the most sensitive scalars available.
    ka, kb = sorted(da.keys()), sorted(db.keys())
    print(f"2. demos           local={len(ka)}  cluster={len(kb)}")
    if len(ka) != len(kb):
        hard.append(f"demo counts differ ({len(ka)} vs {len(kb)})")
    common = [k for k in ka if k in kb]
    lens_a = [int(da[k].attrs["num_samples"]) for k in common]
    lens_b = [int(db[k].attrs["num_samples"]) for k in common]
    bad_len = [(k, x, y) for k, x, y in zip(common, lens_a, lens_b) if x != y]
    print(f"   num_samples     {len(common) - len(bad_len)}/{len(common)} identical")
    if bad_len:
        hard.append(f"{len(bad_len)} demo(s) differ in length, e.g. {bad_len[:3]}: "
                    f"trajectories diverged")

    # 3-5. numeric blocks.
    worst = {"initial_state": 0.0, "actions": 0.0, "states": 0.0}
    for k in common:
        worst["initial_state"] = max(
            worst["initial_state"],
            compare_block(da[k], db[k], "initial_state", ATOL_INIT, f"3. initial_state[{k}]", hard))
        if "actions" in da[k] and "actions" in db[k] and da[k]["actions"].shape == db[k]["actions"].shape:
            x = np.asarray(da[k]["actions"], dtype=np.float64)
            y = np.asarray(db[k]["actions"], dtype=np.float64)
            worst["actions"] = max(worst["actions"], float(np.abs(x - y).max()))
        worst["states"] = max(
            worst["states"],
            compare_block(da[k], db[k], "states", ATOL_STATE, f"5. states[{k}]", soft))
    print(f"3. initial_state   max|delta|={worst['initial_state']:.3e}  (atol {ATOL_INIT:.0e})")
    print(f"4. actions         max|delta|={worst['actions']:.3e}  (atol {ATOL_ACTION:.0e})")
    print(f"5. states          max|delta|={worst['states']:.3e}  (atol {ATOL_STATE:.0e})")
    if worst["actions"] > ATOL_ACTION:
        soft.append(f"actions max|delta| {worst['actions']:.3e} exceeds {ATOL_ACTION:.0e}")

    # 6. pixels. Not expected to be bit-identical; the question is how far off, and whether the
    #    difference is a uniform shift (exposure/hue -> the policy sees a different world) or
    #    high-frequency sampling noise (-> it does not).
    print("6. pixels")
    verdict = "AGREE"
    if common:
        idx = sorted({int(i * (len(common) - 1) / max(args.demos - 1, 1)) for i in range(args.demos)})
        cams = [n for n, it in da[common[0]]["obs"].items()
                if it.dtype == np.uint8 and it.ndim == 4]
        for cam in sorted(cams):
            maes, p99s, shifts = [], [], []
            for i in idx:
                k = common[i]
                A, B = da[k]["obs"][cam], db[k]["obs"][cam]
                if A.shape != B.shape:
                    hard.append(f"{cam}[{k}]: shape {A.shape} vs {B.shape}")
                    continue
                T = A.shape[0]
                for t in sorted({0, T // 2, T - 1})[: args.frames]:
                    x = np.asarray(A[t], dtype=np.float64)
                    y = np.asarray(B[t], dtype=np.float64)
                    d = np.abs(x - y)
                    maes.append(d.mean())
                    p99s.append(np.percentile(d, 99))
                    shifts.append(np.abs(x.mean(axis=(0, 1)) - y.mean(axis=(0, 1))).max())
            if not maes:
                continue
            mae, p99, shift = float(np.mean(maes)), float(np.max(p99s)), float(np.max(shifts))
            print(f"   {cam:12s} MAE={mae:6.3f}  p99={p99:7.3f}  max channel-mean shift={shift:6.3f}")
            if mae > MAE_TOLERABLE or shift > CHANNEL_SHIFT_MAX:
                verdict = "DIVERGENT"
            elif (mae > MAE_AGREE or p99 > P99_AGREE) and verdict != "DIVERGENT":
                verdict = "TOLERABLE"

    fa.close()
    fb.close()

    print()
    for p in soft:
        print(f"  WARN  {p}")
    if hard:
        for p in hard:
            print(f"  STOP  {p}")
        print("PARITY_FAILED: the two machines do not generate the same demonstrations. "
              "Do not mix cluster- and workstation-generated arms in one comparison.")
        return 1
    if verdict == "AGREE":
        print("PARITY_AGREE: trajectories identical and pixels within noise -- cluster- and "
              "workstation-generated arms are freely mixable.")
        return 0
    if verdict == "TOLERABLE":
        print("PARITY_TOLERABLE: trajectories identical, pixels differ in sampling but not in "
              "level. Each arm is generated entirely on one machine, so this is usable -- but "
              "record it, and run the G2b retraining control before committing the wave.")
        return 0
    print("PARITY_DIVERGENT: the rendered images differ in level, not just in sampling. Treat the "
          "generating machine as a confound: either generate every compared arm on the cluster "
          "(regenerating L2 and L3b), or keep all datagen local.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
