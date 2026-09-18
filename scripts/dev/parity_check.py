#!/usr/bin/env python3
"""Compare two Mimic-generated HDF5 datasets episode-for-episode: gate G2, cluster vs workstation.

Why this exists. The reverse-ablation arms are generated on Leonardo's A100s but are compared
against L2 and L3b, whose demos were generated on the workstation's RTX 4090. If the two machines
produce different demonstrations -- different trajectories, or merely different pixels -- then a
success-rate difference between arms confounds "which disturbance was removed" with "which GPU
rendered the training images". That is structurally the same class of artefact as D27's
pose-redundancy bug, and it is worth one 30-minute job to rule out.

Run the SAME leg on both machines (same env key, same seed, same --num_envs; --num_envs is part of
the experiment because attempts are distributed across the async env pool) and point this at the
two files.

**Episodes are matched by initial state, never by file index.** The generator runs N envs
asynchronously and writes demos in COMPLETION order, which is a race: the first run of this check
compared demo_k to demo_k, found 17 of 40 "divergent", and declared a parity failure that did not
exist. Matching on the initial pose instead showed 37 of those 40 were the same episode in a
different slot, with identical lengths. The three genuinely different episodes come from the
accept/reject boundary -- a marginal attempt that succeeds on one machine and fails on the other
makes the generator draw one more pose to fill its quota.

What that means for the verdict: a handful of unmatched episodes is EXPECTED and harmless, because
the poses are drawn from the same seeded distribution and which 40 of them survive is immaterial.
What would not be harmless is a systematic shift -- many unmatched episodes, or a different failure
count -- because that would mean one machine rejects harder cases than the other and the training
distributions differ in difficulty.

usage: parity_check.py LOCAL.hdf5 CLUSTER.hdf5 [--demos 3] [--frames 3] [--pose-atol 1e-5]
"""

from __future__ import annotations

import argparse
import sys

import h5py
import numpy as np

ATOL_ACTION = 1e-4   # Mimic's transformed plan, on a matched episode
ATOL_STATE = 1e-3    # 1 mm: PhysX GPU reduction order may differ by architecture
MIN_MATCHED_FRAC = 0.90
# Pixel verdict bands, in 0-255 units, on MATCHED episode pairs.
MAE_AGREE, P99_AGREE = 1.0, 8.0
MAE_TOLERABLE, CHANNEL_SHIFT_MAX = 3.0, 2.0


def initial_vector(demo) -> np.ndarray:
    """Flatten every leaf of initial_state into one vector, in sorted key order (task-agnostic)."""
    parts = []

    def walk(g, prefix=""):
        for name in sorted(g):
            item = g[name]
            if isinstance(item, h5py.Group):
                walk(item, f"{prefix}{name}/")
            else:
                parts.append(np.asarray(item, dtype=np.float64).ravel())

    walk(demo["initial_state"])
    return np.concatenate(parts)


def match_episodes(da, db, atol: float):
    """Greedy one-to-one match of cluster episodes to local ones by initial state."""
    ka, kb = sorted(da), sorted(db)
    va = np.stack([initial_vector(da[k]) for k in ka])
    pairs, unmatched = [], []
    taken = set()
    for k in kb:
        dist = np.abs(va - initial_vector(db[k])).max(axis=1)
        order = np.argsort(dist)
        hit = next((int(i) for i in order if i not in taken and dist[i] < atol), None)
        if hit is None:
            unmatched.append((k, float(dist[order[0]])))
        else:
            taken.add(hit)
            pairs.append((ka[hit], k))
    return pairs, unmatched


def leaves(group, prefix=""):
    for name in sorted(group):
        item = group[name]
        if isinstance(item, h5py.Group):
            yield from leaves(item, f"{prefix}{name}/")
        else:
            yield f"{prefix}{name}", item


def block_delta(a_demo, b_demo, block: str) -> float:
    if block not in a_demo or block not in b_demo:
        return 0.0
    al, bl = dict(leaves(a_demo[block])), dict(leaves(b_demo[block]))
    if al.keys() != bl.keys():
        return float("inf")
    worst = 0.0
    for key in al:
        x, y = np.asarray(al[key], dtype=np.float64), np.asarray(bl[key], dtype=np.float64)
        if x.shape != y.shape:
            return float("inf")
        worst = max(worst, float(np.abs(x - y).max()))
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("local")
    ap.add_argument("cluster")
    ap.add_argument("--demos", type=int, default=3, help="matched pairs sampled for pixels")
    ap.add_argument("--frames", type=int, default=3, help="frames per sampled pair")
    ap.add_argument("--pose-atol", type=float, default=1e-5)
    args = ap.parse_args()

    fa, fb = h5py.File(args.local, "r"), h5py.File(args.cluster, "r")
    da, db = fa["data"], fb["data"]
    hard: list[str] = []
    soft: list[str] = []

    # 1. counts. data.attrs["total"] is the sum of every demo's num_samples (verified 7511 on
    #    L3bv00), not an attempt count -- the accept/reject tally only reaches the log.
    print(f"1. demos           local={len(da)}  cluster={len(db)}")
    print(f"   total timesteps local={da.attrs.get('total')}  cluster={db.attrs.get('total')}")
    if len(da) != len(db):
        hard.append(f"demo counts differ ({len(da)} vs {len(db)})")

    # 2. match by initial state. Order is a race; identity is the pose.
    pairs, unmatched = match_episodes(da, db, args.pose_atol)
    frac = len(pairs) / max(len(db), 1)
    same_slot = sum(1 for a, b in pairs if a == b)
    print(f"2. matched by initial pose  {len(pairs)}/{len(db)} ({frac:.0%}), "
          f"{same_slot} of them in the same file slot")
    if unmatched:
        d = [x for _, x in unmatched]
        print(f"   unmatched: {len(unmatched)} (nearest-pose distance {min(d):.2e}..{max(d):.2e}) "
              f"-- expected from the accept/reject boundary")
    if frac < MIN_MATCHED_FRAC:
        hard.append(f"only {frac:.0%} of cluster episodes have a counterpart locally "
                    f"(want >= {MIN_MATCHED_FRAC:.0%}): the pose stream is not reproducing")

    # 3-5. numeric agreement on MATCHED pairs only.
    bad_len = [(a, b) for a, b in pairs
               if int(da[a].attrs["num_samples"]) != int(db[b].attrs["num_samples"])]
    print(f"3. matched lengths {len(pairs) - len(bad_len)}/{len(pairs)} identical")
    if bad_len:
        hard.append(f"{len(bad_len)} matched episode(s) differ in length, e.g. {bad_len[:3]}: "
                    f"same initial state, different trajectory -- physics is not reproducing")

    act = st = 0.0
    for a, b in pairs:
        if "actions" in da[a] and da[a]["actions"].shape == db[b]["actions"].shape:
            act = max(act, float(np.abs(np.asarray(da[a]["actions"], dtype=np.float64)
                                        - np.asarray(db[b]["actions"], dtype=np.float64)).max()))
        st = max(st, block_delta(da[a], db[b], "states"))
    print(f"4. actions         max|delta|={act:.3e}  (atol {ATOL_ACTION:.0e})")
    print(f"5. states          max|delta|={st:.3e}  (atol {ATOL_STATE:.0e})")
    if act > ATOL_ACTION:
        soft.append(f"actions max|delta| {act:.3e} exceeds {ATOL_ACTION:.0e}")
    if st > ATOL_STATE:
        soft.append(f"states max|delta| {st:.3e} exceeds {ATOL_STATE:.0e}")

    # 6. pixels, on matched pairs of equal length. Not expected to be bit-identical; the question
    #    is whether the difference is a uniform shift (exposure/hue -- the policy sees a different
    #    world) or high-frequency sampling noise (it does not).
    print("6. pixels (matched pairs)")
    verdict = "AGREE"
    usable = [(a, b) for a, b in pairs
              if int(da[a].attrs["num_samples"]) == int(db[b].attrs["num_samples"])]
    if usable:
        step = max(len(usable) // max(args.demos, 1), 1)
        sample = usable[::step][: args.demos]
        cams = sorted(n for n, it in da[sample[0][0]]["obs"].items()
                      if it.dtype == np.uint8 and it.ndim == 4)
        for cam in cams:
            maes, p99s, shifts = [], [], []
            for a, b in sample:
                A, B = da[a]["obs"][cam], db[b]["obs"][cam]
                T = A.shape[0]
                for t in sorted({0, T // 2, T - 1})[: args.frames]:
                    x = np.asarray(A[t], dtype=np.float64)
                    y = np.asarray(B[t], dtype=np.float64)
                    d = np.abs(x - y)
                    maes.append(d.mean())
                    p99s.append(np.percentile(d, 99))
                    shifts.append(np.abs(x.mean(axis=(0, 1)) - y.mean(axis=(0, 1))).max())
            mae, p99, shift = float(np.mean(maes)), float(np.max(p99s)), float(np.max(shifts))
            print(f"   {cam:12s} MAE={mae:6.3f}  p99={p99:7.3f}  "
                  f"max channel-mean shift={shift:6.3f}   ({len(sample)} pairs)")
            if mae > MAE_TOLERABLE or shift > CHANNEL_SHIFT_MAX:
                verdict = "DIVERGENT"
            elif (mae > MAE_AGREE or p99 > P99_AGREE) and verdict != "DIVERGENT":
                verdict = "TOLERABLE"
    else:
        hard.append("no matched pair of equal length to compare pixels on")

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
        print("PARITY_AGREE: matched episodes are identical and pixels are within noise -- "
              "cluster- and workstation-generated arms are freely mixable.")
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
