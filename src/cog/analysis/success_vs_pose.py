"""Does eval success depend on the initial pose, in the way the generation bias predicts?

Motivation. A level whose success rate plateaus far below 1.0 has two very different explanations:
the policy cannot do the task, or the policy was never shown the part of the state space it is being
tested on. The second is measurable here, because generation keeps its rejected attempts and the
frozen eval sets commit their per-episode initial states.

The test joins two committed artifacts:
  * `configs/eval_sets/<level>.json` -- initial object pose per (batch, env), the frozen benchmark;
  * `results/eval_<TASK>_<LEVEL>_n<N>_080000.json` -- per-episode success, indexed by (batch, env).

and reports success rate binned by an initial-state coordinate, alongside the distribution of that
same coordinate among the demos the policy actually trained on. If success collapses exactly where
the training demos thin out, the plateau is a train/eval coverage problem rather than a policy limit.

Concrete case this was written for: T2 (drawer_stow) L1 plateaus at ~0.24 from N=25 to N=400 while
holding 400 unique poses (so not the D27 redundancy bug). Its generation is yaw-skewed -- retained
attempts have yaw sd 0.392 against the rejected attempts' 0.509, i.e. generation succeeded
preferentially at small |yaw| -- which predicts the policy should fail at large |yaw|.

usage:
  python -m cog.analysis.success_vs_pose --task T2 --level L1 --n 400
  python -m cog.analysis.success_vs_pose --task T2 --level L1 --n 400 --dim yaw --bins 4
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from .gen_bias import HDF5_DIR, initial_poses

REPO = pathlib.Path(__file__).resolve().parents[3]
EVAL_SETS = REPO / "configs" / "eval_sets"
RESULTS = REPO / "results"
DIMS = {"x": 0, "y": 1, "yaw": 2}


def _yaw(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def eval_set_poses(task: str, level: str) -> dict[tuple[int, int], np.ndarray]:
    """(batch, env) -> [x, y, yaw] from the committed snapshot.

    Positions are stored in WORLD coordinates, so they carry the env-grid offset; only yaw is
    directly comparable to the demo data, which is why yaw is the default dimension.
    """
    stem = level if task == "T1" else f"{task}_{level}"
    path = EVAL_SETS / f"{stem}.json"
    d = json.loads(path.read_text())
    key = "object_quat" if "object_quat" in d["batches"][0] else "cup_quat"
    pkey = "object_pos" if "object_pos" in d["batches"][0] else "cup_pos"
    out = {}
    for b, batch in enumerate(d["batches"]):
        pos = np.asarray(batch[pkey])
        yaw = _yaw(np.asarray(batch[key]))
        for i in range(len(pos)):
            out[(b, i)] = np.array([pos[i, 0], pos[i, 1], yaw[i]])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--level", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--dim", default="yaw", choices=sorted(DIMS))
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--abs", action="store_true", default=True,
                    help="bin by |value| (default: yaw magnitude is the quantity of interest)")
    args = ap.parse_args()
    j = DIMS[args.dim]

    res = json.loads((RESULTS / f"eval_{args.task}_{args.level}_n{args.n}_080000.json").read_text())
    poses = eval_set_poses(args.task, args.level)

    rows = []
    for o in res["outcomes"]:
        p = poses.get((o["batch"], o["env"]))
        if p is None:
            continue
        v = abs(p[j]) if args.abs else p[j]
        rows.append((v, bool(o["success"])))
    if not rows:
        print("no episodes could be joined to the eval-set snapshot")
        return
    vals = np.array([r[0] for r in rows])
    succ = np.array([r[1] for r in rows])

    label = f"|{args.dim}|" if args.abs else args.dim
    print(f"{args.task} {args.level} N={args.n}: {len(rows)} episodes joined, "
          f"pooled SR {succ.mean():.3f}")
    edges = np.quantile(vals, np.linspace(0, 1, args.bins + 1))
    edges[-1] += 1e-9
    print(f"\nsuccess rate by {label} (equal-count bins over the EVAL distribution):")
    for k in range(args.bins):
        m = (vals >= edges[k]) & (vals < edges[k + 1])
        if m.sum() == 0:
            continue
        print(f"  {label} in [{edges[k]:.3f},{edges[k + 1]:.3f})  n={m.sum():>3}  "
              f"SR={succ[m].mean():.3f}")

    # what the policy was actually trained on
    stem = args.level if args.task == "T1" else f"{args.task}_{args.level}"
    demos = initial_poses(HDF5_DIR / f"{stem}.hdf5")
    if demos is not None:
        dv = np.abs(demos[:, j]) if args.abs else demos[:, j]
        print(f"\ntrained-on demos ({len(dv)}): {label} mean {dv.mean():.3f}, sd {dv.std():.3f}, "
              f"p90 {np.quantile(dv, 0.9):.3f}, max {dv.max():.3f}")
        print(f"eval episodes      ({len(vals)}): {label} mean {vals.mean():.3f}, "
              f"sd {vals.std():.3f}, p90 {np.quantile(vals, 0.9):.3f}, max {vals.max():.3f}")
        frac = float((vals > np.quantile(dv, 0.9)).mean())
        print(f"\n{100 * frac:.0f}% of eval episodes sit above the demos' 90th percentile of {label}")


if __name__ == "__main__":
    main()
