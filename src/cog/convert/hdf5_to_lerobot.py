"""Convert Isaac Lab robomimic-style HDF5 demos to LeRobotDataset v3 (lerobot==0.4.4).

Design (docs/PLAN.md, spec 06):
- ONE dataset per level containing ALL demos; per-N cells subselect episodes at
  train time via --dataset.episodes with the committed shuffle order (seed 0).
- observation.state = proprio only: eef_pos(3)+eef_quat(4)+gripper_pos(2) = 9.
  Privileged object state stored under info.* keys (NOT observation.*) so
  lerobot's feature->policy-input mapping can never wire it into the policy.
- Images: uint8 HWC 128x128 from HDF5, encoded h264 (portability).
- Normalization stats are computed over the full pool (consistency across cells;
  documented design choice).

Usage (cog_lerobot or cog_isaac env; no Isaac needed):
  python -m cog.convert.hdf5_to_lerobot --input data/hdf5/L0_gen.hdf5 \
      --root data/lerobot/cup_place_L0 --repo_id local/cup_place_L0 --fps 20
For L3: pass multiple --input files (per-variant merge, alternating order).
Pass --task drawer_stow for Task 2 (different privileged keys and task string).
"""

import argparse
import json
import random
from pathlib import Path

import h5py
import hashlib
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Per-task differences are exactly: the language string and which privileged object
# keys the recorder wrote. Everything else (proprio state, action, both cameras,
# joint_pos) is shared, so the policy input spec is identical across tasks.
# name -> (hdf5 obs key, dim, component names)
TASK_SPECS = {
    "cup_place": {
        "task_str": "place the cup on the green target marker",
        "privileged": {
            "info.cup_pos": ("cup_pos", 3, ["x", "y", "z"]),
            "info.cup_quat": ("cup_quat", 4, ["w", "x", "y", "z"]),
        },
    },
    "drawer_stow": {
        "task_str": "open the drawer and stow the box inside",
        "privileged": {
            "info.object_pos": ("object_pos", 3, ["x", "y", "z"]),
            "info.object_quat": ("object_quat", 4, ["w", "x", "y", "z"]),
            "info.drawer_joint_pos": ("drawer_joint_pos", 1, ["drawer"]),
        },
    },
    "push_target": {
        "task_str": "push the puck onto the green target marker",
        "privileged": {
            "info.object_pos": ("object_pos", 3, ["x", "y", "z"]),
            "info.object_quat": ("object_quat", 4, ["w", "x", "y", "z"]),
            # the target moves every episode, so its pose is privileged state the vision
            # policy must instead read off the table camera
            "info.target_pos": ("target_pos", 3, ["x", "y", "z"]),
        },
    },
}

BASE_FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (9,),
        "names": ["eef_x", "eef_y", "eef_z", "eef_qw", "eef_qx", "eef_qy", "eef_qz", "grip_l", "grip_r"],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
    },
    "observation.images.table_cam": {
        "dtype": "video", "shape": (128, 128, 3), "names": ["height", "width", "channels"],
    },
    "observation.images.wrist_cam": {
        "dtype": "video", "shape": (128, 128, 3), "names": ["height", "width", "channels"],
    },
    "info.joint_pos": {"dtype": "float32", "shape": (9,), "names": [f"q{i}" for i in range(9)]},
}


def features_for(task: str, emb_dim: int = 0, include_privileged: bool = True) -> dict:
    feats = dict(BASE_FEATURES)
    if include_privileged:
        for name, (_key, dim, names) in TASK_SPECS[task]["privileged"].items():
            feats[name] = {"dtype": "float32", "shape": (dim,), "names": names}
    if emb_dim:
        # Language channel (candidate A): the exact key observation.environment_state
        # is load-bearing -- lerobot types it FeatureType.ENV and DiffusionPolicy
        # appends it to global_cond; any other observation.* float key is silently
        # ignored (dataset_to_policy_features / robot_state_feature are name-exact).
        # ENV features pass IDENTITY normalization (no "ENV" in normalization_mapping),
        # so the unit-norm embedding reaches the policy unchanged.
        feats["observation.environment_state"] = {
            "dtype": "float32",
            "shape": (emb_dim,),
            "names": [f"clip{i:03d}" for i in range(emb_dim)],
        }
    return feats


def episode_frames(f: h5py.File, demo: str, task: str,
                   instruction: str | None = None,
                   embedding: np.ndarray | None = None,
                   include_privileged: bool = True):
    spec = TASK_SPECS[task]
    g = f[f"data/{demo}"]
    obs = g["obs"]
    T = g["actions"].shape[0]
    state = np.concatenate(
        [obs["eef_pos"][:], obs["eef_quat"][:], obs["gripper_pos"][:]], axis=1
    ).astype(np.float32)
    for t in range(T):
        frame = {
            "observation.state": state[t],
            "action": g["actions"][t].astype(np.float32),
            "observation.images.table_cam": obs["table_cam"][t],
            "observation.images.wrist_cam": obs["wrist_cam"][t],
            "info.joint_pos": obs["joint_pos"][t].astype(np.float32),
            # constant within the episode -- save_episode() requires a single task
            "task": instruction if instruction is not None else spec["task_str"],
        }
        if embedding is not None:
            frame["observation.environment_state"] = embedding
        if include_privileged:
            for name, (key, _dim, _names) in spec["privileged"].items():
                frame[name] = obs[key][t].astype(np.float32)
        yield frame


def load_instruction_set(path: str):
    """Load + cross-check the frozen instruction artifacts. Returns (spec, strings, embs)."""
    spec_path = Path(path)
    spec = json.loads(spec_path.read_text())
    npz = np.load(spec_path.parent / spec["embeddings_file"])
    strings: dict[str, list[str]] = {}
    embs: dict[str, np.ndarray] = {}
    for task, ss in spec["tasks"].items():
        e = npz[task]
        stored = [str(s) for s in npz[f"{task}__instructions"]]
        assert stored == ss, f"{task}: npz strings differ from JSON -- regenerate or fix version"
        assert e.shape == (len(ss), spec["embedding"]["dim"]) and e.dtype == np.float32
        assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-4), f"{task}: embeddings not unit-norm"
        strings[task], embs[task] = ss, e
    return spec, strings, embs


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="source HDF5 file(s); >1 = variant merge")
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--task", nargs="+", choices=sorted(TASK_SPECS), default=["cup_place"],
                    help="selects task string + privileged keys; default keeps T1 behaviour. "
                         "Either 1 value (applies to every input) or exactly one per --input "
                         "(multi-task probe mode; requires --instructions)")
    ap.add_argument("--instructions", default=None,
                    help="path to frozen configs/instructions/instructions_vN.json: assign one of "
                         "the 20 synonyms per episode (task string + observation.environment_state "
                         "embedding). Default off = today's constant-string behaviour")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--shuffle_seed", type=int, default=0)
    ap.add_argument("--max_episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--allow_failed", action="store_true",
                    help="permit input filenames containing '_failed' (blocked by default: a glob like L3v0*.hdf5 also matches L3v00_failed.hdf5)")
    args = ap.parse_args()
    bad = [p for p in args.input if "_failed" in Path(p).name]
    if bad and not args.allow_failed:
        raise SystemExit(f"refusing failed-demo inputs (use --allow_failed to override): {bad}")

    if len(args.task) == 1:
        tasks_per_file = [args.task[0]] * len(args.input)
    elif len(args.task) == len(args.input):
        tasks_per_file = list(args.task)
    else:
        raise SystemExit(f"--task must have 1 value or one per --input ({len(args.input)}), got {len(args.task)}")
    multi_task = len(set(tasks_per_file)) > 1
    if multi_task and not args.instructions:
        raise SystemExit("multi-task conversion requires --instructions (task-discriminating strings)")

    instr_spec = instr_strings = instr_embs = None
    if args.instructions:
        if not multi_task and len(args.input) > 1:
            # 20 instructions round-robined over a k-variant interleave (k=10 for L3)
            # would make instruction a perfect predictor of variant (gcd trap, D18
            # lesson). An L3-lang assignment needs its own design -- refuse loudly.
            raise SystemExit("--instructions with a same-task multi-file merge is not supported: "
                             "instruction index would confound with the variant interleave")
        instr_spec, instr_strings, instr_embs = load_instruction_set(args.instructions)
        for t in set(tasks_per_file):
            assert t in instr_strings, f"{t} missing from {args.instructions}"

    root = Path(args.root)
    if root.exists():
        raise SystemExit(f"refusing to overwrite existing dataset root {root} (no-delete rule)")

    # build global episode list: (file_idx, demo_name), interleaved across input
    # files so nested-N prefixes stay variant-balanced for L3
    per_file: list[list[str]] = []
    handles = [h5py.File(p, "r") for p in args.input]
    try:
        for f in handles:
            demos = sorted(f["data"].keys(), key=lambda d: int(d.split("_")[1]))
            per_file.append(demos)
        rng = random.Random(args.shuffle_seed)
        for demos in per_file:
            rng.shuffle(demos)
        order: list[tuple[int, str]] = []
        longest = max(len(d) for d in per_file)
        for i in range(longest):
            for fi, demos in enumerate(per_file):
                if i < len(demos):
                    order.append((fi, demos[i]))
        if args.max_episodes:
            order = order[: args.max_episodes]

        emb_dim = instr_spec["embedding"]["dim"] if instr_spec else 0
        # Probe (multi-task) roots drop the per-task privileged info.* keys: feature
        # dicts must be uniform across episodes, and info.* is never policy input (D5).
        include_privileged = not multi_task
        ds = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            features=features_for(tasks_per_file[0], emb_dim, include_privileged),
            root=str(root),
            robot_type="franka",
            use_videos=True,
            image_writer_threads=8,
            vcodec="h264",
        )
        instr_records = []
        try:
            # per-task counter % K over the interleaved order: every nested-N prefix
            # stays instruction-balanced within each task (n100 -> exactly 5 each)
            counters: dict[str, int] = {}
            for ep_idx, (fi, demo) in enumerate(order):
                task = tasks_per_file[fi]
                instruction = embedding = None
                if instr_strings:
                    k = counters.get(task, 0)
                    counters[task] = k + 1
                    instr_idx = k % len(instr_strings[task])
                    instruction = instr_strings[task][instr_idx]
                    embedding = instr_embs[task][instr_idx]
                    instr_records.append({"episode_index": ep_idx, "task_name": task,
                                          "instruction_index": instr_idx, "instruction": instruction})
                for frame in episode_frames(handles[fi], demo, task,
                                            instruction, embedding, include_privileged):
                    ds.add_frame(frame)
                ds.save_episode()
                if (ep_idx + 1) % 25 == 0:
                    print(f"[convert] {ep_idx + 1}/{len(order)} episodes", flush=True)
        finally:
            ds.finalize()

        manifest = {
            "task": args.task[0] if len(args.task) == 1 else args.task,
            "inputs": args.input,
            "shuffle_seed": args.shuffle_seed,
            "episode_order": [{"episode_index": i, "file": args.input[fi], "demo": d}
                              for i, (fi, d) in enumerate(order)],
        }
        if args.instructions:
            spec_path = Path(args.instructions)
            manifest["instructions"] = {
                "file": str(spec_path),
                "version": instr_spec["version"],
                "model": instr_spec["model"],
                "sha256_json": sha256_file(spec_path),
                "sha256_npz": sha256_file(spec_path.parent / instr_spec["embeddings_file"]),
                "privileged_dropped": multi_task,
                "episodes": instr_records,
            }
        (root / "conversion_manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"[convert] DONE {len(order)} episodes -> {root}", flush=True)
    finally:
        for f in handles:
            f.close()


if __name__ == "__main__":
    main()
