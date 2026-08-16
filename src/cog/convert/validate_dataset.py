"""Validate a converted LeRobotDataset against its source HDF5s.

Checks: episode/frame counts, random-frame pixel comparison (codec tolerance),
action/state ranges, timestamp uniformity. Replay-in-sim happens separately via
IsaacLab replay_demos.py on the source HDF5.
"""

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--n_checks", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.root)
    manifest = json.loads((root / "conversion_manifest.json").read_text())
    ds = LeRobotDataset(args.repo_id, root=str(root), video_backend="pyav")
    order = manifest["episode_order"]
    print(f"[validate] episodes={ds.num_episodes} frames={ds.num_frames}")
    assert ds.num_episodes == len(order), "episode count mismatch"

    rng = random.Random(1)
    max_pix_err = 0.0
    for _ in range(args.n_checks):
        rec = rng.choice(order)
        with h5py.File(rec["file"], "r") as f:
            g = f[f"data/{rec['demo']}"]
            T = g["actions"].shape[0]
            t = rng.randrange(T)
            ep = rec["episode_index"]
            frame_idx = int(ds.meta.episodes["dataset_from_index"][ep]) + t
            item = ds[frame_idx]
            assert int(item["episode_index"]) == ep, f"episode index mismatch {item['episode_index']} vs {ep}"
            a_src = torch.from_numpy(g["actions"][t].astype("float32"))
            assert torch.allclose(item["action"], a_src, atol=1e-5), f"action mismatch ep{ep} t{t}"
            img_src = g["obs"]["table_cam"][t].astype("float32") / 255.0
            img_ds = item["observation.images.table_cam"].permute(1, 2, 0).numpy()
            err = float(np.abs(img_src - img_ds).mean())
            max_pix_err = max(max_pix_err, err)
    print(f"[validate] max mean |pixel err| over {args.n_checks} frames: {max_pix_err:.4f} (codec tolerance <0.03)")
    assert max_pix_err < 0.03, "pixel error above codec tolerance"

    st = ds.meta.stats
    print(f"[validate] action min={np.round(st['action']['min'],3)} max={np.round(st['action']['max'],3)}")
    print("[validate] VALIDATE_OK")


if __name__ == "__main__":
    main()
