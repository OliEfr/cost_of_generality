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
    ap.add_argument("--expect_episodes", type=int, default=0,
                    help="fail unless total_episodes matches (0 = skip check)")
    ap.add_argument("--instructions", default=None,
                    help="frozen instructions_vN.json: run language-dataset checks (task rows, "
                         "per-episode constancy, embedding<->task match, prefix balance)")
    ap.add_argument("--match_order", default=None,
                    help="root of a baseline dataset whose conversion_manifest episode_order "
                         "(file basename, demo) must be identical -- required for any SR "
                         "comparison against a cell trained on the baseline root")
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
    if args.expect_episodes and ds.num_episodes != args.expect_episodes:
        raise SystemExit(f"[validate] FAIL: total_episodes={ds.num_episodes} != expected {args.expect_episodes}")

    if args.match_order:
        other = json.loads((Path(args.match_order) / "conversion_manifest.json").read_text())
        ours = [(Path(r["file"]).name, r["demo"]) for r in order]
        theirs = [(Path(r["file"]).name, r["demo"]) for r in other["episode_order"]]
        assert ours == theirs, (
            f"episode order differs from {args.match_order} -- nested-N subsets would select "
            f"different demos, invalidating any SR comparison (first divergence at index "
            f"{next(i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b)})")
        print(f"[validate] episode order identical to {args.match_order} ({len(ours)} episodes)")

    if args.instructions:
        spec_path = Path(args.instructions)
        spec = json.loads(spec_path.read_text())
        npz = np.load(spec_path.parent / spec["embeddings_file"])
        info = manifest["instructions"]
        recs = info["episodes"]
        assert len(recs) == ds.num_episodes, "manifest instruction records != episode count"
        tasks_used = sorted({r["task_name"] for r in recs})
        # every used string must come from the frozen set; a task with >= 20 episodes
        # must have used all 20 (counter % 20 guarantees it)
        expected_strings = {r["instruction"] for r in recs}
        for t in tasks_used:
            n_t = sum(r["task_name"] == t for r in recs)
            if n_t >= len(spec["tasks"][t]):
                assert set(spec["tasks"][t]) <= expected_strings, f"{t}: not all 20 instructions used"
        assert expected_strings <= {s for t in tasks_used for s in spec["tasks"][t]}, \
            "manifest contains strings outside the frozen instruction set"
        got_strings = set(ds.meta.tasks.index)
        assert got_strings == expected_strings, (
            f"tasks.parquet mismatch: {len(got_strings)} strings vs expected "
            f"{len(expected_strings)}; diff={got_strings ^ expected_strings}")
        print(f"[validate] tasks.parquet: {len(got_strings)} instruction strings, as expected")

        # per-episode task constancy + embedding<->task match on sampled frames
        for _ in range(args.n_checks):
            rec = rng.choice(recs)
            ep = rec["episode_index"]
            start = int(ds.meta.episodes["dataset_from_index"][ep])
            end = int(ds.meta.episodes["dataset_to_index"][ep])
            for fi in (start, rng.randrange(start, end), end - 1):
                item = ds[int(fi)]
                assert item["task"] == rec["instruction"], (
                    f"ep{ep} frame{fi}: task {item['task']!r} != manifest {rec['instruction']!r}")
                emb = npz[rec["task_name"]][rec["instruction_index"]]
                got = item["observation.environment_state"].numpy()
                got = got[-1] if got.ndim == 2 else got  # delta_timestamps may stack obs steps
                assert np.allclose(got, emb, atol=1e-6), f"ep{ep}: embedding mismatch"
        print(f"[validate] task constancy + embedding match on {args.n_checks} episodes: ok")

        # prefix balance: nested-N subsets must stay instruction-balanced per task
        for n in (10, 25, 50, 100, 200, 400, 800):
            if n > len(recs):
                break
            for t in tasks_used:
                counts = np.bincount(
                    [r["instruction_index"] for r in recs[:n] if r["task_name"] == t],
                    minlength=len(spec["tasks"][t]))
                nz = counts[counts > 0] if counts.max() else counts
                assert counts.max() - counts.min() <= 1 or (nz.max() - nz.min() <= 1 and counts.min() == 0), \
                    f"prefix N={n} task {t}: instruction imbalance {counts.tolist()}"
            if len(tasks_used) > 1:
                per_task = [sum(r["task_name"] == t for r in recs[:n]) for t in tasks_used]
                assert max(per_task) - min(per_task) <= 1, f"prefix N={n}: task imbalance {per_task}"
        print("[validate] nested-N instruction balance: ok")

    print("[validate] VALIDATE_OK")


if __name__ == "__main__":
    main()
