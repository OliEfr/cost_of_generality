"""Quantify how much of LeRobot 0.4.4's video decode cost is per-call container opening.

G5a found training decode-bound (data_s 0.388 s vs updt_s 0.071 s at batch 64, GPU 0% median
util) and the source shows why: datasets/video_utils.decode_video_frames_torchvision builds a
torchvision VideoReader PER CALL on a single 82,916-frame mp4 and closes it again. The
torchcodec path is the only one with a VideoDecoderCache -- and torchcodec will not load
without system ffmpeg.

This measures the ceiling of ANY caching fix, CPU-only, no GPU and no cluster needed:
  A) what LeRobot does now: one open+seek+decode per request
  B) the same requests through ONE reused reader
The ratio B/A is the most a decoder cache could ever buy, whether we get it from torchcodec,
from splitting the videos per episode, or from a local wrapper.

  python scripts/dev/decode_bench.py [--dataset L0] [--n 60]
"""

import argparse
import random
import time
from pathlib import Path

import torch
import torchvision
from lerobot.datasets.video_utils import decode_video_frames

REPO = Path(__file__).resolve().parents[2]


def bench_lerobot_path(video: Path, ts_pairs, tolerance_s):
    """Exactly what training does: one decode_video_frames call per item."""
    t0 = time.perf_counter()
    for ts in ts_pairs:
        decode_video_frames(video, ts, tolerance_s, backend="pyav")
    return time.perf_counter() - t0


def bench_reused_reader(video: Path, ts_pairs, tolerance_s):
    """Same frames, but the container is opened ONCE and seeked repeatedly."""
    torchvision.set_video_backend("pyav")
    reader = torchvision.io.VideoReader(str(video), "video")
    t0 = time.perf_counter()
    for ts in ts_pairs:
        first, last = min(ts), max(ts)
        reader.seek(first, keyframes_only=True)
        got = []
        for frame in reader:
            got.append(frame["data"])
            if frame["pts"] >= last:
                break
    dt = time.perf_counter() - t0
    reader.container.close()
    return dt


def compare_backends(video: Path, ts_pairs, tolerance_s):
    """Assert pyav and torchcodec return the SAME frames, not merely similar timings.

    They disagree by construction: the pyav path seeks and matches by timestamp, while the
    torchcodec path converts to an index with round(ts * average_fps) and uses
    seek_mode="approximate". If that ever picks a neighbouring frame, training silently
    consumes different data and nothing in the pipeline complains -- exactly the class of
    bug that cost this project real time before (silent frame misalignment in merged
    episodes). So the switch is not allowed to happen on trust.
    """
    n_bad = 0
    max_diff = 0.0
    for ts in ts_pairs:
        a = decode_video_frames(video, ts, tolerance_s, backend="pyav")
        b = decode_video_frames(video, ts, tolerance_s, backend="torchcodec")
        if a.shape != b.shape:
            n_bad += 1
            print(f"  SHAPE MISMATCH at ts={ts}: {a.shape} vs {b.shape}")
            continue
        af = a.float()
        bf = b.float()
        d = (af - bf).abs().max().item()
        max_diff = max(max_diff, d)
        if not torch.equal(a, b):
            n_bad += 1
            if n_bad <= 3:
                print(f"  FRAME MISMATCH at ts={ts}: max abs diff {d}")
    print(f"\ncompared {len(ts_pairs)} fetches: {n_bad} mismatched, max abs pixel diff {max_diff}")
    if n_bad == 0:
        print("BACKENDS_IDENTICAL")
    else:
        print("BACKENDS_DIFFER -- do NOT switch backend without understanding this")
    return n_bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="L0")
    ap.add_argument("--n", type=int, default=60, help="number of item-fetches to simulate")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--compare-backends", action="store_true",
                    help="also assert pyav and torchcodec decode identical frames")
    ap.add_argument("--root", default=None,
                    help="dataset root override (e.g. $FAST/cog/datasets on the cluster)")
    args = ap.parse_args()

    root = Path(args.root) if args.root else REPO / "data" / "lerobot"
    vids = sorted((root / args.dataset / "videos").glob("*/chunk-*/file-*.mp4"))
    if not vids:
        raise SystemExit(f"no videos found for dataset {args.dataset}")
    video = vids[0]

    # n_obs_steps=2 => each item wants frames (t-1, t), which is what the real loader asks for.
    import av
    with av.open(str(video)) as c:
        nframes = c.streams.video[0].frames
    print(f"video: {video}")
    print(f"frames: {nframes}  size: {video.stat().st_size / 1e6:.1f} MB")

    random.seed(0)
    idx = [random.randint(1, nframes - 2) for _ in range(args.n)]
    ts_pairs = [[(i - 1) / args.fps, i / args.fps] for i in idx]
    tolerance_s = 1e-4 + 1.0 / args.fps  # generous: we are timing, not validating alignment

    if args.compare_backends:
        compare_backends(video, ts_pairs, tolerance_s)

    a = bench_lerobot_path(video, ts_pairs, tolerance_s)
    b = bench_reused_reader(video, ts_pairs, tolerance_s)

    per_a, per_b = a / args.n * 1000, b / args.n * 1000
    print(f"\nA) per-call open (what LeRobot 0.4.4 does): {a:.2f} s total, {per_a:.2f} ms/item")
    print(f"B) one reused reader                      : {b:.2f} s total, {per_b:.2f} ms/item")
    print(f"\nspeedup from caching the container: {a / b:.1f}x")
    # Translate into the number that actually matters: a training step fetches
    # batch x n_cameras items.
    for batch, ncam in ((64, 2),):
        print(f"\nat batch={batch}, {ncam} cameras -> {batch * ncam} item-fetches/step")
        print(f"  A) {per_a * batch * ncam / 1000:.3f} s/step of decode")
        print(f"  B) {per_b * batch * ncam / 1000:.3f} s/step of decode")
        print("  (measured data_s on the A100 was 0.388 s/step with 8 parallel workers)")


if __name__ == "__main__":
    main()
