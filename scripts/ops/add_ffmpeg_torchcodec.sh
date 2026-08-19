#!/bin/bash
# Add conda-forge ffmpeg to the pinned cluster training env so torchcodec can load.
#
# Why: G5a measured the training loop as decode-bound (data_s 0.388 s vs updt_s 0.071 s at
# batch 64, GPU 0% median util). Root cause is in lerobot 0.4.4's pyav path
# (datasets/video_utils.py): it constructs a torchvision VideoReader PER CALL on an
# 82,916-frame container and closes it again, ~128 opens per batch. The torchcodec path is the
# only one with a VideoDecoderCache. torchcodec cannot load on RHEL 8.8 because the OS ships
# no ffmpeg shared libs -- torchcodec 0.10.0 bundles libtorchcodec_core{4..8}.so, so ANY
# ffmpeg major 4-8 satisfies it. conda-forge ffmpeg 6.1.2 verified installable on Leonardo.
#
# Run ONLY when no job is using the env: conda relinks files in place and would break a
# running training step.
#   ssh leonardo 'bash $WORK/cog/repo/scripts/ops/add_ffmpeg_torchcodec.sh'
set -euo pipefail
COG="${WORK}/cog"
ENV_NAME="${COG_TRAIN_ENV:-cog_lerobot}"
export CONDA_PKGS_DIRS="${COG}/conda_pkgs"
export PIP_CACHE_DIR="${COG}/pip_cache"

source "${COG}/miniforge3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo "[ffmpeg] env=${ENV_NAME} python=$(python -V 2>&1)"

conda install -y -c conda-forge "ffmpeg=6.*"

# LOAD-BEARING: conda does NOT put $CONDA_PREFIX/lib on the dynamic linker path, and
# torchcodec dlopen()s libavutil by soname, so it still reports "FFmpeg is not properly
# installed" with ffmpeg sitting right there in the env (observed in cog_lerobot06 before this
# line existed). Anything that uses the torchcodec backend must export this -- train.sbatch
# included.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
echo "[ffmpeg] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "[ffmpeg] ffmpeg: $(ffmpeg -version 2>/dev/null | head -1)"

# The pins are the contract with the local eval env (D22). conda's solver is free to move pip
# packages as collateral, so re-assert them and repair with pip if it did.
python - <<'PY'
import sys, importlib
want = {"torch": "2.7.0+cu128", "torchvision": "0.22.0+cu128",
        "numpy": "1.26.4", "av": "15.1.0", "lerobot": "0.4.4"}
got = {}
for k in want:
    try:
        got[k] = importlib.import_module(k).__version__
    except Exception as e:
        got[k] = f"IMPORT FAILED: {e}"
for k, v in got.items():
    flag = "OK " if got[k] == want[k] else "DRIFTED"
    print(f"[ffmpeg]   {flag} {k} = {v} (want {want[k]})")
drift = {k: got[k] for k in want if got[k] != want[k]}
# Do not fail here: the whole point is to report drift so it can be repaired deliberately.
print(f"[ffmpeg] DRIFT={drift}" if drift else "[ffmpeg] PINS_INTACT")
PY

echo "[ffmpeg] torchcodec check:"
python - <<'PY'
try:
    import torchcodec
    from torchcodec.decoders import VideoDecoder   # noqa: F401
    print(f"[ffmpeg]   torchcodec {torchcodec.__version__} = USABLE")
    print("TORCHCODEC_USABLE")
except Exception as e:
    print(f"[ffmpeg]   torchcodec = UNUSABLE ({type(e).__name__}: {str(e)[:200]})")
    print("TORCHCODEC_UNUSABLE")
PY

# Prove it can actually decode OUR data, not merely import: a real random-access read of the
# 82k-frame L0 video. An import that succeeds and a decode that fails would be the worst
# possible outcome to discover inside a queued 12 h job.
python - <<'PY'
import os, time, glob
from torchcodec.decoders import VideoDecoder
base = os.environ.get("FAST", "") + "/cog/datasets/L0/videos"
vids = sorted(glob.glob(base + "/*/chunk-000/file-000.mp4"))
if not vids:
    print("[ffmpeg]   no video found under", base); raise SystemExit(0)
d = VideoDecoder(vids[0])
n = d.metadata.num_frames or 0
print(f"[ffmpeg]   {os.path.basename(vids[0])}: {n} frames, {d.metadata.average_fps} fps")
idx = [0, n//4, n//2, 3*n//4, n-1] if n > 5 else [0]
t0 = time.time()
for i in idx:
    fr = d[i]
print(f"[ffmpeg]   random-access decode of {len(idx)} frames OK, shape {tuple(fr.shape)}, "
      f"{(time.time()-t0)/len(idx)*1000:.1f} ms/frame (cached decoder)")
print("TORCHCODEC_DECODE_OK")
PY
