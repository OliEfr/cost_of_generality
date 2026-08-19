#!/bin/bash
# Build the cluster TRAINING env: miniforge + py3.11 + torch 2.7.0/cu128 + lerobot 0.4.4.
#
# Runs ON a Leonardo LOGIN node -- compute nodes have no internet (CLAUDE.md rule 4), so a
# login-node install is the only route. Mirrors the locally verified combo in docs/PINS.md
# version-for-version, so a checkpoint trained here loads in the local cog_isaac eval env.
#
#   ssh leonardo 'setsid nohup bash $WORK/cog/repo/scripts/ops/build_cluster_env.sh \
#                   > $WORK/cog/logs/env_build.log 2>&1 < /dev/null &'
#
# Idempotent: re-running skips miniforge and env creation if they already exist, and pip
# is a no-op when the pins are already satisfied.
set -euo pipefail

COG="${WORK}/cog"
MF="${COG}/miniforge3"
ENV_NAME="${COG_TRAIN_ENV:-cog_lerobot}"
mkdir -p "${COG}/logs"

# $HOME has a 50 GB quota and pip/conda caches run to several GB. Keep both in $WORK.
export PIP_CACHE_DIR="${COG}/pip_cache"
export CONDA_PKGS_DIRS="${COG}/conda_pkgs"
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "[env] START $(date -Is) on $(hostname)"

if [ ! -x "${MF}/bin/conda" ]; then
  echo "[env] installing miniforge -> ${MF}"
  cd "${COG}"
  # NOTE a real reproducibility limit: this URL is `latest`, so a rebuild months from now gets a
  # different conda/python base than the one that produced our results (observed: conda 26.3.2,
  # python 3.11.15, constructor dated 2026-06-01). Override COG_MINIFORGE_URL to pin an exact
  # release when exact reproduction matters. The authoritative record of what actually ran is
  # docs/env/cluster_cog_lerobot.freeze.txt, not this line.
  curl -fsSL -o miniforge.sh \
    "${COG_MINIFORGE_URL:-https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh}"
  # -f is load-bearing: $WORK/cog/miniforge3 is pre-created by the directory-tree step, and
  # the installer aborts with "File or directory already exists" on an existing prefix
  # unless -f is given (-b = batch, -f = no error if the prefix exists). Verified 2026-08-19.
  bash miniforge.sh -b -f -p "${MF}"
  rm -f miniforge.sh
else
  echo "[env] miniforge already present at ${MF}"
fi
# shellcheck disable=SC1091
source "${MF}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[env] creating conda env ${ENV_NAME} (python 3.11)"
  conda create -y -n "${ENV_NAME}" python=3.11
else
  echo "[env] conda env ${ENV_NAME} already exists"
fi
conda activate "${ENV_NAME}"
echo "[env] python: $(python -V 2>&1) at $(which python)"

# torch FIRST from the cu128 index, WITH its dependencies: the cu128 build pulls
# nvidia-*-cu12 12.8.x wheels, and installing torch later with --no-deps would leave
# whatever CUDA minor versions the default PyPI torch had already put there -> import-time
# .so failures on the compute node. lerobot SECOND: torch 2.7.0+cu128 satisfies its
# torch<2.11,>=2.2.1, and torchcodec 0.10.0 declares no torch pin at all (checked in its
# METADATA), so pip has no reason to move torch. Verified locally: torchcodec's .so cannot
# load but its import is lazy and we use the pyav backend anyway (PINS video_backend).
echo "[env] pip install torch 2.7.0 / torchvision 0.22.0 from the cu128 index"
pip install --no-input torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

echo "[env] pip install lerobot==0.4.4"
pip install --no-input "lerobot==0.4.4"

# numpy/av/opencv are pinned to the local eval env's exact versions (D22 parity). opencv is
# in the list because the resolver picked 4.12.0.88 here vs 4.11.0.86 locally; 4.12 imports
# fine against numpy 1.26.4 despite its metadata claiming numpy>=2 (checked on the login
# node), so this pin is parity, not a fix. rerun-sdk still warns about numpy>=2 -- it does
# locally too, and nothing in the training path imports it.
echo "[env] pinning numpy==1.26.4, av==15.1.0, opencv-python-headless==4.11.0.86 (local parity)"
pip install --no-input "numpy==1.26.4" "av==15.1.0" "opencv-python-headless==4.11.0.86"

# Artifact-not-exit-code check (CLAUDE.md rule 10 / D6): import everything the training
# entry point touches and assert the pins, so a silently-clobbered torch fails HERE and not
# eight hours into a queued A100 job. CUDA is NOT checked here: login nodes have no GPU.
python - <<'PY'
import importlib, sys
import torch, numpy, av, lerobot, torchvision
want = {"torch": "2.7.0+cu128", "torchvision": "0.22.0+cu128",
        "numpy": "1.26.4", "av": "15.1.0", "lerobot": "0.4.4"}
got = {"torch": torch.__version__, "torchvision": torchvision.__version__,
       "numpy": numpy.__version__, "av": av.__version__, "lerobot": lerobot.__version__}
bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
for k, v in got.items():
    print(f"[env]   {k} = {v}")
# The training entry point must be importable by module path (train.sbatch calls
# `python -m lerobot.scripts.lerobot_train`; the 0.4.3-and-earlier path does not exist).
importlib.import_module("lerobot.scripts.lerobot_train")
print("[env]   lerobot.scripts.lerobot_train importable = True")
if bad:
    print(f"[env] VERSION MISMATCH {bad}", file=sys.stderr)
    sys.exit(1)
PY

# --- ffmpeg + torchcodec: REQUIRED, and previously missing from this script -------------------
# Without these two the env builds "successfully" and then trains 5x slower, because
# COG_VIDEO_BACKEND=torchcodec (frozen in configs/train/diffusion_base.sh) cannot load. They were
# originally applied by hand while debugging, which meant this script did NOT reproduce the env
# that actually ran the calibration. Folded in on 2026-08-19 after an audit found the gap.
#
# torchcodec MUST be 0.5: lerobot 0.4.4 allows >=0.2.1,<0.11 and pip picks 0.10.0, which is built
# against torch 2.10 and fails with `libtorchcodec_core6.so: undefined symbol: _ZN3c10...`. 0.7.0
# imports but its custom ops are broken (`no fallback function is registered for schema
# torchcodec_ns::_convert_to_tensor`) -- so "it imports" is not a sufficient test; a decode is.
# ffmpeg must be pinned to a MAJOR: bare `conda install ffmpeg` now resolves to 9.x, whose
# libavutil.so.61 no torchcodec wheel below 0.16 supports.
echo "[env] conda-forge ffmpeg 6 (torchcodec needs an ffmpeg; RHEL 8.8 ships none)"
conda install -y -c conda-forge "ffmpeg=6.*"
echo "[env] pip install torchcodec==0.5 (ABI-matched to torch 2.7; --no-deps so torch is untouched)"
pip install --no-input --no-deps "torchcodec==0.5"

# Anything using the torchcodec backend must also export
# LD_LIBRARY_PATH=$CONDA_PREFIX/lib, because conda does not put it on the linker path and
# torchcodec dlopen()s ffmpeg by soname. train.sbatch and the smoke scripts do this.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
python - <<'PYTC'
import torch, torchcodec
from torchcodec.decoders import VideoDecoder   # noqa: F401  -- import alone is not enough...
print(f"[env]   torchcodec {torchcodec.__version__} loads against torch {torch.__version__}")
PYTC

echo "ENV_BUILD_OK $(date -Is)"
date -Is > "${COG}/logs/ENV_BUILD_OK"
