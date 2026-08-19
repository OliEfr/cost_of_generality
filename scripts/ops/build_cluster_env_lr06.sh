#!/bin/bash
# EXPERIMENT (not the frozen training path): build a SECOND, isolated cluster env with
# python 3.12 + the newest LeRobot, purely to A/B the dataloader against the pinned 0.4.4.
#
# Why this exists: G5a measured the 0.4.4 training loop as decode-bound (data_s 0.458 s vs
# updt_s 0.071 s at batch 64, GPU ~87% idle). LeRobot 0.6.x claims a much faster dataloader
# but requires py>=3.12, which the eval side cannot have (Isaac Sim 5.1 needs py3.11 -- G1b).
# So we measure the gain BEFORE paying the migration cost, per the user's instruction
# ("smoke test with 0.5 first before committing - maybe it doesnt improve").
#
# Touches nothing the working env uses: separate conda env name, same miniforge prefix.
#   ssh leonardo 'setsid nohup bash $WORK/cog/repo/scripts/ops/build_cluster_env_lr06.sh \
#                   > $WORK/cog/logs/env_build_lr06.log 2>&1 < /dev/null &'
set -euo pipefail

COG="${WORK}/cog"
MF="${COG}/miniforge3"
ENV_NAME="cog_lerobot06"
LR_VERSION="${COG_LR06_VERSION:-0.6.1}"
mkdir -p "${COG}/logs"
export PIP_CACHE_DIR="${COG}/pip_cache"
export CONDA_PKGS_DIRS="${COG}/conda_pkgs"
export PIP_DISABLE_PIP_VERSION_CHECK=1

echo "[lr06] START $(date -Is) on $(hostname); target lerobot==${LR_VERSION}"
source "${MF}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.12
fi
conda activate "${ENV_NAME}"
echo "[lr06] python: $(python -V 2>&1)"

# Same torch as the pinned env so the comparison isolates the LeRobot version, not the
# CUDA/torch stack. cu128 verified working on this driver (D22).
pip install --no-input torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install --no-input "lerobot==${LR_VERSION}"

# ffmpeg from conda-forge so torchcodec has libavutil: torchcodec fails to load in BOTH our
# envs (OSError: libavutil.so.56) because RHEL 8.8 ships no ffmpeg shared libs. If 0.6's
# faster decoding is torchcodec-only, the upgrade is worthless without this, so install it
# here and let the smoke report which backends actually work. Contained in this env only.
conda install -y -c conda-forge "ffmpeg=6.*" || echo "[lr06] WARN: conda ffmpeg failed"

python - <<'PY'
import importlib, sys
mods = {}
for name in ("torch", "numpy", "lerobot"):
    try:
        m = importlib.import_module(name); mods[name] = getattr(m, "__version__", "?")
    except Exception as e:
        mods[name] = f"IMPORT FAILED: {e}"
for k, v in mods.items():
    print(f"[lr06]   {k} = {v}")
# Which decode backends are actually usable here? This is the whole point of the experiment.
try:
    from torchcodec.decoders import VideoDecoder      # noqa: F401
    print("[lr06]   torchcodec = USABLE")
except Exception as e:
    print(f"[lr06]   torchcodec = UNUSABLE ({type(e).__name__}: {str(e)[:120]})")
try:
    import av
    print(f"[lr06]   pyav = USABLE ({av.__version__})")
except Exception as e:
    print(f"[lr06]   pyav = UNUSABLE ({type(e).__name__})")
# The training entry point moved once already (0.4.4 renamed scripts.train ->
# scripts.lerobot_train), so never assume it: probe both.
for path in ("lerobot.scripts.lerobot_train", "lerobot.scripts.train"):
    try:
        importlib.import_module(path); print(f"[lr06]   entry {path} = OK")
    except Exception as e:
        print(f"[lr06]   entry {path} = MISSING ({type(e).__name__})")
PY

echo "LR06_BUILD_OK $(date -Is)"
date -Is > "${COG}/logs/LR06_BUILD_OK"
