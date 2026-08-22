#!/bin/bash
#SBATCH --job-name=cog_smoke_mtdit
#SBATCH --account=euhpc_b38_106
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# Candidate B throughput probe (D30): 200 steps of multi_task_dit at one batch size on
# a real A100, measuring steps/s + peak VRAM + offline-CLIP resolution on a COMPUTE
# node. Decides the batch for the B2 80k verify run. dbg QOS: 30 min cap.
#   sbatch slurm/smoke_mtdit.sbatch 64
set -euo pipefail

BATCH="${1:?usage: smoke_mtdit.sbatch BATCH}"
WORK="${COG_WORK:-$WORK/cog}"
OUT="${WORK}/checkpoints/smoke_mtdit_b${BATCH}"
DS="${COG_DATASETS:-$FAST/cog/datasets}/L1_i20"

export WANDB_MODE=offline
export WANDB_DIR="${WORK}/wandb_offline"
export HF_HUB_OFFLINE=1
export HF_HOME="${WORK}/hf_cache"
export TOKENIZERS_PARALLELISM=false

source "${WORK}/miniforge3/etc/profile.d/conda.sh"
conda activate "${COG_TRAIN_ENV:-cog_lerobot}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${WORK}/repo/src:${PYTHONPATH:-}"
source "${WORK}/repo/configs/train/lang_dit_b.sh"

# background VRAM/util sampler (pattern from smoke_batch_lr.sbatch)
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 2 \
  > "${WORK}/logs/smoke_mtdit_b${BATCH}_gpu.csv" &
SAMPLER=$!

EPISODES="$(python - << 'PY'
print(",".join(str(i) for i in range(100)))
PY
)"

rm -rf "${OUT}"
mkdir -p "$(dirname "${OUT}")"
# smoke overrides MUST come after ${COG_DIT_FLAGS}: draccus is last-flag-wins
srun python -m lerobot.scripts.lerobot_train \
  --policy.discover_packages_path=lerobot_policy_mtdit \
  --dataset.repo_id=local/L1_i20 \
  --dataset.root="${DS}" \
  --dataset.episodes="[${EPISODES}]" \
  --dataset.video_backend=torchcodec \
  --output_dir="${OUT}" \
  --job_name="smoke_mtdit_b${BATCH}" \
  --batch_size="${BATCH}" \
  --wandb.enable=false \
  ${COG_DIT_FLAGS} \
  --steps=200 --save_freq=200 --log_freq=20
kill "${SAMPLER}" 2>/dev/null || true
echo "SMOKE_MTDIT_DONE batch=${BATCH}"
