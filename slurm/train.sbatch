#!/bin/bash
#SBATCH --job-name=cog_train
#SBATCH --account=euhpc_b38_106
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# One study cell: train the frozen diffusion policy on (TASK, LEVEL, NDEMOS).
#   sbatch slurm/train.sbatch T1 L0 25
# TASK is the dataset prefix: T1 -> cup_place (bare level names), T2/T3 -> prefixed.
#
# UNVERIFIED until G0 clears and this runs once on Leonardo -- every path and module
# name below is from docs/PINS.md and the plan, not from a successful run. First use
# should be a single cell, not the matrix.
set -euo pipefail

TASK="${1:?usage: train.sbatch TASK LEVEL NDEMOS}"
LEVEL="${2:?}"
NDEMOS="${3:?}"

WORK="${COG_WORK:-$WORK/cog}"
RUN_ID="${TASK,,}_${LEVEL}_n${NDEMOS}_s0"
OUT="${WORK}/checkpoints/${RUN_ID}"
case "$TASK" in
  T1) DS_NAME="${LEVEL}" ;;                 # T1 datasets are bare: L0..L3
  T2|T3) DS_NAME="${TASK}_${LEVEL}" ;;
  *) echo "unknown TASK ${TASK}"; exit 2 ;;
esac
DS="${COG_DATASETS:-$FAST/cog/datasets}/${DS_NAME}"

# Compute nodes have NO internet (plan section: cluster). Both of these are mandatory:
# a single online call blocks the whole job until it times out.
export WANDB_MODE=offline
export WANDB_DIR="${WORK}/wandb_offline"
export HF_HUB_OFFLINE=1
export HF_HOME="${WORK}/hf_cache"
export TOKENIZERS_PARALLELISM=false

source "${WORK}/miniforge3/etc/profile.d/conda.sh"
conda activate "${COG_TRAIN_ENV:-cog_lerobot}"
source /home/admin_07/cost_of_generality/configs/train/diffusion_base.sh 2>/dev/null \
  || source "${WORK}/repo/configs/train/diffusion_base.sh"

# The first N episodes of the committed shuffle order (seed 0) -- the nested-subset
# protocol (D4), so a larger cell is a strict superset of a smaller one.
EPISODES="$(python - "$NDEMOS" <<'PY'
import sys
print(",".join(str(i) for i in range(int(sys.argv[1]))))
PY
)"

RESUME=""
if [ -d "${OUT}/checkpoints/last" ]; then
  echo "[train] existing checkpoint found -> resuming"
  RESUME="--resume=true"
fi

mkdir -p "${OUT}"
echo "[train] run=${RUN_ID} dataset=${DS} episodes=0..$((NDEMOS-1)) batch=${COG_DP_BATCH} lr=${COG_DP_LR}"
srun python -m lerobot.scripts.train \
  --dataset.repo_id="local/${DS_NAME}" \
  --dataset.root="${DS}" \
  --dataset.episodes="[${EPISODES}]" \
  --output_dir="${OUT}" \
  --job_name="${RUN_ID}" \
  --batch_size="${COG_DP_BATCH}" \
  --optimizer.lr="${COG_DP_LR}" \
  --wandb.enable=true \
  --wandb.project=cost_of_generality \
  --wandb.disable_artifact=true \
  ${RESUME} \
  ${COG_DP_FLAGS}
echo "[train] DONE ${RUN_ID}"
