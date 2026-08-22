#!/bin/bash
#SBATCH --job-name=cog_lang_a
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
# Candidate A verify cell (language-conditioning investigation, D30). Sibling of
# slurm/train.sbatch -- that file is the production matrix path and stays untouched.
#   sbatch slurm/train_lang_a.sbatch T1 L1 100
# Differences vs train.sbatch: RUN_ID/DS_NAME carry the _i20 stem, the sourced config
# is lang_diffusion_a.sh (which just sources the frozen diffusion_base.sh), and
# COG_RUN_ID / COG_DS_NAME env overrides exist for the probe cell:
#   COG_RUN_ID=t1t3_probe_i20_n200_s0 COG_DS_NAME=probe_T1T3_L1_i20 \
#     sbatch slurm/train_lang_a.sbatch T1 L1 200
set -euo pipefail

TASK="${1:?usage: train_lang_a.sbatch TASK LEVEL NDEMOS}"
LEVEL="${2:?}"
NDEMOS="${3:?}"

WORK="${COG_WORK:-$WORK/cog}"
RUN_ID="${COG_RUN_ID:-${TASK,,}_${LEVEL}_i20_n${NDEMOS}_s0}"
OUT="${WORK}/checkpoints/${RUN_ID}"
case "$TASK" in
  T1) DEFAULT_DS="${LEVEL}_i20" ;;
  T2|T3) DEFAULT_DS="${TASK}_${LEVEL}_i20" ;;
  *) echo "unknown TASK ${TASK}"; exit 2 ;;
esac
DS_NAME="${COG_DS_NAME:-${DEFAULT_DS}}"
DS="${COG_DATASETS:-$FAST/cog/datasets}/${DS_NAME}"

# Compute nodes have NO internet: both mandatory (see train.sbatch).
export WANDB_MODE=offline
export WANDB_DIR="${WORK}/wandb_offline"
export HF_HUB_OFFLINE=1
export HF_HOME="${WORK}/hf_cache"
export TOKENIZERS_PARALLELISM=false

source "${WORK}/miniforge3/etc/profile.d/conda.sh"
conda activate "${COG_TRAIN_ENV:-cog_lerobot}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
source /home/admin_07/cost_of_generality/configs/train/lang_diffusion_a.sh 2>/dev/null \
  || source "${WORK}/repo/configs/train/lang_diffusion_a.sh"

EPISODES="$(python - "$NDEMOS" <<'PY'
import sys
print(",".join(str(i) for i in range(int(sys.argv[1]))))
PY
)"

# Resume: config comes entirely from the checkpoint; the flags asserted are COG_DP_FLAGS,
# which lang_diffusion_a.sh leaves IDENTICAL to the frozen config, so the existing guard
# applies verbatim (input_features are dataset-driven and not asserted -- by design).
CKPT_CFG="${OUT}/checkpoints/last/pretrained_model/train_config.json"
if [ -f "${CKPT_CFG}" ]; then
  echo "[train] existing checkpoint found -> resuming ${RUN_ID} from ${CKPT_CFG}"
  ASSERT_PY="/home/admin_07/cost_of_generality/scripts/ops/assert_resume_config.py"
  [ -f "${ASSERT_PY}" ] || ASSERT_PY="${WORK}/repo/scripts/ops/assert_resume_config.py"
  if ! python "${ASSERT_PY}" "${CKPT_CFG}"; then
    echo "[train] ABORT ${RUN_ID}: checkpoint config != frozen config (see above)" >&2
    exit 2
  fi
  srun python -m lerobot.scripts.lerobot_train \
    --config_path="${CKPT_CFG}" \
    --resume=true
else
  mkdir -p "$(dirname "${OUT}")"
  echo "[train] run=${RUN_ID} dataset=${DS} episodes=0..$((NDEMOS-1)) batch=${COG_DP_BATCH} lr=${COG_DP_LR}"
  srun python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="local/${DS_NAME}" \
    --dataset.root="${DS}" \
    --dataset.episodes="[${EPISODES}]" \
    --dataset.video_backend="${COG_VIDEO_BACKEND:-pyav}" \
    --output_dir="${OUT}" \
    --job_name="${RUN_ID}" \
    --batch_size="${COG_DP_BATCH}" \
    --policy.optimizer_lr="${COG_DP_LR}" \
    --wandb.enable=true \
    --wandb.mode=offline \
    --wandb.project=cost_of_generality \
    --wandb.disable_artifact=true \
    ${COG_DP_FLAGS}
fi
echo "[train] DONE ${RUN_ID}"
