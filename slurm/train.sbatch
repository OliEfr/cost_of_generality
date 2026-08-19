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
# conda does not add $CONDA_PREFIX/lib to the linker path, and torchcodec dlopen()s ffmpeg by
# soname -- without this it reports "FFmpeg is not properly installed" with ffmpeg sitting in
# the env (verified 2026-08-19). Harmless for the pyav backend, so it is set unconditionally.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
source /home/admin_07/cost_of_generality/configs/train/diffusion_base.sh 2>/dev/null \
  || source "${WORK}/repo/configs/train/diffusion_base.sh"

# The first N episodes of the committed shuffle order (seed 0) -- the nested-subset
# protocol (D4), so a larger cell is a strict superset of a smaller one.
EPISODES="$(python - "$NDEMOS" <<'PY'
import sys
print(",".join(str(i) for i in range(int(sys.argv[1]))))
PY
)"

# Four bugs fixed here on 2026-08-19 by reading the installed 0.4.4 source (see D22 and
# the journal entry of the same date). All four were silent or late-failing:
#   1. `python -m lerobot.scripts.train` does NOT exist at 0.4.4 -- the module is
#      lerobot.scripts.lerobot_train (console script `lerobot-train`).
#   2. `mkdir -p "${OUT}"` guaranteed a FileExistsError: configs/train.py:119 raises when
#      output_dir is an existing dir and --resume is not set. Only the PARENT may be made.
#   3. `--resume=true` alone raises "A config_path is expected when resuming a run"
#      (configs/train.py:89-95). Resume takes the config FROM the checkpoint, so the resume
#      invocation must be config_path-only -- other CLI flags are ignored there anyway.
#   4. `--optimizer.lr` is SILENTLY overwritten: with use_policy_training_preset=true
#      (the default), validate() replaces cfg.optimizer with policy.get_optimizer_preset()
#      (configs/train.py:134-136), which reads policy.optimizer_lr. So the real LR knob is
#      --policy.optimizer_lr. Passing --optimizer.lr would have trained every G5a-scaled
#      cell at the 1e-4 default with no warning at all.
# Plus: WandBConfig.mode defaults to None and rl/wandb_utils.py:109 then passes
# mode="online" EXPLICITLY to wandb.init, overriding the WANDB_MODE=offline env var set
# above. On an internet-less compute node that hangs. --wandb.mode=offline is mandatory.
CKPT_CFG="${OUT}/checkpoints/last/pretrained_model/train_config.json"
if [ -f "${CKPT_CFG}" ]; then
  echo "[train] existing checkpoint found -> resuming ${RUN_ID} from ${CKPT_CFG}"
  # A config_path-only resume takes the ENTIRE config from the checkpoint, so an edit to
  # configs/train/diffusion_base.sh made after this cell first ran would be silently ignored
  # here. Assert the checkpoint still matches the frozen config before spending GPU-h on it.
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
