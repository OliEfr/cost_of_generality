#!/bin/bash
#SBATCH --job-name=cog_train_mtdit
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
# Candidate B (language-conditioned) cell: train the backported multi_task_dit plugin on
# (TASK, LEVEL, NDEMOS). Sibling of slurm/train.sbatch (which stays frozen for the
# diffusion cells); differences: sources configs/train/lang_dit_b.sh, RUN_ID gets a
# _mtdit suffix, PYTHONPATH exposes the in-repo plugin, the plugin discover flag is on
# BOTH invocations, and the resume guard is the mtdit-specific one.
#   sbatch slurm/train_lang_dit.sbatch T1 L1_i20 100
# TASK is the dataset prefix: T1 -> bare level names, T2/T3 -> prefixed.
#
# 24 h walltime: if throughput lands near the ~1 steps/s dataloader-bound figure of the
# diffusion cells, 80k steps ~ 23 h; the resume path below covers an overrun.
#
# Prerequisites on the cluster (see plan, candidate B provisioning):
#   - transformers==4.57.6 installed in cog_lerobot (login node, then PINS.md row)
#   - CLIP ViT-B/16 staged into $WORK/cog/hf_cache/hub (scripts/ops/sync_up.sh hf)
#   - offline dress rehearsal on the login node with HF_HUB_OFFLINE=1 before spending a job
set -euo pipefail

TASK="${1:?usage: train_lang_dit.sbatch TASK LEVEL NDEMOS}"
LEVEL="${2:?}"
NDEMOS="${3:?}"

WORK="${COG_WORK:-$WORK/cog}"
RUN_ID="${TASK,,}_${LEVEL}_n${NDEMOS}_s0_mtdit"
OUT="${WORK}/checkpoints/${RUN_ID}"
case "$TASK" in
  T1) DS_NAME="${LEVEL}" ;;                 # T1 datasets are bare: L0..L3, L1_i20, ...
  T2|T3) DS_NAME="${TASK}_${LEVEL}" ;;
  *) echo "unknown TASK ${TASK}"; exit 2 ;;
esac
DS="${COG_DATASETS:-$FAST/cog/datasets}/${DS_NAME}"

# Compute nodes have NO internet. Both of these are mandatory: a single online call
# blocks the whole job until it times out. HF_HUB_OFFLINE also makes the CLIP text/vision
# towers and the tokenizer load from $WORK/cog/hf_cache instead of trying the Hub.
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
source /home/admin_07/cost_of_generality/configs/train/lang_dit_b.sh 2>/dev/null \
  || source "${WORK}/repo/configs/train/lang_dit_b.sh"

# The in-repo plugin package (src/lerobot_policy_mtdit). Both the PYTHONPATH and the
# --policy.discover_packages_path flag are needed on EVERY invocation, including resume:
# the discover flag is consumed by lerobot's parser wrapper before config parsing, so it
# is legal (and required) alongside a config_path-only resume.
export PYTHONPATH="${WORK}/repo/src:${PYTHONPATH:-}"

# The first N episodes of the committed shuffle order (seed 0) -- the nested-subset
# protocol (D4), so a larger cell is a strict superset of a smaller one.
EPISODES="$(python - "$NDEMOS" <<'PY'
import sys
print(",".join(str(i) for i in range(int(sys.argv[1]))))
PY
)"

# Same four 0.4.4 gotchas as train.sbatch (module name, no mkdir of OUT, config_path-only
# resume, --policy.optimizer_lr is the real LR knob) -- see train.sbatch and D22.
CKPT_CFG="${OUT}/checkpoints/last/pretrained_model/train_config.json"
if [ -f "${CKPT_CFG}" ]; then
  echo "[train_mtdit] existing checkpoint found -> resuming ${RUN_ID} from ${CKPT_CFG}"
  # A config_path-only resume takes the ENTIRE config from the checkpoint; assert it still
  # matches lang_dit_b.sh first. NOT assert_resume_config.py -- that one is bound to the
  # diffusion cells' COG_DP_FLAGS and cannot vouch for a mtdit checkpoint.
  ASSERT_PY="/home/admin_07/cost_of_generality/scripts/ops/assert_resume_config_mtdit.py"
  [ -f "${ASSERT_PY}" ] || ASSERT_PY="${WORK}/repo/scripts/ops/assert_resume_config_mtdit.py"
  if ! python "${ASSERT_PY}" "${CKPT_CFG}"; then
    echo "[train_mtdit] ABORT ${RUN_ID}: checkpoint config != lang_dit_b.sh (see above)" >&2
    exit 2
  fi
  srun python -m lerobot.scripts.lerobot_train \
    --policy.discover_packages_path=lerobot_policy_mtdit \
    --config_path="${CKPT_CFG}" \
    --resume=true
else
  mkdir -p "$(dirname "${OUT}")"
  echo "[train_mtdit] run=${RUN_ID} dataset=${DS} episodes=0..$((NDEMOS-1)) batch=${COG_DIT_BATCH}"
  srun python -m lerobot.scripts.lerobot_train \
    --policy.discover_packages_path=lerobot_policy_mtdit \
    --dataset.repo_id="local/${DS_NAME}" \
    --dataset.root="${DS}" \
    --dataset.episodes="[${EPISODES}]" \
    --dataset.video_backend="${COG_VIDEO_BACKEND:-pyav}" \
    --output_dir="${OUT}" \
    --job_name="${RUN_ID}" \
    --batch_size="${COG_DIT_BATCH}" \
    --wandb.enable=true \
    --wandb.mode=offline \
    --wandb.project=cost_of_generality \
    --wandb.disable_artifact=true \
    ${COG_DIT_FLAGS}
fi
echo "[train_mtdit] DONE ${RUN_ID}"
