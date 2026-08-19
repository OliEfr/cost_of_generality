#!/bin/bash
#SBATCH --job-name=cog_eval
#SBATCH --account=euhpc_b38_106
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# Evaluate a cell's checkpoint(s) on the FROZEN eval set. Default = the LAST checkpoint only
# (80k), per the 2026-08-19 user directive; pass explicit step names for the one-time
# multi-checkpoint comparison. See D24.
#   sbatch slurm/eval.sbatch T1 L0 100
#
# ONLY valid if gate G5b passed (Isaac renders on A100). If it failed, run the same loop
# locally on the 4090 instead -- the sbatch header is the only cluster-specific part.
#
# UNVERIFIED until G0 clears and G5b passes.
set -uo pipefail

TASK="${1:?usage: eval.sbatch TASK LEVEL NDEMOS [STEP...]}"
LEVEL="${2:?}"
NDEMOS="${3:?}"
shift 3 || true
# User directive 2026-08-19: full-scale cells evaluate the LAST checkpoint only; the
# 40k/60k/80k comparison is run ONCE, on one cell, to check that 80k is not systematically
# worse than an earlier checkpoint (D24). So the default is 080000 and the comparison is
# requested explicitly:
#   sbatch slurm/eval.sbatch T1 L0 100                        # full-scale: 80k only
#   sbatch slurm/eval.sbatch T1 L0 25 040000 060000 080000    # the one-time comparison
STEPS_TO_EVAL=("$@")
if [ ${#STEPS_TO_EVAL[@]} -eq 0 ]; then STEPS_TO_EVAL=(080000); fi
echo "[eval] checkpoints to evaluate: ${STEPS_TO_EVAL[*]}"

WORK="${COG_WORK:-$WORK/cog}"
RUN_ID="${TASK,,}_${LEVEL}_n${NDEMOS}_s0"
CKPT_DIR="${WORK}/checkpoints/${RUN_ID}/checkpoints"
RESULTS="${WORK}/results"
case "$TASK" in
  T1) ENV_FAMILY="Cog-CupPlace";   EVAL_KEY="${LEVEL}" ;;
  T2) ENV_FAMILY="Cog-DrawerStow"; EVAL_KEY="T2_${LEVEL}" ;;
  T3) ENV_FAMILY="Cog-PushTarget"; EVAL_KEY="T3_${LEVEL}" ;;
  *) echo "unknown TASK ${TASK}"; exit 2 ;;
esac

export OMNI_KIT_ACCEPT_EULA=YES     # Kit blocks forever on its EULA prompt under a TTY
export HF_HUB_OFFLINE=1
export HF_HOME="${WORK}/hf_cache"
mkdir -p "${RESULTS}"

source "${WORK}/miniforge3/etc/profile.d/conda.sh"
conda activate "${COG_EVAL_ENV:-cog_isaac}"

for step in "${STEPS_TO_EVAL[@]}"; do
  CKPT="${CKPT_DIR}/${step}/pretrained_model"
  if [ ! -d "${CKPT}" ]; then echo "[eval] missing ${CKPT}, skipping"; continue; fi
  OUT="${RESULTS}/eval_${EVAL_KEY}_n${NDEMOS}_${step}.json"
  if [ -s "${OUT}" ]; then echo "[eval] ${OUT} exists, skipping"; continue; fi
  echo "[eval] ${RUN_ID} step ${step}"
  srun python -m cog.eval.rollout_eval \
    --task "${ENV_FAMILY}-${LEVEL}-IK-Rel-Visuomotor-v0" \
    --checkpoint "${CKPT}" \
    --num_inference_steps 10 \
    --out "${OUT}" \
    --headless --enable_cameras < /dev/null
  # Kit can exit 0 after a fatal exception, so the artifact is the signal, not $?.
  if [ -s "${OUT}" ]; then echo "EVAL_OK ${EVAL_KEY} n${NDEMOS} ${step}"
  else echo "EVAL_FAILED ${EVAL_KEY} n${NDEMOS} ${step} (no result file)"; fi
done
echo "[eval] DONE ${RUN_ID}"
