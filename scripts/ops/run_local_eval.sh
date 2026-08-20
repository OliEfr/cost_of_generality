#!/usr/bin/env bash
# Evaluate a cell's checkpoints on the LOCAL 4090, waiting for GPU headroom first.
#
#   scripts/ops/run_local_eval.sh t1_L0_n25_s0 L0 25 040000 060000 080000
#
# Why the wait: a foreign eval job (lp-eval) shares this GPU and must never be disturbed
# (CLAUDE.md rule 2). The frozen protocol is num_envs=20 x 5 batches = 100 episodes and num_envs
# comes from configs/eval_sets/protocol.json, NOT a flag -- changing it would change which seeds
# define the benchmark (rule 8), so our footprint is not adjustable. Hence: wait for room rather
# than shrink. User approved waiting (2026-08-19).
#
# Launch under tmux (rule 10). A tmux shell has no conda, so conda is sourced explicitly here.
set -uo pipefail

# Overridable so a worktree-isolated session evaluates the checkpoints IT pulled and writes its
# results into its own tree (see the same change in sync_up.sh / sync_down.sh). The cog package is
# imported from ${REPO}, so this also decides which code runs the rollout.
REPO="${COG_REPO:-/home/admin_07/cost_of_generality}"
RUN_ID="${1:?usage: run_local_eval.sh RUN_ID LEVEL NDEMOS [STEP...]}"
LEVEL="${2:?}"
NDEMOS="${3:?}"
shift 3
STEPS=("$@"); [ ${#STEPS[@]} -eq 0 ] && STEPS=(080000)

MIN_FREE_MIB="${COG_EVAL_MIN_FREE_MIB:-14000}"
MAX_WAIT_MIN="${COG_EVAL_MAX_WAIT_MIN:-720}"
# RUN_ID already encodes the task (t1_/t2_/t3_), so derive everything from it rather than adding a
# parameter that a caller could pass inconsistently with the checkpoint path.
#
# MAX_STEPS matters and is NOT cosmetic: each task's env cfg sets its own episode_length_s at 20 Hz
# (cup_place 30 s = 600 steps, drawer_stow 60 s = 1200, push_target 40 s = 800), and the generated
# demos run 186-219 / 675-743 / 307-399 steps respectively. rollout_eval defaults to --max_steps 600,
# which happens to match T1 exactly -- but would truncate EVERY drawer_stow episode before the task
# can finish and report a near-zero success rate that looks like "T2 is very hard" rather than a
# harness error.
TAG="${RUN_ID%%_*}"
case "${TAG}" in
  t1) GYM_PREFIX="Cog-CupPlace";   MAX_STEPS=600  ;;
  t2) GYM_PREFIX="Cog-DrawerStow"; MAX_STEPS=1200 ;;
  t3) GYM_PREFIX="Cog-PushTarget"; MAX_STEPS=800  ;;
  *) echo "cannot derive task from RUN_ID '${RUN_ID}' (expected t1_/t2_/t3_ prefix)" >&2; exit 2 ;;
esac
TASK="${GYM_PREFIX}-${LEVEL}-IK-Rel-Visuomotor-v0"
OUTDIR="${REPO}/results"; mkdir -p "${OUTDIR}"
LOG="${REPO}/ops/local_eval_${RUN_ID}.log"; mkdir -p "${REPO}/ops"

echo "[eval] $(date -Is) run=${RUN_ID} task=${TASK} steps=${STEPS[*]}" | tee -a "${LOG}"
echo "[eval] waiting for >= ${MIN_FREE_MIB} MiB free VRAM (max ${MAX_WAIT_MIN} min)" | tee -a "${LOG}"

waited=0
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  free=$(( total - used ))
  if [ "${free}" -ge "${MIN_FREE_MIB}" ]; then
    echo "[eval] $(date -Is) headroom OK: ${free} MiB free -> starting" | tee -a "${LOG}"
    break
  fi
  if [ "${waited}" -ge "${MAX_WAIT_MIN}" ]; then
    echo "[eval] GAVE UP after ${MAX_WAIT_MIN} min; only ${free} MiB free" | tee -a "${LOG}"
    echo "LOCAL_EVAL_WAIT_TIMEOUT" | tee -a "${LOG}"; exit 3
  fi
  # every 30 min, say something so a human reading the log knows it is alive and why
  if [ $(( waited % 30 )) -eq 0 ]; then
    echo "[eval] $(date -Is) still waiting: ${free} MiB free, need ${MIN_FREE_MIB} (${waited} min)" | tee -a "${LOG}"
  fi
  sleep 120; waited=$(( waited + 2 ))
done

# shellcheck disable=SC1091
source /home/admin_07/miniconda3/etc/profile.d/conda.sh
conda activate cog_isaac
export OMNI_KIT_ACCEPT_EULA=YES
export HF_HUB_OFFLINE=1
# `cog` is an editable install pointing at the MAIN checkout, and this repo uses a src/ layout, so
# `cd ${REPO}` alone does NOT make this tree's code win. Without this, a COG_REPO-directed run reads
# this tree's checkpoints but executes the main checkout's cog package.
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
cd "${REPO}"

for step in "${STEPS[@]}"; do
  CKPT="${REPO}/experiments/runs/${RUN_ID}/checkpoints/${step}/pretrained_model"
  # The task MUST be in the filename. Without it, T2's L0/N=25 result is written to exactly the
  # path T1's already occupies -- and since this script skips a cell whose output exists, T2 would
  # have silently inherited T1's success rates with no error anywhere. Caught before T2 ran; the 24
  # T1 files were migrated to this scheme at the same time.
  OUT="${OUTDIR}/eval_${TAG^^}_${LEVEL}_n${NDEMOS}_${step}.json"
  if [ ! -d "${CKPT}" ]; then echo "[eval] MISSING ${CKPT}, skipping" | tee -a "${LOG}"; continue; fi
  if [ -s "${OUT}" ]; then echo "[eval] ${OUT} exists, skipping" | tee -a "${LOG}"; continue; fi
  echo "[eval] $(date -Is) START step ${step}" | tee -a "${LOG}"
  python -m cog.eval.rollout_eval \
    --task "${TASK}" \
    --checkpoint "${CKPT}" \
    --num_inference_steps 10 \
    --max_steps "${MAX_STEPS}" \
    --out "${OUT}" \
    --headless --enable_cameras >> "${LOG}" 2>&1
  rc=$?
  # Kit exits 0 after fatal errors (D6): the artifact is the verdict, not $?.
  if [ -s "${OUT}" ]; then
    echo "[eval] $(date -Is) EVAL_OK ${LEVEL} n${NDEMOS} ${step} (rc=${rc})" | tee -a "${LOG}"
    python - "${OUT}" <<'PY' | tee -a "${LOG}"
import json, sys
d = json.load(open(sys.argv[1]))
sr = d.get("success_rate", d.get("sr"))
n = d.get("n_episodes", d.get("num_episodes"))
print(f"[eval]   -> SR={sr} over n={n}")
PY
  else
    echo "[eval] $(date -Is) EVAL_FAILED ${LEVEL} n${NDEMOS} ${step} (rc=${rc}, no artifact)" | tee -a "${LOG}"
  fi
done
echo "LOCAL_EVAL_DONE ${RUN_ID}" | tee -a "${LOG}"
