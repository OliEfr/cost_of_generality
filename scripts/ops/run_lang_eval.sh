#!/usr/bin/env bash
# Language-conditioning investigation eval driver (clone of run_local_eval.sh).
# Writes to results/diagnostics/ ON PURPOSE: these are NOT study cells and must stay
# outside the eval_T*_L*_n*_*.json globs of update_registry_from_evals.py / curves.py.
#
#   scripts/ops/run_lang_eval.sh RUN_ID {t1|t2|t3} LEVEL OUT_BASENAME [STEP]
#
#   RUN_ID        checkpoint dir under ${REPO}/experiments/runs/ (e.g. t1_L1_i20_n100_s0)
#   {t1|t2|t3}    task family (probe run_ids don't encode it -- pass it explicitly)
#   LEVEL         gym level key (e.g. L1)
#   OUT_BASENAME  result filename inside results/diagnostics/
#   STEP          checkpoint step (default 080000; D24 = last checkpoint only)
#
# Env knobs:
#   COG_LANG_INSTRUCTIONS  instructions json (default configs/instructions/instructions_v1.json;
#                          set to "none" for a language-less regression eval)
#   COG_LANG_SWAP          probe mismatch: draw instructions from this task's set
#   COG_EVAL_MIN_FREE_MIB  VRAM admission threshold (default 10000, admits 2 concurrent evals)
#   COG_REPO               tree whose code+checkpoints run (default main checkout)
set -uo pipefail

REPO="${COG_REPO:-/home/admin_07/cost_of_generality}"
# NB: no braces in the :? message -- a '}' inside ${1:?...} terminates the expansion
# early and appends the rest of the message to the VALUE (cost: one failed eval launch).
RUN_ID="${1:?usage: run_lang_eval.sh RUN_ID t1|t2|t3 LEVEL OUT_BASENAME [STEP]}"
TAG="${2:?}"
LEVEL="${3:?}"
OUT_BASE="${4:?}"
STEP="${5:-080000}"

INSTR="${COG_LANG_INSTRUCTIONS:-${REPO}/configs/instructions/instructions_v1.json}"
SWAP="${COG_LANG_SWAP:-}"
MIN_FREE_MIB="${COG_EVAL_MIN_FREE_MIB:-10000}"
MAX_WAIT_MIN="${COG_EVAL_MAX_WAIT_MIN:-720}"

case "${TAG}" in
  t1) GYM_PREFIX="Cog-CupPlace";   MAX_STEPS=600  ;;
  t2) GYM_PREFIX="Cog-DrawerStow"; MAX_STEPS=1200 ;;
  t3) GYM_PREFIX="Cog-PushTarget"; MAX_STEPS=800  ;;
  *) echo "task family must be t1|t2|t3, got '${TAG}'" >&2; exit 2 ;;
esac
TASK="${GYM_PREFIX}-${LEVEL}-IK-Rel-Visuomotor-v0"
OUTDIR="${REPO}/results/diagnostics"; mkdir -p "${OUTDIR}"
OUT="${OUTDIR}/${OUT_BASE}"
LOG="${REPO}/ops/lang_eval_${OUT_BASE%.json}.log"; mkdir -p "${REPO}/ops"
CKPT="${REPO}/experiments/runs/${RUN_ID}/checkpoints/${STEP}/pretrained_model"

echo "[langeval] $(date -Is) run=${RUN_ID} task=${TASK} step=${STEP} out=${OUT}" | tee -a "${LOG}"
if [ ! -d "${CKPT}" ]; then echo "[langeval] MISSING ${CKPT}" | tee -a "${LOG}"; exit 2; fi
if [ -s "${OUT}" ]; then echo "[langeval] ${OUT} exists, refusing to overwrite" | tee -a "${LOG}"; exit 2; fi

waited=0
while :; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  free=$(( total - used ))
  if [ "${free}" -ge "${MIN_FREE_MIB}" ]; then
    echo "[langeval] $(date -Is) headroom OK: ${free} MiB free -> starting" | tee -a "${LOG}"; break
  fi
  if [ "${waited}" -ge "${MAX_WAIT_MIN}" ]; then
    echo "LANG_EVAL_WAIT_TIMEOUT" | tee -a "${LOG}"; exit 3
  fi
  [ $(( waited % 30 )) -eq 0 ] && echo "[langeval] waiting: ${free} MiB free, need ${MIN_FREE_MIB} (${waited} min)" | tee -a "${LOG}"
  sleep 120; waited=$(( waited + 2 ))
done

# shellcheck disable=SC1091
source /home/admin_07/miniconda3/etc/profile.d/conda.sh
conda activate cog_isaac
export OMNI_KIT_ACCEPT_EULA=YES
export HF_HUB_OFFLINE=1
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
cd "${REPO}"

INSTR_ARGS=()
if [ "${INSTR}" != "none" ]; then
  INSTR_ARGS+=(--instructions "${INSTR}")
  [ -n "${SWAP}" ] && INSTR_ARGS+=(--swap_instructions_from "${SWAP}")
fi

python -m cog.eval.rollout_eval \
  --task "${TASK}" \
  --checkpoint "${CKPT}" \
  --num_inference_steps 10 \
  --max_steps "${MAX_STEPS}" \
  --out "${OUT}" \
  "${INSTR_ARGS[@]}" \
  --headless --enable_cameras >> "${LOG}" 2>&1
rc=$?
# Kit exits 0 after fatal errors (D6): the artifact is the verdict, not $?.
if [ -s "${OUT}" ]; then
  echo "[langeval] $(date -Is) LANG_EVAL_OK ${OUT_BASE} (rc=${rc})" | tee -a "${LOG}"
  python - "${OUT}" <<'PY' | tee -a "${LOG}"
import json, sys
d = json.load(open(sys.argv[1]))
print(f"[langeval]   -> SR={d['success_rate']:.3f} over n={d['episodes']}")
PY
else
  echo "[langeval] $(date -Is) LANG_EVAL_FAILED ${OUT_BASE} (rc=${rc}, no artifact)" | tee -a "${LOG}"
fi
echo "LANG_EVAL_DONE ${OUT_BASE}" | tee -a "${LOG}"
