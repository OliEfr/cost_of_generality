#!/usr/bin/env bash
# D24 re-check on the FIXED harness: t1_L0_n25 @ 040000 and 060000 (080000 already in the sweep).
# Sequential, gated on >=10 GB free VRAM per eval; coexists with the phase-2 L3 drivers (3rd slot).
set -uo pipefail
REPO="/home/admin_07/cost_of_generality/.claude/worktrees/results-analysis"
PY="/home/admin_07/miniconda3/envs/cog_isaac/bin/python"
LOG="${REPO}/ops/d24_recheck.log"
export PATH="/home/admin_07/miniconda3/envs/cog_isaac/bin:${PATH}"
export OMNI_KIT_ACCEPT_EULA=YES HF_HUB_OFFLINE=1 PYTHONPATH="${REPO}/src"
cd "${REPO}"
say() { echo "[d24] $(date -Is) $*" | tee -a "${LOG}"; }

for step in 040000 060000; do
  CKPT="${REPO}/experiments/runs/t1_L0_n25_s0/checkpoints/${step}/pretrained_model"
  OUT="${REPO}/results/eval_T1_L0_n25_${step}_fixed.json"
  [ -s "${OUT}" ] && { say "D24_SKIP ${step} exists"; continue; }
  [ -d "${CKPT}" ] || { say "D24_MISSING ${CKPT}"; continue; }
  waited=0
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    [ $(( total - used )) -ge 10000 ] && break
    [ ${waited} -ge 720 ] && { say "D24_WAIT_TIMEOUT"; exit 3; }
    sleep 120; waited=$(( waited + 2 ))
  done
  say "START ${step}"
  timeout -s KILL 60m "${PY}" -m cog.eval.rollout_eval \
    --task Cog-CupPlace-L0-IK-Rel-Visuomotor-v0 --checkpoint "${CKPT}" \
    --num_inference_steps 10 --max_steps 600 --out "${OUT}" \
    --headless --enable_cameras < /dev/null >> "${LOG}" 2>&1
  rc=$?
  if [ -s "${OUT}" ]; then say "D24_EVAL_OK ${step} (rc=${rc})"; else say "D24_EVAL_FAILED ${step} (rc=${rc})"; fi
done
say "D24_DONE"
