#!/usr/bin/env bash
# T2 follow-up evals on the LOCAL 4090 (user-authorised 2026-08-21): the L2<->L1 cross-evals
# that split "cleaner demos" from "eval distribution", plus stage-instrumented re-runs of
# L0/L1/L2 at N=400 for the per-stage funnel (drawer opened -> lifted -> over drawer -> stowed).
#
# Frozen protocol untouched: same eval sets, same seeds, num_envs=20 x 5 (rule 8). Stage
# instrumentation only READS sim state (rollout_eval.py --stages). Checkpoints are read from the
# MAIN checkout (never written); code + results live in the results-analysis worktree.
#
# tmux checklist (docs/running_jobs.md): absolute interpreter, EULA env var, stdin from
# /dev/null, mkdir -p first, success MARKERS not exit codes, timeout -s KILL with exit 137
# treated as OK when the artifact exists (Kit shutdown hangs).
set -uo pipefail

REPO="/home/admin_07/cost_of_generality/.claude/worktrees/results-analysis"
CKPT_ROOT="/home/admin_07/cost_of_generality/experiments/runs"   # read-only
OUTDIR="${REPO}/results"
LOG="${REPO}/ops/t2_followup.log"
mkdir -p "${OUTDIR}" "${REPO}/ops"

# >=10 GB free admits one eval (~7 GB) with margin for Isaac's scene-load spike; threshold
# established 2026-08-20 when two concurrent evals were authorised. We run ONE at a time.
MIN_FREE_MIB="${COG_EVAL_MIN_FREE_MIB:-10000}"
MAX_WAIT_MIN="${COG_EVAL_MAX_WAIT_MIN:-720}"

export PATH="/home/admin_07/miniconda3/envs/cog_isaac/bin:${PATH}"
export OMNI_KIT_ACCEPT_EULA=YES
export HF_HUB_OFFLINE=1
export PYTHONPATH="${REPO}/src"
cd "${REPO}"

say() { echo "[t2fu] $(date -Is) $*" | tee -a "${LOG}"; }

wait_headroom() {
  local waited=0 used total free
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    free=$(( total - used ))
    if [ "${free}" -ge "${MIN_FREE_MIB}" ]; then say "headroom OK: ${free} MiB free"; return 0; fi
    if [ "${waited}" -ge "${MAX_WAIT_MIN}" ]; then say "T2FU_WAIT_TIMEOUT (${free} MiB free)"; return 1; fi
    if [ $(( waited % 30 )) -eq 0 ]; then say "waiting: ${free} MiB free, need ${MIN_FREE_MIB} (${waited} min)"; fi
    sleep 120; waited=$(( waited + 2 ))
  done
}

# ckpt_run_id  eval_level  out_basename   (all N=400, checkpoint 080000, --stages on)
QUEUE=(
  "t2_L2_n400_s0 L1 eval_T2_xeval_L2n400_onL1_080000.json"
  "t2_L1_n400_s0 L1 eval_T2_L1_n400_080000_stages.json"
  "t2_L2_n400_s0 L2 eval_T2_L2_n400_080000_stages.json"
  "t2_L1_n400_s0 L2 eval_T2_xeval_L1n400_onL2_080000.json"
  "t2_L0_n400_s0 L0 eval_T2_L0_n400_080000_stages.json"
)

say "T2FU_START queue=${#QUEUE[@]} evals"
for entry in "${QUEUE[@]}"; do
  read -r run_id level out_name <<< "${entry}"
  CKPT="${CKPT_ROOT}/${run_id}/checkpoints/080000/pretrained_model"
  OUT="${OUTDIR}/${out_name}"
  TASK="Cog-DrawerStow-${level}-IK-Rel-Visuomotor-v0"
  if [ ! -d "${CKPT}" ]; then say "T2FU_MISSING_CKPT ${CKPT}"; continue; fi
  if [ -s "${OUT}" ]; then say "T2FU_SKIP ${out_name} exists"; continue; fi
  wait_headroom || exit 3
  say "START ckpt=${run_id} task=${TASK} -> ${out_name}"
  timeout -s KILL 150m \
    /home/admin_07/miniconda3/envs/cog_isaac/bin/python -m cog.eval.rollout_eval \
      --task "${TASK}" \
      --checkpoint "${CKPT}" \
      --num_inference_steps 10 \
      --max_steps 1200 \
      --stages \
      --out "${OUT}" \
      --headless --enable_cameras < /dev/null >> "${LOG}" 2>&1
  rc=$?
  if [ -s "${OUT}" ]; then
    say "T2FU_EVAL_OK ${out_name} (rc=${rc})"
    /home/admin_07/miniconda3/envs/cog_isaac/bin/python - "${OUT}" <<'PY' | tee -a "${LOG}"
import json, sys
d = json.load(open(sys.argv[1]))
print(f"[t2fu]   -> SR={d['successes']}/{d['episodes']}={d['success_rate']:.3f} stages={d.get('stages')}")
PY
  else
    say "T2FU_EVAL_FAILED ${out_name} (rc=${rc}, no artifact)"
  fi
done
say "T2FU_DONE"
