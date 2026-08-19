#!/bin/bash
#SBATCH --job-name=cog_smoke_resume
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
# G5a part 3: prove resume-across-requeue works, in minutes instead of over a 24 h walltime.
#   sbatch slurm/smoke_resume.sbatch
#
# This exercises the exact code path a requeued 80k job takes, which is worth testing on its
# own merits because that path was BROKEN until today: train.sbatch passed a bare
# `--resume=true`, and configs/train.py:89-95 raises "A config_path is expected when resuming a
# run". A node failure at hour 11 would have turned into a job that could never restart.
#
# Method: train 400 steps but kill it at ~250 (SIGKILL, like a walltime kill), leaving a
# checkpoint at step 200; then resume and require that it (a) starts from 200, not 0, and
# (b) finishes at 400.
set -euo pipefail

COG="${WORK}/cog"
RES="${COG}/results"; mkdir -p "${RES}"
RUN_ID="_smoke_resume_${SLURM_JOB_ID}"
OUT="${COG}/checkpoints/${RUN_ID}"
DS_NAME="${COG_SMOKE_DS:-L0}"
DS="${COG_DATASETS:-$FAST/cog/datasets}/${DS_NAME}"
LOG1="${RES}/g5a_resume_phase1_${SLURM_JOB_ID}.log"
LOG2="${RES}/g5a_resume_phase2_${SLURM_JOB_ID}.log"

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HF_HOME="${COG}/hf_cache"
export TOKENIZERS_PARALLELISM=false

source "${COG}/miniforge3/etc/profile.d/conda.sh"
conda activate "${COG_TRAIN_ENV:-cog_lerobot}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
source "${COG}/repo/configs/train/diffusion_base.sh"

# Strip everything this test overrides. num_workers is NOT overridden here, but steps/save_freq
# are, and draccus is last-wins -- the same trap that invalidated the num_workers measurement.
FLAGS="$(printf '%s\n' ${COG_DP_FLAGS} \
  | grep -v -E '^--steps=|^--save_freq=|^--log_freq=' | tr '\n' ' ')"
EPISODES="$(python -c 'print(",".join(str(i) for i in range(25)))')"

echo "=== phase 1: train to 400, killed at ~250 (simulated walltime kill) ==="
set +e
timeout -s KILL 420 python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="local/${DS_NAME}" \
  --dataset.root="${DS}" \
  --dataset.episodes="[${EPISODES}]" \
  --dataset.video_backend="${COG_VIDEO_BACKEND:-pyav}" \
  --output_dir="${OUT}" \
  --job_name="${RUN_ID}" \
  --batch_size="${COG_DP_BATCH}" \
  --policy.optimizer_lr="${COG_DP_LR}" \
  --steps=400 --save_freq=100 --log_freq=25 \
  --wandb.enable=false \
  ${FLAGS} > "${LOG1}" 2>&1 &
TRAIN_PID=$!
# Kill once a checkpoint at step 200 exists -- deterministic, unlike sleeping a fixed time.
for _ in $(seq 1 120); do
  if [ -d "${OUT}/checkpoints/000000200" ]; then
    echo "[resume] checkpoint 000000200 present -> killing phase 1 (pid ${TRAIN_PID})"
    kill -9 "${TRAIN_PID}" 2>/dev/null
    break
  fi
  kill -0 "${TRAIN_PID}" 2>/dev/null || { echo "[resume] phase 1 exited on its own"; break; }
  sleep 5
done
wait "${TRAIN_PID}" 2>/dev/null
set -e

echo "[resume] checkpoints after phase 1:"; ls "${OUT}/checkpoints/" 2>&1
LAST_STEP1="$(python - "${OUT}" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "checkpoints" / "last" / "training_state" / "training_step.json"
print(json.load(open(p))["step"] if p.exists() else -1)
PY
)"
echo "[resume] step recorded in checkpoints/last = ${LAST_STEP1}"
if [ "${LAST_STEP1}" -lt 100 ]; then
  echo "G5A_RESUME_FAILED: phase 1 never produced a usable checkpoint"; exit 1
fi

echo "=== phase 2: resume via config_path (the path train.sbatch now takes) ==="
CKPT_CFG="${OUT}/checkpoints/last/pretrained_model/train_config.json"
ls -l "${CKPT_CFG}"
set +e
python -m lerobot.scripts.lerobot_train --config_path="${CKPT_CFG}" --resume=true \
  > "${LOG2}" 2>&1
RC2=$?
set -e
echo "[resume] phase 2 rc=${RC2}"

# Assert on the ARTIFACT and on the LOG's first step, not on the exit code: a resume that
# silently restarted from 0 would also exit 0 and would quietly discard 200 steps of work.
FIRST_STEP2="$(grep -oE 'step:[0-9]+' "${LOG2}" | head -1 | cut -d: -f2 || true)"
LAST_STEP2="$(python - "${OUT}" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]) / "checkpoints" / "last" / "training_state" / "training_step.json"
print(json.load(open(p))["step"] if p.exists() else -1)
PY
)"
echo "[resume] phase 2 first logged step=${FIRST_STEP2:-none}, final checkpoint step=${LAST_STEP2}"
tail -5 "${LOG2}"

if [ "${RC2}" -ne 0 ]; then
  echo "G5A_RESUME_FAILED: phase 2 exited ${RC2}"; sed -n '1,40p' "${LOG2}"; exit 1
fi
if [ "${LAST_STEP2}" -ne 400 ]; then
  echo "G5A_RESUME_FAILED: expected final step 400, got ${LAST_STEP2}"; exit 1
fi
if [ -z "${FIRST_STEP2}" ] || [ "${FIRST_STEP2}" -le "${LAST_STEP1}" ]; then
  echo "G5A_RESUME_FAILED: resumed run's first logged step (${FIRST_STEP2:-none}) is not"
  echo "  beyond the checkpoint step (${LAST_STEP1}) -- it restarted instead of resuming"
  exit 1
fi

echo "G5A_RESUME_PASSED  (killed at ${LAST_STEP1}, resumed at ${FIRST_STEP2}, finished at ${LAST_STEP2})"
rm -rf "${OUT}"
