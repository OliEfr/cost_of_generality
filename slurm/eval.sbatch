#!/bin/bash
#SBATCH --job-name=cog_eval
#SBATCH --account=euhpc_b38_106
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# ONE SLICE of one eval cell, under the D31 uniform-slice protocol.
#
#   sbatch slurm/eval.sbatch TASK ARM NDEMOS SLICE [STEP]
#   sbatch slurm/eval.sbatch T1 L2 100 3          # flat arm, batch 3
#   sbatch slurm/eval.sbatch T1 BC 100 3          # variant arm: env BCv03 on batch 3 (D18 diagonal)
#
# A cell is TEN slices: slice s runs in its own process, executes one unscored warm-up batch, then
# scores batch s (reset seed 5000+s, 20 envs). Ten slices = 200 scored episodes. Flat and variant
# arms differ only in whether the env id carries v0<SLICE>, which is why this is one code path.
#
# Why slices at all: the first batch in a process is genuinely depressed (t1_L1_n100_s0 scores
# [0.50, 0.90, 0.95, 0.95, 1.00] across its five batches WITH the t==0 phantom guard active, journal
# 2026-08-22). The old protocol ran flat cells as 5 batches in one process but variant cells as one
# batch per process, so only the variant arms paid that cost in full -- which is the confound that
# broke Finding 4. Here every scored batch sits in the same position of an identically warmed
# process, in every arm.
#
# Rule 8 is untouched: the frozen snapshots already contain batches 0-9 for every level. What
# changes is which committed rows are read and how they are grouped into processes -- the same kind
# of change D18 itself was.
#
# This file was rewritten from slurm/bench_eval_a100_dbg.sbatch (the only verified container
# recipe). The pre-FoldSpace version is kept as slurm/eval_legacy_singularity.sbatch; it uses plain
# singularity and an inner srun that silently drops the GPU, so do not copy from it.
set -uo pipefail

TASK="${1:?usage: eval.sbatch TASK ARM NDEMOS SLICE [STEP]}"
ARM="${2:?missing ARM}"
NDEMOS="${3:?missing NDEMOS}"
SLICE="${4:?missing SLICE (0-9)}"
STEP="${5:-080000}"
[ "${SLICE}" -ge 0 ] && [ "${SLICE}" -le 9 ] || { echo "EVAL_FAILED: SLICE must be 0-9"; exit 2; }

source "${WORK:-/leonardo_work/EUHPC_B38_106}/cog/repo/slurm/lib_cog_container.sh"
cog_task_spec "${TASK}" || exit 2

# Warm-up knobs. One full batch is the conservative default; gate G6 calibrates whether a shorter
# cap suffices, and whatever it decides is passed in by the launcher so the whole sweep shares one
# setting. Every result records what it ran with, so two protocols can never be pooled by accident.
WARMUP_BATCHES="${COG_WARMUP_BATCHES:-1}"
WARMUP_MAX_STEPS="${COG_WARMUP_MAX_STEPS:-0}"
WARMUP_RENDERS="${COG_WARMUP_RENDERS:-0}"
SUFFIX="${COG_EVAL_SUFFIX:-u200}"

RUN_ID="${TASK,,}_${ARM}_n${NDEMOS}_s0"
CKPT="${COG}/checkpoints/${RUN_ID}/checkpoints/${STEP}/pretrained_model"
PART="${RESULTS_DIR}/_partials"
mkdir -p "${PART}"

# The ENV arm and the CHECKPOINT arm are usually the same, but not always: the G2b control trains
# on a cluster-generated dataset called ctl_L1 and must be evaluated in the ordinary L1 env, which
# is the whole point of the control. COG_EVAL_ENV_ARM overrides only the env id.
ENV_ARM="${COG_EVAL_ENV_ARM:-${ARM}}"

# The checkpoint stem and the gym key are NOT the same token for L3b. D27 regenerated L3 under
# per-variant seeds and D29 renamed the ARTIFACTS -- datasets, run_ids, result files -- to L3b, but
# the gym ids were never renamed: levels.py registers L3v00..L3v09 in all three tasks and no L3bv*
# exists anywhere. So `eval.sbatch T2 L3b 400 0` must load checkpoint t2_L3b_n400_s0 and env L3v00.
# Mapped here rather than at each call site: a convention that lives only in the caller is precisely
# what D29 says drifts, and this one fails as a gym registration error ten slices at a time.
[ "${ENV_ARM}" = "L3b" ] && ENV_ARM="L3"
case "${ENV_ARM}" in
  AC|BC|L3) VKEY=$(printf "%sv%02d" "${ENV_ARM}" "${SLICE}") ;;
  *)        VKEY="${ENV_ARM}" ;;
esac
ENVID="${GYM}-${VKEY}-IK-Rel-Visuomotor-v0"
OUT="${PART}/eval_${TASK}_${ARM}_n${NDEMOS}_${STEP}_${SUFFIX}_s${SLICE}.json"

if [ -s "${OUT}" ]; then echo "EVAL_SKIP $(basename "${OUT}")"; exit 0; fi
cog_require "${SIF}" "${CKPT}" "${ASSET_ROOT}" || { echo "EVAL_FAILED ${RUN_ID} s${SLICE}: missing prerequisite"; exit 1; }

cog_job_scratch || exit 1
export COG_TIMEOUT=$(cog_timeout_s)

# One scored batch, seeded 5000+SLICE. Written per job rather than shared, so two concurrent slices
# can never read a half-written protocol file.
PROTO="${COG_SCRATCH}/proto_seed$(( 5000 + SLICE )).json"
cat > "${PROTO}" <<JSON
{"num_envs": 20, "batches": 1, "base_seed": $(( 5000 + SLICE )),
 "comment": "D31 uniform slice ${SLICE}/9 of ${ARM}: env ${VKEY} scored on batch ${SLICE} only. For a variant arm this is D18's diagonal (variant v <- batch v); for a flat arm it is the same batch read from the same frozen snapshot. Ten slices pool to 200 episodes."}
JSON

echo "=== node $(hostname)  job ${SLURM_JOB_ID} ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "[eval] ${RUN_ID} step=${STEP} slice=${SLICE} env=${ENVID} warmup=${WARMUP_BATCHES}"

T0=$(date +%s)
cog_gpu_exec python -u -m cog.eval.rollout_eval \
    --task "${ENVID}" --checkpoint "${CKPT}" --protocol "${PROTO}" \
    --num_inference_steps 10 --max_steps "${MAX_STEPS}" ${STAGES} \
    --warmup_batches "${WARMUP_BATCHES}" --warmup_max_steps "${WARMUP_MAX_STEPS}" \
    --warmup_renders "${WARMUP_RENDERS}" \
    --out "${OUT}" --headless --enable_cameras \
    "--kit_args=\"${COG_KIT_ARGS}\""
RC=$?
echo "EVAL_WALL_SECONDS=$(( $(date +%s) - T0 )) rc=${RC}"

# The artifact is the verdict (Kit exits 0 after fatal errors and usually hangs on shutdown).
if [ -s "${OUT}" ]; then
  echo "EVAL_OK $(basename "${OUT}")"
else
  echo "EVAL_FAILED ${RUN_ID} s${SLICE} rc=${RC}"
  exit 1
fi
