#!/bin/bash
#SBATCH --job-name=cog_wprobe
#SBATCH --account=euhpc_b38_106
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# Render-vs-physics probe for the first-batch depression. See scripts/dev/warmup_frame_probe.py.
#   sbatch slurm/warmup_probe.sbatch [TASK_ENV_ID]
set -uo pipefail
ENVID="${1:-Cog-CupPlace-L1-IK-Rel-Visuomotor-v0}"
source "${WORK:-/leonardo_work/EUHPC_B38_106}/cog/repo/slurm/lib_cog_container.sh"
cog_job_scratch || exit 2
export COG_TIMEOUT=$(cog_timeout_s)
OUT="${WORK}/cog/results/_probes/warmup_frames_$(echo "${ENVID}" | tr -d '-').npz"
echo "=== node $(hostname)  job ${SLURM_JOB_ID} ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
T0=$(date +%s)
cog_gpu_exec python -u -m scripts.dev.warmup_frame_probe \
    --task "${ENVID}" --out "${OUT}" --num_envs 20 --seed 5000 --cycles 3 --steps 30 \
    --headless --enable_cameras "--kit_args=\"${COG_KIT_ARGS}\""
RC=$?
echo "PROBE_WALL_SECONDS=$(( $(date +%s) - T0 )) rc=${RC}"
[ -s "${OUT}" ] && echo "PROBE_ARTIFACT_OK ${OUT}" || { echo "PROBE_FAILED rc=${RC}"; exit 1; }
