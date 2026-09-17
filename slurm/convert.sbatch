#!/bin/bash
#SBATCH --job-name=cog_convert
#SBATCH --account=euhpc_b38_106
#SBATCH --partition=dcgp_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# HDF5 -> LeRobotDataset for ONE arm.  sbatch slurm/convert.sbatch TASK ARM
#
# NO GPU. hdf5_to_lerobot imports h5py/numpy/lerobot and encodes h264 through LeRobot's PyAV path
# (lerobot/datasets/video_utils.py) -- no torch, no CUDA, no Kit, no Vulkan. So this runs on the CPU
# partition and costs no boost GPU-h. The container is still required: the cluster's cog_lerobot
# conda env has neither isaacsim nor h5py.
#
# If dcgp_usr_prod is not available to this allocation (probe it -- the account lists no dcgp_qos_*),
# resubmit with:  sbatch -p boost_usr_prod -q normal ...   (no --gres, so the node's GPUs stay
# schedulable for other jobs).
#
# Parallelism is one job per dataset. h264 encode is single-core, so splitting a dataset gains
# nothing while running the arms concurrently gains everything (running_jobs.md rule 7).
set -uo pipefail

TASK="${1:?usage: convert.sbatch TASK ARM}"
ARM="${2:?missing ARM}"

source "${WORK:-/leonardo_work/EUHPC_B38_106}/cog/repo/slurm/lib_cog_container.sh"
case "${TASK}" in
  T1) TASK_KIND=cup_place;   PREFIX=""    ;;
  T2) TASK_KIND=drawer_stow; PREFIX="T2_" ;;
  T3) TASK_KIND=push_target; PREFIX="T3_" ;;
  *) echo "CONVERT_FAILED unknown task ${TASK}"; exit 2 ;;
esac
KEY="${PREFIX}${ARM}"
ROOT="${DS_DIR}/${KEY}"

# Inputs listed EXPLICITLY. Never glob: RecorderManager writes <name>_failed.hdf5 beside every
# <name>.hdf5, and a glob once produced a 455-episode dataset that looked fine (running_jobs.md
# rule 6). The variant list is also what guarantees the merge stays variant-balanced, so a missing
# file must abort rather than silently produce a 9-variant dataset.
INPUTS=()
case "${ARM}" in
  AC|BC|L3|L3b)
    for i in $(seq -w 0 9); do INPUTS+=("${HDF5_DIR}/${KEY}v${i}.hdf5"); done ;;
  *)
    INPUTS+=("${HDF5_DIR}/${KEY}.hdf5") ;;
esac
cog_require "${SIF}" "${INPUTS[@]}" || { echo "CONVERT_FAILED ${KEY}: missing input"; exit 3; }

if [ -d "${ROOT}" ]; then
  # hdf5_to_lerobot refuses to overwrite an existing root, so this only makes the reason legible.
  echo "CONVERT_SKIP ${KEY} (dataset root already exists at ${ROOT})"; exit 0
fi

echo "=== node $(hostname)  job ${SLURM_JOB_ID} ==="
echo "[convert] ${KEY} <- ${#INPUTS[@]} hdf5 file(s) -> ${ROOT}"
T0=$(date +%s)
cog_cpu_exec python -u -m cog.convert.hdf5_to_lerobot \
    --task "${TASK_KIND}" --input "${INPUTS[@]}" \
    --root "${ROOT}" --repo_id "local/${KEY}" --fps 20
CONVERT_RC=$?
echo "CONVERT_${KEY}_EXIT=${CONVERT_RC}"

cog_cpu_exec python -u -m cog.convert.validate_dataset \
    --root "${ROOT}" --repo_id "local/${KEY}" --expect_episodes 400
VALIDATE_RC=$?
echo "VALIDATE_${KEY}_EXIT=${VALIDATE_RC}"
echo "CONVERT_WALL_SECONDS=$(( $(date +%s) - T0 ))"

if [ "${CONVERT_RC}" -eq 0 ] && [ "${VALIDATE_RC}" -eq 0 ]; then
  echo "VALIDATE_OK ${KEY}"
else
  echo "CONVERT_FAILED ${KEY} convert=${CONVERT_RC} validate=${VALIDATE_RC}"
  exit 1
fi
