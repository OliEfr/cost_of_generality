#!/bin/bash
#SBATCH --job-name=cog_build_sif
#SBATCH --account=euhpc_b38_106
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# Convert a docker archive to a Singularity image, ON A COMPUTE NODE.
#   sbatch slurm/build_sif.sbatch isaac-sim-5.1.0.tar isaac-sim-5.1.0.sif
#
# Why not on the login node: the first attempt there failed. `mksquashfs` wants real RAM and
# fast local disk, and a login node gives neither -- it is shared by ~100 users, its memory is
# capped, and $SINGULARITY_TMPDIR on Lustre made mksquashfs emit an endless stream of
# "Unrecognised xattr prefix lustre.lov" while unpacking ~17 GB of small files. A compute node
# has 120 GB of RAM and node-local scratch. No fakeroot is needed: converting a docker archive
# is a pure format conversion, not a %post build (fakeroot is unmapped for this account anyway).
set -uo pipefail

TAR="${1:?usage: build_sif.sbatch <archive.tar> <out.sif>}"
SIF_NAME="${2:?}"
C="${WORK}/cog/containers"
LOG="${WORK}/cog/logs/build_sif_${SLURM_JOB_ID}.log"

echo "=== node $(hostname) ==="
echo "TMPDIR=${TMPDIR:-unset}"
df -h "${TMPDIR:-/tmp}" /tmp 2>&1 | tail -3
free -g | head -2

# Prefer node-local scratch for the unpack: it avoids Lustre xattrs entirely and is far faster
# for the hundreds of thousands of small files in an Isaac image. Fall back to $WORK only if
# the local filesystem cannot hold ~2.5x the archive size.
NEED_GB=$(( $(stat -c%s "${C}/${TAR}") / 1073741824 * 5 / 2 ))
LOCAL_DIR="${TMPDIR:-/tmp}/sif_build_$$"
AVAIL_GB=$(df -BG --output=avail "${TMPDIR:-/tmp}" | tail -1 | tr -dc '0-9')
echo "archive needs ~${NEED_GB} GB of scratch; ${AVAIL_GB} GB available on ${TMPDIR:-/tmp}"
if [ "${AVAIL_GB:-0}" -ge "${NEED_GB}" ]; then
  export SINGULARITY_TMPDIR="${LOCAL_DIR}"
  echo "using NODE-LOCAL scratch: ${SINGULARITY_TMPDIR}"
else
  export SINGULARITY_TMPDIR="${C}/tmp"
  echo "local scratch too small; falling back to Lustre: ${SINGULARITY_TMPDIR}"
fi
export SINGULARITY_CACHEDIR="${C}/cache"
mkdir -p "${SINGULARITY_TMPDIR}" "${SINGULARITY_CACHEDIR}" "$(dirname "${LOG}")"

cd "${C}"
# NO PIPE HERE. The login-node attempt piped singularity through `tail` and then read $? --
# which reports tail's status, so a failed build printed BUILD_RC=0. Redirect to a file and
# capture the real exit code.
echo "=== singularity build (full log: ${LOG}) ==="
singularity build --force "${SIF_NAME}" "docker-archive://${TAR}" > "${LOG}" 2>&1
RC=$?
echo "singularity build rc=${RC}"

# Lustre xattr warnings are noise from mksquashfs; drop them so a real error is visible.
grep -vF 'Unrecognised xattr prefix lustre.lov' "${LOG}" | tail -25

rm -rf "${LOCAL_DIR}" 2>/dev/null

# The artifact is the verdict, not the exit code.
if [ -s "${C}/${SIF_NAME}" ]; then
  ls -lh "${C}/${SIF_NAME}"
  echo "=== does it run? ==="
  singularity exec "${C}/${SIF_NAME}" bash -lc 'echo inside; cat /etc/os-release | head -2; ls /isaac-sim/python.sh' < /dev/null
  echo "SIF_BUILD_OK ${SIF_NAME}"
else
  echo "SIF_BUILD_FAILED ${SIF_NAME} (rc=${RC}); full log at ${LOG}"
  exit 1
fi
