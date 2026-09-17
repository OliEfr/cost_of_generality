#!/bin/bash
#SBATCH --job-name=cog_fs_vk2
#SBATCH --account=euhpc_b38_106
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:12:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# Follow-up to probe_foldspace_vulkan.sbatch (job 58035749), which showed:
#   - FoldSpace vnc.sif: loader works but only Mesa ICDs -> llvmpipe (CPU) device;
#   - our cog-env sif: rc=-9 under BOTH runtimes -> the runtime was never the variable.
# August's disproofs (journal 2026-08-19 22:15) were all Kit-level. This job separates
# DRIVER-level from Kit-level failure with the plain ctypes loader client:
#   H)  bare host, no container: can any Vulkan client see the A100 on a boost node?
#   B1) our sif + system singularity, host ICD dir bound over /etc/vulkan/icd.d (standard
#       loader search path, no env vars, bare-soname library_path resolves via --nv libs)
#   B2) same as B1 under the FoldSpace toolbox apptainer
#   A2) vnc.sif + nvidia ICD added the same way (its loader stack demonstrably works ->
#       if the A100 enumerates here, GPU Vulkan in containers is possible on these nodes)
set -uo pipefail

FS=/leonardo/prod/opt/tools/foldspace/1.0
APPTAINER="${FS}/apptainer/bin/apptainer"
VNC_SIF="${FS}/tools/vnc/build/vnc.sif"
COG="${WORK}/cog"
COG_SIF="${COG}/containers/cog-env-5.1.0.sif"
PROBE="${COG}/repo/scripts/dev/vk_probe.py"
ICD_DIR=/usr/share/vulkan/icd.d

module purge
module load cintools 2>/dev/null || echo "(cintools module not available -- continuing)"

echo "=== HOST: $(hostname) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "-- host ICDs:"; ls ${ICD_DIR} 2>&1

echo
echo "=== H) bare host, no container ==="
python3 "${COG}/repo/scripts/dev/vk_probe.py" || echo "leg H rc=$?"

echo
echo "=== B1) cog-env sif + system singularity, host ICDs at /etc/vulkan/icd.d ==="
singularity exec --nv -B "${COG}:${COG}" -B "${ICD_DIR}:/etc/vulkan/icd.d" \
  "${COG_SIF}" python3 "${PROBE}" < /dev/null || echo "leg B1 rc=$?"

echo
echo "=== B2) cog-env sif + toolbox apptainer, host ICDs at /etc/vulkan/icd.d ==="
"${APPTAINER}" exec --nv -B "${COG}:${COG}" -B "${ICD_DIR}:/etc/vulkan/icd.d" \
  "${COG_SIF}" python3 "${PROBE}" < /dev/null || echo "leg B2 rc=$?"

echo
echo "=== A2) vnc.sif + toolbox apptainer, nvidia ICD added at /etc/vulkan/icd.d ==="
"${APPTAINER}" exec --nv -B "${COG}:${COG}" -B "${ICD_DIR}:/etc/vulkan/icd.d" \
  --scratch /run/ "${VNC_SIF}" bash -c '
  command -v python3 >/dev/null && python3 '"${PROBE}"' || echo "(no python3 in vnc.sif)"
' < /dev/null || echo "leg A2 rc=$?"

echo "FS_VK2_PROBE_DONE"
