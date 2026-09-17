#!/bin/bash
#SBATCH --job-name=cog_fs_vk
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
# CINECA support (reply to the D25 Vulkan ticket, received 2026-09-17) suggests testing Vulkan on
# boost nodes via their new FoldSpace toolbox, whose VNC tool runs a vnc.sif under the toolbox's
# OWN Apptainer 1.5.3 (the G5b failure was under system SingularityPRO 4.3.1). This probe answers,
# on one node, whether the container RUNTIME was the variable all along:
#   A) FoldSpace vnc.sif under toolbox apptainer (their suggested verification, headless)
#   B) our cog-env-5.1.0.sif under toolbox apptainer   <- the result we actually need
#   C) our cog-env-5.1.0.sif under system singularity  <- control; expect the old failure
# Each leg runs scripts/dev/vk_probe.py (ctypes vkCreateInstance + device enumeration), the exact
# call that returned ERROR_INCOMPATIBLE_DRIVER in August.
set -uo pipefail

FS=/leonardo/prod/opt/tools/foldspace/1.0
APPTAINER="${FS}/apptainer/bin/apptainer"
VNC_SIF="${FS}/tools/vnc/build/vnc.sif"
COG="${WORK}/cog"
COG_SIF="${COG}/containers/cog-env-5.1.0.sif"
PROBE="${COG}/repo/scripts/dev/vk_probe.py"

module purge
module load cintools 2>/dev/null || echo "(cintools module not available -- continuing)"

echo "=== HOST: $(hostname) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "toolbox apptainer: $(${APPTAINER} --version 2>&1)"
echo "system singularity: $(singularity --version 2>&1)"

echo
echo "=== A) FoldSpace vnc.sif under toolbox apptainer --nv ==="
# Mirror the tool's own before.sh: host dirs at /host/*, module tree bound in.
export APPTAINER_BINDPATH="/etc:/host/etc,/usr:/host/usr,/var:/host/var,/opt:/host/opt,/leonardo,/leonardo_work,/leonardo_scratch"
"${APPTAINER}" exec --nv --scratch /run/ "${VNC_SIF}" bash -c '
  echo "-- icd.d inside:"; ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1 | head -8
  echo "-- vulkaninfo: $(command -v vulkaninfo || echo absent)"
  command -v vulkaninfo >/dev/null && vulkaninfo --summary 2>&1 | sed -n "1,25p"
  command -v python3 >/dev/null && python3 '"${PROBE}"' || echo "(no python3 in vnc.sif)"
' < /dev/null || echo "leg A rc=$?"

echo
echo "=== B) cog-env-5.1.0.sif under toolbox apptainer --nv ==="
unset APPTAINER_BINDPATH
"${APPTAINER}" exec --nv -B "${COG}:${COG}" -B /leonardo_work:/leonardo_work "${COG_SIF}" bash -c '
  echo "-- icd.d inside:"; ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1 | head -8
  python3 '"${PROBE}"'
' < /dev/null || echo "leg B rc=$?"

echo
echo "=== C) control: cog-env-5.1.0.sif under system singularity --nv ==="
singularity exec --nv -B "${COG}:${COG}" -B /leonardo_work:/leonardo_work "${COG_SIF}" bash -c '
  echo "-- icd.d inside:"; ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1 | head -8
  python3 '"${PROBE}"'
' < /dev/null || echo "leg C rc=$?"

echo "FS_VK_PROBE_DONE"
