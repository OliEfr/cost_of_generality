#!/bin/bash
#SBATCH --job-name=cog_vk
#SBATCH --account=euhpc_b38_106
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:08:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# Why: the G5b gate showed Kit starting inside cog-env-5.1.0.sif and then tearing itself down
# (`Recursive unloadAllPlugins()`, exit 0) before frames_qa.py's module-level os.makedirs ever ran.
# That is a failure during Kit startup, and the usual cause under Singularity is Vulkan: the RTX
# renderer needs the NVIDIA Vulkan ICD, which `--nv` does not always inject. This probe answers
# whether Vulkan can see the GPU inside the container -- if the ICD is simply missing, the fix is a
# bind plus VK_ICD_FILENAMES and needs NO image rebuild.
set -uo pipefail
SIF="${WORK}/cog/containers/cog-env-5.1.0.sif"

echo "=== HOST (compute node $(hostname)) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
for d in /usr/share/vulkan/icd.d /etc/vulkan/icd.d; do
  echo "-- ${d}:"; ls "${d}" 2>&1 | head -5
done
echo "-- host nvidia GL/VK libs:"
ls /usr/lib64/libGLX_nvidia.so* /usr/lib64/libnvidia-glvkspirv.so* 2>&1 | head -4

echo
echo "=== INSIDE CONTAINER with --nv ==="
singularity exec --nv -B "${WORK}:${WORK}" "${SIF}" bash /dev/stdin < /dev/null <<'INNER' || echo "inner rc=$?"
echo "-- icd.d inside:"; ls /usr/share/vulkan/icd.d 2>&1 | head -8
echo "-- VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-unset}"
echo "-- vulkan in ldconfig: $(ldconfig -p 2>/dev/null | grep -ci vulkan)"
echo "-- glvkspirv in ldconfig: $(ldconfig -p 2>/dev/null | grep -c glvkspirv)"
echo "-- GLX_nvidia in ldconfig: $(ldconfig -p 2>/dev/null | grep -c GLX_nvidia)"
python - <<'PY'
import ctypes
try:
    ctypes.CDLL("libvulkan.so.1"); print("-- libvulkan.so.1 loads OK")
except OSError as e:
    print("-- libvulkan load FAILED:", e)
PY
INNER
echo "VK_PROBE_DONE"
