#!/bin/bash
#SBATCH --job-name=cog_icd
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
# Kit fails with `vkCreateInstance failed / ERROR_INCOMPATIBLE_DRIVER` and
# `Failed to create any GPU devices` even with the host ICD bound and visible. An ICD json names a
# driver library; if THAT library (or one of its deps) is not among the ~46 libs `--nv` injects, the
# loader finds no usable driver and reports exactly this. So: print the json, then try to dlopen the
# library it names, from inside the container. The dlopen error names the missing dependency.
set -uo pipefail
SIF="${WORK}/cog/containers/cog-env-5.1.0.sif"
ICD_DIR=/usr/share/vulkan/icd.d
ICD_JSON="${ICD_DIR}/nvidia_icd.x86_64.json"

echo "=== host: $(hostname) ==="
echo "--- ICD json content ---"; cat "${ICD_JSON}" 2>&1
LIB=$(python3 -c "import json,sys; print(json.load(open('${ICD_JSON}'))['ICD']['library_path'])" 2>/dev/null \
      || grep -oE '"library_path"[^,}]*' "${ICD_JSON}")
echo "--- library_path = ${LIB}"
echo "--- host copy: $(ls -l /usr/lib64/${LIB} 2>&1 | tail -1)"

echo
echo "=== inside container (--nv, ICD bound) ==="
singularity exec --nv -B "${WORK}:${WORK}" -B "${ICD_DIR}:${ICD_DIR}" \
  --env VK_ICD_FILENAMES="${ICD_JSON}" --env VK_DRIVER_FILES="${ICD_JSON}" \
  --env COG_VK_LIB="${LIB}" \
  "${SIF}" bash /dev/stdin < /dev/null <<'INNER'
echo "-- injected libs: $(ls /.singularity.d/libs 2>/dev/null | wc -l)"
echo "-- looking for ${COG_VK_LIB}:"
ls -l /.singularity.d/libs/${COG_VK_LIB}* 2>&1 | tail -2
python - <<'PY'
import ctypes, os
lib = os.environ.get("COG_VK_LIB", "libGLX_nvidia.so.0")
for cand in (lib, f"/.singularity.d/libs/{lib}"):
    try:
        ctypes.CDLL(cand); print(f"-- dlopen OK: {cand}")
    except OSError as e:
        print(f"-- dlopen FAILED {cand}: {e}")
# Also try the ICD driver entry the way the loader would
try:
    ctypes.CDLL("libvulkan.so.1"); print("-- libvulkan.so.1 OK")
except OSError as e:
    print("-- libvulkan FAILED:", e)
PY
echo "-- libgomp present? $(ls /usr/lib/x86_64-linux-gnu/libgomp.so.1 2>&1 | tail -1)"
INNER
echo "ICD_PROBE_DONE"
