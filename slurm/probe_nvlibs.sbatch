#!/bin/bash
#SBATCH --job-name=cog_nvlibs
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
# Only blocker left for cluster eval: `vkCreateInstance -> ERROR_INCOMPATIBLE_DRIVER`, so Kit gets
# no GPU device and silently renders on the CPU (our scene took 274 s instead of ~20 s). torch sees
# the A100 fine, so this is Kit's Vulkan path alone. The ICD is valid and its library dlopens, which
# leaves: the NVIDIA Vulkan driver needs libnvidia-* companions that `--nv` did not inject (it ships
# only 46 libs). This lists what the host has versus what got injected, so the gap is a fact rather
# than a guess.
set -uo pipefail
SIF="${WORK}/cog/containers/cog-env-5.1.0.sif"
OUT="${WORK}/cog/nvlib_diff.txt"

echo "=== host nvidia userspace libs ==="
ls /usr/lib64/ | grep -E "^lib(nvidia|GLX_nvidia|EGL_nvidia|GLESv[12]_nvidia|cuda|nvcuvid|nvoptix|nvidia-ml)" | sort > /tmp/host_nv.txt
wc -l < /tmp/host_nv.txt; head -40 /tmp/host_nv.txt

echo
echo "=== injected into the container by --nv ==="
singularity exec --nv -B "${WORK}:${WORK}" "${SIF}" \
  bash -lc 'ls /.singularity.d/libs/ 2>/dev/null | sort' < /dev/null > /tmp/inj.txt
wc -l < /tmp/inj.txt; cat /tmp/inj.txt

echo
echo "=== host libs NOT injected (candidates for the Vulkan failure) ==="
comm -23 /tmp/host_nv.txt /tmp/inj.txt | tee "${OUT}"
echo
echo "written to ${OUT}"
echo "NVLIB_PROBE_DONE"
