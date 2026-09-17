#!/bin/bash
#SBATCH --job-name=cog_fs_kit
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
# A/B against debug_a100_kit.sbatch (G5b stage A, failed 2026-08-19 with vkCreateInstance
# ERROR_INCOMPATIBLE_DRIVER -> CPU rendering): identical image and payload (NVIDIA's own
# camera.py example, pixel-judged), the ONE changed variable is the container runtime --
# the FoldSpace toolbox Apptainer 1.5.3 instead of system SingularityPRO 4.3.1.
# PASS here means cluster-side Isaac eval is back on the table (revisit D25).
set -uo pipefail

FS=/leonardo/prod/opt/tools/foldspace/1.0
APPTAINER="${FS}/apptainer/bin/apptainer"
COG="${WORK}/cog"
SIF="${COG}/containers/isaac-sim-5.1.0.sif"
OUTDIR="${COG}/results/fs_kit_${SLURM_JOB_ID}"
ISAAC_HOME="${COG}/isaac_home"
mkdir -p "${OUTDIR}" "${ISAAC_HOME}"

module purge
module load cintools 2>/dev/null || echo "(cintools module not available -- continuing)"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
echo "runtime: $(${APPTAINER} --version 2>&1)"
if [ ! -f "${SIF}" ]; then echo "FS_KIT_FAILED: no image at ${SIF}"; exit 1; fi

export OMNI_KIT_ACCEPT_EULA=YES

echo
echo "=== headless camera render, 601 steps, under toolbox apptainer ==="
cd "${OUTDIR}"
timeout 1200 "${APPTAINER}" exec --nv \
  -B "${COG}:${COG}" -B /leonardo_work:/leonardo_work \
  --pwd "${OUTDIR}" \
  --env HOME="${ISAAC_HOME}" --env OMNI_KIT_ACCEPT_EULA=YES \
  "${SIF}" bash -lc \
  '/isaac-sim/python.sh /isaac-sim/standalone_examples/api/isaacsim.sensors.camera/camera.py --test --disable-output' \
  < /dev/null
RC=$?
echo "render rc=${RC} (124 = timed out)"

echo
echo "=== vulkan failure signature scan (empty = no Vulkan errors) ==="
grep -i "INCOMPATIBLE_DRIVER\|Failed to create any GPU devices" \
  "${ISAAC_HOME}/.nvidia-omniverse/logs"/*/*.log 2>/dev/null | tail -5 || true

echo
echo "=== artifacts ==="
ls -l "${OUTDIR}" 2>&1

# Judge the PIXELS, not the exit code (D6: Kit exits 0 after fatal errors; degraded A100
# rendering would write a valid-but-blank PNG). Same gate as debug_a100_kit.sbatch.
source "${COG}/miniforge3/etc/profile.d/conda.sh"
conda activate "${COG_TRAIN_ENV:-cog_lerobot}"
python - "${OUTDIR}" <<'PYEOF'
import sys, glob, os
import numpy as np
outdir = sys.argv[1]
pngs = sorted(glob.glob(os.path.join(outdir, "camera.frame*.png")))
print(f"[gate] {len(pngs)} PNG(s) found")
if not pngs:
    print("FS_KIT_FAILED: the renderer produced no image at all")
    raise SystemExit(1)
try:
    from PIL import Image
except ImportError:
    print("[gate] PIL unavailable; falling back to file-size heuristic only")
    Image = None
ok = 0
for p in pngs:
    size = os.path.getsize(p)
    if Image is None:
        print(f"[gate] {os.path.basename(p)}: {size} bytes")
        ok += size > 5000
        continue
    a = np.asarray(Image.open(p).convert("RGB")).astype(np.float32)
    uniq = len(np.unique(a.reshape(-1, 3), axis=0))
    print(f"[gate] {os.path.basename(p)}: {size} B, shape {a.shape}, "
          f"mean {a.mean():.1f}, std {a.std():.1f}, {uniq} unique colours")
    ok += (a.std() > 3.0 and uniq > 50)
print(f"[gate] {ok}/{len(pngs)} frames look like real renders")
if ok == 0:
    print("FS_KIT_FAILED: images exist but are blank/uniform")
    raise SystemExit(1)
print("FS_KIT_PASSED")
PYEOF
