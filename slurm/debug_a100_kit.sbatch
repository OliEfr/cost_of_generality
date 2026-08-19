#!/bin/bash
#SBATCH --job-name=cog_a100_kit
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
# G5b stage A: does Isaac Sim render at all on an A100? Isaac Sim is officially unsupported on
# A100 (no RT cores; upstream issues #3421 crash and #1519 degraded rendering), so this is the
# hypothesis that decides whether cluster-side eval is possible. Tested on the BASE isaac-sim
# image before investing in a larger IsaacLab+cog image.
#
#   sbatch slurm/debug_a100_kit.sbatch
#
# Uses NVIDIA's own shipped example (standalone_examples/api/isaacsim.sensors.camera/camera.py,
# `--test` = 601 steps) rather than a script of mine, so a failure cannot be blamed on my use of
# an API I guessed at. The example writes camera.frame*.png every 100 steps; those PNGs are the
# artifact this gate judges -- existence AND pixel content, because "degraded rendering" would
# produce a file that exists and is blank.
set -uo pipefail

COG="${WORK}/cog"
SIF="${COG}/containers/isaac-sim-5.1.0.sif"
OUTDIR="${COG}/results/a100_kit_${SLURM_JOB_ID}"
# Kit writes several GB of shader/compute caches under $HOME. $HOME here has a 50 GB quota, so
# point HOME at $WORK instead -- filling the home quota would break login, not just this job.
ISAAC_HOME="${COG}/isaac_home"
mkdir -p "${OUTDIR}" "${ISAAC_HOME}"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
if [ ! -f "${SIF}" ]; then echo "G5B_A_FAILED: no image at ${SIF}"; exit 1; fi
ls -lh "${SIF}"

export OMNI_KIT_ACCEPT_EULA=YES

echo
echo "=== NVIDIA's own compatibility check (informational; it may report A100 unsupported) ==="
timeout 300 singularity exec --nv \
  -B "${COG}:${COG}" -B /leonardo_work:/leonardo_work \
  --env HOME="${ISAAC_HOME}" --env OMNI_KIT_ACCEPT_EULA=YES \
  "${SIF}" bash -lc '/isaac-sim/isaac-sim.compatibility_check.sh --no-window 2>&1 | tail -40' \
  < /dev/null
echo "compatibility_check rc=$?"

echo
echo "=== headless camera render, 601 steps ==="
cd "${OUTDIR}"
timeout 1200 singularity exec --nv \
  -B "${COG}:${COG}" -B /leonardo_work:/leonardo_work \
  --pwd "${OUTDIR}" \
  --env HOME="${ISAAC_HOME}" --env OMNI_KIT_ACCEPT_EULA=YES \
  "${SIF}" bash -lc \
  '/isaac-sim/python.sh /isaac-sim/standalone_examples/api/isaacsim.sensors.camera/camera.py --test --disable-output' \
  < /dev/null
RC=$?
echo "render rc=${RC} (124 = timed out)"

echo
echo "=== artifacts ==="
ls -l "${OUTDIR}" 2>&1

# Judge the PIXELS, not the exit code: Kit is known to exit 0 after fatal errors (D6), and a
# renderer degraded to a blank frame still writes a valid PNG.
source "${COG}/miniforge3/etc/profile.d/conda.sh"
conda activate "${COG_TRAIN_ENV:-cog_lerobot}"
python - "${OUTDIR}" <<'PYEOF'
import sys, glob, os
import numpy as np
outdir = sys.argv[1]
pngs = sorted(glob.glob(os.path.join(outdir, "camera.frame*.png")))
print(f"[gate] {len(pngs)} PNG(s) found")
if not pngs:
    print("G5B_A_FAILED: the renderer produced no image at all")
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
    # A working render of this scene (ground plane + three coloured cubes) has real structure:
    # many distinct colours and a non-trivial spread. A black/uniform frame is the signature of
    # the degraded-rendering failure mode, and it would pass a mere "file exists" check.
    uniq = len(np.unique(a.reshape(-1, 3), axis=0))
    print(f"[gate] {os.path.basename(p)}: {size} B, shape {a.shape}, "
          f"mean {a.mean():.1f}, std {a.std():.1f}, {uniq} unique colours")
    ok += (a.std() > 3.0 and uniq > 50)
print(f"[gate] {ok}/{len(pngs)} frames look like real renders")
if ok == 0:
    print("G5B_A_FAILED: images exist but are blank/uniform -- degraded rendering on A100")
    raise SystemExit(1)
print("G5B_A_PASSED")
PYEOF
