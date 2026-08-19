#!/bin/bash
#SBATCH --job-name=cog_a100_gate
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
# G5b, for real this time: can Isaac Sim render OUR scene on an A100?
#   sbatch slurm/debug_a100_cogenv.sbatch
#
# PASS -> evaluation can run on the cluster (72 -> 24+2 evals after D24, ~9 GPU-h).
# FAIL -> sim stages stay on the local 4090, which is the plan's accepted fallback; training is
#         unaffected either way.
#
# This supersedes debug_a100_kit.sbatch, which targeted the NVIDIA isaac-sim image's own
# standalone_examples tree. The image we actually ship (cog-env-5.1.0.sif) is the working local
# conda env inside ubuntu:24.04, so the right test is our own `frames_qa.py` -- the artifact the
# original G5b gate was specified around.
set -uo pipefail

COG="${WORK}/cog"
SIF="${COG}/containers/cog-env-5.1.0.sif"
REPO_IN="/home/admin_07/cost_of_generality"       # the repo's path INSIDE the image
QA_OUT="${COG}/qa_gate_${SLURM_JOB_ID}"           # writable stand-in for the repo's ops/
ISAAC_HOME="${COG}/isaac_home"
mkdir -p "${QA_OUT}" "${ISAAC_HOME}"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
if [ ! -f "${SIF}" ]; then echo "G5B_FAILED: no image at ${SIF}"; exit 1; fi
ls -lh "${SIF}"

# frames_qa.py hardcodes OUT="${REPO_IN}/ops/qa" and the image is read-only, so bind a writable
# directory over the repo's ops/ (which .dockerignore excluded anyway). os.makedirs then creates
# qa/ inside the bind. No code change needed.
# HOME is redirected because Kit writes GB of shader cache and $HOME has a 50 GB quota.
echo
echo "=== frames_qa inside the container ==="
timeout 1500 srun singularity exec --nv \
  -B "${COG}:${COG}" -B /leonardo_work:/leonardo_work \
  -B "${QA_OUT}:${REPO_IN}/ops" \
  --env HOME="${ISAAC_HOME}" --env OMNI_KIT_ACCEPT_EULA=YES \
  "${SIF}" bash -lc "cd ${REPO_IN} && python scripts/dev/frames_qa.py --level L0 --resets 2 --headless --enable_cameras" \
  < /dev/null
RC=$?
echo "frames_qa rc=${RC} (124 = timed out)"

echo
echo "=== artifacts ==="
find "${QA_OUT}" -type f 2>/dev/null | head; ls -l "${QA_OUT}/qa" 2>/dev/null

# Judge PIXELS, not the exit code: Kit exits 0 after fatal errors (D6), and the documented A100
# failure mode is DEGRADED rendering, which writes a perfectly valid blank PNG.
source "${COG}/miniforge3/etc/profile.d/conda.sh"
conda activate "${COG_TRAIN_ENV:-cog_lerobot}"
python - "${QA_OUT}" <<'PYEOF'
import glob, os, sys
import numpy as np
outdir = sys.argv[1]
pngs = sorted(glob.glob(os.path.join(outdir, "**", "*.png"), recursive=True))
print(f"[gate] {len(pngs)} PNG(s)")
if not pngs:
    print("G5B_FAILED: renderer produced no image"); raise SystemExit(1)
from PIL import Image
ok = 0
for p in pngs:
    a = np.asarray(Image.open(p).convert("RGB")).astype(np.float32)
    uniq = len(np.unique(a.reshape(-1, 3), axis=0))
    print(f"[gate] {os.path.basename(p)}: {a.shape} mean {a.mean():.1f} std {a.std():.1f} "
          f"{uniq} unique colours")
    ok += (a.std() > 3.0 and uniq > 50)
print(f"[gate] {ok}/{len(pngs)} look like real renders")
if ok == 0:
    print("G5B_FAILED: images blank/uniform -- degraded rendering on A100"); raise SystemExit(1)
print("G5B_PASSED")
PYEOF
