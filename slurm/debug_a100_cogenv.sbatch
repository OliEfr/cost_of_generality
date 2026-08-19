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
# VULKAN: measured on a compute node (job 52958496) -- the host has
# /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json, but INSIDE the container with --nv that directory
# does not exist and VK_ICD_FILENAMES is unset. libvulkan.so.1 loads, so the loader is there with no
# ICD and no driver behind it => zero Vulkan devices => Kit's RTX renderer cannot initialise and Kit
# tears itself down (`Recursive unloadAllPlugins()`) and exits **0**, before frames_qa.py's
# module-level os.makedirs ever runs. That looks identical to "the A100 cannot render" and is not.
# Fix needs no image rebuild: bind the host ICD directory and point the loader at the NVIDIA ICD.
# READ-ONLY FILESYSTEM: Kit writes into its OWN install tree, which lives inside the squashfs and
# is therefore read-only. Its first log lines give it away:
#   Loading user config located at: '<env>/site-packages/isaacsim/kit/data/Kit/Isaac-Sim/5.1/user.config.json'
#   [carb] Logging to file: '<env>/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/kit_*.log'
# Locally that tree is a writable conda env, so nobody notices. In a .sif every one of those writes
# fails, and Kit unwinds (`Recursive unloadAllPlugins()`) and exits **0** before AppLauncher returns
# -- which is why frames_qa.py's module-level os.makedirs never ran.
#
# Fix: bind writable host directories over the three dirs Kit writes (logs, data, cache). NOT
# --writable-tmpfs: its overlay is capped by `sessiondir max size` (tens of MB) and Kit's shader
# cache is far larger. $WORK has 3 TB.
# kit/data is SEEDED from the image first -- binding over it would otherwise hide user.config.json
# and Kit's extension registry.
ENV_IN=/home/admin_07/miniconda3/envs/cog_isaac/lib/python3.11/site-packages/isaacsim
KIT_RW="${COG}/kit_rw"
if [ ! -d "${KIT_RW}/data" ]; then
  echo "seeding ${KIT_RW} from the image (first run only)"
  mkdir -p "${KIT_RW}/logs" "${KIT_RW}/data" "${KIT_RW}/cache"
  singularity exec "${SIF}" bash -lc "cp -a ${ENV_IN}/kit/data/. /seed_data/ 2>/dev/null || true"     --bind "${KIT_RW}/data:/seed_data" < /dev/null ||   singularity exec -B "${KIT_RW}/data:/seed_data" "${SIF}"     bash -lc "cp -a ${ENV_IN}/kit/data/. /seed_data/ 2>/dev/null || true" < /dev/null
  singularity exec -B "${KIT_RW}/cache:/seed_cache" "${SIF}"     bash -lc "cp -a ${ENV_IN}/kit/cache/. /seed_cache/ 2>/dev/null || true" < /dev/null
  echo "seeded: data=$(find ${KIT_RW}/data -type f 2>/dev/null | wc -l) files, cache=$(find ${KIT_RW}/cache -type f 2>/dev/null | wc -l) files"
fi
KIT_BINDS=(-B "${KIT_RW}/logs:${ENV_IN}/kit/logs"
           -B "${KIT_RW}/data:${ENV_IN}/kit/data"
           -B "${KIT_RW}/cache:${ENV_IN}/kit/cache")

# Two fixes found by reading Kit's OWN log (only readable once the writable binds above existed)
# plus the upstream issue tracker -- neither needs an image rebuild:
#
# 1. `libgomp.so.1: cannot open shared object file`, 18 times, which then cascaded into 81 further
#    failures of Isaac's own libs (libomni.usd.so x73, libusd_hd.so x6, librtx.hydra.so, libosdCPU).
#    ONE missing system package, `libgomp1`, explained 99 of 101 errors. Rather than rebuild 24 GB
#    to add it, the library is staged on $WORK and prepended to LD_LIBRARY_PATH: it is copied from
#    the workstation's Ubuntu 24.04, the same distro the image is built from, so it is ABI-exact.
#    (It should still go into the Dockerfile for the final image -- a bind is a test, not a design.)
#
# 2. `vkCreateInstance failed / ERROR_INCOMPATIBLE_DRIVER / Vulkan 1.1 is not supported`.
#    NOT a plumbing fault: the probe (job 52969503) showed the ICD is valid
#    (api_version 1.3.242) and `libGLX_nvidia.so.0` is injected by --nv and dlopens fine inside the
#    container. It is the documented misreport for **driver 535.255 and newer** -- Leonardo runs
#    535.274.02 -- for which NVIDIA's workaround is `--/rtx/verifyDriverVersion/enabled=false`.
#    Passed via IsaacLab's `--kit_args`, so frames_qa.py needs no change. NOTE the `=` form:
#    argparse rejects a value beginning with `--` when separated by a space (it reads it as the
#    next option and exits 2), so `--kit_args "--/rtx/..."` never reaches Kit -- the run died
#    with rc=2 before Kit started, which is why no new Kit log appeared. IsaacLab's own docs
#    show the space-separated form; it does not work.
#
VK_ICD_HOST=/usr/share/vulkan/icd.d
VK_ICD_SRC="${VK_ICD_HOST}/nvidia_icd.x86_64.json"
echo "host ICD: $(ls ${VK_ICD_SRC} 2>&1)"

# The stock ICD names its driver by BARE SONAME:
#     {"ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.242"}}
# which the Vulkan loader resolves through LD_LIBRARY_PATH. `--nv` puts the driver in
# /.singularity.d/libs and adds that to LD_LIBRARY_PATH, and a bare `dlopen("libGLX_nvidia.so.0")`
# from a plain shell inside the container does succeed (probe 52969503). But **Kit rewrites
# LD_LIBRARY_PATH for its own process**, so by the time its bundled loader
# (omni.gpu_foundation/bin/deps/libvulkan.so.1) resolves the ICD, the soname no longer resolves and
# the loader reports ERROR_INCOMPATIBLE_DRIVER -- indistinguishable from a genuinely unsupported GPU.
# Rewriting library_path to an ABSOLUTE path removes the search entirely.
#
# Three Vulkan hypotheses are now DISPROVED by measurement rather than argued away: the ICD was
# genuinely absent (fixed; no change to the error); NVIDIA's documented 535.255+ driver-version
# misreport workaround `--/rtx/verifyDriverVersion/enabled=false` (no change -- it gates Kit's own
# check, not vkCreateInstance); and the ICD's bare soname vs Kit's rewritten LD_LIBRARY_PATH
# (absolute path; no change). So stop guessing and make the loader talk: VK_LOADER_DEBUG makes the
# Vulkan loader log every ICD it discovers and its reason for rejecting each one.
#
# Aside for the record: Apptainer documents `--nvccli` + NVIDIA_DRIVER_CAPABILITIES=graphics as the
# way to get Vulkan/GL, but `nvidia-container-cli` is not installed on Leonardo, so that route is
# closed. Leonardo's /etc/singularity/nvliblist.conf does already list the graphics libs, so plain
# `--nv` injects them -- the libraries were never the problem, only their resolution.
VK_ICD_JSON="${COG}/nvidia_icd_abs.json"
python3 - "${VK_ICD_SRC}" "${VK_ICD_JSON}" <<'PYICD'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))
d["ICD"]["library_path"] = "/.singularity.d/libs/libGLX_nvidia.so.0"
json.dump(d, open(dst, "w"), indent=4)
print("rewrote ICD ->", dst, d["ICD"])
PYICD

echo "=== container GPU/Vulkan visibility with the ICD bound ==="
singularity exec --nv \
  -B "${COG}:${COG}" -B "${VK_ICD_HOST}:${VK_ICD_HOST}" \
  --env VK_ICD_FILENAMES="${VK_ICD_JSON}" --env VK_DRIVER_FILES="${VK_ICD_JSON}" \
  "${SIF}" bash -lc 'echo " singularity libs: $(ls /.singularity.d/libs 2>/dev/null | wc -l) entries";
    echo " LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}";
    echo " GLX_nvidia present: $(ls /.singularity.d/libs/libGLX_nvidia* 2>/dev/null | wc -l)";
    echo " glvkspirv present: $(ls /.singularity.d/libs/libnvidia-glvkspirv* 2>/dev/null | wc -l)";
    echo " ICD visible: $(ls ${VK_ICD_FILENAMES} 2>&1)"' < /dev/null

timeout 1500 srun singularity exec --nv \
  -B "${COG}:${COG}" -B /leonardo_work:/leonardo_work \
  -B "${QA_OUT}:${REPO_IN}/ops" \
  -B "${VK_ICD_HOST}:${VK_ICD_HOST}" \
  "${KIT_BINDS[@]}" \
  --env HOME="${ISAAC_HOME}" --env OMNI_KIT_ACCEPT_EULA=YES \
  --env VK_ICD_FILENAMES="${VK_ICD_JSON}" --env VK_DRIVER_FILES="${VK_ICD_JSON}" \
  -B "${COG}/extralibs:/opt/extralibs" \
  --env VK_LOADER_DEBUG="${COG_VK_LOADER_DEBUG:-error,warn}" \
  "${SIF}" bash -lc "export LD_LIBRARY_PATH=/opt/extralibs:\${LD_LIBRARY_PATH}; cd ${REPO_IN} && python scripts/dev/frames_qa.py --level L0 --resets 2 --headless --enable_cameras --kit_args=\"--/rtx/verifyDriverVersion/enabled=false\"" \
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
