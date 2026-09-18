# Shared container recipe for every Isaac job on Leonardo. SOURCED by the sbatch files, never run.
#
# One copy, because the recipe is five interacting facts that each cost hours to find
# (docs/cluster_eval.md, journal 2026-09-17):
#   1. the FoldSpace toolbox Apptainer 1.5.3, NOT the system SingularityPRO -- under
#      SingularityPRO vkCreateInstance always returns ERROR_INCOMPATIBLE_DRIVER;
#   2. the STOCK host Vulkan ICD bound at /etc/vulkan/icd.d -- a rewritten absolute-path ICD
#      loads but then fails vk_icdGetInstanceProcAddr, which reads as "found no drivers";
#   3. --/rtx/verifyDriverVersion/enabled=false -- Kit misparses driver 535.274.02 as "535.18"
#      and refuses the RTX renderer;
#   4. the cluster training env's CUDA-12 cuDNN bound over the image's CUDA-13 one -- the image
#      was built from a local env carrying both, and the cu13 build cannot initialise on driver
#      535, so every conv2d dies with CUDNN_STATUS_NOT_INITIALIZED;
#   5. no inner srun -- a job step does not inherit --gres, so the GPU disappears.
# A drifted second copy fails by rendering on the CPU or dying in a conv. Both look like
# something else entirely, which is why this is a library and not a convention.

WORK="${WORK:-/leonardo_work/EUHPC_B38_106}"
FAST="${FAST:-/leonardo_scratch/fast/EUHPC_B38_106}"
COG="${WORK}/cog"
APPTAINER=/leonardo/prod/opt/tools/foldspace/1.0/apptainer/bin/apptainer
SIF="${COG}/containers/cog-env-5.1.0.sif"

# Paths INSIDE the image are the workstation's absolute paths: the image copies the conda env and
# the repo to their original locations so nothing has to be relocated (docker/Dockerfile.cog_env).
REPO_IN=/home/admin_07/cost_of_generality
ENV_IN=/home/admin_07/miniconda3/envs/cog_isaac/lib/python3.11/site-packages/isaacsim
NV_LIBS=/home/admin_07/miniconda3/envs/cog_isaac/lib/python3.11/site-packages/nvidia
CUDNN_OK="${COG}/miniforge3/envs/cog_lerobot/lib/python3.11/site-packages/nvidia/cudnn/lib"
# cuDNN 9 dispatches to libcudnn_ops/cnn/engines_* by soname at run time, so their directory has to
# be findable; the rest are added rather than guessing which the dispatcher wants next.
CUDNN_LD="${NV_LIBS}/cudnn/lib:${NV_LIBS}/cublas/lib:${NV_LIBS}/cuda_runtime/lib:${NV_LIBS}/cufft/lib:${NV_LIBS}/curand/lib:${NV_LIBS}/cusolver/lib:${NV_LIBS}/cusparse/lib:${NV_LIBS}/nvjitlink/lib"

ASSET_ROOT="${COG}/isaac_assets/Assets/Isaac/5.1"
HDF5_DIR="${COG_HDF5_DIR:-${FAST}/cog/hdf5}"
DS_DIR="${COG_DS_DIR:-${FAST}/cog/datasets}"
RESULTS_DIR="${COG_RESULTS_DIR:-${COG}/results}"

# Compute nodes are offline, so the asset root must point at the staged mirror or Kit spends 300 s
# timing out against the Omniverse S3 before falling back. The '=' form is mandatory: Isaac's
# AppLauncher parses --kit_args="..." but not --kit_args "...".
COG_KIT_ARGS="--/rtx/verifyDriverVersion/enabled=false --/persistent/isaac/asset_root/default=${ASSET_ROOT} --/persistent/isaac/asset_root/cloud=${ASSET_ROOT}"

cog_require() {
  local missing=0 p
  for p in "$@"; do
    [ -e "${p}" ] || { echo "MISSING ${p}" >&2; missing=1; }
  done
  return "${missing}"
}

# Per-job Kit scratch. The 2026-09-17 gate ran ONE job against a single shared $WORK/cog/kit_rw and
# --home $WORK/cog/isaac_home; a wave runs ~60 at once against the same shader cache and home, which
# is a contention mode nothing has tested. Each job therefore gets its own copy.
# It lives on $FAST (NVMe, 925 GB free), not on the node's /tmp: node-local scratch is only ~10 GB
# and four GPU jobs share a node, so 4 x 1.4 GB would leave no room for anything else.
cog_job_scratch() {
  COG_SCRATCH="${COG_JOB_SCRATCH:-${FAST}/cog/jobscratch/${SLURM_JOB_ID:-manual}}"
  case "${COG_SCRATCH}" in
    */cog/jobscratch/*) : ;;
    *) echo "refusing scratch outside */cog/jobscratch/: ${COG_SCRATCH}" >&2; return 1 ;;
  esac
  rm -rf "${COG_SCRATCH}"
  mkdir -p "${COG_SCRATCH}"
  # Seeded WARM from the templates: a cold Kit boot is 36 s against 13 s warm, and the templates
  # must stay pristine, so nothing is ever copied back.
  cp -a "${COG}/kit_rw" "${COG_SCRATCH}/kit_rw"
  cp -a "${COG}/isaac_home" "${COG_SCRATCH}/isaac_home"
  trap 'rm -rf "${COG_SCRATCH}"' EXIT
}

# Seconds left before Slurm kills the job, minus a 5 min tail so the verdict block still runs.
# Every Isaac launch is wrapped in this: Kit ignores SIGTERM once hung (observed holding a slot for
# a full 30 min walltime), so the wrapper uses -s KILL and the job reports rather than vanishing.
cog_timeout_s() {
  local end now
  end=$(scontrol show job "${SLURM_JOB_ID:-0}" -o 2>/dev/null | tr ' ' '\n' | awk -F= '/^EndTime=/{print $2}')
  end=$(date -d "${end}" +%s 2>/dev/null || echo 0)
  now=$(date +%s)
  echo $(( end > now + 600 ? end - now - 300 : 3300 ))
}

# GPU + RTX container. COG_TIMEOUT (seconds) wraps the launch; COG_EXTRA_BINDS adds binds.
#
# The command is joined with "$*" into an inner `bash -c`, so an argument containing spaces must
# carry its own quotes -- which is exactly what --kit_args needs anyway ("--kit_args=\"...\"").
# Every other argument in this study is a single token.
cog_gpu_exec() {
  timeout -s KILL "${COG_TIMEOUT:-3300}" \
  "${APPTAINER}" exec --nv \
    -B "${COG}:${COG}" -B /leonardo_work:/leonardo_work -B "${FAST}:${FAST}" \
    -B /usr/share/vulkan/icd.d:/etc/vulkan/icd.d \
    -B "${COG}/repo/src:${REPO_IN}/src" \
    -B "${COG}/repo/configs:${REPO_IN}/configs" \
    -B "${COG}/repo/scripts:${REPO_IN}/scripts" \
    -B "${COG_SCRATCH}/kit_rw/logs:${ENV_IN}/kit/logs" \
    -B "${COG_SCRATCH}/kit_rw/data:${ENV_IN}/kit/data" \
    -B "${COG_SCRATCH}/kit_rw/cache:${ENV_IN}/kit/cache" \
    -B "${COG}/extralibs:/opt/extralibs" \
    -B "${CUDNN_OK}:${NV_LIBS}/cudnn/lib" \
    "${COG_EXTRA_BINDS[@]+"${COG_EXTRA_BINDS[@]}"}" \
    --home "${COG_SCRATCH}/isaac_home" \
    --env OMNI_KIT_ACCEPT_EULA=YES --env HF_HUB_OFFLINE=1 --env WANDB_MODE=offline \
    --env HDF5_USE_FILE_LOCKING=FALSE \
    --env PYTHONPATH="${REPO_IN}/src" \
    "${SIF}" bash -c "export LD_LIBRARY_PATH=/opt/extralibs:${CUDNN_LD}:\${LD_LIBRARY_PATH}; cd ${REPO_IN} && $*" \
    < /dev/null
}

# CPU-only container: no --nv, no ICD, no Kit scratch. Used by conversion, which imports only
# h5py/numpy/lerobot and encodes h264 through LeRobot's PyAV path. The image is still required --
# the cluster's cog_lerobot conda env has neither isaacsim nor h5py.
cog_cpu_exec() {
  "${APPTAINER}" exec \
    -B "${COG}:${COG}" -B /leonardo_work:/leonardo_work -B "${FAST}:${FAST}" \
    -B "${COG}/repo/src:${REPO_IN}/src" \
    -B "${COG}/repo/configs:${REPO_IN}/configs" \
    -B "${COG}/repo/scripts:${REPO_IN}/scripts" \
    --home "${COG_SCRATCH:-${FAST}/cog/jobscratch/${SLURM_JOB_ID:-manual}}" \
    --env HF_HUB_OFFLINE=1 --env HDF5_USE_FILE_LOCKING=FALSE \
    --env "COG_DATA_HDF5=${HDF5_DIR}" \
    --env PYTHONPATH="${REPO_IN}/src" \
    "${SIF}" bash -c "cd ${REPO_IN} && $*" \
    < /dev/null
}

# Count demos in an HDF5 without needing h5py outside the image. Prints -1 if unreadable.
cog_count_demos() {
  "${APPTAINER}" exec -B "${FAST}:${FAST}" -B /leonardo_work:/leonardo_work "${SIF}" \
    python -c "
import h5py, sys
try:
    with h5py.File(sys.argv[1], 'r') as f:
        print(len(f['data'].keys()))
except Exception:
    print(-1)
" "$1" 2>/dev/null | tail -1
}

# Gym prefix, episode step cap and stage instrumentation per task. max_steps is NOT cosmetic --
# T2's episodes run ~680 steps, so a 600-step cap would score long successes as failures.
cog_task_spec() {
  case "$1" in
    T1) GYM=Cog-CupPlace;   MAX_STEPS=600;  STAGES="";         SRC_STEM=L2_source_annotated    ;;
    T2) GYM=Cog-DrawerStow; MAX_STEPS=1200; STAGES="--stages"; SRC_STEM=T2_L2_source_annotated ;;
    T3) GYM=Cog-PushTarget; MAX_STEPS=800;  STAGES="";         SRC_STEM=T3_L2_source_annotated ;;
    *) echo "unknown task '$1' (want T1|T2|T3)" >&2; return 2 ;;
  esac
}
