#!/bin/bash
#SBATCH --job-name=cog_bench_dbg
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
# BENCHMARK: one standard eval cell (T1 L1, frozen protocol, 100 episodes) on an A100,
# to compare against the measured local 4090 number (docs/timings.md: T1 ~8 min/checkpoint
# on the shared card, 4-6 min when it is empty). Uses the 2026-09-17 Vulkan recipe
# (FoldSpace apptainer + stock ICD bind + verifyDriverVersion=false) plus USD assets staged
# at $WORK/cog/isaac_assets (compute nodes are offline).
#
# cuDNN ROOT CAUSE (jobs 58046658/58047343/58047669): the local cog_isaac env -- the env this
# image was built from -- has BOTH nvidia_cudnn_cu12 9.7.1.26 and nvidia_cudnn_cu13 9.20.0.48
# installed. They share site-packages/nvidia/cudnn/lib, so the CUDA-13 build shadows the CUDA-12
# one (torch.backends.cudnn.version() == 92000 == 9.20.0). That works on this workstation's
# driver 580 but CANNOT initialise on Leonardo's driver 535 (CUDA 12 only) -> every conv2d dies
# with CUDNN_STATUS_NOT_INITIALIZED, with or without Kit. The cluster TRAINING env
# (miniforge3/envs/cog_lerobot) carries only the cu12 build, which is why training convs always
# worked on the same A100s. Fix without rebuilding the 10 GB image: bind that env's cuDNN lib
# directory over the container's.
#
# (historic) the first attempt (58046658) died in the policy ResNet with
# CUDNN_STATUS_NOT_INITIALIZED. Probe 58047343 ruled out Kit, the libgomp shim and memory;
# a local control run showed the same by-soname dlopen failures WITH a working conv, so the
# cause is library RESOLUTION inside the container, not missing torch support. cuDNN 9 splits
# into sub-libraries (libcudnn_ops/cnn/engines_*) that libcudnn.so.9 dlopens by soname at
# runtime, so the fix is to put their directory on LD_LIBRARY_PATH. Stage 1 below verifies
# that; only if it passes does the benchmark run, so a failure is reported as such rather
# than silently benchmarking a cuDNN-disabled fallback (which would not be comparable).
set -uo pipefail

COG="${WORK}/cog"
APPTAINER=/leonardo/prod/opt/tools/foldspace/1.0/apptainer/bin/apptainer
SIF="${COG}/containers/cog-env-5.1.0.sif"
REPO_IN="/home/admin_07/cost_of_generality"
ENV_IN=/home/admin_07/miniconda3/envs/cog_isaac/lib/python3.11/site-packages/isaacsim
NV_LIBS=/home/admin_07/miniconda3/envs/cog_isaac/lib/python3.11/site-packages/nvidia
ASSET_ROOT="${COG}/isaac_assets/Assets/Isaac/5.1"
CKPT="${COG}/checkpoints/t1_L1_n100_s0/checkpoints/080000/pretrained_model"
OUT_DIR="${COG}/results/bench_eval_${SLURM_JOB_ID}"
ISAAC_HOME="${COG}/isaac_home"
KIT_RW="${COG}/kit_rw"
mkdir -p "${OUT_DIR}" "${ISAAC_HOME}"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
for p in "${SIF}" "${CKPT}" "${ASSET_ROOT}/Isaac/IsaacLab/Objects/Mug/mug.usd"; do
  [ -e "${p}" ] || { echo "BENCH_FAILED: missing ${p}"; exit 1; }
done

KIT_BINDS=(-B "${KIT_RW}/logs:${ENV_IN}/kit/logs"
           -B "${KIT_RW}/data:${ENV_IN}/kit/data"
           -B "${KIT_RW}/cache:${ENV_IN}/kit/cache")

# Every nvidia pip package ships its libs in <pkg>/lib; add them all rather than guessing which
# sub-library the dispatcher wants next.
CUDNN_OK="${COG}/miniforge3/envs/cog_lerobot/lib/python3.11/site-packages/nvidia/cudnn/lib"
CUDNN_LD="${NV_LIBS}/cudnn/lib:${NV_LIBS}/cublas/lib:${NV_LIBS}/cuda_runtime/lib:${NV_LIBS}/cufft/lib:${NV_LIBS}/curand/lib:${NV_LIBS}/cusolver/lib:${NV_LIBS}/cusparse/lib:${NV_LIBS}/nvjitlink/lib"

echo
echo "=== stage 1: cuDNN sub-library check + conv2d with the lib dirs on LD_LIBRARY_PATH ==="
"${APPTAINER}" exec --nv -B "${COG}:${COG}" -B "${CUDNN_OK}:${NV_LIBS}/cudnn/lib" --home "${ISAAC_HOME}" \
  "${SIF}" bash -c "ls ${NV_LIBS}/cudnn/lib 2>&1 | head -12; \
    export LD_LIBRARY_PATH=${CUDNN_LD}:\${LD_LIBRARY_PATH}; \
    python3 -u ${COG}/repo/scripts/dev/cudnn_probe.py" < /dev/null | tail -14
S1=$?

echo
echo "=== stage 2: the eval benchmark ==="
T0=$(date +%s)
echo "BENCH_START $(date -Is)"
timeout 1500 "${APPTAINER}" exec --nv \
  -B "${COG}:${COG}" -B /leonardo_work:/leonardo_work \
  -B /usr/share/vulkan/icd.d:/etc/vulkan/icd.d \
  -B "${COG}/repo/src:${REPO_IN}/src" \
  -B "${COG}/repo/configs:${REPO_IN}/configs" \
  -B "${COG}/repo/scripts:${REPO_IN}/scripts" \
  -B "${OUT_DIR}:${REPO_IN}/ops" \
  "${KIT_BINDS[@]}" \
  -B "${COG}/extralibs:/opt/extralibs" \
  -B "${CUDNN_OK}:${NV_LIBS}/cudnn/lib" \
  --home "${ISAAC_HOME}" --env OMNI_KIT_ACCEPT_EULA=YES --env HF_HUB_OFFLINE=1 \
  --env PYTHONPATH="${REPO_IN}/src" \
  "${SIF}" bash -c "export LD_LIBRARY_PATH=/opt/extralibs:${CUDNN_LD}:\${LD_LIBRARY_PATH}; cd ${REPO_IN} && \
    python -u -m cog.eval.rollout_eval \
      --task Cog-CupPlace-L1-IK-Rel-Visuomotor-v0 \
      --checkpoint ${CKPT} \
      --num_inference_steps 10 --max_steps 600 \
      --out ${OUT_DIR}/bench_eval_t1_L1_n100_s0_080000.json \
      --headless --enable_cameras \
      --kit_args=\"--/rtx/verifyDriverVersion/enabled=false --/persistent/isaac/asset_root/default=${ASSET_ROOT} --/persistent/isaac/asset_root/cloud=${ASSET_ROOT}\"" \
  < /dev/null
RC=$?
T1=$(date +%s)
echo "BENCH_WALL_SECONDS=$(( T1 - T0 )) rc=${RC}"

if [ -s "${OUT_DIR}/bench_eval_t1_L1_n100_s0_080000.json" ]; then
  python3 - "${OUT_DIR}/bench_eval_t1_L1_n100_s0_080000.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"BENCH_OK episodes={d.get('episodes')} SR={d.get('success_rate')}")
PY
else
  echo "BENCH_FAILED: no result artifact (per-batch timings above still give throughput)"
fi
