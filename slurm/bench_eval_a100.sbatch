#!/bin/bash
#SBATCH --job-name=cog_bench_eval
#SBATCH --account=euhpc_b38_106
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# BENCHMARK: one standard eval cell (T1 L1, frozen protocol, 100 episodes) on an A100,
# to compare against the measured local 4090 number (docs/timings.md: ~14 min/checkpoint,
# GPU shared with the foreign policy server). Uses the 2026-09-17 Vulkan recipe
# (FoldSpace toolbox apptainer + stock ICD bind + verifyDriverVersion=false, journal) plus
# locally-staged USD assets at $WORK/cog/isaac_assets (offline nodes; 118 MB mirror of the
# four subtrees T1 references). Diagnostics only -- the result json goes to results/diagnostics
# naming, NOT a study cell.
set -uo pipefail

COG="${WORK}/cog"
APPTAINER=/leonardo/prod/opt/tools/foldspace/1.0/apptainer/bin/apptainer
SIF="${COG}/containers/cog-env-5.1.0.sif"
REPO_IN="/home/admin_07/cost_of_generality"
ENV_IN=/home/admin_07/miniconda3/envs/cog_isaac/lib/python3.11/site-packages/isaacsim
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

T0=$(date +%s)
echo "BENCH_START $(date -Is)"
timeout 9000 "${APPTAINER}" exec --nv \
  -B "${COG}:${COG}" -B /leonardo_work:/leonardo_work \
  -B /usr/share/vulkan/icd.d:/etc/vulkan/icd.d \
  -B "${COG}/repo/src:${REPO_IN}/src" \
  -B "${COG}/repo/configs:${REPO_IN}/configs" \
  -B "${COG}/repo/scripts:${REPO_IN}/scripts" \
  -B "${OUT_DIR}:${REPO_IN}/ops" \
  "${KIT_BINDS[@]}" \
  -B "${COG}/extralibs:/opt/extralibs" \
  --home "${ISAAC_HOME}" --env OMNI_KIT_ACCEPT_EULA=YES --env HF_HUB_OFFLINE=1 \
  --env PYTHONPATH="${REPO_IN}/src" \
  "${SIF}" bash -c "export LD_LIBRARY_PATH=/opt/extralibs:\${LD_LIBRARY_PATH}; cd ${REPO_IN} && \
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
  echo "BENCH_FAILED: no result artifact"
fi
