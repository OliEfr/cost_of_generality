#!/bin/bash
#SBATCH --job-name=cog_datagen
#SBATCH --account=euhpc_b38_106
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# ONE Isaac Lab Mimic generation leg on an A100: one sub-level env, one seed, one output file.
#
#   sbatch slurm/datagen.sbatch TASK KEY STEM SEED TRIALS [NUM_ENVS]
#   sbatch slurm/datagen.sbatch T1 ACv03 ACv03  1303 40  8
#   sbatch slurm/datagen.sbatch T1 L3v00 parity_L3bv00 1000 40 8      # the G2 parity rerun
#
# KEY is the gym id fragment (a registered SUB_LEVELS key); STEM is the output filename stem. They
# differ whenever an arm is written under a name other than its env's -- which is how the corrected
# L3b arm was generated from the unchanged L3v** envs (D27), and how a parity rerun writes beside
# the file it is being compared against instead of over it.
#
# ONE JOB PER VARIANT, not one per arm. The local waves batch ten variants into one process because
# a camera-enabled Kit boot costs 3.5-4 min on the workstation; in this container it is 13 s warm /
# 36 s cold (docs/timings.md, 2026-09-17), so that rationale is gone and per-variant jobs buy 10x
# parallelism, dbg-QOS eligibility for the short tasks, and a blast radius of one leg instead of ten.
#
# --seed IS LOAD-BEARING (D27): upstream generate_dataset.py seeds only from
# env.cfg.datagen_config.seed, so ten unseeded calls replay ONE pose stream and 400 demos collapse to
# ~43 unique initial poses. Verify every new arm with
#   python -m cog.analysis.gen_bias --levels <stem>     # UNIQUE ~= 400, no redundancy flag
set -uo pipefail

TASK="${1:?usage: datagen.sbatch TASK KEY STEM SEED TRIALS [NUM_ENVS]}"
KEY="${2:?missing KEY (gym id fragment)}"
STEM="${3:?missing STEM (output filename stem)}"
SEED="${4:?missing SEED}"
TRIALS="${5:?missing TRIALS}"
NUM_ENVS="${6:-8}"

source "${WORK:-/leonardo_work/EUHPC_B38_106}/cog/repo/slurm/lib_cog_container.sh"
cog_task_spec "${TASK}" || exit 2

SRC="${HDF5_DIR}/${SRC_STEM}.hdf5"
OUT="${HDF5_DIR}/${STEM}.hdf5"
ENVID="${GYM}-${KEY}-IK-Rel-Visuomotor-Mimic-v0"

echo "=== node $(hostname)  job ${SLURM_JOB_ID} ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "[gen] task=${TASK} env=${ENVID} stem=${STEM} seed=${SEED} trials=${TRIALS} envs=${NUM_ENVS}"
cog_require "${SIF}" "${SRC}" "${ASSET_ROOT}" || { echo "GEN_FAILED ${STEM}: missing prerequisite"; exit 1; }

# Resumable, but never silently (ported from scripts/ops/gen_L3_wave.sh): a complete file is
# skipped, a SHORT one aborts. Folding a short variant into the merged dataset would quietly
# unbalance the nested subsets, and nothing downstream would notice.
if [ -s "${OUT}" ]; then
  ND=$(cog_count_demos "${OUT}")
  if [ "${ND}" = "${TRIALS}" ]; then echo "GEN_SKIP ${STEM} (already ${TRIALS} demos)"; exit 0; fi
  echo "GEN_REFUSE ${STEM}: exists with ${ND} demos, expected ${TRIALS} or absent -- move it aside" >&2
  echo "GEN_REFUSE ${STEM}"
  exit 1
fi

mkdir -p "${HDF5_DIR}"
cog_job_scratch || exit 1
export COG_TIMEOUT=$(cog_timeout_s)

T0=$(date +%s)
echo "GEN_START $(date -Is)  timeout=${COG_TIMEOUT}s"
cog_gpu_exec python -u "${REPO_IN}/src/cog/datagen/vendored/generate_dataset.py" \
    --device cuda --task "${ENVID}" \
    --input_file "${SRC}" --output_file "${OUT}" \
    --seed "${SEED}" --generation_num_trials "${TRIALS}" --num_envs "${NUM_ENVS}" \
    --headless --enable_cameras \
    "--kit_args=\"${COG_KIT_ARGS}\""
RC=$?
T1=$(date +%s)

# Judge the ARTIFACT, never $? -- Kit exits 0 after fatal exceptions and drops its final progress
# flush (D16), so both the exit code and the log tail understate what happened.
ND=$(cog_count_demos "${OUT}")
NF=$(cog_count_demos "${HDF5_DIR}/${STEM}_failed.hdf5")
echo "GEN_WALL_SECONDS=$(( T1 - T0 )) rc=${RC}"
if [ "${ND}" = "${TRIALS}" ]; then
  echo "GEN_OK ${STEM} demos=${ND} failed=${NF} seed=${SEED} seconds=$(( T1 - T0 ))"
else
  echo "GEN_FAILED ${STEM} demos=${ND} expected=${TRIALS} failed=${NF} rc=${RC}"
  exit 1
fi
