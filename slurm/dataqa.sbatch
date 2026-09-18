#!/bin/bash
#SBATCH --job-name=cog_dataqa
#SBATCH --account=euhpc_b38_106
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# Dataset QA for generated arms: generation statistics (rule 9) and the D27 pose-redundancy check.
#
#   sbatch slurm/dataqa.sbatch AC BC T3_AC T3_BC
#
# NO --gres: both tools are h5py + numpy over the HDF5, no GPU. The container is still needed --
# the cluster's cog_lerobot env has no h5py -- and it cannot run on a LOGIN node, where apptainer
# fails with "maximum number of user namespaces exceeded". Hence a job.
#
# Why this is not optional (D27): ten generation legs seeded identically replay ONE pose stream, so
# 400 demos can carry 43 unique initial poses while every count-based check passes. gen_bias is the
# check that would have caught it, and D27 says it belongs in dataset QA rather than being run ad
# hoc. UNIQUE must be ~= the demo count, with no redundancy flag.
#
# Generation SR comes from the HDF5 episode counts, never from a generator log: carb drops the final
# progress flush, so log tails understate by up to 19 demos (D16).
set -uo pipefail

source "${WORK:-/leonardo_work/EUHPC_B38_106}/cog/repo/slurm/lib_cog_container.sh"
LEVELS=("$@")
[ "${#LEVELS[@]}" -gt 0 ] || { echo "usage: dataqa.sbatch LEVEL [LEVEL...]"; exit 2; }

COG_SCRATCH="${FAST}/cog/jobscratch/${SLURM_JOB_ID:-manual}"
mkdir -p "${COG_SCRATCH}"
trap 'rm -rf "${COG_SCRATCH}"' EXIT

echo "=== node $(hostname)  job ${SLURM_JOB_ID} ==="
echo "[qa] levels: ${LEVELS[*]}   hdf5: ${HDF5_DIR}"

echo
echo "=== gen_stats (rule 9: counts from the HDF5, never from the logs) ==="
cog_cpu_exec python -u "${REPO_IN}/scripts/dev/gen_stats.py" --out "${COG}/results/gen_stats_cluster.csv"
echo "GEN_STATS_EXIT=$?"

echo
echo "=== gen_bias (D27: unique initial poses per arm) ==="
cog_cpu_exec python -u -m cog.analysis.gen_bias --levels "${LEVELS[@]}" \
  --out "${COG}/results/gen_bias_cluster.csv"
echo "GEN_BIAS_EXIT=$?"
echo "DATAQA_DONE"
