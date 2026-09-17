#!/bin/bash
#SBATCH --job-name=cog_freeze
#SBATCH --account=euhpc_b38_106
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# Freeze the eval-set snapshots for ONE arm (D11).  sbatch slurm/freeze_eval_sets.sbatch TASK ARM
#
#   sbatch slurm/freeze_eval_sets.sbatch T2 AC      # -> ACv00..ACv09, 10 Kit sessions
#   sbatch slurm/freeze_eval_sets.sbatch T1 L2      # -> L2, 1 Kit session
#
# STATE env, so --enable_cameras is deliberately absent: no RTX, no rendering, ~3 s per Kit boot
# locally and 13-36 s here. The GPU is still required -- freeze_eval_sets.py builds the env on
# args_cli.device and the poses come off the CUDA generator, so freezing on the CPU would produce a
# different, non-comparable benchmark. The Vulkan binds are kept because they are a harmless
# superset of the one verified recipe, and dropping them would be an untested change.
#
# One Kit session PER SUB-LEVEL: a second gym.make in the same app hangs (docs/timings.md).
# Raw snapshots land under $WORK/cog/results/_evalsets_raw/<TASK>/ -- the leading underscore keeps
# them clear of the eval_*.json globs in curves.py and update_registry_from_evals.py. They are
# MERGED LOCALLY after sync_down, so the git-tracked configs/eval_sets/*.json are written in the
# checkout rather than in the rsync mirror.
set -uo pipefail

TASK="${1:?usage: freeze_eval_sets.sbatch TASK ARM}"
ARM="${2:?missing ARM}"

source "${WORK:-/leonardo_work/EUHPC_B38_106}/cog/repo/slurm/lib_cog_container.sh"
cog_task_spec "${TASK}" || exit 2
case "${TASK}" in
  T1) TASK_KIND=cup_place ;; T2) TASK_KIND=drawer_stow ;; T3) TASK_KIND=push_target ;;
esac

case "${ARM}" in
  AC|BC|L3|L3b) KEYS=$(for i in $(seq -w 0 9); do printf "%sv%s " "${ARM}" "${i}"; done) ;;
  *)            KEYS="${ARM}" ;;
esac

RAW="${RESULTS_DIR}/_evalsets_raw/${TASK}"
mkdir -p "${RAW}"
cog_require "${SIF}" "${ASSET_ROOT}" || { echo "FREEZE_FAILED ${TASK} ${ARM}: missing prerequisite"; exit 1; }
cog_job_scratch || exit 1
export COG_TIMEOUT=900

echo "=== node $(hostname)  job ${SLURM_JOB_ID} ==="
echo "[freeze] task=${TASK} arm=${ARM} keys: ${KEYS}"

fails=0
for key in ${KEYS}; do
  if [ -s "${RAW}/${key}.json" ]; then echo "FREEZE_SKIP ${key} (snapshot exists)"; continue; fi
  cog_gpu_exec python -u "${REPO_IN}/scripts/dev/freeze_eval_sets.py" \
      --task "${GYM}-${key}-IK-Rel-v0" --task_kind "${TASK_KIND}" \
      --out "${RAW}/${key}.json" --headless \
      "--kit_args=\"${COG_KIT_ARGS}\"" > "${RAW}/${key}.log" 2>&1
  # Kit exits 0 after fatal exceptions, so the marker -- not $? -- is the verdict.
  if [ "$(grep -c EVALSET_OK "${RAW}/${key}.log")" -gt 0 ] && [ -s "${RAW}/${key}.json" ]; then
    echo "FREEZE_OK ${key}"
  else
    echo "FREEZE_FAILED ${key} (no EVALSET_OK marker; see ${RAW}/${key}.log)"
    fails=$((fails + 1))
  fi
done

if [ "${fails}" -eq 0 ]; then
  echo "FREEZE_ARM_OK ${TASK} ${ARM}"
else
  echo "FREEZE_ARM_FAILED ${TASK} ${ARM} (${fails} sub-level(s))"
  exit 1
fi
