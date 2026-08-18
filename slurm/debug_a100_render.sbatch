#!/bin/bash
#SBATCH --job-name=cog_a100_render
#SBATCH --account=euhpc_b38_106
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# GATE G5b: does Isaac Sim render on an A100 at all? Isaac Sim is officially unsupported
# there (no RT cores; issues #3421 crash / #1519 degraded rendering), and the documented
# workaround is antialiasing OFF -- which our visuomotor cfgs already set.
#
# PASS  -> demo generation and evaluation can also run on the cluster.
# FAIL  -> sim stages stay on the local 4090 (accepted fallback); training is unaffected
#          either way, so this gate never blocks the matrix.
set -uo pipefail

WORK="${COG_WORK:-$WORK/cog}"
export OMNI_KIT_ACCEPT_EULA=YES     # Kit blocks forever on the EULA prompt under a TTY
IMG="${WORK}/containers/isaaclab.sif"

echo "[G5b] node=$(hostname) gpu:"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# 8 episodes of the frozen T1 L0 visuomotor env: enough to prove RGB tensors come back
# non-degenerate, which is the only question this gate asks.
srun singularity exec --nv \
  -B "${WORK}:${WORK}" -B /leonardo_work:/leonardo_work \
  "${IMG}" bash -lc "
    export OMNI_KIT_ACCEPT_EULA=YES
    cd ${WORK}/repo && python scripts/dev/frames_qa.py --level L0 --resets 2 \
      --headless --enable_cameras < /dev/null
  "
RC=$?
echo "[G5b] frames_qa exit=${RC}"
# Kit can exit 0 after a fatal exception, so the artifact is the real signal, not $?.
if [ -s "${WORK}/repo/ops/qa/frames_L0.png" ]; then
  echo "G5B_PASSED: rendered frames written"
else
  echo "G5B_FAILED: no frame artifact -- keep sim stages local"
fi
