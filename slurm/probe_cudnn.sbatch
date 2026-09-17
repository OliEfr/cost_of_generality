#!/bin/bash
#SBATCH --job-name=cog_cudnn
#SBATCH --account=euhpc_b38_106
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# The A100 eval benchmark (58046658) renders fine but dies in the policy's ResNet with
# CUDNN_STATUS_NOT_INITIALIZED. Training never exercised this path on the cluster (it ran
# in the $WORK conda env, not the .sif), so torch-conv-in-container is untested here.
# Leg A = plain conv, no Kit, no extralibs. Leg B = same + the /opt/extralibs LD_LIBRARY_PATH
# prepend the eval uses (libgomp shim), to see whether that shim shadows a cuDNN dependency.
set -uo pipefail
COG="${WORK}/cog"
APPTAINER=/leonardo/prod/opt/tools/foldspace/1.0/apptainer/bin/apptainer
SIF="${COG}/containers/cog-env-5.1.0.sif"
PROBE="${COG}/repo/scripts/dev/cudnn_probe.py"
ISAAC_HOME="${COG}/isaac_home"

echo "=== node $(hostname) ==="; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo; echo "=== A) container, NO Kit, NO extralibs ==="
"${APPTAINER}" exec --nv -B "${COG}:${COG}" --home "${ISAAC_HOME}" \
  "${SIF}" python3 -u "${PROBE}" < /dev/null || echo "leg A rc=$?"

echo; echo "=== B) container + extralibs on LD_LIBRARY_PATH (as the eval runs it) ==="
"${APPTAINER}" exec --nv -B "${COG}:${COG}" -B "${COG}/extralibs:/opt/extralibs" \
  --home "${ISAAC_HOME}" \
  "${SIF}" bash -c "export LD_LIBRARY_PATH=/opt/extralibs:\${LD_LIBRARY_PATH}; python3 -u ${PROBE}" \
  < /dev/null || echo "leg B rc=$?"

echo "CUDNN_PROBE_JOB_DONE"
