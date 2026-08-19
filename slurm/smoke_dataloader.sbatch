#!/bin/bash
#SBATCH --job-name=cog_smoke_dl
#SBATCH --account=euhpc_b38_106
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# G5a part 2: the batch/LR sweep showed the loop is DECODE-BOUND (batch 64: data_s 0.458 s
# vs updt_s 0.071 s, GPU ~87% idle, 80k steps ~= 11.8 h). Batch size cannot fix that. This
# job measures the two levers that can:
#   (a) more dataloader workers -- needs no code or version change
#   (b) a newer LeRobot (py3.12 env cog_lerobot06) -- claims a faster dataloader
#
#   sbatch slurm/smoke_dataloader.sbatch cog_lerobot   "8 16 24" pyav
#   sbatch slurm/smoke_dataloader.sbatch cog_lerobot06 "8 24"    pyav
#
# 32 CPUs are requested so num_workers can actually scale (billing scales with cores: this
# job bills 32 core-h per hour instead of 8, i.e. ~1.3 GPU-h-equivalents for 20 minutes).
# Batch is FIXED at 64 -- the point is steps/s at constant batch, not another batch sweep.
set -euo pipefail

ENV_NAME="${1:?usage: smoke_dataloader.sbatch ENV_NAME \"NW LIST\" BACKEND}"
NW_LIST="${2:-8 16 24}"
BACKEND="${3:-pyav}"

COG="${WORK}/cog"
RES="${COG}/results"; mkdir -p "${RES}"
CSV="${RES}/g5a_dataloader.csv"
DS_NAME="${COG_SMOKE_DS:-L0}"
DS="${COG_DATASETS:-$FAST/cog/datasets}/${DS_NAME}"
NEPS="${COG_SMOKE_NEPS:-25}"
STEPS="${COG_SMOKE_STEPS:-150}"
BATCH=64

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HF_HOME="${COG}/hf_cache"
export TOKENIZERS_PARALLELISM=false

source "${COG}/miniforge3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
source "${COG}/repo/configs/train/diffusion_base.sh"

LR_VER="$(python -c 'import lerobot; print(lerobot.__version__)')"
echo "=== node $(hostname) | env ${ENV_NAME} | lerobot ${LR_VER} | backend ${BACKEND} ==="
nproc
# The entry point moved between versions (0.4.4 renamed scripts.train -> lerobot_train), so
# resolve it instead of assuming; a wrong guess here would look like a performance result.
ENTRY=""
for cand in lerobot.scripts.lerobot_train lerobot.scripts.train; do
  if python -c "import importlib,sys; importlib.import_module('${cand}')" 2>/dev/null; then
    ENTRY="${cand}"; break
  fi
done
if [ -z "${ENTRY}" ]; then echo "NO TRAIN ENTRY POINT FOUND in ${ENV_NAME}" >&2; exit 3; fi
echo "entry: ${ENTRY}"

# num_workers MUST be stripped: it is in the frozen config, draccus takes the LAST
# occurrence of a flag, and ${SMOKE_FLAGS} is expanded AFTER --num_workers below. Leaving it
# in made every arm of the first run silently execute at num_workers=8 while reporting 8/16/24
# -- three identical rows that looked like "more workers does not help". Verified by grepping
# the resolved config in the run log (num_workers': 8 in the "nw=16" arm).
SMOKE_FLAGS="$(printf '%s\n' ${COG_DP_FLAGS} \
  | grep -v -E '^--steps=|^--save_freq=|^--log_freq=|^--num_workers=' | tr '\n' ' ')"
EPISODES="$(python - "$NEPS" <<'PY'
import sys
print(",".join(str(i) for i in range(int(sys.argv[1]))))
PY
)"

if [ ! -f "${CSV}" ]; then
  echo "env,lerobot,backend,num_workers,batch,steps,updt_s,data_s,steady_steps_per_s,est_80k_h,peak_vram_mib,median_util_pct,final_loss" > "${CSV}"
fi

for NW in ${NW_LIST}; do
  OUT="${COG}/checkpoints/_smoke_dl_${ENV_NAME}_${BACKEND}_nw${NW}_${SLURM_JOB_ID}"
  LOG="${RES}/g5a_dl_${ENV_NAME}_${BACKEND}_nw${NW}_${SLURM_JOB_ID}.log"
  UTIL="${RES}/g5a_dl_${ENV_NAME}_${BACKEND}_nw${NW}_${SLURM_JOB_ID}.util"
  echo "=== ${ENV_NAME} ${BACKEND} num_workers=${NW} ==="

  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 2 > "${UTIL}" 2>/dev/null &
  SMI_PID=$!
  set +e
  python -m "${ENTRY}" \
    --dataset.repo_id="local/${DS_NAME}" \
    --dataset.root="${DS}" \
    --dataset.episodes="[${EPISODES}]" \
    --dataset.video_backend="${BACKEND}" \
    --output_dir="${OUT}" \
    --job_name="smoke_dl_nw${NW}" \
    --batch_size="${BATCH}" \
    --num_workers="${NW}" \
    --steps="${STEPS}" --log_freq=25 --save_checkpoint=false \
    --wandb.enable=false \
    ${SMOKE_FLAGS} > "${LOG}" 2>&1
  RC=$?
  set -e
  kill "${SMI_PID}" 2>/dev/null || true; wait "${SMI_PID}" 2>/dev/null || true

  if [ "${RC}" -ne 0 ]; then
    echo "  FAILED rc=${RC}; tail:"; tail -20 "${LOG}"
    echo "${ENV_NAME},${LR_VER},${BACKEND},${NW},${BATCH},${STEPS},FAILED,FAILED,FAILED,FAILED,FAILED,FAILED,rc${RC}" >> "${CSV}"
    continue
  fi

  # Trust the RESOLVED config, not the flag we passed. LeRobot echoes the full config; if the
  # requested num_workers or backend did not survive flag precedence, the timing number is
  # meaningless and must not be written to the CSV as if it were real.
  GOT_NW="$(grep -o "num_workers': *[0-9]*" "${LOG}" | head -1 | grep -o '[0-9]*$' || true)"
  if [ -n "${GOT_NW}" ] && [ "${GOT_NW}" != "${NW}" ]; then
    echo "  ABORT: requested num_workers=${NW} but the resolved config says ${GOT_NW}"
    echo "${ENV_NAME},${LR_VER},${BACKEND},${NW},${BATCH},${STEPS},NWMISMATCH${GOT_NW},NWMISMATCH,NA,NA,NA,NA,NA" >> "${CSV}"
    continue
  fi

  python - "${ENV_NAME}" "${LR_VER}" "${BACKEND}" "${NW}" "${BATCH}" "${STEPS}" "${LOG}" "${UTIL}" "${CSV}" <<'PY'
import re, sys, statistics
env, ver, backend, nw, batch, steps, log, util, csv = sys.argv[1:]
txt = open(log, errors="replace").read()
# Use the LAST logged interval, not the whole run: the first intervals include dataloader
# worker spin-up and page-cache warming, which would flatter a low num_workers arm.
upd = [float(x) for x in re.findall(r"updt_s:([0-9.]+)", txt)]
dat = [float(x) for x in re.findall(r"data_s:([0-9.]+)", txt)]
loss = re.findall(r"loss:\s*([0-9.eE+-]+)", txt)
if not upd or not dat:
    open(csv, "a").write(f"{env},{ver},{backend},{nw},{batch},{steps},NOLOG,NOLOG,NOLOG,NOLOG,NA,NA,NA\n")
    print("  no timing lines parsed"); raise SystemExit(0)
u, d = upd[-1], dat[-1]
step_s = u + d
sps = 1.0/step_s if step_s else 0.0
est = 80000*step_s/3600.0
mem, ut = [], []
for line in open(util, errors="replace"):
    p = [x.strip() for x in line.split(",")]
    if len(p) == 2 and p[0].isdigit() and p[1].isdigit():
        ut.append(int(p[0])); mem.append(int(p[1]))
peak = max(mem) if mem else "NA"
med = int(statistics.median(ut[3:])) if len(ut) > 6 else (int(statistics.median(ut)) if ut else "NA")
row = (f"{env},{ver},{backend},{nw},{batch},{steps},{u:.3f},{d:.3f},"
       f"{sps:.2f},{est:.1f},{peak},{med},{loss[-1] if loss else 'NA'}")
open(csv, "a").write(row+"\n")
print("  " + row)
PY
  rm -rf "${OUT}"
done

echo "=== ${CSV} ==="; cat "${CSV}"
echo "SMOKE_DATALOADER_DONE"
