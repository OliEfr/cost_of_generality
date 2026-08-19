#!/bin/bash
#SBATCH --job-name=cog_smoke_blr
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
# G5a part 1: pick batch size + LR by measurement, then freeze them for all 24 cells.
#   sbatch slurm/smoke_batch_lr.sbatch
# Sweeps batch 64/128/256 with sqrt-scaled LR for a few hundred steps each and reports
# samples/s, peak VRAM and GPU utilization. Also discharges the D22 VERIFY item (cu128
# wheels on a CUDA-12.2 driver). ~0.5 GPU-h. Writes $WORK/cog/results/g5a_batch_lr.csv.
set -euo pipefail

COG="${WORK}/cog"
RES="${COG}/results"; mkdir -p "${RES}"
CSV="${RES}/g5a_batch_lr.csv"
DS_NAME="${COG_SMOKE_DS:-L0}"
DS="${COG_DATASETS:-$FAST/cog/datasets}/${DS_NAME}"
NEPS="${COG_SMOKE_NEPS:-25}"
STEPS="${COG_SMOKE_STEPS:-200}"

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HF_HOME="${COG}/hf_cache"
export TOKENIZERS_PARALLELISM=false

source "${COG}/miniforge3/etc/profile.d/conda.sh"
conda activate "${COG_TRAIN_ENV:-cog_lerobot}"
source "${COG}/repo/configs/train/diffusion_base.sh"

echo "=== node ==="; hostname; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# D22 VERIFY: the cu128 wheels are newer than this driver's CUDA 12.2 and are expected to
# work only through CUDA minor version compatibility. Prove it with a real GPU matmul BEFORE
# spending the sweep, so a failure costs 30 s and not 30 min. Fallback is torch 2.7.0+cu126.
python - <<'PY'
import sys, torch
print("[gpu] torch", torch.__version__, "| cuda build", torch.version.cuda)
print("[gpu] is_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("[gpu] D22 VERIFY FAILED: no CUDA device visible", file=sys.stderr); sys.exit(1)
print("[gpu] device", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
a = torch.randn(4096, 4096, device="cuda"); b = torch.randn(4096, 4096, device="cuda")
c = (a @ b).sum().item()
torch.cuda.synchronize()
assert c == c, "matmul produced NaN"          # NaN != NaN
print(f"[gpu] matmul ok, checksum {c:.4e}")
print("[gpu] D22_VERIFY_PASSED")
PY

# --steps/--save_freq/--log_freq come from the frozen config; strip them so the smoke can
# override without relying on draccus's last-flag-wins behaviour (which is not documented).
SMOKE_FLAGS="$(printf '%s\n' ${COG_DP_FLAGS} \
  | grep -v -E '^--steps=|^--save_freq=|^--log_freq=' | tr '\n' ' ')"

EPISODES="$(python - "$NEPS" <<'PY'
import sys
print(",".join(str(i) for i in range(int(sys.argv[1]))))
PY
)"

echo "batch,lr,steps,wall_s,samples_per_s,steps_per_s,peak_vram_mib,median_util_pct,final_loss" > "${CSV}"

for BATCH in 64 128 256; do
  # sqrt scaling from the 1e-4 @ 64 baseline (plan: P5 / configs/train/diffusion_base.sh)
  LR="$(python -c "import math; print(f'{1e-4*math.sqrt(${BATCH}/64):.6g}')")"
  OUT="${COG}/checkpoints/_smoke_blr_b${BATCH}_${SLURM_JOB_ID}"
  LOG="${RES}/g5a_b${BATCH}_${SLURM_JOB_ID}.log"
  UTIL="${RES}/g5a_b${BATCH}_${SLURM_JOB_ID}.util"
  echo "=== batch=${BATCH} lr=${LR} steps=${STEPS} ==="

  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -l 2 > "${UTIL}" 2>/dev/null &
  SMI_PID=$!

  T0=$(date +%s)
  set +e
  python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="local/${DS_NAME}" \
    --dataset.root="${DS}" \
    --dataset.episodes="[${EPISODES}]" \
    --dataset.video_backend="${COG_VIDEO_BACKEND:-pyav}" \
    --output_dir="${OUT}" \
    --job_name="smoke_b${BATCH}" \
    --batch_size="${BATCH}" \
    --policy.optimizer_lr="${LR}" \
    --steps="${STEPS}" --log_freq=25 --save_checkpoint=false \
    --wandb.enable=false \
    ${SMOKE_FLAGS} > "${LOG}" 2>&1
  RC=$?
  set -e
  T1=$(date +%s)
  kill "${SMI_PID}" 2>/dev/null || true
  wait "${SMI_PID}" 2>/dev/null || true

  WALL=$((T1-T0))
  if [ "${RC}" -ne 0 ]; then
    echo "  FAILED rc=${RC}; last lines:"; tail -15 "${LOG}"
    echo "${BATCH},${LR},${STEPS},${WALL},FAILED,FAILED,FAILED,FAILED,FAILED" >> "${CSV}"
    continue
  fi

  python - "${BATCH}" "${LR}" "${STEPS}" "${WALL}" "${LOG}" "${UTIL}" "${CSV}" <<'PY'
import re, sys, statistics
batch, lr, steps, wall, log, util, csv = sys.argv[1:]
batch, steps, wall = int(batch), int(steps), int(wall)
txt = open(log, errors="replace").read()
# LeRobot logs lines like "step:200 smpl:12K ... loss:0.123 ...". Take the last loss seen.
losses = re.findall(r"loss:\s*([0-9.eE+-]+)", txt)
final = losses[-1] if losses else "NA"
mem, ut = [], []
for line in open(util, errors="replace"):
    p = [x.strip() for x in line.split(",")]
    if len(p) == 2 and p[0].isdigit() and p[1].isdigit():
        ut.append(int(p[0])); mem.append(int(p[1]))
peak = max(mem) if mem else "NA"
# Drop the first few samples: they cover dataset/video-decoder startup, not training.
med = int(statistics.median(ut[3:])) if len(ut) > 6 else (int(statistics.median(ut)) if ut else "NA")
sps = steps*batch/wall if wall else 0
stps = steps/wall if wall else 0
row = f"{batch},{lr},{steps},{wall},{sps:.1f},{stps:.3f},{peak},{med},{final}"
open(csv, "a").write(row+"\n")
print("  " + row)
PY
  rm -rf "${OUT}"
done

echo "=== $CSV ==="
cat "${CSV}"
echo "SMOKE_BATCH_LR_DONE"
