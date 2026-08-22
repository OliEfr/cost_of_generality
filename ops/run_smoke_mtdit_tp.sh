#!/bin/bash
# B1 gate 5: throughput probe -- 100-step mtdit runs at batch 16/32/64 on local/L1,
# with a 1 Hz nvidia-smi per-process VRAM sampler. Skips a batch size if <10 GiB free.
set -uo pipefail
WT=/home/admin_07/cost_of_generality/.claude/worktrees/lang-cand-b
source "$WT/configs/train/lang_dit_b.sh"
export PYTHONPATH="$WT/src"
export TOKENIZERS_PARALLELISM=false
PY=/home/admin_07/miniconda3/envs/cog_isaac/bin/python
EPISODES="$($PY -c 'print(",".join(str(i) for i in range(25)))')"
for B in 16 32 64; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$FREE" -lt 10240 ]; then echo "TP_SKIP_b${B} free=${FREE}MiB"; continue; fi
  ( while true; do nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits; sleep 1; done > "$WT/ops/tp_vram_b${B}.samples" ) &
  SAMPLER=$!
  $PY -m lerobot.scripts.lerobot_train \
    --policy.discover_packages_path=lerobot_policy_mtdit \
    --dataset.repo_id=local/L1 \
    --dataset.root=/home/admin_07/cost_of_generality/data/lerobot/L1 \
    --dataset.episodes="[${EPISODES}]" \
    --dataset.video_backend=pyav \
    --output_dir="$WT/experiments/runs/smoke_mtdit_tp_b${B}" \
    --job_name="smoke_mtdit_tp_b${B}" \
    --wandb.enable=false \
    ${COG_DIT_FLAGS} \
    --steps=100 \
    --save_freq=100 \
    --log_freq=20 \
    --batch_size="$B" > "$WT/ops/tp_train_b${B}.log" 2>&1
  echo "TP_EXIT_b${B}=$?"
  kill "$SAMPLER" 2>/dev/null
done
echo TPPROBE_DONE
