#!/bin/bash
# B1 gate 3: 300-step mtdit train on local/L1 (episodes 0..24, batch 16, pyav).
set -uo pipefail
WT=/home/admin_07/cost_of_generality/.claude/worktrees/lang-cand-b
source "$WT/configs/train/lang_dit_b.sh"
export PYTHONPATH="$WT/src"
export TOKENIZERS_PARALLELISM=false
EPISODES="$(/home/admin_07/miniconda3/envs/cog_isaac/bin/python -c 'print(",".join(str(i) for i in range(25)))')"
/home/admin_07/miniconda3/envs/cog_isaac/bin/python -m lerobot.scripts.lerobot_train \
  --policy.discover_packages_path=lerobot_policy_mtdit \
  --dataset.repo_id=local/L1 \
  --dataset.root=/home/admin_07/cost_of_generality/data/lerobot/L1 \
  --dataset.episodes="[${EPISODES}]" \
  --dataset.video_backend=pyav \
  --output_dir="$WT/experiments/runs/smoke_mtdit_300" \
  --job_name=smoke_mtdit_300 \
  --wandb.enable=false \
  ${COG_DIT_FLAGS} \
  --steps=300 \
  --save_freq=150 \
  --log_freq=25 \
  --batch_size=16
echo "SMOKE300_EXIT=$?"
