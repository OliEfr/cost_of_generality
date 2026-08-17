#!/bin/bash
# T2 eval-set freeze: one Kit session per sub-level (a second gym.make in one
# session hangs). State envs boot in ~3 s, so the whole wave is ~2 min.
# timeout-kill covers the known Kit shutdown hang; EVALSET_OK is the success marker.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cog_isaac
cd "$(dirname "$0")/../.."
mkdir -p ops/eval_sets_raw_t2
for key in L0 L1 L2 L3v00 L3v01 L3v02 L3v03 L3v04 L3v05 L3v06 L3v07 L3v08 L3v09; do
  out="ops/eval_sets_raw_t2/${key}.json"
  timeout -k 30 900 python scripts/dev/freeze_eval_sets.py \
    --task "Cog-DrawerStow-${key}-IK-Rel-v0" --task_kind drawer_stow \
    --out "$out" --headless \
    > "ops/eval_sets_raw_t2/${key}.log" 2>&1
  rc=$?
  ok=$(grep -c EVALSET_OK "ops/eval_sets_raw_t2/${key}.log")
  echo "FREEZE_T2_${key}_EXIT=${rc} OK=${ok}"
done
echo FREEZE_T2_ALL_DONE
