#!/bin/bash
# One Kit session per (sub-)level (a second gym.make in one session hangs).
# timeout-kill covers the known Kit shutdown hang; EVALSET_OK is the success marker.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cog_isaac
cd "$(dirname "$0")/../.."
mkdir -p ops/eval_sets_raw
for key in L0 L1 L2 L3v00 L3v01 L3v02 L3v03 L3v04 L3v05 L3v06 L3v07 L3v08 L3v09; do
  out="ops/eval_sets_raw/${key}.json"
  timeout -k 30 900 python scripts/dev/freeze_eval_sets.py \
    --task "Cog-CupPlace-${key}-IK-Rel-v0" --out "$out" --headless \
    > "ops/eval_sets_raw/${key}.log" 2>&1
  rc=$?
  ok=$(grep -c EVALSET_OK "ops/eval_sets_raw/${key}.log")
  echo "FREEZE_${key}_EXIT=${rc} OK=${ok}"
done
echo FREEZE_ALL_DONE
