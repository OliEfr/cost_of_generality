#!/bin/bash
# T3 eval-set freeze: one Kit session per sub-level (a second gym.make in one session hangs).
# State envs boot in ~3 s, so the whole wave is ~2 min.
#
# Absolute interpreter (no conda function in a non-interactive shell), explicit EULA
# acceptance and null stdin (Kit blocks on its EULA prompt when it sees a TTY, e.g. in tmux).
set -u
export OMNI_KIT_ACCEPT_EULA=YES
PY=/home/admin_07/miniconda3/envs/cog_isaac/bin/python
cd "$(dirname "$0")/../.."
mkdir -p ops/eval_sets_raw_t3
for key in L0 L1 L2 L3v00 L3v01 L3v02 L3v03 L3v04 L3v05 L3v06 L3v07 L3v08 L3v09; do
  out="ops/eval_sets_raw_t3/${key}.json"
  timeout -k 30 900 "$PY" scripts/dev/freeze_eval_sets.py \
    --task "Cog-PushTarget-${key}-IK-Rel-v0" --task_kind push_target \
    --out "$out" --headless < /dev/null \
    > "ops/eval_sets_raw_t3/${key}.log" 2>&1
  rc=$?
  ok=$(grep -c EVALSET_OK "ops/eval_sets_raw_t3/${key}.log")
  echo "FREEZE_T3_${key}_EXIT=${rc} OK=${ok}"
  # Kit can exit 0 after a fatal exception, so EVALSET_OK -- not $? -- is the real signal.
  if [ "${ok}" -eq 0 ]; then echo "FREEZE_T3_${key}_FAILED (no EVALSET_OK marker)"; fi
done
echo FREEZE_T3_ALL_DONE
