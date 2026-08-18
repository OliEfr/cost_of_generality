#!/bin/bash
# Push code + datasets to Leonardo. Datasets go to $FAST (read-hot, NVMe); code to $WORK.
#   scripts/ops/sync_up.sh code            # repo only (fast, safe to repeat)
#   scripts/ops/sync_up.sh datasets T1 T2  # LeRobot datasets for the named tasks
#
# Requires a live certificate (48 h; renew from the laptop with ~/cineca_login.sh).
set -euo pipefail
REMOTE=leonardo
REPO=/home/admin_07/cost_of_generality
WORK_REMOTE='$WORK/cog'
FAST_REMOTE='$FAST/cog/datasets'

what="${1:?usage: sync_up.sh code|datasets [TASK...]}"; shift || true

case "$what" in
  code)
    # data/ and third_party/ are excluded: datasets move separately and IsaacLab is
    # installed on the remote, not copied.
    rsync -az --delete --info=stats1 \
      --exclude 'data/' --exclude 'third_party/' --exclude '.git/' \
      --exclude '__pycache__/' --exclude '*.hdf5' --exclude 'ops/' \
      "${REPO}/" "${REMOTE}:${WORK_REMOTE}/repo/"
    ;;
  datasets)
    for task in "$@"; do
      case "$task" in
        T1) sets="L0 L1 L2 L3" ;;
        T2) sets="T2_L0 T2_L1 T2_L2 T2_L3" ;;
        T3) sets="T3_L0 T3_L1 T3_L2 T3_L3" ;;
        *) echo "unknown task ${task}"; exit 2 ;;
      esac
      for s in $sets; do
        echo "[sync] ${s}"
        rsync -az --info=stats1 "${REPO}/data/lerobot/${s}/" \
          "${REMOTE}:${FAST_REMOTE}/${s}/"
      done
    done
    ;;
  *) echo "unknown target ${what}"; exit 2 ;;
esac
echo "SYNC_UP_OK ${what}"
