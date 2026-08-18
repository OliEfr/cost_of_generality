#!/bin/bash
# Pull results back from Leonardo. Deliberately NARROW: result JSONs and the pruned
# best/last checkpoints only, never whole checkpoint trees (the plan budgets ~60 GB on the
# cluster after pruning, and a full pull would dwarf the local disk budget).
#   scripts/ops/sync_down.sh results        # eval JSONs + registry updates (small, frequent)
#   scripts/ops/sync_down.sh checkpoints T1_L0_n100_s0   # one run's best+last
set -euo pipefail
REMOTE=leonardo
REPO=/home/admin_07/cost_of_generality
WORK_REMOTE='$WORK/cog'

what="${1:?usage: sync_down.sh results|checkpoints [RUN_ID...]}"; shift || true

case "$what" in
  results)
    mkdir -p "${REPO}/results"
    rsync -az --info=stats1 "${REMOTE}:${WORK_REMOTE}/results/" "${REPO}/results/"
    echo "[sync] $(ls "${REPO}/results" | wc -l) result files locally"
    ;;
  checkpoints)
    for run in "$@"; do
      dest="${REPO}/experiments/runs/${run}"
      mkdir -p "${dest}"
      # only the two checkpoints the protocol needs; --prune-empty-dirs keeps it tidy
      rsync -az --info=stats1 --prune-empty-dirs \
        --include '*/' --include '080000/**' --include 'last/**' --exclude '*' \
        "${REMOTE}:${WORK_REMOTE}/checkpoints/${run}/" "${dest}/"
      echo "[sync] ${run} -> ${dest}"
    done
    ;;
  *) echo "unknown target ${what}"; exit 2 ;;
esac
echo "SYNC_DOWN_OK ${what}"
