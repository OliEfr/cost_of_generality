#!/bin/bash
# Pull results back from Leonardo. Deliberately NARROW: result JSONs and the pruned
# best/last checkpoints only, never whole checkpoint trees (the plan budgets ~60 GB on the
# cluster after pruning, and a full pull would dwarf the local disk budget).
#   scripts/ops/sync_down.sh results        # eval JSONs + registry updates (small, frequent)
#   scripts/ops/sync_down.sh checkpoints T1_L0_n100_s0   # one run's best+last
set -euo pipefail
REMOTE=leonardo
# Overridable for the same reason as in sync_up.sh: a worktree-isolated session must be able to
# land its pulls in its own tree rather than silently in the main checkout.
REPO="${COG_REPO:-/home/admin_07/cost_of_generality}"
# Which checkpoint steps to pull. D24 changed the protocol to LAST-CHECKPOINT-ONLY for full-scale
# runs (the one-off 40k/60k/80k comparison is done, and the last checkpoint won), so the matrix
# needs 080000 alone -- pulling three would triple 3 GB/cell x 24 cells for nothing. Set
# COG_SYNC_STEPS="040000 060000 080000" to restore the best-of-3 pull for a comparison run.
COG_SYNC_STEPS="${COG_SYNC_STEPS:-080000}"
# See sync_up.sh: rsync never shell-expands the REMOTE path, so resolve $WORK here.
read -r _work_base < <(ssh "${REMOTE}" 'echo "$WORK"')
if [ -z "${_work_base:-}" ]; then
  echo "could not resolve \$WORK on ${REMOTE} (no certificate, or no project association?)" >&2
  exit 3
fi
WORK_REMOTE="${_work_base}/cog"

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
      # The protocol evaluates the LAST THREE checkpoints (40k/60k/80k) and reports
      # best-of-three, so all three must come down -- not just 080000. This previously listed
      # only 080000 and last, which would have silently starved the local-4090 eval fallback of
      # two thirds of its inputs and quietly turned best-of-3 into last-only.
      # Directory names are SIX digits (040000, not 00040000) -- verified against a real
      # checkpoint on 2026-08-19; docs/specs/06_lerobot_044.md shows 9 and is wrong.
      # Only pretrained_model/ is pulled: that is all eval loads. training_state/ is ~2 GB per
      # checkpoint of optimizer state used solely for RESUMING, which happens on the cluster, so
      # pulling it would triple the transfer (9 GB -> 3 GB per cell) for nothing.
      step_includes=()
      for s in ${COG_SYNC_STEPS}; do
        step_includes+=(--include "${s}/pretrained_model/**")
      done
      rsync -az --info=stats1 --prune-empty-dirs \
        --include '*/' \
        "${step_includes[@]}" \
        --exclude '*' \
        "${REMOTE}:${WORK_REMOTE}/checkpoints/${run}/" "${dest}/"
      echo "[sync] ${run} -> ${dest}"
    done
    ;;
  *) echo "unknown target ${what}"; exit 2 ;;
esac
echo "SYNC_DOWN_OK ${what}"
