#!/usr/bin/env bash
# Pool all 108 D31 cells: 18 (task, arm) pairs x 6 demo counts, ten slices each.
#
# Run on a Leonardo LOGIN node. pool_variant_eval.py is deliberately 3.6-compatible for exactly
# this (no torch, no Kit, no GPU -- it reads ten small JSONs and writes one), so the 108 calls are
# seconds of work and do not need to be 108 Slurm jobs with afterany dependencies. The plan's
# dependency-job design was written when the sweep was expected to take days; it drained in hours.
#
# Idempotent twice over: the pooler prints POOL_SKIP if the output already exists and exits 5 with
# INCOMPLETE if any of the ten slices is missing. So this can be run repeatedly while the sweep is
# still draining -- each run pools whatever has become complete and leaves the rest alone. That is
# the point: a cell is poolable the moment its tenth slice lands, not when the last cell finishes.
set -uo pipefail
REPO="${WORK}/cog/repo"
PART="${WORK}/cog/results/_partials"
RES="${WORK}/cog/results"
OK=0; SKIP=0; INC=0; BAD=0
for TASK in T1 T2 T3; do
  for ARM in L0 L1 L2 L3b AC BC; do
    for N in 10 25 50 100 200 400; do
      OUT=$(python3 "${REPO}/scripts/ops/pool_variant_eval.py" \
              --partials "${PART}" --results "${RES}" \
              --task "${TASK}" --arm "${ARM}" --n "${N}" --step 080000 --suffix u200 2>&1)
      RC=$?
      case "${RC}" in
        0) if echo "${OUT}" | grep -q POOL_SKIP; then SKIP=$((SKIP+1)); else OK=$((OK+1)); echo "${OUT}"; fi ;;
        5) INC=$((INC+1)); echo "${OUT}" | head -1 ;;
        *) BAD=$((BAD+1)); echo "POOL_ERROR ${TASK} ${ARM} n${N}: ${OUT}" ;;
      esac
    done
  done
done
echo "=== pooled=${OK} already_present=${SKIP} incomplete=${INC} errors=${BAD} of 108 ==="
[ "${BAD}" -eq 0 ] || exit 1
