#!/usr/bin/env bash
# Durable tmux layer for the rfix4 test (D33: renders instead of a warm-up batch).
#
# Counts ARTIFACTS, not sacct states (D16: a job can exit 0 without producing its output), and
# reports each failure once rather than on every pass. Self-terminates at 30/30 with an empty
# queue, so a finished wave does not leave a watcher running for days.
set -uo pipefail
REPO="/home/admin_07/cost_of_generality"
LOG="${REPO}/ops/rfix4_watch.log"
SEEN="${REPO}/ops/.rfix4_failed_seen"
IDS=$(awk -F, '/rfix4/ {print $13}' "${REPO}/experiments/cluster_jobs.csv" | sort -n | paste -sd, -)
touch "${SEEN}"

say() { echo "[$(date '+%Y-%m-%d %H:%M')] $*" | tee -a "${LOG}"; }
say "watching 30 rfix4 slices: ${IDS}"

while :; do
  DONE=$(timeout 60 ssh leonardo 'ls $WORK/cog/results/_partials/*_rfix4_s*.json 2>/dev/null | wc -l' 2>/dev/null || echo "?")
  QUEUED=$(timeout 60 ssh leonardo "squeue -j ${IDS} -h -o %T 2>/dev/null | wc -l" 2>/dev/null || echo "?")
  BAD=$(timeout 60 ssh leonardo "sacct -j ${IDS} -X -n -o JobID,State | awk '\$2!=\"COMPLETED\" && \$2!=\"PENDING\" && \$2!=\"RUNNING\" {print \$1, \$2}'" 2>/dev/null || true)
  if [ -n "${BAD}" ]; then
    NEW=$(comm -13 <(sort "${SEEN}") <(echo "${BAD}" | sort))
    if [ -n "${NEW}" ]; then
      say "NEW FAILURES:"; echo "${NEW}" | tee -a "${LOG}"
      echo "${BAD}" | sort > "${SEEN}"
    fi
  fi
  say "artifacts ${DONE}/30 | still queued ${QUEUED}"
  if [ "${DONE}" = "30" ] && [ "${QUEUED}" = "0" ]; then
    say "WAVE COMPLETE -- 30/30 artifacts, queue empty. Watcher exiting."
    break
  fi
  sleep 600
done
