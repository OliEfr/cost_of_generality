#!/usr/bin/env bash
# Durable tmux layer for the D31 eval sweep (CLAUDE.md rule 10, third layer).
#
# The event watcher (Monitor) expires after 30 min and the hourly cron only samples;
# this is the one that runs unattended for the whole sweep. It polls every 5 min and
# appends ONE line per tick to ops/eval_d31_progress.log, plus a line per NEW terminal
# failure -- transition-based, for the reason recorded in docs/running_jobs.md: a
# state-based grep re-fires every tick and trains the reader to ignore the channel.
#
# Counts the ARTIFACTS (_u200d32_s*.json), never the exit codes -- D16: a Kit job exits 0
# after a fatal error, so "COMPLETED" is not evidence that a slice was scored.
set -u
LOG=/home/admin_07/cost_of_generality/ops/eval_d31_progress.log
SEEN=/home/admin_07/cost_of_generality/ops/.eval_d31_failed_seen
TOTAL=1080
touch "$SEEN"
while :; do
  SNAP=$(timeout 120 ssh leonardo '
    echo "PARTIALS $(ls $WORK/cog/results/_partials/ 2>/dev/null | grep -c "_u200d32_s")"
    echo "POOLED $(ls $WORK/cog/results/ 2>/dev/null | grep -c "_u200d32\.json$")"
    echo "QUEUE $(squeue -u $USER -h -n cog_eval -o "%T" | sort | uniq -c | tr "\n" " ")"
    sacct -X -S now-3days -n -o JobID,State --name cog_eval 2>/dev/null \
      | awk "/FAILED|TIMEOUT|NODE_FAIL|OUT_OF_ME|CANCELLED/ {print \"BAD \" \$1 \" \" \$2}"
  ' 2>/dev/null)
  TS=$(date "+%Y-%m-%d %H:%M")
  P=$(echo "$SNAP" | awk '/^PARTIALS/{print $2}')
  echo "$TS  slices ${P:-?}/$TOTAL  pooled $(echo "$SNAP" | awk '/^POOLED/{print $2}')  | $(echo "$SNAP" | grep '^QUEUE' | cut -d' ' -f2-)" >> "$LOG"
  echo "$SNAP" | awk '/^BAD/{print $2}' | sort -u > /tmp/.eval_d31_bad.$$
  NEW=$(comm -13 "$SEEN" /tmp/.eval_d31_bad.$$)
  [ -n "$NEW" ] && echo "$TS  NEW FAILURES: $(echo $NEW | tr '\n' ' ')" >> "$LOG"
  sort -u "$SEEN" /tmp/.eval_d31_bad.$$ > "$SEEN.tmp" && mv "$SEEN.tmp" "$SEEN"
  rm -f /tmp/.eval_d31_bad.$$
  # Stop once every slice has an artifact AND nothing is left in the queue.
  if [ "${P:-0}" -ge "$TOTAL" ] && [ -z "$(echo "$SNAP" | grep '^QUEUE' | cut -d' ' -f2-)" ]; then
    echo "$TS  EVAL SWEEP COMPLETE -- $P/$TOTAL slices" >> "$LOG"; break
  fi
  sleep 300
done
