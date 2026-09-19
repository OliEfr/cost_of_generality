#!/usr/bin/env bash
# D31 reverse-ablation wave status. Prints a compact, ACTIONABLE report:
# what changed since the last run, what is running, and what is now unblocked.
#
# Run from cron (hourly) and by hand. State lives in ops/status_d31.state so the
# "NEW" section only reports transitions -- an hourly report that reprints the same
# 40 lines is one nobody reads, which is how a finished wave goes unnoticed.
set -u
REPO=/home/admin_07/cost_of_generality
OPS=$REPO/ops
STATE=$OPS/status_d31.state
TS=$(date '+%Y-%m-%d %H:%M')
touch "$STATE"

if ! ssh -o BatchMode=yes -o ConnectTimeout=20 leonardo true 2>/dev/null; then
  echo "$TS  D31: ssh leonardo unavailable (cert expired?) -- cannot check"
  echo "- $TS  D31 status check could not reach leonardo (renew cert: ssh st07; ~/cineca_login.sh)" >> "$OPS/ALERTS.md"
  exit 0
fi

SNAP=$(ssh -o BatchMode=yes leonardo '
  echo "###QUEUE"
  squeue -u $USER -h -o "%j %T" | sort | uniq -c
  echo "###TERMINAL"
  sacct -X -S $(date -d "7 days ago" +%F) -n -o JobID,JobName%14,State 2>/dev/null \
    | grep -E "cog_train|cog_eval|cog_convert" | awk "{print \$1, \$2, \$3}"
  echo "###CKPT"
  for t in t1 t2 t3; do for l in AC BC; do for n in 10 25 50 100 200 400; do
    [ -d "$WORK/cog/checkpoints/${t}_${l}_n${n}_s0/checkpoints/080000/pretrained_model" ] && echo "CKPT ${t}_${l}_n${n}_s0"
  done; done; done
  echo "###DATASETS"
  # VALIDATE_OK, not the directory: hdf5_to_lerobot creates meta/ on the first episode, so a
  # still-encoding dataset looks finished on disk. The marker is the only completion signal
  # (same reason rule 9 reads gen_stats.csv rather than generator logs).
  grep -h "VALIDATE_OK" $WORK/cog/logs/cog_convert-*.out 2>/dev/null \
    | awk "{print \$2}" | grep -E "^(AC|BC|T2_AC|T2_BC|T3_AC|T3_BC)$" | sort -u | sed "s/^/DS /"
  echo "###PARTIALS"
  ls $WORK/cog/results/_partials/*_u200d32_s*.json 2>/dev/null | wc -l
  echo "###POOLED"
  ls $WORK/cog/results/eval_*_u200d32.json 2>/dev/null | wc -l
  echo "###DISK"
  df -BG --output=target,avail $WORK $FAST 2>/dev/null | tail -2
' 2>/dev/null)

sec() { echo "$SNAP" | sed -n "/^###$1\$/,/^###/p" | grep -v '^###'; }

echo "=== $TS  D31 wave status ==="

# --- terminal-state transitions since last run (the part that must never be missed)
sec TERMINAL | grep -E "COMPLETED|FAILED|TIMEOUT|CANCELLED|NODE_FAIL|OUT_OF_ME" | sort > /tmp/.d31_term.$$
# First run seeds the state silently: every job that ever finished is not news, and an
# opening burst of 39 "COMPLETED" alerts teaches the reader to skip the channel.
if [ ! -s "$STATE" ]; then
  cat /tmp/.d31_term.$$ > "$STATE"; rm -f /tmp/.d31_term.$$
  echo "-- state seeded ($(wc -l < "$STATE") historical jobs); reporting transitions from here --"
  sec QUEUE | sed 's/^/   /'
  exit 0
fi
NEW=$(comm -13 <(sort "$STATE") /tmp/.d31_term.$$)
if [ -n "$NEW" ]; then
  BAD=$(echo "$NEW" | grep -E "FAILED|TIMEOUT|NODE_FAIL|OUT_OF_ME")
  NBAD=$(echo "$NEW" | grep -cE "FAILED|TIMEOUT|NODE_FAIL|OUT_OF_ME")
  NDONE=$(echo "$NEW" | grep -c "COMPLETED")
  # COMPLETED is SUMMARISED, the bad states are LISTED. During the eval sweep an hour is
  # 100+ finished slices, and printing each one buries the single FAILED line in the middle
  # of a screen of good news -- the same alert-fatigue failure the transition check was
  # added to fix, just relocated from "every hour" to "every line". The COMPLETED job ids
  # stay in $STATE and in ops/status_d31.log; only the terminal display is condensed.
  echo "-- NEW terminal states --"
  [ "${NDONE:-0}" -gt 0 ] && echo "   ${NDONE} COMPLETED (ids in $STATE)"
  # CANCELLED was collected into the state file and reported in NEITHER bucket -- not counted as
  # done, not listed as bad -- so a scheduler-side cancellation (preemption, a node drain, an admin
  # kill) vanished silently. Deliberate cancellations are rare enough that listing them costs
  # nothing, and the one thing this check exists to prevent is a terminal state nobody sees.
  NCAN=$(echo "$NEW" | grep -c "CANCELLED")
  [ "${NCAN:-0}" -gt 0 ] && { echo "   ${NCAN} CANCELLED:"; echo "$NEW" | grep "CANCELLED" | head -5 | sed 's/^/     /'; }
  [ "${NBAD:-0}" -gt 0 ] && { echo "   ${NBAD} ended BADLY:"; echo "$BAD" | sed 's/^/     /'; }
  [ "${NBAD:-0}" -gt 0 ] && echo "- $TS  D31: $NBAD job(s) ended badly -- see ops/status_d31.log" >> "$OPS/ALERTS.md"
  # No ALERTS.md line for COMPLETED during a sweep: 1,080 slices finishing normally is the
  # expected case, not an event. The "-- unblocked --" section below is what says a phase
  # is ready, and it reads artifacts rather than job states.
else
  echo "-- no new terminal states --"
fi
cat /tmp/.d31_term.$$ > "$STATE"; rm -f /tmp/.d31_term.$$

# --- what is in flight
echo "-- queue --"; sec QUEUE | sed 's/^/   /'

# --- phase progress
NDS=$(sec DATASETS | wc -l); NCK=$(sec CKPT | wc -l)
NPA=$(sec PARTIALS | tr -dc '0-9'); NPO=$(sec POOLED | tr -dc '0-9')
echo "-- progress --"
echo "   convert  ${NDS}/6 datasets"
echo "   train    ${NCK}/36 cells at step 080000"
echo "   eval     ${NPA:-0}/1080 slices, ${NPO:-0}/108 pooled"

# --- what is unblocked right now
# "Unblocked" means SUBMITTED-ness, not trained-ness. Keying the T2 line on checkpoints instead
# kept it lit for the whole training wave and told the next hourly tick to re-launch cells that
# were already queued. The registry is the record of what has been submitted -- launch_matrix
# writes a row per cell at submit time -- so ask it, not the filesystem.
NSUB=$(python3 -c "
import csv
rows = list(csv.DictReader(open('$REPO/experiments/registry.csv')))
print(sum(1 for r in rows if '_AC_n' in r['run_id'] or '_BC_n' in r['run_id']))
" 2>/dev/null || echo 0)
echo "-- unblocked --"
echo "   submitted ${NSUB}/36 cells"
if [ "$NDS" -lt 6 ]; then
  echo "   (waiting on conversion)"
elif [ "${NSUB:-0}" -lt 36 ]; then
  echo "   training can launch for the missing cells: launch_matrix.py --task T{1,2,3} --levels AC BC"
elif [ "$NCK" -lt 36 ]; then
  echo "   (waiting on training: ${NCK}/36 at step 080000)"
elif [ "${NPA:-0}" -eq 0 ]; then
  echo "   EVAL SWEEP can launch (all 36 cells trained) -- ~276 GPU-h, ASK FIRST"
elif [ "${NPA:-0}" -lt 1080 ]; then
  echo "   (eval in flight: ${NPA}/1080 slices)"
fi

sec DISK | sed 's/^/   disk /'
