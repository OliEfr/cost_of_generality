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
  ls $WORK/cog/results/_partials/*_u200_s*.json 2>/dev/null | wc -l
  echo "###POOLED"
  ls $WORK/cog/results/eval_*_u200.json 2>/dev/null | wc -l
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
  echo "-- NEW terminal states --"
  echo "$NEW" | sed 's/^/   /'
  BAD=$(echo "$NEW" | grep -cE "FAILED|TIMEOUT|NODE_FAIL|OUT_OF_ME")
  DONE=$(echo "$NEW" | grep -c "COMPLETED")
  [ "${BAD:-0}" -gt 0 ] && echo "- $TS  D31: $BAD job(s) ended badly -- see ops/status_d31.log" >> "$OPS/ALERTS.md"
  [ "${DONE:-0}" -gt 0 ] && echo "- $TS  D31: $DONE job(s) COMPLETED -- next phase may be unblocked" >> "$OPS/ALERTS.md"
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
echo "-- unblocked --"
[ "$NDS" -eq 6 ] && [ "$NCK" -lt 36 ] && echo "   T2 training can launch: launch_matrix.py --task T2 --levels AC BC"
[ "$NCK" -eq 36 ] && [ "${NPA:-0}" -eq 0 ] && echo "   EVAL SWEEP can launch (all 36 cells trained)"
[ "$NDS" -lt 6 ] && echo "   (waiting on conversion)"

sec DISK | sed 's/^/   disk /'
