#!/usr/bin/env bash
# Hourly health watcher for cost_of_generality. Installed in user crontab.
# Appends to ops/watchdog.log (cron) and raises issues in ops/ALERTS.md.
set -u
REPO=/home/admin_07/cost_of_generality
OPS=$REPO/ops
TS=$(date '+%Y-%m-%d %H:%M')
alert() { echo "- $TS  $1" >> "$OPS/ALERTS.md"; echo "ALERT: $1"; }
note()  { echo "$TS  $1"; }

mkdir -p "$OPS"
touch "$OPS/ALERTS.md"

# 1. Leonardo certificate status (never blocks; jobs unaffected by cert lapse)
CERT_OUT=$(/home/admin_07/cineca_login.sh --status 2>&1 || true)
HRS=$(echo "$CERT_OUT" | grep -oP '\d+(?=h)' | head -1)
if echo "$CERT_OUT" | grep -qi 'expired\|not found\|no certificate'; then
  alert "Leonardo cert EXPIRED/missing - renew from laptop: ssh st07 then ~/cineca_login.sh"
elif [ -n "${HRS:-}" ] && [ "$HRS" -lt 12 ]; then
  alert "Leonardo cert <${HRS}h remaining - renew from laptop soon (ssh st07; ~/cineca_login.sh)"
else
  note "cert ok (${HRS:-?}h)"
fi

# 2. Cluster checks (only if ssh works)
if ssh -o BatchMode=yes -o ConnectTimeout=15 leonardo true 2>/dev/null; then
  # G0: Slurm association watch (until it passes once)
  if [ ! -f "$OPS/G0_PASSED" ]; then
    if ssh -o BatchMode=yes leonardo "sacctmgr -nP show assoc user=ohausdoe format=account" 2>/dev/null | grep -q euhpc_b38_106; then
      date > "$OPS/G0_PASSED"
      alert "G0 PASSED: Slurm association with euhpc_b38_106 is LIVE - cluster phases unblocked!"
    else
      note "G0 pending (no b38 association yet)"
    fi
  fi
  # Job status snapshot
  ssh -o BatchMode=yes leonardo "squeue --me -o '%.10i %.20j %.8T %.10M %.6D %R' 2>/dev/null; sacct -X -S \$(date -d '2 days ago' +%F) --format=JobID,JobName%20,State,Elapsed -n 2>/dev/null | tail -20" \
      > "$OPS/cluster_status.txt" 2>/dev/null
  # Failed jobs in last 2 days?
  if grep -qE 'FAILED|NODE_FAIL|OUT_OF_ME' "$OPS/cluster_status.txt" 2>/dev/null; then
    alert "cluster jobs in FAILED/NODE_FAIL state - see ops/cluster_status.txt"
  fi
  # Daily budget snapshot (first run after midnight)
  if [ ! -f "$OPS/saldo.txt" ] || [ -n "$(find "$OPS/saldo.txt" -mtime +0 2>/dev/null)" ]; then
    ssh -o BatchMode=yes leonardo "saldo -b 2>/dev/null" > "$OPS/saldo.txt" 2>/dev/null || true
  fi
else
  note "ssh leonardo unavailable (cert expired or network); skipping cluster checks"
fi

# 3. Local disk
AVAIL_GB=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
if [ "${AVAIL_GB:-0}" -lt 80 ]; then
  alert "local disk low: ${AVAIL_GB}G free - ask user for cleanup (no-delete rule)"
else
  note "disk ok (${AVAIL_GB}G free)"
fi

# 4. Local GPU + our datagen tmux sessions
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader > "$OPS/gpu_status.txt" 2>/dev/null || true
for s in $(tmux ls 2>/dev/null | grep '^cog_' | cut -d: -f1); do
  note "tmux session alive: $s"
done
