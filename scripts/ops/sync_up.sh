#!/bin/bash
# Push code + datasets to Leonardo. Datasets go to $FAST (read-hot, NVMe); code to $WORK.
#   scripts/ops/sync_up.sh code            # repo only (fast, safe to repeat)
#   scripts/ops/sync_up.sh datasets T1 T2  # LeRobot datasets for the named tasks
#
# Requires a live certificate (48 h; renew from the laptop with ~/cineca_login.sh).
set -euo pipefail
REMOTE=leonardo
# Overridable so a git-worktree checkout can be the source of truth for a sync. Hardcoding the
# main checkout meant a worktree's edits were silently NOT what reached the cluster -- the job
# would then run the OLD frozen config while the local branch showed the new one (caught
# 2026-08-19 while flipping use_separate_rgb_encoder_per_camera).
REPO="${COG_REPO:-/home/admin_07/cost_of_generality}"
echo "[sync_up] source repo: ${REPO}"
# rsync hands the destination path to the remote rsync as a plain argument; it is NEVER
# shell-expanded there, so a literal '$WORK' gets resolved against $HOME and the transfer
# dies with mkdir "...userexternal/ohausdoe/$WORK/cog/repo" failed (verified 2026-08-19).
# Resolve the bases locally in one ssh round-trip instead.
read -r _work_base _fast_base < <(ssh "${REMOTE}" 'echo "$WORK" "$FAST"')
if [ -z "${_work_base:-}" ] || [ -z "${_fast_base:-}" ]; then
  echo "could not resolve \$WORK/\$FAST on ${REMOTE} (no certificate, or no project association?)" >&2
  exit 3
fi
WORK_REMOTE="${_work_base}/cog"
FAST_REMOTE="${_fast_base}/cog/datasets"

what="${1:?usage: sync_up.sh code|datasets [TASK...]}"; shift || true

case "$what" in
  code)
    # data/ and third_party/ are excluded: datasets move separately and IsaacLab is
    # installed on the remote, not copied.
    # '*.out' protects Slurm job logs: sbatch's default --output=%x-%j.out is relative to the
    # SUBMIT directory, which is $WORK/cog/repo, so --delete would otherwise erase every job
    # log on the next code push. An --exclude'd path is skipped in BOTH directions, so
    # excluding it protects it from --delete (that is what --delete-excluded would override).
    # LEADING SLASHES ARE LOAD-BEARING: an rsync pattern with no '/' in it matches the
    # final path component at ANY depth, so a bare 'ops/' also ate scripts/ops/ and left
    # the cluster without launch_matrix.py (verified 2026-08-19). Anchor to the root.
    rsync -az --delete --info=stats1 \
      --exclude '/data/' --exclude '/third_party/' --exclude '.git/' \
      --exclude '/experiments/runs/' \
      --exclude '__pycache__/' --exclude '*.hdf5' --exclude '/ops/' \
      --exclude '*.out' \
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
