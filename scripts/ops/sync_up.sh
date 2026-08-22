#!/bin/bash
# Push code + datasets to Leonardo. Datasets go to $FAST (read-hot, NVMe); code to $WORK.
#   scripts/ops/sync_up.sh code            # repo only (fast, safe to repeat)
#   scripts/ops/sync_up.sh datasets T1 T2  # LeRobot datasets for the named tasks
#   scripts/ops/sync_up.sh hf              # CLIP ViT-B/16 HF cache -> $WORK/cog/hf_cache/hub
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
    # Datasets live in the MAIN checkout's data/ (gitignored, shared between worktrees), NOT under
    # COG_REPO -- pointing COG_REPO at a worktree for the code sync would otherwise make this rsync
    # an empty source directory, which silently uploads nothing.
    LEROBOT="${COG_DATA_LEROBOT:-/home/admin_07/cost_of_generality/data/lerobot}"
    for task in "$@"; do
      case "$task" in
        T1) sets="L0 L1 L2 L3" ;;
        T2) sets="T2_L0 T2_L1 T2_L2 T2_L3" ;;
        T3) sets="T3_L0 T3_L1 T3_L2 T3_L3" ;;
        # anything else is taken as a literal dataset name, so a single regenerated arm can be
        # pushed on its own: sync_up.sh datasets L3b T2_L3b
        *) sets="$task" ;;
      esac
      for s in $sets; do
        [ -d "${LEROBOT}/${s}" ] || { echo "[sync] MISSING ${LEROBOT}/${s}" >&2; exit 3; }
        echo "[sync] ${s} <- ${LEROBOT}/${s}"
        rsync -az --info=stats1 "${LEROBOT}/${s}/" \
          "${REMOTE}:${FAST_REMOTE}/${s}/"
      done
    done
    ;;
  hf)
    # Stage the CLIP ViT-B/16 weights + tokenizer for candidate B (multi_task_dit):
    # compute nodes are offline (HF_HUB_OFFLINE=1) and train_lang_dit.sbatch points
    # HF_HOME at $WORK/cog/hf_cache, so the model must be pre-staged in hub/ layout.
    # No --delete: the remote cache may hold other models we did not stage.
    HF_MODEL_DIR="${COG_HF_CACHE:-$HOME/.cache/huggingface}/hub/models--openai--clip-vit-base-patch16"
    [ -d "${HF_MODEL_DIR}" ] || { echo "[sync] MISSING ${HF_MODEL_DIR} (download CLIP locally first)" >&2; exit 3; }
    echo "[sync] hf <- ${HF_MODEL_DIR}"
    rsync -az --info=stats1 "${HF_MODEL_DIR}" \
      "${REMOTE}:${WORK_REMOTE}/hf_cache/hub/"
    ;;
  *) echo "unknown target ${what}"; exit 2 ;;
esac
echo "SYNC_UP_OK ${what}"
