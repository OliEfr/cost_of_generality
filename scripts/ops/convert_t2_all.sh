#!/bin/bash
# T2 HDF5 -> LeRobot conversion + validation.
#   convert_t2_all.sh            -> all four levels, sequentially
#   convert_t2_all.sh T2_L1      -> just that level (used to run levels in parallel;
#                                   h264 encode is single-core, the box has 32 threads)
#
# Inputs are listed EXPLICITLY, never globbed: RecorderManager writes
# `<name>_failed.hdf5` beside every `<name>.hdf5`, and a glob like T2_L3v0*.hdf5
# silently pulls in the failures (this cost us a bad 455-episode L3 dataset once --
# see docs/journal.md). The converter also refuses `_failed` names as a backstop.
#
# Absolute interpreter on purpose: a tmux/cron bash is non-interactive and has no
# conda shell function, so `conda activate` here fails and leaves `python` undefined.
set -u
cd /home/admin_07/cost_of_generality

PY=/home/admin_07/miniconda3/envs/cog_isaac/bin/python
if [ ! -x "$PY" ]; then echo "FATAL: no interpreter at $PY"; exit 1; fi
"$PY" -c "import lerobot, h5py" || { echo "FATAL: env missing lerobot/h5py"; exit 1; }

L3_INPUTS=(
  data/hdf5/T2_L3v00.hdf5 data/hdf5/T2_L3v01.hdf5 data/hdf5/T2_L3v02.hdf5
  data/hdf5/T2_L3v03.hdf5 data/hdf5/T2_L3v04.hdf5 data/hdf5/T2_L3v05.hdf5
  data/hdf5/T2_L3v06.hdf5 data/hdf5/T2_L3v07.hdf5 data/hdf5/T2_L3v08.hdf5
  data/hdf5/T2_L3v09.hdf5
)

inputs_for () {
  case "$1" in
    T2_L0) echo data/hdf5/T2_L0.hdf5 ;;
    T2_L1) echo data/hdf5/T2_L1.hdf5 ;;
    T2_L2) echo data/hdf5/T2_L2.hdf5 ;;
    # L3 pools all ten variants into one dataset; the converter interleaves them so
    # any nested-N prefix stays variant-balanced.
    T2_L3) echo "${L3_INPUTS[@]}" ;;
    *) echo ""; ;;
  esac
}

convert_and_validate () {
  local key="$1"
  local root="data/lerobot/${key}"
  local ins; ins=$(inputs_for "${key}")
  if [ -z "${ins}" ]; then echo "FATAL: unknown level ${key}"; return; fi
  echo "=== ${key} start $(date '+%F %H:%M:%S') ==="
  # shellcheck disable=SC2086
  "$PY" -m cog.convert.hdf5_to_lerobot --task drawer_stow \
    --input ${ins} --root "${root}" --repo_id "local/${key}" --fps 20
  local ce=$?
  echo "CONVERT_${key}_EXIT=${ce}"
  if [ "${ce}" -ne 0 ]; then echo "SKIP_VALIDATE_${key} (conversion failed)"; return; fi
  "$PY" -m cog.convert.validate_dataset --root "${root}" \
    --repo_id "local/${key}" --expect_episodes 400
  echo "VALIDATE_${key}_EXIT=$?"
  echo "=== ${key} end $(date '+%F %H:%M:%S') ==="
}

KEYS=("$@")
if [ "${#KEYS[@]}" -eq 0 ]; then KEYS=(T2_L0 T2_L1 T2_L2 T2_L3); fi
for k in "${KEYS[@]}"; do convert_and_validate "$k"; done
echo T2_CONVERT_DONE
