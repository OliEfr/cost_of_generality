#!/bin/bash
# T3 HDF5 -> LeRobot conversion + validation.
#   convert_t3_all.sh            -> all four levels, sequentially
#   convert_t3_all.sh T3_L1      -> just that level (levels are run in PARALLEL: h264
#                                   encode is single-core and the box has 32 threads;
#                                   measured on T2, 4x throughput at unchanged per-level speed)
#
# Inputs listed EXPLICITLY, never globbed: RecorderManager writes `<name>_failed.hdf5`
# beside every `<name>.hdf5`, and a glob once produced a bad 455-episode dataset.
#
# Absolute interpreter on purpose: a tmux/cron bash is non-interactive and has no conda
# shell function, so `conda activate` fails and leaves `python` undefined.
set -u
cd /home/admin_07/cost_of_generality

PY=/home/admin_07/miniconda3/envs/cog_isaac/bin/python
if [ ! -x "$PY" ]; then echo "FATAL: no interpreter at $PY"; exit 1; fi
"$PY" -c "import lerobot, h5py" || { echo "FATAL: env missing lerobot/h5py"; exit 1; }

L3_INPUTS=(
  data/hdf5/T3_L3v00.hdf5 data/hdf5/T3_L3v01.hdf5 data/hdf5/T3_L3v02.hdf5
  data/hdf5/T3_L3v03.hdf5 data/hdf5/T3_L3v04.hdf5 data/hdf5/T3_L3v05.hdf5
  data/hdf5/T3_L3v06.hdf5 data/hdf5/T3_L3v07.hdf5 data/hdf5/T3_L3v08.hdf5
  data/hdf5/T3_L3v09.hdf5
)

inputs_for () {
  case "$1" in
    T3_L0) echo data/hdf5/T3_L0.hdf5 ;;
    T3_L1) echo data/hdf5/T3_L1.hdf5 ;;
    T3_L2) echo data/hdf5/T3_L2.hdf5 ;;
    T3_L3) echo "${L3_INPUTS[@]}" ;;
    *) echo "" ;;
  esac
}

convert_and_validate () {
  local key="$1"
  local root="data/lerobot/${key}"
  local ins; ins=$(inputs_for "${key}")
  if [ -z "${ins}" ]; then echo "FATAL: unknown level ${key}"; return; fi
  echo "=== ${key} start $(date '+%F %H:%M:%S') ==="
  # shellcheck disable=SC2086
  "$PY" -m cog.convert.hdf5_to_lerobot --task push_target \
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
if [ "${#KEYS[@]}" -eq 0 ]; then KEYS=(T3_L0 T3_L1 T3_L2 T3_L3); fi
for k in "${KEYS[@]}"; do convert_and_validate "$k"; done
echo T3_CONVERT_DONE
