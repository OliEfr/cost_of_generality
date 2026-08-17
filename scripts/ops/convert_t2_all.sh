#!/bin/bash
# T2 HDF5 -> LeRobot conversion + validation for all four levels.
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

convert_and_validate () {
  local key="$1"; shift
  local root="data/lerobot/${key}"
  echo "=== ${key} start $(date '+%F %H:%M:%S') ==="
  "$PY" -m cog.convert.hdf5_to_lerobot --task drawer_stow \
    --input "$@" --root "${root}" --repo_id "local/${key}" --fps 20
  local ce=$?
  echo "CONVERT_${key}_EXIT=${ce}"
  if [ "${ce}" -ne 0 ]; then
    echo "SKIP_VALIDATE_${key} (conversion failed)"
    return
  fi
  "$PY" -m cog.convert.validate_dataset --root "${root}" \
    --repo_id "local/${key}" --expect_episodes 400
  echo "VALIDATE_${key}_EXIT=$?"
  echo "=== ${key} end $(date '+%F %H:%M:%S') ==="
}

convert_and_validate T2_L0 data/hdf5/T2_L0.hdf5
convert_and_validate T2_L1 data/hdf5/T2_L1.hdf5
convert_and_validate T2_L2 data/hdf5/T2_L2.hdf5
# L3 = one dataset pooling all ten variants; the converter interleaves them so any
# nested-N prefix stays variant-balanced.
convert_and_validate T2_L3 \
  data/hdf5/T2_L3v00.hdf5 data/hdf5/T2_L3v01.hdf5 data/hdf5/T2_L3v02.hdf5 \
  data/hdf5/T2_L3v03.hdf5 data/hdf5/T2_L3v04.hdf5 data/hdf5/T2_L3v05.hdf5 \
  data/hdf5/T2_L3v06.hdf5 data/hdf5/T2_L3v07.hdf5 data/hdf5/T2_L3v08.hdf5 \
  data/hdf5/T2_L3v09.hdf5
echo T2_CONVERT_DONE
