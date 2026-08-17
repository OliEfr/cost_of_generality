#!/bin/bash
# T2 source demos + annotation (VERIFY d in the same chain).
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cog_isaac
cd "$(dirname "$0")/../.."
timeout -k 30 3600 python src/cog/datagen/record_drawer_source_demos.py \
  --task Cog-DrawerStow-L2-IK-Rel-v0 --num_envs 1 --num_demos 18 \
  --dataset_file data/hdf5/T2_L2_source.hdf5 --seed 7 --headless \
  > ops/t2_record_sources.log 2>&1
echo "RECORD_EXIT=$?"
grep -E "DONE|expert_SR" ops/t2_record_sources.log | tail -2
timeout -k 30 3600 python src/cog/datagen/vendored/annotate_demos.py \
  --device cuda --task Cog-DrawerStow-L2-IK-Rel-Mimic-v0 --auto \
  --input_file data/hdf5/T2_L2_source.hdf5 --output_file data/hdf5/T2_L2_source_annotated.hdf5 \
  --headless > ops/t2_annotate.log 2>&1
echo "ANNOTATE_EXIT=$?"
tail -3 ops/t2_annotate.log
echo T2_SOURCES_DONE
