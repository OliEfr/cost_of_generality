#!/usr/bin/env bash
# L3 generation wave: 10 sub-variant envs x 40 successes = 400 demos total (D2).
# Same annotated L2 sources for provenance control (D9).
set -u
source /home/admin_07/miniconda3/etc/profile.d/conda.sh
conda activate cog_isaac
export PYTHONUNBUFFERED=1
cd /home/admin_07/cost_of_generality/third_party/IsaacLab
for i in 0 1 2 3 4 5 6 7 8 9; do
  V=$(printf "L3v%02d" "$i")
  date
  ./isaaclab.sh -p /home/admin_07/cost_of_generality/src/cog/datagen/vendored/generate_dataset.py \
    --task "Cog-CupPlace-${V}-IK-Rel-Visuomotor-Mimic-v0" \
    --input_file /home/admin_07/cost_of_generality/data/hdf5/L2_source_annotated.hdf5 \
    --output_file "/home/admin_07/cost_of_generality/data/hdf5/${V}.hdf5" \
    --generation_num_trials 40 --num_envs 8 --headless --enable_cameras
  echo "GEN_${V}_EXIT=$?"
  date
done
echo L3_WAVE_DONE
