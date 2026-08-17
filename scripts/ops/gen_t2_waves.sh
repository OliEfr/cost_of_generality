#!/bin/bash
# T2 full datagen: L0/L1/L2 at 400 successes each + 10 L3 variants at 40 each.
# Visuomotor Mimic envs, same annotated L2 sources for every level (D9).
# Estimated ~13-16 h total on the 4090 (long episodes x ~30% gen SR).
set -u
cd /home/admin_07/cost_of_generality
for KEY in L0 L1 L2; do
  date
  ./third_party/IsaacLab/isaaclab.sh -p src/cog/datagen/vendored/generate_dataset.py \
    --device cuda --task "Cog-DrawerStow-${KEY}-IK-Rel-Visuomotor-Mimic-v0" \
    --input_file data/hdf5/T2_L2_source_annotated.hdf5 \
    --output_file "data/hdf5/T2_${KEY}.hdf5" \
    --generation_num_trials 400 --num_envs 8 --headless --enable_cameras
  echo "GEN_T2_${KEY}_EXIT=$?"
done
for i in 0 1 2 3 4 5 6 7 8 9; do
  V=$(printf "L3v%02d" "$i")
  date
  ./third_party/IsaacLab/isaaclab.sh -p src/cog/datagen/vendored/generate_dataset.py \
    --device cuda --task "Cog-DrawerStow-${V}-IK-Rel-Visuomotor-Mimic-v0" \
    --input_file data/hdf5/T2_L2_source_annotated.hdf5 \
    --output_file "data/hdf5/T2_${V}.hdf5" \
    --generation_num_trials 40 --num_envs 8 --headless --enable_cameras
  echo "GEN_T2_${V}_EXIT=$?"
done
echo T2_WAVES_DONE
