#!/bin/bash
# T3 full datagen: L0/L1/L2 at 400 successes each + 10 L3 variants at 40 each.
# Visuomotor Mimic envs, same annotated L2 sources for every level (D9).
#
# isaaclab.sh resolves its own interpreter, so no conda activation is needed here (unlike
# the pure-python drivers -- see the convert_t2_all.sh note about non-interactive shells).
#
# Estimated ~3-4 h total: episodes are 265-342 steps (a quarter of T2's) and generation SR
# is 93% (against T2's 31%), so a 400-demo level should take well under an hour.
set -u
cd /home/admin_07/cost_of_generality
for KEY in L0 L1 L2; do
  date
  ./third_party/IsaacLab/isaaclab.sh -p src/cog/datagen/vendored/generate_dataset.py \
    --device cuda --task "Cog-PushTarget-${KEY}-IK-Rel-Visuomotor-Mimic-v0" \
    --input_file data/hdf5/T3_L2_source_annotated.hdf5 \
    --output_file "data/hdf5/T3_${KEY}.hdf5" \
    --generation_num_trials 400 --num_envs 8 --headless --enable_cameras
  echo "GEN_T3_${KEY}_EXIT=$?"
done
for i in 0 1 2 3 4 5 6 7 8 9; do
  V=$(printf "L3v%02d" "$i")
  date
  ./third_party/IsaacLab/isaaclab.sh -p src/cog/datagen/vendored/generate_dataset.py \
    --device cuda --task "Cog-PushTarget-${V}-IK-Rel-Visuomotor-Mimic-v0" \
    --input_file data/hdf5/T3_L2_source_annotated.hdf5 \
    --output_file "data/hdf5/T3_${V}.hdf5" \
    --generation_num_trials 40 --num_envs 8 --headless --enable_cameras
  echo "GEN_T3_${V}_EXIT=$?"
done
echo T3_WAVES_DONE
