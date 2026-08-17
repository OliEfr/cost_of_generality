#!/bin/bash
# T3 full datagen: L0/L1/L2 at 400 successes each + 10 L3 variants at 40 each.
# Visuomotor Mimic envs, same annotated L2 sources for every level (D9).
#
# isaaclab.sh requires `python` ON PATH (it shells out to it), and a tmux/cron bash has no
# conda function, so the first launch of this wave failed all 13 legs instantly with
# "python: command not found". Prepending the env's bin to PATH is more robust than
# `conda activate`, which needs the shell hook to have been sourced. Second time this exact
# trap has bitten (see the convert_t2_all.sh entry) -- hence doing it by PATH, not by conda.
#
# Estimated ~3-4 h total: episodes are 265-342 steps (a quarter of T2's) and generation SR
# is 93% (against T2's 31%), so a 400-demo level should take well under an hour.
set -u
export PATH="/home/admin_07/miniconda3/envs/cog_isaac/bin:$PATH"
command -v python >/dev/null || { echo "FATAL: no python on PATH"; exit 1; }
# Kit prompts "Do you accept the EULA? (Yes/No)" and BLOCKS FOREVER when it sees a TTY --
# which a tmux pane provides and a piped Bash-tool shell does not. That is exactly why the
# same command ran fine interactively and hung in tmux. Set the documented acceptance
# variable AND give every launch a null stdin, so no Kit prompt can stall a wave again.
export OMNI_KIT_ACCEPT_EULA=YES
cd /home/admin_07/cost_of_generality
for KEY in L0 L1 L2; do
  date
  ./third_party/IsaacLab/isaaclab.sh -p src/cog/datagen/vendored/generate_dataset.py \
    --device cuda --task "Cog-PushTarget-${KEY}-IK-Rel-Visuomotor-Mimic-v0" \
    --input_file data/hdf5/T3_L2_source_annotated.hdf5 \
    --output_file "data/hdf5/T3_${KEY}.hdf5" \
    --generation_num_trials 400 --num_envs 8 --headless --enable_cameras < /dev/null
  echo "GEN_T3_${KEY}_EXIT=$?"
done
for i in 0 1 2 3 4 5 6 7 8 9; do
  V=$(printf "L3v%02d" "$i")
  date
  ./third_party/IsaacLab/isaaclab.sh -p src/cog/datagen/vendored/generate_dataset.py \
    --device cuda --task "Cog-PushTarget-${V}-IK-Rel-Visuomotor-Mimic-v0" \
    --input_file data/hdf5/T3_L2_source_annotated.hdf5 \
    --output_file "data/hdf5/T3_${V}.hdf5" \
    --generation_num_trials 40 --num_envs 8 --headless --enable_cameras < /dev/null
  echo "GEN_T3_${V}_EXIT=$?"
done
echo T3_WAVES_DONE
