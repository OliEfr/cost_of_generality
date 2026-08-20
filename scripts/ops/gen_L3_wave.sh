#!/usr/bin/env bash
# L3 generation wave: 10 sub-variant envs x 40 successes = 400 demos total (D2).
# Same annotated L2 sources for provenance control (D9).
#
# --seed IS LOAD-BEARING (D27). Without it every variant run replays the same pose stream, because
# upstream generate_dataset.py seeds only from env.cfg.datagen_config.seed. The first L3 wave was
# generated that way and produced 400 demos over just 43 UNIQUE initial poses instead of 400, so
# L3's demo axis was ~9x redundant and its success rate plateaued at a pose-coverage ceiling that
# looked like a generality ceiling. Seeds are offset by task and kept away from the eval seeds
# (5000-5009, configs/eval_sets/protocol.json) so training and eval poses stay disjoint.
# Verify after regenerating:  python -m cog.analysis.gen_bias --levels L3   (UNIQUE must be ~400)
set -u
source /home/admin_07/miniconda3/etc/profile.d/conda.sh
conda activate cog_isaac
export PYTHONUNBUFFERED=1
SEED_BASE="${COG_GEN_SEED_BASE:-1000}"    # T1=1000, T2=2000, T3=3000
# Output prefix is overridable so the CORRECTED arm can be generated without overwriting the
# pose-redundant one, which we keep as a fixed-N pose-diversity ablation (D27) and which is the
# provenance of 6 already-trained cells. The gym env id is unchanged either way -- only the seed and
# the output filename differ.
OUT_PREFIX="${COG_L3_OUT_PREFIX:-L3v}"
REPO="${COG_REPO:-/home/admin_07/cost_of_generality}"
DATA="${COG_DATA:-/home/admin_07/cost_of_generality/data/hdf5}"
echo "[gen] code from ${REPO}  data to ${DATA}"
cd /home/admin_07/cost_of_generality/third_party/IsaacLab
for i in 0 1 2 3 4 5 6 7 8 9; do
  V=$(printf "L3v%02d" "$i")             # env id fragment: always the registered L3v** variant
  OUT=$(printf "%s%02d" "${OUT_PREFIX}" "$i")   # output file stem: L3v00 or e.g. L3bv00
  if [ -s "${DATA}/${OUT}.hdf5" ]; then
    echo "REFUSING to overwrite existing ${OUT}.hdf5 -- set COG_L3_OUT_PREFIX to a fresh prefix." >&2
    exit 1
  fi
  date
  ./isaaclab.sh -p "${REPO}/src/cog/datagen/vendored/generate_dataset.py" \
    --task "Cog-CupPlace-${V}-IK-Rel-Visuomotor-Mimic-v0" \
    --input_file "${DATA}/L2_source_annotated.hdf5" \
    --output_file "${DATA}/${OUT}.hdf5" \
    --seed "$((SEED_BASE + i))" \
    --generation_num_trials 40 --num_envs 8 --headless --enable_cameras
  echo "GEN_${OUT}_EXIT=$?"
  date
done
echo L3_WAVE_DONE
