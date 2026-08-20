#!/usr/bin/env bash
# L3 generation wave for ONE task: 10 sub-variant envs x 40 successes = 400 demos total (D2).
# Same annotated L2 sources for provenance control (D9).
#
#   gen_L3_wave.sh [T1|T2|T3]           # default T1
#
# L3-ONLY BY DESIGN. gen_t2_waves.sh / gen_t3_waves.sh regenerate L0-L2 as well and have no
# no-clobber guard, so using them to redo L3 would overwrite four valid datasets. This script touches
# nothing but the ten L3 variant files, and refuses to overwrite even those.
#
# --seed IS LOAD-BEARING (D27). Without it every variant run replays the same pose stream, because
# upstream generate_dataset.py seeds only from env.cfg.datagen_config.seed. The first L3 wave was
# generated that way and produced 400 demos over just 43-48 UNIQUE initial poses instead of 400, so
# L3's demo axis was ~9x redundant and its success rate plateaued at a pose-coverage ceiling that
# looked like a generality ceiling. Seeds are offset per task and kept away from the eval seeds
# (5000-5009, configs/eval_sets/protocol.json) so training and eval poses stay disjoint.
#
# Verify after regenerating:  python -m cog.analysis.gen_bias --levels <L3 stem>
# UNIQUE must be ~400 and no redundancy flag.
set -uo pipefail
TASK="${1:-T1}"

# code follows COG_REPO (a worktree's generator fix must actually run); data stays in the main
# checkout, which is shared and gitignored and already holds the existing datasets.
REPO="${COG_REPO:-/home/admin_07/cost_of_generality}"
MAIN=/home/admin_07/cost_of_generality
DATA="${COG_DATA:-${MAIN}/data/hdf5}"

case "${TASK}" in
  T1) GYM=Cog-CupPlace;   SRC="${DATA}/L2_source_annotated.hdf5";    STEM=L3v;    SEED_BASE=1000 ;;
  T2) GYM=Cog-DrawerStow; SRC="${DATA}/T2_L2_source_annotated.hdf5"; STEM=T2_L3v; SEED_BASE=2000 ;;
  T3) GYM=Cog-PushTarget; SRC="${DATA}/T3_L2_source_annotated.hdf5"; STEM=T3_L3v; SEED_BASE=3000 ;;
  *) echo "usage: gen_L3_wave.sh [T1|T2|T3]" >&2; exit 2 ;;
esac
# Output stem is overridable so a CORRECTED arm can be generated without overwriting the
# pose-redundant one, which is the provenance of already-trained cells and is kept as a fixed-N
# pose-diversity ablation (D27). The gym env id never changes -- only the seed and the filename.
STEM="${COG_L3_OUT_STEM:-${STEM}}"

source /home/admin_07/miniconda3/etc/profile.d/conda.sh
conda activate cog_isaac
export PYTHONUNBUFFERED=1
export OMNI_KIT_ACCEPT_EULA=YES HF_HUB_OFFLINE=1
echo "[gen] task=${TASK} gym=${GYM} stem=${STEM} seeds=${SEED_BASE}..$((SEED_BASE + 9))"
echo "[gen] code=${REPO}  data=${DATA}  src=${SRC}"
[ -s "${SRC}" ] || { echo "missing source demos: ${SRC}" >&2; exit 3; }

cd "${MAIN}"
for i in 0 1 2 3 4 5 6 7 8 9; do
  V=$(printf "L3v%02d" "$i")                # env id fragment: the registered L3v** variant
  OUT=$(printf "%s%02d" "${STEM}" "$i")     # output stem, e.g. L3v00 or L3bv00
  if [ -s "${DATA}/${OUT}.hdf5" ]; then
    echo "REFUSING to overwrite existing ${OUT}.hdf5 -- set COG_L3_OUT_STEM to a fresh stem." >&2
    exit 1
  fi
  date
  ./third_party/IsaacLab/isaaclab.sh -p "${REPO}/src/cog/datagen/vendored/generate_dataset.py" \
    --device cuda --task "${GYM}-${V}-IK-Rel-Visuomotor-Mimic-v0" \
    --input_file "${SRC}" \
    --output_file "${DATA}/${OUT}.hdf5" \
    --seed "$((SEED_BASE + i))" \
    --generation_num_trials 40 --num_envs 8 --headless --enable_cameras < /dev/null
  echo "GEN_${OUT}_EXIT=$?"
  date
done
echo "L3_WAVE_DONE_${TASK}"
