#!/usr/bin/env bash
# Copy annotate/generate scripts from the IsaacLab checkout and insert the
# cog import (spec 02: custom task packages need one import line added).
set -euo pipefail
SRC=/home/admin_07/cost_of_generality/third_party/IsaacLab/scripts/imitation_learning/isaaclab_mimic
DST=/home/admin_07/cost_of_generality/src/cog/datagen/vendored
mkdir -p "$DST"
for f in annotate_demos.py generate_dataset.py; do
  cp "$SRC/$f" "$DST/$f"
  sed -i 's/^import isaaclab_mimic.envs  # noqa: F401$/import isaaclab_mimic.envs  # noqa: F401\nimport cog.tasks.cup_place  # noqa: F401  (COG: registers Cog-CupPlace-* env IDs)\nimport cog.tasks.drawer_stow  # noqa: F401  (COG: registers Cog-DrawerStow-* env IDs)\nimport cog.tasks.push_target  # noqa: F401  (COG: registers Cog-PushTarget-* env IDs)/' "$DST/$f"
  grep -q "cog.tasks.cup_place" "$DST/$f" || { echo "PATCH FAILED for $f"; exit 1; }
  grep -q "cog.tasks.drawer_stow" "$DST/$f" || { echo "PATCH FAILED for $f"; exit 1; }
  grep -q "cog.tasks.push_target" "$DST/$f" || { echo "PATCH FAILED for $f"; exit 1; }
done
echo "vendored scripts ready in $DST"
