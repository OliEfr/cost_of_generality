"""Assert that every SUB_LEVELS key of every task registers its four gym ids, and that each
reverse-ablation arm really has the dimension it claims to have removed switched OFF.

Why this needs a Kit boot: `assets.py` imports `isaaclab.sim`, which imports `carb`, which only
exists once a SimulationApp has started. A state-only app boots in ~3 s (docs/timings.md) and needs
no cameras and no Vulkan, so this is cheap enough to run on every levels.py edit.

Verdict goes to a FILE as well as stdout: Kit's fastShutdown discards buffered stdout, so a
print-only check reports nothing at all (observed here, and the same reason
scripts/dev/render_smoke_offline.py writes rtx_smoke.txt).

usage: python scripts/dev/check_levels.py [--out ops/check_levels.txt]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--out", default="ops/check_levels.txt")
_args, _ = _pre.parse_known_args()
_OUT = pathlib.Path(_args.out)
_OUT.parent.mkdir(parents=True, exist_ok=True)
_OUT.write_text("")


def emit(msg: str) -> None:
    print(msg, flush=True)
    with _OUT.open("a") as fh:
        fh.write(msg + "\n")

from isaacsim import SimulationApp

_app = SimulationApp({"headless": True})

import gymnasium as gym  # noqa: E402

import cog.tasks.cup_place  # noqa: E402,F401  (registration side effect)
import cog.tasks.drawer_stow  # noqa: E402,F401
import cog.tasks.push_target  # noqa: E402,F401
from cog.tasks.cup_place.levels import SUB_LEVELS as T1  # noqa: E402
from cog.tasks.drawer_stow.levels import SUB_LEVELS as T2  # noqa: E402
from cog.tasks.push_target.levels import SUB_LEVELS as T3  # noqa: E402

SUFFIXES = ("IK-Rel-v0", "IK-Rel-Visuomotor-v0", "IK-Rel-Mimic-v0", "IK-Rel-Visuomotor-Mimic-v0")
TASKS = (("Cog-CupPlace", T1), ("Cog-DrawerStow", T2), ("Cog-PushTarget", T3))

# (removed-dimension accessor, kept-dimension accessor) per task, as (field, how to read a range)
DEGENERATE = {
    "Cog-CupPlace": {
        "AC": ("goal_pose_range", "cup_pose_range"),
        "BC": ("cup_pose_range", "goal_pose_range"),
    },
    "Cog-DrawerStow": {
        "AC": ("cabinet_pose_range", "object_pose_range"),
        "BC": ("object_pose_range", "cabinet_pose_range"),
    },
    "Cog-PushTarget": {
        "AC": ("bearing_range", "puck_pose_range"),
        "BC": ("puck_pose_range", "bearing_range"),
    },
}


def _spans(rng) -> list[float]:
    """Width of every axis of a pose-range dict, or of a (lo, hi) bearing tuple."""
    if isinstance(rng, dict):
        return [hi - lo for lo, hi in rng.values()]
    return [rng[1] - rng[0]]


def main() -> int:
    bad: list[str] = []

    for prefix, levels in TASKS:
        for key in levels:
            missing = [s for s in SUFFIXES if f"{prefix}-{key}-{s}" not in gym.registry]
            if missing:
                bad.append(f"{prefix}-{key}: missing {missing}")
            if "-" in key:
                bad.append(f"{prefix}-{key}: key contains '-', which breaks task.split('-')[2]")

        arms = sorted({k[:2] for k in levels if k.startswith(("ACv", "BCv"))})
        if arms != ["AC", "BC"]:
            bad.append(f"{prefix}: expected arms AC+BC, found {arms}")
        for arm in arms:
            keys = [k for k in levels if k.startswith(arm + "v")]
            if len(keys) != 10:
                bad.append(f"{prefix} {arm}: {len(keys)} variants, expected 10")
            off_field, on_field = DEGENERATE[prefix][arm]
            for k in keys:
                sub = levels[k]
                if any(s > 1e-9 for s in _spans(getattr(sub, off_field))):
                    bad.append(f"{prefix}-{k}: {off_field} is NOT degenerate (dimension not removed)")
                if not any(s > 1e-9 for s in _spans(getattr(sub, on_field))):
                    bad.append(f"{prefix}-{k}: {on_field} IS degenerate (kept dimension is off)")
        # slurm/eval.sbatch maps the L3b CHECKPOINT stem back to the L3 GYM key, because D29
        # renamed the artifacts and not the registrations. Pin both halves of that asymmetry here
        # so the mapping can never be "fixed" by renaming the envs without the sbatch noticing.
        if "L3v00" not in levels:
            bad.append(f"{prefix}: L3v00 is not registered -- eval.sbatch's L3b->L3 map is broken")
        if any(k.startswith("L3bv") for k in levels):
            bad.append(f"{prefix}: an L3bv* key exists -- eval.sbatch maps L3b->L3 and would now "
                       f"evaluate the wrong env")

        emit(f"{prefix}: {len(levels)} sub-levels, {len(levels) * 4} gym ids, arms {arms}")

    if bad:
        for b in bad:
            emit(f"  BAD  {b}")
        emit(f"LEVELS_FAILED ({len(bad)} problems)")
        return 1
    emit("LEVELS_OK")
    return 0


if __name__ == "__main__":
    rc = main()
    _app.close()
    sys.exit(rc)
