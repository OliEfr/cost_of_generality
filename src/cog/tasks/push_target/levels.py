"""Task 3 generality ladder: push a puck a FIXED 20 cm to a target disk.

The fixed push distance is forced by Mimic, not by taste (D19): a rigid transform
carries no scale, so the stroke length baked into a source demo is the stroke length
you get. Holding |puck - target| constant makes that stroke correct at every level;
only the puck's POSITION (L1) and the target's BEARING around the puck (L2) vary.

A useful side effect: because the target is DERIVED from the puck at a fixed radius,
the puck can never spawn inside the target region, so no rejection sampling is needed
and no subtask signal can be true at t=0 (which crashes Mimic's pool loader).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .assets import DEFAULT_PUCK, PUCK_VARIANTS, TARGET_MARKER_Z

# Fixed puck->target distance (m). Any change invalidates every recorded source demo.
PUSH_DISTANCE = 0.20

# Workspace, chosen inside T1's proven-reachable band (cup x 0.35-0.65, y -0.25-0.15):
# the puck sits near the robot and is pushed away in +y, so the blade stands off at
# y - 0.075 and the target lands at y + 0.20 -- both inside the reachable region.
PUCK_FIXED = (0.42, -0.10)
PUCK_RANGE = {"x": (0.36, 0.48), "y": (-0.16, -0.04)}      # 12 x 12 cm

# Bearing of the target from the puck, radians, measured from +x. pi/2 = straight +y,
# which is the direction the 2026-08-17 probe verified a blade push works in.
BEARING_FIXED = math.pi / 2
BEARING_RANGE = (math.pi / 2 - 0.70, math.pi / 2 + 0.70)    # +-40 deg


@dataclass(frozen=True)
class SubLevelCfg:
    key: str                    # "L0", "L3v04" -> gym id fragment
    level: str                  # parent level L0..L3
    puck_variant: str
    puck_pose_range: dict       # x/y/z/yaw ranges for the puck
    bearing_range: tuple        # (lo, hi) radians; equal values = fixed


def _fixed_xy(x: float, y: float, z: float) -> dict:
    return {"x": (x, x), "y": (y, y), "z": (z, z), "yaw": (0.0, 0.0)}


def _mk(key: str, level: str, variant: str, rng: dict | None, bearing: tuple) -> SubLevelCfg:
    puck_z = PUCK_VARIANTS[variant].half_height + 0.002
    if rng is None:
        ppr = _fixed_xy(*PUCK_FIXED, puck_z)
    else:
        # Puck yaw is deliberately NOT randomized: a cylinder is yaw-symmetric, so it
        # would be a visually and physically inert axis (the D17 mistake).
        ppr = dict(rng) | {"z": (puck_z, puck_z), "yaw": (0.0, 0.0)}
    return SubLevelCfg(key=key, level=level, puck_variant=variant,
                       puck_pose_range=ppr, bearing_range=bearing)


L3_VARIANTS: list[str] = list(PUCK_VARIANTS)   # all ten: 5 radii x 2 heights

SUB_LEVELS: dict[str, SubLevelCfg] = {}
SUB_LEVELS["L0"] = _mk("L0", "L0", DEFAULT_PUCK, None, (BEARING_FIXED, BEARING_FIXED))
SUB_LEVELS["L1"] = _mk("L1", "L1", DEFAULT_PUCK, PUCK_RANGE, (BEARING_FIXED, BEARING_FIXED))
SUB_LEVELS["L2"] = _mk("L2", "L2", DEFAULT_PUCK, PUCK_RANGE, BEARING_RANGE)
for _i, _v in enumerate(L3_VARIANTS):
    SUB_LEVELS[f"L3v{_i:02d}"] = _mk(f"L3v{_i:02d}", "L3", _v, PUCK_RANGE, BEARING_RANGE)

assert len(L3_VARIANTS) == 10, "downstream tooling hardcodes ten L3 variants"


def level_members(level: str) -> list[SubLevelCfg]:
    return [s for s in SUB_LEVELS.values() if s.level == level]
