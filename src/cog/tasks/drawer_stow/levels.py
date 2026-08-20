"""Generality levels for drawer_stow: single source of truth (mirrors cup_place).

Ladder (D12): T2-L0 all fixed | T2-L1 +object pose | T2-L2 +cabinet pose |
T2-L3 +object variants (2 box sizes x 5 colors as sub-envs).

Coordinates are env-local (robot base at origin, ground z=0). The object zone
sits on the plinth top and is capped at x<=0.26 so the vertical grasp corridor
stays clear of the opened drawer (front face >= 0.325 at 0.2 m pull with the
cabinet at its nearest randomized pose). Cabinet yaw is randomized about pi
(drawers toward the robot); its range is capped so the swept front corner
never reaches the plinth footprint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .assets import BOX_VARIANTS, DEFAULT_BOX, PLINTH_TOP_Z

OBJ_FIXED = (0.21, 0.45)
OBJ_RANGE = {"x": (0.16, 0.26), "y": (0.36, 0.54), "yaw": (-0.785, 0.785)}  # 10 x 18 cm
CABINET_FIXED = (0.9, 0.0)
CABINET_RANGE = {"x": (0.85, 0.95), "y": (-0.06, 0.06),
                 "yaw": (math.pi - 0.13, math.pi + 0.13)}  # +-7.5 deg
DRAWER_OPEN_TARGET = 0.28      # expert pull target (m); success threshold is 0.15.
                               # Deep pull moves the drawer wall behind the descent corridor
                               # (arm carry envelope ends x~0.32; wall inner at 0.575-open+0.01
                               # must sit below that minus box half). 0.33 is the deepest the
                               # pull reliably reaches before the base-proximity stall (0.36+
                               # hit the 6 s bail in run 17)


def _fixed(x: float, y: float, z: float, yaw: float = 0.0) -> dict:
    return {"x": (x, x), "y": (y, y), "z": (z, z), "yaw": (yaw, yaw)}


@dataclass(frozen=True)
class SubLevelCfg:
    key: str                    # e.g. "L0", "L3v04" -> gym id fragment
    level: str                  # parent level: L0..L3
    box_variant: str
    object_pose_range: dict
    cabinet_pose_range: dict


def _mk(key: str, level: str, variant: str, obj_range: dict | None, cab_range: dict | None) -> SubLevelCfg:
    obj_z = PLINTH_TOP_Z + BOX_VARIANTS[variant].half_size + 0.002
    if obj_range is None:
        opr = _fixed(*OBJ_FIXED, obj_z)
    else:
        opr = dict(obj_range) | {"z": (obj_z, obj_z)}
    if cab_range is None:
        cpr = _fixed(*CABINET_FIXED, 0.4, math.pi)
    else:
        cpr = dict(cab_range) | {"z": (0.4, 0.4)}
    return SubLevelCfg(key=key, level=level, box_variant=variant,
                       object_pose_range=opr, cabinet_pose_range=cpr)


L3_VARIANTS: list[str] = [
    # colours per assets.COLORS, positionally stable across the D28 recolouring
    f"box_{s}_{c}" for s in ("s", "m") for c in ("red", "orange", "blue", "magenta", "purple")
]

SUB_LEVELS: dict[str, SubLevelCfg] = {}
SUB_LEVELS["L0"] = _mk("L0", "L0", DEFAULT_BOX, None, None)
SUB_LEVELS["L1"] = _mk("L1", "L1", DEFAULT_BOX, OBJ_RANGE, None)
SUB_LEVELS["L2"] = _mk("L2", "L2", DEFAULT_BOX, OBJ_RANGE, CABINET_RANGE)
for _i, _v in enumerate(L3_VARIANTS):
    SUB_LEVELS[f"L3v{_i:02d}"] = _mk(f"L3v{_i:02d}", "L3", _v, OBJ_RANGE, CABINET_RANGE)


def level_members(level: str) -> list[SubLevelCfg]:
    return [s for s in SUB_LEVELS.values() if s.level == level]
