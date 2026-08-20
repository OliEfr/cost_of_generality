"""Generality levels: THE single source of truth for the study distributions.

User-approved design (2026-08-16): 4 levels.
  L0 all fixed | L1 +cup pose (30x40 cm, yaw) | L2 +goal pose (20x20 cm)
  L3 +object variation (cylinder sizes x colors, + mug meshes pending QA)
L3 is realized as per-variant sub-environments (L3v00..) whose datasets and
eval sets are merged; distribution semantics = uniform over variants x poses.

All coordinates are env-local (robot base at origin, table top at z=0).
Eval sets sampled from these SAME distributions (in-distribution protocol).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .assets import CUP_VARIANTS, DEFAULT_CUP

# workspace geometry (m)
CUP_FIXED = (0.45, -0.15)
GOAL_FIXED = (0.50, 0.25)
CUP_RANGE = {"x": (0.35, 0.65), "y": (-0.25, 0.15), "yaw": (-1.57, 1.57)}  # 30 x 40 cm
GOAL_RANGE = {"x": (0.40, 0.60), "y": (0.10, 0.30)}                        # 20 x 20 cm
MIN_SEPARATION = 0.14


def _fixed(x: float, y: float, z: float, yaw: float = 0.0) -> dict:
    return {"x": (x, x), "y": (y, y), "z": (z, z), "yaw": (yaw, yaw)}


@dataclass(frozen=True)
class SubLevelCfg:
    """One registered env family: a level (or an L3 variant sub-level)."""

    key: str                    # e.g. "L0", "L3v04" -> gym id fragment
    level: str                  # parent level: L0..L3
    cup_variant: str
    cup_pose_range: dict
    goal_pose_range: dict
    min_separation: float = MIN_SEPARATION


def _mk(key: str, level: str, variant: str, cup_range: dict | None, goal_range: dict | None) -> SubLevelCfg:
    half_h = CUP_VARIANTS[variant].half_height
    cup_z = half_h + 0.002
    from .assets import GOAL_MARKER_Z

    if cup_range is None:
        cpr = _fixed(*CUP_FIXED, cup_z)
    else:
        cpr = dict(cup_range) | {"z": (cup_z, cup_z)}
    if goal_range is None:
        gpr = _fixed(*GOAL_FIXED, GOAL_MARKER_Z)
    else:
        gpr = dict(goal_range) | {"z": (GOAL_MARKER_Z, GOAL_MARKER_Z)}
    return SubLevelCfg(key=key, level=level, cup_variant=variant,
                       cup_pose_range=cpr, goal_pose_range=gpr)


# L3 variant set: 2 cylinder sizes x 5 colors (10). Mug meshes (mug_s, mug_m)
# are appended only after P3 grasp/render QA -- edit L3_VARIANTS then.
# Colour order matches assets.COLORS positionally so variant indices are unchanged by the D28
# recolouring (v01 was cyl_s_green, is now cyl_s_orange). Keep in sync with COLORS.
L3_VARIANTS: list[str] = [
    f"cyl_{s}_{c}" for s in ("s", "m") for c in ("red", "orange", "blue", "magenta", "purple")
]

SUB_LEVELS: dict[str, SubLevelCfg] = {}
SUB_LEVELS["L0"] = _mk("L0", "L0", DEFAULT_CUP, None, None)
SUB_LEVELS["L1"] = _mk("L1", "L1", DEFAULT_CUP, CUP_RANGE, None)
SUB_LEVELS["L2"] = _mk("L2", "L2", DEFAULT_CUP, CUP_RANGE, GOAL_RANGE)
for _i, _v in enumerate(L3_VARIANTS):
    SUB_LEVELS[f"L3v{_i:02d}"] = _mk(f"L3v{_i:02d}", "L3", _v, CUP_RANGE, GOAL_RANGE)


def level_members(level: str) -> list[SubLevelCfg]:
    return [s for s in SUB_LEVELS.values() if s.level == level]
