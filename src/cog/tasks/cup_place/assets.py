"""Cup variant registry + goal marker spawn cfg.

Default cup (L0-L2) is a procedural cylinder tumbler: offline-safe (no cloud
USD), robust top-down grasp, no handle/yaw grasp confound (decision journaled
2026-08-16). The mug USD variants join L3 only after render/grasp QA.

All spawn cfgs follow spec 05 (verified field names, v2.3.0).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# Cluster/offline override (spec 05 section 4): local asset mirror root.
ASSET_ROOT = os.environ.get("COG_ASSET_ROOT", ISAACLAB_NUCLEUS_DIR)

CUP_RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)

COLORS: dict[str, tuple[float, float, float]] = {
    "red": (0.80, 0.05, 0.05),
    "green": (0.05, 0.60, 0.05),
    "blue": (0.05, 0.10, 0.80),
    "yellow": (0.85, 0.80, 0.05),
    "purple": (0.50, 0.05, 0.60),
}


@dataclass(frozen=True)
class CupVariant:
    name: str
    half_height: float          # m; cup center height when resting upright
    grasp_z_offset: float       # grasp point above cup center
    spawn: object               # spawner cfg


def _cylinder(name: str, radius: float, height: float, color: tuple) -> CupVariant:
    return CupVariant(
        name=name,
        half_height=height / 2,
        grasp_z_offset=0.015,
        spawn=sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            axis="Z",
            rigid_props=CUP_RIGID_PROPS,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2, dynamic_friction=1.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color, roughness=0.5, metallic=0.0
            ),
            semantic_tags=[("class", "cup")],
        ),
    )


def _mug(name: str, scale: float) -> CupVariant:
    # half_height/grasp for the mug are provisional until bbox inspection (spec 05 sec 1.3)
    return CupVariant(
        name=name,
        half_height=0.05 * scale,
        grasp_z_offset=0.02 * scale,
        spawn=UsdFileCfg(
            usd_path=f"{ASSET_ROOT}/Objects/Mug/mug.usd",
            scale=(scale, scale, scale),
            rigid_props=CUP_RIGID_PROPS,
            semantic_tags=[("class", "cup")],
        ),
    )


CUP_VARIANTS: dict[str, CupVariant] = {}
for _cname, _rgb in COLORS.items():
    CUP_VARIANTS[f"cyl_s_{_cname}"] = _cylinder(f"cyl_s_{_cname}", 0.027, 0.080, _rgb)
    CUP_VARIANTS[f"cyl_m_{_cname}"] = _cylinder(f"cyl_m_{_cname}", 0.031, 0.090, _rgb)
CUP_VARIANTS["mug_s"] = _mug("mug_s", 0.80)
CUP_VARIANTS["mug_m"] = _mug("mug_m", 0.95)

DEFAULT_CUP = "cyl_m_red"

GOAL_MARKER_SPAWN = sim_utils.CylinderCfg(
    radius=0.06,
    height=0.004,
    axis="Z",
    rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
    collision_props=None,  # pure visual: gripper/cup pass through; cup rests on the table
    visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.10, 0.70, 0.10), roughness=0.9, metallic=0.0
    ),
    semantic_tags=[("class", "goal")],
)
GOAL_MARKER_Z = 0.003  # bottom 1 mm above tabletop; avoids z-fighting half-disk artifact
