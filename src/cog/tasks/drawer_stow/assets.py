"""Sektion cabinet cfg + stow-object variant registry + plinth (side table).

Cabinet: stock Isaac Lab Sektion articulation, root moved to x=0.9 so a 0.2 m
drawer pull leaves the handle at x~0.26 (reachable; at the stock x=0.8 the
handle would end up at x~0.19, near the base). The USD lives under
Isaac/Props/ on Nucleus (NOT Isaac/IsaacLab/), hence a second asset-root seam.

Stow objects are procedural boxes (offline-safe, like the cup cylinders):
2 sizes x 5 colors for T2-L3. Sizes keep the yaw-aligned grasp width under the
0.08 m gripper span.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from ..cup_place.assets import COLORS, CUP_RIGID_PROPS

# Cluster/offline override for assets rooted at .../Isaac (cup_place's
# COG_ASSET_ROOT is rooted one level deeper, at .../Isaac/IsaacLab).
ISAAC_ASSET_ROOT = os.environ.get("COG_ISAAC_ASSET_ROOT", ISAAC_NUCLEUS_DIR)

# --- cabinet (stock cfg fields verbatim except init pos; recon 2026-08-17) ---
CABINET_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Cabinet",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_ROOT}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.9, 0.0, 0.4),
        rot=(0.0, 0.0, 0.0, 1.0),  # yaw 180 deg: drawers face the robot
        joint_pos={
            "door_left_joint": 0.0,
            "door_right_joint": 0.0,
            "drawer_bottom_joint": 0.0,
            "drawer_top_joint": 0.0,
        },
    ),
    actuators={
        # stiffness 0: stock's 10.0 acts as a return spring toward joint 0 and
        # silently re-closes the drawer after release; viscous damping holds it
        "drawers": ImplicitActuatorCfg(
            joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
            effort_limit=87.0, velocity_limit=100.0, stiffness=0.0, damping=8.0,
        ),
        "doors": ImplicitActuatorCfg(
            joint_names_expr=["door_left_joint", "door_right_joint"],
            effort_limit=87.0, velocity_limit=100.0, stiffness=10.0, damping=2.5,
        ),
    },
)

# empirical geometry (ops/cabinet_geometry.json), all relative to cabinet ROOT:
CABINET_TOP_Z = 0.402          # top surface above root
CABINET_FRONT_X = 0.325        # front face ahead of root (in cabinet local +x)
DRAWER_TRAVEL = 0.40           # drawer_top_joint upper limit
DRAWER_BODY_Z = 0.3172         # drawer_top body origin above root
DRAWER_CAVITY_FLOOR_Z = -0.062   # cavity floor rel drawer_top body origin
DRAWER_CAVITY_RIM_Z = 0.062      # wall top rel drawer_top body origin
DRAWER_CAVITY_HALF_X = 0.22
DRAWER_CAVITY_HALF_Y = 0.30

# --- plinth (side table the object starts on; static, offline-safe) ---
PLINTH_CENTER = (0.24, 0.45, 0.20)
PLINTH_SIZE = (0.24, 0.30, 0.40)   # top surface z=0.40, x in [0.12,0.36], y in [0.30,0.60]
PLINTH_SPAWN = sim_utils.CuboidCfg(
    size=PLINTH_SIZE,
    rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.9),
    visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.45, 0.38, 0.30), roughness=0.8, metallic=0.0
    ),
    semantic_tags=[("class", "plinth")],
)
PLINTH_TOP_Z = PLINTH_CENTER[2] + PLINTH_SIZE[2] / 2  # 0.40


@dataclass(frozen=True)
class BoxVariant:
    name: str
    half_size: float            # m; cube half edge
    grasp_z_offset: float       # grasp point above box center
    spawn: object


def _box(name: str, edge: float, color: tuple) -> BoxVariant:
    return BoxVariant(
        name=name,
        half_size=edge / 2,
        grasp_z_offset=-0.005,  # grip slightly below center: the carried box hangs
        # higher relative to the TCP, buying wall clearance on the stow traverse
        spawn=sim_utils.CuboidCfg(
            size=(edge, edge, edge),
            rigid_props=CUP_RIGID_PROPS,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2, dynamic_friction=1.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color, roughness=0.5, metallic=0.0
            ),
            semantic_tags=[("class", "stow_object")],
        ),
    )


BOX_VARIANTS: dict[str, BoxVariant] = {}
for _cname, _rgb in COLORS.items():
    BOX_VARIANTS[f"box_s_{_cname}"] = _box(f"box_s_{_cname}", 0.045, _rgb)
    BOX_VARIANTS[f"box_m_{_cname}"] = _box(f"box_m_{_cname}", 0.058, _rgb)

DEFAULT_BOX = "box_m_red"
