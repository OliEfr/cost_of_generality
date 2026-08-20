"""Puck variant registry + target-marker spawn cfg for Task 3 (push-to-target).

All geometry is procedural (`sim_utils` primitives), so nothing here depends on
Nucleus cloud assets — same reasoning as D1.

Geometry follows the empirical probe of 2026-08-17 (docs/journal.md):
  * The `ee_frame` TCP sits at the grasp point between the fingertips; the finger
    BODY origins are +4.5 cm above it. A TCP z of 0.048 contacted a cup spanning
    z 0..0.084, i.e. TCP z IS the contact height, directly.
  * A tall object tips: the T1 cup (r 0.031 x h 0.090, i.e. taller than wide) rolled
    to 90 deg mid-push. Pucks here are wider than tall so they cannot tip, and the
    contact height is deliberately BELOW the centre of mass (CONTACT_H_FRAC) because
    tipping torque about the leading base edge grows with contact height.
"""

from __future__ import annotations

from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

PUCK_RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)

# Push contact height as a fraction of puck height. Low on purpose: the tipping
# torque about the leading base edge scales with contact height, and 0.40 keeps the
# fingertips >= 1.5 cm clear of the tabletop for the shortest variant.
CONTACT_H_FRAC = 0.40

# Constant across every variant so the L3 axis is GEOMETRY, not inertia: if mass
# tracked volume, an L3 result would confound shape with how hard the puck is to move.
PUCK_MASS = 0.15

# High-ish friction is a reliability choice: the puck stops when the push stops,
# which the `settled` success clause depends on. A slippery puck coasts through the
# target disk, and generation OR-latches success across timesteps (D19 item 7).
PUCK_FRICTION_STATIC = 0.8
PUCK_FRICTION_DYNAMIC = 0.7

# Marker green is reserved for the target disk, so pucks never use it -- but excluding pure green is
# not enough (D28). YELLOW carries G=0.80, higher than the marker's own 0.70, and measured on T1 it
# was the second-worst aliaser after green (0.15 vs red's 0.95). Replaced with magenta, keeping this
# dict's ORDER intact: the variant loop below assigns colours by index, and index 4 (orange) is
# DEFAULT_PUCK, which T3's L0-L2 datasets were generated with and which must not change.
# INVARIANT: green channel <= 0.40 for every entry (orange is the loosest, and is the default).
PUCK_COLORS: dict[str, tuple[float, float, float]] = {
    "red": (0.80, 0.05, 0.05),
    "blue": (0.05, 0.10, 0.80),
    "magenta": (0.90, 0.10, 0.55),     # was yellow (G=0.80)
    "purple": (0.50, 0.05, 0.60),
    "orange": (0.90, 0.40, 0.05),      # DEFAULT_PUCK's colour -- position 4 is pinned
}


@dataclass(frozen=True)
class PuckVariant:
    name: str
    radius: float
    height: float
    spawn: object

    @property
    def half_height(self) -> float:
        return self.height / 2

    @property
    def contact_z(self) -> float:
        """World z of the TCP for a push contact on this variant."""
        return CONTACT_H_FRAC * self.height


def _puck(name: str, radius: float, height: float, color: tuple) -> PuckVariant:
    return PuckVariant(
        name=name,
        radius=radius,
        height=height,
        spawn=sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            axis="Z",
            rigid_props=PUCK_RIGID_PROPS,
            mass_props=sim_utils.MassPropertiesCfg(mass=PUCK_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=PUCK_FRICTION_STATIC,
                dynamic_friction=PUCK_FRICTION_DYNAMIC,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color, roughness=0.7, metallic=0.0
            ),
            semantic_tags=[("class", "puck")],
        ),
    )


# L3 axis = GEOMETRY, unlike T1/T2 where the ten variants were 2 shapes x 5 colours
# and colour is physics-inert (D17/D18). Here the ten variants are 5 radii x 2 heights,
# all ten physically distinct, with colour cycled so appearance varies too but is not
# the axis. Radius spans 0.032..0.058 (a 1.8x range); a changed radius shifts where the
# puck surface is relative to its centre, which is what the push standoff must absorb.
# Radii chosen by MEASUREMENT. Expert SR falls monotonically once the puck gets wide
# than the blade can control (8x64-episode gate, 2026-08-17):
#   r=0.032 -> 88%,  r=0.045 -> 92-94%,  r=0.052 -> 73-83%,  r=0.058 -> 63-75%.
# A ~2 cm blade simply cannot keep a 12 cm-wide disc on line: contact is a short chord of
# a shallow arc, so any lateral offset spins the puck instead of translating it. The axis
# is therefore spaced across the reliable band 0.032-0.045 (a 1.4x radius range, still
# more physical variation than T2's 1.2x box-edge range), keeping ten variants so no
# downstream tooling changes.
PUCK_RADII = (0.032, 0.035, 0.038, 0.042, 0.045)
PUCK_HEIGHTS = (0.040, 0.055)

# Largest radius drives the approach standoff so that NO variant is penetrated when a
# source stroke recorded on one variant is rigidly replayed onto another (D19 item 8).
MAX_PUCK_RADIUS = max(PUCK_RADII)

PUCK_VARIANTS: dict[str, PuckVariant] = {}
_cnames = list(PUCK_COLORS)
for _i, _h in enumerate(PUCK_HEIGHTS):
    for _j, _r in enumerate(PUCK_RADII):
        _name = f"puck_r{int(_r * 1000):02d}_h{int(_h * 1000):02d}"
        PUCK_VARIANTS[_name] = _puck(_name, _r, _h, PUCK_COLORS[_cnames[(_i * 5 + _j) % 5]])

DEFAULT_PUCK = "puck_r45_h40"

# Pure-visual disk, identical pattern to cup_place's goal marker: kinematic, no
# collision, so the puck slides over it and the gripper passes through. It must be a
# RigidObjectCfg (not AssetBaseCfg) to appear in scene.get_state()["rigid_object"] and
# therefore in Mimic's object poses.
TARGET_MARKER_RADIUS = 0.06
TARGET_MARKER_SPAWN = sim_utils.CylinderCfg(
    radius=TARGET_MARKER_RADIUS,
    height=0.004,
    axis="Z",
    rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
    collision_props=None,
    visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.10, 0.70, 0.10), roughness=0.9, metallic=0.0
    ),
    semantic_tags=[("class", "goal")],
)
TARGET_MARKER_Z = 0.003  # 1 mm above the tabletop; avoids z-fighting (cup_place lesson)

# Success radius. Same 5 cm gate as T1, and it is what absorbs the two unavoidable
# stroke errors: radius mismatch between source and target variant (<= 2.6 cm) and
# contact slip during the open-loop transformed stroke.
SUCCESS_RADIUS = 0.05
