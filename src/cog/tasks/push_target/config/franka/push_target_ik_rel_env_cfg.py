"""Concrete Franka IK-Rel cfgs for push_target, one subclass per SubLevelCfg."""

from isaaclab.assets import RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

from ... import mdp
from ...assets import PUCK_VARIANTS, SUCCESS_RADIUS, TARGET_MARKER_SPAWN, TARGET_MARKER_Z
from ...levels import BEARING_FIXED, PUCK_FIXED, PUSH_DISTANCE, SUB_LEVELS, SubLevelCfg
from ...push_target_env_cfg import PushTargetEnvCfg


@configclass
class EventCfg:
    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        # cup_place's ready pose: the robot is table-mounted at z=0 exactly as in T1,
        # and this pose is the one T1's 400-demo waves were generated from.
        params={"default_pose": [0.0, -0.2, -0.15, -2.5, -0.02, 2.35, 0.7, 0.04, 0.04]},
    )
    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={"mean": 0.0, "std": 0.02, "asset_cfg": SceneEntityCfg("robot")},
    )
    randomize_puck_and_target = EventTerm(
        func=mdp.randomize_puck_and_target,
        mode="reset",
        params={},  # filled per sub-level in __post_init__
    )


@configclass
class FrankaPushTargetIKRelEnvCfg(PushTargetEnvCfg):
    """Parameterized by class attribute ``sub_level_key`` (subclasses override it)."""

    sub_level_key: str = "L0"

    def __post_init__(self):
        super().__post_init__()
        sub: SubLevelCfg = SUB_LEVELS[self.sub_level_key]
        variant = PUCK_VARIANTS[sub.puck_variant]

        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        puck_z = sub.puck_pose_range["z"][0]
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Puck",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[PUCK_FIXED[0], PUCK_FIXED[1], puck_z], rot=[1, 0, 0, 0]
            ),
            spawn=variant.spawn,
        )
        # Target marker must be a RigidObjectCfg (not AssetBaseCfg) so that it appears in
        # scene.get_state()["rigid_object"], hence in Mimic's object poses.
        import math

        self.scene.target_marker = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TargetMarker",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[
                    PUCK_FIXED[0] + PUSH_DISTANCE * math.cos(BEARING_FIXED),
                    PUCK_FIXED[1] + PUSH_DISTANCE * math.sin(BEARING_FIXED),
                    TARGET_MARKER_Z,
                ],
                rot=[1, 0, 0, 0],
            ),
            spawn=TARGET_MARKER_SPAWN,
        )

        # ee frame sensor (cup_place order: end_effector, right, left)
        ee_marker_cfg = FRAME_MARKER_CFG.copy()
        ee_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        ee_marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=ee_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
                ),
            ],
        )

        self.events = EventCfg()
        self.events.randomize_puck_and_target.params = {
            "object_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("target_marker"),
            "puck_pose_range": sub.puck_pose_range,
            "bearing_range": sub.bearing_range,
        }

        self.terminations.success.params = {
            "success_radius": SUCCESS_RADIUS,
            "max_lin_vel": 0.02,
        }
        # Puck off the table: tabletop is z=0, so a puck whose centre drops below
        # -0.05 has left the surface.
        self.terminations.puck_lost.params = {
            "object_cfg": SceneEntityCfg("object"),
            "min_height": -0.05,
        }


def _make_state_cfg(key: str):
    return configclass(
        type(f"FrankaPushTargetIKRelEnvCfg_{key}", (FrankaPushTargetIKRelEnvCfg,), {"sub_level_key": key})
    )


STATE_CFGS = {key: _make_state_cfg(key) for key in SUB_LEVELS}
