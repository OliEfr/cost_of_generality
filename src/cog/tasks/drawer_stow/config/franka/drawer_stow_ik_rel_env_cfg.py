"""Concrete Franka IK-Rel cfgs for drawer_stow, one subclass per SubLevelCfg."""

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
from ...assets import BOX_VARIANTS, CABINET_CFG
from ...drawer_stow_env_cfg import DrawerStowEnvCfg
from ...levels import SUB_LEVELS, SubLevelCfg


@configclass
class EventCfg:
    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400]
        },
    )
    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={"mean": 0.0, "std": 0.02, "asset_cfg": SceneEntityCfg("robot")},
    )
    randomize_cabinet_and_object = EventTerm(
        func=mdp.randomize_cabinet_and_object,
        mode="reset",
        params={},  # filled per sub-level in __post_init__
    )


@configclass
class FrankaDrawerStowIKRelEnvCfg(DrawerStowEnvCfg):
    """Parameterized by class attribute ``sub_level_key`` (subclasses override it)."""

    sub_level_key: str = "L0"

    def __post_init__(self):
        super().__post_init__()
        sub: SubLevelCfg = SUB_LEVELS[self.sub_level_key]
        variant = BOX_VARIANTS[sub.box_variant]

        # robot + IK-Rel actions (identical to cup_place / stack)
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

        # cabinet + stow object (variant)
        self.scene.cabinet = CABINET_CFG.replace(prim_path="{ENV_REGEX_NS}/Cabinet")
        obj_z = sub.object_pose_range["z"][0]
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.21, 0.45, obj_z], rot=[1, 0, 0, 0]),
            spawn=variant.spawn,
        )

        # handle frame (stock cabinet offset: grasp-ready TCP pose)
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/CabinetFrameTransformer"
        self.scene.cabinet_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Cabinet/sektion",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_top",
                    name="drawer_handle_top",
                    offset=OffsetCfg(pos=(0.305, 0.0, 0.01), rot=(0.5, 0.5, -0.5, -0.5)),
                ),
            ],
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

        # events: level distribution
        self.events = EventCfg()
        self.events.randomize_cabinet_and_object.params = {
            "cabinet_cfg": SceneEntityCfg("cabinet"),
            "object_cfg": SceneEntityCfg("object"),
            "cabinet_pose_range": sub.cabinet_pose_range,
            "object_pose_range": sub.object_pose_range,
        }

        # success thresholds per variant
        self.terminations.success.params = {
            "min_drawer_open": 0.15,
            "object_half_size": variant.half_size,
        }


def _make_state_cfg(key: str):
    return configclass(
        type(f"FrankaDrawerStowIKRelEnvCfg_{key}", (FrankaDrawerStowIKRelEnvCfg,), {"sub_level_key": key})
    )


STATE_CFGS = {key: _make_state_cfg(key) for key in SUB_LEVELS}
