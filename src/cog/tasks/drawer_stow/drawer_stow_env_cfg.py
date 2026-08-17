"""Base (robot-agnostic) env cfg for drawer_stow. Mirrors cup_place_env_cfg."""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from . import mdp
from .assets import PEDESTAL_HEIGHT, PEDESTAL_SPAWN, PLINTH_CENTER, PLINTH_SPAWN


@configclass
class DrawerStowSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    cabinet: ArticulationCfg = MISSING       # set by franka cfg (from assets.CABINET_CFG)
    cabinet_frame: FrameTransformerCfg = MISSING
    # object is added by the franka config subclass (variant-dependent)

    pedestal = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Pedestal",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0, 0, PEDESTAL_HEIGHT / 2], rot=[1, 0, 0, 0]),
        spawn=PEDESTAL_SPAWN,
    )
    plinth = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plinth",
        init_state=RigidObjectCfg.InitialStateCfg(pos=list(PLINTH_CENTER), rot=[1, 0, 0, 0]),
        spawn=PLINTH_SPAWN,
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0]),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_pos = ObsTerm(func=mdp.object_pos, params={"object_cfg": SceneEntityCfg("object")})
        object_quat = ObsTerm(func=mdp.object_quat, params={"object_cfg": SceneEntityCfg("object")})
        drawer_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
        )
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        drawer_opened_1 = ObsTerm(
            func=mdp.drawer_opened,
            params={
                "asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
                "threshold": 0.15,
            },
        )
        grasp_2 = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.05, "asset_cfg": SceneEntityCfg("object")},
    )
    success = DoneTerm(func=mdp.object_stowed_in_drawer)  # params per variant in franka cfg


@configclass
class DrawerStowEnvCfg(ManagerBasedRLEnvCfg):
    scene: DrawerStowSceneCfg = DrawerStowSceneCfg(num_envs=8, env_spacing=3.0, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    rewards = None
    events = None  # set by franka cfg
    curriculum = None

    def __post_init__(self):
        self.decimation = 5
        self.episode_length_s = 60.0  # 1200 policy steps at 20 Hz: the ramp-paced
        # expert needs ~670 steps for the NEAR object draw; far draws add ~150
        self.sim.dt = 0.01
        self.sim.render_interval = 2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.gripper_joint_names = ["panda_finger_.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005
