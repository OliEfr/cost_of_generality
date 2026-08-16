"""Visuomotor variants: table + wrist cameras at 128x128 (cup_place conventions).

table_cam sits left of the robot (y<0), elevated, looking across at the plinth
(object zone, y>0), the cabinet front/handle, and the drawer pull path.
Framing to be verified by frames_qa BEFORE any datagen (lesson from cup_place's
clipped goal marker).
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from ... import mdp
from ...levels import SUB_LEVELS
from .drawer_stow_ik_rel_env_cfg import FrankaDrawerStowIKRelEnvCfg

IMG_H = 128
IMG_W = 128


@configclass
class FrankaDrawerStowVisuomotorEnvCfg(FrankaDrawerStowIKRelEnvCfg):
    sub_level_key: str = "L0"

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.table_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False},
        )
        self.observations.policy.wrist_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )

        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=IMG_H,
            width=IMG_W,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
                clipping_range=(0.1, 2),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15), rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"
            ),
        )
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=IMG_H,
            width=IMG_W,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=26.0,
                clipping_range=(0.1, 4),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.15, -0.90, 1.20),
                rot=(0.50646, -0.84961, 0.12643, -0.07536),  # look-at (0.50, 0.25, 0.55)
                convention="ros",
            ),
        )

        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"
        self.image_obs_list = ["table_cam", "wrist_cam"]
        self.scene.num_envs = 8


def _make_visuomotor_cfg(key: str):
    return configclass(
        type(f"FrankaDrawerStowVisuomotorEnvCfg_{key}", (FrankaDrawerStowVisuomotorEnvCfg,), {"sub_level_key": key})
    )


VISUOMOTOR_CFGS = {key: _make_visuomotor_cfg(key) for key in SUB_LEVELS}
