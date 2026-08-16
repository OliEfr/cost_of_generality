"""Visuomotor variants: table + wrist cameras at 128x128, images in the policy group.

Camera placement/intrinsics copied from stack_ik_rel_visuomotor_env_cfg.py (spec 01
section 4), resolution raised 84 -> 128.
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from ... import mdp
from ...levels import SUB_LEVELS
from .cup_place_ik_rel_env_cfg import FrankaCupPlaceIKRelEnvCfg

IMG_H = 128
IMG_W = 128


@configclass
class FrankaCupPlaceVisuomotorEnvCfg(FrankaCupPlaceIKRelEnvCfg):
    sub_level_key: str = "L0"

    def __post_init__(self):
        super().__post_init__()

        # image obs terms appended to the policy group (recorded per spec 03)
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
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
                clipping_range=(0.1, 4),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.0, 0.0, 0.4), rot=(0.35355, -0.61237, -0.61237, 0.35355), convention="ros"
            ),
        )

        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"
        self.image_obs_list = ["table_cam", "wrist_cam"]
        # cameras are expensive: safe default, overridden by --num_envs
        self.scene.num_envs = 8


def _make_visuomotor_cfg(key: str):
    return configclass(
        type(f"FrankaCupPlaceVisuomotorEnvCfg_{key}", (FrankaCupPlaceVisuomotorEnvCfg,), {"sub_level_key": key})
    )


VISUOMOTOR_CFGS = {key: _make_visuomotor_cfg(key) for key in SUB_LEVELS}
