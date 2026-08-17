"""Visuomotor variants for push_target: table + wrist cameras at 128x128.

Camera placement and intrinsics are Task 1's VERBATIM. That is deliberate: T1's
table_cam framing was QA'd over 1600 episodes (D10) and the push workspace is a subset
of T1's cup/goal workspace, so the framing is already known to cover it. Reusing it
also keeps the visual domain identical across tasks, so cross-task comparisons are not
confounded by camera changes.
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from ... import mdp
from ...levels import SUB_LEVELS
from .push_target_ik_rel_env_cfg import FrankaPushTargetIKRelEnvCfg

IMG_H = 128
IMG_W = 128


@configclass
class FrankaPushTargetVisuomotorEnvCfg(FrankaPushTargetIKRelEnvCfg):
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
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=24.0,
                clipping_range=(0.1, 4),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.0, 0.04, 0.4), rot=(0.35355, -0.61237, -0.61237, 0.35355), convention="ros"
            ),
        )

        self.rerender_on_reset = True
        self.sim.render.antialiasing_mode = "OFF"
        self.image_obs_list = ["table_cam", "wrist_cam"]
        self.scene.num_envs = 8


def _make_visuomotor_cfg(key: str):
    return configclass(
        type(
            f"FrankaPushTargetVisuomotorEnvCfg_{key}",
            (FrankaPushTargetVisuomotorEnvCfg,),
            {"sub_level_key": key},
        )
    )


VISUOMOTOR_CFGS = {key: _make_visuomotor_cfg(key) for key in SUB_LEVELS}
