"""Success termination for drawer_stow: drawer open + object inside the cavity.

The cavity check runs in the drawer_top BODY frame (so it is correct for any
cabinet pose and opening amount); bounds from ops/cabinet_geometry.json.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from ..assets import (
    DRAWER_CAVITY_FLOOR_Z,
    DRAWER_CAVITY_HALF_X,
    DRAWER_CAVITY_HALF_Y,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_stowed_in_drawer(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet"),
    min_drawer_open: float = 0.15,
    object_half_size: float = 0.029,
    max_lin_vel: float = 0.15,
) -> torch.Tensor:
    """True when drawer_top is open >= min_drawer_open, the object rests inside
    its cavity, and the gripper has released."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    cabinet: Articulation = env.scene[cabinet_cfg.name]

    joint_ids, _ = cabinet.find_joints(["drawer_top_joint"])
    drawer_open = cabinet.data.joint_pos[:, joint_ids[0]] >= min_drawer_open

    body_ids, _ = cabinet.find_bodies(["drawer_top"])
    drawer_pos = cabinet.data.body_pos_w[:, body_ids[0]]
    drawer_quat = cabinet.data.body_quat_w[:, body_ids[0]]
    obj_local = math_utils.quat_apply_inverse(drawer_quat, obj.data.root_pos_w - drawer_pos)

    in_x = obj_local[:, 0].abs() < DRAWER_CAVITY_HALF_X
    in_y = obj_local[:, 1].abs() < DRAWER_CAVITY_HALF_Y
    # resting on the cavity floor (small band above it; below the rim)
    rest_z = DRAWER_CAVITY_FLOOR_Z + object_half_size
    in_z = (obj_local[:, 2] > rest_z - 0.02) & (obj_local[:, 2] < rest_z + 0.05)

    settled = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=1) < max_lin_vel

    finger_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    finger_pos = robot.data.joint_pos[:, finger_ids]
    released = torch.all(
        torch.isclose(
            finger_pos,
            torch.full_like(finger_pos, env.cfg.gripper_open_val),
            atol=1e-3,
            rtol=1e-3,
        ),
        dim=1,
    )

    return drawer_open & in_x & in_y & in_z & settled & released
