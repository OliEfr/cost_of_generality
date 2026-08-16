"""Success termination for cup_place.

Modeled on place/mdp/terminations.py::object_a_is_into_b and
stack cubes_stacked (spec 01 section 3, spec 04 section 3.4): xy distance +
height band + upright + settled velocity + gripper released.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cup_placed_at_goal(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cup_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_marker"),
    xy_threshold: float = 0.05,
    height_threshold: float = 0.015,
    height_diff: float = 0.045,
    max_tilt_rad: float = 0.52,
    max_lin_vel: float = 0.10,
) -> torch.Tensor:
    """True when the cup rests upright on the goal marker with the gripper open.

    height_diff: expected cup-center height above the goal marker center
    (cup half height; override per cup variant via term params).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    cup: RigidObject = env.scene[cup_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]

    pos_diff = cup.data.root_pos_w - goal.data.root_pos_w
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    h_dist = torch.abs(pos_diff[:, 2] - height_diff)

    # uprightness: angle between cup z-axis and world z
    roll, pitch, _ = math_utils.euler_xyz_from_quat(cup.data.root_quat_w)
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))
    upright = (roll.abs() < max_tilt_rad) & (pitch.abs() < max_tilt_rad)

    settled = torch.linalg.vector_norm(cup.data.root_lin_vel_w, dim=1) < max_lin_vel

    # gripper released: both finger joints back at open position (stack convention)
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

    return (xy_dist < xy_threshold) & (h_dist < height_threshold) & upright & settled & released
