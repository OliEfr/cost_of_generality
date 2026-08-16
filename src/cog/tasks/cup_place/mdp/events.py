"""Reset events: joint randomization of cup + goal with per-asset ranges.

Clone of stack's franka_stack_events.randomize_object_pose (spec 01 section 2)
extended with per-asset pose ranges; min_separation is enforced across BOTH
assets because they are sampled in one call.
"""

from __future__ import annotations

import math
import random
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _sample_pose(pose_range: dict) -> list[float]:
    ranges = [pose_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")]
    return [random.uniform(r[0], r[1]) for r in ranges]


def randomize_cup_and_goal(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    cup_cfg: SceneEntityCfg = SceneEntityCfg("cup"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_marker"),
    cup_pose_range: dict | None = None,
    goal_pose_range: dict | None = None,
    min_separation: float = 0.12,
    max_sample_tries: int = 5000,
):
    if env_ids is None:
        return
    cup_pose_range = cup_pose_range or {}
    goal_pose_range = goal_pose_range or {}

    for cur_env in env_ids.tolist():
        for _ in range(max_sample_tries):
            cup_pose = _sample_pose(cup_pose_range)
            goal_pose = _sample_pose(goal_pose_range)
            dist = math.dist(cup_pose[:2], goal_pose[:2])
            if dist >= min_separation:
                break
        for asset_cfg, pose in ((cup_cfg, cup_pose), (goal_cfg, goal_pose)):
            asset = env.scene[asset_cfg.name]
            pose_tensor = torch.tensor([pose], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(
                pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5]
            )
            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1),
                env_ids=torch.tensor([cur_env], device=env.device),
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device),
                env_ids=torch.tensor([cur_env], device=env.device),
            )
