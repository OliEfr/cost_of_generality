"""Reset events: cabinet root pose + closed joints, object pose on the plinth.

Zones are disjoint by construction (levels.py), so no separation rejection
sampling is needed. Cabinet joints are explicitly re-closed each reset because
the pose write alone does not touch joint state.
"""

from __future__ import annotations

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


def randomize_cabinet_and_object(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    cabinet_cfg: SceneEntityCfg = SceneEntityCfg("cabinet"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    cabinet_pose_range: dict | None = None,
    object_pose_range: dict | None = None,
):
    if env_ids is None:
        return
    cabinet_pose_range = cabinet_pose_range or {}
    object_pose_range = object_pose_range or {}

    cabinet = env.scene[cabinet_cfg.name]
    obj = env.scene[object_cfg.name]
    for cur_env in env_ids.tolist():
        ids = torch.tensor([cur_env], device=env.device)
        for asset, pose in ((cabinet, _sample_pose(cabinet_pose_range)),
                            (obj, _sample_pose(object_pose_range))):
            pose_tensor = torch.tensor([pose], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(
                pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5]
            )
            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1), env_ids=ids
            )
            asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=ids)
        # drawers/doors closed at start
        jp = torch.zeros(1, cabinet.num_joints, device=env.device)
        cabinet.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
