"""Reset event: puck pose, with the target disk DERIVED at a fixed distance.

The target is not sampled independently -- it is placed PUSH_DISTANCE away from the
puck along a sampled bearing (D19). That keeps the stroke length constant (which is
what makes Mimic's rigid, scale-free transform correct at every level) and guarantees
the puck never starts inside the target region.
"""

from __future__ import annotations

import math
import random
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from ..assets import TARGET_MARKER_Z
from ..levels import PUSH_DISTANCE

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_puck_and_target(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
    puck_pose_range: dict | None = None,
    bearing_range: tuple | None = None,
):
    if env_ids is None:
        return
    puck_pose_range = puck_pose_range or {}
    bearing_range = bearing_range or (math.pi / 2, math.pi / 2)

    obj = env.scene[object_cfg.name]
    marker = env.scene[target_cfg.name]

    if not hasattr(env, "_push_reset_xy") or env._push_reset_xy.shape[0] != env.num_envs:
        env._push_reset_xy = torch.zeros(env.num_envs, 2, device=env.device)
        env._push_contact_latch = torch.zeros(env.num_envs, 1, dtype=torch.bool, device=env.device)

    for cur_env in env_ids.tolist():
        ids = torch.tensor([cur_env], device=env.device)
        px = random.uniform(*puck_pose_range.get("x", (0.0, 0.0)))
        py = random.uniform(*puck_pose_range.get("y", (0.0, 0.0)))
        pz = random.uniform(*puck_pose_range.get("z", (0.0, 0.0)))
        bearing = random.uniform(*bearing_range)

        origin = env.scene.env_origins[cur_env, 0:3]
        puck_xyz = torch.tensor([[px, py, pz]], device=env.device) + origin
        upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
        obj.write_root_pose_to_sim(torch.cat([puck_xyz, upright], dim=-1), env_ids=ids)
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=ids)

        tx = px + PUSH_DISTANCE * math.cos(bearing)
        ty = py + PUSH_DISTANCE * math.sin(bearing)
        tgt_xyz = torch.tensor([[tx, ty, TARGET_MARKER_Z]], device=env.device) + origin
        marker.write_root_pose_to_sim(torch.cat([tgt_xyz, upright], dim=-1), env_ids=ids)
        marker.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=ids)

        # Reference for the contact signal, and clear its latch for this env.
        env._push_reset_xy[cur_env, 0] = puck_xyz[0, 0]
        env._push_reset_xy[cur_env, 1] = puck_xyz[0, 1]
        env._push_contact_latch[cur_env] = False
