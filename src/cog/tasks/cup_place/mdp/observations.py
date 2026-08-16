"""Task observation terms (env-origin-relative frames, matching stack conventions)."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_pos(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg = SceneEntityCfg("cup")) -> torch.Tensor:
    """Object root position relative to env origin. Shape (num_envs, 3)."""
    obj = env.scene[object_cfg.name]
    return obj.data.root_pos_w - env.scene.env_origins


def object_quat(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg = SceneEntityCfg("cup")) -> torch.Tensor:
    """Object root orientation (w, x, y, z). Shape (num_envs, 4)."""
    obj = env.scene[object_cfg.name]
    return obj.data.root_quat_w
