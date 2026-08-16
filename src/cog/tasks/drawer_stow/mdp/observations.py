"""Task observation terms for drawer_stow."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def drawer_opened(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
    threshold: float = 0.15,
) -> torch.Tensor:
    """Boolean subtask signal: the configured joint is past ``threshold``. (N, 1)."""
    asset = env.scene[asset_cfg.name]
    return (asset.data.joint_pos[:, asset_cfg.joint_ids] >= threshold).any(dim=1, keepdim=True)
