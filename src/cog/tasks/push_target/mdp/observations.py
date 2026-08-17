"""Task observation terms for push_target."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_pos(
    env: "ManagerBasedRLEnv",
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
) -> torch.Tensor:
    """Target disk centre, env-relative. (N, 3)."""
    marker = env.scene[target_cfg.name]
    return marker.data.root_pos_w - env.scene.env_origins


def puck_to_target(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
) -> torch.Tensor:
    """Planar puck->target vector. (N, 2). Privileged: recorded, not fed to the
    vision policy, which must read the target off the table camera instead."""
    obj = env.scene[object_cfg.name]
    marker = env.scene[target_cfg.name]
    return (marker.data.root_pos_w - obj.data.root_pos_w)[:, :2]


def puck_contacted(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    min_displacement: float = 0.015,
) -> torch.Tensor:
    """Boolean: the puck has moved at least ``min_displacement`` from where it was
    reset, i.e. contact has been established and the stroke is under way. (N, 1).

    Kept even though D19 chose a SINGLE-subtask decomposition, so switching to the
    two-subtask fallback is a config edit rather than a code change. It is latched
    monotone (max over the episode) and false at t=0 by construction, which is what
    Mimic requires of a non-final subtask signal: an always-true signal makes the
    boundary diffs all-zero and crashes the pool loader, and a chattering one gets
    segmented at the first NONZERO diff rather than the first rising edge.
    """
    obj = env.scene[object_cfg.name]
    if not hasattr(env, "_push_reset_xy") or env._push_reset_xy.shape[0] != env.num_envs:
        # Before the first reset event runs there is no reference; report "no contact".
        return torch.zeros(env.num_envs, 1, dtype=torch.bool, device=env.device)
    moved = torch.linalg.vector_norm(obj.data.root_pos_w[:, :2] - env._push_reset_xy, dim=1)
    now = (moved >= min_displacement).unsqueeze(-1)
    env._push_contact_latch = torch.logical_or(
        getattr(env, "_push_contact_latch", torch.zeros_like(now)), now
    )
    return env._push_contact_latch
