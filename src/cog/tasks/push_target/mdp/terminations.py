"""Success termination for push_target: puck inside the target disk AND settled.

`settled` is load-bearing, not cosmetic. Mimic OR-latches success across every
timestep of a generated episode, so without a velocity clause a puck that slides
THROUGH the disk and off the far side would be recorded as a success (D19 item 7).

There is deliberately no `released` clause: the pusher's fingers stay closed for the
whole episode, so a gripper-open test could never fire.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from ..assets import SUCCESS_RADIUS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def puck_on_target(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_marker"),
    success_radius: float = SUCCESS_RADIUS,
    max_lin_vel: float = 0.02,
) -> torch.Tensor:
    """True when the puck centre is within ``success_radius`` (planar) of the target
    centre and has essentially stopped moving.

    A blade-clearance clause ("the pusher has backed off", the non-prehensile analogue of
    T1/T2's `released`) was TRIED and REVERTED on 2026-08-17: it cost 5-35 points of expert
    SR across levels, because episodes that legitimately succeed as the puck settles were
    then also required to complete a full retreat inside the episode budget. Its purpose --
    keeping mid-stroke, disk-edge successes out of the SOURCE demos -- is better served by
    selecting sources on final placement error at recording time, which costs nothing and
    does not distort the success definition the whole study is measured against.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    marker: RigidObject = env.scene[target_cfg.name]

    planar_err = torch.linalg.vector_norm(
        (obj.data.root_pos_w - marker.data.root_pos_w)[:, :2], dim=1
    )
    speed = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=1)
    return torch.logical_and(planar_err <= success_radius, speed <= max_lin_vel)


def puck_off_table(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    min_height: float = -0.05,
) -> torch.Tensor:
    """Failure guard: the puck has been shoved off the tabletop."""
    obj: RigidObject = env.scene[object_cfg.name]
    return (obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]) < min_height
