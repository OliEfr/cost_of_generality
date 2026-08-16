"""Gym registration for all drawer_stow env IDs (state + visuomotor, per sub-level)."""

import gymnasium as gym

from ...levels import SUB_LEVELS
from .drawer_stow_ik_rel_env_cfg import STATE_CFGS
from .drawer_stow_ik_rel_visuomotor_env_cfg import VISUOMOTOR_CFGS

for _key in SUB_LEVELS:
    gym.register(
        id=f"Cog-DrawerStow-{_key}-IK-Rel-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": STATE_CFGS[_key]},
        disable_env_checker=True,
    )
    gym.register(
        id=f"Cog-DrawerStow-{_key}-IK-Rel-Visuomotor-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": VISUOMOTOR_CFGS[_key]},
        disable_env_checker=True,
    )
