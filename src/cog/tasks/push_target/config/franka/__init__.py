"""Gym registration for all push_target env IDs (state + visuomotor, per sub-level)."""

import gymnasium as gym

from ...levels import SUB_LEVELS
from .push_target_ik_rel_env_cfg import STATE_CFGS
from .push_target_ik_rel_visuomotor_env_cfg import VISUOMOTOR_CFGS

# NB the family name must be a single dash-free token: the recorder drivers derive the
# sub-level with task.split("-")[2], so "Cog-PushTarget-L0-..." resolves correctly
# whereas "Cog-Push-Target-L0-..." would silently pick the wrong SUB_LEVELS key.
for _key in SUB_LEVELS:
    gym.register(
        id=f"Cog-PushTarget-{_key}-IK-Rel-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": STATE_CFGS[_key]},
        disable_env_checker=True,
    )
    gym.register(
        id=f"Cog-PushTarget-{_key}-IK-Rel-Visuomotor-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={"env_cfg_entry_point": VISUOMOTOR_CFGS[_key]},
        disable_env_checker=True,
    )
