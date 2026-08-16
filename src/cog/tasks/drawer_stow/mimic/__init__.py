"""Mimic env registration for drawer_stow (state + visuomotor, per sub-level)."""

import gymnasium as gym

from ..levels import SUB_LEVELS
from .drawer_stow_mimic_env_cfg import MIMIC_STATE_CFGS, MIMIC_VISUOMOTOR_CFGS

for _key in SUB_LEVELS:
    gym.register(
        id=f"Cog-DrawerStow-{_key}-IK-Rel-Mimic-v0",
        entry_point="cog.tasks.drawer_stow.mimic.drawer_stow_mimic_env:FrankaDrawerStowIKRelMimicEnv",
        kwargs={"env_cfg_entry_point": MIMIC_STATE_CFGS[_key]},
        disable_env_checker=True,
    )
    gym.register(
        id=f"Cog-DrawerStow-{_key}-IK-Rel-Visuomotor-Mimic-v0",
        entry_point="cog.tasks.drawer_stow.mimic.drawer_stow_mimic_env:FrankaDrawerStowIKRelMimicEnv",
        kwargs={"env_cfg_entry_point": MIMIC_VISUOMOTOR_CFGS[_key]},
        disable_env_checker=True,
    )
