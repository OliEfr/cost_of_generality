"""Mimic env registration for cup_place (state + visuomotor, per sub-level)."""

import gymnasium as gym

from ..levels import SUB_LEVELS
from .cup_place_mimic_env_cfg import MIMIC_STATE_CFGS, MIMIC_VISUOMOTOR_CFGS

for _key in SUB_LEVELS:
    gym.register(
        id=f"Cog-CupPlace-{_key}-IK-Rel-Mimic-v0",
        entry_point="cog.tasks.cup_place.mimic.cup_place_mimic_env:FrankaCupPlaceIKRelMimicEnv",
        kwargs={"env_cfg_entry_point": MIMIC_STATE_CFGS[_key]},
        disable_env_checker=True,
    )
    gym.register(
        id=f"Cog-CupPlace-{_key}-IK-Rel-Visuomotor-Mimic-v0",
        entry_point="cog.tasks.cup_place.mimic.cup_place_mimic_env:FrankaCupPlaceIKRelMimicEnv",
        kwargs={"env_cfg_entry_point": MIMIC_VISUOMOTOR_CFGS[_key]},
        disable_env_checker=True,
    )
