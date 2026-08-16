"""Mimic cfgs: task cfg + MimicEnvCfg with 2-subtask grasp->place (spec 02 section 2)."""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from ..levels import SUB_LEVELS
from ..config.franka.cup_place_ik_rel_env_cfg import STATE_CFGS
from ..config.franka.cup_place_ik_rel_visuomotor_env_cfg import VISUOMOTOR_CFGS


def _install_mimic(self):
    self.datagen_config.name = f"cog_cup_place_{self.sub_level_key}"
    self.datagen_config.generation_guarantee = True
    self.datagen_config.generation_keep_failed = True
    self.datagen_config.generation_num_trials = 10
    self.datagen_config.generation_select_src_per_subtask = True
    self.datagen_config.generation_transform_first_robot_pose = False
    self.datagen_config.generation_interpolate_from_last_target_pose = True
    self.datagen_config.seed = 1

    self.subtask_configs["franka"] = [
        SubTaskConfig(
            object_ref="cup",
            subtask_term_signal="grasp_1",
            subtask_term_offset_range=(10, 20),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.02,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        ),
        SubTaskConfig(
            object_ref="goal_marker",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            action_noise=0.02,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        ),
    ]


def _make_mimic_cfg(name: str, base_cls):
    def __post_init__(self):
        base_cls.__post_init__(self)
        _install_mimic(self)

    return configclass(type(name, (base_cls, MimicEnvCfg), {"__post_init__": __post_init__}))


MIMIC_STATE_CFGS = {
    key: _make_mimic_cfg(f"FrankaCupPlaceIKRelMimicEnvCfg_{key}", STATE_CFGS[key]) for key in SUB_LEVELS
}
MIMIC_VISUOMOTOR_CFGS = {
    key: _make_mimic_cfg(f"FrankaCupPlaceVisuomotorMimicEnvCfg_{key}", VISUOMOTOR_CFGS[key])
    for key in SUB_LEVELS
}
