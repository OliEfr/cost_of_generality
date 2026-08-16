"""Mimic cfgs: task cfg + MimicEnvCfg, 3 subtasks (open -> grasp -> stow).

Consecutive segments sharing a reference frame gain nothing from splitting, so
the handle-grasp + pull segments form ONE subtask (ref cabinet); then grasp
(ref object); then stow (ref cabinet — the drawer opening amount is generated
consistently, so cabinet-root-relative stow transforms are well-posed).
"""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from ..levels import SUB_LEVELS
from ..config.franka.drawer_stow_ik_rel_env_cfg import STATE_CFGS
from ..config.franka.drawer_stow_ik_rel_visuomotor_env_cfg import VISUOMOTOR_CFGS


def _install_mimic(self):
    self.datagen_config.name = f"cog_drawer_stow_{self.sub_level_key}"
    self.datagen_config.generation_guarantee = True
    self.datagen_config.generation_keep_failed = True
    self.datagen_config.generation_num_trials = 10
    self.datagen_config.generation_select_src_per_subtask = True
    self.datagen_config.generation_transform_first_robot_pose = False
    self.datagen_config.generation_interpolate_from_last_target_pose = True
    self.datagen_config.seed = 1

    common = dict(
        selection_strategy="nearest_neighbor_object",
        selection_strategy_kwargs={"nn_k": 3},
        action_noise=0.02,
        num_interpolation_steps=5,
        num_fixed_steps=0,
        apply_noise_during_interpolation=False,
    )
    self.subtask_configs["franka"] = [
        SubTaskConfig(
            object_ref="cabinet",
            subtask_term_signal="drawer_opened_1",
            subtask_term_offset_range=(10, 20),
            **common,
        ),
        SubTaskConfig(
            object_ref="object",
            subtask_term_signal="grasp_2",
            subtask_term_offset_range=(10, 20),
            **common,
        ),
        SubTaskConfig(
            object_ref="cabinet",
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            **common,
        ),
    ]


def _make_mimic_cfg(name: str, base_cls):
    def __post_init__(self):
        base_cls.__post_init__(self)
        _install_mimic(self)

    return configclass(type(name, (base_cls, MimicEnvCfg), {"__post_init__": __post_init__}))


MIMIC_STATE_CFGS = {
    key: _make_mimic_cfg(f"FrankaDrawerStowIKRelMimicEnvCfg_{key}", STATE_CFGS[key]) for key in SUB_LEVELS
}
MIMIC_VISUOMOTOR_CFGS = {
    key: _make_mimic_cfg(f"FrankaDrawerStowVisuomotorMimicEnvCfg_{key}", VISUOMOTOR_CFGS[key])
    for key in SUB_LEVELS
}
