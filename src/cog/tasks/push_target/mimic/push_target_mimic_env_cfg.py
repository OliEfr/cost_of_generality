"""Mimic cfgs for push_target: ONE subtask over the whole episode (D19).

Why a single subtask, when T1 used two and T2 used three: for a push, every gain from
splitting is outweighed by a specific failure mode.
  * Mimic inserts a free-space interpolation between subtasks
    (num_interpolation_steps=5, interpolate_from_last_target_pose=True), which would
    jump the EEF mid-stroke and break contact.
  * Non-final subtask signals must be latched, monotone AND false at t=0; a signal true
    at t=0 makes the boundary diffs all-zero and crashes DataGenInfoPool._add_episode
    with a bare IndexError.
  * Boundaries come from the first NONZERO diff, not the first rising edge, so a
    contact predicate that chatters (as contact predicates do) segments at the wrong step.
A single segment anchored on `push_frame` has no boundary to get wrong and keeps the
approach, descent and stroke in one rigid, internally consistent transform.
"""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from ..levels import SUB_LEVELS
from ..config.franka.push_target_ik_rel_env_cfg import STATE_CFGS
from ..config.franka.push_target_ik_rel_visuomotor_env_cfg import VISUOMOTOR_CFGS


def _install_mimic(self):
    self.datagen_config.name = f"cog_push_target_{self.sub_level_key}"
    self.datagen_config.generation_guarantee = True
    self.datagen_config.generation_keep_failed = True
    self.datagen_config.generation_num_trials = 10
    self.datagen_config.generation_select_src_per_subtask = True
    self.datagen_config.generation_transform_first_robot_pose = False
    self.datagen_config.generation_interpolate_from_last_target_pose = True
    self.datagen_config.seed = 1

    self.subtask_configs["franka"] = [
        SubTaskConfig(
            object_ref="push_frame",
            subtask_term_signal=None,       # final (and only) subtask
            subtask_term_offset_range=(0, 0),  # asserted to be (0,0) on the last subtask
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 3},
            # Lower than T1/T2's 0.02: this is an open-loop contact-rich stroke, and
            # noise injected mid-push translates directly into puck placement error
            # against a 5 cm success disk.
            action_noise=0.01,
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
    key: _make_mimic_cfg(f"FrankaPushTargetIKRelMimicEnvCfg_{key}", STATE_CFGS[key]) for key in SUB_LEVELS
}
MIMIC_VISUOMOTOR_CFGS = {
    key: _make_mimic_cfg(f"FrankaPushTargetVisuomotorMimicEnvCfg_{key}", VISUOMOTOR_CFGS[key])
    for key in SUB_LEVELS
}
