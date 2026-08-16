"""Mimic env: subclass of FrankaCubeStackIKRelMimicEnv (cup_place pattern).

Only get_subtask_term_signals is overridden; eef pose/action conversion is
inherited (reads policy eef_pos/eef_quat, which our cfg provides).
NOTE VERIFY (d): subtask object_ref="cabinet" is an Articulation — the datagen
pose path must accept it (smoke-tested before any generation).
"""

from collections.abc import Sequence

import torch

from isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env import FrankaCubeStackIKRelMimicEnv


class FrankaDrawerStowIKRelMimicEnv(FrankaCubeStackIKRelMimicEnv):
    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        subtask_terms = self.obs_buf["subtask_terms"]
        return {
            "drawer_opened_1": subtask_terms["drawer_opened_1"][env_ids],
            "grasp_2": subtask_terms["grasp_2"][env_ids],
        }
