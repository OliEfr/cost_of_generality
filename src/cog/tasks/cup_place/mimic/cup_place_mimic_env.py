"""Mimic env: subclass of FrankaCubeStackIKRelMimicEnv (spec 02 section 1).

Only get_subtask_term_signals must be overridden (annotate --auto checks the
override via __func__ identity); eef pose/action conversion methods are
inherited and read obs policy eef_pos/eef_quat, which our cfg provides.
"""

from collections.abc import Sequence

import torch

from isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env import FrankaCubeStackIKRelMimicEnv


class FrankaCupPlaceIKRelMimicEnv(FrankaCubeStackIKRelMimicEnv):
    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        subtask_terms = self.obs_buf["subtask_terms"]
        return {"grasp_1": subtask_terms["grasp_1"][env_ids]}
