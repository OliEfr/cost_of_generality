"""Mimic env: subclass of FrankaCubeStackIKRelMimicEnv (cup_place pattern).

Only get_subtask_term_signals is overridden; eef pose/action conversion is
inherited (reads policy eef_pos/eef_quat, which our cfg provides).
NOTE VERIFY (d): subtask object_ref="cabinet" is an Articulation — the datagen
pose path must accept it (smoke-tested before any generation).
"""

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env import FrankaCubeStackIKRelMimicEnv


class FrankaDrawerStowIKRelMimicEnv(FrankaCubeStackIKRelMimicEnv):
    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """Base implementation enumerates rigid objects only; the cabinet is an
        Articulation and must be added for object_ref="cabinet" subtasks
        (VERIFY d resolved: without this the pose dict lacks the cabinet)."""
        if env_ids is None:
            env_ids = slice(None)
        object_pose_matrix = super().get_object_poses(env_ids)
        cab_state = self.scene.get_state(is_relative=True)["articulation"]["cabinet"]
        object_pose_matrix["cabinet"] = PoseUtils.make_pose(
            cab_state["root_pose"][env_ids, :3],
            PoseUtils.matrix_from_quat(cab_state["root_pose"][env_ids, 3:7]),
        )
        return object_pose_matrix

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        subtask_terms = self.obs_buf["subtask_terms"]
        return {
            "drawer_opened_1": subtask_terms["drawer_opened_1"][env_ids],
            "grasp_2": subtask_terms["grasp_2"][env_ids],
        }
