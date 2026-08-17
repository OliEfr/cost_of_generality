"""Mimic env for push_target: publishes the SYNTHETIC PUSH FRAME (D19).

This is the crux of Task 3. Mimic expresses each subtask's EEF trajectory relative to
ONE 4x4 reference pose and rigidly re-applies it (data_generator.py:52-83). A push
depends on TWO bodies -- the puck and the target -- so neither raw body pose works as
the reference: anchoring on the puck reproduces the source demo's puck->target VECTOR
(so a moved target is never reached), and anchoring on the target loses the approach.

`push_frame` resolves this: origin at the puck centre, +x pointing at the target,
roll/pitch stripped. A source stroke is then "advance along +x of this frame", and
re-applying it rigidly in a new scene pushes along the NEW puck->target direction.
Direction adapts for free; distance cannot (a rigid transform has no scale), which is
exactly why levels.py holds |puck - target| constant.

Stripping roll/pitch matters: the puck's own quaternion tumbles and spins freely while
it slides, and a reference frame inheriting that would rotate the whole replayed stroke
out of the table plane.
"""

from collections.abc import Sequence

import torch

import isaaclab.utils.math as PoseUtils
from isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env import FrankaCubeStackIKRelMimicEnv


class FrankaPushTargetIKRelMimicEnv(FrankaCubeStackIKRelMimicEnv):
    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """Adds `push_frame` alongside the raw rigid-object poses."""
        if env_ids is None:
            env_ids = slice(None)
        object_pose_matrix = super().get_object_poses(env_ids)

        puck_pos = self.scene["object"].data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        target_pos = (
            self.scene["target_marker"].data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        )

        delta = target_pos[:, :2] - puck_pos[:, :2]
        yaw = torch.atan2(delta[:, 1], delta[:, 0])
        zeros = torch.zeros_like(yaw)
        # roll = pitch = 0: keep the frame level regardless of how the puck tumbles.
        quat = PoseUtils.quat_from_euler_xyz(zeros, zeros, yaw)

        object_pose_matrix["push_frame"] = PoseUtils.make_pose(
            puck_pos, PoseUtils.matrix_from_quat(quat)
        )
        return object_pose_matrix

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Single-subtask design: the only subtask is the final one, which by Mimic's
        convention carries `subtask_term_signal=None`, so no signal is consumed here.
        The contact predicate is still returned -- annotate_demos.py --auto requires
        this method to be genuinely overridden (it checks __func__ identity), and
        publishing it keeps the two-subtask fallback a pure config change.
        """
        if env_ids is None:
            env_ids = slice(None)
        return {"contact_1": self.obs_buf["subtask_terms"]["contact_1"][env_ids]}
