"""Warp-kernel pick->place state machine expert (spec 04 section 3).

Fixes applied vs the stock lift_cube_sm: correct per-state wait constants,
frozen grasp target after grasp begins, terminated|truncated handling is the
caller's job, symmetric-cup top-down grasp (DOWN_QUAT), world-frame hovers.
Emits ABSOLUTE EE pose targets; the driver converts to IK-Rel deltas
(convert_abs_to_rel_actions) so recorded actions match the Mimic env exactly.
"""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_unique

wp.init()

DOWN_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)  # gripper pointing straight down


class GripperState:
    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class SmState:
    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)
    APPROACH_ABOVE_GOAL = wp.constant(5)
    LOWER_TO_GOAL = wp.constant(6)
    RELEASE_OBJECT = wp.constant(7)
    RETREAT = wp.constant(8)
    DONE = wp.constant(9)


class SmWait:
    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.4)
    LIFT_OBJECT = wp.constant(0.4)
    APPROACH_ABOVE_GOAL = wp.constant(0.6)
    LOWER_TO_GOAL = wp.constant(0.5)
    RELEASE_OBJECT = wp.constant(0.5)
    RETREAT = wp.constant(0.5)


@wp.func
def near(a: wp.vec3, b: wp.vec3, threshold: float) -> bool:
    return wp.length(a - b) < threshold


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    grasp_pose: wp.array(dtype=wp.transform),
    goal_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    hover_offset: wp.array(dtype=wp.transform),
    place_hover_offset: wp.array(dtype=wp.transform),
    position_threshold: float,
):
    tid = wp.tid()
    state = sm_state[tid]
    ee_p = wp.transform_get_translation(ee_pose[tid])
    if state == SmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= SmWait.REST:
            sm_state[tid] = SmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == SmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(hover_offset[tid], grasp_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if near(ee_p, wp.transform_get_translation(des_ee_pose[tid]), position_threshold):
            if sm_wait_time[tid] >= SmWait.APPROACH_ABOVE_OBJECT:
                sm_state[tid] = SmState.APPROACH_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == SmState.APPROACH_OBJECT:
        des_ee_pose[tid] = grasp_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if near(ee_p, wp.transform_get_translation(des_ee_pose[tid]), position_threshold):
            if sm_wait_time[tid] >= SmWait.APPROACH_OBJECT:
                sm_state[tid] = SmState.GRASP_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == SmState.GRASP_OBJECT:
        des_ee_pose[tid] = grasp_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= SmWait.GRASP_OBJECT:
            sm_state[tid] = SmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == SmState.LIFT_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(hover_offset[tid], grasp_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        if near(ee_p, wp.transform_get_translation(des_ee_pose[tid]), position_threshold):
            if sm_wait_time[tid] >= SmWait.LIFT_OBJECT:
                sm_state[tid] = SmState.APPROACH_ABOVE_GOAL
                sm_wait_time[tid] = 0.0
    elif state == SmState.APPROACH_ABOVE_GOAL:
        des_ee_pose[tid] = wp.transform_multiply(place_hover_offset[tid], goal_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        if near(ee_p, wp.transform_get_translation(des_ee_pose[tid]), position_threshold):
            if sm_wait_time[tid] >= SmWait.APPROACH_ABOVE_GOAL:
                sm_state[tid] = SmState.LOWER_TO_GOAL
                sm_wait_time[tid] = 0.0
    elif state == SmState.LOWER_TO_GOAL:
        des_ee_pose[tid] = goal_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        if near(ee_p, wp.transform_get_translation(des_ee_pose[tid]), position_threshold):
            if sm_wait_time[tid] >= SmWait.LOWER_TO_GOAL:
                sm_state[tid] = SmState.RELEASE_OBJECT
                sm_wait_time[tid] = 0.0
    elif state == SmState.RELEASE_OBJECT:
        des_ee_pose[tid] = goal_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= SmWait.RELEASE_OBJECT:
            sm_state[tid] = SmState.RETREAT
            sm_wait_time[tid] = 0.0
    elif state == SmState.RETREAT:
        des_ee_pose[tid] = wp.transform_multiply(place_hover_offset[tid], goal_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if near(ee_p, wp.transform_get_translation(des_ee_pose[tid]), position_threshold):
            if sm_wait_time[tid] >= SmWait.RETREAT:
                sm_state[tid] = SmState.DONE
                sm_wait_time[tid] = 0.0
    elif state == SmState.DONE:
        des_ee_pose[tid] = wp.transform_multiply(place_hover_offset[tid], goal_pose[tid])
        gripper_state[tid] = GripperState.OPEN
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


def _to_wp_transform(pose_wxyz: torch.Tensor) -> torch.Tensor:
    """(N,7) pos+quat(wxyz) -> (N,7) warp transform layout pos+quat(xyzw)."""
    return pose_wxyz[:, [0, 1, 2, 4, 5, 6, 3]].contiguous()


class PlaceSm:
    """Vectorized cup-place state machine."""

    def __init__(self, dt: float, num_envs: int, device, position_threshold: float = 0.012):
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold
        self.dt = torch.full((num_envs,), dt, device=device)
        self.sm_state = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.sm_wait_time = torch.zeros(num_envs, device=device)
        self.des_ee_pose = torch.zeros(num_envs, 7, device=device)
        self.des_gripper_state = torch.zeros(num_envs, device=device)
        self.frozen_grasp_pose = None  # (N,7) wxyz, latched once grasping starts

        hover = torch.zeros(num_envs, 7, device=device)
        hover[:, 2] = 0.10
        hover[:, 6] = 1.0  # identity quat in xyzw layout (qw last)
        self.hover_offset = hover
        place_hover = torch.zeros(num_envs, 7, device=device)
        place_hover[:, 2] = 0.12
        place_hover[:, 6] = 1.0
        self.place_hover_offset = place_hover

        self.dt_wp = wp.from_torch(self.dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.hover_offset_wp = wp.from_torch(self.hover_offset, wp.transform)
        self.place_hover_offset_wp = wp.from_torch(self.place_hover_offset, wp.transform)

    def reset_idx(self, env_ids):
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0
        if self.frozen_grasp_pose is not None:
            self.frozen_grasp_pose[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, grasp_pose: torch.Tensor, goal_pose: torch.Tensor) -> torch.Tensor:
        """All inputs (N,7) pos+quat(wxyz), env-local frame. Returns (N,8) abs pose + gripper."""
        if self.frozen_grasp_pose is None:
            self.frozen_grasp_pose = grasp_pose.clone()
        # update the latched grasp target only while not yet grasping
        pregrasp = self.sm_state <= SmState.APPROACH_OBJECT
        self.frozen_grasp_pose[pregrasp] = grasp_pose[pregrasp]

        ee_wp = wp.from_torch(_to_wp_transform(ee_pose), wp.transform)
        grasp_wp = wp.from_torch(_to_wp_transform(self.frozen_grasp_pose), wp.transform)
        goal_wp = wp.from_torch(_to_wp_transform(goal_pose), wp.transform)
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.dt_wp, self.sm_state_wp, self.sm_wait_time_wp,
                ee_wp, grasp_wp, goal_wp,
                self.des_ee_pose_wp, self.des_gripper_state_wp,
                self.hover_offset_wp, self.place_hover_offset_wp,
                self.position_threshold,
            ],
            device=self.dt_wp.device,
        )
        wp.synchronize()
        des_pose_wxyz = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        return torch.cat([des_pose_wxyz, self.des_gripper_state.unsqueeze(-1)], dim=-1)


def convert_abs_to_rel_actions(
    abs_target: torch.Tensor, tcp_pos: torch.Tensor, tcp_quat: torch.Tensor
) -> torch.Tensor:
    """(N,8) abs pose+gripper -> (N,7) IK-Rel action, mimicking
    FrankaCubeStackIKRelMimicEnv.target_eef_pose_to_action (spec 04 section 3.5)."""
    delta_pos = (abs_target[:, 0:3] - tcp_pos).clamp(-0.05, 0.05)
    dquat = quat_mul(abs_target[:, 3:7], quat_conjugate(tcp_quat))
    delta_rot = axis_angle_from_quat(quat_unique(dquat)).clamp(-0.2, 0.2)
    return torch.cat([delta_pos, delta_rot, abs_target[:, 7:8]], dim=-1)
