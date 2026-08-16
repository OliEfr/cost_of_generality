"""Scripted drawer-stow expert (torch, vectorized over envs).

Differences vs the stock open_cabinet_sm: offsets composed in the HANDLE frame
(correct under cabinet yaw randomization), a ramped pull latched at grasp time
until the drawer joint passes its target (stock used a single -1.5 cm world
offset), then a cup_place-style yaw-aligned top grasp of the box and a place
into the open cavity. All traverses between the handle and the plinth run at
TRAVERSE_Z, above the open drawer rim (~0.78 m), to avoid clipping its walls.

Emits ABSOLUTE TCP pose targets; the driver converts with
cup_place.state_machine.convert_abs_to_rel_actions (same IK-Rel contract).
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import quat_apply, quat_mul

from ..cup_place.state_machine import DOWN_QUAT_WXYZ  # noqa: F401  (re-export for driver)
from .assets import DRAWER_CAVITY_FLOOR_Z, DRAWER_CAVITY_RIM_Z
from .levels import DRAWER_OPEN_TARGET


class Sm:
    REST = 0
    APPROACH_INFRONT_HANDLE = 1
    APPROACH_HANDLE = 2
    GRASP_HANDLE = 3
    PULL_DRAWER = 4
    RELEASE_HANDLE = 5
    RETREAT_FROM_HANDLE = 6
    APPROACH_ABOVE_OBJECT = 7
    APPROACH_OBJECT = 8
    GRASP_OBJECT = 9
    LIFT_OBJECT = 10
    APPROACH_ABOVE_DRAWER = 11
    LOWER_INTO_DRAWER = 12
    RELEASE_OBJECT = 13
    RETREAT_UP = 14
    DONE = 15


# minimum dwell per state (s); position-gated states also need `near`
WAIT = {
    Sm.REST: 0.3,
    Sm.APPROACH_INFRONT_HANDLE: 0.5,
    Sm.APPROACH_HANDLE: 0.5,
    Sm.GRASP_HANDLE: 0.5,
    Sm.PULL_DRAWER: 0.0,           # joint-gated
    Sm.RELEASE_HANDLE: 0.4,
    Sm.RETREAT_FROM_HANDLE: 0.3,
    Sm.APPROACH_ABOVE_OBJECT: 0.4,
    Sm.APPROACH_OBJECT: 0.5,
    Sm.GRASP_OBJECT: 0.4,
    Sm.LIFT_OBJECT: 0.3,
    Sm.APPROACH_ABOVE_DRAWER: 0.5,
    Sm.LOWER_INTO_DRAWER: 0.5,
    Sm.RELEASE_OBJECT: 0.5,
    Sm.RETREAT_UP: 0.3,
}

HANDLE_APPROACH_DIST = 0.10     # in front of the handle, handle frame -z
PULL_RATE = 0.10                # m/s commanded pull speed
PULL_OVERSHOOT = 0.04           # command a little past the joint target
PULL_TIMEOUT = 6.0              # bail (episode will fail success) after this
TRAVERSE_Z = 0.92               # safe TCP height above the open drawer rim
OBJECT_DROP_CLEARANCE = 0.02    # object bottom above cavity floor at release


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    """(N,) yaw -> (N,4) wxyz quat about world z."""
    half = yaw * 0.5
    q = torch.zeros(yaw.shape[0], 4, device=yaw.device)
    q[:, 0] = torch.cos(half)
    q[:, 3] = torch.sin(half)
    return q


def _box_grasp_quat(obj_quat: torch.Tensor) -> torch.Tensor:
    """Top-down TCP quat with yaw aligned to the box faces (mod pi/2)."""
    # yaw from quat (wxyz): atan2(2(wz+xy), 1-2(y^2+z^2))
    w, x, y, z = obj_quat.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    half_pi = math.pi / 2
    yaw_eff = yaw - torch.round(yaw / half_pi) * half_pi
    down = torch.tensor(DOWN_QUAT_WXYZ, device=obj_quat.device).expand(obj_quat.shape[0], 4)
    return quat_mul(_yaw_quat(yaw_eff), down)


class DrawerStowSm:
    """Vectorized drawer-stow state machine. All poses (N,7) pos+quat wxyz, env-local."""

    def __init__(self, dt: float, num_envs: int, device, position_threshold: float = 0.012):
        self.dt = dt
        self.num_envs = num_envs
        self.device = device
        self.thr = position_threshold
        self.state = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.wait = torch.zeros(num_envs, device=device)
        self.pull_progress = torch.zeros(num_envs, device=device)
        self.latched_handle = torch.zeros(num_envs, 7, device=device)
        self.latched_grasp = torch.zeros(num_envs, 7, device=device)

    def reset_idx(self, env_ids):
        self.state[env_ids] = Sm.REST
        self.wait[env_ids] = 0.0
        self.pull_progress[env_ids] = 0.0
        self.latched_handle[env_ids] = 0.0
        self.latched_grasp[env_ids] = 0.0

    def compute(
        self,
        ee_pose: torch.Tensor,        # (N,7) TCP pose
        handle_pose: torch.Tensor,    # (N,7) grasp-ready handle frame (cabinet_frame sensor)
        drawer_joint: torch.Tensor,   # (N,) drawer_top_joint position
        object_pose: torch.Tensor,    # (N,7) stow object root pose
        drawer_body_pose: torch.Tensor,  # (N,7) drawer_top body pose (cavity frame)
        grasp_z_offset: float,
        object_half_size: float,
    ) -> torch.Tensor:
        """Returns (N,8) absolute TCP pose target + gripper (+1 open / -1 close)."""
        N, dev = self.num_envs, self.device
        s = self.state
        des = torch.zeros(N, 8, device=dev)
        des[:, 7] = 1.0  # default open

        # latch handle target until the grasp begins; latch object grasp likewise
        pre_handle = s <= Sm.APPROACH_HANDLE
        self.latched_handle[pre_handle] = handle_pose[pre_handle]
        obj_grasp = torch.cat(
            [object_pose[:, 0:2],
             (object_pose[:, 2] + grasp_z_offset).unsqueeze(-1),
             _box_grasp_quat(object_pose[:, 3:7])], dim=-1)
        pre_obj = s <= Sm.APPROACH_OBJECT
        self.latched_grasp[pre_obj] = obj_grasp[pre_obj]

        lh = self.latched_handle
        lg = self.latched_grasp
        pull_dir = quat_apply(lh[:, 3:7], torch.tensor([0.0, 0.0, -1.0], device=dev).expand(N, 3))

        # cavity stow target from the (moving) drawer body frame
        stow_xy = drawer_body_pose[:, 0:2]
        floor_z = drawer_body_pose[:, 2] + DRAWER_CAVITY_FLOOR_Z
        rim_z = drawer_body_pose[:, 2] + DRAWER_CAVITY_RIM_Z
        stow_tcp_z = floor_z + object_half_size + OBJECT_DROP_CLEARANCE + grasp_z_offset

        def assign(mask, pos, quat, grip):
            des[mask, 0:3] = pos[mask]
            des[mask, 3:7] = quat[mask]
            des[mask, 7] = grip

        m = s == Sm.REST
        assign(m, ee_pose[:, 0:3], ee_pose[:, 3:7], 1.0)

        m = s == Sm.APPROACH_INFRONT_HANDLE
        approach = lh[:, 0:3] + pull_dir * HANDLE_APPROACH_DIST
        assign(m, approach, lh[:, 3:7], 1.0)

        m = (s == Sm.APPROACH_HANDLE) | (s == Sm.GRASP_HANDLE)
        assign(m, lh[:, 0:3], lh[:, 3:7], 1.0)
        des[s == Sm.GRASP_HANDLE, 7] = -1.0

        m = s == Sm.PULL_DRAWER
        pull_pos = lh[:, 0:3] + pull_dir * self.pull_progress.unsqueeze(-1)
        assign(m, pull_pos, lh[:, 3:7], -1.0)
        self.pull_progress[m] = torch.clamp(
            self.pull_progress[m] + PULL_RATE * self.dt,
            max=DRAWER_OPEN_TARGET + PULL_OVERSHOOT,
        )

        release_pos = lh[:, 0:3] + pull_dir * self.pull_progress.unsqueeze(-1)
        # retreat straight up (an extra -x offset at z 0.92 with a horizontal
        # wrist is IK-awkward near the base pillar); rotate to the down-facing
        # object-grasp quat during the ascent
        retreat_pos = torch.cat([release_pos[:, 0:2],
                                 torch.full((N, 1), TRAVERSE_Z, device=dev)], dim=-1)
        assign(s == Sm.RELEASE_HANDLE, release_pos, lh[:, 3:7], 1.0)
        assign(s == Sm.RETREAT_FROM_HANDLE, retreat_pos, lg[:, 3:7], 1.0)

        m = s == Sm.APPROACH_ABOVE_OBJECT
        above_obj = torch.cat([lg[:, 0:2], torch.full((N, 1), TRAVERSE_Z, device=dev)], dim=-1)
        assign(m, above_obj, lg[:, 3:7], 1.0)

        m = (s == Sm.APPROACH_OBJECT) | (s == Sm.GRASP_OBJECT)
        assign(m, lg[:, 0:3], lg[:, 3:7], 1.0)
        des[s == Sm.GRASP_OBJECT, 7] = -1.0

        m = s == Sm.LIFT_OBJECT
        assign(m, above_obj, lg[:, 3:7], -1.0)

        m = s == Sm.APPROACH_ABOVE_DRAWER
        above_drawer = torch.cat([stow_xy, torch.full((N, 1), TRAVERSE_Z, device=dev)], dim=-1)
        assign(m, above_drawer, lg[:, 3:7], -1.0)

        m = s == Sm.LOWER_INTO_DRAWER
        stow_pos = torch.cat([stow_xy, stow_tcp_z.unsqueeze(-1)], dim=-1)
        assign(m, stow_pos, lg[:, 3:7], -1.0)

        m = s == Sm.RELEASE_OBJECT
        assign(m, stow_pos, lg[:, 3:7], 1.0)

        m = (s == Sm.RETREAT_UP) | (s == Sm.DONE)
        assign(m, above_drawer, lg[:, 3:7], 1.0)

        # ---- transitions (evaluated on a frozen copy: one step per call max) ----
        s0 = self.state.clone()
        near = (ee_pose[:, 0:3] - des[:, 0:3]).norm(dim=-1) < self.thr
        waited = torch.zeros(N, dtype=torch.bool, device=dev)
        for st, w in WAIT.items():
            waited |= (s0 == st) & (self.wait >= w)

        # states whose transition needs only the dwell time, not position
        TIME_ONLY = (Sm.REST, Sm.GRASP_HANDLE, Sm.GRASP_OBJECT,
                     Sm.RELEASE_HANDLE, Sm.RELEASE_OBJECT)
        NEXT = {
            Sm.REST: Sm.APPROACH_INFRONT_HANDLE,
            Sm.APPROACH_INFRONT_HANDLE: Sm.APPROACH_HANDLE,
            Sm.APPROACH_HANDLE: Sm.GRASP_HANDLE,
            Sm.GRASP_HANDLE: Sm.PULL_DRAWER,
            Sm.PULL_DRAWER: Sm.RELEASE_HANDLE,
            Sm.RELEASE_HANDLE: Sm.RETREAT_FROM_HANDLE,
            Sm.RETREAT_FROM_HANDLE: Sm.APPROACH_ABOVE_OBJECT,
            Sm.APPROACH_ABOVE_OBJECT: Sm.APPROACH_OBJECT,
            Sm.APPROACH_OBJECT: Sm.GRASP_OBJECT,
            Sm.GRASP_OBJECT: Sm.LIFT_OBJECT,
            Sm.LIFT_OBJECT: Sm.APPROACH_ABOVE_DRAWER,
            Sm.APPROACH_ABOVE_DRAWER: Sm.LOWER_INTO_DRAWER,
            Sm.LOWER_INTO_DRAWER: Sm.RELEASE_OBJECT,
            Sm.RELEASE_OBJECT: Sm.RETREAT_UP,
            Sm.RETREAT_UP: Sm.DONE,
        }
        pull_done = (drawer_joint >= DRAWER_OPEN_TARGET) | (self.wait >= PULL_TIMEOUT)
        for from_state, to_state in NEXT.items():
            if from_state == Sm.PULL_DRAWER:
                go = (s0 == from_state) & pull_done
            elif from_state in TIME_ONLY:
                go = (s0 == from_state) & waited
            else:
                go = (s0 == from_state) & waited & near
            self.state[go] = to_state
            self.wait[go] = 0.0

        self.wait += self.dt
        return des
