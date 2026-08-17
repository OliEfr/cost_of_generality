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
    STAGE_FOR_STOW = 11
    APPROACH_ABOVE_DRAWER = 12
    LOWER_INTO_DRAWER = 13
    RELEASE_OBJECT = 14
    RETREAT_UP = 15
    DONE = 16


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
    Sm.STAGE_FOR_STOW: 0.3,
    Sm.APPROACH_ABOVE_DRAWER: 0.5,
    Sm.LOWER_INTO_DRAWER: 0.5,
    Sm.RELEASE_OBJECT: 0.5,
    Sm.RETREAT_UP: 0.3,
}

HANDLE_APPROACH_DIST = 0.10     # in front of the handle, handle frame -z
RETRY_MIN_OPEN = 0.24           # re-grasp and pull again below this opening -- checked
                                # right after the retreat, BEFORE the drawer's slow creep
                                # (~4-6 cm over a long episode) eats the margin
MAX_PULL_RETRIES = 2
PULL_RATE = 0.12                # m/s commanded pull speed
PULL_OVERSHOOT = 0.02           # command a little past the joint target (limit 0.40)
PULL_TIMEOUT = 6.0              # bail (episode will fail success) after this
OBJECT_DROP_CLEARANCE = 0.02    # object bottom above cavity floor at release

# Waypoints chosen to keep DLS out of extension singularities (debug sessions
# 2026-08-17): rotate to the down-quat LOW and CLOSE (RETREAT_WP), hover the
# plinth low (OBJ_HOVER_Z), climb to stow height at SMALL radius (STAGE_WP),
# then translate out over the drawer wall at constant z.
RETREAT_WP = (0.20, 0.20, 0.88)
STAGE_END = (0.21, 0.10, 0.92)  # z: the arm's practical hold ceiling (higher unwraps the elbow);
                                # x: carried-box leading edge (x+half) must clear the wall line
                                # 0.575-open during the ascent -- 0.25 clipped it at open 0.30
OBJ_HOVER_Z = 0.62
STOW_TRAVERSE_Z = 0.92          # with the 0.20 m pedestal this is mid-workspace: the carried
                                # box crosses the 0.785 wall top with ~10 cm to spare at any x
HANDLE_TO_FACE = 0.03           # handle frame sits 3 cm in front of the drawer face
STOW_BEHIND_FACE = 0.08         # drop well clear of the wall's inner face: the descent
                                # starts with up to ~2 cm XY lag, and a box edge that
                                # overlaps the wall wedges it shut (run 14: drawer 0.31->0)
STOW_RAMP_RATE = 0.09           # m/s ascent ramp (stage leg; free space, sag non-critical)
TRAVERSE_RATE = 0.03            # m/s wall-crossing leg: slower = tighter z tracking (less sag)


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
        self.stow_progress = torch.zeros(num_envs, device=device)
        self.stage_progress = torch.zeros(num_envs, device=device)
        self.stage_start = torch.zeros(num_envs, 3, device=device)
        self.latched_stow = torch.zeros(num_envs, 2, device=device)
        self.lower_progress = torch.zeros(num_envs, device=device)
        self.lower_start = torch.zeros(num_envs, 3, device=device)
        self.pull_retries = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.latched_handle = torch.zeros(num_envs, 7, device=device)
        self.latched_grasp = torch.zeros(num_envs, 7, device=device)

    def reset_idx(self, env_ids):
        self.state[env_ids] = Sm.REST
        self.wait[env_ids] = 0.0
        self.pull_progress[env_ids] = 0.0
        self.stow_progress[env_ids] = 0.0
        self.stage_progress[env_ids] = 0.0
        self.stage_start[env_ids] = 0.0
        self.latched_stow[env_ids] = 0.0
        self.lower_progress[env_ids] = 0.0
        self.lower_start[env_ids] = 0.0
        self.pull_retries[env_ids] = 0
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

        # latch handle target until the grasp begins; latch object grasp likewise.
        # The grasp roll is flipped 180 deg about the approach (z) axis: the
        # frame-transformer orientation as-is drives panda_joint6 to its 3.75 rad
        # limit, which then blocks the post-release lift (DLS cannot escape).
        flip_z = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).expand(N, 4)
        handle_grasp = torch.cat(
            [handle_pose[:, 0:3], quat_mul(handle_pose[:, 3:7], flip_z)], dim=-1)
        pre_handle = s <= Sm.APPROACH_HANDLE
        self.latched_handle[pre_handle] = handle_grasp[pre_handle]
        obj_grasp = torch.cat(
            [object_pose[:, 0:2],
             (object_pose[:, 2] + grasp_z_offset).unsqueeze(-1),
             _box_grasp_quat(object_pose[:, 3:7])], dim=-1)
        pre_obj = s <= Sm.APPROACH_OBJECT
        self.latched_grasp[pre_obj] = obj_grasp[pre_obj]

        lh = self.latched_handle
        lg = self.latched_grasp
        # neutral top-down quat for hover/carry: holding the yaw-ALIGNED grasp
        # quat through the traverse shifts the arm's extension surface with the
        # box yaw, and at +-45 deg the carry equilibrium stops short of the
        # descent-clearance gate (L1/L2 gate collapse: env-dependent failures).
        # A cube only needs alignment during the grasp itself.
        down = torch.tensor(DOWN_QUAT_WXYZ, device=dev).expand(N, 4)
        pull_dir = quat_apply(lh[:, 3:7], torch.tensor([0.0, 0.0, -1.0], device=dev).expand(N, 3))

        # cavity stow target: the drawer BODY origin sits inside the cabinet, so
        # anchor on the live handle pose instead (frame transformer tracks the
        # drawer): from the handle, HANDLE_TO_FACE + STOW_BEHIND_FACE toward the
        # cabinet lands inside the pulled-out cavity for any cabinet pose
        stow_live = handle_pose[:, 0:2] - pull_dir[:, 0:2] * (HANDLE_TO_FACE + STOW_BEHIND_FACE)
        # freeze the target once the traverse starts: if the carried object grazes
        # the drawer wall, a live handle-anchored target chases the closing drawer
        # (runaway feedback observed in debug run 9)
        pre_stow = s <= Sm.STAGE_FOR_STOW
        self.latched_stow[pre_stow] = stow_live[pre_stow]
        stow_xy = self.latched_stow
        floor_z = drawer_body_pose[:, 2] + DRAWER_CAVITY_FLOOR_Z
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
        # rotate to the down-facing grasp quat LOW and CLOSE (in front of the
        # drawer wall) -- rotating while rising to a high target drives DLS
        # into elbow-extension local minima
        retreat_pos = torch.tensor(RETREAT_WP, device=dev).expand(N, 3)
        assign(s == Sm.RELEASE_HANDLE, release_pos, lh[:, 3:7], 1.0)
        assign(s == Sm.RETREAT_FROM_HANDLE, retreat_pos, lg[:, 3:7], 1.0)

        m = s == Sm.APPROACH_ABOVE_OBJECT
        above_obj = torch.cat([lg[:, 0:2], torch.full((N, 1), OBJ_HOVER_Z, device=dev)], dim=-1)
        assign(m, above_obj, down, 1.0)

        m = (s == Sm.APPROACH_OBJECT) | (s == Sm.GRASP_OBJECT)
        assign(m, lg[:, 0:3], lg[:, 3:7], 1.0)
        des[s == Sm.GRASP_OBJECT, 7] = -1.0

        m = s == Sm.LIFT_OBJECT
        assign(m, above_obj, lg[:, 3:7], -1.0)
        self.stage_start[m] = above_obj[m]  # latch ramp-1 origin at the lift top

        # both stow legs are RAMPED: a far target lets DLS unwrap the elbow into
        # the straight branch, whose max height at radius ~0.4 is below the wall
        m = s == Sm.STAGE_FOR_STOW
        stage_end = torch.tensor(STAGE_END, device=dev).expand(N, 3)
        seg1 = stage_end - self.stage_start
        seg1_len = seg1.norm(dim=-1).clamp(min=1e-6)
        frac1 = (self.stage_progress / seg1_len).clamp(max=1.0)
        stage_pos = self.stage_start + seg1 * frac1.unsqueeze(-1)
        assign(m, stage_pos, down, -1.0)
        self.stage_progress[m] = self.stage_progress[m] + STOW_RAMP_RATE * self.dt
        ramp1_done = frac1 >= 1.0

        m = s == Sm.APPROACH_ABOVE_DRAWER
        stage_xy = torch.tensor(STAGE_END[:2], device=dev).expand(N, 2)
        span = stow_xy - stage_xy
        span_len = span.norm(dim=-1).clamp(min=1e-6)
        frac = (self.stow_progress / span_len).clamp(max=1.0)
        ramp_xy = stage_xy + span * frac.unsqueeze(-1)
        above_drawer = torch.cat([ramp_xy, torch.full((N, 1), STOW_TRAVERSE_Z, device=dev)], dim=-1)
        assign(m, above_drawer, down, -1.0)
        self.stow_progress[m] = self.stow_progress[m] + TRAVERSE_RATE * self.dt
        ramp_done = frac >= 1.0

        in_traverse = s == Sm.APPROACH_ABOVE_DRAWER
        self.lower_start[in_traverse] = ee_pose[in_traverse, 0:3]  # release pose latch

        # descent is a ramped DIAGONAL from wherever the traverse equilibrium
        # ended toward the in-cavity drop point: at the start height the box
        # bottom is above the wall top, and the +x component clears the wall
        # edge before contact height; the unreachable-at-height x becomes
        # reachable as z drops
        # hold the ACHIEVED pose (latched at gate pass) while releasing -- the
        # ramp targets are deliberately unreachable and would cause drift
        m = s == Sm.RELEASE_OBJECT
        assign(m, self.lower_start, down, 1.0)

        retreat_up = self.lower_start + torch.tensor([0.0, 0.0, 0.03], device=dev)
        m = (s == Sm.RETREAT_UP) | (s == Sm.DONE)
        assign(m, retreat_up, down, 1.0)

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
            Sm.LIFT_OBJECT: Sm.STAGE_FOR_STOW,
            Sm.STAGE_FOR_STOW: Sm.APPROACH_ABOVE_DRAWER,
            # no LOWER state: descending through the wall-top plane wedges the
            # box on the wall's edge whenever the rear clearance (~mm) is inside
            # tracking noise (runs 14-20). The cavity is a CONTAINER: releasing
            # from carry height drops the cube ~12 cm onto the drawer floor,
            # the walls catch it, and a cube rests identically on any face.
            Sm.APPROACH_ABOVE_DRAWER: Sm.RELEASE_OBJECT,
            Sm.RELEASE_OBJECT: Sm.RETREAT_UP,
            Sm.RETREAT_UP: Sm.DONE,
        }
        pull_done = (drawer_joint >= DRAWER_OPEN_TARGET) | (self.wait >= PULL_TIMEOUT)
        # pull outcome varies with the reset joint jitter (elbow-branch luck,
        # occasional handle slip: openings 0.15-0.34 observed); a shallow pull
        # starves the stow geometry, so re-grasp and pull the remainder
        retry = ((s0 == Sm.RETREAT_FROM_HANDLE) & waited & near
                 & (drawer_joint < RETRY_MIN_OPEN)
                 & (self.pull_retries < MAX_PULL_RETRIES))
        # retries go STRAIGHT to the handle: with a part-open drawer the handle
        # sits near x~0.22 and the in-front staging point (-0.10 further) is
        # unreachable; the short remaining stroke doesn't need staging
        self.state[retry] = Sm.APPROACH_HANDLE
        self.wait[retry] = 0.0
        self.pull_retries[retry] += 1
        self.pull_progress[retry] = 0.0
        for from_state, to_state in NEXT.items():
            if from_state == Sm.PULL_DRAWER:
                go = (s0 == from_state) & pull_done
            elif from_state == Sm.STAGE_FOR_STOW:
                go = (s0 == from_state) & waited & near & ramp1_done
            elif from_state == Sm.APPROACH_ABOVE_DRAWER:
                # gate on the PHYSICAL descent condition: the carried box's
                # trailing edge must clear the drawer wall line, measured along
                # the pull direction from the live handle pose (z is
                # over-commanded for sag compensation, so `near` cannot pass,
                # and target-distance proxies wedged the box in runs 14-18)
                back_dir = -pull_dir[:, 0:2]
                back_dir = back_dir / back_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                depth = ((ee_pose[:, 0:2] - handle_pose[:, 0:2]) * back_dir).sum(dim=-1)
                clear = depth >= HANDLE_TO_FACE + 0.012 + object_half_size + 0.004
                go = (s0 == from_state) & waited & clear & ramp_done
            elif from_state in TIME_ONLY:
                go = (s0 == from_state) & waited
            else:
                go = (s0 == from_state) & waited & near
            if from_state == Sm.RETREAT_FROM_HANDLE:
                go &= ~retry
            self.state[go] = to_state
            self.wait[go] = 0.0

        self.wait += self.dt
        return des
