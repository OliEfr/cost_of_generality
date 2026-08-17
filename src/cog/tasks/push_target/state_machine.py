"""Scripted push expert for push_target.

Design notes carried over from the Task-1/Task-2 experts (docs/journal.md):
  * Emit ABSOLUTE TCP targets; the driver converts with
    cup_place.state_machine.convert_abs_to_rel_actions so recorded actions match the
    Mimic env exactly (D7).
  * Evaluate transitions on a FROZEN copy of the state vector. Aliasing the live tensor
    let one compute() cascade through every gate at once (T2 root cause 1).
  * RAMP every long translation via a progress buffer. Commanding a far target in one
    step makes the DLS IK unwrap the elbow into a damped stall (T2 runs 8-13).
  * Gate descent on a MEASURED physical quantity, not on the commanded target. The
    2026-08-17 probe stalled 8 cm above the object with an open-loop descent and pushed
    air; the TCP z reported by ee_frame IS the contact height, so it can be gated directly.

Push-specific:
  * The blade is the two closed fingers. Its broad face must be normal to the push
    direction, so the whole approach is rotated about world z by (bearing - pi/2)
    relative to the reset orientation -- pi/2 because +y is the bearing the probe
    verified, so at that bearing the rotation is the identity.
  * The stand-off uses MAX_PUCK_RADIUS, not the current variant's radius, so a stroke
    recorded on one L3 variant never starts inside a larger one when Mimic replays it.
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_from_angle_axis, quat_mul

from .assets import MAX_PUCK_RADIUS

# --- states ---
REST = 0
CLOSE_BLADE = 1
APPROACH_XY = 2
DESCEND = 3
PUSH = 4
RETREAT = 5
DONE = 6

# --- tuning ---
SETTLE_TICKS = 10            # let the reset transient die before moving
CLOSE_TICKS = 25             # fingers 0.04 -> 0.0 at the action rate
TRAVERSE_Z = 0.10            # safe height for the XY approach (clears the 5.5 cm puck by 4.5 cm)
STANDOFF = MAX_PUCK_RADIUS + 0.017   # 0.075 m behind the puck centre
APPROACH_RATE = 0.030        # m per control tick, ramped
DESCEND_RATE = 0.020
PUSH_RATE = 0.015            # probe-verified: moved the object smoothly, no stall
# The commanded advance is capped to this much AHEAD of the arm's MEASURED progress.
# Without it the ramp advances 0.015/tick unconditionally while the arm -- IK-Rel scale
# 0.5, one waypoint per step, loaded against contact friction -- follows at roughly a
# third of that. Traced 2026-08-17: commanded 0.180 m vs 0.045 m actual at step 195,
# ramp pinned at its cap by step 210, so `spent` fired and the machine retreated with
# the puck still 9 cm short. It only reached the target because the runaway command kept
# dragging the blade forward during RETREAT.
# How far past the contact point the blade is commanded. This sets the steady-state
# penetration into the puck, hence the push force, hence the speed. At 0.020 the stroke
# crawled at 0.7 mm/tick (traced 2026-08-17) -- 20x slower than PUSH_RATE, because the
# aim sat only ~1.5 cm ahead of the blade and the commanded step is limited by that gap,
# not by the rate. Episodes then timed out mid-push and mid-descent.
PUSH_LEAD = 0.050
# Blade half-thickness: the two closed fingers form a plate ~2 cm across, so the contact
# point sits this far from the puck surface when the TCP is on the push line.
BLADE_HALF = 0.012
# Lateral chase limit. The stroke is a PURSUIT controller (below), and without a bound a
# puck that squirts far sideways would drag the blade into an ever-widening arc.
CHASE_LIMIT = 0.06
XY_TOL = 0.006
# Descent gate. Measured steady-state IK tracking error on the descent is ~6 mm, so a
# 4 mm gate was a coin flip: the arm parks at contact_z + 0.006 and |err| never drops
# below tolerance, so DESCEND hangs until the episode times out with the puck untouched.
# This is the same class of bug that pinned T1's expert at 47% (a 13 mm tracking error
# against a 12 mm threshold). Three defences, all needed:
#   1. command BELOW the contact height, so the error signal keeps driving down;
#   2. gate with tolerance > the tracking error;
#   3. a hard tick budget that advances the state anyway -- a blade a few mm high still
#      pushes a 40-55 mm puck, so proceeding beats hanging.
Z_TOL = 0.008
DESCEND_PRESS = 0.010
DESCEND_TIMEOUT = 140
PUSH_TIMEOUT = 340
PUSH_CAP = 1.6               # stroke cap as a multiple of the nominal push distance
# Stop distance for the stroke. Deliberately far tighter than the 5 cm success gate:
# whatever residual error the expert leaves is the error every generated demo inherits,
# and Mimic then adds its own slip on top.
PUSH_STOP_ERR = 0.018
# Proportional braking. With a FIXED lead the controller had no notion of "nearly there":
# on L2 the error bottomed out at 5.3 cm and was then shoved to 18 cm PAST the target
# (traced 2026-08-17), so the stop test never fired and the stroke ran to its cap. The
# commanded penetration is now proportional to the remaining error, so the push force
# fades out as the puck arrives and the blade ends up merely touching it.
PUSH_BRAKE_GAIN = 0.25
# Floor on how far the blade presses PAST the modelled contact point. Penetration is
# (lead_eff - BLADE_HALF), so a purely proportional law fades the push force to zero at
# ~3 cm of error -- before the stop test at 1.8 cm can fire -- and it fades at a
# radius-dependent point because a 2 cm blade meets a small puck's sharp curvature
# differently than a large puck's flat one. That inverted the failure mode: large pucks
# failed with a fixed lead, small pucks failed with a proportional one. A constant floor
# keeps a light push alive all the way to the stop test for every radius.
PUSH_MIN_PRESS = 0.006
# Cap on how far the blade may be commanded PAST the puck surface, scaled by the puck.
# The penetration a lead implies is (lead_eff - BLADE_HALF), up to 3.8 cm at full lead.
# For a 3.2 cm-radius puck that is past its CENTRE: the blade is told to occupy the space
# the puck is in, and knocks it away instead of pushing it (failures clustered at 11-16 cm
# of final error, traced 2026-08-17). You cannot press 3.8 cm into a 3.2 cm object, so the
# cap scales with radius -- which also makes the L3 geometry axis behave uniformly.
PUSH_MAX_PEN = 0.030
PUSH_PEN_RADIUS_FRAC = 0.6
RETREAT_BACK = 0.06
RETREAT_UP = 0.08
RETREAT_TICKS = 45


class PushSm:
    """Batched push state machine. compute() returns (N,8): abs TCP pose + gripper."""

    def __init__(self, dt: float, num_envs: int, device, push_distance: float = 0.20):
        self.dt = dt
        self.num_envs = num_envs
        self.device = device
        self.push_distance = push_distance

        self.state = torch.full((num_envs,), REST, dtype=torch.long, device=device)
        self.ticks = torch.zeros(num_envs, dtype=torch.long, device=device)
        # captured at the first compute() after a reset: the reset orientation is the
        # one the probe validated, so every desired quat is derived from it
        self.base_quat = torch.zeros(num_envs, 4, device=device)
        self.have_base = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # latched geometry, frozen at APPROACH so a puck nudged mid-stroke cannot make
        # the target chase it (T2 run 9: a live-anchored target chased the drawer away)
        self.dir_xy = torch.zeros(num_envs, 2, device=device)
        self.standoff_xy = torch.zeros(num_envs, 2, device=device)
        self.contact_z = torch.zeros(num_envs, device=device)
        self.pushed = torch.zeros(num_envs, device=device)      # ramp progress (m)
        self.approach = torch.zeros(num_envs, device=device)    # ramp progress (m)
        self.approach_from = torch.zeros(num_envs, 3, device=device)

    def reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == 0:
            return
        self.state[env_ids] = REST
        self.ticks[env_ids] = 0
        self.have_base[env_ids] = False
        self.pushed[env_ids] = 0.0
        self.approach[env_ids] = 0.0

    def compute(
        self,
        ee_pose: torch.Tensor,      # (N,7) TCP pose, env-relative, wxyz
        puck_pose: torch.Tensor,    # (N,7) puck root pose, env-relative
        target_pos: torch.Tensor,   # (N,3) target disk centre, env-relative
        contact_z: float,           # variant contact height (assets.PuckVariant.contact_z)
        puck_radius: float,          # variant radius: sets where the blade meets the puck
        success_radius: float,
    ) -> torch.Tensor:
        N, dev = self.num_envs, self.device
        self.radius = puck_radius
        s = self.state.clone()          # FROZEN: at most one transition per call
        self.ticks += 1

        tcp = ee_pose[:, 0:3]
        tcp_quat = ee_pose[:, 3:7]
        puck_xy = puck_pose[:, 0:2]

        newly = ~self.have_base
        if newly.any():
            self.base_quat[newly] = tcp_quat[newly]
            self.have_base[newly] = True

        delta = target_pos[:, 0:2] - puck_xy
        dist = torch.linalg.vector_norm(delta, dim=1, keepdim=True).clamp_min(1e-6)
        dir_now = delta / dist
        bearing = torch.atan2(dir_now[:, 1], dir_now[:, 0])

        # desired orientation: reset quat rotated about world z by (bearing - pi/2)
        zaxis = torch.zeros(N, 3, device=dev)
        zaxis[:, 2] = 1.0
        yaw_delta = bearing - torch.pi / 2
        des_quat = quat_mul(quat_from_angle_axis(yaw_delta, zaxis), self.base_quat)

        des = torch.zeros(N, 8, device=dev)
        des[:, 3:7] = des_quat
        des[:, 7] = -1.0                      # closed by default; REST opens below
        des[:, 0:3] = tcp                     # default: hold position

        # ---------------- REST: settle, fingers open ----------------
        m = s == REST
        if m.any():
            des[m, 7] = 1.0
            done = m & (self.ticks >= SETTLE_TICKS)
            self.state[done] = CLOSE_BLADE
            self.ticks[done] = 0

        # ---------------- CLOSE_BLADE: fingers to a blade ----------------
        m = s == CLOSE_BLADE
        if m.any():
            des[m, 7] = -1.0
            done = m & (self.ticks >= CLOSE_TICKS)
            if done.any():
                # latch the push geometry now, before anything can be nudged
                self.dir_xy[done] = dir_now[done]
                self.standoff_xy[done] = puck_xy[done] - dir_now[done] * STANDOFF
                self.contact_z[done] = contact_z
                self.approach_from[done] = tcp[done]
                self.approach[done] = 0.0
            self.state[done] = APPROACH_XY
            self.ticks[done] = 0

        # ---------------- APPROACH_XY: ramped traverse above the stand-off ----------------
        m = s == APPROACH_XY
        if m.any():
            goal = torch.cat([self.standoff_xy, torch.full((N, 1), TRAVERSE_Z, device=dev)], dim=1)
            leg = goal - self.approach_from
            leg_len = torch.linalg.vector_norm(leg, dim=1).clamp_min(1e-6)
            self.approach[m] = torch.minimum(self.approach[m] + APPROACH_RATE, leg_len[m])
            frac = (self.approach / leg_len).clamp(0.0, 1.0).unsqueeze(-1)
            des[m, 0:3] = (self.approach_from + leg * frac)[m]
            err = torch.linalg.vector_norm(goal[:, 0:2] - tcp[:, 0:2], dim=1)
            done = m & (self.approach >= leg_len - 1e-6) & (err < XY_TOL)
            self.state[done] = DESCEND
            self.ticks[done] = 0

        # ---------------- DESCEND: to contact height, gated on MEASURED tcp z ----------------
        m = s == DESCEND
        if m.any():
            hold_xy = self.standoff_xy
            press_to = self.contact_z - DESCEND_PRESS
            step_z = (press_to - tcp[:, 2]).clamp(-DESCEND_RATE, DESCEND_RATE)
            des[m, 0] = hold_xy[m, 0]
            des[m, 1] = hold_xy[m, 1]
            des[m, 2] = (tcp[:, 2] + step_z)[m]
            low_enough = tcp[:, 2] <= self.contact_z + Z_TOL
            done = m & (low_enough | (self.ticks >= DESCEND_TIMEOUT))
            if done.any():
                self.pushed[done] = 0.0
            self.state[done] = PUSH
            self.ticks[done] = 0

        # ---------------- PUSH: closed-loop pursuit of the puck's far side ----------------
        # An open-loop straight stroke along the latched direction reached only 22% on the
        # largest puck (gate run 2026-08-17): a ~2 cm blade pushing a cylinder lets the
        # puck squirt sideways, and a bigger radius gives that slip more leverage. So the
        # stroke continuously re-aims: each tick, command the blade to a point just inside
        # the puck's near surface along the CURRENT puck->target line. The puck only moves
        # when pushed and the command follows it, so the controller is self-sustaining and
        # self-correcting. The source demo is then a gently curved path in the push frame,
        # which Mimic can still replay rigidly.
        m = s == PUSH
        if m.any():
            cap = self.push_distance * PUSH_CAP + STANDOFF
            proj = ((tcp[:, 0:2] - self.standoff_xy) * self.dir_xy).sum(dim=1).clamp_min(0.0)

            err_mag = torch.linalg.vector_norm(target_pos[:, 0:2] - puck_xy, dim=1, keepdim=True)
            max_pen = min(PUSH_MAX_PEN, PUSH_PEN_RADIUS_FRAC * self.radius)
            lead_eff = (BLADE_HALF + PUSH_MIN_PRESS + PUSH_BRAKE_GAIN * err_mag).clamp(
                max=min(PUSH_LEAD, BLADE_HALF + max_pen)
            )
            contact_pt = puck_xy - dir_now * (self.radius + BLADE_HALF)
            aim = contact_pt + dir_now * lead_eff
            # bound how far the blade may be dragged off the latched line
            off_line = aim - (self.standoff_xy + self.dir_xy * proj.unsqueeze(-1))
            lateral = off_line - (off_line * self.dir_xy).sum(dim=1, keepdim=True) * self.dir_xy
            lat_norm = torch.linalg.vector_norm(lateral, dim=1, keepdim=True).clamp_min(1e-6)
            clipped = lateral * (lat_norm.clamp(max=CHASE_LIMIT) / lat_norm)
            aim = aim - lateral + clipped

            # rate-limit the command so it never sprints away from the arm
            step_xy = (aim - tcp[:, 0:2]).clamp(-PUSH_RATE, PUSH_RATE)
            des[m, 0:2] = (tcp[:, 0:2] + step_xy)[m]
            des[m, 2] = self.contact_z[m]

            on_target = torch.linalg.vector_norm(target_pos[:, 0:2] - puck_xy, dim=1) <= PUSH_STOP_ERR
            spent = proj >= cap - 1e-6
            done = m & (on_target | spent | (self.ticks >= PUSH_TIMEOUT))
            if done.any():
                self.pushed[done] = proj[done]
            self.state[done] = RETREAT
            self.ticks[done] = 0

        # ---------------- RETREAT: back off and lift, then hold ----------------
        m = s == RETREAT
        if m.any():
            frac = (self.ticks.float() / RETREAT_TICKS).clamp(0.0, 1.0).unsqueeze(-1)
            back = -self.dir_xy * RETREAT_BACK * frac
            stroke_end = self.standoff_xy + self.dir_xy * self.pushed.unsqueeze(-1)
            des[m, 0:2] = (stroke_end + back)[m]
            des[m, 2] = (self.contact_z + RETREAT_UP * frac.squeeze(-1))[m]
            done = m & (self.ticks >= RETREAT_TICKS)
            self.state[done] = DONE
            self.ticks[done] = 0

        # ---------------- DONE: hold clear ----------------
        m = s == DONE
        if m.any():
            des[m, 0:3] = tcp[m]

        return des
