"""Task-3 expert gate: run PushSm on a state env and report success rate.

  python scripts/dev/t3_smoke.py --level L0 --num_envs 8 --headless
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--level", type=str, default="L0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--episodes", type=int, default=8)
parser.add_argument("--trace", action="store_true")
parser.add_argument("--debug_env", type=int, default=-1, help="per-tick trace for one env")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import cog.tasks.push_target  # noqa: F401
from cog.tasks.cup_place.state_machine import convert_abs_to_rel_actions
from cog.tasks.push_target.assets import PUCK_VARIANTS, SUCCESS_RADIUS
from cog.tasks.push_target.levels import PUSH_DISTANCE, SUB_LEVELS
from cog.tasks.push_target.state_machine import PushSm
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

task = f"Cog-PushTarget-{args_cli.level}-IK-Rel-v0"
env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
env = gym.make(task, cfg=env_cfg).unwrapped
variant = PUCK_VARIANTS[SUB_LEVELS[args_cli.level].puck_variant]
print(f"[smoke] {task} puck={variant.name} r={variant.radius} h={variant.height} "
      f"contact_z={variant.contact_z:.4f}", flush=True)

sm = PushSm(dt=env.step_dt, num_envs=env.num_envs, device=env.device,
            push_distance=PUSH_DISTANCE)
env.reset()
sm.reset_idx(torch.arange(env.num_envs, device=env.device))

successes = 0
finished = 0
bearings_ok, bearings_bad = [], []
max_steps = int(env_cfg.episode_length_s / env.step_dt) + 5
step = 0
while finished < args_cli.episodes * env.num_envs and step < max_steps * args_cli.episodes:
    step += 1
    org = env.scene.env_origins
    ee = env.scene["ee_frame"]
    ee_pose = torch.cat([ee.data.target_pos_w[:, 0] - org, ee.data.target_quat_w[:, 0]], dim=-1)
    puck = env.scene["object"]
    puck_pose = torch.cat([puck.data.root_pos_w - org, puck.data.root_quat_w], dim=-1)
    tgt = env.scene["target_marker"].data.root_pos_w - org

    prev_err = torch.linalg.vector_norm((puck_pose[:, 0:2] - tgt[:, 0:2]), dim=1)
    prev_state = sm.state.clone()
    prev_bearing = torch.atan2(sm.dir_xy[:, 1], sm.dir_xy[:, 0])
    # recover the puck's start position from the latched stand-off, then measure how far
    # it has travelled ALONG the push direction: > PUSH_DISTANCE is overshoot, < is short
    from cog.tasks.push_target.state_machine import STANDOFF
    puck_start = sm.standoff_xy + sm.dir_xy * STANDOFF
    prev_travel = ((puck_pose[:, 0:2] - puck_start) * sm.dir_xy).sum(dim=1)
    prev_puck = puck_pose[:, 0:2].clone()
    abs_target = sm.compute(ee_pose, puck_pose, tgt, variant.contact_z, variant.radius, SUCCESS_RADIUS)
    action = convert_abs_to_rel_actions(abs_target, ee_pose[:, 0:3], ee_pose[:, 3:7])
    if args_cli.debug_env >= 0:
        d = args_cli.debug_env
        if step % 15 == 0 or int(sm.state[d]) in (3, 4):
            if step % 15 == 0:
                print(f"[dbg] s{step:4d} st={int(sm.state[d])} "
                      f"des=({abs_target[d,0]:+.3f},{abs_target[d,1]:+.3f},{abs_target[d,2]:+.3f}) "
                      f"tcp=({ee_pose[d,0]:+.3f},{ee_pose[d,1]:+.3f},{ee_pose[d,2]:+.3f}) "
                      f"puck=({puck_pose[d,0]:+.3f},{puck_pose[d,1]:+.3f}) "
                      f"tgt=({tgt[d,0]:+.3f},{tgt[d,1]:+.3f}) err={prev_err[d]*100:5.2f} "
                      f"pushed={float(sm.pushed[d]):.3f}", flush=True)
    obs, rew, terminated, truncated, info = env.step(action)

    dones = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1)
    if len(dones):
        succ = env.termination_manager.get_term("success")[dones]
        successes += int(succ.sum().item())
        finished += len(dones)
        for i, d in enumerate(dones.tolist()):
            b = float(torch.rad2deg(prev_bearing[d]))
            (bearings_ok if bool(succ[i]) else bearings_bad).append(b)
        if args_cli.trace:
            for i, d in enumerate(dones.tolist()):
                print(f"[smoke] env {d} done: success={bool(succ[i])} "
                      f"final_err={prev_err[d]*100:.1f} cm state_at_done={int(prev_state[d])} "
                      f"timeout={bool(truncated[d])} "
                      f"bearing={float(torch.rad2deg(prev_bearing[d])):.0f}deg "
                      f"travel={float(prev_travel[d])*100:.1f}cm/20.0", flush=True)
        sm.reset_idx(dones)

    if step % 100 == 0:
        states = torch.bincount(sm.state, minlength=7).tolist()
        print(f"[smoke] step {step} states {states} done {finished} ok {successes} "
              f"err_med {prev_err.median()*100:.1f} cm", flush=True)

print(f"[smoke] RESULT {successes}/{finished} expert successes", flush=True)
# SR BINNED by |bearing-90|. Reporting the MEAN bearing of failures is uninformative
# because the sampled range is symmetric about 90 deg: any bearing-driven loss still
# averages ~90. Binned success rate is the measurement that actually separates them.
if bearings_ok or bearings_bad:
    bins = [(0, 10), (10, 25), (25, 45)]
    print("[smoke] SR by |bearing-90deg|:", flush=True)
    for lo, hi in bins:
        ok = sum(1 for b in bearings_ok if lo <= abs(b - 90) < hi)
        bad = sum(1 for b in bearings_bad if lo <= abs(b - 90) < hi)
        tot = ok + bad
        rate = f"{100*ok/tot:.0f}%" if tot else "n/a"
        print(f"[smoke]   {lo:2d}-{hi:2d} deg: {ok:3d}/{tot:3d} = {rate}", flush=True)
print("T3_SMOKE_DONE", flush=True)
env.close()
simulation_app.close()
