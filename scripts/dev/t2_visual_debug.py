"""One visuomotor drawer_stow episode with frame dumps + transition diagnostics."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Cog-DrawerStow-L0-IK-Rel-Visuomotor-v0")
parser.add_argument("--outdir", type=str, default="ops/t2_debug")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import cog.tasks.drawer_stow  # noqa: F401
from cog.tasks.cup_place.state_machine import convert_abs_to_rel_actions
from cog.tasks.drawer_stow.assets import BOX_VARIANTS
from cog.tasks.drawer_stow.levels import SUB_LEVELS
from cog.tasks.drawer_stow.state_machine import DrawerStowSm, Sm

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

os.makedirs(args_cli.outdir, exist_ok=True)
sub_key = args_cli.task.split("-")[2]
variant = BOX_VARIANTS[SUB_LEVELS[sub_key].box_variant]
env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
obs, _ = env.reset(seed=0)
drw = env.scene["cabinet"].actuators["drawers"]
print(f"[dbg] drawer actuator stiffness={drw.stiffness.tolist()} damping={drw.damping.tolist()}", flush=True)

sm = DrawerStowSm(dt=env_cfg.sim.dt * env_cfg.decimation, num_envs=1, device=env.device)
actions = torch.zeros(1, 7, device=env.device)
cab_joint_ids, _ = env.scene["cabinet"].find_joints(["drawer_top_joint"])
cab_body_ids, _ = env.scene["cabinet"].find_bodies(["drawer_top"])
STATE_NAMES = {v: k for k, v in vars(Sm).items() if isinstance(v, int)}
prev_state = -1

for step in range(820):
    obs, _, terminated, truncated, _ = env.step(actions)
    if (terminated | truncated).any():
        print(f"[dbg] episode ended at step {step}", flush=True)
        break
    origins = env.scene.env_origins
    ee = env.scene["ee_frame"]
    tcp_pos = ee.data.target_pos_w[..., 0, :] - origins
    tcp_quat = ee.data.target_quat_w[..., 0, :]
    ee_pose = torch.cat([tcp_pos, tcp_quat], dim=-1)
    cab_frame = env.scene["cabinet_frame"]
    handle_pose = torch.cat([cab_frame.data.target_pos_w[..., 0, :] - origins,
                             cab_frame.data.target_quat_w[..., 0, :]], dim=-1)
    cabinet = env.scene["cabinet"]
    drawer_joint = cabinet.data.joint_pos[:, cab_joint_ids[0]]
    drawer_body_pose = torch.cat([cabinet.data.body_pos_w[:, cab_body_ids[0]] - origins,
                                  cabinet.data.body_quat_w[:, cab_body_ids[0]]], dim=-1)
    o = env.scene["object"]
    object_pose = torch.cat([o.data.root_pos_w - origins, o.data.root_quat_w], dim=-1)
    abs_target = sm.compute(ee_pose, handle_pose, drawer_joint, object_pose, drawer_body_pose,
                            grasp_z_offset=variant.grasp_z_offset, object_half_size=variant.half_size)
    actions = convert_abs_to_rel_actions(abs_target, tcp_pos, tcp_quat)

    st = int(sm.state[0])
    if st != prev_state:
        fingers = env.scene["robot"].data.joint_pos[0, -2:].tolist()
        print(f"[dbg] step {step}: -> {STATE_NAMES[st]} | tcp={tcp_pos[0].tolist()} "
              f"des={abs_target[0, :3].tolist()} desq={abs_target[0, 3:7].tolist()} "
              f"handle={handle_pose[0].tolist()} drawer={float(drawer_joint[0]):.3f} "
              f"fingers={[round(f, 4) for f in fingers]}", flush=True)
        prev_state = st
    if step % 50 == 0:
        dist = float((tcp_pos[0] - abs_target[0, :3]).norm())
        jp = env.scene["robot"].data.joint_pos[0, :7]
        print(f"[dbg] step {step}: {STATE_NAMES[st]} dist={dist:.4f} tcp={[round(x,3) for x in tcp_pos[0].tolist()]} "
              f"des={[round(x,3) for x in abs_target[0,:3].tolist()]} drawer={float(drawer_joint[0]):.3f} "
              f"joints={[round(float(x),2) for x in jp]}", flush=True)
    if step % 25 == 0 or st != prev_state:
        img = obs["policy"]["table_cam"][0].cpu().numpy().astype(np.uint8)
        Image.fromarray(img).save(f"{args_cli.outdir}/f{step:04d}_{STATE_NAMES[st]}.png")

print(f"[dbg] final drawer={float(drawer_joint[0]):.3f} state={STATE_NAMES[int(sm.state[0])]}", flush=True)
print("[dbg] DONE", flush=True)
env.close()
simulation_app.close()
