"""Empirical Sektion-cabinet geometry: body world poses, drawer joint limits,
top-surface height, drawer cavity bounds (closed + open) — drives drawer_stow
scene constants."""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK = "Isaac-Open-Drawer-Franka-IK-Rel-v0"
env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
env = gym.make(TASK, cfg=env_cfg).unwrapped
env.reset(seed=0)

cab = env.scene["cabinet"]
report = {
    "body_names": cab.body_names,
    "joint_names": cab.joint_names,
    "joint_limits": cab.data.joint_pos_limits[0].cpu().tolist(),
    "default_root": cab.data.default_root_state[0].cpu().tolist(),
}

def poses(tag):
    d = {}
    for i, name in enumerate(cab.body_names):
        d[name] = cab.data.body_pos_w[0, i].cpu().round(decimals=4).tolist()
    d["handle_frame_target"] = env.scene["cabinet_frame"].data.target_pos_w[0, 0].cpu().round(decimals=4).tolist()
    report[tag] = d

poses("closed_body_pos_w")

# force drawer_top open to 0.3 and settle
idx = cab.joint_names.index("drawer_top_joint")
jp = cab.data.joint_pos.clone()
jp[0, idx] = 0.3
cab.write_joint_state_to_sim(jp, torch.zeros_like(jp))
for _ in range(30):
    env.sim.step()
cab.update(env.physics_dt)
env.scene["cabinet_frame"].update(env.physics_dt)
poses("open03_body_pos_w")

# USD bbox of the cabinet prim
from pxr import Usd, UsdGeom
stage = env.sim.stage
prim = stage.GetPrimAtPath("/World/envs/env_0/Cabinet")
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
report["cabinet_world_bbox"] = {"min": list(box.GetMin()), "max": list(box.GetMax())}
for sub in ("sektion", "drawer_top", "drawer_handle_top"):
    p = stage.GetPrimAtPath(f"/World/envs/env_0/Cabinet/{sub}")
    if p.IsValid():
        b = cache.ComputeWorldBound(p).ComputeAlignedRange()
        report[f"bbox_{sub}"] = {"min": list(b.GetMin()), "max": list(b.GetMax())}

with open(args_cli.out, "w") as f:
    json.dump(report, f, indent=1)
print("INSPECT_OK", flush=True)
env.close()
simulation_app.close()
