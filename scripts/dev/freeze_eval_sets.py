"""Freeze one (sub-)level's eval-set snapshot: for each protocol batch seed,
reset the STATE env with 20 envs and record cup/goal initial poses. The frozen
benchmark is (env cfg + seed protocol); this JSON makes drift detectable."""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--batches", type=int, default=10)
parser.add_argument("--base_seed", type=int, default=5000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import cog.tasks.cup_place  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=20)
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

batches = []
for b in range(args_cli.batches):
    env.reset(seed=args_cli.base_seed + b)
    batches.append({
        "seed": args_cli.base_seed + b,
        "cup_pos": env.scene["cup"].data.root_pos_w.cpu().numpy().round(6).tolist(),
        "cup_quat": env.scene["cup"].data.root_quat_w.cpu().numpy().round(6).tolist(),
        "goal_pos": env.scene["goal_marker"].data.root_pos_w.cpu().numpy().round(6).tolist(),
    })
with open(args_cli.out, "w") as f:
    json.dump({"task": args_cli.task, "num_envs": 20, "batches": batches}, f, indent=1)
print(f"EVALSET_OK {args_cli.task} batches={len(batches)}", flush=True)
env.close()
simulation_app.close()
