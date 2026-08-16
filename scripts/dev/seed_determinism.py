"""VERIFY (c): env.reset(seed=k) must reproduce identical scene state,
within a process and across processes. Writes poses to --out for diffing."""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Cog-CupPlace-L2-IK-Rel-v0")
parser.add_argument("--out", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import cog.tasks.cup_place  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=4)
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped


def snap():
    return {
        "cup": env.scene["cup"].data.root_pos_w.cpu().numpy().round(6).tolist(),
        "goal": env.scene["goal_marker"].data.root_pos_w.cpu().numpy().round(6).tolist(),
        "joints": env.scene["robot"].data.joint_pos.cpu().numpy().round(6).tolist(),
    }


env.reset(seed=5000)
a = snap()
env.reset(seed=7777)
env.reset(seed=5000)
b = snap()
within = a == b
print(f"[determinism] within-process seed-5000 repeat identical: {within}", flush=True)
with open(args_cli.out, "w") as f:
    json.dump({"within_process_identical": within, "snapshot": a}, f, indent=1)
env.close()
simulation_app.close()
