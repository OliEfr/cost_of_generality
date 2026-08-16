"""Render reset frames across levels for visual QA (camera framing, marker visibility)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--level", type=str, default="L0")
parser.add_argument("--resets", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import cog.tasks.cup_place  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

OUT = "/home/admin_07/cost_of_generality/ops/qa"
os.makedirs(OUT, exist_ok=True)

key = args_cli.level
task = f"Cog-CupPlace-{key}-IK-Rel-Visuomotor-v0"
env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=1)
env = gym.make(task, cfg=env_cfg).unwrapped
level_rows = []
for r in range(args_cli.resets):
    obs, _ = env.reset(seed=100 + r)
    for _ in range(3):  # settle + render
        obs, *_ = env.step(torch.zeros(1, 7, device=env.device))
    t = obs["policy"]["table_cam"][0].cpu().numpy()
    w = obs["policy"]["wrist_cam"][0].cpu().numpy()
    level_rows.append(np.concatenate([t, w], axis=0))  # stack table over wrist
row = np.concatenate(level_rows, axis=1)
Image.fromarray(row).save(f"{OUT}/frames_{key}.png")
print(f"[qa] {key} saved", flush=True)
