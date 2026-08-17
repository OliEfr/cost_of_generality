"""Freeze one (sub-)level's eval-set snapshot: for each protocol batch seed,
reset the STATE env with 20 envs and record the task's randomized initial poses.
The frozen benchmark is (env cfg + seed protocol); this JSON makes drift detectable.

--task_kind selects which scene entities get snapshotted: cup_place records the cup
and goal marker, drawer_stow records the box plus the cabinet root and its joints
(the cabinet is an Articulation, so its drawer state has to be captured too)."""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--batches", type=int, default=10)
parser.add_argument("--base_seed", type=int, default=5000)
parser.add_argument("--task_kind", choices=("cup_place", "drawer_stow"), default="cup_place")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

import cog.tasks.cup_place  # noqa: F401
import cog.tasks.drawer_stow  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=20)
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

def snapshot(kind):
    def arr(x):
        return x.cpu().numpy().round(6).tolist()

    if kind == "cup_place":
        return {
            "cup_pos": arr(env.scene["cup"].data.root_pos_w),
            "cup_quat": arr(env.scene["cup"].data.root_quat_w),
            "goal_pos": arr(env.scene["goal_marker"].data.root_pos_w),
        }
    return {
        "object_pos": arr(env.scene["object"].data.root_pos_w),
        "object_quat": arr(env.scene["object"].data.root_quat_w),
        "cabinet_pos": arr(env.scene["cabinet"].data.root_pos_w),
        "cabinet_quat": arr(env.scene["cabinet"].data.root_quat_w),
        "cabinet_joint_pos": arr(env.scene["cabinet"].data.joint_pos),
    }


batches = []
for b in range(args_cli.batches):
    env.reset(seed=args_cli.base_seed + b)
    batches.append({"seed": args_cli.base_seed + b, **snapshot(args_cli.task_kind)})
with open(args_cli.out, "w") as f:
    json.dump({"task": args_cli.task, "task_kind": args_cli.task_kind,
               "num_envs": 20, "batches": batches}, f, indent=1)
print(f"EVALSET_OK {args_cli.task} batches={len(batches)}", flush=True)
env.close()
simulation_app.close()
