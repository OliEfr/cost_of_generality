"""P2 smoke: register + instantiate + step cup_place envs.

Usage: ./isaaclab.sh -p scripts/dev/smoke_env.py --headless [--visuomotor --enable_cameras]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--visuomotor", action="store_true")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=25)
parser.add_argument("--task", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import json
import os
import torch

import cog.tasks.cup_place  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

ids = sorted(i for i in gym.registry if i.startswith("Cog-CupPlace"))
print(f"[smoke] {len(ids)} Cog-CupPlace IDs registered; e.g. {ids[:3]} ... {ids[-1]}")

task = args_cli.task or (
    "Cog-CupPlace-L0-IK-Rel-Visuomotor-v0" if args_cli.visuomotor else "Cog-CupPlace-L0-IK-Rel-v0"
)
env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
env = gym.make(task, cfg=env_cfg).unwrapped
obs, _ = env.reset()
print("[smoke] policy obs terms:")
for k, v in obs["policy"].items():
    print(f"    {k}: {tuple(v.shape)} {v.dtype}")
print("[smoke] subtask_terms:", {k: tuple(v.shape) for k, v in obs["subtask_terms"].items()})

actions = torch.zeros(env.num_envs, 7, device=env.device)
succ_any = False
for i in range(args_cli.steps):
    obs, _, term, trunc, _ = env.step(actions)
    if i == 5:
        cup = env.scene["cup"].data.root_pos_w - env.scene.env_origins
        goal = env.scene["goal_marker"].data.root_pos_w - env.scene.env_origins
        print(f"[smoke] cup pos env0: {cup[0].tolist()}")
        print(f"[smoke] goal pos env0: {goal[0].tolist()}")
result = {
    "task": task,
    "n_ids": len(ids),
    "policy_obs": {k: [list(v.shape), str(v.dtype)] for k, v in obs["policy"].items()},
    "subtask_terms": {k: list(v.shape) for k, v in obs["subtask_terms"].items()},
    "ok": True,
}
if args_cli.visuomotor:
    img = obs["policy"]["table_cam"]
    result["table_cam"] = [list(img.shape), str(img.dtype), int(img.min().item()), int(img.max().item())]
    print(f"[smoke] table_cam: {tuple(img.shape)} {img.dtype}", flush=True)
tag = "visuo" if args_cli.visuomotor else "state"
os.makedirs("/home/admin_07/cost_of_generality/ops", exist_ok=True)
with open(f"/home/admin_07/cost_of_generality/ops/smoke_result_{tag}.json", "w") as f:
    json.dump(result, f, indent=1)
print("[smoke] SMOKE_ENV_OK", flush=True)
env.close()
simulation_app.close()
