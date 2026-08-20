"""Batched rollout evaluation of a LeRobot diffusion policy in any of the study's task envs.

Frozen protocol (configs/eval_sets/protocol.json): per level, `batches` x
`num_envs` episodes; batch b resets the vectorized env with seed base_seed+b,
which deterministically reproduces the same initial conditions for every
checkpoint/cell (global-RNG event sampling; verified determinism at G4).
Success = the env's `success` termination fired at least once (latched);
failure = timeout first. DDIM num_inference_steps set explicitly at load
(post-load gotcha: DiffusionModel copies config at init).

Run: ./isaaclab.sh -p src/cog/eval/rollout_eval.py --task Cog-CupPlace-L0-IK-Rel-Visuomotor-v0 \
        --checkpoint <...>/checkpoints/080000/pretrained_model --out results/eval_L0_n100_080000.json \
        --headless --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--protocol", default="/home/admin_07/cost_of_generality/configs/eval_sets/protocol.json")
parser.add_argument("--num_inference_steps", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=600)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import os

import gymnasium as gym
import torch

import importlib

# Gym ids are registered as an import SIDE EFFECT of each task package, so the package matching
# --task has to be imported before gym.make. This was hardcoded to cup_place, which meant no T2/T3
# eval could ever run: the failure surfaces as
#   NameNotFound: Environment `Cog-PushTarget-L2-IK-Rel-Visuomotor` doesn't exist.
#                 Did you mean: `Cog-CupPlace-L2-IK-Rel-Visuomotor`?
# which reads like a typo in the task name rather than a missing import, and Kit still exits 0
# (D6). Found the first time a T2/T3 cell was evaluated -- the T1 path had always worked, so the
# earlier T2/T3 readiness audit (max_steps, result filenames, PYTHONPATH) never exercised this.
_TASK_MODULES = {
    "Cog-CupPlace": "cog.tasks.cup_place",
    "Cog-DrawerStow": "cog.tasks.drawer_stow",
    "Cog-PushTarget": "cog.tasks.push_target",
}
_prefix = next((p for p in _TASK_MODULES if args_cli.task.startswith(p)), None)
if _prefix is None:
    # Unknown prefix: register everything rather than guess, so a new task family still evaluates.
    for _m in _TASK_MODULES.values():
        importlib.import_module(_m)
else:
    importlib.import_module(_TASK_MODULES[_prefix])

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


def obs_to_batch(obs, device):
    pol = obs["policy"]
    state = torch.cat([pol["eef_pos"], pol["eef_quat"], pol["gripper_pos"]], dim=-1).float()
    batch = {"observation.state": state.to(device)}
    for cam in ("table_cam", "wrist_cam"):
        img = pol[cam]  # (B,H,W,C) uint8
        batch[f"observation.images.{cam}"] = (
            img.permute(0, 3, 1, 2).float() / 255.0
        ).to(device)
    return batch


def main():
    proto = json.load(open(args_cli.protocol))
    num_envs, batches, base_seed = proto["num_envs"], proto["batches"], proto["base_seed"]

    policy = DiffusionPolicy.from_pretrained(
        args_cli.checkpoint,
        cli_overrides=[
            "--noise_scheduler_type=DDIM",
            f"--num_inference_steps={args_cli.num_inference_steps}",
        ],
    )
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args_cli.checkpoint)
    dev = next(policy.parameters()).device

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    outcomes = []
    for b in range(batches):
        obs, _ = env.reset(seed=base_seed + b)
        policy.reset()
        success = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        finished = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        for _ in range(args_cli.max_steps):
            with torch.inference_mode():
                batch = pre(obs_to_batch(obs, dev))
                action = post(policy.select_action(batch))
            obs, _, terminated, truncated, _ = env.step(action.to(env.device))
            succ_now = env.termination_manager.get_term("success")
            success |= succ_now & ~finished
            finished |= terminated | truncated
            if bool(finished.all()):
                break
        outcomes.extend(
            {"batch": b, "env": i, "success": bool(success[i])} for i in range(num_envs)
        )
        sr_so_far = sum(o["success"] for o in outcomes) / len(outcomes)
        print(f"[eval] batch {b+1}/{batches}  running SR={sr_so_far:.3f}", flush=True)

    n = len(outcomes)
    k = sum(o["success"] for o in outcomes)
    result = {
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "num_inference_steps": args_cli.num_inference_steps,
        "protocol": proto,
        "episodes": n,
        "successes": k,
        "success_rate": k / n,
        "outcomes": outcomes,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[eval] DONE SR={k}/{n}={k/n:.3f} -> {args_cli.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
