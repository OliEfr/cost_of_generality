"""Diagnose expert failures: per-episode final SM state, lift/place outcome."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Cog-CupPlace-L0-IK-Rel-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--episodes", type=int, default=24)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import cog.tasks.cup_place  # noqa: F401
from cog.tasks.cup_place.assets import CUP_VARIANTS
from cog.tasks.cup_place.levels import SUB_LEVELS
from cog.tasks.cup_place.state_machine import (
    DOWN_QUAT_WXYZ,
    PlaceSm,
    convert_abs_to_rel_actions,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

STATE_NAMES = [
    "REST", "APPR_ABOVE_OBJ", "APPR_OBJ", "GRASP", "LIFT",
    "APPR_ABOVE_GOAL", "LOWER", "RELEASE", "RETREAT", "DONE",
]


def main():
    sub_key = args_cli.task.split("-")[2]
    variant = CUP_VARIANTS[SUB_LEVELS[sub_key].cup_variant]
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    sm = PlaceSm(
        dt=env_cfg.sim.dt * env_cfg.decimation,
        num_envs=env.num_envs,
        device=env.device,
        position_threshold=0.012,
    )
    down_quat = torch.tensor([DOWN_QUAT_WXYZ], device=env.device).repeat(env.num_envs, 1)
    actions = torch.zeros(env.num_envs, 7, device=env.device)
    max_cup_z = torch.zeros(env.num_envs, device=env.device)
    ep_len = torch.zeros(env.num_envs, device=env.device)

    finished = 0
    stats = {}
    while simulation_app.is_running() and finished < args_cli.episodes:
        _, _, terminated, truncated, _ = env.step(actions)
        dones = terminated | truncated

        origins = env.scene.env_origins
        ee = env.scene["ee_frame"]
        tcp_pos = ee.data.target_pos_w[..., 0, :] - origins
        tcp_quat = ee.data.target_quat_w[..., 0, :]
        cup_pos = env.scene["cup"].data.root_pos_w - origins
        goal_pos = (env.scene["goal_marker"].data.root_pos_w - origins).clone()
        max_cup_z = torch.maximum(max_cup_z, cup_pos[:, 2])
        ep_len += 1

        if dones.any():
            succ = env.termination_manager.get_term("success")
            ids = dones.nonzero(as_tuple=False).squeeze(-1)
            for i in ids.tolist():
                st = STATE_NAMES[int(sm.sm_state[i])]
                d_des = torch.norm(tcp_pos[i] - sm.des_ee_pose[i, 0:3]).item()
                dxy = torch.norm(cup_pos[i, :2] - goal_pos[i, :2]).item()
                lifted = max_cup_z[i].item() > variant.half_height * 2 + 0.03
                key = (st, bool(succ[i].item()), lifted)
                stats[key] = stats.get(key, 0) + 1
                print(
                    f"[ep] env={i} len={int(ep_len[i])} success={bool(succ[i].item())} "
                    f"state={st} |ee-des|={d_des:.3f} cup_goal_xy={dxy:.3f} "
                    f"max_cup_z={max_cup_z[i]:.3f} lifted={lifted}",
                    flush=True,
                )
                finished += 1
            max_cup_z[ids] = 0.0
            ep_len[ids] = 0
            sm.reset_idx(ids)

        abs_target = sm.compute(
            torch.cat([tcp_pos, tcp_quat], dim=-1),
            torch.cat(
                [cup_pos + torch.tensor([0.0, 0.0, variant.grasp_z_offset], device=env.device), down_quat],
                dim=-1,
            ),
            torch.cat(
                [goal_pos + torch.tensor([0.0, 0.0, variant.half_height + variant.grasp_z_offset + 0.006], device=env.device), down_quat],
                dim=-1,
            ),
        )
        actions = convert_abs_to_rel_actions(abs_target, tcp_pos, tcp_quat)

    print("[summary] (state, success, lifted) -> count", flush=True)
    for k in sorted(stats, key=str):
        print(f"[summary] {k}: {stats[k]}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
