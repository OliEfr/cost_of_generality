"""Record scripted-expert source demos for a cup_place task (spec 04 section 6.2, Pattern B).

Multi-env parallel collection with automatic success-only export: the success
termination stays ACTIVE, so RecorderManager auto-exports successful episodes
on reset and silently drops failures (time_out is the failure path).

Run inside the cog_isaac env:
  ./isaaclab.sh -p <repo>/src/cog/datagen/record_source_demos.py \
      --task Cog-CupPlace-L0-IK-Rel-v0 --num_envs 8 --num_demos 10 \
      --dataset_file <repo>/data/hdf5/L0_source.hdf5 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Cog-CupPlace-L0-IK-Rel-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--num_demos", type=int, default=10)
parser.add_argument("--dataset_file", type=str, default="./data/hdf5/source.hdf5")
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import random

import gymnasium as gym
import numpy as np
import torch

import cog.tasks.cup_place  # noqa: F401  (registers Cog-CupPlace-* env IDs)
from cog.tasks.cup_place.assets import CUP_VARIANTS
from cog.tasks.cup_place.levels import SUB_LEVELS
from cog.tasks.cup_place.state_machine import DOWN_QUAT_WXYZ, PlaceSm, convert_abs_to_rel_actions

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    sub_key = args_cli.task.split("-")[2]  # Cog-CupPlace-<KEY>-...
    variant = CUP_VARIANTS[SUB_LEVELS[sub_key].cup_variant]

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = os.path.dirname(os.path.abspath(args_cli.dataset_file))
    env_cfg.recorders.dataset_filename = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

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

    target = args_cli.num_demos
    step_count = 0
    while simulation_app.is_running():
        _, _, terminated, truncated, _ = env.step(actions)
        dones = terminated | truncated
        if dones.any():
            sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

        origins = env.scene.env_origins
        ee = env.scene["ee_frame"]
        tcp_pos = ee.data.target_pos_w[..., 0, :] - origins
        tcp_quat = ee.data.target_quat_w[..., 0, :]
        ee_pose = torch.cat([tcp_pos, tcp_quat], dim=-1)

        cup_pos = env.scene["cup"].data.root_pos_w - origins
        grasp_pos = cup_pos + torch.tensor([0.0, 0.0, variant.grasp_z_offset], device=env.device)
        grasp_pose = torch.cat([grasp_pos, down_quat], dim=-1)

        goal_pos = (env.scene["goal_marker"].data.root_pos_w - origins).clone()
        goal_pos[:, 2] = goal_pos[:, 2] + variant.half_height + 0.006
        goal_pose = torch.cat([goal_pos, down_quat], dim=-1)

        abs_target = sm.compute(ee_pose, grasp_pose, goal_pose)
        actions = convert_abs_to_rel_actions(abs_target, tcp_pos, tcp_quat)

        step_count += 1
        exported = env.recorder_manager.exported_successful_episode_count
        if step_count % 200 == 0:
            print(f"[record] steps={step_count} exported={exported}/{target}")
        if exported >= target:
            break

    print(f"[record] DONE: {env.recorder_manager.exported_successful_episode_count} demos -> "
          f"{env_cfg.recorders.dataset_export_dir_path}/{env_cfg.recorders.dataset_filename}.hdf5")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
