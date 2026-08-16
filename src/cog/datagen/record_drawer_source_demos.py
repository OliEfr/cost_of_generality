"""Record scripted-expert source demos for a drawer_stow task (cup_place pattern).

Run inside the cog_isaac env:
  python src/cog/datagen/record_drawer_source_demos.py \
      --task Cog-DrawerStow-L2-IK-Rel-v0 --num_envs 1 --num_demos 15 \
      --dataset_file data/hdf5/T2_L2_source.hdf5 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Cog-DrawerStow-L0-IK-Rel-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_demos", type=int, default=10)
parser.add_argument("--dataset_file", type=str, default="./data/hdf5/t2_source.hdf5")
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

import cog.tasks.drawer_stow  # noqa: F401  (registers Cog-DrawerStow-* env IDs)
from cog.tasks.cup_place.state_machine import convert_abs_to_rel_actions
from cog.tasks.drawer_stow.assets import BOX_VARIANTS
from cog.tasks.drawer_stow.levels import SUB_LEVELS
from cog.tasks.drawer_stow.state_machine import DrawerStowSm

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    sub_key = args_cli.task.split("-")[2]  # Cog-DrawerStow-<KEY>-...
    variant = BOX_VARIANTS[SUB_LEVELS[sub_key].box_variant]

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = os.path.dirname(os.path.abspath(args_cli.dataset_file))
    env_cfg.recorders.dataset_filename = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    sm = DrawerStowSm(
        dt=env_cfg.sim.dt * env_cfg.decimation,
        num_envs=env.num_envs,
        device=env.device,
        position_threshold=0.012,
    )
    actions = torch.zeros(env.num_envs, 7, device=env.device)
    cab_joint_ids, _ = env.scene["cabinet"].find_joints(["drawer_top_joint"])
    cab_body_ids, _ = env.scene["cabinet"].find_bodies(["drawer_top"])

    target = args_cli.num_demos
    step_count = 0
    attempts = 0
    while simulation_app.is_running():
        _, _, terminated, truncated, _ = env.step(actions)
        dones = terminated | truncated
        if dones.any():
            attempts += int(dones.sum().item())
            sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

        origins = env.scene.env_origins
        ee = env.scene["ee_frame"]
        tcp_pos = ee.data.target_pos_w[..., 0, :] - origins
        tcp_quat = ee.data.target_quat_w[..., 0, :]
        ee_pose = torch.cat([tcp_pos, tcp_quat], dim=-1)

        cab_frame = env.scene["cabinet_frame"]
        handle_pose = torch.cat(
            [cab_frame.data.target_pos_w[..., 0, :] - origins,
             cab_frame.data.target_quat_w[..., 0, :]], dim=-1)

        cabinet = env.scene["cabinet"]
        drawer_joint = cabinet.data.joint_pos[:, cab_joint_ids[0]]
        drawer_body_pose = torch.cat(
            [cabinet.data.body_pos_w[:, cab_body_ids[0]] - origins,
             cabinet.data.body_quat_w[:, cab_body_ids[0]]], dim=-1)

        obj = env.scene["object"]
        object_pose = torch.cat([obj.data.root_pos_w - origins, obj.data.root_quat_w], dim=-1)

        abs_target = sm.compute(
            ee_pose, handle_pose, drawer_joint, object_pose, drawer_body_pose,
            grasp_z_offset=variant.grasp_z_offset,
            object_half_size=variant.half_size,
        )
        actions = convert_abs_to_rel_actions(abs_target, tcp_pos, tcp_quat)

        step_count += 1
        exported = env.recorder_manager.exported_successful_episode_count
        if step_count % 200 == 0:
            print(f"[record] steps={step_count} exported={exported}/{target} attempts={attempts}")
        if exported >= target:
            break

    n_exp = env.recorder_manager.exported_successful_episode_count
    sr = n_exp / max(attempts, 1)
    print(f"[record] DONE: {n_exp} demos, attempts={attempts}, expert_SR={sr:.2f} -> "
          f"{env_cfg.recorders.dataset_export_dir_path}/{env_cfg.recorders.dataset_filename}.hdf5")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
