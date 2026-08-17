"""Record scripted-expert source demos for a push_target task (T2 pattern).

Run inside the cog_isaac env:
  python src/cog/datagen/record_push_source_demos.py \
      --task Cog-PushTarget-L2-IK-Rel-v0 --num_envs 1 --num_demos 20 \
      --dataset_file data/hdf5/T3_L2_source.hdf5 --headless

`--max_final_err` filters on SOURCE QUALITY, which matters more here than for the other
two tasks: Mimic replays a source trajectory rigidly, so a demo that leaves the puck near
the edge of the 5 cm success disk hands that error to every generated copy. Episodes can
legitimately succeed at ~5 cm (the puck stalls just inside the disk and counts as settled),
so those successes are real but make poor templates. Rather than tighten the env's success
criterion -- which was tried and cost 5-35 points of expert SR, see docs/journal.md
2026-08-17 -- sources are selected here on measured final placement error.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Cog-PushTarget-L2-IK-Rel-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_demos", type=int, default=20)
parser.add_argument("--dataset_file", type=str, default="./data/hdf5/t3_source.hdf5")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--max_final_err", type=float, default=0.025,
                    help="report successful episodes whose final |puck-target| exceeds this")
parser.add_argument("--source_success_radius", type=float, default=0.020,
                    help="success radius used FOR RECORDING ONLY (the level's own gate stays "
                         "at 5 cm for generation and evaluation)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import random

import gymnasium as gym
import numpy as np
import torch

import cog.tasks.push_target  # noqa: F401  (registers Cog-PushTarget-* env IDs)
from cog.tasks.cup_place.state_machine import convert_abs_to_rel_actions
from cog.tasks.push_target.assets import PUCK_VARIANTS, SUCCESS_RADIUS
from cog.tasks.push_target.levels import PUSH_DISTANCE, SUB_LEVELS
from cog.tasks.push_target.state_machine import PushSm

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    sub_key = args_cli.task.split("-")[2]  # Cog-PushTarget-<KEY>-...
    variant = PUCK_VARIANTS[SUB_LEVELS[sub_key].puck_variant]

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = os.path.dirname(os.path.abspath(args_cli.dataset_file))
    env_cfg.recorders.dataset_filename = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    # Tighten the success radius FOR RECORDING ONLY. Measured 2026-08-17: with the level's
    # 5 cm gate, source demos came out at a median final error of 5.01 cm -- success fires
    # the moment the puck stalls just inside the disk, ending the episode before the expert
    # pushes to centre. Mimic replays a source rigidly, so each generated copy would inherit
    # ~5 cm against a 5 cm gate and fail on any slip. Recording against a 2 cm radius makes
    # the expert push to centre; generation and evaluation keep the level's real 5 cm gate.
    env_cfg.terminations.success.params["success_radius"] = args_cli.source_success_radius

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    sm = PushSm(
        dt=env_cfg.sim.dt * env_cfg.decimation,
        num_envs=env.num_envs,
        device=env.device,
        push_distance=PUSH_DISTANCE,
    )
    actions = torch.zeros(env.num_envs, 7, device=env.device)

    target = args_cli.num_demos
    step_count = 0
    attempts = 0
    rejected = 0
    err_log = []
    while simulation_app.is_running():
        origins = env.scene.env_origins
        obj = env.scene["object"]
        marker = env.scene["target_marker"]
        # measured BEFORE stepping: Isaac Lab resets inside step(), so a post-step read
        # belongs to the next episode (this cost two bad diagnostics on 2026-08-17)
        pre_err = torch.linalg.vector_norm(
            (obj.data.root_pos_w - marker.data.root_pos_w)[:, :2], dim=1
        )

        _, _, terminated, truncated, _ = env.step(actions)
        dones = terminated | truncated
        if dones.any():
            attempts += int(dones.sum().item())
            for d in dones.nonzero(as_tuple=False).squeeze(-1).tolist():
                if bool(terminated[d]):
                    err_log.append(float(pre_err[d]))
                    if float(pre_err[d]) > args_cli.max_final_err:
                        rejected += 1
            sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

        ee = env.scene["ee_frame"]
        tcp_pos = ee.data.target_pos_w[..., 0, :] - origins
        tcp_quat = ee.data.target_quat_w[..., 0, :]
        ee_pose = torch.cat([tcp_pos, tcp_quat], dim=-1)
        puck_pose = torch.cat([obj.data.root_pos_w - origins, obj.data.root_quat_w], dim=-1)
        target_pos = marker.data.root_pos_w - origins

        abs_target = sm.compute(
            ee_pose, puck_pose, target_pos,
            contact_z=variant.contact_z,
            puck_radius=variant.radius,
            success_radius=SUCCESS_RADIUS,
        )
        actions = convert_abs_to_rel_actions(abs_target, tcp_pos, tcp_quat)

        step_count += 1
        exported = env.recorder_manager.exported_successful_episode_count
        if step_count % 200 == 0:
            print(f"[record] steps={step_count} exported={exported}/{target} "
                  f"attempts={attempts} over_err={rejected}", flush=True)
        if exported >= target:
            break

    n_exp = env.recorder_manager.exported_successful_episode_count
    sr = n_exp / max(attempts, 1)
    med = float(np.median(err_log)) if err_log else float("nan")
    print(f"[record] DONE: {n_exp} demos, attempts={attempts}, expert_SR={sr:.2f}", flush=True)
    print(f"[record] final placement error over successes: median {med*100:.2f} cm, "
          f"{rejected} of {len(err_log)} exceeded --max_final_err "
          f"({args_cli.max_final_err*100:.1f} cm) and should be dropped at annotation",
          flush=True)
    print(f"[record] -> {env_cfg.recorders.dataset_export_dir_path}/"
          f"{env_cfg.recorders.dataset_filename}.hdf5", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
