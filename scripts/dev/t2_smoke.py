"""Drawer-stow bring-up smoke: env creation + obs check + expert episodes."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Cog-DrawerStow-L0-IK-Rel-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--episodes", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import cog.tasks.drawer_stow  # noqa: F401
from cog.tasks.cup_place.state_machine import convert_abs_to_rel_actions
from cog.tasks.drawer_stow.assets import BOX_VARIANTS
from cog.tasks.drawer_stow.levels import SUB_LEVELS
from cog.tasks.drawer_stow.state_machine import DrawerStowSm, Sm

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

sub_key = args_cli.task.split("-")[2]
variant = BOX_VARIANTS[SUB_LEVELS[sub_key].box_variant]
env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
obs, _ = env.reset(seed=0)
print("[smoke] policy obs:", {k: tuple(v.shape) for k, v in obs["policy"].items()}, flush=True)
print("[smoke] subtask obs:", {k: tuple(v.shape) for k, v in obs["subtask_terms"].items()}, flush=True)

sm = DrawerStowSm(dt=env_cfg.sim.dt * env_cfg.decimation, num_envs=env.num_envs, device=env.device)
actions = torch.zeros(env.num_envs, 7, device=env.device)
cab_joint_ids, _ = env.scene["cabinet"].find_joints(["drawer_top_joint"])
cab_body_ids, _ = env.scene["cabinet"].find_bodies(["drawer_top"])

STATE_NAMES = {v: k for k, v in vars(Sm).items() if isinstance(v, int)}
finished = 0
succ = 0
step = 0
last_drawer = [0.0] * env.num_envs
last_depth = [0.0] * env.num_envs
ep_steps = [0] * env.num_envs
req = 0.03 + 0.012 + variant.half_size + 0.004
while finished < args_cli.episodes and step < 6000:
    _, _, terminated, truncated, _ = env.step(actions)
    dones = terminated | truncated
    if dones.any():
        ids = dones.nonzero(as_tuple=False).squeeze(-1)
        for i in ids.tolist():
            s_flag = bool(env.termination_manager.get_term("success")[i])
            print(f"[smoke] episode end env{i}: success={s_flag} sm_state={STATE_NAMES.get(int(sm.state[i]), '?')} "
                  f"drawer_pre_reset={last_drawer[i]:.3f} depth={last_depth[i]:.3f} req={req:.3f} "
                  f"ep_steps={ep_steps[i]}", flush=True)
            ep_steps[i] = 0
            finished += 1
            succ += int(s_flag)
        sm.reset_idx(ids)
    origins = env.scene.env_origins
    ee = env.scene["ee_frame"]
    tcp_pos = ee.data.target_pos_w[..., 0, :] - origins
    tcp_quat = ee.data.target_quat_w[..., 0, :]
    ee_pose = torch.cat([tcp_pos, tcp_quat], dim=-1)
    cab_frame = env.scene["cabinet_frame"]
    handle_pose = torch.cat([cab_frame.data.target_pos_w[..., 0, :] - origins,
                             cab_frame.data.target_quat_w[..., 0, :]], dim=-1)
    cabinet = env.scene["cabinet"]
    drawer_joint = cabinet.data.joint_pos[:, cab_joint_ids[0]]
    drawer_body_pose = torch.cat([cabinet.data.body_pos_w[:, cab_body_ids[0]] - origins,
                                  cabinet.data.body_quat_w[:, cab_body_ids[0]]], dim=-1)
    obj = env.scene["object"]
    object_pose = torch.cat([obj.data.root_pos_w - origins, obj.data.root_quat_w], dim=-1)
    abs_target = sm.compute(ee_pose, handle_pose, drawer_joint, object_pose, drawer_body_pose,
                            grasp_z_offset=variant.grasp_z_offset, object_half_size=variant.half_size)
    actions = convert_abs_to_rel_actions(abs_target, tcp_pos, tcp_quat)
    back = -torch.nn.functional.normalize(
        torch.stack([torch.cos(torch.zeros(env.num_envs, device=env.device)),
                     torch.zeros(env.num_envs, device=env.device)], dim=-1), dim=-1)
    for i in range(env.num_envs):
        last_drawer[i] = float(drawer_joint[i])
        d = float(((tcp_pos[i, 0:2] - handle_pose[i, 0:2]) * torch.tensor([1.0, 0.0], device=env.device)).sum())
        last_depth[i] = d
        ep_steps[i] += 1
    step += 1
    if step % 200 == 0:
        print(f"[smoke] step {step}: sm states={[STATE_NAMES.get(int(x), '?') for x in sm.state]} "
              f"drawer={[round(float(d), 3) for d in drawer_joint]}", flush=True)

print(f"[smoke] RESULT: {succ}/{finished} expert successes", flush=True)
env.close()
simulation_app.close()
