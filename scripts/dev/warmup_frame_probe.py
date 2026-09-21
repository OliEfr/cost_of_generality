"""Is the first-batch depression a RENDER effect or a PHYSICS effect?

The depression is known to follow process position rather than the poses (August probe: a fresh
process scored 0.35 on seed 5001, which scores 0.85 warm). It has never been attributed to a
subsystem; "renderer/physics warm-up" is a label, not a measurement.

This probe removes the policy from the loop so that the two candidates separate. It resets the
same env with the same seed several times in one process and drives every cycle with an IDENTICAL
zero-action sequence, so the trajectories cannot diverge through the policy. Any difference
between cycle 1 and cycle 2 is therefore the process's own state, and it lands in exactly one of
two observable places:

  * the CAMERA tensors differ  -> the renderer has not converged in the first cycle
  * the RIGID-BODY poses differ -> the physics has not converged in the first cycle

Both can be true. Neither being true would mean the depression needs the policy in the loop and is
not a raw-observation effect at all, which is itself a useful answer.

Run (inside the container, as the eval jobs do):
  python -m scripts.dev.warmup_frame_probe --task Cog-CupPlace-L1-IK-Rel-Visuomotor-v0 \
      --out $WORK/cog/results/_probes/warmup_frames.npz --headless --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--seed", type=int, default=5000)
parser.add_argument("--cycles", type=int, default=3, help="resets of the same seed in one process")
parser.add_argument("--steps", type=int, default=30, help="zero-action steps per cycle")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_TASK_MODULES = {
    "Cog-CupPlace": "cog.tasks.cup_place",
    "Cog-DrawerStow": "cog.tasks.drawer_stow",
    "Cog-PushTarget": "cog.tasks.push_target",
}
_prefix = next((p for p in _TASK_MODULES if args_cli.task.startswith(p)), None)
if _prefix is None:
    raise SystemExit(f"no task module registered for {args_cli.task}")
__import__(_TASK_MODULES[_prefix])

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

CAPTURE_T = (0, 1, 2, 5, 10, 20, 29)
CAMS = ("table_cam", "wrist_cam")


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    act_dim = env.action_manager.total_action_dim
    zero = torch.zeros((args_cli.num_envs, act_dim), device=env.device)

    frames = {}   # (cycle, cam, t) -> uint8 (B,H,W,C)
    poses = {}    # (cycle, t) -> float32 (B, 7) root pose of the manipulandum
    body = "object" if "object" in env.scene.keys() else list(env.scene.rigid_objects)[0]
    print(f"[probe] task={args_cli.task} envs={args_cli.num_envs} act_dim={act_dim} "
          f"tracking scene['{body}']", flush=True)

    for c in range(args_cli.cycles):
        obs, _ = env.reset(seed=args_cli.seed)
        for t in range(args_cli.steps):
            if t in CAPTURE_T:
                pol = obs["policy"]
                for cam in CAMS:
                    frames[(c, cam, t)] = pol[cam].detach().cpu().numpy().astype(np.uint8)
                poses[(c, t)] = env.scene[body].data.root_state_w[:, :7].detach().cpu().numpy()
            obs, _, _, _, _ = env.step(zero)
        print(f"[probe] cycle {c} done", flush=True)

    # ---- comparisons, cycle 0 (cold) against every later cycle
    report = {"task": args_cli.task, "seed": args_cli.seed, "num_envs": args_cli.num_envs,
              "cycles": args_cli.cycles, "steps": args_cli.steps, "pairs": []}
    for c in range(1, args_cli.cycles):
        for t in CAPTURE_T:
            row = {"cycle_a": 0, "cycle_b": c, "t": t}
            for cam in CAMS:
                a = frames[(0, cam, t)].astype(np.int16)
                b = frames[(c, cam, t)].astype(np.int16)
                d = np.abs(a - b)
                row[f"{cam}_mae"] = float(d.mean())
                row[f"{cam}_p99"] = float(np.percentile(d, 99))
                row[f"{cam}_max"] = int(d.max())
                row[f"{cam}_pct_pixels_differ"] = float((d > 0).mean() * 100.0)
                row[f"{cam}_channel_mean_shift"] = [
                    float(x) for x in (b.astype(np.float64).mean(axis=(0, 1, 2))
                                       - a.astype(np.float64).mean(axis=(0, 1, 2)))
                ]
            pa, pb = poses[(0, t)], poses[(c, t)]
            row["pose_max_abs_delta"] = float(np.abs(pa - pb).max())
            row["pose_xyz_max_abs_delta"] = float(np.abs(pa[:, :3] - pb[:, :3]).max())
            report["pairs"].append(row)

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    np.savez_compressed(
        args_cli.out,
        report=json.dumps(report),
        **{f"c{c}_{cam}_t{t}": frames[(c, cam, t)][:2] for (c, cam, t) in frames},
        **{f"pose_c{c}_t{t}": poses[(c, t)] for (c, t) in poses},
    )
    print(json.dumps(report, indent=1), flush=True)
    print(f"PROBE_OK {args_cli.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
