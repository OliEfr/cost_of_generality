"""Does an extra sim.render() before the first observation remove the cold-frame defect?

D33 leaves this open: the wrist camera's first frame in a fresh process is a different image, not a
noisy one, which reads like a camera transform that has not caught up with the post-reset robot
pose. If so, driving the renderer a few more times before the first observation is read should make
the cold frame converge on the warm one, and the warm-up BATCH could shrink to a few render calls.

This probe only measures; it changes nothing in the eval path. Three reset cycles at one seed in
one process. In every cycle the observation is captured straight from reset() and again after
1, 2, 4 and 8 extra render calls. The yardstick is the renderer's own noise floor, measured as
cycle 1 vs cycle 2 (~0.62 MAE in the D33 probe): the fix works if cycle 0 after k renders lands on
the warm frame at that floor.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--seed", type=int, default=5000)
parser.add_argument("--cycles", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

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

CAMS = ("table_cam", "wrist_cam")
RENDERS = (0, 1, 2, 4, 8)  # cumulative extra render() calls before the capture


def grab(env, obs=None):
    """Camera tensors as the policy would see them right now."""
    if obs is None:
        for name in CAMS:
            sensor = env.scene.sensors.get(name) if hasattr(env.scene, "sensors") else None
            if sensor is not None:
                try:
                    sensor.update(dt=0.0, force_recompute=True)
                except TypeError:
                    sensor.update(0.0)
        obs = env.observation_manager.compute()
    pol = obs["policy"]
    return {c: pol[c].detach().cpu().numpy().astype(np.uint8) for c in CAMS}


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    print(f"[probe] task={args_cli.task} envs={args_cli.num_envs} "
          f"sensors={list(getattr(env.scene, 'sensors', {}) or {})}", flush=True)

    caps = {}  # (cycle, k_renders, cam) -> uint8 array
    for c in range(args_cli.cycles):
        obs, _ = env.reset(seed=args_cli.seed)
        done = 0
        for k in RENDERS:
            while done < k:
                env.sim.render()
                done += 1
            g = grab(env, obs if k == 0 else None)
            for cam in CAMS:
                caps[(c, k, cam)] = g[cam]
        print(f"[probe] cycle {c} captured at renders {RENDERS}", flush=True)

    def mae(a, b):
        return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())

    report = {"task": args_cli.task, "seed": args_cli.seed, "renders": list(RENDERS), "rows": []}
    for cam in CAMS:
        floor = mae(caps[(1, 0, cam)], caps[(2, 0, cam)])
        for k in RENDERS:
            row = {
                "cam": cam,
                "extra_renders": k,
                # cold after k renders vs the warm frame as the pipeline produces it today
                "cold_k_vs_warm0": mae(caps[(0, k, cam)], caps[(1, 0, cam)]),
                # cold after k renders vs warm after the same k renders
                "cold_k_vs_warm_k": mae(caps[(0, k, cam)], caps[(1, k, cam)]),
                # does the extra render move the warm frame at all?
                "warm_k_vs_warm0": mae(caps[(1, k, cam)], caps[(1, 0, cam)]),
                "noise_floor": floor,
            }
            row["ratio_to_floor"] = row["cold_k_vs_warm0"] / floor if floor else float("nan")
            report["rows"].append(row)

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    np.savez_compressed(args_cli.out, report=json.dumps(report),
                        **{f"c{c}_r{k}_{cam}": caps[(c, k, cam)][:2] for (c, k, cam) in caps})
    print(json.dumps(report, indent=1), flush=True)
    print(f"PROBE_OK {args_cli.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
