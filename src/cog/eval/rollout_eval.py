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
parser.add_argument(
    "--stages",
    action="store_true",
    help="drawer_stow only: record per-episode stage latches (drawer opened, object "
    "lifted, object over open drawer) alongside the official success",
)
parser.add_argument(
    "--instructions",
    default=None,
    help="frozen configs/instructions/instructions_vN.json: assign instruction "
    "idx=(batch+env)%%K per episode, inject the string (batch['task']) and, iff the "
    "checkpoint declares observation.environment_state, the frozen embedding. "
    "Layered on top of the frozen protocol seeds; default off = today's behaviour",
)
parser.add_argument(
    "--instruction_task",
    default=None,
    help="task key into the instructions file (default: derived from the gym prefix)",
)
parser.add_argument(
    "--swap_instructions_from",
    default=None,
    help="probe mismatch condition: draw instructions from THIS task's set instead "
    "of the env's own (e.g. push_target instructions in the cup_place env)",
)
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

from lerobot.policies.factory import get_policy_class, make_pre_post_processors


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

    # Stage instrumentation reads sim state the policy never sees; it cannot alter the
    # rollout. Thresholds mirror the success termination (min_drawer_open=0.15) and the
    # cavity bounds; "lifted" = 5 cm above the episode's initial object height.
    stages_on = args_cli.stages and args_cli.task.startswith("Cog-DrawerStow")
    if args_cli.stages and not stages_on:
        print(f"[eval] --stages ignored: no stage definitions for {args_cli.task}", flush=True)
    if stages_on:
        import isaaclab.utils.math as math_utils
        from cog.tasks.drawer_stow.assets import DRAWER_CAVITY_HALF_X, DRAWER_CAVITY_HALF_Y

    # Generic policy loading: language-less diffusion checkpoints take the exact same
    # path as before; a multi_task_dit checkpoint (candidate B) needs the in-repo
    # plugin imported first so its @PreTrainedConfig.register_subclass runs.
    with open(os.path.join(args_cli.checkpoint, "config.json")) as f:
        policy_type = json.load(f)["type"]
    if policy_type == "multi_task_dit":
        import lerobot_policy_mtdit  # noqa: F401  (src/ is on PYTHONPATH alongside cog)
    policy = get_policy_class(policy_type).from_pretrained(
        args_cli.checkpoint,
        cli_overrides=[
            "--noise_scheduler_type=DDIM",
            f"--num_inference_steps={args_cli.num_inference_steps}",
        ],
    )
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args_cli.checkpoint)
    dev = next(policy.parameters()).device

    # Language injection (default: none). The env_state channel is checkpoint-driven:
    # a candidate-A checkpoint declares observation.environment_state and CANNOT run
    # without embeddings; a language-less checkpoint must never receive the key.
    use_env_state = policy.config.env_state_feature is not None
    instr_strings = instr_embs = None
    instr_meta = {}
    if args_cli.instructions:
        import hashlib
        from pathlib import Path

        import numpy as np

        spec_path = Path(args_cli.instructions)
        spec = json.loads(spec_path.read_text())
        npz_path = spec_path.parent / spec["embeddings_file"]
        npz = np.load(npz_path)
        env_kind = _TASK_MODULES[_prefix].rsplit(".", 1)[-1] if _prefix else None
        task_kind = args_cli.instruction_task or env_kind
        lookup_kind = args_cli.swap_instructions_from or task_kind
        instr_strings = spec["tasks"][lookup_kind]
        instr_embs = torch.from_numpy(npz[lookup_kind]).float().to(dev)
        instr_meta = {
            "file": str(spec_path),
            "sha256_json": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            "sha256_npz": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
            "instruction_task": task_kind,
            "swap_instructions_from": args_cli.swap_instructions_from,
            "assignment": "(batch+env)%K",
            "K": len(instr_strings),
        }
        print(f"[eval] instructions: {lookup_kind} x{len(instr_strings)} "
              f"(env_state injection: {use_env_state})", flush=True)
    elif use_env_state:
        raise SystemExit("checkpoint declares observation.environment_state (language-"
                         "conditioned candidate A) -- pass --instructions")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    outcomes = []
    for b in range(batches):
        obs, _ = env.reset(seed=base_seed + b)
        policy.reset()
        idx_list = batch_tasks = batch_embs = None
        if instr_strings:
            # (b+i)%K rotates instructions across env columns batch-to-batch, so each
            # instruction gets its 5 episodes in 5 different (batch, column) slots;
            # poses are drawn fresh per reset -> no instruction<->pose confound (D18).
            idx_list = [(b + i) % len(instr_strings) for i in range(num_envs)]
            batch_tasks = [instr_strings[j] for j in idx_list]
            if use_env_state:
                batch_embs = instr_embs[idx_list]
        success = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        finished = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        if stages_on:
            cab, obj = env.scene["cabinet"], env.scene["object"]
            jid = cab.find_joints(["drawer_top_joint"])[0][0]
            bid = cab.find_bodies(["drawer_top"])[0][0]
            obj_z0 = obj.data.root_pos_w[:, 2].clone()
            opened = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
            lifted = torch.zeros_like(opened)
            over = torch.zeros_like(opened)
            max_open = torch.zeros(num_envs, device=env.device)
            max_lift = torch.zeros(num_envs, device=env.device)
            t_open, t_lift, t_over, t_succ = (
                torch.full((num_envs,), -1, dtype=torch.long, device=env.device) for _ in range(4)
            )
        for t in range(args_cli.max_steps):
            with torch.inference_mode():
                raw = obs_to_batch(obs, dev)
                if instr_strings:
                    raw["task"] = batch_tasks
                    if use_env_state:
                        raw["observation.environment_state"] = batch_embs
                batch = pre(raw)
                action = post(policy.select_action(batch))
            obs, _, terminated, truncated, _ = env.step(action.to(env.device))
            succ_now = env.termination_manager.get_term("success")
            # BATCH-BOUNDARY CARRYOVER BUG (found 2026-08-21): on the FIRST step after a
            # manual env.reset(), get_term("success") still returns the previous batch's
            # value, so every env that genuinely succeeded in batch b-1 latches a phantom
            # success at t=0 of batch b (verified: 20/20 batch transitions, phantom
            # episodes never even lift the object). Genuine success at t=0 is physically
            # impossible in all three tasks (shortest demo >= 150 steps), so drop it.
            # All evals before this fix are affected; see docs/journal.md 2026-08-21.
            if t == 0:
                succ_now = torch.zeros_like(succ_now)
            if stages_on:
                alive = ~finished  # same latching semantics as the official success
                jpos = cab.data.joint_pos[:, jid]
                opos = obj.data.root_pos_w
                local = math_utils.quat_apply_inverse(
                    cab.data.body_quat_w[:, bid], opos - cab.data.body_pos_w[:, bid]
                )
                open_now = jpos >= 0.15
                lift_now = (opos[:, 2] - obj_z0) >= 0.05
                over_now = (
                    (local[:, 0].abs() < DRAWER_CAVITY_HALF_X)
                    & (local[:, 1].abs() < DRAWER_CAVITY_HALF_Y)
                    & open_now
                )
                for now, latch, t_first in (
                    (open_now, opened, t_open),
                    (lift_now, lifted, t_lift),
                    (over_now, over, t_over),
                    (succ_now, success, t_succ),
                ):
                    t_first[now & ~latch & alive] = t
                opened |= open_now & alive
                lifted |= lift_now & alive
                over |= over_now & alive
                max_open = torch.where(alive, torch.maximum(max_open, jpos), max_open)
                max_lift = torch.where(alive, torch.maximum(max_lift, opos[:, 2] - obj_z0), max_lift)
            success |= succ_now & ~finished
            finished |= terminated | truncated
            if bool(finished.all()):
                break
        if stages_on:
            outcomes.extend(
                {
                    "batch": b,
                    "env": i,
                    "success": bool(success[i]),
                    "drawer_opened": bool(opened[i]),
                    "object_lifted": bool(lifted[i]),
                    "object_over_drawer": bool(over[i]),
                    "max_drawer_open": round(float(max_open[i]), 4),
                    "max_object_lift": round(float(max_lift[i]), 4),
                    "t_open": int(t_open[i]),
                    "t_lift": int(t_lift[i]),
                    "t_over": int(t_over[i]),
                    "t_success": int(t_succ[i]),
                    **({"instruction_index": idx_list[i]} if idx_list else {}),
                }
                for i in range(num_envs)
            )
        else:
            outcomes.extend(
                {"batch": b, "env": i, "success": bool(success[i]),
                 **({"instruction_index": idx_list[i]} if idx_list else {})}
                for i in range(num_envs)
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
    if instr_meta:
        result["instructions"] = instr_meta
        per: dict[int, list[int]] = {}
        for o in outcomes:
            s = per.setdefault(o["instruction_index"], [0, 0])
            s[0] += o["success"]
            s[1] += 1
        result["per_instruction"] = {
            str(j): {"successes": s[0], "episodes": s[1]} for j, s in sorted(per.items())
        }
    if stages_on:
        result["stages"] = {
            key: sum(o[key] for o in outcomes) / n
            for key in ("drawer_opened", "object_lifted", "object_over_drawer")
        }
        print(f"[eval] stage rates: {result['stages']}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[eval] DONE SR={k}/{n}={k/n:.3f} -> {args_cli.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
