"""Batched rollout evaluation of a LeRobot diffusion policy in any of the study's task envs.

Frozen protocol (configs/eval_sets/protocol.json): per level, `batches` x
`num_envs` episodes; batch b resets the vectorized env with seed base_seed+b,
which deterministically reproduces the same initial conditions for every
checkpoint/cell (global-RNG event sampling; verified determinism at G4).
Success = the env's `success` termination fired at least once (latched);
failure = timeout first. DDIM num_inference_steps set explicitly at load
(post-load gotcha: DiffusionModel copies config at init).

--warmup_batches runs unscored batches first, in the same process (D31). The first batch
in a process is genuinely depressed, so under the uniform-slice protocol -- one scored
batch per process for every arm, flat or per-variant -- a warm-up is what makes cells
comparable. Default 0 reproduces the pre-2026-09-17 behaviour exactly.

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
    "--warmup_batches",
    type=int,
    default=0,
    help="unscored batches to run before the scored ones, in the SAME process. The first "
    "batch in a process is genuinely depressed (D31); under the uniform-slice protocol "
    "every scored batch is first-in-process unless this is set",
)
parser.add_argument(
    "--warmup_seed",
    type=int,
    default=4900,
    help="warm-up batch w resets with warmup_seed+w. Default 4900 keeps warm-up poses "
    "clear of the eval block (5000-5009) and of every generation seed block",
)
parser.add_argument(
    "--warmup_renders",
    type=int,
    default=0,
    help="extra env.sim.render() calls after each scored reset, before the first observation "
    "is read. D33: the renderer is one call behind the post-reset transform, so the first "
    "frame of a fresh process shows a stale wrist view (measured 105x the renderer's own "
    "noise floor; one extra call takes it to 1.6x, four to the floor). This is the cheap "
    "alternative to --warmup_batches, which costs a full batch -- 46% of an eval sweep",
)
parser.add_argument(
    "--warmup_max_steps",
    type=int,
    default=0,
    help="step cap for a warm-up batch; 0 = same as --max_steps. A shorter cap is cheaper "
    "and may suffice -- calibrate before relying on it (gate G6)",
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
import time
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


def drive_renderer(env, n):
    """Run the renderer n extra times and re-read the observation it feeds the policy.

    The sensors cache on a timestamp that render() does not advance, so they are forced; without
    that the recomputed observation is the same stale buffer.
    """
    if n <= 0:
        return None
    for _ in range(n):
        env.sim.render()
    for name in ("table_cam", "wrist_cam"):
        sensor = getattr(env.scene, "sensors", {}).get(name)
        if sensor is None:
            continue
        try:
            sensor.update(dt=0.0, force_recompute=True)
        except TypeError:
            sensor.update(0.0)
    return env.observation_manager.compute()


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


# Retired. A step window was the wrong shape of fix: the stale flag is latched for the whole
# episode, not for k steps, so any window simply moves the phantom latch to its far edge (measured:
# guard 1 -> earliest success 1, guard 10 -> earliest success 10, SR unchanged to three decimals).
# Kept as 0 so the key stays in every result and a reader can tell corrected files from the two
# generations of broken ones.
PHANTOM_GUARD_STEPS = 0
# Any success latching before this is physically impossible and means the guard has been outrun
# by a new failure mode. Loud, because this bug has now escaped twice by being silent.
IMPLAUSIBLE_SUCCESS_STEP = 100


def main():
    proto = json.load(open(args_cli.protocol))
    num_envs, batches, base_seed = proto["num_envs"], proto["batches"], proto["base_seed"]
    warmup_max_steps = args_cli.warmup_max_steps or args_cli.max_steps
    # Echo the warm-up into the RESULT's protocol block (never into protocol.json, which is
    # frozen): two results with different warm-up are not comparable, so a reader must be
    # able to tell them apart without consulting the job script.
    proto_out = dict(proto)
    proto_out["warmup"] = {
        "batches": args_cli.warmup_batches,
        "seed": args_cli.warmup_seed,
        "max_steps": warmup_max_steps if args_cli.warmup_batches else 0,
    }
    # Stamped into every result so a corrected number can never be silently compared against an
    # uncorrected one -- results written before 2026-09-19 have no such key and were scored with
    # a 1-step guard, which is the bug.
    proto_out["phantom_guard_steps"] = PHANTOM_GUARD_STEPS
    # The signal the SR was computed from. u200 results have no such key and were scored off the
    # sticky get_term() latch; u200g10 has guard_steps=10 and the same latch one window later.
    proto_out["success_signal"] = "terminated & get_term('success')"
    proto_out["warmup_renders"] = args_cli.warmup_renders

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

    # Warm-up (D31), unscored. The first batch in a process is genuinely depressed --
    # t1_L1_n100_s0 scores [0.50, 0.90, 0.95, 0.95, 1.00] across its five batches WITH the
    # t==0 phantom guard active, so this is not a scoring artefact (journal 2026-08-22).
    # The uniform-slice protocol runs one scored batch per process, which would leave every
    # episode first-in-process; these batches put the scored one in the same warm state in
    # every cell, flat and per-variant alike. Nothing here is recorded.
    for w in range(args_cli.warmup_batches):
        t_warm = time.time()
        obs, _ = env.reset(seed=args_cli.warmup_seed + w)
        policy.reset()
        warm_tasks = warm_embs = None
        if instr_strings:
            widx = [(w + i) % len(instr_strings) for i in range(num_envs)]
            warm_tasks = [instr_strings[j] for j in widx]
            if use_env_state:
                warm_embs = instr_embs[widx]
        warm_done = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        for t in range(warmup_max_steps):
            with torch.inference_mode():
                raw = obs_to_batch(obs, dev)
                if warm_tasks is not None:
                    raw["task"] = warm_tasks
                    if warm_embs is not None:
                        raw["observation.environment_state"] = warm_embs
                action = post(policy.select_action(pre(raw)))
            obs, _, terminated, truncated, _ = env.step(action.to(env.device))
            warm_done |= terminated | truncated
            if bool(warm_done.all()):
                break
        print(
            f"[eval] warmup {w+1}/{args_cli.warmup_batches} (unscored, seed "
            f"{args_cli.warmup_seed + w})  {time.time()-t_warm:.1f}s",
            flush=True,
        )

    outcomes = []
    t_start = time.time()
    for b in range(batches):
        t_batch = time.time()
        obs, _ = env.reset(seed=base_seed + b)
        obs = drive_renderer(env, args_cli.warmup_renders) or obs
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
        # When each episode's success first latched, recorded for EVERY task and not only when
        # --stages is on. This is the instrumentation that would have caught the 2026-09-19
        # carryover eight months earlier: the bug was visible only in T2's stage timestamps, on
        # three cells nobody re-read, and was invisible in T1/T3 because a bare success flag
        # cannot say WHEN. It costs one int per episode.
        t_first_succ = torch.full((num_envs,), -1, dtype=torch.long, device=env.device)
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
            # THE SUCCESS SIGNAL. `terminated` is the only fresh one: TerminationManager.compute()
            # clears _terminated_buf at the top of every call. get_term() reads _term_dones, which
            # compute() writes ONLY for rows where a term fired --
            #
            #     rows = value.nonzero(as_tuple=True)[0]
            #     if rows.numel() > 0:
            #         self._term_dones[rows] = False
            #         self._term_dones[rows, i] = True
            #
            # -- and never clears otherwise, while reset() does not touch it at all. So
            # get_term("success") is a STICKY, CROSS-EPISODE record of "the last reason this env's
            # episode ended", not a per-step signal: once an env succeeds it reads True forever,
            # through resets, in every later batch. That is the whole bug (D32), and it is why the
            # 2026-08-21 and the first 2026-09-19 fix both failed -- they suppressed the latched
            # flag for a window of steps, and it was still latched on the far side of the window.
            #
            # AND-ing with `terminated` is exact rather than defensive: at a step where terminated
            # is true, compute() has just overwritten _term_dones one-hot for exactly those rows,
            # so get_term names the term that fired THIS step; at any other step the conjunction is
            # false whatever the stale latch says. Generic across all three tasks, and it needs no
            # duplicate copy of the success predicate.
            succ_now = terminated & env.termination_manager.get_term("success")
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
            t_first_succ[succ_now & ~success & ~finished] = t
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
                    "t_first_success": int(t_first_succ[i]),
                    **({"instruction_index": idx_list[i]} if idx_list else {}),
                }
                for i in range(num_envs)
            )
        else:
            outcomes.extend(
                {"batch": b, "env": i, "success": bool(success[i]),
                 "t_first_success": int(t_first_succ[i]),
                 **({"instruction_index": idx_list[i]} if idx_list else {})}
                for i in range(num_envs)
            )
        sr_so_far = sum(o["success"] for o in outcomes) / len(outcomes)
        print(
            f"[eval] batch {b+1}/{batches}  running SR={sr_so_far:.3f}  "
            f"batch {time.time()-t_batch:.1f}s  total {time.time()-t_start:.1f}s",
            flush=True,
        )

    n = len(outcomes)
    k = sum(o["success"] for o in outcomes)
    result = {
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "num_inference_steps": args_cli.num_inference_steps,
        "protocol": proto_out,
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
    # A success inside the implausible window means the guard has been outrun. Loud and recorded,
    # because this bug has now escaped twice by failing silently.
    early = [o["t_first_success"] for o in outcomes
             if o["success"] and 0 <= o["t_first_success"] < IMPLAUSIBLE_SUCCESS_STEP]
    result["earliest_success_step"] = min(
        [o["t_first_success"] for o in outcomes if o["success"] and o["t_first_success"] >= 0],
        default=-1,
    )
    if early:
        result["PHANTOM_SUSPECT"] = len(early)
        print(f"[eval] *** WARNING: {len(early)} success(es) latched before step "
              f"{IMPLAUSIBLE_SUCCESS_STEP} (steps {sorted(set(early))[:10]}) -- the phantom guard "
              f"of {PHANTOM_GUARD_STEPS} steps did not cover the carryover. This result is "
              f"SUSPECT. ***", flush=True)
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
