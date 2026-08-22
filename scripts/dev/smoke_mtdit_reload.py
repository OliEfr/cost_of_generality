#!/usr/bin/env python
"""B1 reload smoke for the lerobot_policy_mtdit plugin (candidate B backport).

Mirrors the eval harness's loading pattern (src/cog/eval/rollout_eval.py:99-106):
`<PolicyClass>.from_pretrained(ckpt, cli_overrides=[DDIM, 10 steps])` +
`make_pre_post_processors(policy.config, pretrained_path=ckpt)`, then one
`select_action` on a dummy 20-env observation batch (the eval protocol's env count):
state (20,9), two (20,3,128,128) cameras, "task" = canonical string x20.
Gate: postprocessed action has shape (20, 7). Prints MTDIT_RELOAD_OK on success
(assert on the marker, never on the exit code).

Run:  PYTHONPATH=<repo>/src <cog_isaac>/bin/python scripts/dev/smoke_mtdit_reload.py [ckpt_dir]
Default ckpt: the 300-step smoke's checkpoints/000300/pretrained_model.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

MIN_FREE_GIB = 10
NUM_ENVS = 20
CANONICAL_TASK = "place the cup on the green target marker"


def assert_gpu_headroom() -> None:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
    )
    free_gib = int(out.strip().splitlines()[0]) / 1024
    assert free_gib >= MIN_FREE_GIB, f"only {free_gib:.1f} GiB free on GPU (< {MIN_FREE_GIB})"
    print(f"[reload] GPU headroom OK: {free_gib:.1f} GiB free")


def main() -> None:
    default_ckpt = (
        Path(__file__).resolve().parents[2]
        / "experiments/runs/smoke_mtdit_300/checkpoints/000300/pretrained_model"
    )
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else default_ckpt
    assert (ckpt / "model.safetensors").is_file(), f"no checkpoint at {ckpt}"

    assert_gpu_headroom()

    import torch

    from lerobot_policy_mtdit import MultiTaskDiTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    # Load with the eval-time inference overrides (DDPM-100 train -> DDIM-10 eval),
    # exactly like rollout_eval.py does for the diffusion cells.
    policy = MultiTaskDiTPolicy.from_pretrained(
        str(ckpt),
        cli_overrides=["--noise_scheduler_type=DDIM", "--num_inference_steps=10"],
    )
    assert policy.config.noise_scheduler_type == "DDIM", policy.config.noise_scheduler_type
    assert policy.objective.num_inference_steps == 10, policy.objective.num_inference_steps
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler

    assert isinstance(policy.objective.noise_scheduler, DDIMScheduler)
    print("[reload] checkpoint loaded with DDIM-10 overrides")

    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(ckpt))
    print(f"[reload] processors loaded: pre={len(pre.steps)} steps, post={len(post.steps)} steps")

    policy.to("cuda")
    policy.eval()
    policy.reset()

    # Dummy 20-env observation batch, raw (pre-pipeline) format: batched tensors + task
    # strings. AddBatchDimensionProcessorStep is a no-op on already-batched inputs.
    raw = {
        "observation.state": torch.rand(NUM_ENVS, 9),
        "observation.images.table_cam": torch.rand(NUM_ENVS, 3, 128, 128),
        "observation.images.wrist_cam": torch.rand(NUM_ENVS, 3, 128, 128),
        "task": [CANONICAL_TASK] * NUM_ENVS,
    }
    batch = pre(raw)
    assert "observation.language.tokens" in batch, sorted(batch.keys())
    assert batch["observation.language.tokens"].shape[0] == NUM_ENVS
    with torch.no_grad():
        action = policy.select_action(batch)
    action = post(action)
    assert action.shape == (NUM_ENVS, 7), f"action shape {tuple(action.shape)} != (20, 7)"
    assert torch.isfinite(action).all(), "non-finite actions"
    print(f"[reload] select_action OK: action shape {tuple(action.shape)}, device {action.device}")

    print("MTDIT_RELOAD_OK")


if __name__ == "__main__":
    main()
