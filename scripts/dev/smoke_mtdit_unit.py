#!/usr/bin/env python
"""B1 unit smoke for the lerobot_policy_mtdit plugin (candidate B backport).

Proves, on the pinned lerobot 0.4.4 in cog_isaac, that:
  1. importing the plugin registers "multi_task_dit" with PreTrainedConfig;
  2. a config with the study's features (state (9,), 2x 128x128 cams, action (7,)) and
     the mandatory resize-256/crop-224 geometry KEEPS its crop (validate_features
     silently disables cropping when crop > effective size — the CLIP 224 trap);
  3. the policy instantiates (CLIP ViT-B/16 text+vision from the local HF cache) and the
     conditioning vector provably contains the 512-d text projection;
  4. a training forward+backward on a random batch with pre-tokenized language keys
     yields a finite loss (tokenization is processor-side in real training —
     TokenizerProcessorStep reads batch["task"] — so the unit test feeds
     observation.language.{tokens,attention_mask} directly).

Run:  PYTHONPATH=<repo>/src <cog_isaac>/bin/python scripts/dev/smoke_mtdit_unit.py
Prints MTDIT_UNIT_OK on success (assert on the marker, never on the exit code).
"""

import subprocess
import sys
import time
from pathlib import Path

# Make the run robust to a missing PYTHONPATH: the plugin lives in <repo>/src.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

MIN_FREE_GIB = 10  # GPU etiquette: the foreign eval job must keep its headroom.


def assert_gpu_headroom() -> None:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True
    )
    free_gib = int(out.strip().splitlines()[0]) / 1024
    assert free_gib >= MIN_FREE_GIB, f"only {free_gib:.1f} GiB free on GPU (< {MIN_FREE_GIB})"
    print(f"[smoke] GPU headroom OK: {free_gib:.1f} GiB free")


def load_tokenizer(name: str, attempts: int = 6, wait_s: int = 30):
    """The parent session may still hold a HF download lock on the CLIP cache: retry."""
    from transformers import AutoTokenizer

    for i in range(attempts):
        try:
            return AutoTokenizer.from_pretrained(name)
        except Exception as e:  # noqa: BLE001 - lock/partial-cache errors vary by hf_hub version
            if i == attempts - 1:
                raise
            print(f"[smoke] tokenizer load failed ({e}); retry {i + 1}/{attempts} in {wait_s}s")
            time.sleep(wait_s)


def main() -> None:
    assert_gpu_headroom()

    import torch

    import lerobot_policy_mtdit
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.utils.constants import (
        ACTION,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    # 1. Registration
    cfg_cls = PreTrainedConfig.get_choice_class("multi_task_dit")
    assert cfg_cls is lerobot_policy_mtdit.MultiTaskDiTConfig, cfg_cls
    print(f"[smoke] registered: {cfg_cls.__module__}.{cfg_cls.__name__}")

    # 2. Config with study features + resize/crop geometry
    config = cfg_cls(
        n_obs_steps=2,
        horizon=20,
        n_action_steps=16,
        image_resize_shape=(256, 256),
        image_crop_shape=(224, 224),
        image_crop_is_random=True,
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(9,)),
            "observation.images.table_cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
            "observation.images.wrist_cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        device="cuda",
        push_to_hub=False,
    )
    assert config.drop_n_last_frames == 3, config.drop_n_last_frames  # 20 - 16 - 2 + 1

    # 3. Policy instantiation. validate_features runs inside __init__ and silently sets
    #    image_crop_shape=None when crop > effective size — assert the crop SURVIVED.
    policy = lerobot_policy_mtdit.MultiTaskDiTPolicy(config).to("cuda")
    assert config.image_crop_shape == (224, 224), (
        f"crop was silently disabled (image_crop_shape={config.image_crop_shape}); "
        "128px frames would hit CLIP's fixed 224x224 pos-embeds raw"
    )
    # Decisive conditioning check: (state 9 + vision 768*2 cams + text 512) * n_obs 2.
    expected_cond = (9 + 768 * 2 + 512) * 2
    got_cond = policy.observation_encoder.conditioning_dim
    assert got_cond == expected_cond, f"conditioning_dim {got_cond} != {expected_cond}"
    n_params = sum(p.numel() for p in policy.parameters())
    n_train = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[smoke] policy built: {n_params / 1e6:.1f}M params ({n_train / 1e6:.1f}M trainable), "
          f"conditioning_dim={got_cond} (includes 512-d text projection)")

    # 4. Training forward on a random batch with pre-tokenized language
    tokenizer = load_tokenizer(config.text_encoder_name)
    bsz = 4
    enc = tokenizer(
        ["place the cup on the green target marker"] * bsz,
        padding=config.tokenizer_padding,
        max_length=config.tokenizer_max_length,
        truncation=config.tokenizer_truncation,
        return_tensors="pt",
    )
    dev = torch.device("cuda")
    batch = {
        OBS_STATE: torch.randn(bsz, 2, 9, device=dev),
        "observation.images.table_cam": torch.rand(bsz, 2, 3, 128, 128, device=dev),
        "observation.images.wrist_cam": torch.rand(bsz, 2, 3, 128, 128, device=dev),
        ACTION: torch.randn(bsz, 20, 7, device=dev),
        OBS_LANGUAGE_TOKENS: enc["input_ids"].to(dev),
        OBS_LANGUAGE_ATTENTION_MASK: enc["attention_mask"].to(dev),
    }
    policy.train()
    loss, _ = policy.forward(batch)
    assert torch.isfinite(loss), f"non-finite loss: {loss}"
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf"))
    assert torch.isfinite(grad_norm), f"non-finite grad norm: {grad_norm}"
    # Frozen text tower must have no grads; its learnable projection must have them.
    assert policy.observation_encoder.text_encoder.projection.weight.grad is not None
    assert all(p.grad is None for p in policy.observation_encoder.text_encoder.text_encoder.parameters())
    print(f"[smoke] training forward: loss={loss.item():.4f}, grad_norm={grad_norm.item():.2f}")

    print("MTDIT_UNIT_OK")


if __name__ == "__main__":
    main()
