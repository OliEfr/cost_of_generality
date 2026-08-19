# Frozen diffusion-policy hyperparameters (identical for EVERY cell).
# NOTE: source + expand $COG_DP_FLAGS under BASH (sbatch default). zsh does not
# word-split unquoted vars — local zsh callers must use `bash -c` or ${=COG_DP_FLAGS}.
#
# FROZEN 2026-08-19 by G5a (CLAUDE.md rule 7). Batch and LR are no longer placeholders:
# they were decided by measurement on one A100, see docs/journal.md 2026-08-19 18:05.
#
# Draccus resolves a repeated flag LAST-WINS, so anything appended after ${COG_DP_FLAGS}
# silently overrides what is set here. That already caused one invalid measurement
# (--num_workers). Callers that must override a value have to STRIP it from this list first.
export COG_DP_FLAGS="
  --policy.type=diffusion
  --policy.push_to_hub=false
  --policy.device=cuda
  --policy.n_obs_steps=2
  --policy.horizon=16
  --policy.n_action_steps=8
  --policy.drop_n_last_frames=7
  --policy.vision_backbone=resnet18
  --policy.crop_shape=[112,112]
  --policy.crop_is_random=true
  --policy.use_group_norm=true
  --policy.spatial_softmax_num_keypoints=32
  --policy.use_separate_rgb_encoder_per_camera=false
  --policy.do_mask_loss_for_padding=false
  --policy.noise_scheduler_type=DDPM
  --policy.num_train_timesteps=100
  --policy.beta_schedule=squaredcos_cap_v2
  --policy.prediction_type=epsilon
  --steps=80000
  --save_freq=20000
  --log_freq=200
  --eval_freq=0
  --num_workers=8
  --seed=0
"
# The last two policy flags above are pinned at lerobot 0.4.4's OWN defaults
# (use_separate_rgb_encoder_per_camera=False, do_mask_loss_for_padding=False), so they change
# nothing today. They are written out because lerobot 0.6.0 silently flipped five diffusion
# defaults -- horizon 16->64, n_action_steps 8->32, use_group_norm True->False,
# pretrained_backbone_weights None->ImageNet, use_separate_rgb_encoder_per_camera False->True.
# The first three were already pinned here; use_separate_rgb_encoder_per_camera was not.
# (do_mask_loss_for_padding is NOT one of the five -- it is False in every release from 0.4.4
# to 0.6.1; it is pinned here purely defensively.) Any future version bump would otherwise have
# changed the ARCHITECTURE mid-study with no error and no log line.
# Still unpinned deliberately: `pretrained_backbone_weights` (None at 0.4.4). Passing None
# through the CLI risks draccus decoding the string "null", which would be worse than relying
# on the default -- so it is instead ASSERTED against the calibration run's saved config.json.
# That is safe for a further reason, verified 2026-08-19: the three encoder flips are COUPLED.
# modeling_diffusion.py raises "You can't replace BatchNorm in a pretrained model without
# ruining the weights!" when use_group_norm=True and pretrained_backbone_weights is set, which
# is exactly why 0.6.0 had to flip use_group_norm to False to default the ImageNet weights on.
# Since we pin use_group_norm=true, a bump to 0.6.x FAILS LOUDLY at model construction instead
# of silently training a different encoder. Full 29-default cross-version table: docs/PINS.md.
# Also verified: lerobot 0.5.0/0.5.1 are identical to 0.4.4 on every diffusion default, so the
# rejected 0.5 bump would have changed no architecture either.

# --- Frozen by G5a on measured evidence (job 52878355, 200 steps/arm, L0/N=25, one A100) ---
# batch:  64 -> 0.962 steps/s | 128 -> 0.862 | 256 -> 0.385   (higher batch is SLOWER per step)
# VRAM:   13.5 / 14.5 / 17.1 GiB of 64 GiB  -> memory is never the binding constraint
# GPU util: 0 % median at every batch size  -> the loop is dataloader-bound, not compute-bound
# With the protocol fixed at 80k STEPS, the smallest sensible batch minimises wall-clock, and
# a larger batch buys no throughput because samples/s is flat. So the plan's "scale batch up
# as far as the A100 allows" is inapplicable to this workload -- see D23.
export COG_DP_BATCH=64
export COG_DP_LR=1e-4         # = DiffusionConfig.optimizer_lr default; no sqrt scaling needed
                              # at batch 64. NB the knob is --policy.optimizer_lr, NOT
                              # --optimizer.lr, which validate() silently overwrites.

# Video decode backend. torchcodec is ~31x faster per call than pyav here because it keeps a
# module-level VideoDecoderCache, while 0.4.4's pyav path rebuilds a reader for EVERY frame
# fetch on an 82,916-frame container (~32 ms/open x 128 fetches per step). Requires
# conda-forge ffmpeg + an ABI-matched torchcodec + LD_LIBRARY_PATH=$CONDA_PREFIX/lib.
# Set to pyav to fall back; the two backends are verified to return identical frames.
export COG_VIDEO_BACKEND="${COG_VIDEO_BACKEND:-torchcodec}"
