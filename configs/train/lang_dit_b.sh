# Candidate B (language-conditioned) hyperparameters: multi_task_dit backported as the
# in-repo plugin src/lerobot_policy_mtdit (lerobot 0.5.2 checkout commit fc6c94c, run on
# pinned 0.4.4). Provenance: this file mirrors configs/train/diffusion_base.sh in style
# but is a SEPARATE, non-frozen config for the language-conditioning investigation only
# (plan: add-another-disturbance-dimension; the frozen diffusion_base.sh is untouched).
#
# NOTE: source + expand $COG_DIT_FLAGS under BASH (sbatch default). zsh does not
# word-split unquoted vars — local zsh callers must use `bash -c` or ${=COG_DIT_FLAGS}.
# Draccus resolves a repeated flag LAST-WINS: anything appended after ${COG_DIT_FLAGS}
# silently overrides what is set here.
#
# The plugin must be discoverable on BOTH fresh and resume invocations:
#   PYTHONPATH=<repo>/src  +  --policy.discover_packages_path=lerobot_policy_mtdit
# (the sbatch passes the discover flag OUTSIDE these flags because the resume branch is
# config_path-only and still needs it).
#
# Field names verified against src/lerobot_policy_mtdit/configuration_multi_task_dit.py.
#
# CRITICAL image geometry: CLIP ViT-B/16 has fixed 224x224 position embeddings. Our
# frames are 128x128, and MultiTaskDiTConfig.validate_features SILENTLY DISABLES
# cropping when crop > effective image size — the raw 128px input then crashes/misbehaves
# at the first CLIP forward. So resize >= crop = 224 is mandatory:
# resize [256,256] + random crop [224,224] keeps the study baseline's 0.875 crop ratio.
#
# Policy-tuned preset kept deliberately (lr 2e-5, NOT the study's 1e-4; hidden 512 / 6
# layers / 8 heads / RoPE / DDPM-100; vision tower fine-tuned at 0.1x lr; ONE shared CLIP
# encoder for both cameras — a deliberate asymmetry vs D26's per-camera ResNets, flagged
# in the report). horizon=20 / n_action_steps=16 / n_obs_steps=2 adapts the policy's
# 30 Hz defaults (32/24) to our 20 Hz control rate. drop_n_last_frames=3 is written out
# defensively but equals the config's own auto-calc (horizon - n_action_steps -
# n_obs_steps + 1 = 20-16-2+1).
export COG_DIT_FLAGS="
  --policy.type=multi_task_dit
  --policy.push_to_hub=false
  --policy.device=cuda
  --policy.n_obs_steps=2
  --policy.horizon=20
  --policy.n_action_steps=16
  --policy.drop_n_last_frames=3
  --policy.objective=diffusion
  --policy.noise_scheduler_type=DDPM
  --policy.num_train_timesteps=100
  --policy.beta_schedule=squaredcos_cap_v2
  --policy.prediction_type=epsilon
  --policy.hidden_dim=512
  --policy.num_layers=6
  --policy.num_heads=8
  --policy.use_rope=true
  --policy.dropout=0.1
  --policy.image_resize_shape=[256,256]
  --policy.image_crop_shape=[224,224]
  --policy.image_crop_is_random=true
  --policy.vision_encoder_name=openai/clip-vit-base-patch16
  --policy.text_encoder_name=openai/clip-vit-base-patch16
  --policy.use_separate_rgb_encoder_per_camera=false
  --policy.vision_encoder_lr_multiplier=0.1
  --policy.optimizer_lr=2e-5
  --policy.do_mask_loss_for_padding=false
  --steps=80000
  --save_freq=20000
  --log_freq=200
  --eval_freq=0
  --num_workers=8
  --seed=0
"

# Batch is passed separately (like COG_DP_BATCH). 64 is the starting point; the B2
# cluster dbg smoke measures 64/128/192 before the full run — record in docs/timings.md.
export COG_DIT_BATCH="${COG_DIT_BATCH:-64}"

# Video decode backend: same rationale as diffusion_base.sh (torchcodec keeps a decoder
# cache; local torchcodec is broken -> local smokes use pyav explicitly).
export COG_VIDEO_BACKEND="${COG_VIDEO_BACKEND:-torchcodec}"
