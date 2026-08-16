# Frozen diffusion-policy hyperparameters (identical for EVERY cell).
# BATCH/LR are placeholders until the G5a A100 utilization smoke locks them
# (sqrt-LR scaling from lr=1e-4 @ batch 64). Everything else is FINAL.
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
export COG_DP_BATCH=64        # G5a: raise to 128/256 if A100 allows
export COG_DP_LR=1e-4         # G5a: scale by sqrt(batch/64)
