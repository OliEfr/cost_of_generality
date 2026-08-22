# Candidate A (language-conditioning investigation, D30): the frozen diffusion recipe,
# UNCHANGED. This file exists to document that language conditioning in candidate A is
# entirely DATASET-DRIVEN: the *_i20 datasets carry a per-frame
# observation.environment_state (512-d frozen CLIP text embedding, unit-norm), lerobot
# 0.4.4 auto-populates cfg.input_features from ds_meta.features (factory.py:458-472),
# and DiffusionPolicy natively appends any ENV feature to global_cond
# (modeling_diffusion.py:172-186, 246-282). ENV features get IDENTITY normalization
# (no "ENV" key in normalization_mapping), so the embedding reaches the U-Net unchanged.
#
# ZERO flag changes vs the frozen study config -- sourcing it is the whole config.
# The only training-time difference vs a study cell is the dataset root (*_i20).
source "$(dirname "${BASH_SOURCE[0]}")/diffusion_base.sh"
