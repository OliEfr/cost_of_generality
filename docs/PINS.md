# Version pins

| Component | Pin | Why | Status |
|---|---|---|---|
| Isaac Sim | 5.1.0 (pip, pypi.nvidia.com) | Combo used by official LeRobot x IsaacLab-Arena integration; works with driver 580.173.02; 6.x needs >=595 | installed+verified |
| Isaac Lab | v2.3.0 tag (commit 3c6e67b) | Pairs with Isaac Sim 5.1; has isaaclab_mimic; 3.0 is beta | installed+verified |
| Python (sim env) | 3.11 | Isaac Sim 5.1 requirement | -- |
| torch (sim env) | 2.7.0 cu128 | Isaac Lab 2.3 recommendation | -- |
| LeRobot | ==0.4.4 | newest release with requires_python >=3.10 (0.5.0+ needs >=3.12); has LeRobotDataset v3 + diffusion + lerobot-train; single version for train (cluster) AND eval (inside cog_isaac py3.11). Coexistence with isaacsim verified (torch untouched). | verified 2026-08-16 |
| numpy | ==1.26.4 | numpy 2.4.6 segfaults Kit (pinocchio ABI via dex_retargeting import); see decisions.md D3 | pinned 2026-08-16 |
| transformers | <5 (4.57.6) | transformers 5.x needs huggingface-hub>=1.5, lerobot pins <0.36; isaaclab dep only | pinned 2026-08-16 |
| torchcodec | not installed locally; cluster: try ==0.4.0 else pyav backend | torch-2.7 pairing per compat matrix (verify at G5a) | open |
| seed | 0 everywhere | single-seed directive | -- |

Exact resolved package lists: `docs/pins/` (conda env exports, added after G1).
