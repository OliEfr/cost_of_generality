# Version pins

| Component | Pin | Why | Status |
|---|---|---|---|
| Isaac Sim | 5.1.0 (pip, pypi.nvidia.com) | Combo used by official LeRobot x IsaacLab-Arena integration; works with driver 580.173.02; 6.x needs >=595 | installing |
| Isaac Lab | v2.3.0 tag | Pairs with Isaac Sim 5.1; has isaaclab_mimic; 3.0 is beta | cloning |
| Python (sim env) | 3.11 | Isaac Sim 5.1 requirement | -- |
| torch (sim env) | 2.7.0 cu128 | Isaac Lab 2.3 recommendation | -- |
| LeRobot | ==0.4.4 | newest release with requires_python >=3.10 (0.5.0+ needs >=3.12); has LeRobotDataset v3 + diffusion + lerobot-train; single version for train (cluster) AND eval (inside cog_isaac py3.11). Install-compat check pending after isaacsim install. | decided (G1b) 2026-08-16 |
| seed | 0 everywhere | single-seed directive | -- |

Exact resolved package lists: `docs/pins/` (conda env exports, added after G1).
