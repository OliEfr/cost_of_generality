# Lab journal

## 2026-08-16
- Plan approved (docs/PLAN.md). Cluster Slurm association still missing (user
  emailed CINECA support; watcher installed). Local phases P1-P4 started.
- Verified: 559 GB free disk, ~21 GB GPU headroom, foreign eval job alive (untouched).
- Repo created; P1 env install + IsaacLab clone kicked off in background.

## 2026-08-16 (afternoon) — P1 done, P2 env verified, pipeline tooling written
- **P1/G1 PASSED.** cog_isaac env: Isaac Sim 5.1.0 + Isaac Lab v2.3.0 + lerobot 0.4.4
  coexist (torch 2.7.0+cu128 preserved). Headless camera rendering verified 10 min
  crash-free on the stack visuomotor env.
- **Install fixes (details in decisions.md D3):** numpy pinned 1.26.4 (numpy 2.4.6
  segfaulted Kit via pinocchio ABI), transformers<5 (hub conflict), EULA env vars
  persisted on the conda env. Residual pip-check conflicts are in unused corners
  (rerun-sdk, stable-baselines3, torchaudio, packaging/click exact pins) — accepted.
- **cup_place package written + smoke-tested:** 52 gym IDs (state/visuomotor/mimic
  x L0,L1,L2,L3v00-09); obs dict incl. eef_pos/eef_quat + grasp_1 subtask signal;
  both cams 128x128 uint8 range 0-255. One fix: ActionsCfg needed type annotations.
- **Frames QA:** L0 fixed-pose and L1 cup-randomization verified visually; framing OK
  in both cams. Found+fixed: goal disk z-fighting (half-disk render) -> thicker disk,
  center z=0.003. L2/L3 renders still running.
- **Manual contract verification** (replacing the review workflow, which died when all
  3 subagents hit the session usage limit; subagents unavailable until ~17:30):
  exported_successful_episode_count is a @property (int) ✓; mimic IK-Rel delta formula
  matches our convert_abs_to_rel_actions (target-curr pos, matrix->axis-angle rot) ✓;
  RecorderManager.record_pre_reset auto-reads the "success" term + auto-exports ✓;
  object_grasped/event signatures match ✓.
- **Kit runtime gotchas (cost ~30 min each, avoid):** (1) python stdout is block-
  buffered and LOST on Kit's os._exit when redirected -> always PYTHONUNBUFFERED=1
  and/or write JSON result files; (2) a second gym.make in one Kit session hangs ->
  one env per app launch, loop in bash.
- **Converter/eval tooling committed:** hdf5_to_lerobot (nested-N via train-time
  episode subselection), validate_dataset, frozen DP config (80k steps), rollout_eval
  with frozen protocol (100 eps = 5x20 envs, seeds 5000+b).
- Cluster: Slurm association STILL missing (user mailed support); watchdog polls hourly.
- NEXT: G2 expert run (record_source_demos on L0), then annotate/generate smoke.
