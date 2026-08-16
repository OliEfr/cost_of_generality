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

## 2026-08-16 (evening) — frames QA round 2: marker clipping + joint-noise finding

- **Goal marker was clipped by the table cam, not z-fighting (round 2).** After the
  z-fight fix, L0/L1/L2 renders still showed a "half-disk" marker. Pixel analysis of
  `ops/qa/frames_*.png` showed the green blob's bbox hits x=127 in every frame: the
  marker extends past the RIGHT image edge. Root cause: stock stack-task table cam
  (pos y=0, aperture 20.955 / ~47 deg) is centered on y=0, but our workspace is
  asymmetric: goal-marker rim reaches y=+0.36 (L2 max y 0.30 + r 0.06), cup rim
  y=-0.28. Fix: cam shifted +4 cm in y and horizontal_aperture 20.955 -> 24.0
  (~53 deg). Covers y in [-0.315, +0.395] with ~3.5 cm margin both sides. Changed
  BEFORE any demo is recorded, so no data has mixed intrinsics.
- **L0 wrist views differ across resets — explained, kept (-> decisions.md D8).**
  `randomize_franka_joint_state` (stock stack event, Gaussian std 0.02 rad on reset)
  perturbs the arm start pose at every level incl. L0. Kept deliberately: without it
  L0 demos would be bit-identical and SR-vs-N degenerate; it applies uniformly to all
  levels so it cancels in cost ratios. L0 = "fixed task + natural motor noise".
- QA sweep note: frames_qa levels each hit the 480 s timeout AFTER writing their PNG
  (Kit hangs on close; exit 137 is benign) but `timeout -s KILL` on isaaclab.sh
  orphans the python child holding ~4.8 GB VRAM. Killing orphans was blocked by the
  session policy; they must be cleaned up manually if they accumulate
  (`pgrep -af frames_qa`). Worst-case VRAM still fits under 24 GB for this sweep.

## 2026-08-16 — adversarial review round 2 (workflow wf_5b659661-27b): 2 minor findings

6 agents (3 reviewers vs IsaacLab v2.3.0 source + per-finding verification), no
blockers/majors. Confirmed minors, both in source-demo recording:
1. **Parallel overshoot** — with num_envs=8 several envs can succeed in the same
   step; RecorderManager exports all of them before the script's `exported>=target`
   check, so the HDF5 can hold a few more than --num_demos.
2. **Batch-size replay divergence** — sources recorded in 8-env batched PhysX are
   replayed open-loop by annotate_demos.py at num_envs=1; PhysX is not bit-identical
   across batch sizes, so annotate's success re-check may drop episodes (loud, not
   silent). Upstream mimic workflow records sources single-env.

**Policy adopted:** final source demos are recorded with `--num_envs 1` and
over-recorded (~15 for a 10-target), keeping the episodes that survive annotation
(matches upstream guidance; kills both findings). 8-env recording remains for
expert-SR gate measurements only (throwaway files).

## 2026-08-16 — G2 debug: expert stalled at 47% SR on L0 — place-height bug found & fixed

First live run of the SM expert (8 envs, L0): expert_SR=0.47 (20/43). Diagnostic
(`scripts/dev/sm_diag.py`, logs per-episode final SM state + lift/place outcome)
showed a SINGLE failure mode: every failure stalled in LOWER with |ee-des|=0.013 vs
the 0.012 near() gate; every success fired in RELEASE at ~189 steps with 0.009 error.

**Root cause:** the place target `goal_z + half_height + 0.006` computes the desired
CUP-CENTER height but was fed to the SM as the TCP target. The TCP grasps the cup
grasp_z_offset (1.5 cm) ABOVE its center, so the commanded TCP pose pushed the cup
~6 mm into the table — unreachable, steady-state error ~13 mm, LOWER never advanced.
Successes were finger-slip luck. **Fix:** TCP place target = goal_z + half_height +
grasp_z_offset + 0.006 (recorder + sm_diag). The tainted g2_sr_L0.hdf5 was deleted
(regenerable throwaway; final sources come from L2 single-env per D9).

Lesson recorded: SM targets are TCP poses — every object-height computation must add
the TCP-to-object offset of the current grasp.

**Retest after fix:** L0 diag 24/24 successes, all reaching RELEASE at ~189 steps
(9.5 s sim), tracking error ~9 mm — well inside the 12 mm gate. L1/L2 diags queued.

**G2 SR criterion MET (2026-08-16):** post-fix expert diag — L0 24/24, L1 32/32,
L2 32/32 (88/88 overall; gate needs >=90%). All successes reach RELEASE at
~190 steps / ~9.5 s sim. G2b started: 15 L2 source demos single-env (D9), then
auto-annotate replay re-check.

**G2 PASSED (2026-08-16):** G2b recorded 15/15 L2 source demos single-env
(expert_SR=1.00, data/hdf5/L2_source.hdf5) and annotate --auto exported 15/15
annotated episodes (L2_source_annotated.hdf5) — zero replay attrition, validating
D9's single-env policy. Gate criteria: expert SR >=90% on L0-L2 (88/88 diag) +
sources replay correctly (15/15). Next: P3 Mimic generation smoke on L0.

**P3 state-generation smoke PASSED (2026-08-16):** Mimic generated 11/12 successful
L0 demos (91.7% gen SR, G3 floor is 30%) from the 15 annotated L2 sources, stopping
at the 10-success target. All subclass contracts exercised live (datagen pool load,
subtask transforms, target_eef_pose_to_action). Visuomotor generation smoke queued.

**P3 visuomotor generation smoke PASSED (2026-08-16):** 10/12 successes (83.3% gen
SR). Output HDF5 carries both 128x128x3 uint8 camera streams (live pixels, new
framing) + full proprio/object obs; ~8.8 MB/demo => ~3.5 GB per 400-demo level.
Failed episodes exported separately (*_failed.hdf5) — useful for QA.

**Full L0 generation LAUNCHED 18:40 (tmux cog_gen_L0):** 400-success target, 8 envs,
log ops/gen/L0.log. Estimated ~3-4 h from smoke throughput. Converter smoke on the
10-demo file running concurrently on CPU (G4 prep).

**L0 DATASET COMPLETE (2026-08-16 19:02):** 400/463 successes = 86.4% generation SR,
3.5 GB (L0.hdf5) + 550 MB failed pool, in 22 min wall (8 envs) — startup dominated
the smoke, so real throughput is ~20x my estimate; full 4-level datagen fits in ~2 h,
not days. Converter smoke: 10 episodes -> valid LeRobotDataset v3 tree
(videos/meta/manifest). L1+L2 generation wave launched 22:31 (tmux cog_gen_L1L2).
Gen-SR-per-level so far: L0 86.4%.

**Converter validation PASSED (2026-08-16):** VALIDATE_OK on the smoke conversion —
10 eps / 2064 frames, max mean |pixel err| 0.0166 (< 0.03 codec tolerance), action
ranges identical to HDF5. Full L0 conversion (400 eps) started on CPU concurrently
with the L1/L2 generation wave.

**L0 LeRobot dataset VALIDATED (2026-08-16):** 400 eps / 82,916 frames, pixel err
0.0151 < 0.03, VALIDATE_OK. 81 MB after h264 (~40x vs HDF5) — cluster sync trivial.

