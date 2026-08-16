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

**Gotcha (2026-08-16): zsh vs $COG_DP_FLAGS.** The session shell is zsh, which does
not word-split unquoted variables: `lerobot-train $COG_DP_FLAGS` passed the whole
multi-line flag block as ONE argument ("unrecognized arguments"). Fix: run train
invocations under `bash -c` locally (sbatch is bash, unaffected). Noted in
configs/train/diffusion_base.sh header. G4 train smoke relaunched under bash.

**Gotcha (2026-08-16): torchcodec unusable locally.** lerobot's default video
backend torchcodec fails at first batch (libavutil.so.57 missing — no system FFmpeg
shared libs; conda ffmpeg would endanger the numpy==1.26.4 pin). Fix:
`--dataset.video_backend=pyav` (pyav 15.1.0 bundles its own ffmpeg). Applied to the
G4 smoke; PINS.md updated; cluster backend decided at G5a by throughput.

**G4 train smoke RUNNING (2026-08-16 22:58):** diffusion policy training live on
L0/N=25 via --dataset.episodes (pyav backend), loss 0.579 @ step 200. Epoch math
confirms subselection empirically: 12.8k samples / 5.2k subset frames = epoch 2.47
as logged. ~2.8 steps/s data-bound while sharing the box with L2 generation
(updt_s 0.123 vs data_s 0.262) — pyav decode throughput is a G5a consideration.

**G4 train leg DONE / L1+L2 datasets DONE (2026-08-16 23:27):** 5k-step smoke
finished exit 0, loss 0.579 -> 0.098, checkpoint saved (~30 min sharing the GPU with
generation). L1 400 demos (85.8% gen SR), L2 400 demos (85.5%). Launched in
parallel: L3 wave (tmux cog_gen_L3, 10 sub-variants x 40), G4 eval smoke (20 eps,
reduced 2x10 smoke protocol in ops/ — frozen eval sets untouched), L1/L2
conversion+validation on CPU. Gen-SR per level so far: L0 86.4, L1 85.8, L2 85.5.

**G4 PASSED (2026-08-16 23:29):** eval smoke SR=16/20=0.80 on L0 with the 5k-step /
N=25 policy (DDIM-10, seeded batches 5000-5001). Entire pipeline validated:
env -> expert -> mimic gen -> conversion -> episodes= training -> checkpoint reload
-> batched seeded eval -> JSON. Determinism check (VERIFY c) running: seeded-reset
snapshot diff, within- and cross-process. Registry row added for the smoke run.

**VERIFY (c) CLOSED (2026-08-16):** seeded-reset determinism confirmed within- and
cross-process (identical cup/goal/joint snapshots for seed 5000 on L2). All P4
VERIFY items now closed; local pipeline fully proven. Remaining before the matrix:
L3 wave + conversions (running), G5 cluster bring-up (blocked on Slurm association).


**L3 WAVE DONE (2026-08-16 23:52):** all 10 variants exit 0, wall 23:27:05->23:52:39
= 25 min 34 s (~2.6 min/variant — 65-70 min estimate was ~3x too high; warm Kit
start is ~1.5-2 min, not ~4; timings.md corrected). Per-variant gen SR: v00-v04
(small cylinder x 5 colors) each 40/45 = 88.9%; v05-v09 (large cylinder x 5 colors)
each 40/46 = 87.0%. Identical attempt counts within each size group => generation
outcome depends only on geometry, not color — expected (color is render-only) and a
free determinism sanity check. Overall L3: 400/455 = 87.9% gen SR.
Gen-SR-per-level finding now complete: L0 86.4 / L1 85.8 / L2 85.5 / L3 87.9 —
essentially flat across the ladder; Mimic generation does not get harder with our
randomization ranges (worth a sentence in the report; the data-cost curves cannot be
explained by generation-side attrition). L3 conversion (10-file interleaved merge,
shuffle_seed 0) + validation launched. L1 converted+validated (VALIDATE_OK); L2
conversion running. Hourly fallback cron armed per user request (checks tmux/logs/
GPU/disk/G0/cert) in addition to event watchers.

**L1+L2 CONVERSION CHAIN DONE (2026-08-17 ~00:0x):** L2 converted (75 MB) and
validated — VALIDATE_OK. LeRobot datasets now green for L0/L1/L2 (400 eps each,
~75-81 MB); L3 interleaved-merge conversion in flight.

**GOTCHA — glob pulled failed demos into L3 (2026-08-17 ~00:20):** first L3
conversion used `--input data/hdf5/L3v0*.hdf5`, which also matches
`L3v0X_failed.hdf5` -> dataset had 455 eps (400 successes + 55 fails) yet
validation printed VALIDATE_OK (validator checks integrity, not provenance).
Caught only via the episodes=455 count. Tainted data/lerobot/L3 deleted,
reconverted from the explicit success-only list. Hardened: converter now refuses
`*_failed*` inputs without --allow_failed; validator gained --expect_episodes.
Rule for future agents: NEVER glob HDF5 inputs; RecorderManager writes
`<name>_failed.hdf5` next to every `<name>.hdf5`. L0/L1/L2 verified clean
(info.json total_episodes=400 each).

**L3 REBUILT CLEAN + ALL CONVERSIONS GREEN (2026-08-17 ~00:40):** L3 reconverted
from success-only inputs: 400 eps / 74,740 frames, pixel err 0.0160, VALIDATE_OK
incl. new --expect_episodes 400 guard. LeRobot datasets L0-L3 all validated
(400 eps each). Remaining for G3: dataset QA (visual grids, action ranges,
coverage) + per-level eval-set freeze.

**G3 QA PASS (2026-08-17 ~01:10):** scripts/dev/dataset_qa.py over all 4 levels:
coverage matches spec exactly (L1 cup span 29.9x39.9 cm vs 30x40; L2 goal
19.8x19.9 vs 20x20; L0 all fixed; L3 slightly narrower — 400 draws via 10
sub-runs). Final placement err max 3.71 cm vs 5 cm gate, all 1600 eps. Action
ranges sane (yaw deltas grow with yaw randomization as expected). Visual grids +
coverage scatters in ops/qa/. Two QA gotchas: (a) generator-exhaustion bug in my
own QA script (list() over a yielding with-block closes the h5 handles); (b) L3
grid appeared to show missing cups — full-sweep pixel analysis showed the real
rate is 1/400 invisible + ~1.4% marginal at the far corner (D10: keep camera,
level-uniform + train/eval-matched). Eval-set freeze wave launched (tmux
cog_eval_freeze, 14 Kit sessions: L0-L2 + 10 L3 sub-envs, state envs, 10 batches
x 20 envs each) -> configs/eval_sets/{L}.json per D11.

**G3 PASSED — P3 COMPLETE FOR TASK 1 (2026-08-17 ~02:00):** eval-set freeze wave:
13/13 sub-levels EXIT=0, merged to configs/eval_sets/{L0,L1,L2,L3}.json (D11
format: 10 batches x 20 envs of cup/goal init poses per sub-level). Snapshot
invariance matches spec exactly: L0 cup+goal frozen across seeds (0.0 cm); L1 cup
varies 37.9 cm / goal frozen; L2 cup 38.8 + goal 19.5 cm; L3 sub-envs draw
independent streams. G3 criteria all green: 400 clean demos/level (validated
LeRobot sets, --expect_episodes), gen SR 85.5-87.9% >> 30% floor, QA pass, eval
sets frozen. Task 1 data phase is DONE. Next: P5/G5 cluster bring-up (blocked on
G0 Slurm association), then the 24-run matrix.

**TIMING CORRECTION (2026-08-17):** the eval-set freeze wave took 84 s TOTAL
(~6.5 s/leg), not ~2.5 min/leg as first committed — state-only headless Kit
(isaaclab.python.headless.kit, no cameras) boots in ~3 s; the 3.5-4 min startup
cost is entirely the camera/RTX stack. timings.md fixed. Investigation footnote:
log-file mtimes clustered at wave end looked impossible for serial legs until the
in-log carb timestamps showed each leg really ran start-to-finish in ~6.5 s;
EVALSET_OK sits mid-file because carb's buffered log lines flush at shutdown,
after python's flushed print. Trust in-log timestamps over stream order.

**TASK 2 KICKOFF (2026-08-17 ~02:15):** asset/reference recon done (subagent).
Sektion cabinet: joints [door_left, door_right, drawer_bottom, drawer_top],
drawer_top travel ~0.40 m, handle frame drawer_handle_top with grasp-ready
offset (0.305,0,0.01 / rot 0.5,0.5,-0.5,-0.5); asset is Nucleus-cloud-only.
No in-tree cabinet Mimic env. Stock SM insufficient (world-frame offsets, 1.5 cm
pull). D12 finalized with addendum + VERIFY (d) (Articulation as object_ref).
Next: empirical cabinet geometry inspection, then scaffold src/cog/tasks/drawer_stow.

**TASK 2 SCAFFOLD COMPLETE (2026-08-17 ~03:00):** src/cog/tasks/drawer_stow/
written, full cup_place structure mirrored (assets/levels/env cfgs/mdp/mimic/SM)
+ record_drawer_source_demos.py + t2_smoke.py. Design points beyond D12:
- Object can't live on the cabinet top (down-grasp at z .85/r .7 near-singular)
  nor on the ground under the drawer's pull path -> procedural PLINTH at
  (0.24,0.45), top z 0.40; object zone x<=0.26 keeps the vertical grasp corridor
  clear of the opened drawer front (>=0.325 at 0.2 m pull, nearest cabinet pose).
- Cabinet root moved 0.8->0.9 (handle ends at x~0.26 after pull, not 0.19).
- Cabinet rand capped (+-5 cm, +-7.5 deg yaw) so its swept corner misses the plinth.
- SM: torch-based 16 states; handle-frame offsets (yaw-safe), ramped pull latched
  at grasp (0.10 m/s to joint 0.20), traverses at z 0.92 above the drawer rim,
  yaw-aligned box grasp (mod pi/2). 3 Mimic subtasks: open(cabinet)/grasp(object)/
  stow(cabinet); signals drawer_opened_1, grasp_2.
- Custom mdp.drawer_opened obs term (no such helper upstream).
Smoke (env create + 2 expert episodes on L0) running in tmux cog_t2smoke.

**T2 EXPERT DEBUG SESSION (2026-08-17 ~03:30-05:30):** three root causes found
and fixed via the visuomotor frame-dump debug loop (t2_visual_debug.py):
1. SM transition cascade: `s = self.state` aliased the live tensor, so one
   compute() could fall through ALL gates (near/waited precomputed against
   REST's trivially-true target). Fix: evaluate every gate on a frozen s0 copy;
   max one transition per step. (Commit 14a826c.)
2. Sektion drawer actuator is a RETURN SPRING: stock stiffness 10 targets
   joint 0, silently re-closing the drawer after release. stiffness=0 +
   damping=8 makes it hold position like a real drawer. Verified live gains
   [[0,0]]/[[8,8]] in-sim. (4211e98.)
3. Handle grasp roll drives panda_joint6 to its 3.752 rad limit (visible as
   joints[5]=3.75 pinned in the trace); the post-release lift then needs j6
   PAST the limit and DLS stalls at dist 0.17 forever. The bar admits two
   grasp rolls; flipping the TCP 180 deg about the approach axis lands j6
   near 0.6. Debug pattern that found it: print joint vector every 50 steps —
   a pinned coordinate at a round number is a limit, not an IK failure.
Also: "final" state prints after env auto-reset — post-reset reads are the
RESET scene, not the episode end (drawer 0.000 red herring). Pull phase works:
grip 0.010/0.014 on the bar, ramped pull to joint 0.20-0.21 in ~2.7 s.
