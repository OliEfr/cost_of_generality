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

**T2 STOW-TRAVERSE DEBUG (2026-08-17 ~06:00-08:00, runs 6-10):** the hard part
of drawer_stow is carrying the box OVER the 0.779 m drawer wall. Hard-won DLS
(IK-Rel) mechanics on the Panda, for future experts:
- Configuration-branch trap: chasing a FAR high target makes DLS unwrap the
  elbow into the straight branch (j4 -> -0.47), whose max TCP height at radius
  0.4 is ~0.78 -- below the wall. No joint limit is hit; it is a damped stall
  (dist frozen, all joints mid-range). Diagnosis signature: j2~0, j4~-0.47.
- Fix that works: RAMP every long segment (pull-style); ramped targets keep the
  elbow bent (j4 -1.4..-1.0 observed through the same region that stalled).
- Even ramped, holding z while translating outward sags 2-4 cm (z is traded in
  the damped LSQ). Command higher than needed (0.84 for a 0.80 requirement) and
  slow the ramp over the critical region (0.03 m/s).
- Wall clearance arithmetic: carried-box bottom = TCP - (half + |grasp_off|);
  gripping BELOW center (grasp_z_offset -0.005) buys ~1.3 cm.
- Anchor drop targets on LATCHED poses: a live handle-anchored target chases
  the drawer away if the box grazes the wall (runaway feedback, run 9:
  drawer 0.248 -> 0.128 with the target retreating 0.37 -> 0.49).
Handle phase is stable across all runs: grasp 0.012/0.012, pull to 0.248
(joint-gated at 0.20 + overshoot), release, retreat. Object grasp cycle stable:
fingers 0.0277/0.029 on the 5.8 cm box, held through all subsequent states.

**T2 GEOMETRY RESOLUTION (2026-08-17 ~02:30):** runs 15-18 showed the 5.8 cm
box is ~1 cm infeasible: pull ceiling ~0.335 (base-proximity stall), drawer
creeps ~3 cm closed after release (decaying drift, present even at damping 25
-- treat as environmental), carry equilibrium x~0.32. D13: boxes shrunk to
4.0/4.8 cm; physical-clearance descent gate. Drawer damping 8->25 (drift
mitigation + knock resistance).

**T2 FIRST EXPERT SUCCESS (2026-08-17 ~03:15, debug run 21):** episode
terminated at step 673 (6 steps after RELEASE_OBJECT, far before the 800-step
timeout) -- the success term fired: drawer 0.31 open, cube dropped ~12 cm into
the cavity, settled, gripper released. The winning design change: NO descent
into the cavity. The drawer is a container -- release from carry height
(box bottom ~1.2 cm above the wall top, footprint inside the cavity) and let
the walls catch the falling cube; a cube rests identically on any face, and
the success check is position+settled, not gentleness. This eliminates the
entire wedge-prone wall-crossing descent that consumed runs 14-20.
4-episode state-env confirmation smoke running.

**T2 L1/L2 GATE DIAGNOSIS (2026-08-17 ~04:30):** instrumented per-episode
pre-reset readouts ended three wrong hypotheses (carry-quat yaw, time budget,
drawer creep -- each a partial factor at best): successes pull the drawer to
0.31-0.34, failures to 0.15-0.29, and the descent-clearance gate then honestly
refuses. The pull's stall depth varies with the reset joint jitter
(elbow-branch luck; one episode showed a handle slip at 0.148). Fix: PULL
RETRY -- after the retreat, if the opening is < 0.30 and retries < 2, re-grasp
the handle (now nearer, easier) and pull the remaining travel. Fresh arm
configuration each attempt. Lesson recorded: instrument before theorizing;
pre-reset state must be captured explicitly (post-reset reads are the new
episode).

**T2 DECISION-TIME DIAGNOSIS + PEDESTAL (2026-08-17 ~05:30):** proper
instrumentation (drawer@release / drawer@traverse) showed EVERY pull reaches
exactly 0.350 — the variance was post-release: transit corridors clipping the
opened drawer volume + draw-dependent creep, and the constraint algebra
(face-handle offset fixed at 13.15 cm) proves the ground-mounted stow window
is empty by ~2 cm at any pull depth. D14: 0.20 m robot pedestal — the
systemic fix. Carry heights raised (traverse cmd 0.92), pull target relaxed
to 0.28, retry threshold 0.24. Pedestal-geometry gate running.

**T2 END-TO-END SUCCESS AT PEDESTAL 0.08 (2026-08-17 ~09:05, run 29):**
episode success at step 650: approach->grasp (fingers 0.012)->pull 0.287->
retreat->ramped+slerped object leg->box grasp (0.023)->stage->traverse at
z 0.879 (1.2 cm tracking, 7.5 cm wall clearance)->drop-release->success term.
Joints mid-range all episode. The 0.20 m pedestal detour (runs 22-28) and its
lessons recorded in D14-revised. Full 16-episode L0/L1/L2 gate running.

**G2-T2 PASSED 48/48 (2026-08-17 ~09:40):** expert SR 16/16 on each of L0, L1,
L2 (zero failures; gate bar >=90%). The drawer_stow expert is robust across
object pose (10x18 cm + yaw +-45 deg) and cabinet pose (+-5 cm, +-7.5 deg yaw)
randomization. Next: L2 source recording (single-env, over-record 18 keep 15,
D9) -> annotate --auto (VERIFY d: Articulation object_ref) -> camera QA ->
Mimic gen smoke.

**T2 L3 VARIANTS PASS (2026-08-17):** 4/4 expert successes on L3v00 (4.0 cm
box) and L3v05 (4.8 cm box) -- the size-variant axis works; the whole T2
ladder L0-L3 has a working expert.

**T2 MIMIC GENERATION SMOKE PASS (2026-08-17 ~10:30):** 12/12 target demos
generated from the 17 annotated L2 sources on the state env; gen SR ~30-34%
(final tail 7/23 visible + guarantee completion to 12). Meets the G3 floor
(>=30%) — far below T1's 86-88%, as expected for a long-horizon articulated
task; record as a study finding (generation-side difficulty scales with task
complexity). VERIFY (d) fix (cabinet in get_object_poses) held through
generation. Next: visuomotor camera QA, then full 4-level datagen waves.

**T2 CAMERA QA PASS + DATAGEN LAUNCH (2026-08-17 ~11:00):** table_cam frozen at
pos (0.10,-0.85,1.45) look-at (0.45,0.28,0.42) aperture 28: box visible at
rest, carry + open drawer render clearly, expert succeeded on the visuomotor
env twice. Full T2 datagen wave launched (tmux cog_gen_t2): L0/L1/L2 x 400 +
L3v00-09 x 40, all from the same annotated L2 sources (D9). Estimate 13-16 h
(demos ~650 steps x ~30% gen SR ~ 10x T1 per-success cost) -> done overnight.

**T2 WAVE: L0 LEG DONE (2026-08-17 05:49):** 400 successes in 2 h 17 min at
~54.5% gen SR (visuomotor wave SR well above the ~30% state smoke), 8.1 GB.
L1 generating at ~40% SR — the object randomization costs ~15 points of
generation SR, first per-level generation-difficulty signal for T2.

**T2 WAVE: L1 LEG DONE (2026-08-17 ~09:05):** 400 successes at ~44% gen SR
(final visible 394/897), ~3 h 15 min. L2 now generating at ~31% SR. The
per-level generation-SR gradient is emerging clearly: L0 54.5 / L1 ~44 /
L2 ~31 — each T2 randomization axis costs ~11-15 points of generation SR,
in sharp contrast to T1's flat 85-88% across all levels. Strong candidate
finding for the paper: generation-side difficulty scales with task complexity
AND with distribution breadth for long-horizon tasks.

### 2026-08-17 09:52 — T2 wave status check: L0/L1 files verified, L2 mid-flight

Verified the two finished legs directly in HDF5 (not just from log lines):

- `T2_L0.hdf5`: **400 episodes**, action-sequence length min/mean/max = 663/705/724 steps
- `T2_L1.hdf5`: **400 episodes**, length 663/694/724 steps

Episode lengths sit at ~33-36 s of 20 Hz control, comfortably inside the 1200-step
(60 s) timeout, so no episode is finishing by timeout-with-success luck.

Final generation success rates read off the last progress line of each leg:
L0 = 381/699 = **54.5 %**, L1 = 394/897 = **43.9 %**.

L2 is at 118/374 = **31.6 %** after 71 min (started 08:39:34), giving ~1.66
successes/min and an ETA of ~12:40. The 10-variant L3 wave then needs roughly
another 4.5-5 h (400 successes at a similar rate plus ten ~4 min camera-enabled
Kit boots), so the full wave should land ~17:00-17:30 today.

Disk: 473 G free on `/`, `data/hdf5` at 53 G. Note the RecorderManager `_failed`
companions are the bulk of it (L0 6.5 G, L1 10.2 G, L2 4.9 G so far) — they are
regenerable and only needed until per-level gen SR is extracted, so they are the
obvious reclaim target if the 150 G project budget gets tight during T3.

### 2026-08-17 13:05 — T2 WAVE: L2 LEG DONE (400 demos, 30.3 % gen SR); L3 variant wave started

`GEN_T2_L2_EXIT=0` at 12:48. Verified in HDF5: `T2_L2.hdf5` holds **400 episodes**,
action-sequence length min/mean/max = 618/677/743 steps. Final generation counter
384/1267 = **30.3 % gen SR**; wall time 08:39:34 -> 12:48 = **4 h 08 min**.

That closes the three main T2 levels and the per-level gradient is now complete
and measured, not extrapolated:

| Level | successes/attempts | gen SR | wall time |
|---|---|---|---|
| T2 L0 | 381/699 | **54.5 %** | 2 h 16 min |
| T2 L1 | 394/897 | **43.9 %** | 2 h 50 min |
| T2 L2 | 384/1267 | **30.3 %** | 4 h 08 min |

Compare Task 1, same Mimic machinery, same source-demo discipline: L0 86.4 / L1 85.8
/ L2 85.1 / L3 87.9 % — **flat** (spread 2.8 points, and non-monotone, i.e. noise). So the generality tax on data *production* is a
property of the task, not of the generator: on a long-horizon articulated task each
randomization axis costs 11-15 points of generation SR, while on a short pick-place
task extra randomization is free. Every attempt costs the same GPU time, so the L2
dataset cost 1.8x the wall-clock of L0 for the identical 400 demos. This is a
first-class result for the paper (new figure: gen SR vs level, two tasks overlaid)
and it is also a practical warning for anyone planning a Mimic data budget.

L3 variant wave started 12:48 (`Cog-DrawerStow-L3v00`, 40 successes per variant x 10
variants). At 13:05 v00 is at 21/74 = 28.4 %, i.e. ~1.24 successes/min, so ~32 min of
generation plus a ~3.5 min camera-enabled Kit boot per variant -> **~6 h for the ten
variants, landing ~18:45**. Total T2 wave then ~15 h 15 min for 800 demos across 13
sub-levels, which is in line with the 13-16 h estimate written into
`scripts/ops/gen_t2_waves.sh` before launch.

Disk: 455 G free. `T2_L2_failed.hdf5` came out at 17.3 G (the 883 failed attempts) --
the failed companions now total 34 G and are the largest single reclaimable block in
the repo. Their only remaining value was the attempt counts, which are now recorded
in the table above, so they can be dropped whenever space matters.

### 2026-08-17 13:20 — GitHub remote wired up; history rewritten to strip 3 GB of committed checkpoint weights

User created `git@github.com:OliEfr/cost_of_generality.git` and asked to use it. The
repo could not be pushed as it stood: `.git` was **2.8 GB** because commit 75daa95
("Journal: train smoke done...") had committed the G4 smoke checkpoint, including
`optimizer_state.safetensors` (**2035 MB**) and `model.safetensors` (**1018 MB**).
GitHub hard-rejects any blob over 100 MB anywhere in pushed history, so those two
objects had to leave the history — a rewrite was unavoidable, not a preference.

Done non-destructively, in this order:

1. Backed up `.git` (2.8 G) and the whole 3 G checkpoint dir to
   `data/_prepush_backup/` (gitignored, still on disk). Verified the backup repo
   replays: 60 commits, HEAD `ec4dfbc`.
2. Secret-scanned every tracked file before anything left the machine — clean, the
   only hit was a doc *mentioning* `.netrc`, no credentials.
3. `git clone --bare --no-local` into a scratch clone (`--no-local` is required or
   filter-repo refuses: it wants a freshly-packed repo), then
   `git filter-repo --strip-blobs-bigger-than 50M` **inside the clone**, leaving the
   working repo untouched. Result: 2.8 G -> **3.4 MB**, all 61 commits preserved.
4. Verified the rewrite surgically: file lists of original vs cleaned HEAD differ by
   exactly those two paths; the md5 of (blob hash, path) over every *other* tracked
   file is identical in both; all 61 commit subjects identical in order.
5. Pushed to `origin/main` = `25c7a0e`.
6. Re-pointed this working repo without discarding anything: renamed the old branch
   to **`main-prefilter`** (all 61 original commits, big blobs included, still
   reachable locally) and created `main` tracking `origin/main`. Restored the two
   weight files to disk from backup, now covered by
   `experiments/runs/**/*.safetensors` in `.gitignore`.

**Consequence for provenance:** commit hashes quoted in journal entries written
before this point (e.g. b091e9a, f57ee06, 87abc90, ec4dfbc) no longer resolve on
`main` — they live on `main-prefilter` and in
`data/_prepush_backup/git_before_filter_repo`. Commit *messages* are unchanged, so
`git log --grep` finds any of them on either branch.

The running T2 wave was never at risk: it writes only to `data/hdf5/` and `ops/`,
both gitignored, and none of the above touched the tmux session (verified alive and
writing throughout).

### Exact generation SR — stop scraping logs

`T2_L3v00_EXIT=0` at 13:12, and it exposed a measurement bug in how I had been
reporting gen SR. The last progress line visible in the log **understates** the
result, because carb buffers the final prints until shutdown: v00's last visible
line was 21/74 while the file actually holds the full 40 demos.

The `_failed.hdf5` companion is the exact record: attempts = successes + failures.
Counting episodes in both files gives ground truth, and it is cheap (HDF5 key count,
no payload read). Corrected table — the numbers move by only ~0.4 points, but these
are the ones that go in the paper:

| Leg | successes | failures | attempts | gen SR (exact) | log-scraped |
|---|---|---|---|---|---|
| T2 L0 | 400 | 328 | 728 | **54.9 %** | 54.5 % |
| T2 L1 | 400 | 506 | 906 | **44.2 %** | 43.9 % |
| T2 L2 | 400 | 906 | 1306 | **30.6 %** | 30.3 % |
| T2 L3v00 | 40 | 80 | 120 | **33.3 %** | 28.4 % |

L3v00 took 12:48 -> 13:12 = **24 min** for 40 demos (623-743 step episodes), so the
ten-variant wave should finish ~17:00-17:30. v01 booted at 13:12.

Note L3v00's 33.3 % sits *above* L2's 30.6 %: an L3 variant fixes one object size and
colour while keeping L2's pose randomization, so it is narrower than L2, not wider.
The L3 *aggregate* over all ten variants is the L2-plus-object-variation condition —
that aggregate, not any single variant, is what belongs on the gen-SR-vs-level curve.

### 2026-08-17 13:35 — Gen SR now lives in a committed CSV, not in prose

User asked whether the result values were actually recorded in the repo. They were
not: every generation-SR number existed only as prose in `docs/journal.md` and
`docs/timings.md`, and `experiments/registry.csv` is a *training*-run schema
(`sr_40k`/`sr_80k`), with nowhere to put dataset-level statistics. For a number that
becomes a paper figure, that is not good enough — prose gets rewritten and cannot be
re-derived.

Added `scripts/dev/gen_stats.py` -> **`experiments/gen_stats.csv`** (committed). It
recomputes everything from the HDF5 pairs per D16 (successes, failures, attempts,
gen SR, episode-length min/mean/max, finish timestamp, size), so it is idempotent and
safe to re-run as each leg of a wave lands. Files still locked by a running generator
are reported as in-flight and skipped rather than half-read. `--chain-wave T2_
--wave-start ...` fills `wall_min` for a wave whose legs ran back to back from one
script; it is left blank where it is not honestly derivable (the T1 datasets came
from several separate launches, so chaining their mtimes would invent numbers).

The script also prints the per-level aggregate pooled over L3 variants, which is the
only correct way to place L3 on the gen-SR-vs-level curve.

**It immediately caught a third log-scraping error:** Task 1 L2 is **85.1 %**
(400/470), not the 85.5 % carried in the docs since 2026-08-16. Corrected in today's
comparison table. Earlier entries are left as written (they are the honest record of
what was believed then); `experiments/gen_stats.csv` now supersedes every gen-SR
figure quoted anywhere in the docs.

Full ground-truth picture, 17 finished datasets:

| | L0 | L1 | L2 | L3 (pooled) |
|---|---|---|---|---|
| **T1 cup_place** | 86.4 % (400/463) | 85.8 % (400/466) | 85.1 % (400/470) | 87.9 % (400/455) |
| **T2 drawer_stow** | 54.9 % (400/728) | 44.2 % (400/906) | 30.6 % (400/1306) | 33.3 % so far (40/120, v00 only) |

Mean episode length is the other half of the cost story and is now in the CSV too:
T1 ~187-207 steps vs T2 ~675-705, so a T2 attempt costs ~3.5x a T1 attempt in sim
time *before* the SR gap multiplies it. T2 L2 needed 1306 attempts x ~677 steps for
400 demos; T1 L2 needed 470 x ~187.

### 2026-08-17 14:45 — L3 is a weaker generality axis than the plan specifies (needs a user decision)

Four consecutive T2 L3 variants landed at *exactly* 40/120 attempts and 23 min each.
Identical to the attempt is not luck, so I checked the definitions:

- T2 `L3_VARIANTS = [box_{s}_{c} for s in ("s","m") for c in 5 colours]`
  -> v00-v04 = size 0.040 in five colours, v05-v09 = size 0.048 in five colours.
- T1 `L3_VARIANTS = [cyl_{s}_{c} for s in ("s","m") for c in 5 colours]`
  -> same structure: radius 0.027/0.031, height 0.080/0.090.

Colour is a material property with no physical effect, so within a size group the
dynamics are bit-identical and, at a fixed seed, the generator reproduces the same
attempt sequence. The data shows exactly that, in both tasks independently:

| | v00-v04 (size s) | v05-v09 (size m) |
|---|---|---|
| T1 cup_place | 40/45 = 88.9 % (x5, identical) | 40/46 = 87.0 % (x5, identical) |
| T2 drawer_stow | 40/120 = 33.3 % (x4 so far, identical) | not yet generated |

**Prediction to check when v05 lands:** it should break the 40/120 pattern and then
repeat identically for v06-v09. If v05-v09 also come out at 40/120, the size axis is
doing nothing either and L3 is *purely* cosmetic.

Two consequences.

*The good one:* this is a free determinism check on the whole Mimic pipeline. Same
seed + same physics reproduces the same success/failure sequence across ten separate
process launches, hours apart.

*The problem:* **L3 as built is 2 geometries x 5 colours, not the "4 mug meshes x 5
colours x scale 0.9-1.1" the plan specifies.** D1 deferred the mug meshes ("join
L3_VARIANTS only if grasp+render QA passes") and that step was never executed before
P3 closed, so L3's geometric spread is a 10 % (T1) / 20 % (T2) size difference plus
appearance. If an L3 policy turns out to need barely more data than L2, the honest
reading may be "this axis is nearly trivial", not "object generality is cheap" — a
confound sitting directly under a headline claim.

**This is the cheap moment to fix it:** no training has run yet (P6 is blocked on G0),
and regenerating T1 L3 costs ~26 min, T2 L3 ~4 h. After the matrix runs it is
unaffordable. Recorded as **D17 (OPEN)** — needs the user's call, because it changes
what a level *means* in the paper and would require re-freezing the L3 eval sets,
which rule 8 otherwise forbids touching.

My recommendation is option (c) below: don't disturb the finished data phase, describe
L3 honestly, and add geometry as its own level later — that keeps the schedule and
turns the gap into an extra result rather than a caveat.

(a) Regenerate L3 with mug meshes added (2 cyl + 2 mug geometries). Truest to the
    plan. Costs the mug grasp/render QA D1 asked for, a T1+T2 L3 regeneration, an
    L3 eval-set re-freeze, and mug USD is Nucleus-cloud-only, adding cluster
    staging risk.
(b) Leave L3 as is and say nothing. Cheapest, and wrong — the paper would claim an
    object-variation axis it does not have.
(c) Leave L3's data as is, rename the level honestly ("appearance + mild scale") in
    the paper, cite the identical-gen-SR evidence that 5 of 10 variants are
    pixel-only, and add a separate **L4 geometry** level with the mug meshes after
    the Task-1 matrix. Purely additive: touches no frozen benchmark, and directly
    tests appearance-vs-geometry cost, which is a better result than either alone.

### 2026-08-17 15:36 — D17 prediction confirmed: the size axis is real but small

Predicted at 14:45 that T2 L3v05 would break the 40/120 pattern (it starts the
0.048 m box group) and then repeat identically for v06-v09. Both halves held:

| variant group | box size | attempts | gen SR | mean ep len |
|---|---|---|---|---|
| v00-v04 | 0.040 m | 120 (x5, identical) | 33.3 % | 675 |
| v05-v06 | 0.048 m | 125 (x2 so far, identical) | 32.0 % | 679 |

So L3 does contain two genuinely distinct physical conditions — the bigger box costs
5 extra attempts per 40 demos and runs 4 steps longer per episode — but the effect is
small (1.3 points of gen SR) and there are only two of them, five colours deep each.
That sharpens D17 rather than changing it: the axis is not null, it is thin. Option
(b) "leave it and say nothing" is now definitively out, since I can quantify exactly
how thin it is.

The within-group determinism across ten independent process launches is now confirmed
in both directions (identical inside a group, reproducibly different between groups),
which is a stronger pipeline-determinism check than I could have designed on purpose.

### 2026-08-17 16:42 — T2 DATAGEN WAVE COMPLETE: 13/13 legs, 1600 demos, 13 h 10 min

`T2_WAVES_DONE`, all thirteen `GEN_T2_*_EXIT=0`. Wave ran 03:32 -> 16:42 =
**13 h 10 min** unattended on the shared 4090, producing 1600 demos (400 per level,
L3 as 10 x 40 variants). Inside the 13-16 h pre-launch estimate.

Final generation SR, exact per D16 (`experiments/gen_stats.csv`, 26 datasets):

| Level | T1 cup_place | T2 drawer_stow |
|---|---|---|
| L0 | 86.4 % (400/463) | **54.9 %** (400/728) |
| L1 | 85.8 % (400/466) | **44.2 %** (400/906) |
| L2 | 85.1 % (400/470) | **30.6 %** (400/1306) |
| L3 (pooled) | 87.9 % (400/455) | **32.7 %** (400/1225) |

L3 pooled (32.7 %) sits just above L2 (30.6 %) because each L3 variant fixes one box
geometry while keeping L2's pose randomization — narrower per variant, so this is not
a reversal of the downward trend (see D17 on how thin that axis is).

**The headline contrast is now complete and measured on both tasks:** T1 generation SR
is flat across the whole generality ladder (85.1-87.9 %, spread 2.8 points,
non-monotone = noise), while T2 falls 54.9 -> 44.2 -> 30.6 % as randomization axes are
added, 11-15 points each. Generality taxes *data production* on a long-horizon
articulated task and is free on a short pick-place task.

Because every attempt costs the same sim time, this compounds: T2 L2 needed 1306
attempts x ~677 steps for its 400 demos, against T1 L2's 470 x ~187 — **10x the
simulation work for the same dataset size**, of which ~2.8x is episode length and
~2.8x is the SR penalty.

Per-leg wall times are in `experiments/gen_stats.csv` (`wall_min`): L0 137, L1 171,
L2 249, each L3 variant 23-24 min (the ten variant launches pay ~35 min of pure Kit
boot between them).

Next: convert all four T2 levels to LeRobot and validate (`scripts/ops/convert_t2_all.sh`).
