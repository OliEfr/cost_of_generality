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

### 2026-08-17 16:45 — T2 conversion launched (and a repeat tmux-environment bite)

First launch of `scripts/ops/convert_t2_all.sh` died in under a second: all four legs
returned 127 with `python: command not found`. Cause: the script did
`source ~/miniforge3/etc/profile.d/conda.sh` but **conda on this box is
`~/miniconda3`**, and a tmux-spawned bash is non-interactive so it has no conda shell
function to fall back on. My interactive calls had been working only because the
session shell already had conda initialised — the classic "works when I type it,
fails in tmux" trap, and the second time a tmux launch has failed for environment
reasons (the eval-freeze wave died on a missing redirect directory).

Fix: call the interpreter by absolute path,
`/home/admin_07/miniconda3/envs/cog_isaac/bin/python`, and assert up front that it
exists and can import lerobot+h5py. Also made validation skip (rather than run and
fail confusingly) when its conversion failed. Nothing was created by the bad run —
the four dataset roots did not exist — so the re-run started clean.

**Rule for future tmux/cron work in this repo: never rely on `conda activate` in a
non-interactive shell; use the absolute env python and fail fast if it is missing.**
The bad log is kept as `ops/convert_t2_failed_env.log`.

Relaunched 16:44, converting T2_L0 -> `data/lerobot/T2_L0` with validation
(`--expect_episodes 400`) chained per level. T2 episodes average ~680 frames vs T1's
~190, so expect roughly 3.5x T1's per-level conversion cost; will record the measured
number in timings.md when L0 lands.

### 2026-08-17 16:58 — T2 conversion parallelized four ways (8 h -> ~1 h)

The sequential chain's first progress checkpoint gave the real rate: 25/400 episodes
in 7.3 min, i.e. **~2 h per level and ~8 h for all four**. Checked the machine before
accepting that: the converter runs at ~103 % CPU (h264 encode is single-core) on a
32-thread box at load ~4. Nothing about this job is inherently serial — the four
levels are independent datasets — so serializing them wasted ~24 of 32 threads.

Restarted as four independent tmux sessions (`cog_cv_T2_L0..L3`), one level each,
after removing the 12-min-old partial `T2_L0` root (the converter refuses to write
into an existing root, by design). `scripts/ops/convert_t2_all.sh` now takes optional
level keys, so `... T2_L1` runs one level and no arguments keeps the old all-levels
behaviour.

Aggregate throughput is ~4x with per-level speed *unchanged* (L0: 33 MB in 5 min
parallel vs 38 MB in 7.3 min sequential), load only ~5.5/32. All four should land
~17:45 instead of ~00:45.

**Self-inflicted lesson worth writing down:** the stop step used
`pkill -f "hdf5_to_lerobot"`, and that pattern also matched *my own shell command
line*, so pkill killed the shell running it (exit 144) before it reached the cleanup
`rm`. No damage — the converters did die as intended and only the cleanup was lost —
but `pkill -f <pattern>` inside a command that itself contains the pattern is a
foot-gun. Kill by PID from `pgrep`, or exclude the wrapper.

Two nags for the user, unchanged: the orphan `frames_qa` PID 2083049 has now been
running 23 h at ~110 % CPU holding 4.8 GB of VRAM (it is now also stealing a core from
conversion; I am classifier-blocked from killing it), and **D17 is still open**.

### 2026-08-17 17:45 — the 24 h "orphan" was a hung Kit shutdown, not a running job

Finally diagnosed the `frames_qa.py --level L1` process (PID 2083049) that had been
burning 110 % CPU and 4.8 GB of VRAM since Sunday 17:36. It was **not** still working:
its output `ops/qa/frames_L1.png` was written at 17:49, thirteen minutes after launch,
and the process then failed to exit — 201 threads, state R, spinning forever. Every
artifact it was supposed to produce has been on disk and committed since Sunday.

So a job that looked alive for 24 h had actually finished in 13 min. **Liveness checks
must look at the output file, not at CPU%:** a spinning Isaac process proves nothing.
This is the mirror image of the earlier false-stall alarm, where a *quiet* log made me
think a healthy job had died and the file mtime proved it alive. Same lesson from both
directions — trust artifacts over process state.

`kill` (SIGTERM) did nothing: the hang takes the signal handling down with it. It
needed `kill -9`. After that, VRAM went 7.3 GB -> 2.4 GB used (only the foreign eval
job's 1.6 GB left), freeing ~4.8 GB and a core that the four conversion jobs were
competing with.

Operational rule: after any `--enable_cameras` Isaac script finishes writing its
output, confirm the process actually exited; if it is still spinning, SIGKILL it.
Otherwise these accumulate and quietly eat the GPU for days.

### 2026-08-17 18:05 — What the generality ladder actually varies: poses and paint, never geometry

Prompted by the user asking whether the drawer varies too. Auditing both tasks'
level definitions against the code:

**T2 scene contents.** Manipulated object: one cube, 4.0 or 4.8 cm edge, 0.12 kg, five
colours. Fixtures: the stock Isaac Lab **Sektion** cabinet (root x=0.9, yawed 180 deg to
face the robot; its top drawer is the target), a fixed plinth 24x30x40 cm at
(0.24, 0.45) whose top at z=0.40 holds the box, an 8 cm pedestal under the Franka, and
the ground plane.

**What varies, per level:**

| | robot start | box pose | cabinet pose | box size/colour | drawer geometry | drawer start |
|---|---|---|---|---|---|---|
| L0 | joint noise | fixed | fixed | fixed | same | closed |
| L1 | joint noise | 10x18 cm, yaw +-45 deg | fixed | fixed | same | closed |
| L2 | joint noise | 10x18 cm, yaw +-45 deg | +-5 cm x, +-6 cm y, +-7.5 deg yaw | fixed | same | closed |
| L3 | joint noise | 10x18 cm, yaw +-45 deg | +-5 cm x, +-6 cm y, +-7.5 deg yaw | 2 sizes x 5 colours | same | closed |

So the drawer **does** vary in *where it is* -- and that is the most expensive axis in
the whole study, costing 13 points of generation SR (44.2 -> 30.6 %), more than the
object-pose axis (10.7 points) and far more than the object axis (~1 point). But the
drawer never varies in *what it is*: same Sektion asset, same drawer box, same handle,
same 0-0.4 m travel, and the reset always writes every cabinet joint to 0, so it always
starts fully closed. Never partially open, never a different cabinet.

**The honest characterization of the ladder is therefore: it varies pose and
appearance, never geometry or kinematics.** That is one coherent story rather than two
separate gaps (the thin object axis is just the visible symptom), and it is the right
framing for both the paper's limitations section and the open decision: the missing
axis is not "mug meshes", it is *shape/kinematic variation* as a class -- different
object geometries, different cabinets, partially-open starting drawers.

This strengthens the recommendation already on the table: describe the current ladder
as pose+appearance, and add geometry as a distinct additional level rather than trying
to retrofit it into the existing one. It also suggests the cheapest possible geometry
axis for T2 is not a second cabinet asset (another cloud-asset dependency) but
**randomizing the drawer's initial opening** -- one line in the reset event, no new
assets, and it directly attacks the kinematic-state assumption the expert leans on.
Noted for the user's decision; not implemented.

### 2026-08-17 18:20 — CORRECTION: Mimic's rigid-transform assumption is *itself* the constraint on which generality axes we can have

User pushed back on my 18:05 suggestion that pre-opening the drawer is "one line in
the reset event", asking whether changing objects significantly or pre-opening the
drawer breaks MimicGen. They are right and I was wrong. Read the actual math in
`third_party/IsaacLab/.../datagen/data_generator.py`
(`transform_source_data_segment_using_object_pose`, lines 52-83):

    src_eef_rel_obj = src_eef_poses @ inv(src_obj_pose)     # source eef in object frame
    new_eef_poses   = src_eef_rel_obj @ cur_obj_pose        # re-applied to current object

Per subtask, the source end-effector trajectory is expressed relative to **one 4x4
reference pose** and rigidly re-applied to that reference's current pose. Two
assumptions follow, and both are load-bearing:

1. **The needed gripper pose is a fixed rigid offset from the reference frame** -- i.e.
   object geometry is unchanged, or changed so little the same offset still works.
2. **The reference pose captures all relevant state.** It is a single rigid pose; any
   degree of freedom not encoded in it is invisible to the transform.

**Pre-opening the drawer breaks assumption 2 as currently configured.** Subtask 1 uses
`object_ref="cabinet"`, and our `get_object_poses` supplies the cabinet **root** pose,
which does not move when the drawer slides. A drawer starting 10 cm open therefore
leaves the reference pose identical while the handle has moved 10 cm -- the transformed
trajectory would reach for the closed-drawer handle position and miss by exactly the
initial opening. Subtask 3 (stow) has the same blind spot: it also references the
cabinet root, so the drawer's opening at stow time is invisible to it.

It is fixable in principle -- expose the drawer body/handle frame (we already publish a
handle FrameTransformer) and point subtasks 1 and 3 at it, so the opening lives *inside*
the reference pose. But it is not one line, and two further issues remain:
the pull *stroke* is baked into the source segment (starting part-open plus a full
stroke drives toward the 0.4 m travel limit and changes the final opening, which shifts
the stow clearance), and the `drawer_opened` termination signal is an absolute
threshold, so a drawer starting past it satisfies subtask 1 at t=0 and the segmentation
degenerates. The threshold would have to become a delta from the initial opening, or
the randomization be capped below it.

**Significant object geometry change breaks assumption 1.** Subtask 2's grasp is a
fixed offset in the box frame. Cube 4.0 -> 4.8 cm is fine (flat faces, grip 5 mm below
centre). A mug with a handle needs a genuinely different grasp relative to its own
frame, and one that depends on yaw -- exactly the confound D1 flagged. One source-demo
set cannot cover mug and cylinder; you would need **per-geometry source demos**, i.e.
re-run the scripted-expert tuning per shape. For T2 that tuning was 29 debug iterations
for a single geometry.

**This reframes the open decision and is a genuine paper finding, not an excuse.** The
reason our ladder varies pose and appearance rather than geometry is not oversight: it
is that **the data generator's object-centric rigid-transform assumption makes pose
variation nearly free to generate and geometry variation expensive.** Any MimicGen-style
pipeline has this property, which is why datasets built this way vary placements, not
shapes. Stating that explicitly -- with our measured numbers, where pose axes cost
11-13 points of generation SR and the appearance axis costs ~1 -- is a contribution
about the method, and it is a more honest and more useful result than quietly bolting on
a half-working geometry axis.

Revised option set for the user, cheapest first:
(1) **Widen the box scale within the cube family** (e.g. 3.5-5.5 cm instead of
    4.0/4.8). Mimic-safe, no new source demos, no new assets; bounded above by the stow
    corridor (5.8 cm was infeasible, D13). Makes the existing axis less thin without
    touching the method's assumptions.
(2) **Accept the ladder as pose+appearance and report the constraint as a finding.**
    Zero cost, scientifically honest, adds a result about MimicGen-style pipelines.
(3) **Drawer initial opening**: needs the reference-frame change *and* a delta-based
    termination signal; medium effort, real risk of a fresh debug loop.
(4) **New geometries (mug)**: needs per-geometry source demos and per-shape expert
    tuning; expensive, and re-introduces the yaw/handle grasp confound.
My recommendation is now **(1) + (2)**: widen the scale range if we ever regenerate L3
for another reason, and otherwise report the constraint rather than fight it.

### 2026-08-17 18:35 — DECISION CLOSED: ladder stays as built, gap recorded as a limitation

User's call: keep everything as is, note it as a limitation. No data regenerated, no
level definitions touched, no frozen eval set disturbed.

Landed:
- `docs/decisions.md` D17 flipped OPEN -> RESOLVED with the rationale and the two
  follow-ups explicitly declined (mug meshes; randomized drawer start), plus the note
  that if L3 is ever rebuilt for another reason the box edge range should be widened
  to 3.5-5.5 cm while it is open.
- **`paper/limitations.md` created** — the first file in `paper/`. Five entries: (1) the
  axes vary placement and appearance, never shape or mechanism, with the per-level table
  and the byte-identical-colour-runs evidence; (2) *why* — Mimic's rigid object-centric
  transform, written out with the actual math and the measured per-axis generation-SR
  costs (object pose 10.7 pts, fixture pose 13.6 pts, appearance ~1 pt); (3) one seed
  per cell; (4) fixed 80k-step budget rather than convergence; (5) in-distribution,
  simulation-only evaluation.

Entry 2 is deliberately written as a *result*, not an apology: pose generality is
expensive to learn but cheap to generate, geometry generality is not cheaply generatable
under this method at all. That asymmetry plausibly explains why MimicGen-style datasets
in the literature vary placements rather than shapes, and it means demo-count studies
built on such pipelines are systematically better evidence about spatial generality than
about object generality.

`paper/limitations.md` is the running list from here on — new limitations get appended
as they land rather than reconstructed at writing time.

### 2026-08-17 18:56 — T2 CONVERSION COMPLETE: 4/4 levels, 400 eps each, all VALIDATE_OK

All four `CONVERT_T2_*_EXIT=0` and `VALIDATE_T2_*_EXIT=0`, 16:53 -> 18:55.

| dataset | episodes | frames | size |
|---|---|---|---|
| T2_L0 | 400 | 281,987 | 356 MB |
| T2_L1 | 400 | 277,661 | 351 MB |
| T2_L2 | 400 | 270,744 | 343 MB |
| T2_L3 | 400 | 270,745 | 344 MB |

Parallelizing paid off exactly as predicted: **2 h wall instead of ~8 h**, load never
above 5 of 32 threads.

L3 merge verified from `conversion_manifest.json`: exactly **40 episodes per variant**,
and the episode order cycles v00..v09 repeatedly, so every nested-N prefix is
variant-balanced (N=10 -> one per variant, N=200 -> twenty each). That is the property
D2 promised and it is now checked rather than assumed.

Remaining for the T2 data phase: dataset QA (adapt `scripts/dev/dataset_qa.py`, which
asserts T1's cup-to-goal final distance) and freezing the 13 T2 eval sets.

### 2026-08-17 19:20 — T2 eval sets frozen; found and fixed a 20-distinct-poses bug in the L3 protocol

T2 freeze wave: **13/13 sub-levels `EXIT=0`, all `EVALSET_OK`**, merged to
`configs/eval_sets/T2_{L0,L1,L2,L3}.json`. Generalized `freeze_eval_sets.py` with
`--task_kind` (drawer_stow snapshots the box pose plus the cabinet root *and its joint
positions*, since the cabinet is an Articulation) and parameterized
`merge_eval_sets.py` with `--raw`/`--prefix` so T1's frozen files cannot be touched --
verified by md5 before/after (all five T1 files OK).

Snapshot invariance is exactly per spec, checked **per-env across seeds** (my first
check compared across the 20 parallel envs and reported 12 m "spreads" -- that was the
check being wrong, not the data; env origins are spaced in a grid):

| | box xyz | box yaw | cabinet xyz | cabinet yaw |
|---|---|---|---|---|
| T2_L0 | frozen | frozen | frozen | frozen |
| T2_L1 | 9.7 x 17.5 cm | varies | frozen | frozen |
| T2_L2 | 9.7 x 17.5 cm | varies | 9.8 x 11.6 cm | varies |

Drawer joints are 0.0 in every frozen batch, confirming every eval episode starts closed.

**The real find: the L3 eval protocol was much weaker than it looked.** "Batch 0 on each
of the 10 sub-envs, pooled (200 eps)" gives only **20 distinct object poses**, because
L3 variants share the pose RNG stream — batch 0 is the same 20 poses ten times over,
once per appearance. L0-L2's 100-episode standard eval has 100 distinct poses. So L3's
apparent 200-episode precision was ~20 independent spatial draws, and L3 was not
comparable to the other levels on the study's dominant axis. This affects **both tasks
identically** and was sitting inside the headline curve.

Fixed for free by reading the *diagonal* — variant v uses batch v — which yields 200
episodes with **200 distinct poses** and all ten variants (verified for both tasks). The
frozen snapshots already held 10 batches per variant, so this changes which committed
rows the protocol reads, not the data: T1's `L3.json` diff is exactly two protocol
strings with `variants` byte-identical. No eval had run yet, so nothing is invalidated.
Recorded as D18, along with the guidance that cross-level headline comparisons use each
level's 200-episode set (equal spatial coverage).

**This also retracts an earlier claim of mine.** The G3 entry said "L3 sub-envs draw
independent streams" — they do not. Same seed, same poses. Two wrong claims about
variant independence in two days (this and the T1 gen-SR figures); the pattern is that I
asserted a property that *sounded* right for per-variant sub-environments instead of
querying the artifacts. The artifacts were available both times.

### 2026-08-17 19:25 — G3 PASSED FOR TASK 2 — the data phase of the study is COMPLETE

Verified checklist, all green:

1. **HDF5 pools:** 1600 demos across 13 legs (400 x L0/L1/L2, 10 x 40 for L3).
2. **LeRobot datasets:** `data/lerobot/T2_{L0,L1,L2,L3}`, 400 episodes each, every one
   `VALIDATE_OK` with `--expect_episodes 400`; L3 verified variant-balanced (40 per
   variant, interleaved order).
3. **Eval sets frozen:** `configs/eval_sets/T2_{L0,L1,L2}.json` (10 batches x 20 envs
   each) + `T2_L3.json` (10 variants x 10 batches), invariance matching spec per-level,
   drawer closed in every frozen batch, L3 on the corrected diagonal protocol (D18).
4. **`experiments/gen_stats.csv`:** 13 T2 rows with exact successes/failures/attempts.
5. **QA:** 12 artifacts (grid, coverage, drawer-opening histogram per level); asserts
   pass on all four levels; T1 QA re-run as a regression after the refactor, unchanged.

**Both tasks now have complete, validated, QA'd datasets and frozen benchmarks.** The
study's entire local data phase is done. Everything remaining is GPU work that is
blocked on the cluster association (G0):
- P5/G5a: batch-size/LR smoke on one A100, then one full 80k run to calibrate GPU-h.
- P5/G5b: the A100 rendering gate (decides whether eval runs on cluster or stays local).
- P6: the 24-run Task-1 matrix, then the Task-2 matrix.

Task 3 (push-to-target) remains the only implementation work available locally, and it
is the sensible next thing to build while the cluster is blocked.

Totals for the T2 phase, for the record: env+expert built from scratch, 29 expert debug
iterations, 13 h 10 min of unattended generation, 2 h of parallel conversion, and three
methodological errors caught and corrected along the way (log-scraped gen SR, the
20-distinct-poses L3 eval protocol, and the false claim that L3 variants draw
independent RNG streams).

### 2026-08-17 19:40-21:20 — Task 3 (push_target) built; expert at 78-98% and closing

Full package written: `src/cog/tasks/push_target/` (assets, levels, mdp/{observations,
events,terminations}, env cfg, franka state+visuomotor cfgs, mimic env + cfg, state
machine) plus `scripts/dev/t3_smoke.py`. Design is D19. Empirical grounding first: the
2026-08-17 probe measured that the `ee_frame` TCP z IS the contact height and that the T1
cup TIPS at 90 deg when pushed, which is why the object is a wide flat puck.

**Expert debugging, in order — every fix came from a trace, and every hypothesis I
formed without one was wrong:**

1. **Runaway ramp.** The push ramp advanced 0.015/tick unconditionally while the arm
   followed at a third of that (commanded 0.180 m vs 0.045 m actual). The ramp pinned at
   its cap in ~26 ticks, `spent` fired, and the machine retreated mid-push; the puck only
   arrived because the runaway command dragged the blade forward during RETREAT. Fixed by
   bounding the commanded advance to PUSH_LEAD ahead of MEASURED progress, and by reading
   `spent` off measured travel. Final error 3.6 cm -> 0.8 cm.
2. **Masked/full-width tensor mix.** `self.pushed[m] = torch.minimum(self.pushed[m] + rate,
   lead)[m]` only works when every env is in the same state -- which a 2-env smoke test
   guarantees and an 8-env gate does not. Compute at full width, then mask-assign.
3. **Lateral squirt.** An open-loop straight stroke let a 2 cm blade slip off a cylinder
   (22% on the largest puck). Replaced with a closed-loop pursuit: each tick, aim at a
   point just inside the puck's near surface along the CURRENT puck->target line.
4. **Descent hang.** Measured steady-state IK tracking error on the descent is ~6 mm and
   the gate was 4 mm, so DESCEND hung until timeout with the puck untouched -- the exact
   bug class that pinned T1's expert at 47% (13 mm error vs 12 mm gate). Three defences:
   command BELOW the contact height, gate looser than the tracking error, and a tick
   budget that advances the state regardless.
5. **Crawl.** With a 2 cm lead the aim sat ~1.5 cm ahead of the blade, so the commanded
   step was limited by that gap, not the rate: 0.7 mm/tick, 20x too slow, timing out
   mid-stroke. Raised the lead; also raised approach/descend rates and the episode budget
   20 s -> 30 s.
6. **Overshoot.** A FIXED lead has no notion of "nearly there": error bottomed at 5.3 cm
   and was then shoved 18 cm PAST the target, so the stop test never fired. Added
   proportional braking.
7. **Braking killed the push, radius-dependently.** Penetration is (lead_eff -
   BLADE_HALF), which a purely proportional law fades to zero at ~3 cm of error, before
   the 1.8 cm stop test. This INVERTED the failure mode -- large pucks failed with a fixed
   lead, small pucks with a proportional one. Added a constant press floor.
8. **Penetration exceeded the object.** Full lead commands the blade 3.8 cm past the
   surface; for a 3.2 cm-radius puck that is past its CENTRE, so the blade tried to occupy
   the puck's space and knocked it away (failures clustered at 11-16 cm). Capped
   penetration at min(3 cm, 0.6 x radius). This removed the geometry sensitivity: the
   three radii now score 80 / 77.5 / 80 instead of 57.5 / 75 / 90.

**Gate status:** L0 95%, L1 97.5%, L2 85%, L3v00 80%, L3v04 77.5%, L3v09 80%.
L0/L1 pass; L2 and L3 are 5-12 points short of the 90% gate. Since L1 (no bearing
randomization) scores 97.5% and L2 (bearing +-40 deg) 85%, the bearing axis carries
almost all of the remaining loss -- geometry now costs only ~5 points. Next diagnostic is
whether the loss concentrates at extreme bearings (reachability / wrist reconfiguration)
rather than being uniform.

**Note for the record:** this took 8 fix cycles against T2's 29, and every one was found
by tracing decision-time state rather than by reasoning about the mechanism. The two
times I predicted a cause without a trace (lateral squirt, then the timeout) I was wrong
about which variable mattered.

### 2026-08-17 21:20-23:10 — Task 3 expert: final gate numbers and three more traced fixes

Continuing the push expert. Gate, 80 episodes per level (SE ~3.5 points):

| level | expert SR | |
|---|---|---|
| L0 | 93.8 % | pass |
| L1 | 93.8 % | pass |
| L2 | 85.0 % | 5 short |
| L3v00 (r 0.032) | 77.5 % | worst variant |
| L3v04 (r 0.045, h 0.040) | 92.5 % | pass |
| L3v09 (r 0.045, h 0.055) | 88.8 % | borderline |

Fixes since the last entry, all trace-driven:

9. **Retreat lurch.** At the push->retreat transition the commanded pose jumped to a
   PROJECTION-derived point that sits ahead of the arm whenever it lags, so the retreat
   began with a forward shove. Failing episodes overran to 25-36 cm of travel against a
   20 cm nominal. Now the retreat starts from the MEASURED TCP.
10. **Bearing axis capped by measurement.** Binned expert SR by |bearing - 90 deg| on L2:
    0-10 deg 94 %, 10-25 deg 95 %, 25-45 deg 75 %. Beyond ~25 deg the stroke runs toward
    the edge of the arm's comfortable workspace. BEARING_RANGE narrowed +-40 -> +-25 deg,
    which is a 50 deg arc of push directions -- still a real axis, now a serviceable one.
    **Method note:** my first bearing diagnostic reported the MEAN bearing of failures
    (98 deg vs 89 deg for successes) and I read that as exonerating the bearing axis. It
    was uninformative, not exculpatory: the sampled range is symmetric about 90 deg, so any
    bearing-driven loss still averages ~90. Binned SR was the measurement that separated
    them. Reporting a mean where the mechanism predicts a symmetric split proves nothing.
11. **Geometry axis capped by measurement.** Expert SR falls monotonically with radius:
    0.032 -> 88 %, 0.045 -> 92-94 %, 0.052 -> 73-83 %, 0.058 -> 63-75 %. A ~2 cm blade
    cannot keep a 12 cm-wide disc on line -- contact is a short chord of a shallow arc, so
    lateral offset spins the puck rather than translating it. PUCK_RADII re-spaced across
    the reliable band 0.032-0.045 (1.4x range, vs T2's 1.2x box edge), still ten variants.
12. **Hidden coupling: standoff <- MAX_PUCK_RADIUS.** Re-spacing the radii silently shrank
    the descent stand-off from 3.0 cm to 1.7 cm, so the descending blade clipped the puck's
    rim and the SAME physical variant went from 92 % to 69 %. The stand-off is a DESCENT
    CLEARANCE and is now its own constant, not a function of the variant set. Lesson: a
    constant derived from a data set changes meaning when the data set changes.

**A change tried and REVERTED, recorded because the reasoning was sound but the cost was
not:** success gained a blade-clearance clause (the non-prehensile analogue of T1/T2's
`released`), motivated by real evidence -- episodes were succeeding mid-stroke the instant
a puck stalled at the disk edge, having travelled 15 of 20 cm, and such demos are poor
Mimic sources because every generated copy inherits a ~5 cm error against a 5 cm gate. But
the clause cost 5-35 points of expert SR (L2 92 -> 55 %), because episodes that legitimately
succeed as the puck settles then also had to finish a full retreat inside the episode
budget. Raising the budget 30 -> 40 s recovered only part of it. Reverted: the goal is
better served by SELECTING sources on final placement error at recording time, which costs
nothing and does not distort the success definition the study is measured against. The
40 s budget was kept as harmless headroom.

**Recurring mistake, three times in one build:** I wrote a gate tighter than the motion
could achieve -- descent gate 4 mm vs 6 mm tracking error; push stop 1.8 cm vs a braking law
that faded to zero at 3 cm; retreat 6 cm vs an 8 cm clearance requirement. Each looked
correct in isolation. The check that would have caught all three: before adding a
threshold, compute what the motion actually delivers and require margin.

**Next:** decide whether to close the L2/L3v00 gap or accept ~85-90 % (sources are recorded
on ONE level and reused per D9, so 85 % suffices to produce ~20 clean sources); then camera
QA, source recording, annotation, generation smoke, and the pipeline seams (`gen_stats.py`
T3_ prefix, converter TASK_SPECS entry, `dataset_qa.py` qa_push_target, `freeze_eval_sets.py`
task_kind branch, ops wave scripts).

### 2026-08-17 22:36-23:20 — T3 pipeline seams, source demos, and the source-quality trap

Machine was idle (T2 done, cluster blocked), so continued the Task-3 build.

**Four pipeline seams patched**, all of which the recon had flagged as silent-failure risks:
- `gen_stats.py`: prefix table `{T2_: drawer_stow, T3_: push_target}`. Unpatched, every
  `T3_*.hdf5` would have been filed as `cup_place` with level `T3_L0` -- no error, just a
  wrong CSV, and that CSV is the study's source of truth for generation SR.
- `hdf5_to_lerobot.py`: `push_target` TASK_SPECS entry (privileged `object_pos/quat` plus
  `target_pos`, since the target moves every episode and the vision policy must read it
  off the camera).
- `freeze_eval_sets.py`: a real `push_target` branch. The existing `snapshot()` had no
  default, so merely adding the choice would have returned the drawer_stow dict and failed
  on `scene['cabinet']`.
- `dataset_qa.py`: `qa_push_target` + the T3 level table. It measures travel in the PUSH
  frame, not world coordinates -- a world-frame measurement from L2 on would just
  re-measure the bearing distribution.
- `setup_vendored.sh`: patches the `cog.tasks.push_target` import into the vendored
  annotate/generate scripts (3 cog imports now); re-run and verified.

**Source demos recorded: `data/hdf5/T3_L2_source.hdf5`, 20 demos.** Getting them usable
exposed the sharpest issue of the build:

The first recording run reported a perfect **expert SR of 1.00 (20/20)** with a **median
final placement error of 5.01 cm** -- exactly the 5 cm success gate, with 18 of 20 demos
worse than 2.5 cm. Success fires the instant the puck stalls just inside the disk, ending
the episode before the expert pushes to centre. Those episodes are genuine successes and
useless as Mimic templates: Mimic replays a source rigidly, so every generated copy would
inherit ~5 cm of error against a 5 cm gate and fail on any slip. **A perfect expert SR was
hiding systematically bad data** -- the metric I had been tuning against for fifteen cycles
was not the metric that matters for source demos.

Fixed in the right place: the recorder tightens the success radius to 2 cm FOR RECORDING
ONLY (`--source_success_radius`), leaving the level's real 5 cm gate untouched for
generation and evaluation. Re-recorded: **median final error 1.93 cm, 0 of 20 above the
2.5 cm bar**, at an expert SR of 0.69 against the harder 2 cm gate (the 5 cm gate still
scores 85-94%). This also retires the reverted blade-clearance clause for good -- the same
goal, achieved by selection at recording time instead of by distorting the success
criterion.

**One latent bug found by reading rather than by failing:** APPROACH_XY had no tick budget,
though DESCEND and PUSH both got one. Its gate is XY_TOL = 6 mm and the arm's steady-state
tracking error is ~6 mm, so it could hang indefinitely; the traverse height had been
masking it. Given a 160-tick budget.

**Next:** annotate the sources (`--auto`) and run a generation smoke test. That is the
first real test of D19's single-subtask design -- whether Mimic accepts a one-element
`subtask_configs` and whether the synthetic push frame reproduces strokes at new bearings.

### 2026-08-17 23:40-23:55 — T3 MIMIC PIPELINE VALIDATED: 93 % generation SR, the highest of the three tasks

Annotation and generation both work, and the result overturns what I expected.

**Annotation:** `T3_L2_source_annotated.hdf5`, **17 of 20** sources accepted (three dropped
on replay). Mimic accepts a ONE-element `subtask_configs` -- the single-subtask design of
D19 is legal, which was the biggest open risk in the design. The annotated file's structure
is byte-for-byte the same shape as T2's known-good one (`datagen_info` is computed at
generation time, not stored; my expectation that it would appear in the file was wrong).

**Generation smoke, state env:** 12/12 successes, 0 failures, median final placement error
1.65 cm.
**Generation smoke, VISUOMOTOR env (what the wave uses):** **40 successes / 43 attempts =
93.0 % generation SR**, median final error 1.75 cm, max 4.90 cm (inside the 5 cm gate).
Episode lengths 265-342 steps. Full obs contract present: table_cam, wrist_cam, eef_pos,
eef_quat, gripper_pos, joint_pos, joint_vel, object_pos, object_quat, target_pos.

**Camera QA passed on the reused T1 framing** (`ops/qa/T3_smoke_grid.png`): in all sampled
episodes the yellow puck and the green target disk are both clearly visible and well
separated at t=0, and the final frames show the puck resting on the disk. Green
(target-disk) pixel count per first frame over 20 episodes: min 156, median 237, max 316 --
no episode has a clipped or occluded target. No re-aiming needed, so T3 inherits T1's
frozen camera exactly, which also keeps the visual domain identical across tasks.

### The finding: generation SR measures DESIGN FIT to Mimic, not task difficulty

Generation success rate across the three tasks:

| task | what it is | gen SR |
|---|---|---|
| T1 cup_place | short prehensile pick-and-place | 85-88 % |
| T2 drawer_stow | long-horizon articulated, 3 subtasks | **31-55 %** |
| T3 push_target | non-prehensile contact-rich | **93 %** |

Task 3 is by far the hardest to CONTROL -- it took fifteen fix cycles to get a scripted
expert to 85-94 %, against T1's handful -- and yet it has the highest generation SR of the
three. The reason is that its design was built around Mimic's rigid single-reference
transform rather than in spite of it: one subtask so there are no boundaries to mis-segment
and no interpolation jump mid-stroke, a synthetic reference frame that encodes the push
direction so direction adapts for free, a constant stroke length because a rigid transform
carries no scale, and sources selected for placement quality so no template hands its error
to its copies.

That reframes the T2 finding rather than contradicting it. What costs generation SR is not
"generality" or "difficulty" in the abstract, but **the number of independent
pose-dependent relations a task requires that a single rigid reference per subtask cannot
express**. T2's stow needs the cabinet pose AND the drawer opening AND the box pose, across
three chained segments; each added randomization axis degrades a transform that was already
approximating. T3 needs exactly one relation, and it is the one the reference frame encodes.

Practical consequence for anyone budgeting Mimic data: generation SR is largely a design
variable, not a fact about the task. It is worth spending a day on the subtask
decomposition and reference frames before spending three on generation compute.

**Next:** the T3 datagen wave (13 sub-levels; at 93 % SR and ~300-step episodes this should
be far cheaper than T2's 13 h -- estimate ~3-4 h), then conversion, QA and eval-set freeze.

### 2026-08-18 00:00 — T3 datagen wave launched (third attempt); two tmux-environment traps

The wave is running: L0 at **97.3 % generation SR** a few minutes in, GPU 57 %, output
growing. Estimated ~3 h for all 13 legs (~33 min per 400-demo level, plus the ten L3 boots),
so it should land around 03:00. Watcher armed.

It took three launches, and both failures were environment, not logic:

1. **`python: command not found`, all 13 legs exit 1 instantly.** `isaaclab.sh` shells out
   to `python` and needs it ON PATH; a tmux bash has no conda function. The earlier T2 wave
   only worked because it inherited an already-activated environment from the launching
   shell. Fixed by exporting the env's bin onto PATH inside the script -- more robust than
   `conda activate`, which needs the shell hook sourced. This is the same trap that killed
   the first T2 conversion launch; the lesson had been journaled and I still hit it, because
   last time the fix was "call the absolute interpreter" and that does not work when a
   third-party script insists on the bare name.
2. **Hung forever on Kit's EULA prompt.** `Do you accept the EULA? (Yes/No):` blocks when
   Kit sees a TTY -- which a tmux pane provides and the piped Bash-tool shell does not. That
   is precisely why the identical command ran fine interactively minutes earlier and then
   hung in tmux, with the session alive, no GPU use, and no output file: a stall that looks
   like a slow boot. Fixed with `export OMNI_KIT_ACCEPT_EULA=YES` **and** `< /dev/null` on
   every launch.

**Rule for this repo: a tmux/cron launch differs from an interactive one in at least three
ways -- no conda, a TTY, and a clean env -- so any script meant for tmux must set its own
PATH, accept the EULA explicitly, and redirect stdin from /dev/null.**

**Also repeated a foot-gun I had already written down:** `pkill -f <pattern>` where the
pattern appears in my own command line killed my own shell mid-sequence (exit 144), leaving
the edit and relaunch un-run. Replaced with: `pgrep`, then verify each candidate's
`/proc/<pid>/cmdline`, then `kill` by PID. Worth noting that the first attempt at even that
was wrong -- I quoted the PID list so `/proc/$PIDS/cmdline` expanded to a single bogus path.

### 2026-08-18 00:36 — T3 L0 leg done: 98.5 % generation SR, 29 min for 400 demos

`GEN_T3_L0_EXIT=0`. Exact counts from the HDF5 pair (D16): **400 successes / 406 attempts =
98.5 % gen SR**, mean episode 317 steps, 5.21 GB, **29 min** wall. L1 now generating at
~90 %.

For scale, the same 400 demos by task:

| | gen SR | attempts for 400 demos | mean ep len | wall per level |
|---|---|---|---|---|
| T1 L0 | 86.4 % | 463 | 207 | ~25 min |
| T2 L0 | 54.9 % | 728 | 705 | 137 min |
| **T3 L0** | **98.5 %** | **406** | **317** | **29 min** |

T3 L0 needs 6 wasted attempts to produce 400 demos; T2 L0 needed 328 and T2 L2 needed 906.
The 13-leg T3 wave should therefore finish in ~3 h against T2's 13 h 10 min, for the same
1600 demos -- a 4.4x saving that comes almost entirely from designing the task around
Mimic's transform rather than from the task being easier (it is not; its expert took 15 fix
cycles against T2's 29 but at a far lower success ceiling).

### 2026-08-18 02:06 — T3 DATAGEN WAVE COMPLETE: 13/13 legs, 1600 demos in 2 h 09 min

All thirteen `GEN_T3_*_EXIT=0`, `T3_WAVES_DONE`. Wave ran 23:56 -> 02:05 = **2 h 09 min**
for the same 1600 demos that took T2 **13 h 10 min** -- a **6.1x** speed-up.

Exact per-leg SR (D16, from the HDF5 pairs):

| level | successes/attempts | gen SR | wall |
|---|---|---|---|
| T3 L0 | 400/406 | **98.5 %** | 29 min |
| T3 L1 | 400/422 | **94.8 %** | 32 min |
| T3 L2 | 400/421 | **95.0 %** | 31 min |
| T3 L3 (pooled 10 variants) | 400/452 | **88.5 %** | 3-4 min each |

Per-variant L3 SR ranges 80.0-93.0 % with no monotone trend in radius or height, i.e. the
residual spread is sampling noise at n=40-50, not a geometry effect. Notably the expert's
own SR *did* fall monotonically with radius -- which is why the radii were re-spaced into
0.032-0.045 -- and having done that, GENERATION is now flat across the geometry axis.

### Three tasks, three distinct generation-SR signatures

| level | T1 cup_place | T2 drawer_stow | T3 push_target |
|---|---|---|---|
| L0 | 86.4 % | 54.9 % | **98.5 %** |
| L1 | 85.8 % | 44.2 % | 94.8 % |
| L2 | 85.1 % | 30.6 % | 95.0 % |
| L3 | 87.9 % | 32.7 % | 88.5 % |
| **pattern** | flat | steep collapse | near-flat, high |

This is now a three-point result rather than a two-point contrast, and it says the
determining factor is not task difficulty but **how many independent pose-dependent
relations the task requires versus how many a single rigid reference per subtask can
express**. T2 chains three subtasks over a cabinet pose, a drawer opening and a box pose,
so every added randomization axis degrades an already-approximating transform. T3 needs
exactly one relation and encodes it in the reference frame, so its curve barely moves --
despite being the hardest task to control (15 expert fix cycles, and a scripted expert that
tops out at 85-94 % where T1's exceeds 98 %).

**The paper claim this supports:** generation SR is a property of the DATA PIPELINE's fit to
the task, not of the task's intrinsic difficulty, and it is therefore a design variable. The
practical corollary for anyone budgeting Mimic data: a day spent on subtask decomposition
and reference frames bought 11 hours of generation compute here.

Conversion of all four levels launched in parallel (the T2 measurement: 4x throughput at
unchanged per-level speed, 2 h instead of 8 h).
