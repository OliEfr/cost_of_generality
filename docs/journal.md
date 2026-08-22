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

### 2026-08-18 02:50 — G3 PASSED FOR TASK 3 — ALL THREE TASKS' DATA PHASES ARE COMPLETE

Verified checklist, all green:

1. **HDF5 pools:** 1600 demos across 13 legs (400 x L0/L1/L2, 10 x 40 for L3).
2. **LeRobot datasets:** `data/lerobot/T3_{L0,L1,L2,L3}`, 400 episodes each, every one
   `VALIDATE_OK` at `--expect_episodes 400`; conversion took **38 min** for all four in
   parallel (124-128k frames each). L3 verified variant-balanced: exactly 40 per variant,
   interleaved v00..v09.
3. **Eval sets frozen:** `configs/eval_sets/T3_{L0,L1,L2}.json` + `T3_L3.json`, on the
   corrected diagonal protocol (D18) -- verified 200 episodes / **200 distinct poses**.
   Invariance per spec: L0 all frozen; L1 puck 11.8 x 11.6 cm with the target following it
   rigidly (it is derived at a fixed distance and bearing); L2 target spread widens to
   23.1 x 11.6 cm as the bearing varies. All nine T1/T2 eval files md5-verified UNCHANGED.
4. **`experiments/gen_stats.csv`:** 13 T3 rows with exact counts.
5. **QA:** 13 artifacts; asserts pass on all four levels. Travel along the push axis
   medians 18.4-19.2 cm against a 20 cm nominal; final placement error medians 1.67-2.29 cm,
   p95 3.6-4.3 cm.

**Study total: 4800 validated demonstrations** -- 1600 per task, 12 levels, 39 frozen
eval sub-levels. Every GPU-bound stage that can run locally is now done. The study is
blocked entirely on G0 (the Slurm association) for training.

**Two QA/tooling issues found and fixed while closing out:**
- One episode per level on L1/L2 ends 0.01-0.04 mm past the 5 cm gate. Not a defect: the
  env's success fires when the puck is inside AND settled, and the LAST RECORDED frame
  catches a fraction of a millimetre of post-success coast. The QA assert now carries a 5 mm
  settle tolerance and reports the overrun count instead of failing on it -- a real defect
  would show as many episodes or a large excess.
- **The freeze wave reported `EXIT=0` for all 13 legs while all 13 had died** on
  `NameNotFound` (I had added the `--task_kind push_target` branch to
  `freeze_eval_sets.py` but never its `import cog.tasks.push_target` registration line).
  An Isaac script can exit 0 after a fatal exception because Kit's shutdown path swallows
  it, so **exit codes are not a trustworthy success signal for Kit scripts -- the printed
  marker is**. The wave script now explicitly flags a missing `EVALSET_OK`. Worth
  generalising: every Isaac wave in this repo greps for a marker, and this is why.

### 2026-08-18 04:40 — Cluster-side artifacts authored ahead of G0 (UNVERIFIED)

Everything local is finished and the study is blocked on the Slurm association, so the hour
went on the artifacts that sit on the critical path the moment G0 clears. `slurm/` was empty
and there was no launch or sync tooling; now:

- **`slurm/train.sbatch`** — one study cell, `sbatch slurm/train.sbatch T1 L0 25`.
  `-A euhpc_b38_106` explicit (CLAUDE.md rule 4), 1 GPU / 8 cpus / 64 G / 24 h,
  auto-`--resume` if a checkpoint exists, `WANDB_MODE=offline` + `HF_HUB_OFFLINE=1` (a
  single online call would hang the job on a compute node with no internet), and the
  nested-subset episode list generated as `0..N-1` from the committed shuffle order (D4), so
  a larger cell is a strict superset of a smaller one.
- **`slurm/debug_a100_render.sbatch`** — the G5b gate on `boost_qos_dbg`, 30 min. Runs
  `frames_qa.py` inside the Singularity image and decides PASS/FAIL on **whether the frame
  artifact exists**, not on the exit code, because Kit can exit 0 after a fatal exception
  (learned tonight from the freeze wave). Sets `OMNI_KIT_ACCEPT_EULA=YES` and redirects
  stdin, the two things that stalled the T3 wave.
- **`scripts/ops/launch_matrix.py`** — submits a task's 24-cell grid as one parallel wave,
  writes a registry row per cell with its job id, and SKIPS cells already present and not
  failed, so re-running after a partial submission is idempotent. Dry-run verified: 24
  cells for T1.
- **`scripts/ops/sync_up.sh`** — code to `$WORK/cog/repo` (excluding data, third_party,
  ops), datasets to `$FAST/cog/datasets` per task.

**All four are UNVERIFIED against the real cluster** -- every path, module name and flag
comes from `docs/PINS.md` and the plan, not from a successful run. Noted in the file headers.
First use after G0 must be a SINGLE cell, not the matrix, and the A100 render gate before any
sim work is planned there.

### 2026-08-18 05:50 — Analysis module built, and a Monte Carlo of the study's RESOLUTION

`src/cog/analysis/curves.py` written and tested end-to-end against synthetic eval files with
a known ground-truth logistic: it reads the `eval_<level>_n<N>_<step>.json` schema
`rollout_eval.py` writes, collapses checkpoints to best-of-last-three (carrying the
last-checkpoint SR alongside), fits a logistic in log N, and reports N*(50/80/90 %) plus cost
ratios, printing ">400" rather than extrapolating when a curve never crosses in range.

Then, because a single synthetic draw suggested the fit-based N* was systematically low, I
ran a proper Monte Carlo (`scripts/dev/nstar_resolution.py`, 400 trials per condition):

| estimator | bias | \|err\| p90 (100 eval eps) | \|err\| p90 (200 eval eps) |
|---|---|---|---|
| interpolated crossing | **+3 to +5 %** | 27-30 % | 22-23 % |
| logistic fit | **0.0 %** | 15-23 % | 12-15 % |

**The fit is unbiased and ~40 % tighter**; the interpolated crossing is biased high because a
single noisy cell moves it a long way when the grid doubles between cells. So the paper should
report the FIT as primary N*, with the interpolated crossing as an assumption-free secondary.
The module now prints both.

**This also corrects a conclusion I had drawn two steps earlier from ONE draw**, where the fit
looked systematically 13-23 % low. It was noise. Comparing two estimators on a single sample
is worthless -- the same error class as judging expert SR from a 64-episode run earlier
tonight. If I am comparing estimators or configurations, it needs many trials or it needs no
conclusion.

### What the study can and cannot resolve

For a true cost ratio of 2.0x between two levels, at 100 eval episodes per cell:
**median estimate 2.03x, with 90 % of estimates in [1.57x, 2.65x]**.

So the design comfortably resolves a 2x data-cost difference, and **an effect below about
1.3x is not distinguishable from no effect at all** at this eval budget. That matters for how
the results must be worded: if a level's measured ratio comes out near 1.2x, the honest
statement is "below our resolution", not "generality is free". Doubling to 200 episodes (the
protocol's headline rerun) cuts N* error p90 from ~17 % to ~13 % -- worth spending on the
cells that carry a claim.

Worth having run this BEFORE the matrix rather than after: it is a statement about the design,
and while the eval budget is still a choice it is actionable.

### 2026-08-18 06:45 — Figures module, and the study's first finished paper figure

`src/cog/analysis/figures.py`: one independent function per figure, each skipped with a
message rather than faked when its inputs are absent. `fig_sr_vs_n` waits on
`experiments/curves.csv` (needs evals, so it is correctly skipped today); **`fig_gen_sr` is
ready now from real data** and is written to `paper/figures/fig_gen_sr_vs_level.png`.

It plots generation SR against generality level for all three tasks -- T1 flat at 85-88 %,
T2 collapsing 54.9 -> 30.6 %, T3 near-flat at 88-98 % -- with the 30 % G3 floor marked. The
figure makes the point visually that the ordering is not by task difficulty: T3, the hardest
task to control, generates best, because its design matches Mimic's single rigid reference.
That is the study's first complete, publishable result, drawn entirely from measurements
rather than estimates.

Small fix after inspecting the render: T1 and T3 converge at L3 (87.9 vs 88.5) and their value
labels overlapped, so per-task label offsets now push them apart.

### 2026-08-18 07:40 — Cluster tooling complete (all still unverified)

Closed the last two gaps: **`slurm/eval.sbatch`** (evaluates a cell's 40k/60k/80k checkpoints
on the frozen eval set, skipping results that already exist so a requeue is idempotent, and
deciding success on the RESULT FILE rather than the exit code because Kit can exit 0 after a
fatal exception) and **`scripts/ops/sync_down.sh`** (deliberately narrow: eval JSONs freely,
but only the 80k and last checkpoints per run, never whole checkpoint trees).

The full toolchain now exists end to end: datagen waves, conversion + validation, dataset QA,
eval-set freezing, rollout eval, curves/N* analysis, figures, Slurm train/eval/render-gate
templates, launch matrix, sync up/down, watchdog. Everything from G0 onward is authored but
**unverified against the real cluster** -- the first post-G0 actions should be, in order:
`sbatch --test-only` (G0 itself), the A100 render gate (G5b), ONE training cell (G5a), and only
then `launch_matrix.py`.

### 2026-08-18 08:35 — Leonardo certificate has EXPIRED

`~/cineca_login.sh --status` now reports NO VALID CERTIFICATE (it lapsed at 08:33 as
predicted). Consequences, none of them data-threatening: no `ssh leonardo`, so no `squeue`,
no `saldo -b` budget ledger, no `sbatch --test-only` G0 probe, and no sync in either
direction. The hourly watchdog's cluster checks will report unavailable until it is renewed.

Nothing local is affected -- all three tasks' datasets, eval sets and the full toolchain are
complete and committed. Renewal needs the user's laptop (tunnelled session, then
`~/cineca_login.sh`); the plain form works fine on an expired cert since it treats it as zero
seconds remaining.

### 2026-08-18 08:40 — Cert renewed; G0 probed precisely and still FAILING (but informatively)

Certificate renewed by the user: **valid 47 h 58 m** (until 2026-08-20 08:36). `ssh leonardo`
works again (`ohausdoe` on login05).

G0 probed properly for the first time in days, and the two halves disagree in a way that is
worth quoting verbatim to CINECA support:

- `sacctmgr -n show assoc user=ohausdoe` returns **only `euhpc_b34+`** (the expired B34
  project). There is NO association for B38.
- `sbatch --test-only -A euhpc_b38_106 -p boost_usr_prod --gres=gpu:1` ->
  **"allocation failure: Invalid account or account/partition combination specified"**.
- BUT `saldo -b` **does** list `EUHPC_B38_106`, 112,000 local hours, 0 consumed, valid
  20260729-20261029.

So the grant exists, is funded and is unconsumed, and the user is attached to it in the
ACCOUNTING database -- but Slurm has no association for them on it, which is what actually
authorises job submission. That is a sharper request for the PI or support than "I can't
submit": *saldo shows me EUHPC_B38_106 but sacctmgr shows no association for my user on that
account, so sbatch rejects it as an invalid account/partition combination.*

Scheduling changed per user request: the hourly pipeline watch (c08cfc1f) is **cancelled** --
nothing runs locally any more and all three data phases are closed, so it only produced noise.
Replaced with a **three-hourly compute-access check** (1676c977) that probes G0, and on
success writes `ops/G0_PASSED`, syncs code up, runs the A100 render gate, and then STOPS to
check with the user before anything expensive. Session-only, auto-expires in 7 days.

### 2026-08-18 08:50 — Monitoring discipline written into the agent instructions

User: "remember to set up hourly health watchers for all larger jobs. Also write this in the
project documentation / agent instructions for future agents."

- **CLAUDE.md rule 10:** every job over ~10 min gets all THREE of a tmux session, an event
  watcher, and an hourly cron fallback -- because each has failed alone here (a watcher that
  never fired left a finished wave unnoticed; an hourly poll alone would have missed the T3
  wave dying one second after launch).
- **New `docs/running_jobs.md`**, the launch checklist, written from the failures rather than
  from principle. Eight sections: the three monitoring layers with copy-pasteable snippets;
  the three ways a tmux shell differs from an interactive one (no conda hook, it HAS a TTY so
  Kit blocks on its EULA prompt, clean env) with the symptom and fix for each; exit codes lie
  and markers do not; liveness is an artifact question (mtime), not a process question
  (a spinning process ran 24 h after finishing its work in 13 min); never `pkill -f` with a
  pattern your own command line contains; never glob HDF5 inputs; parallelise across levels
  not within; and where the measured numbers live.

Rule 10 points at the doc explicitly, and the doc's opening line is "every rule here was paid
for", with dates back to the journal entries holding the traces -- so a future agent can check
the evidence rather than take it on faith.

### 2026-08-18 17:20 — Cluster-side blocker is now TWO things, not one (G0 probe degraded)

Routine three-hourly access check turned up two new cluster facts. Both matter for planning,
so recording them even though no pipeline action was taken.

**1. `sacctmgr` is currently unusable — the G0 probe is degraded, not clean.**

```
sacctmgr: error: slurm_persist_conn_open_without_init: failed to open persistent
  connection to host:slurm-slurmdbd.userservices.svc.kube.local:6819: Connection refused
```

Reproduced on an immediate retry. slurmdbd (the Slurm accounting daemon, which is where
associations live) is refusing connections. `sbatch --test-only -A euhpc_b38_106` still fails
with the same "Invalid account or account/partition combination", and `sinfo`/`squeue`/`saldo`
all work — so slurmctld is healthy and this is specific to slurmdbd.

Consequence for the gate: **today's probe cannot distinguish "the association is still
missing" from "the association exists but slurmctld can't load it while slurmdbd is down."**
Previous checks were clean negatives; this one is not. Do not read a failure during a slurmdbd
outage as evidence about the PI's UserDB action. Re-probe once `sacctmgr` answers again.

**2. The whole `boost_usr_prod` partition is drained — a live GPU-detection incident.**

```
sinfo -p boost_usr_prod -o "%T %D %E"
drained$    1  gres/gpu count reported lower than configured (0 < 4)
draining  173  gres/gpu count reported lower than configured (0 < 4)
drained   348  gres/gpu count reported lower than configured (0 < 4)
```

522 of 522 visible nodes, i.e. **zero schedulable nodes**; every node reports 0 GPUs against
4 configured. 739 jobs are still listed as RUNNING, which is consistent rather than
contradictory: `draining` nodes keep their current jobs and accept no new ones (739 running on
~174 draining nodes is ~4 per node = 1-GPU jobs on 4-GPU nodes). Nothing new can start.

So even the moment G0 passes, **no job will start until this clears** — the association and
the drain are independent blockers. Neither is actionable from here; both are CINECA-side.
The reservations list shows only long-standing `cin_sanity`/MAINT entries (nothing that
explains a cluster-wide GPU drain), so this looks like an unplanned incident, not a scheduled
window. **-> WRONG, corrected 30 min later: it is the scheduled Slurm upgrade. See the
2026-08-18 17:55 entry below before acting on anything in this paragraph.**

Planning impact: none on the critical path yet, because the association is still the outer
blocker and all local data phases are closed. But the drain is the thing to check *first*
after G0 passes — submitting the A100 render gate into a fully drained partition would just
queue indefinitely and look like a broken sbatch script on its unverified first use, which is
exactly the confusion to avoid. Check `sinfo -p boost_usr_prod -t idle,mix` before believing
any queue behaviour.

### 2026-08-18 17:55 — Correction: the drain is the announced Slurm upgrade, not a GPU fault

User forwarded the CINECA notice ("Leonardo: Slurm upgrade - UPDATE", 18 Aug 2026): the
upgrade had been delayed by "complications encountered during the preparatory activities", so
Slurm stayed up all morning, and they are *now* upgrading **the controller and database
daemons**. Job submission and queue queries unavailable; running jobs unaffected; pending jobs
stay queued and will not start until the service is restored.

**What I got wrong:** I read `gres/gpu count reported lower than configured (0 < 4)` across
every node plus a dead slurmdbd as an unplanned hardware/driver incident. It is the announced
maintenance. The node-side symptom is what a Slurm upgrade looks like from a login node —
slurmd restarting under a version transition stops reporting its GRES, so slurmctld drains the
node. I over-read a maintenance artifact as a fault, on the strength of "no reservation
explains it" — but a daemon upgrade needs no node reservation, so absence of a reservation was
never evidence for a fault. Cheap lesson: check the service-status mail before diagnosing the
cluster.

Confirming evidence 20 min later — the node states are churning, which a hardware fault would
not do:

```
17:20   drained$ 1  draining 173  drained 348                      (522 visible)
17:45   drained$ 1  maint 3  completing 1  draining 20  drained 188  reserved 9   (222 visible)
```

`sinfo --version` still reports `slurm 23.11.10-BullSequana.1.2.1`, i.e. mid-flight.
`squeue -u ohausdoe` answers (empty, as expected). `sacctmgr` still refuses.

**Standing correction to the G0 procedure:** while slurmdbd is down, a G0 failure is
**uninformative** — associations live in slurmdbd, so a `sbatch --test-only` rejection during
the upgrade says nothing about whether the PI added us. The three-hourly check keeps running
(it is the restoration detector), but its negatives are to be logged, not interpreted, until
`sacctmgr` answers again. **The first meaningful G0 probe is the first one after `sacctmgr`
responds** — and it is worth taking seriously, because a controller/DB upgrade reloads
associations from the database, so if the UserDB request has already been processed it could
appear at exactly that moment.

The two-independent-blockers point from the previous entry still stands in substance (a passing
G0 does not imply a schedulable partition, so check `sinfo -p boost_usr_prod -t idle,mix`
before trusting queue behaviour on the sbatch scripts' unverified first use) — only the *cause*
of the drain was misattributed, and the expected clearing is now "when maintenance ends"
rather than "unknown".

### 2026-08-18 20:20 — Upgrade phase 2: slurmdbd back, slurmctld down. Association still absent.

Three-hourly check. The failure mode has moved, which is why this one is worth recording rather
than logging as another negative.

```
sacctmgr -n show assoc user=ohausdoe   ->  euhpc_b34_046  boost_qos_bprod,boost_qos_dbg+
sbatch --test-only -A euhpc_b38_106 ...  ->  (no output)   RC=124   <- killed by `timeout`
scontrol ping                            ->  (no output)   RC=124   <- killed by `timeout`
sinfo --version                          ->  slurm 23.11.10-BullSequana.1.2.1  (local binary)
sinfo -p boost_usr_prod                  ->  empty
```

So **slurmdbd is back** (sacctmgr answers, and answers correctly — it still lists the B34
association) while **slurmctld is now unreachable** (`scontrol ping` hangs). That matches the
notice's "controller and database daemons" as a two-stage rollout, DB first.

**What this means for G0, and it is a genuine change:** `sacctmgr` talks to slurmdbd directly,
not through the controller, so **the association read is trustworthy again even though the
sbatch probe cannot run.** As of now there is still **no `euhpc_b38_106` association for
ohausdoe** — a real negative, not an artifact like the 17:20 one. The PI/UserDB request has not
landed yet. (Mild caveat: the DB is freshly restarted mid-upgrade; confirm with sbatch once the
controller returns.)

**Two diagnostic traps caught here, both worth remembering:**

1. **`sbatch --test-only` printing nothing is NOT a pass.** On success it prints
   `sbatch: Job N to start at <time> using <n> processors on nodes <list>`; on rejection it
   prints an allocation-failure line. Silence means it never reached the controller — here,
   `timeout` killing a hang. The first probe of this round returned empty output and I nearly
   had a "no error" reading on my hands; the RC settled it. **Always capture the RC** — G0 is
   `RC==0 AND a "Job N to start" line`, never "no error text".
2. **I piped `sinfo`/`squeue` through `head`, so their `$?` was head's, not theirs** — those two
   RC=0 values in the probe output mean nothing. Only the two unpiped commands
   (`sbatch --test-only`, `scontrol ping`) carried real exit codes, and both were 124. Same
   family of mistake as reading a stale error message after an auto-reset: check *what* the
   number you are reading is actually attached to.

No action taken; nothing submitted (nothing could be). Next check continues on the three-hourly
schedule. The probe to trust is `scontrol ping` succeeding -> then re-run the full G0.

### 2026-08-18 23:20 — Upgrade DONE, partition healthy, G0 still fails: association is the sole blocker

The post-upgrade probe I flagged as "the first meaningful one" has now run. It is negative, and
this time nothing is confounding it.

**Slurm is fully back:**
```
scontrol ping                      -> Slurmctld(primary) at slurm-controller is UP   RC=0
sacctmgr -n show assoc ohausdoe    -> euhpc_b34_046  boost_qos_bprod,boost_qos_dbg,+  RC=0
sinfo -p boost_usr_prod            -> idle 702 | mixed 319 | allocated 2214 | draining 17 | ...
```
The GPU drain from the 17:20 entry is gone (702 idle nodes vs zero schedulable), confirming
again that it was the upgrade, not hardware.

**G0 fails, with a control to prove the probe itself is sound:**
```
-A euhpc_b38_106 -> "allocation failure: Invalid account or account/partition combination
                     specified"                                              RC=1
-A euhpc_b34_046 -> "sbatch: error: invalid account or expired budget"
                     "allocation failure: Unspecified error"                 RC=1
```
Ran the expired B34 account as a control deliberately. The two accounts fail with **different
messages**, which is the informative part: B34 (which *has* an association) reaches
account/budget validation and is rejected for the expired budget, while B38 is rejected one step
earlier as an invalid account/partition combination. So the `-p boost_usr_prod --gres=gpu:1
--cpus-per-task=8 --mem=64G` parts of the submit line are well-formed and accepted -- the
rejection is specifically about the missing association, not my sbatch syntax. (Caveat: with no
valid account available I still cannot exercise the *success* path, so the "Job N to start at"
branch of the G0 criterion remains untested.)

**Why this probe mattered more than the previous ones:** a controller+DB upgrade reloads
associations from the database, so if the UserDB request had been processed at any point it would
be visible now. It is not. That excludes the "it exists but is not propagated" hypothesis that
made the 17:20 check uninterpretable. **The Slurm association for `EUHPC_B38_106` has genuinely
not been created.** Both cluster-side blockers are now resolved down to one, and it is the one
only the PI or CINECA support can clear.

Exact wording for the request (unchanged, still the sharpest framing):
> `saldo -b` lists EUHPC_B38_106 for my user (112,000 local h, 0 consumed, valid 20260729-
> 20261029), but `sacctmgr show assoc user=ohausdoe` shows no association on that account --
> only the expired euhpc_b34_046 -- so `sbatch -A euhpc_b38_106 -p boost_usr_prod` is rejected
> as an invalid account/partition combination. Please add ohausdoe to EUHPC_B38_106 in Slurm.

Everything on our side is ready and waiting: all three tasks' data phases closed (4,800 demos,
39 frozen eval sub-levels), and the cluster scripts authored but unverified. First action the
moment G0 passes is still `sync_up.sh code` + the ~30 min A100 render gate on `boost_qos_dbg`,
then STOP for user approval before the 8 GPU-h calibration run.

## 2026-08-19 15:30 — G0 PASSED: Slurm association is live; cluster phases unblocked

The `EUHPC_B38_106` Slurm association was created some time before 15:07 today. The
hourly watchdog detected it first (wrote `ops/G0_PASSED` + an `ops/ALERTS.md` line at
15:07); the three-hourly compute-access check confirmed it independently at 15:26.
Elapsed block: ~3.5 days from first detection (2026-08-16) to resolution.

**G0 evidence — both halves of the criterion, not merely an absence of errors:**

```
sacctmgr -n -P show assoc user=ohausdoe format=Account,Partition,QOS
  euhpc_b34_046||boost_qos_bprod,boost_qos_dbg,boost_qos_lprod,normal
  euhpc_b38_106||boost_qos_bprod,boost_qos_dbg,boost_qos_lprod,normal   <-- NEW
sbatch --test-only -A euhpc_b38_106 -p boost_usr_prod --gres=gpu:1 --cpus-per-task=8 --mem=64G
  sbatch: Job 52803454 to start at 2026-08-26T05:17:15 using 8 processors on nodes lrdn2541
  RC=0
```

`saldo -b`: EUHPC_B38_106, 112,000 local h, 0 consumed, valid 20260729-20261029.
All four QOS we need (`boost_qos_dbg` for gates, `boost_qos_bprod` for the matrix,
`boost_qos_lprod` for any 4-day run) are attached to the new association.

**FINDING — the queue is deep, and this changes wall-clock planning, not cost.**
`boost_usr_prod` shows 1,575 pending / 2,050 running jobs. `--test-only` estimates our
start at 2026-08-26 (~6 days out) for `normal` QOS and 2026-08-25 even under
`boost_qos_dbg`. Two caveats before anyone plans around those dates: (a) `--test-only`
start estimates are pessimistic by construction -- they assume every job ahead consumes
its full walltime and give no credit for backfill, which a 30-min 1-GPU job is an ideal
candidate for; (b) the estimate is a snapshot of a queue that just reopened after the
Slurm upgrade, so it is inflated by the backlog that accumulated while submissions were
refused. Real queue latency has to be *measured*, not predicted -- the A100 render gate
is now doing double duty as that measurement. Plan implication: the 24-run matrix's
"~1 day at reasonable queue depth" assumption (P6) is the number most at risk, and it is
a wall-clock risk only -- GPU-h cost is unchanged.

**Cluster tree created** (did not exist; `rsync --delete` creates only the final path
component, so this had to precede any sync):
`$WORK/cog/{repo,miniforge3,envs,checkpoints,wandb_offline,results,containers,hf_cache,datasets_backup}`
plus `$FAST/cog/datasets`. Both areas verified writable by touch+rm.
Quota: `$WORK` = /leonardo_work/EUHPC_B38_106, 3 T (4 K used); `$FAST` = 1 T. Ample.

**Verified gotcha that did NOT bite:** `$WORK`/`$FAST`/`$SCRATCH` *do* expand under a
non-interactive `ssh leonardo '...'` (they resolve via profile scripts that run even for
command-mode ssh). `sync_up.sh` relies on this -- it single-quotes `$WORK/cog` and lets
the remote shell expand it. Had they been login-shell-only, the rsync would have written
to a relative directory named by an empty string instead of failing loudly. Checked
before first use rather than after.

**-> Half of that inference was WRONG, corrected 15 min later. The ssh-expansion claim
holds, but rsync never passes its destination path through a shell, so `sync_up.sh` was
in fact broken and failed on first use. Verifying ssh told me nothing about rsync. See
the 2026-08-19 15:45 entry before trusting this paragraph.**

Note `$WORK` now points at EUHPC_B38_106; before the association it did not resolve to
this project's area. Anything cached from an earlier probe of `$WORK` is stale.

## 2026-08-19 15:45 — Gate (a) code sync PASSED after two real bugs; G5b BLOCKED on a missing container

### Bug 1: rsync does not shell-expand the remote path (both sync scripts were broken)

First real invocation of `sync_up.sh code` died with:

```
rsync: mkdir "/leonardo/home/userexternal/ohausdoe/$WORK/cog/repo" failed: No such file or directory (2)
rsync error: error in file IO (code 11)
```

The literal string `$WORK` was resolved against `$HOME`. My earlier entry today reasoned
that single-quoting `'$WORK/cog'` was fine "because the remote shell expands it" -- half
right, and the wrong half is the one that mattered. `ssh leonardo 'echo $WORK'` *does*
expand (verified). But rsync hands the destination to the remote `rsync --server` as a
protected argument; no shell ever sees it. Verifying the ssh behaviour told me nothing
about the rsync behaviour, and I treated the one as evidence for the other.

Fix in `sync_up.sh` and `sync_down.sh` (both carried it): resolve the bases locally in one
ssh round-trip and fail loudly if they come back empty --

```bash
read -r _work_base _fast_base < <(ssh "${REMOTE}" 'echo "$WORK" "$FAST"')
[ -z "${_work_base:-}" ] && { echo "could not resolve \$WORK ..." >&2; exit 3; }
WORK_REMOTE="${_work_base}/cog"
```

The empty-check matters: a cert expiry or a lost association would otherwise silently
produce `/cog/repo` and write somewhere unintended.

### Bug 2: rsync does not read .gitignore -- 3 GB of local smoke checkpoint went up

The fixed sync then succeeded but moved **2.9 GB**. Cause: `experiments/runs/g4_smoke_L0_n25/`
(a 2.0 GB optimizer state + 1.0 GB model from the G4 local smoke) is gitignored, so it never
showed in `git status` and I had been treating "clean tree" as "small tree". rsync has no
notion of gitignore. Added `--exclude 'experiments/runs/'` and removed the remote copy
(inside `$WORK/cog`, our own mirror per CLAUDE.md rule 1; the local original is untouched
and it is a regenerable smoke artifact either way). Re-sync now sends 5 KB against a
3.3 MB tree -- idempotent. **Gate (a) PASSED.**

Lesson worth generalising: for rsync, `.gitignore` and `git status` are not inventories.
Size the payload with `du` before the first push of any tree, not after.

### G5b (A100 render gate): BLOCKED, not failed

`$WORK/cog/containers/isaaclab.sif` does not exist, and nothing else usable is on the
system -- the only `.sif` files reachable are `jim/TRELLIS/trellis.sif` and
`jim/ubuntu24.sif` in the **expired B34** area: another user's directory, unrelated
content, not ours to touch. `singularity` itself is fine (SingularityPRO 4.3.1-1.el8).
`$WORK/cog/miniforge3/` is also empty, so there is no cluster training env yet either.
Per instruction I am reporting the gap rather than improvising a ~20 GB Isaac image build.

**Recommendation: deprioritise G5b, do the training env first.** The reasoning changed
now that all datagen is finished. G5b originally gated *demo generation* on the cluster,
which was the expensive rendering workload -- but every dataset for all three tasks is
already generated locally (12/12 present, 2.2 GB total in LeRobot form). The only
remaining cluster-side use for Isaac would be the 72 checkpoint evals, and those have an
accepted local fallback (~2-3 d serialised on the 4090). So G5b now buys at most ~2 days
of eval wall-clock, in exchange for building a large container for a GPU on which Isaac
Sim is officially unsupported and may fail anyway. Training is the long pole and does not
need Isaac at all -- it needs miniforge + LeRobot + torch, which is a login-node install
costing zero GPU-h.

### Dataset inventory (relevant to the next sync)

12/12 datasets present: T1 L0-L3 ~75-81 MB each, T2 ~343-356 MB, T3 ~115-120 MB;
**2.2 GB total**. `$FAST` quota is 1 T, so the read-hot staging plan has ~500x headroom
and the "<20 GB" budget line in the plan was far too conservative.

### Bug 3 (same gate, found by verifying rather than trusting): unanchored rsync excludes

After the sync reported success I checked *what had actually arrived* rather than trusting
`SYNC_UP_OK`, and found `$WORK/cog/repo/scripts/` contained only `dev/` --
**`scripts/ops/` was missing entirely, including `launch_matrix.py`**, the script that
submits the whole 24-run matrix.

Cause: rsync filter-rule semantics. A pattern with no `/` in it is matched against the
*final path component at any depth*; only a pattern containing a slash is matched against
the path from the transfer root. So `--exclude 'ops/'`, written to skip the top-level
`ops/` log directory, also matched `scripts/ops/`. Same latent trap in `data/` and
`third_party/` (harmless today -- no nested dirs by those names -- but wrong for the same
reason). Fixed by anchoring: `/ops/`, `/data/`, `/third_party/`, `/experiments/runs/`.
`.git/`, `__pycache__/` and `*.hdf5` stay unanchored deliberately -- those we do want
dropped at any depth.

Re-sync now lands 13 ops scripts, `launch_matrix.py` among them; `repo` = 3.5 MB with
top-level `ops/` and `data/` correctly absent. **Gate (a) PASSED for real.**

This bug would not have surfaced until matrix-launch time, as a "command not found" after
G0/G5a had been signed off -- the most expensive place to discover it. Three bugs in one
25-line script on its first real use is the argument for one-small-thing-at-a-time against
the real cluster, and for making the *artifact* the check rather than the exit code (the
same rule CLAUDE.md rule 9 and the Kit-exits-0 gotcha already encode).

### Login-node facts for the env build (P5 prerequisite)

`ulimit`: max memory size 1 GB (Linux typically does not enforce `-m`), stack 100 MB,
65536 open files; node has 128 cores / 502 GB with ~217 GB available. Compute nodes have
no internet, so a login-node install is the only route for pip/conda; if cgroup limits
kill a large install, the fallback is to download wheels on the login node and install
from cache inside an `srun` step. Not yet attempted -- recorded so the attempt is planned
rather than improvised.

---

## 2026-08-19 17:35 — Cluster bring-up: queue latency measured (the earlier warning was wrong), A100 node facts, four real bugs in `train.sbatch`

User approved (2026-08-19, after the cycle-11 report) steps 1-4 of the proposed sequence:
cluster training env, dataset sync, batch/LR utilization smoke, 80k calibration run -- plus
a go-ahead to attempt the optional cluster-eval path afterwards. Cert renewed by the user
from their laptop, now valid until **2026-08-21 17:24** (47h59m at 17:24).

### FINDING (retraction): the queue-depth risk I reported does not exist

In the 15:30 entry I flagged `boost_usr_prod`'s 1,575 pending / 2,050 running jobs and the
`sbatch --test-only` estimate of a start at `2026-08-26T05:17` as "the number most at risk"
for P6. That estimate is **wrong by five orders of magnitude**, and I could only find that
out by submitting something.

A 5-minute `nvidia-smi` probe on `boost_qos_dbg` (job 52869585, ~0.02 GPU-h):

```
submitted 2026-08-19T17:30:12
started   2026-08-19T17:30:16     <- 4 seconds
ended     2026-08-19T17:30:18     Elapsed 00:00:02, State COMPLETED
AllocTRES billing=8,cpu=8,gres/gpu=1,mem=64G
```

**4 seconds, not 6 days.** `--test-only` reports the reservation-only worst case: it answers
"when could this start if nothing ahead of it ever finished early and no backfill happened",
which on a large machine with thousands of short jobs bears no relation to reality. The
lesson generalises beyond Slurm: *a scheduler's own estimate of itself is not a measurement.*
The cheapest possible real job (2 s of one GPU) was worth more than any amount of reasoning
about queue depth. **P6's "training wave completes in ~1 day" assumption stands; there is no
wall-clock risk to plan around.** I over-warned the user, on a number the machine handed me.

### A100 node facts (lrdn2752, from the same probe)

| Property | Value |
|---|---|
| GPU | NVIDIA A100-SXM-64GB, 65536 MiB |
| Compute capability | 8.0 (sm_80) |
| Driver | 535.274.02 |
| Driver's CUDA | 12.2 |

**Implication for the torch pin.** We install torch 2.7.0+**cu128** to match the local eval
env version-for-version. cu128 > the driver's 12.2, which is fine *only* by CUDA minor
version compatibility (any 12.x runtime on a >=525 driver), and sm_80 is in the cu128 binary
so no PTX JIT is needed. This is a genuine assumption, so it is checked empirically in the
G5a smoke and not before: `torch.cuda.is_available()` plus a real GPU matmul. **Fallback if
it fails: torch 2.7.0+cu126.** That fallback costs nothing scientifically -- the checkpoint
format is a build-independent safetensors dir, so a cu126-trained checkpoint still loads in
the local cu128 eval env. Recorded as D22.

### Four real bugs in `slurm/train.sbatch`, all found by reading the installed 0.4.4 source

The file carried an explicit "UNVERIFIED until G0 clears" banner, and it earned it. None of
these would have been caught by a syntax check; two fail late, and two fail *silently*.

1. **`python -m lerobot.scripts.train` does not exist at 0.4.4.** The module is
   `lerobot.scripts.lerobot_train` (console script `lerobot-train`). Verified against the
   installed tree: `scripts/` contains only `lerobot_train.py` and
   `lerobot_train_tokenizer.py`. Failure mode: instant `No module named`, 8 h of queue
   position thrown away per cell.
2. **`mkdir -p "${OUT}"` guaranteed a crash.** `configs/train.py:119` raises
   `FileExistsError` when `output_dir` is an existing directory and `--resume` is unset. The
   script created the very directory that makes LeRobot refuse to start. Now only the parent
   is created.
3. **`--resume=true` alone cannot work.** `configs/train.py:89-95` raises "A config_path is
   expected when resuming a run" -- resume reads the whole config *from the checkpoint*, so
   the resume invocation must be `--config_path=<ckpt>/pretrained_model/train_config.json
   --resume=true` and nothing else. This is exactly the path a 24 h-walltime requeue takes,
   i.e. the bug would have surfaced only after a 24 h job hit its limit, and then again on
   every requeue: **the G5a resume test would have failed for a reason unrelated to resume.**
4. **`--optimizer.lr` is silently ignored.** With `use_policy_training_preset=true` (the
   default) `configs/train.py:134-136` *replaces* `cfg.optimizer` wholesale with
   `policy.get_optimizer_preset()`, which reads `policy.optimizer_lr` (default 1e-4). So the
   real knob is **`--policy.optimizer_lr`**. Worst bug of the four by a wide margin: after
   G5a raises the batch size, every one of the 24 cells would have trained at lr=1e-4
   instead of the sqrt-scaled value, with no error, no warning, and a plausible-looking loss
   curve. The scientific claim would have been quietly about the wrong hyperparameter.

Plus one more, of the same silent family: `WandBConfig.mode` defaults to `None`, and
`rl/wandb_utils.py:109` then passes `mode="online"` **explicitly** to `wandb.init()`, which
overrides the `WANDB_MODE=offline` env var the script exports. On an internet-less compute
node that blocks until it times out. `--wandb.mode=offline` is now passed on the CLI. The
env var was never sufficient; it only looked sufficient.

All five fixed, with the source line numbers in the comment block, and `bash -n` clean.

### Bug in my own new `build_cluster_env.sh`, caught in 8 seconds by the watcher

First launch died immediately: `ERROR: File or directory already exists:
.../cog/miniforge3`. The directory-tree step in the 15:30 entry had pre-created an *empty*
`$WORK/cog/miniforge3`, and my guard tested for `${MF}/bin/conda` (absent) so it proceeded to
install into a prefix the installer refuses to touch. Fixed with `-b -f -p` (`-f` = do not
error if the prefix exists) rather than by deleting anything. Worth noting that the event
watcher earned its keep at the 60-second mark on a job I would otherwise have assumed was
still downloading: **a fast failure looks exactly like a slow start unless something checks.**

### Install order in the cluster env is load-bearing

torch is installed from the cu128 index **first, with its dependencies**, and lerobot
second. The reverse order (or `--force-reinstall --no-deps` for torch afterwards, which is
what I first wrote) leaves whatever `nvidia-*-cu12` minor versions the default-PyPI torch
had already installed, which is an import-time `.so` failure on the compute node. lerobot
does not move torch: 2.7.0+cu128 satisfies its `torch<2.11.0,>=2.2.1`, and `torchcodec`
0.10.0 -- the one dep that could have dragged torch around -- declares **no** torch
requirement at all (checked in its METADATA before relying on it).

### 2026-08-19 17:50 — Cluster eval (G5b): the container is mandatory and cannot be built on Leonardo

The user approved attempting the optional cluster-eval path after step 4. Two measurements
settle *how* it has to be done, and both were cheap:

**1. Leonardo is RHEL 8.8 with glibc 2.28.** Isaac Sim 5.1 needs glibc >= 2.35 (Ubuntu
22.04). So the cheap route I had hoped for -- repeat the local recipe, `pip install
isaacsim` into a second conda env on `$WORK`, no container at all -- is **impossible**, not
merely awkward. The plan's original instinct (Singularity) was right. Recording this because
"just pip install it like we did locally" is the obvious idea and it would have burned an
hour to find out.

**2. `singularity build --fakeroot` is not available to this account:**

```
FATAL: could not use fakeroot: no valid mapping entry found for ohausdoe (133040)
```

`allow setuid = yes` and the fakeroot CNI config exist in `/etc/singularity/singularity.conf`,
but there is no subuid/subgid mapping for our user, which is a site decision we cannot change.
Consequence: **no def-file build (`%post`) can run on the cluster at all.** Any image has to
arrive already built.

**What still works, and needs neither root nor credentials on the cluster:** unprivileged
`singularity build` from a *docker archive* is a pure format conversion, no `%post`, so it
needs no fakeroot. And the login node reaches both `nvcr.io` and `registry-1.docker.io`
(HTTP 401 on `/v2/` = registry alive, unauthenticated probe -- not a block). Meanwhile this
workstation has Docker 28.3.0, the user is in the `docker` group, and `~/.docker/config.json`
**already holds nvcr.io credentials**.

So the chosen route is: build the image **locally** with Docker, `docker save` it, rsync the
archive up, and `singularity build isaaclab.sif docker-archive://...` on the login node. This
keeps the NGC credential on the workstation -- it never travels to a shared cluster, which
the alternative (`SINGULARITY_DOCKER_PASSWORD` on the login node) would have required. Cost
is ~20 GB of local disk and one ~20 GB upload, 0 GPU-h, and no dependence on a site policy we
cannot influence.

**Sequencing decision:** this comes *after* the 80k calibration run is submitted. Training
needs no Isaac whatsoever, the local-4090 eval fallback is already accepted, and G5b is a
gate that may still fail on its own merits (Isaac Sim is officially unsupported on A100 --
no RT cores). Building a 20 GB image is not allowed to delay the critical path.

### 2026-08-19 18:05 — G5a batch/LR sweep: the plan's batch-scaling premise is refuted; the loop is decode-bound

Job 52878355, `boost_qos_dbg`, 200 steps per arm on L0/N=25, sqrt-LR scaling from 1e-4 @ 64:

| batch | lr | wall_s | samples/s | steps/s | peak VRAM (MiB) | median GPU util | loss @200 |
|---|---|---|---|---|---|---|---|
| 64 | 1e-4 | 208 | 61.5 | **0.962** | 13546 | **0 %** | 0.188 |
| 128 | 1.41e-4 | 232 | 110.3 | 0.862 | 14482 | **0 %** | 0.164 |
| 256 | 2e-4 | 519 | 98.7 | 0.385 | 17060 | **0 %** | 0.149 |

Steady-state per-step split (last logged interval, so worker spin-up is excluded):

| batch | updt_s | data_s | data share |
|---|---|---|---|
| 64 | 0.071 | 0.458 | 87 % |
| 128 | 0.098 | 0.904 | 90 % |
| 256 | 0.450 | 2.438 | 84 % |

**The plan (P5) assumed we would be GPU-bound** and said to "scale batch up as far as A100-64GB
VRAM/throughput allow (e.g., 64->128->256)". That premise is **false for this workload**:

- Peak VRAM at batch 256 is 17 GB of 65 GB. **VRAM is not remotely a constraint** and never
  becomes the deciding variable, so the whole "how large a batch fits" question is moot.
- Median GPU utilization is **0 % at every batch size**. The A100 is idle ~87 % of the time.
- Samples/s is essentially flat (61 -> 110 -> 99 measured; 121 -> 128 -> ~89 on the
  steady-state split). Throughput is set by the dataloader, not the GPU, so a bigger batch
  buys no throughput -- it just makes each of the fixed 80k steps cost proportionally more.

**Decision: batch 64, lr 1e-4.** With the protocol fixed at 80k *steps* (user directive), the
smallest sensible batch minimises wall-clock: 0.962 steps/s -> ~11.8 h/run, against 22 h at
128 and 58 h at 256 (which would not even fit the 24 h walltime). lr 1e-4 also happens to be
`DiffusionConfig.optimizer_lr`'s default, so no sqrt-scaling is applied and the LR-override
bug found earlier this session is doubly defused. `configs/train/diffusion_base.sh` can now
have its two placeholder values frozen at exactly the values already written there.

Note the loss ordering (0.188 > 0.164 > 0.149 at equal *steps*) is expected and is **not** an
argument for a larger batch: at fixed steps a larger batch has seen 2x/4x more samples, so it
should be further along. Comparing at equal steps across batch sizes compares different
sample budgets, which is why batch is frozen for every cell rather than tuned per cell.

### Why it is decode-bound -- root cause found in the 0.4.4 source, not guessed

`data_s` of 0.458 s for a batch of 64 means ~1.8 ms per 128x128 frame (64 samples x 2 cameras
x 2 obs steps = 256 frame fetches). That is far too slow for tiny frames, so I looked instead
of speculating:

1. **The videos are already nearly all-intra.** `ffprobe` on `L0/videos/.../file-000.mp4`:
   h264, 128x128, 20 fps, **82,916 frames** in one 43 MB file, with **41,058 keyframes** --
   the keyframe pattern is literally `1,0,1,0,1,0,...`, i.e. **GOP 2**. So my first idea
   (re-encode all-intra with `-g 1` to make seeking cheap) would buy nearly nothing. Killed
   before implementing it.
2. **The pyav path re-opens the container on every single call.**
   `datasets/video_utils.py:decode_video_frames_torchvision` constructs a
   `torchvision.io.VideoReader(video_path, "video")` per call, seeks, decodes, then
   `reader.container.close()`. Each sample-camera pair therefore pays a full container open
   and index parse **on an 82,916-frame file** -- ~128 opens per batch of 64.
3. **The torchcodec path, and only the torchcodec path, has a `VideoDecoderCache`**
   (`decode_video_frames_torchcodec(..., decoder_cache: VideoDecoderCache | None)`). The
   decoder is reused across calls, which is exactly the cost the pyav path keeps re-paying.

So the bottleneck is not the codec and not the GPU: it is repeated container opening in the
one backend we can currently load. **torchcodec fails to import in both our envs**
(`OSError: libavutil.so.56`) because neither the workstation nor RHEL 8.8 ships ffmpeg shared
libraries -- which is why PINS chose pyav in the first place.

**And that is a fixable dependency problem, not a design constraint.** torchcodec 0.10.0 ships
`libtorchcodec_core{4,5,6,7,8}.so`, i.e. it supports ffmpeg majors 4-8; it only needs *an*
ffmpeg. conda-forge ffmpeg 6.1.2 installs cleanly into a cluster env (verified in the py3.12
experiment env: `libavutil.so.58` now present). So the likely fix is a one-package install,
**with no LeRobot migration at all** -- tested next.

### On the LeRobot version question (user, 2026-08-19)

The user asked why we are on 0.4.4 and noted 0.5+ improved the dataloader. The pin was never a
preference: 0.5.0, 0.6.0 and 0.6.1 all declare `requires_python >=3.12` (checked against PyPI
today), while Isaac Sim 5.1 requires py3.11, and eval loads the policy in the *same process*
as Isaac. One version had to serve both, and 0.4.4 is the newest that runs on 3.11 (G1b).

The user's instruction was to measure before committing ("maybe it doesnt improve"), which is
right, and the extras layout of 0.6.1 already hints at the answer: **`torchcodec` is required
by the `dataset` extra while `av` (pyav) is a separate opt-in `av-dep` extra.** 0.6 is built
around torchcodec. If our decode problem is "torchcodec cannot load", then 0.6 does not solve
it -- ffmpeg does -- and 0.6 without ffmpeg would be *worse* off, not better. Being measured
head-to-head anyway (env `cog_lerobot06`, py3.12, lerobot 0.6.1 + ffmpeg 6.1.2).

Also worth recording: a bare `pip install lerobot==0.6.1` is **not usable for training** -- no
video backend, and `lerobot.scripts.lerobot_train` raises ImportError. The correct install is
`lerobot[dataset,training]`. A migration would therefore not be a version-number change.

### 2026-08-19 18:25 — My own num_workers measurement was invalid; draccus takes the LAST flag

Job 52885440 reported num_workers 8 and 16 as **identical** (data_s 0.388 vs 0.386,
2.18 vs 2.19 steps/s), which I nearly wrote up as "worker count does not help, the bottleneck
must be a serialized resource". It was not a finding, it was a bug in my own smoke script:

`configs/train/diffusion_base.sh` contains `--num_workers=8`, and `${SMOKE_FLAGS}` (derived
from it) is expanded **after** my `--num_workers=${NW}` on the command line. I had stripped
`--steps`, `--save_freq` and `--log_freq` from it precisely because I did not want to depend
on flag precedence -- and then left `--num_workers` in. Draccus resolves duplicates
last-wins, so every arm ran at 8 workers.

Caught by checking the **resolved** config rather than the flag I passed:
`grep "num_workers" <log>` -> `num_workers': 8` inside the arm labelled nw=16. The job was
cancelled rather than left to finish a third identical arm.

Two things now guard against a repeat, because a wrong number that looks plausible is worse
than a crash:
1. `--num_workers=` is stripped from `SMOKE_FLAGS` alongside the others.
2. The smoke **asserts** the resolved `num_workers` equals the requested one and writes
   `NWMISMATCH<n>` into the CSV instead of a timing row if it does not. A measurement that
   cannot prove what it measured does not get to look like data.

Not a total loss: the two accidental repeats of the same configuration show run-to-run
variance is **under 1 %** (2.18 vs 2.19 steps/s), which is worth knowing before treating any
single-arm difference as real. Also confirmed the job really did get 32 CPUs (`nproc` = 32),
so the boring explanation was ruled out before the interesting one was believed.

**Note for the frozen config:** `--num_workers=8` living inside `COG_DP_FLAGS` means it is
part of the frozen hyperparameter set. It is *not* a scientific hyperparameter -- it changes
throughput, not the model -- so if the dataloader work leads to a different worker count, it
must be changed deliberately in one place and recorded, not overridden per job.

### Cost context, so this optimisation does not become a rabbit hole

Do-nothing baseline: 0.962 steps/s at batch 64 with 8 cores -> **~11.8 h per run** (10.2 h by
the steady-state estimate). 24 cells x 10.2 h x 8 cores = ~1,960 core-h = **~245 GPU-h**, and
because the cells are independent and the queue is effectively empty (4 s to start), the whole
wave finishes in **one 10-12 h wall-clock block**. That is already inside the plan's ~200
GPU-h / "~1 day" envelope for T1 and inside the 2,200 GPU-h ceiling even after Tasks 2-3.

So the decode bottleneck is an **optimisation, not a blocker**. It is worth a bounded attempt
because a 3x win compounds across 3 tasks (~735 -> ~250 GPU-h), but it does not gate the
calibration run and will not be allowed to delay it.

### 2026-08-19 18:45 — LeRobot version question answered by measurement; the real bug is one pin

The user asked why we are on 0.4.4 and suggested 0.5 improved the dataloader, then (correctly)
said to smoke-test before committing. Investigated properly, on our own dataset. Findings:

**0.5.0 would have changed nothing.** `lerobot/datasets/video_utils.py` is **byte-identical**
between 0.4.4 and 0.5.0, the `datasets/` file list is unchanged, and there is no
`persistent_workers`/`prefetch_factor` in `configs/train.py`. The v0.5 blog's "10x faster
image training / 3x faster encoding" is entirely **write/record-side** (streaming encode,
hardware encoders), not the training read path. Upgrading to 0.5 would have cost the py3.11
single-env property (G1b) and bought exactly zero.

**0.6.0 does help, and less than the blog implies.** Two independent changes:
- PR **#3588** rewrote the pyav path to use `av.open`/`seek`/`decode` natively instead of
  wrapping `torchvision.io.VideoReader` (motivated by `VideoReader` being removed in
  torchvision 0.26 -- our own run already warns about this). Frame selection (`torch.cdist`
  + `tolerance_s`) is unchanged, so no new misalignment risk.
- PR **#3406** ("2x faster dataloader") adds a `ThreadPoolExecutor` over camera keys,
  `return_uint8`, `persistent_workers=True`, `prefetch_factor=4`, spawn context.

Measured on `data/lerobot/L0` (2 cams, 128x128, batch 64, num_workers 8, `data_s` averaged
over 100 steps after 10 warmup), **pyav on both sides**:

| lerobot | settings | data_s |
|---|---|---|
| 0.4.4 | as we run it | **0.3485 s** |
| 0.6.1 | 0.4.4-like (prefetch 2, no persistent) | 0.0967 s |
| 0.6.1 | full 0.6 defaults | **0.0947 s** |
| 0.6.1 | full 0.6 defaults, parallel decode off | 0.1052 s |

**~3.7x, and essentially all of it is #3588, not the headline #3406.** uint8/persistent/
prefetch contribute ~2 % at 128x128; parallel decode ~13 %. Per-call decode: 0.4.4 = 33.9 ms,
0.6.1 = 12.1 ms.

**But the actual bug is in a version pin, and it is one line.** lerobot 0.4.4 requires
`torchcodec>=0.2.1,<0.11.0`, which resolves to **torchcodec 0.10.0 -- built against torch
2.10**, while we pin torch 2.7.0. Hence the load failure, which is *not* an ffmpeg problem:

```
libtorchcodec_core6.so: undefined symbol: _ZN3c1013MessageLogger6streamB5cxx11Ev
```

`_ZN3c10...` is a **libtorch C++ ABI symbol**. torchcodec dlopens its variants in DESCENDING
ffmpeg order (8,7,6,5,4), so the `libavutil.so.56` line everyone sees is merely the LAST
attempt (ffmpeg 4, which nobody has) -- I had been reading the tail of the traceback and
chasing the wrong layer. Also useful: `libavutil major = ffmpeg major + 52`, so .56=FF4,
.58=FF6, .61=FF9 -- which is why a bare `conda install ffmpeg` (now FF 9.0.1) would be
*useless* and the `=6.*` pin mattered.

**torchcodec 0.7.0 loads against torch 2.7.0+cu128 on Leonardo** (verified directly:
`WORKS 0.7.0`) once conda-forge ffmpeg 6.1.2 is present AND
`LD_LIBRARY_PATH=$CONDA_PREFIX/lib` is exported -- `libtorchcodec_core*.so` carries no
RPATH/RUNPATH, so conda's libs are invisible without it. Per-call decode with a working
torchcodec is **~1.1 ms vs pyav's 33.9 ms (~31x)**, because it keeps a module-level
`_default_decoder_cache` (`video_utils.py:330`: `if decoder_cache is None: decoder_cache =
_default_decoder_cache`) instead of re-opening the container.

**Decision: stay on 0.4.4 and fix the torchcodec pin.** It is the larger win (~31x vs 3.7x),
on the version we already have, with no py3.12 migration, no loss of the single-env property,
and no checkpoint surgery. Recorded as D23.

**Two findings that make me glad we measured instead of upgrading:**

1. **lerobot 0.6.0 silently flipped five diffusion defaults** (PR #3202): `horizon` 16->64,
   `n_action_steps` 8->32, `use_group_norm` True->False, `pretrained_backbone_weights`
   None->`ResNet18_Weights.IMAGENET1K_V1`, `use_separate_rgb_encoder_per_camera` False->True.
   Our frozen config pinned the first three and **not** the last two. A "just bump the
   version for a faster dataloader" change would have silently altered the model architecture
   in the middle of the study. Both missing flags are now pinned at 0.4.4's own values in
   `configs/train/diffusion_base.sh`, so they are inert today and load-bearing later.
2. **A 0.6.1 checkpoint does not load in 0.4.4** without deleting two added config keys
   (`pretrained_revision`, `gradient_checkpointing`); with them removed it loads with an
   identical parameter checksum. The reverse direction (0.4.4 ckpt -> 0.6.1) works untouched.
   Since eval must stay on py3.11 for Isaac, a train-on-0.6 plan would have needed a
   config.json scrub step in the eval path. Good to know, not needed now.

Also recorded for later: 0.6.x keeps pyav fully supported and, from 0.6.1 (PR #4307), replaces
`get_safe_default_codec()` with `get_safe_default_video_backend()`, which *try-imports*
torchcodec and falls back to pyav with a warning instead of crashing in the workers -- exactly
our failure mode. And `pip install lerobot==0.6.1` alone is not trainable: no video backend,
`diffusers` is no longer core, so it needs `lerobot[training,dataset]`.

### 2026-08-19 18:30 — G5a decode fix lands: data_s 0.388 -> 0.003 s (129x), runs go 10.2 h -> 1.6 h

`torchcodec==0.5` + conda-forge ffmpeg 6 + `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`, measured in a
real training loop (job 52896093, L0/N=25, batch 64, num_workers 8, 150 steps):

| backend | updt_s | data_s | steady steps/s | est. 80k | peak VRAM |
|---|---|---|---|---|---|
| pyav | 0.071 | 0.388 | 2.18 | 10.2 h | 13.5 GiB |
| pyav (nw=16*) | 0.070 | 0.386 | 2.19 | 10.1 h | 13.7 GiB |
| **torchcodec** | 0.069 | **0.003** | **13.89** | **1.6 h** | 13.5 GiB |

\* that arm actually ran at 8 workers -- see the 18:25 entry.

**`data_s` fell 129x and the GPU is now the bottleneck**, which is what the user's "GPU
utilization should be high" directive was asking for. `updt_s` is unchanged at 0.069 s, so the
step time is now essentially pure compute: 0.072 s/step -> 13.9 steps/s.

Warm-up is visible and worth knowing: step 25 still reports `updt_s 1.152, data_s 0.480` while
the decoder cache is cold and CUDA kernels autotune; by step 50 it is at 0.069/0.003 and stays
there. Any future throughput measurement must discard the first interval or it will understate
the fix by an order of magnitude -- which is exactly why the smoke parses the LAST logged
interval rather than averaging the run.

**Independent corroboration of frame-identity:** `final_loss` at step 150 is **0.266 in all
three arms** -- pyav, "pyav nw16", and torchcodec. Same seed, same sampler, same loss to three
decimals means the two backends fed the model the same pixels. That is a second, orthogonal
check on top of the direct comparison (40 fetches, max abs pixel diff 0.0, `BACKENDS_IDENTICAL`).

`median_util_pct` still reads 0 in the CSV for the torchcodec arm, and that is a measurement
artifact rather than a result: 150 steps now take ~11 s inside a ~50 s job, so the 2-second
nvidia-smi samples are dominated by dataset creation. The metric was informative while runs
were slow and is useless now that they are fast; `data_s` is the reliable signal.

**Cost and schedule impact** (billing = allocated cores, verified: `billing=8` for 8 cores):

| | per run | 24-cell matrix | 3 tasks (72 cells) |
|---|---|---|---|
| pyav | ~10.2 h, ~10.2 GPU-h | ~245 GPU-h | ~735 GPU-h |
| torchcodec | **~1.6 h, ~1.6 GPU-h** | **~38 GPU-h** | **~115 GPU-h** |

Against the plan's ~200 GPU-h estimate for T1 training and the 2,200 GPU-h approved ceiling,
this turns the budget from "comfortable" into "irrelevant", and the whole T1 wave becomes a
**~2 h wall-clock block** rather than a day. The 80k calibration run now costs ~1.6 GPU-h
instead of the ~8 quoted to the user.

Two process notes worth keeping:
- **"It imports" is not evidence that a native extension matches your torch.** torchcodec 0.7.0
  imported fine and then failed on first decode with `no fallback function is registered for
  schema torchcodec_ns::_convert_to_tensor`. I had already reported 0.7.0 as working on the
  strength of a successful import. Only a decode proves a decoder.
- The frame-equality check was written *before* switching and immediately caught the broken
  0.7.0. Had I trusted the timing numbers alone, a subtly wrong backend could have reached the
  matrix.

### 2026-08-19 18:38 — G5a resume-across-requeue PASSED

Job 52899246, `boost_qos_dbg`, ~4 min:

```
[resume] checkpoint at step 200 -> SIGKILLing phase 1 (pid 2175431)
[resume] step recorded in checkpoints/last = 200
=== phase 2: resume via config_path (the path train.sbatch now takes) ===
G5A_RESUME_PASSED  (killed at 200, resumed at 225, finished at 400)
```

The assertion that matters is `resumed at 225`, not `finished at 400`: a resume that silently
restarted from step 0 would also have finished at 400 and exited 0, quietly discarding the
first 200 steps. The test requires the resumed run's **first logged step** to be beyond the
checkpoint step, so "it ran to completion" cannot be mistaken for "it resumed".

This validates the bug-3 fix from earlier today. Before it, `train.sbatch` passed a bare
`--resume=true`, which `configs/train.py:89-95` rejects with "A config_path is expected when
resuming a run" -- so a node failure or walltime kill at hour 11 of an 80k run would have
produced a job that could never be restarted, and we would have found out only then.

**Two bugs in my own test, both worth recording because both were silent:**
1. I checked for `checkpoints/000000200`, following the 9-digit `000020000` shown in
   `docs/specs/06_lerobot_044.md`. LeRobot actually writes **6 digits** (`000200`). The trigger
   therefore never fired, phase 1 ran to its 400-step target unkilled, and the test would have
   reported a resume failure caused entirely by the test. Now it reads the step out of
   `training_state/training_step.json`, which is format-independent, and it distinguishes
   `G5A_RESUME_INCONCLUSIVE` (never got killed) from a real failure.
2. `timeout -s KILL 420 python ...` put `timeout` in `$!`, and SIGKILL to `timeout` does not
   propagate to its child, so phase 1's python would have kept training in the background while
   phase 2 started on the same output directory -- two writers, one checkpoint dir. `timeout`
   was dropped; python is launched directly.

**G5a status: three of four parts done.** batch/LR frozen (batch 64, lr 1e-4); decode fixed
(torchcodec, 13.89 steps/s); resume verified. The 80k calibration run (`t1_L0_n25_s0`, job
52899856) is in flight -- and it is not a throwaway: L0/N=25 is a real cell of the 24-cell
matrix, so the calibration produces a study result rather than only a timing number.

### 2026-08-19 18:40 — wandb swallows the Slurm log; monitor 80k runs by checkpoint, not by stdout

The calibration run's `.out` file stopped growing at 118 bytes while the job kept running, which
looked exactly like a hang. It is not. `wandb.init()` installs a console redirect --
`wandb_run.py:_redirect():2446 Wrapping output streams` / `2469 Redirects installed` -- so every
subsequent LeRobot log line goes into wandb's own buffers instead of the job's stdout, and in
**offline** mode nothing flushes them to a readable file until the run ends (the offline run dir
contains only `logs/debug*.log`, `run-<id>.wandb` and `files/requirements.txt` while running).

Liveness had to be established another way, and the right tool was `sstat`:

```
52899856.0   AveCPU 00:12:52   MaxRSS 3030448K      <- the srun step is very much alive
```

Note the `.0` step only appears if you do not truncate `sstat` output -- my first look used
`head -4` and showed just `.extern` and `.batch`, which momentarily looked like "the srun step
never started".

**Two consequences, both recorded so no one re-derives them:**

1. **Progress monitoring for the real runs must be artifact-based** -- watch for
   `checkpoints/{020000,040000,060000,080000}` appearing (six-digit names, per the resume-test
   finding). This is the same "assert on the artifact, not the exit code" rule as CLAUDE.md 10,
   arriving from a new direction: here you cannot even *see* the log. The end-of-job
   `[train] DONE <run_id>` line does still reach the `.out`, because the sbatch script echoes it
   after `srun` returns and wandb's redirect has been torn down.
2. **`WANDB_DIR` is ineffective in LeRobot.** `train.sbatch` exports
   `WANDB_DIR=$WORK/cog/wandb_offline`, but LeRobot passes `dir=cfg.output_dir` to `wandb.init`,
   so the offline run actually lands in
   `$WORK/cog/checkpoints/<run_id>/wandb/offline-run-<ts>-<id>/`. That directory stayed empty
   while `$WORK/cog/checkpoints/t1_L0_n25_s0/wandb/` filled up. **The later `wandb sync` must
   therefore point at the per-run checkpoint directories, not at `$WORK/cog/wandb_offline`** --
   syncing the latter would silently upload nothing and report success. Related to the standing
   note that the local `.netrc` key may not match a run's wandb entity: confirm the account
   before trusting any sync.

### 2026-08-19 19:05 — Reviewed the eval path ahead of G5b: two cluster-portability bugs

Read `slurm/eval.sbatch` and `src/cog/eval/rollout_eval.py` while waiting on the calibration,
looking for the same class of never-run-on-the-cluster bug that `train.sbatch` turned out to
have five of. Two found, both fatal on the cluster and both invisible locally:

1. **`eval.sbatch` activates a conda env that cannot exist on Leonardo.** It does
   `conda activate "${COG_EVAL_ENV:-cog_isaac}"`. That is exactly right locally and impossible
   on the cluster: RHEL 8.8's glibc 2.28 is below Isaac Sim 5.1's >= 2.35 requirement, which is
   *why* we are building a Singularity image at all. Cluster eval has to run **inside the sif**:
   `singularity exec --nv -B ... "${SIF}" python -m cog.eval.rollout_eval ...`. The script was
   written before the glibc constraint was known, and its own header says "UNVERIFIED until G0
   clears and G5b passes" -- which it has now earned twice over.
2. **`rollout_eval.py --protocol` defaults to a hardcoded workstation path**
   (`/home/admin_07/cost_of_generality/configs/eval_sets/protocol.json`). On the cluster the
   file lives at `$WORK/cog/repo/configs/eval_sets/protocol.json`. Same failure family as the
   `train.sbatch` line that sources `diffusion_base.sh` from an absolute local path -- there the
   `||` fallback saved it; here there is no fallback. The default should resolve relative to the
   package, with the absolute path as an override.

Both are deferred until the G5b render gate answers, on purpose: if Isaac cannot render on an
A100 the entire cluster-eval path is moot and the accepted fallback (eval locally on the 4090)
uses the conda env and the local protocol path exactly as written. Fixing them first would be
work done on the strength of an unverified assumption -- the mistake this project keeps
catching itself in.

Noted for when it is time: `--time=04:00:00` in `eval.sbatch` covers **three** checkpoints
(40k/60k/80k) x 100 episodes x up to 600 steps of rendered rollout. That is ~180k rendered env
steps in one job and the 4 h budget is a guess, not a measurement. Time one checkpoint before
launching 24 cells' worth.

### 2026-08-19 19:07 — Calibration mid-run: 11.2 steps/s measured, and the run's own config verifies every frozen value

**The user pushed back that the policy "trains so fast", which was the right instinct to check.**
Answered from the 20k checkpoint's own recorded config rather than from my extrapolation.

**Real rate, end-to-end.** Checkpoint `020000` written at 19:06:46; the training loop started
~18:37:07 (wandb init 18:36:58 + the ~9 s gap between init and "Start offline training" observed
in the smoke). 20,000 steps in **1,779 s = 11.2 steps/s**, so 80k steps is **~1.98 h** plus ~2.3
min of startup -> **~2.0 h and ~2.0 GPU-h per cell** (billing 8 = 1 GPU-h per wall-clock hour).

That is **20 % slower than the 13.89 steps/s** the throughput smoke reported, and the reason is
CPU allocation, not anything mysterious: `smoke_dataloader.sbatch` requests
`--cpus-per-task=32` while `train.sbatch` requests 8. With 8 cores, 8 dataloader workers plus the
main process contend, so `data_s` no longer rounds to zero. Measured CPU draw during the run was
~5.9 of 8 cores busy.

**8 cores is nevertheless the right choice, by arithmetic rather than preference:** billing scales
linearly with cores, so 16 cores would buy ~18 % wall-clock for ~2x the cost.

| cpus-per-task | steps/s | h/cell | GPU-h/cell | 23 cells |
|---|---|---|---|---|
| **8** | **11.2 (measured)** | **~2.0** | **~2.0** | **~46** |
| 16 | ~13 (extrapolated) | ~1.7 | ~3.4 | ~79 |

So the earlier "1.6 h / 38 GPU-h" figures were optimistic by the CPU-allocation difference;
**~2.0 h / ~46 GPU-h is the number to plan with.** Still 5x better than the pyav baseline
(10.2 h) and well inside the plan's ~200 GPU-h for T1 training.

**Config verification from `checkpoints/020000` -- every frozen value confirmed:**

```
steps = 80000              batch_size = 64            seed = 0
save_freq = 20000          num_workers = 8            video_backend = torchcodec
optimizer = adam / lr=0.0001                          optimizer_lr = 0.0001
n_obs_steps = 2            horizon = 16               n_action_steps = 8
vision_backbone = resnet18 crop_shape = [112, 112]    use_group_norm = True
num_train_timesteps = 100  noise_scheduler_type = DDPM
pretrained_backbone_weights = None
use_separate_rgb_encoder_per_camera = False           do_mask_loss_for_padding = False
dataset.episodes = 25 (0..24)
```

Three things this settles:
1. **Nothing was silently reduced.** `steps=80000` at `batch_size=64` with the full architecture.
   The run is simply a ~2 h job on an A100 at 128x128 -- cheap compared with the original
   Diffusion Policy setup's 240x320 inputs.
2. **The LR fix works end-to-end.** `optimizer = adam / lr=0.0001` in `train_config.json` proves
   `--policy.optimizer_lr` reached the *actual optimizer*. Bug 4 (validate() overwriting
   `cfg.optimizer` from the policy preset) would have left this at the default silently -- here it
   is confirmed against the artifact rather than the flag.
3. **D23 VERIFY (c) discharged:** `pretrained_backbone_weights = None`. That flag is deliberately
   not passed on the CLI (draccus might decode the string "null"), so asserting it against the
   saved config was the plan, and it holds.

**Also fixed: my sif build failed and my own diagnostic hid it.** The login-node
`singularity build` was piped through `tail -15` and then read `$?`, which reports **tail's**
status -- so a failed build printed `BUILD_RC=0` while producing no image. That is the same
exit-code-hygiene trap already recorded in this journal for `head`, arriving through a different
pipe. The conversion is now a compute job (`slurm/build_sif.sbatch`, job 52915767) with no pipe
before the exit-code capture, 120 GB of RAM, and node-local scratch instead of Lustre -- the
login node is shared by ~100 users with capped memory, and a Lustre `SINGULARITY_TMPDIR` made
mksquashfs emit an endless "Unrecognised xattr prefix lustre.lov" stream while unpacking ~17 GB
of small files.

### 2026-08-19 19:23 — Isaac Sim container built on Leonardo: SIF_BUILD_OK (7.1 GB)

The glibc barrier is cleared. `isaac-sim-5.1.0.sif` is 7.1 GB and runs:

```
=== does it run? ===
inside
PRETTY_NAME="Ubuntu 24.04.2 LTS"
SIF_BUILD_OK isaac-sim-5.1.0.sif
```

Ubuntu 24.04 inside means glibc 2.39, comfortably above Isaac Sim 5.1's >= 2.35 -- which is the
whole reason a container was mandatory on RHEL 8.8 (glibc 2.28).

Route that worked, in order, none of it optional:
1. `docker pull nvcr.io/nvidia/isaac-sim:5.1.0` **locally** (15.1 GB) -- the workstation already
   had nvcr.io credentials, so nothing had to be sent to a shared cluster.
2. `docker save` -> 15,123,856,384 B tar; `rsync -z` up (7.54 GB on the wire, 2.01x compression);
   byte-count verified **identical** on both sides.
3. `singularity build isaac-sim-5.1.0.sif docker-archive://...` on a **compute node**. Converting a
   docker archive needs no `%post` and therefore **no fakeroot** -- which matters because fakeroot
   is unmapped for this account (`no valid mapping entry found for ohausdoe`), so a def-file build
   was never an option.

**Timings, for docs/timings.md:** unpack ~19.4 GB of rootfs in ~10 min, squashfs to 7.1 GB in
~6 min, total job 13:55 at 16 cores = **0.46 GPU-h**. The same conversion on a login node had
**failed**: shared by ~100 users, capped memory, and a Lustre `SINGULARITY_TMPDIR` that made
mksquashfs emit an unbounded "Unrecognised xattr prefix lustre.lov" stream. The compute node's own
`/tmp` was only 10 GB against ~35 GB needed, so the job correctly fell back to Lustre for scratch
and still finished in 14 minutes -- the win was RAM and dedicated cores, not local disk.

Next: `slurm/debug_a100_kit.sbatch` (job 52921497) runs NVIDIA's own shipped camera example
headless for 601 steps and judges the PNGs it writes -- existence *and* pixel content, because the
A100 failure mode on record is degraded rendering, which produces a perfectly valid blank file.

### 2026-08-19 19:25 — G5b stage A failed on a PERMISSION bug, not on A100 rendering

Job 52921497 reported `G5B_A_FAILED: the renderer produced no image at all`. **This is not an
Isaac-on-A100 verdict.** The renderer was never reached:

```
=== NVIDIA's own compatibility check ===
/usr/bin/bash: line 1: /isaac-sim/isaac-sim.compatibility_check.sh: Permission denied
=== headless camera render, 601 steps ===
render rc=126        # 126 = "command found but NOT EXECUTABLE"
[gate] 0 PNG(s) found
```

Diagnosed inside the container in one command:

```
uid=133040(ohausdoe) gid=25200(interactive)
drwxr-x---. 18 root root 1066 Oct 17 2025 /isaac-sim
ls: cannot access '/isaac-sim/python.sh': Permission denied
```

**`/isaac-sim` is mode 750 owned by the image's `isaac-sim` user.** Docker runs containers as
**root** by default, so nobody ever notices; Singularity runs as the **invoking user**, so the
entire Isaac Sim tree is unreadable. Every path under it fails, and the symptom -- a headless
render that produces no image -- is indistinguishable at a glance from the A100 rendering failure
the gate was built to detect. **A gate that cannot separate "no permission" from "cannot render"
would have produced a completely wrong conclusion**, and the one that mattered: we would have
abandoned cluster-side eval, and with it ~2 days of the study's critical path, on a chmod.

The repair is trivial once measured. Inspecting the image as root shows the tree is otherwise
sane: **exactly one directory lacks o+x and only 28 entries lack o+r**; everything else is
755/644. So `chmod a+rx /isaac-sim` plus an `a+rX` pass over those 28 entries touches ~29 files
rather than copying up 15 GB of layer -- worth checking, because a naive `chmod -R a+rX /isaac-sim`
would have forced overlayfs to duplicate the entire image and doubled both the upload and the
conversion.

Now folded into `docker/Dockerfile.cog`, together with a `test -x /isaac-sim/python.sh` so the
build fails at build time if the fix ever stops working. Rebuilding the FULL image (Isaac Sim +
Isaac Lab v2.3.0 + LeRobot 0.4.4 + cog) rather than a chmod-only base, because the round trip
costs the same and the full image is what eval actually needs.

**Generalisable lesson, third time today:** the failure was in the layer *below* the one the error
pointed at. `libavutil.so.56` was really a torch ABI mismatch; `BUILD_RC=0` was really `tail`'s
exit code; "the renderer produced no image" was really a directory mode. In each case the honest
next step was to look one level down rather than act on the surface message.

### 2026-08-19 19:50 — Cluster-eval container: STOPPED at a deliberate bound, and D24 removed most of the reason to continue

**What works.** `isaac-sim-5.1.0.sif` (7.1 GB) builds and runs on Leonardo, and the mode-750
problem is fixed in `docker/Dockerfile.cog`. The `apt`, IsaacLab-install and permission-sweep
layers all build. So Isaac Sim *can* be containerised here.

**What blocks it.** Installing LeRobot 0.4.4 into Kit's python cannot be resolved, and after six
build attempts the cause is clear and is not a pinning mistake. NVIDIA ships several packages
inside `/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/` rather than site-packages, and pip
can neither *reuse* those to satisfy a requirement nor *replace* them safely:

- `packaging` 23.0 -- upgrading it (lerobot needs >=24) deletes prebundle files and **breaks
  Isaac's own vendored torch** (`No module named 'torch._vendor.packaging._structures'`; the
  pristine image imports torch 2.7.0+cu128 fine). `--ignore-installed` works around this: verified
  `packaging 25.0 | torch 2.7.0+cu128` in the same interpreter.
- `imageio` 2.37.0 -- the same trick installs a site-packages copy successfully, and the resolver
  **still** refuses: `Cannot install imageio[ffmpeg]==2.37.0 and lerobot==0.4.4`, with the real
  reason on a line below the one everybody reads: *"some packages in these conflicts have no
  matching distributions available for your environment: imageio"*. Not a version conflict -- pip
  has no imageio it is willing to use for the extras variant. Unchanged under pip 24.3.1 and 26.2.1.
- `numpy` -- unconstrained, pip upgrades 1.26.0 -> 2.4.6 (breaking isaaclab-rl, isaaclab-tasks,
  nvidia-srl-usd, numba: the D3 hazard); pinned to 1.26.4, it collides with `rerun-sdk 0.26.2`,
  which requires numpy>=2. The local env carries that same inconsistency and survives only because
  nothing ever forced a joint solve; `--no-deps` for rerun-sdk reproduces that deliberately.

Each of the six failures was a *different* real problem, which is why I kept going; but they are
all instances of one thing -- **a vendor image whose python environment is not pip-manageable** --
and that is not going to be fixed by a seventh pin.

**Why stopping is now the better call, not just the bounded one.** D24 landed mid-investigation:
full-scale cells evaluate only the last checkpoint, so the eval workload fell from **72
checkpoint-evals to 24+2**. Cluster eval was worth a 20 GB container when it was saving ~2-3 days
of 4090 wall-clock; against 24 evals the local fallback -- already the plan's accepted default if
G5b fails -- costs order hours. **The container's payoff shrank by 3x while its cost stayed the
same.** Continuing would be optimising the branch that stopped mattering.

**Two options remain on the record, neither urgent:**
1. **ZMQ policy-server split** -- the plan's documented G1b fallback (~150 lines). It fits this
   constraint unusually well: the cluster *already* has a working LeRobot env (`cog_lerobot`,
   py3.11, torch cu128, lerobot 0.4.4), so the container would need Isaac Sim + IsaacLab + `cog`
   and **no lerobot at all**, which deletes the entire conflict above. Cost: refactoring
   `rollout_eval.py` into client/server halves.
2. **Eval locally on the 4090** -- zero new code, accepted in the plan, and now ~3x cheaper than
   when it was chosen as the fallback.

**Recommendation: option 2 for T1, and revisit only if Tasks 2-3 make eval throughput binding
again.** The A100 rendering question (issues #3421/#1519) therefore stays formally **unanswered** --
worth stating plainly, because the gate never rendered a frame, and I am not going to record a
guess as a result.

Kept for reuse either way: the sif on `$WORK/cog/containers/`, `slurm/build_sif.sbatch`,
`slurm/debug_a100_kit.sbatch`, and `docker/Dockerfile.cog` with all six fixes and their reasons.

### 2026-08-19 19:55 — Cluster-eval UNBLOCKED: ship the working local env instead of rebuilding it (user's suggestion)

The user asked "can you not build the working image/env locally and then transfer it to the
cluster?" -- and that reframing is what broke the deadlock. My six failed attempts all built
**FROM `nvcr.io/nvidia/isaac-sim:5.1.0`**, i.e. they tried to *reconstruct* the stack inside a
vendor image whose python environment is not pip-manageable. The workstation already **has** the
stack working, in conda env `cog_isaac`. So move it rather than rebuild it.

`docker/Dockerfile.cog_env`: `FROM ubuntu:24.04`, plus Kit's X/GL/Vulkan runtime libs, plus the
env copied to its **identical absolute path**. Built first time, every assertion passing:

```
python 3.11.15
lerobot 0.4.4 | numpy 1.26.4 | torch 2.7.0+cu128
isaaclab importable
cog importable
non-world-readable entries before fix: 4 -> after fix: 0
```

23.8 GB image. **Whole stack importable in one interpreter**, which is precisely what the
NVIDIA-image route could not deliver.

**Why it works, and why the alternatives were unnecessary:**
- **glibc is the only reason a container is needed at all.** RHEL 8.8 has 2.28; Isaac Sim 5.1 needs
  >= 2.35. This workstation is Ubuntu 24.04.2 / **glibc 2.39** -- the same toolchain the binaries
  were compiled against -- and `ubuntu:24.04` supplies 2.39 in the container. Nothing is being asked
  to run on a platform it was not built for.
- **Identical absolute paths remove all relocation.** A conda env bakes its prefix into every
  console-script shebang (`#!/home/admin_07/miniconda3/envs/cog_isaac/bin/python3.11`), and
  `isaaclab`, `isaaclab_tasks`, `isaaclab_mimic`, `isaaclab_assets`, `isaaclab_rl` and `cog` are all
  **editable** installs whose `.pth` files point into `/home/admin_07/cost_of_generality/...`.
  Copying to the same paths means nothing needs rewriting.
- **conda-pack is unnecessary here** (the user raised it): it solves *prefix relocation*, and there
  is no prefix to relocate. More importantly it would not have helped with the actual blocker --
  no amount of path rewriting changes which glibc a `.so` requires. It is still the right tool if
  the env ever has to live at a different path.
- **The USD assets travel with the env.** `isaacsim-asset`, `isaacsim-extscache-kit`,
  `isaacsim-extscache-physics` etc. are pip packages *inside* `cog_isaac`, so nothing has to be
  pre-staged separately -- which matters because compute nodes have no internet.

**CINECA's own documentation endorses this route** (checked after the user suggested reading it):
*"In order to move locally built SIF images on CINECA's clusters, consult the 'Data Transfer'
page."* Build locally, transfer. Fakeroot is not mentioned as a user-available option anywhere,
consistent with our measurement that it is unmapped. The docs also specify **CUDA 12.2 for
Leonardo** and `singularity exec --nv`, both already satisfied (D22 verified cu128 on that driver).

**One refinement left on the table:** CINECA assume you transfer a finished `.sif`. We transfer a
`docker save` tar and convert on a compute node, because this box has no apptainer. A sif is about
half the size (the base image was 15 GB as a tar vs **7.1 GB as a sif**), so building locally would
halve the upload and delete the cluster-side conversion job. It needs `apt install apptainer`, a
system change outside the repo, so it is offered to the user rather than done.

### 2026-08-19 20:12 — Duplicate Slurm job from my own automation; TaskStop does not undo what already happened

The hourly watchdog caught **two `cog_build_sif` jobs running on the same node** (52941771 and
52941866), both converting the same 24 GB archive to the same output path. Cause: I had armed a
background watcher to verify the upload byte count and then auto-submit the conversion. When the
upload finished I verified the byte count myself, decided to submit manually, and called TaskStop
on the watcher first -- but the watcher had **already submitted** a fraction of a second earlier,
and its output was buffered so nothing in its log showed that yet. Stopping a task does not retract
its side effects.

Consequence was waste rather than damage -- each build gets its own `build-temp-<random>` under
`SINGULARITY_TMPDIR`, so they were not corrupting each other's scratch, but both would have written
`cog-env-5.1.0.sif` with `--force` and they were occupying 32 cores of one node between them.
Cancelled the later one (52941866); 52941771 continues.

**Two rules to carry forward:**
1. **Automation that submits jobs must be idempotent.** A watcher that fires `sbatch` should first
   check `squeue` for a job of the same name, or write a claim file. Mine did neither.
2. **Never assume a stopped task did nothing.** After TaskStop, check the world (here: `squeue`),
   not the task's log -- especially when its output is buffered behind a pipe.

This is the third time today that buffering hid a fact I needed (`tail` swallowing a build's real
error, `tail` hiding `singularity`'s exit code, and now a watcher's submission). Worth stating as a
habit: **when a background command's output is piped, treat its log as unavailable until it exits,
and verify from the system instead.**

### 2026-08-19 20:36 — G5a CALIBRATION COMPLETE: 2:00:04, exactly 2.0 GPU-h, loss 0.579 -> 0.004

`t1_L0_n25_s0` (job 52899856) finished: **`COMPLETED 02:00:04`**, `billing=8` -> **2.0 GPU-h
exactly**, all four checkpoints present (`020000 040000 060000 080000 last`) and
`training_step.json` = `{"step": 80000}`. The estimate derived from the 20k checkpoint two hours
earlier was "~1.98 h + ~2.3 min startup"; the outturn was 2:00:04. **The per-cell cost is now a
measurement, not a projection.**

### Loss curve (recovered from wandb's binary log -- see the trick below)

| step | loss | grad_norm | lr | updt_s | data_s |
|---|---|---|---|---|---|
| 200 | 0.579 | 3.088 | 2.0e-05 | 0.210 | 0.077 |
| 7K | 0.096 | 0.264 | 9.8e-05 | 0.069 | 0.014 |
| 15K | 0.075 | 0.253 | 9.3e-05 | 0.069 | 0.014 |
| 22K | 0.059 | 0.270 | 8.3e-05 | 0.069 | 0.015 |
| 36K | 0.034 | 0.272 | 5.8e-05 | 0.069 | 0.020 |
| 51K | 0.015 | 0.282 | 3.0e-05 | 0.069 | 0.019 |
| 65K | 0.006 | 0.270 | 8.6e-06 | 0.069 | 0.014 |
| 80K | **0.004** | 0.212 | 5.2e-10 | 0.069 | 0.014 |

Three conclusions:
1. **It is training properly.** Monotone decrease over 400 logged intervals, grad-norm flat around
   0.22-0.28 after warmup, no divergence or spikes.
2. **The LR schedule is the one we specified.** Warms to ~1e-4 by 7K (cosine, 500-step warmup) and
   decays to 5.2e-10 by 80K, which independently confirms `--policy.optimizer_lr=1e-4` set the
   *peak* -- the third confirmation that bug 4 is dead.
3. **The 20 % throughput gap is now explained exactly, not hand-waved.** `data_s` here is
   **0.014-0.021**, versus 0.003 in the throughput smoke that had 32 cores allocated. Step time
   0.069 + ~0.015 = ~0.085 s -> ~11.8 steps/s, and the measured 11.2-11.4 is that minus checkpoint
   saves. Eight cores genuinely does leave the loader slightly short of the GPU. It is still the
   right allocation (billing is linear in cores), but the *reason* is now on record.

The first interval is again visibly unrepresentative (`updt_s 0.210, data_s 0.077` at step 200) --
the warm-up artifact that any future measurement must discard.

**Worth flagging scientifically:** a final loss of 0.004 on 25 demonstrations is very low, which is
what memorisation looks like. That is expected at small N and is precisely the effect this study
exists to quantify -- it is a reason to watch the N=10/25 cells' *eval* numbers closely, not a
defect.

### How to read wandb's offline logs when there is no output.log

wandb offline writes **no `files/output.log`** while or after a run -- the offline run directory
holds only `logs/debug*.log`, `files/requirements.txt` and the binary `run-<id>.wandb`. Since
wandb's console redirect also silences the Slurm `.out` (see the 18:40 entry), the training log
looks lost. It is not:

```python
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb
ds = datastore.DataStore(); ds.open_for_scan(path_to_run_wandb)
# rec.WhichOneof("record_type") in {"history","output_raw","stats","summary",...}
```

The file here held 2,567 records: **400 `history`** (= 80,000 / log_freq 200, exactly right) and
**419 `output_raw`** -- the captured console stream. The `history` items came back with **empty
`key` fields** (this wandb version stores them under a nested key path), so the practical route is
`output_raw`: concatenate `rec.output_raw.line` and re-parse LeRobot's own log format. That
recovered all 400 rows. Recorded because "wandb ate the log" will happen again on every cluster
run, and syncing to the cloud is not an option from a compute node.

### 2026-08-19 20:50 — G5b: Kit runs in the shipped image but dies during startup; STOPPED at the bound. A100 rendering still UNANSWERED

Two gate attempts against `cog-env-5.1.0.sif` (jobs 52956838, 52959414). Both end the same way:

```
... [Warning] [carb] Potential plugin preload failed: .../libomni.hydra.rtx.plugin.so
... [Warning] [carb] Recursive unloadAllPlugins() detected!
frames_qa rc=0
[gate] 0 PNG(s)
```

**Where this actually is, stated precisely.** This is much further than the NVIDIA-image route ever
got (that failed at `rc=126`, unable to execute anything). Kit now *starts inside the container*:
it reads its user config, initialises carb, loads extensions, warns about the missing X display as
expected for headless, and then tears itself down after ~18 s and **exits 0**.

The decisive detail is not the exit code -- it is that `frames_qa.py`'s **module-level**
`os.makedirs(OUT)` (line 26, immediately after the AppLauncher block) never runs: the writable
directory bound over the repo's `ops/` stays completely empty. So the process is terminated from
*inside* the `AppLauncher`/`SimulationApp` constructor, before any of our code executes. Nothing
our script does is implicated, and no Python traceback is produced -- Kit exits 0 on a fatal
startup error, which is exactly the D6 pattern this project already knew about and which the gate
was written to defend against (it judges pixels, not `$?`, which is why it reported honestly
instead of passing).

**What was found and fixed on the way (a real bug, just not the last one).** The Vulkan ICD was
genuinely missing. Measured on a compute node (job 52958496):

| | host (lrdn0332) | inside container, `--nv` |
|---|---|---|
| `/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json` | present | **directory absent** |
| `VK_ICD_FILENAMES` | -- | **unset** |
| `libvulkan.so.1` | -- | loads fine |

A Vulkan loader with no ICD and no driver sees **zero devices**, which alone is enough to stop the
RTX renderer. Fixed in the gate by binding the host ICD directory and setting
`VK_ICD_FILENAMES` -- no image rebuild needed -- and verified: `ICD visible: .../nvidia_icd.x86_64.json`.
Also corrected a false negative of mine: `--nv` *does* inject the driver, 46 libs into
`/.singularity.d/libs` including `libGLX_nvidia` and `libnvidia-glvkspirv`. My earlier
`ldconfig -p | grep glvkspirv` returned 0 because **ldconfig reads /etc/ld.so.cache, not
LD_LIBRARY_PATH** -- the libs were there all along.

**Stopping here, deliberately.** I set a bound of one diagnostic plus one fix on this optional item
before starting, and both are spent. The remaining gap is a Kit-startup failure inside a
hand-built ubuntu:24.04 base that is missing something the official Isaac Sim image installs -- the
`Potential plugin preload failed` lines for the RTX/hydra plugins are the obvious suspects, and the
honest next step is to obtain the *reference* warning set by running the identical `frames_qa.py`
locally where it works, and diff. That is tractable but it is not a five-minute job.

**The A100 rendering question is therefore still formally UNANSWERED.** Neither gate attempt ever
reached a renderer, so nothing here says anything about issues #3421/#1519. I am not recording a
guess as a result.

**Recommendation unchanged, and now cheaper than when first made:** evaluate T1 on the local 4090.
D24 cut the workload from 72 checkpoint-evals to 24+2, so the fallback the plan already accepted is
now a third of the size it was when it was chosen. Revisit the container only if Tasks 2-3 make
eval throughput binding again -- and if so, start from the *reference-warning diff*, not from
another round of guessing at apt packages.

**Assets kept for whoever resumes it:** `cog-env-5.1.0.sif` (9.8 GB) and `isaac-sim-5.1.0.sif`
(7.1 GB) on `$WORK/cog/containers/`; `slurm/build_sif.sbatch`, `slurm/probe_vulkan.sbatch`,
`slurm/debug_a100_cogenv.sbatch`, `docker/Dockerfile.cog_env` and `docker/Dockerfile.cog`, each
carrying its findings in comments.

### 2026-08-19 21:50 — FIRST STUDY RESULT: L0/N=25 @40k = SR 0.97, and L0 looks saturated

Local eval on the 4090 (the accepted fallback), frozen protocol, 100 episodes:

| cell | checkpoint | successes | SR | Wilson 95% CI |
|---|---|---|---|---|
| t1_L0_n25_s0 | 040000 | 97/100 | **0.970** | [0.915, 0.990] |

`Cog-CupPlace-L0-IK-Rel-Visuomotor-v0`, `num_inference_steps=10`, protocol `num_envs=20 x 5
batches, base_seed=5000`. ~14 min per checkpoint on the 4090 alongside the foreign job.

**This is the first real data point of the study, and it already says something about the design.**
L0 is the easiest level -- fixed mug pose, fixed goal, one mug -- and **25 demonstrations are already
enough for 97 %**. Consequences worth stating before the matrix runs:

- **N\*(s) for L0 is <= 25 for every threshold we planned to report** (50/80/90 %). The L0 row of the
  demos-vs-success surface will be flat across N in {25..400}: saturated.
- Therefore **all of L0's discriminating information sits at N=10**, the one grid point below this.
  That retrospectively justifies including N=10 in the grid, and it means the L0 curve is
  essentially a two-point curve (10 and "saturated").
- The interesting variation must come from **L1-L3**, where randomisation is added. If those also
  saturate early the study has a ceiling problem; if they do not, the cost-of-generality ratios
  N\*(L_k)/N\*(L0) become large precisely because the L0 denominator is small. Either way this is
  the number to watch when the matrix lands.
- It also explains the training loss of 0.004: at N=25 on a fixed-pose task the policy can very
  nearly memorise, and memorisation is sufficient here **because the eval distribution is the
  training distribution** at L0. That is not a defect; it is the baseline the higher levels are
  measured against.

Recorded in the registry as `sr_40k=0.97`, `eval_n=100`. 60k and 80k are running -- the D24
comparison needs all three before the last-checkpoint-only protocol can be judged sound.

### 2026-08-19 22:00 — D24 CHECKPOINT COMPARISON RESOLVED: last-checkpoint-only is sound

The one-time 40k/60k/80k comparison D24 requires, on `t1_L0_n25_s0`, frozen protocol, 100 episodes
each on the local 4090:

| checkpoint | successes | SR | Wilson 95 % CI |
|---|---|---|---|
| 040000 | 97/100 | 0.970 | [0.915, 0.990] |
| 060000 | 95/100 | 0.950 | [0.888, 0.978] |
| **080000** | **98/100** | **0.980** | [0.930, 0.994] |

**Verdict: the last checkpoint is not systematically worse -- here it is the best of the three, and
`best_of_last_3` (0.980) is IDENTICAL to `last_checkpoint` (0.980).** All three intervals overlap
heavily. The spread, 3 points, is the same order as the binomial standard error at n=100 and
p≈0.97 (~1.7 points), so checkpoint choice contributes no more noise than sampling already does.

**So D24 is discharged in the affirmative and the full-scale protocol stands: evaluate 080000 only.**
This is what I committed to reporting either way -- had 80k come in materially below 60k it would
have gone back to the user before the matrix was evaluated, because the headline metric would then
have been measuring late-training degradation rather than data efficiency.

Two caveats kept on the record rather than buried:
1. **One cell, one level, one seed.** This says late training is stable for L0/N=25, a saturated
   cell where all three checkpoints are near ceiling and differences are hard to see by
   construction. A cell in the steep part of the curve (low N, or a harder level) could behave
   differently. If any future cell's number looks anomalous, re-running its 40k/60k is cheap
   (~14 min each locally) and the checkpoints are already saved -- which is exactly why
   `--save_freq=20000` was deliberately left in place under D24.
2. Ceiling effect: at 0.95-0.98 there is very little room for a checkpoint to look better, so this
   test had limited power to detect a *small* late-training gain. It had ample power to detect the
   thing we actually cared about -- a large late-training loss -- and found none.

Registry updated: `sr_40k=0.97 sr_60k=0.95 sr_80k=0.98 sr_best=0.98`, status `done`.

### 2026-08-19 22:15 — G5b, the full Vulkan investigation: one package fixed 99 of 101 errors, five hypotheses disproved, and Kit now runs but on the CPU

Recorded in one entry because an audit found that everything after the 19:50 "stopped at the bound"
note was missing from this journal -- roughly two hours of findings, including the most useful one.
The user asked me to reopen this ("easy fixes?", "you are not the first one attempting this"), which
was the right push: prior art and Kit's own log turned a vague "needs more work" into named causes.

#### The breakthrough that made everything else visible: writable binds

Kit writes its log, `user.config.json`, crash dumps and shader cache **into its own install tree**,
which sits inside the read-only squashfs. Locally that tree is a writable conda env, so nobody ever
notices. Binding writable host dirs over `<env>/site-packages/isaacsim/kit/{logs,data,cache}`
(seeding `data` from the image first, or the bind hides `user.config.json`) fixed it -- Kit promptly
wrote **310 shader-cache files** and, crucially, **its own log**. Every finding below came out of
that log; before it, all we had was stdout, which wandb-style redirection and Kit's exit-0 behaviour
render nearly useless. `--writable-tmpfs` was rejected deliberately: its overlay is capped by
`sessiondir max size` (tens of MB) and the shader cache is far larger.

#### One missing package explained 99 of 101 errors

From Kit's log, all missing shared objects:

| count | library | nature |
|---|---|---|
| 18 | **`libgomp.so.1`** | genuinely missing system lib (`libgomp1`) |
| 73 | `libomni.usd.so` | cascade from the above |
| 6 | `libusd_hd.so` | cascade |
| 1 + 1 | `librtx.hydra.so`, `libosdCPU.so.3.6.0` | cascade |

`ubuntu:24.04` does not ship `libgomp1`; NVIDIA's isaac-sim image does (checked). Rather than rebuild
24 GB to test, the library was staged on `$WORK` and prepended to `LD_LIBRARY_PATH` -- copied from
this workstation's Ubuntu 24.04, i.e. the same distro the image is built from, so ABI-exact. Result:
**libgomp errors 0**, and for the first time our own code executed inside the container
(`frames_qa.py`'s module-level `os.makedirs` created its output dir). Kit went from dying at 43 s to
running 5+ minutes, booting IsaacLab, parsing `FrankaCupPlaceVisuomotorEnvCfg_L0` and **building the
scene**. *(This belongs in the Dockerfile, not a bind, before any production use.)*

#### Where it stands: Kit runs, but on the CPU

```
[Error] carb.graphics-vulkan: VkResult: ERROR_INCOMPATIBLE_DRIVER
[Error] vkCreateInstance failed. Vulkan 1.1 is not supported, or your driver requires an update.
[Error] omni.gpu_foundation_factory: Failed to create any GPU devices, including compatibility mode
[Error] omni.kit.renderer: GPU Foundation is not initialized!
[Error] omni.physx: CUDA libs are present, but no suitable CUDA GPU was found!
[INFO]: Time taken for scene creation : 274.145037 seconds
```

274 s for a scene that takes ~20 s on a GPU is the signature of the CPU fallback. Note torch uses the
same A100 perfectly (D22), so this is Kit's Vulkan path alone.

#### Five hypotheses, each disproved by measurement

| # | hypothesis | test | outcome |
|---|---|---|---|
| 1 | Vulkan ICD absent in container | host has `nvidia_icd.x86_64.json`; container's `/usr/share/vulkan/icd.d` did not exist | **real** -- bound it; error unchanged |
| 2 | driver 535.255+ reports its Vulkan version wrongly (NVIDIA's documented workaround) | `--kit_args=--/rtx/verifyDriverVersion/enabled=false` | no effect -- that gates Kit's *own* check, not `vkCreateInstance` |
| 3 | ICD's bare soname stops resolving because Kit rewrites `LD_LIBRARY_PATH` | rewrote `library_path` to `/.singularity.d/libs/libGLX_nvidia.so.0` | no effect |
| 4 | the loader never read our ICD at all | put the absolute-path ICD **in** `/usr/share/vulkan/icd.d` via bind | no effect |
| 5 | `/dev/nvidia-modeset` not bound (Vulkan needs it, CUDA does not) | listed `/dev/nvidia*` inside the container | **already present** -- `--nv` binds all of them |

Facts established along the way, each worth keeping:
- **The ICD is valid**: `{"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.242"}`, and that
  library **is** injected (46 libs in `/.singularity.d/libs`) and **dlopens successfully** inside the
  container.
- **Kit uses its OWN bundled Vulkan loader** (`omni.gpu_foundation-*/bin/deps/libvulkan.so.1`), and
  that loader **ignores `VK_LOADER_DEBUG` entirely** -- `=all` produced **zero** LOADER lines even
  though the container demonstrably receives our `--env` values. If it ignores that, it very likely
  ignores `VK_ICD_FILENAMES`/`VK_DRIVER_FILES` too, which would retroactively explain why
  hypotheses 3 and 4 changed nothing: the loader was never reading the file we were editing.
- **The host is fully Vulkan-capable**: `nvidia_drm` and `nvidia_modeset` loaded, `/dev/nvidia-modeset`
  and `/dev/nvidia-caps` present, `libGLX_nvidia` / `libnvidia-glcore` / `libnvidia-glvkspirv` all
  installed, Display Mode Disabled (normal for a datacenter GPU), Compute Mode Default.
- **`--nvccli` is not an option here**: Apptainer documents it plus
  `NVIDIA_DRIVER_CAPABILITIES=graphics` as the way to get Vulkan/GL, but `nvidia-container-cli` is
  not installed on Leonardo. Leonardo's `/etc/singularity/nvliblist.conf` *does* already list the
  graphics libs, so plain `--nv` injects them -- the libraries were never the problem.

Also fixed while here: `--kit_args` must use the **`=` form** (`--kit_args=--/rtx/...`); argparse
rejects a space-separated value that begins with `--` and exits 2, so the flag never reaches Kit.
IsaacLab's own docs show the space-separated form; it does not work.

#### Conclusion and recommendation

**A100 rendering remains formally unanswered** -- five attempts, none of which reached a renderer, so
nothing here speaks to upstream issues #3421/#1519. Recording a guess would be worse than recording
nothing.

This is now a **site question**: does Vulkan work inside Singularity on Leonardo at all? CINECA can
answer that in one reply, and we have exactly what a ticket needs -- a minimal reproduction, the
exact error, and five documented dead ends. That is a better next step than a sixth guess.

**Recommendation: evaluate T1 locally** (D25). D24 already cut the workload to 24+2 evals, the local
path is proven end-to-end today, and it costs zero grant hours.

Artifacts kept: `cog-env-5.1.0.sif` (9.8 GB) and `isaac-sim-5.1.0.sif` (7.1 GB) on
`$WORK/cog/containers`; `slurm/{build_sif,debug_a100_cogenv,probe_vulkan,probe_icd,probe_nvlibs}.sbatch`;
`docker/Dockerfile.cog_env`; each carrying its findings in comments.

### 2026-08-19 22:15 — Local eval path is production-ready

`scripts/ops/run_local_eval.sh` runs a cell's checkpoints on the 4090 and **waits for GPU headroom
before starting** (default >= 14 GB free, polled every 2 min). The wait exists because the frozen
protocol fixes `num_envs=20 x 5 batches` and `num_envs` comes from
`configs/eval_sets/protocol.json`, **not** a CLI flag -- so our GPU footprint is not adjustable
without changing which seeds define the benchmark (rule 8). Shrinking was therefore not an option;
waiting was. The foreign `lp-eval` job is never touched (rule 2).

Measured: **~14 min per checkpoint** (100 episodes) on the 4090 while sharing it, ~6.8 GB VRAM.
So the full T1 eval under D24 -- 23 cells x 1 checkpoint -- is roughly **5-6 h of local wall-clock
and zero grant hours**.

Supporting fixes: `slurm/eval.sbatch` gained a container mode (`COG_SIF`) alongside the conda path so
one script serves cluster and local; `sync_down.sh checkpoints` now pulls **all three** protocol
checkpoints (it only pulled 080000, which would have silently degraded best-of-3 to last-only) and
only their `pretrained_model/` subdirs, cutting the transfer from 9 GB to 3 GB per cell because eval
never reads optimizer state.

## 2026-08-19 (later) -- Cross-version audit of the diffusion-policy defaults (0.4.4 vs 0.6.1)

Question raised: what exactly differs in the DP defaults between our pinned 0.4.4 and the latest
release? PINS.md already listed five 0.6.0 flips from PR #3202, but that list had never been
checked against source, and one adjacent claim of mine turned out to be wrong. Verified properly
this time: downloaded the 0.5.0/0.5.1/0.6.0/0.6.1 wheels, extracted
`lerobot/policies/diffusion/`, and tabulated **all 29 DiffusionConfig defaults** across the five
releases (0.4.4 read from the installed `cog_isaac` env).

Result -- six deltas total, no more:

| default | 0.4.4 (ours) | 0.6.1 | lands in |
|---|---|---|---|
| `horizon` | 16 | **64** | 0.6.0 |
| `n_action_steps` | 8 | **32** | 0.6.0 |
| `pretrained_backbone_weights` | None | **ResNet18_Weights.IMAGENET1K_V1** | 0.6.0 |
| `use_group_norm` | True | **False** | 0.6.0 |
| `use_separate_rgb_encoder_per_camera` | False | **True** | 0.6.0 |
| `gradient_checkpointing` | (absent) | False (new field) | **0.6.1** |

Everything else is byte-stable across all five releases: the optimizer preset (1e-4, betas
(0.95,0.999), eps 1e-8, wd 1e-6), cosine schedule + 500 warmup steps, DDPM/100 steps/
squaredcos_cap_v2/epsilon/clip_sample, `num_inference_steps=None`, resnet18, `down_dims`,
kernel 5, `n_groups` 8, embed dim 128, FiLM scale modulation, 32 spatial-softmax keypoints,
`crop_is_random`, `n_obs_steps=2`. `__post_init__` validation is identical too.

Four findings the docs did not have:

1. **0.5.0 and 0.5.1 are identical to 0.4.4 on every DP default.** We already knew their
   `video_utils.py` was byte-identical (D23, so no dataloader gain); now the *policy* side is
   confirmed unchanged as well. The rejected 0.5 bump was an architectural no-op, not merely a
   performance no-op -- which is worth knowing if a 0.5-only bugfix ever becomes attractive.
2. **A correction.** I had earlier written that 0.6.0 flipped `do_mask_loss_for_padding` as well.
   It does not: that default is `False` in every release 0.4.4 through 0.6.1. `diffusion_base.sh`
   pins it, which is harmless and still worth keeping, but it counters no upstream flip. The
   config file's own comment was right; my summary of it was not. Both are now precise.
3. **The three encoder flips are coupled, and our pins turn a bump into a LOUD failure.**
   `modeling_diffusion.py:507-511` (0.6.1; :481-485 in 0.4.4) raises `"You can't replace BatchNorm
   in a pretrained model without ruining the weights!"` when `use_group_norm=True` and
   `pretrained_backbone_weights` is set. That is *why* 0.6.0 had to flip `use_group_norm` to False
   in order to default the ImageNet weights on -- the two cannot both be enabled. Since
   `diffusion_base.sh` pins `use_group_norm=true`, a bump to 0.6.x would abort at model
   construction rather than silently train a different encoder. This retroactively justifies the
   decision to leave `pretrained_backbone_weights` unpinned (draccus would risk decoding "null"):
   the guard rail is the coupling, not the pin.
4. **An upstream inconsistency in 0.6.x.** `drop_n_last_frames` stayed at 7 while `horizon` went
   16 -> 64, so 0.6.x's own inline comment (`# horizon - n_action_steps - n_obs_steps + 1`, i.e.
   64-32-2+1 = 31) contradicts its own default. Anyone training 0.6.x defaults samples windows
   that pad more than the code claims to intend. Irrelevant to this study because we pin horizon,
   n_action_steps, n_obs_steps and drop_n_last_frames explicitly -- which is the point of pinning
   all four rather than relying on the formula.

Net effect on the study: **none of the six deltas touch our runs.** Five of the six are explicitly
pinned in `configs/train/diffusion_base.sh` and the sixth (`gradient_checkpointing`) does not
exist in 0.4.4 and is behaviour-neutral where it does. The calibration run's saved `config.json`
was already verified against these pins at G5a. What changed is the confidence level: the flip
list is now source-verified rather than release-note-derived, and one wrong adjacent claim is
retracted.

## 2026-08-19 (evening) -- Architecture switched to per-camera RGB encoders; full T1 matrix LAUNCHED

User directive: use a separate RGB encoder per camera for the study, re-run the already-trained
cell, extend train walltime because the model has more parameters, and check health hourly.
Decision recorded as D26. Sequence, in the order it had to happen:

**1. Verified 0.4.4 actually supports the path** before changing anything.
`modeling_diffusion.py:176-182` builds `nn.ModuleList([DiffusionRgbEncoder(cfg) for _ in cams])`
and `:252-265` runs each encoder on its own camera stream (`zip(..., strict=True)`), concatenating
the per-camera features. Notable: `global_cond_dim` is `feature_dim * num_images` in **both**
branches, so untying the encoders does NOT widen the U-Net conditioning input.

**2. Measured the cost** by building `DiffusionPolicy` both ways with our real feature spec
(2 cams @128x128 from `data/lerobot/L0/meta/info.json`, crop 112, state 9, action 7):

| | shared | separate | delta |
|---|---|---|---|
| encoder params | 11,197,088 | 22,394,176 | **x2.00** |
| U-Net params | 255,601,287 | 255,601,287 | **identical** |
| total params | 266,798,375 | 277,995,463 | +11.2M (+4.2%) |

fwd/bwd at batch 64 succeeded, peak 3.44 GiB allocated -- VRAM is a non-issue on a 64 GiB A100
(and the earlier 13.5 GiB figure was reserved-including-dataloader, not model activations).

**3. Caught a silent-wrong-result hazard before it fired.** `train.sbatch:79-84` resumes with
`--config_path=<ckpt> --resume=true` -- mandatory at 0.4.4, but on that path the config is taken
ENTIRELY from the checkpoint and all other CLI flags are ignored. `t1_L0_n25_s0` already had
80k-step shared-encoder checkpoints on `$WORK`. Submitting the matrix would therefore have found
that cell "already complete" under the OLD architecture, with the registry, the log line and the
frozen config all claiming the new one. Confirmed by reading the saved configs: all three pulled
checkpoints report `sep_enc=False, group_norm=True, horizon=16, batch=64`.
Two fixes: (a) renamed `$WORK/cog/checkpoints/t1_L0_n25_s0` ->
`t1_L0_n25_s0_sharedenc` (renamed, never deleted -- it is the evidence behind D24 and the L0
saturation finding), and registry row likewise renamed with `status=superseded`; (b) wrote
`scripts/ops/assert_resume_config.py`, called from `train.sbatch` before any resume, which diffs
every `--policy.*` value and `batch_size` in the checkpoint against `COG_DP_FLAGS` and aborts with
exit 2 on mismatch. Tested against the real stale checkpoint both ways: with the new flag it
reports exactly one mismatch (`use_separate_rgb_encoder_per_camera: checkpoint=False frozen=true`)
and nothing else; with the old flag restored, `16 frozen values match`. The single-mismatch result
is itself the evidence that the value normalisation handles bools, `[112,112]`, ints and strings
without false positives. This guard is generic: it catches ANY later edit to the frozen config.

**4. Caught a second silent hazard: `sync_up.sh` hardcoded `REPO=/home/admin_07/cost_of_generality`**
-- the MAIN checkout. This session works in a git worktree (background-job isolation), so syncing
would have shipped the OLD `sep_enc=false` config to the cluster while the local branch showed
`true`, and 24 jobs would have trained the wrong architecture. Now `REPO="${COG_REPO:-...}"`, and
the script echoes which repo it is syncing from. Verified after syncing that
`$WORK/cog/repo/configs/train/diffusion_base.sh:24` really reads `=true` and that the guard script
arrived.

**5. Walltime 06:00:00 -> 12:00:00** per user request. Free insurance: Leonardo bills
cores x ELAPSED, not the reservation, so a bigger limit costs nothing and only a walltime kill
costs a requeue.

**6. Launched the full T1 matrix: 24 cells** (L0-L3 x N=10/25/50/100/200/400, seed 0), job ids
53008600-53008873, all **RUNNING within ~20 s** of submission -- consistent with the 4 s queue
latency probe and again showing `sbatch --test-only`'s "start 2026-08-26" prediction is worthless.
Datasets verified present on `$FAST` first: L0-L3, 400 episodes each. 24 registry rows written.
Expected ~2 h/cell if throughput holds (~48 GPU-h); that estimate is provisional because it was
measured on the shared-encoder architecture.

Note for the next reader: the Slurm logs stay nearly empty because wandb's `_redirect()` swallows
the console stream (recorded earlier today), so liveness must be judged from checkpoint mtimes and
`squeue`, not from log growth.

## 2026-08-19 (late) -- CUDA works inside the .sif; the Vulkan failure is isolated, not GPU access

While assembling a CINECA support ticket I re-read Kit's own log properly and found it reports CUDA
failures alongside the Vulkan one -- "no CUDA-capable device is detected" (`omni.physx.tensors`,
`omni.gpucompute-cuda`), plus "CUDA libs are present, but no suitable CUDA GPU was found!". Taken at
face value that would mean `--nv` GPU access is broken in the container, which is a completely
different ticket from a Vulkan/graphics one. So I tested it instead of assuming (job 53013894,
boost_qos_dbg, 14 s):

Inside `cog-env-5.1.0.sif` with `singularity exec --nv` on lrdn2482:
`nvidia-smi` -> 535.274.02 / A100-SXM-64GB; `/dev/nvidia0..3`, `nvidiactl`, `nvidia-uvm`,
`nvidia-uvm-tools`, `nvidia-modeset`, `nvidia-caps` all present; `libcuda.so.1` injected at
`/.singularity.d/libs/`; **torch 2.7.0+cu128 `is_available=True`, `device_count=1`,
`get_device_name(0)='NVIDIA A100-SXM-64GB'`, and a 1024x1024 matmul succeeded.**

So GPU access via `--nv` is fine and the CUDA errors in Kit's log are DOWNSTREAM FALLOUT of the
Vulkan failure, not an independent cause. The evidence for that ordering is in the log itself:
the CUDA errors all appear AFTER `[omni.gpu_foundation_factory.plugin] Failed to create any GPU
devices`, one of them reports a garbage device ordinal (`device 1294759088`), and
`omni.graph.core` says "unable to get a valid CUDA device id **from the renderer**". Omniverse
enumerates devices through its own Vulkan-based GPU foundation, so when that yields no devices its
CUDA interop has no ordinal to map. This confirms the diagnosis is Vulkan-only -- previously an
assumption, now measured.

Exact failure, verbatim from `kit_rw/logs/Kit/Isaac-Sim/5.1/kit_20260819_214819.log:1680-1720`:
```
[Error] [carb.graphics-vulkan.plugin] VkResult: ERROR_INCOMPATIBLE_DRIVER
[Error] [carb.graphics-vulkan.plugin] vkCreateInstance failed. Vulkan 1.1 is not supported, or your driver requires an update.
[Error] [gpu.foundation.plugin] carb::graphics::createInstance failed.
[Error] [omni.gpu_foundation_factory.plugin] Failed to create any GPU devices, including an attempt with compatibility mode.
```
(twice -- the second is the compatibility-mode retry.)

Two further details found in the same log, both useful for the ticket:
1. **Kit blanks the Vulkan environment variables itself** (`:1604-1608`): `VK_SDK_PATH`,
   `VULKAN_SDK`, `VK_LAYER_PATH`, `VK_INSTANCE_LAYERS`, `VULKAN_HEADERS_INSTALL_DIR` are all logged
   as "Environment variable overridden: <name> = " (empty). This is the mechanism behind the
   earlier finding that Kit ignores loader env vars -- it is not that they are unread, it is that
   Kit actively clears them, so no client-side loader override can work.
2. **All NVIDIA Vulkan support libraries ARE injected** and version-matched to the host driver:
   `libnvidia-glvkspirv.so.535.274.02`, `libnvidia-rtcore`, `libnvidia-glcore`, `libnvidia-glsi`,
   `libnvoptix.so.1`, `libGLX_nvidia.so.{0,535.274.02}`; `nvliblist.conf` lists glvkspirv. So the
   common "missing glvkspirv" explanation for ERROR_INCOMPATIBLE_DRIVER does not apply here.

Net: valid ICD (api_version 1.3.242), all support libs present and version-matched, driver 535.274.02
(Vulkan 1.3-capable), device nodes present, CUDA fully working -- and `vkCreateInstance` still
returns ERROR_INCOMPATIBLE_DRIVER. Every ordinary cause is eliminated, which is what makes this
worth a CINECA ticket rather than more local guessing. Cost of the check: 14 s on one A100.

## 2026-08-20 (00:50) -- T1 matrix mostly trained; eval started, two bugs found in my own harness

**Training.** 16 of 24 cells COMPLETED, 19 have written `080000`, 8 still RUNNING, **zero FAILED**.
Elapsed per completed cell **1:50:24 - 2:05:43**, i.e. ~1.95 h -- better than the 2.35 h projected
from the early-run rate, because throughput improves as the page cache warms. `AllocTRES` is
`billing=8,cpu=8,gres/gpu=1` on every cell, so 8 billing-h per wall-hour = 1 A100-h per wall-hour,
and elapsed hours are GPU-h directly. Matrix therefore lands at **~47 GPU-h**, close to the original
48 estimate rather than the 56 re-forecast.

**Unexpected ordering: the SMALLEST datasets are the slowest.** All four `n10` cells plus
`L1_n400` are the stragglers, while `n400` cells finished first. With a fixed 80k steps the demo
count should not change step cost, so this is dataloader behaviour, not model behaviour -- a
10-episode dataset has very few distinct sampleable windows, so workers cycle the same short
episode list constantly and get less benefit from readahead than a 400-episode dataset streaming
through many files. Worth remembering for Tasks 2-3 scheduling: the cheap-looking cells are not the
fast ones. Not investigated further, since it costs nothing at this scale.

**Eval harness: two bugs, both mine, both caught before any number was recorded.**

1. **Directory-vs-weights race.** My eval driver gated on the checkpoint *directory* existing.
   rsync creates the directory first and only then transfers the 1.1 GB `model.safetensors`, so the
   driver raced my own background sync and launched Isaac on a half-pulled checkpoint ->
   `FileNotFoundError: .../model.safetensors`. Kit still exited **0** (D6 again), so only the
   missing-artifact check caught it. Fixed to gate on `model.safetensors` being non-empty; rsync
   renames into place atomically, so the final name appearing does mean the transfer finished.
2. **Busy-retry with no backoff.** Because a failed cell left no result JSON, the driver's outer
   loop immediately retried the same cell, booting Isaac every ~10 s. Three iterations ran before I
   killed the tmux session. Now a cell that fails is recorded in a `FAILED` map and skipped for the
   rest of the run, and the exit line reports which cells failed.

**A third, more dangerous one: stale eval results from the superseded architecture.**
`results/eval_L0_n25_{040000,060000,080000}.json` were still present from the shared-encoder
calibration run. The driver skips a cell whose result JSON exists, so `t1_L0_n25_s0` -- the very
cell re-run to put L0/N=25 on the new architecture -- would have been silently skipped, and those
three files would later have been read as new-architecture results. Renamed to
`*_sharedenc.json` (preserved, not deleted: they are the D24 evidence). This is the same class of
error as the checkpoint-resume hazard: an artifact from a superseded configuration sitting exactly
where the new one belongs. Lesson recorded: when an architecture changes, sweep for EVERY artifact
keyed by run id -- checkpoints, registry rows AND result JSONs.

**Supporting fixes.** `sync_down.sh` and `run_local_eval.sh` now honour `COG_REPO` like
`sync_up.sh`, so a worktree-isolated session pulls and evaluates in its own tree instead of
silently in the main checkout. `sync_down.sh` also gained `COG_SYNC_STEPS`, defaulting to
`080000` alone: D24 made the protocol last-checkpoint-only, so pulling 40k/60k/80k would have
tripled 1.1 GB x 24 cells for nothing. Verified on one cell: 1.03 GB pulled in ~47 s, only
`080000` present, and its `config.json` reports `use_separate_rgb_encoder_per_camera: true`.

Eval is now running under tmux (`cog_eval`) on `t1_L0_n25_s0`, gated on >=14 GB free VRAM so the
foreign eval job is untouched (rule 2). All 44+ cluster checkpoints checked: **44/44 report
`sep_enc=True`**, no invalid cell anywhere.

## 2026-08-20 (01:30) -- T1 training wave COMPLETE: 24/24, 51.3 GPU-h, zero failures

`MATRIX_QUEUE_EMPTY` after 34 polls. **All 24 cells COMPLETED**, all 24 wrote `080000`, and no cell
failed, timed out, or was requeued -- so the resume path was never exercised in anger.

| | value |
|---|---|
| cells | 24 / 24 COMPLETED |
| elapsed per cell | min 1.84 h, median 2.03 h, max 2.83 h |
| **total** | **51.3 GPU-h** (billing=8 on every cell -> elapsed hours == A100-h) |
| forecast it replaces | 48 (original), 56 (warm-up-based re-forecast) |

So the original estimate was the better one, and the 12 h walltime was never close to binding (worst
cell used 24% of it).

**The small-N slowdown is real and large.** The five slowest cells are 2.50-2.83 h and are the four
`n10` cells plus `L1_n400`; the fastest are 1.84-1.92 h. That is up to **+52%** wall-clock for the
cells with the LEAST data, at identical step count. With 80k steps fixed, demo count cannot change
per-step compute, so this is dataloader behaviour: a 10-episode dataset offers very few distinct
sampleable windows, so the workers cycle a tiny file set and get almost no benefit from readahead,
while a 400-episode dataset streams through many files. Practical consequence for Tasks 2-3: do not
schedule assuming the small-N cells are cheap -- they are the expensive ones.

**Emerging science (L0 complete except N=10).** SR at 080000, 100 episodes, frozen protocol:
`N25=0.97  N50=1.00  N100=1.00  N200=1.00  N400=0.99`. L0 is **saturated from N=25 upward** -- the
0.97 and 0.99 are within Wilson noise of 1.00 (97/100 -> [0.915,0.990]). This reproduces the
shared-encoder finding on the new architecture, and it means **all of L0's discriminating
information sits at N=10**, which is still syncing. A ceiling this hard at the easiest level is
itself a result: it says the L0 task is essentially solved by 25 demos, so the interesting
comparisons are between levels, not within L0.

**Registry defect found and fixed (pre-existing).** Aggregating `gpu_h` threw
`could not convert string to float: 'G4 pipeline validation only ...'`. The `g4_smoke_L0_n25` row had
been written with `sr_best` OMITTED rather than empty, so every field from there on sat one column
left: `gpu_h` held a sentence, `eval_n` held `0.5`, `eval_set` held `20`, and `sr_80k` held the
non-numeric string `0.80(5k-step smoke; 20 eps reduced protocol)`. Realigned, with the parenthetical
moved into `notes`. The row is a smoke test and never enters a curve, but the registry is the
traceability source for every number in the report, so an unparseable row is a defect regardless.
Registry now validates clean: 31 rows, **0 non-numeric values in any numeric field**.

**Also fixed:** `update_registry_from_evals.py` only wrote `gpu_h` when the field was empty, so the
five cells whose rows were created while they were still RUNNING kept their *partial* elapsed. Since
sacct is authoritative and Elapsed only grows, it now takes the sacct value whenever it exceeds the
recorded one.

Spend to date: **55.6 GPU-h** across all 31 rows (51.3 matrix + 4.3 bring-up, calibration and the
superseded shared-encoder run) against the 2,200 ceiling -- 2.5%.

## 2026-08-20 (02:45) -- First real demos-vs-generality surface (13/24 cells evaluated)

SR at 080000, 100 frozen-protocol episodes, DDIM 10 steps, per-camera encoders (D26):

| level | N10 | N25 | N50 | N100 | N200 | N400 |
|---|---|---|---|---|---|---|
| L0 (all fixed) | - | 0.97 | 1.00 | 1.00 | 1.00 | 0.99 |
| L1 (+ mug pose) | - | 0.67 | 0.83 | 0.86 | 0.98 | 0.99 |
| L2 (+ goal pose) | 0.31 | 0.60 | 0.75 | - | - | - |
| L3 (+ object variation) | - | - | - | - | - | - |

**This is the study's central measurement and it is working.** Three things stand out already:

1. **The cost of generality is directly readable.** To reach ~0.98 success, L0 needs ~25-50 demos
   while L1 needs ~200 -- roughly a **4-8x data cost for one added axis of randomisation** (mug XY in
   30x40 cm, yaw +-90 deg). That ratio, computed properly as N*(s) with Wilson intervals, is exactly
   the headline number the study was designed to produce.
2. **The curves are monotone in data, which validates the nested-subset design (D4).** L1 goes
   0.67 -> 0.83 -> 0.86 -> 0.98 -> 0.99 and L2 goes 0.31 -> 0.60 -> 0.75, with no reversals. Because
   dataset N is the first N episodes of one seed-0 shuffle, a bigger N is a strict superset, so these
   curves cannot be explained by resampling luck. The only wobble is L0's 1.00 -> 0.99 at N=400,
   which is a single episode out of 100 and well inside Wilson noise.
3. **Level separation is large and clean.** At N=25 the three levels read 0.97 / 0.67 / 0.60, and at
   N=50 they read 1.00 / 0.83 / 0.75. The levels are not collapsing into each other, which was a real
   risk -- if L1 and L2 had landed on top of each other the ladder would have been badly calibrated
   and the whole 4-level design would have needed rework. It did not.

Also worth noting: L0 saturating at 1.00 by N=50 means L0 contributes almost nothing to the
data-cost curve above N=25, and its only informative cell is N=10 (still queued). That is a design
lesson for Tasks 2-3: the easiest level should start its grid lower, or it spends 5 of 6 cells
measuring a ceiling.

Eval throughput is better than the 14 min/checkpoint measured earlier: 7 cells in the last hour,
~8.6 min each, presumably because the earlier figure included Isaac's first-launch shader cache
work. 13/24 done, **zero eval failures** since the harness fixes.

## 2026-08-20 (03:45) -- L0/L1/L2 evaluated (18/24); all six L3 cells failed on an eval-path gap

**L0, L1, L2 are complete: 18/24 cells evaluated, zero failures.** Then all six L3 cells failed
identically:

```
gymnasium.error.NameNotFound: Environment `Cog-CupPlace-L3-IK-Rel-Visuomotor` doesn't exist.
Did you mean: `Cog-CupPlace-L2-IK-Rel-Visuomotor`?
```

**Not a typo -- a real gap between the design and the eval harness.** L3's generality axis IS the
object, so `levels.py` deliberately expands L3 into **10 sub-levels** `L3v00..L3v09`
(2 cylinder sizes x 5 colours), and registers an env per sub-level. There is intentionally no
single `Cog-CupPlace-L3-*` env. But `run_local_eval.sh` builds its task id as
`f"Cog-CupPlace-{LEVEL}-IK-Rel-Visuomotor-v0"`, which is correct for L0-L2 and cannot work for L3.

Why this stayed invisible until now: **training never touches the env** -- it only reads the
LeRobot dataset -- so all six L3 cells trained perfectly and produced valid checkpoints. The gap
existed only on the eval path, and L3 is the last level evaluated. Worth remembering as a class of
bug: a level whose *data* pipeline is fine can still be unevaluable, and the training wave will
report 24/24 success either way.

**The protocol was already decided; only the implementation was missing.** `protocol.json` and
`configs/eval_sets/L3.json` both specify D18: **variant v is evaluated on batch v** (the diagonal),
pooling to 10 x 20 = **200 episodes with 200 distinct poses**. The eval set explicitly warns that
running batch 0 on all ten variants would also total 200 episodes but yield only **20 distinct
poses**, because the variants share the pose RNG stream. So the diagonal is load-bearing, not a
stylistic choice. Note this makes L3's standard eval 200 episodes where L0-L2 use 100 -- frozen
design (rule 8), and it gives L3 a tighter CI than the other levels by construction.

**Implemented `scripts/ops/run_local_eval_l3.py`**: for each variant it writes a one-batch protocol
with `base_seed = 5000 + v` (which is exactly batch v), runs `cog.eval.rollout_eval` in its own
Isaac process, and pools the ten partials into a single result JSON using the same schema as L0-L2
plus a `per_variant` breakdown -- so `update_registry_from_evals.py` needs no special-casing. One
process per variant because creating a second gym env inside a live Kit instance is unreliable, and
because a single crashed variant then cannot poison the other nine.

Two things it does deliberately:
- **Refuses to write a pooled result if any variant failed**, rather than averaging over a partial
  diagonal. That would silently understate coverage while looking like a valid number. This guard
  fired on the first attempt and is the reason no bad L3 number was ever written.
- Waits for >=14 GB free VRAM before each variant, same as the L0-L2 path (rule 2).

**First attempt failed on a missing env var, not on logic**: every variant died with
`Do you accept the EULA? (Yes/No): Unable to bootstrap inner kit kernel: EOF when reading a line`.
`run_local_eval.sh` exports `OMNI_KIT_ACCEPT_EULA=YES`, but a python `subprocess` does not inherit
a shell export that was never in the environment -- and the resulting error names neither the EULA
setting nor the real cause in any obvious way. Now set explicitly in the subprocess env alongside
`HF_HUB_OFFLINE=1`.

## 2026-08-20 (04:15) -- The surface, 19/24 cells: monotone in BOTH directions

| level | N10 | N25 | N50 | N100 | N200 | N400 |
|---|---|---|---|---|---|---|
| L0 | 0.85 | 0.97 | 1.00 | 1.00 | 1.00 | 0.99 |
| L1 | 0.42 | 0.67 | 0.83 | 0.86 | 0.98 | 0.99 |
| L2 | 0.31 | 0.60 | 0.75 | 0.82 | 0.98 | 1.00 |
| L3 | - | **0.15** | - | - | - | - |

(L0-L2: 100 episodes/cell. L3: 200, per the D18 diagonal.)

**Monotone in N within every level, and monotone in level at every N.** No reversals anywhere except
L0's 1.00 -> 0.99 at N=400, a single episode. For a one-seed study this is about the best structural
outcome available: the nested-subset design (D4) means larger N is a strict superset, so within-level
monotonicity is evidence the curves are data effects rather than seed noise, and the strict level
ordering at every single N means the generality ladder is correctly ordered by difficulty.

**Reading off the data cost at s=80% (informally -- the formal logistic fit and Wilson-bounded
N*(s) come from `cog.analysis.curves`):** L0 crosses 0.80 at or below N=10, L1 at ~N=50, L2 at
~N=100. That is roughly a **5x cost for L1 and 10x for L2 relative to L0**, to reach the same
success rate. This is the study's headline quantity and it is now measured rather than assumed.

**L1 and L2 converge at high N -- an interesting result in itself.** They are consistently ordered
but the gap narrows: 0.42/0.31 at N=10, 0.67/0.60 at N=25, 0.83/0.75 at N=50, 0.86/0.82 at N=100,
then **0.98/0.98 at N=200 and 0.99/1.00 at N=400**. So adding goal randomisation on top of object-pose
randomisation (L1 -> L2) costs real data in the low-N regime but essentially nothing once there are
~200 demos. The two axes are not additive in cost, which is a more interesting finding than a simple
"more generality costs more" story.

**L3 is a different regime.** 0.15 at N=25 where L2 is 0.60. Per-variant SRs are 0.05-0.25 with no
outlier, so this is a uniform difficulty increase across the object set rather than one broken
variant -- worth checking explicitly, because a single mis-scaled mesh would have produced the same
pooled number with a very different meaning. If L3 stays this low it may not reach 80% within
N=400, in which case N*(0.8) must be reported honestly as "> 400" per the plan.

**L0's N=10 cell is the informative one, as predicted:** 0.85, comfortably below ceiling, so L0 does
contribute one usable point to the data-cost curve after all.

Remaining: five L3 cells (n10, n50, n100, n200, n400) running at ~24 min each, ~2 h.

## 2026-08-20 (06:15) -- T1 COMPLETE: all 24 cells trained and evaluated; data-cost curves computed

**Full surface** (SR at 80k; L0-L2 100 episodes/cell, L3 200 via the D18 diagonal):

| level | N10 | N25 | N50 | N100 | N200 | N400 |
|---|---|---|---|---|---|---|
| L0 | 0.85 | 0.97 | 1.00 | 1.00 | 1.00 | 0.99 |
| L1 | 0.42 | 0.67 | 0.83 | 0.86 | 0.98 | 0.99 |
| L2 | 0.31 | 0.60 | 0.75 | 0.82 | 0.98 | 1.00 |
| L3 | 0.03 | 0.15 | 0.48 | 0.47 | 0.53 | 0.45 |

**N*(s) from `cog.analysis.curves`** (logistic fit in log N, preferred because it pools all six
cells; interpolated crossings in brackets):

| level | N*(50%) | N*(80%) | N*(90%) | interpolated N*(90%) |
|---|---|---|---|---|
| L0 | 0 | 2 | 4 | 16 |
| L1 | 15 | 42 | 76 | 133 |
| L2 | 22 | 45 | 69 | 150 |
| L3 | 193 | **>400** | **>400** | >400 |

**Cost ratio vs L0 at 90%** (interpolated crossings): **L1 8.31x, L2 9.38x, L3 not reached**.

### Three findings, in order of how much they change the story

**1. L3 does not have a data cost -- it has a ceiling.** L3 is flat from N=50 to N=400:
0.48, 0.47, 0.53, 0.445, with every Wilson interval overlapping. Crucially this is NOT an artefact of
spreading demos over 10 objects. The converter interleaves variants round-robin so every nested
prefix is variant-balanced, and I verified this exactly from `conversion_manifest.json`: N=10 gives
1 demo per object, N=50 gives 5, N=400 gives **40**. So the plateau spans an **8x increase in
per-object data with no improvement whatsoever**. Whatever limits L3 is not the number of
demonstrations.

  This matters for the plan's "extend to N=800 for unsaturated levels" rule, which implicitly
  assumed unsaturated == still rising. L3 is neither at ceiling nor rising -- it is plateaued at
  ~0.47. **On this evidence an N=800 arm for L3 is not indicated**: 8x more per-object data bought
  nothing, so 2x more is very unlikely to. The informative follow-ups would instead disentangle
  object-count from per-object-count (e.g. 40 demos each of 2 objects vs 8 each of 10), or probe
  capacity (larger encoder, more inference steps). That is a scope decision for the user, not
  something to just do.

**2. The two randomisation axes are not additive: L1 ~= L2.** The logistic fit puts them at
N*(80%) 42 vs 45 and N*(90%) 76 vs 69 -- L2 slightly *below* L1 at 90%, i.e. the ordering reverses
within noise. Their raw curves converge too (0.98/0.98 at N=200, 0.99/1.00 at N=400). So adding goal
randomisation on top of object-pose randomisation costs real data at low N but essentially nothing
asymptotically. A naive "each axis multiplies the cost" model is wrong here.

**3. The grid does not resolve L0, which breaks the 50%/80% cost ratios.** L0 reaches 0.85 at our
smallest N=10, so its interpolated N*(50%) and N*(80%) are both "<=10" -- not numeric, so the ratio
column reports n/a for those targets and only the 90% ratio survives. The logistic fit's L0 values
(0, 2, 4) are extrapolations below the grid and should not be quoted as measurements. Design lesson,
now quantified: **the easiest level needed N=2 and N=5 points**. For Tasks 2-3, either start the grid
lower on L0 or accept that L0 serves only as a qualitative reference rather than a ratio denominator.

### Caveats to carry into the report
- **One seed (seed 0) per cell**, per the user's directive. No seed variance is measured, so
  cell-to-cell differences of a few points are not separable from seed noise. The nested-subset
  design is what makes the *within-level* curves trustworthy despite this.
- L3's SR rests on 200 episodes vs 100 for L0-L2, so its CIs are tighter by construction -- fine for
  within-L3 comparisons, but cross-level statements should use the equal-coverage 200-episode sets
  the protocol defines for headline reruns.
- `sr_40k`/`sr_60k` are empty by design (D24, last-checkpoint-only), so "best-of-3" equals
  "last" in the analysis output; that column is not evidence about checkpoint selection.

Spend: **55.6 GPU-h** (2.5% of the 2,200 ceiling). Local eval added 0 grant hours.

## 2026-08-20 (07:00) -- Tasks 2 and 3 LAUNCHED (48 cells); readiness audit caught three bugs first

User approved the full pipeline for T2 (drawer_stow) and T3 (push_target). Before submitting
anything I audited the pipeline against what T1 had already taught us, and found **three defects,
each of which would have produced plausible-looking but wrong results**.

**1. `--max_steps` would have truncated every T2 episode.** `rollout_eval` defaults to
`--max_steps 600`. That is not a neutral default -- it happens to match cup_place *exactly*
(`episode_length_s = 30.0` at 20 Hz = 600 steps), which is why T1 was fine. But drawer_stow sets
`episode_length_s = 60.0` (1200 steps) and its generated demos run **675-743 steps**; push_target
sets 40.0 s (800 steps) with demos at 307-399. So every single T2 episode would have been cut off
before the task could complete, and T2 would have reported a near-zero success rate across the board.
That is the worst kind of bug available here: it looks exactly like a scientific finding
("drawer_stow is much harder than cup_place"). Now derived per task from the env cfgs: T1 600,
T2 1200, T3 800.

**2. Result filenames had no task component, so T2 would have silently inherited T1's numbers.**
Names were `eval_<LEVEL>_n<N>_<step>.json`. T2's L0/N=25 result would therefore be written to
`eval_L0_n25_080000.json` -- the exact path T1 already occupies -- and because `run_local_eval.sh`
skips any cell whose output file already exists, T2's cells would have been *skipped* and T1's
success rates read back as T2's. No error, no warning, and the surface would have looked plausible.
Names are now `eval_<TASK>_<LEVEL>_n<N>_<step>.json`; the 27 existing T1 files were migrated;
`curves.py` parses the tag (optional, defaulting to T1 so old names still work) and takes a `--task`
filter, because tasks must never be pooled -- each has its own L0 baseline, so one shared table would
compute cost ratios across unrelated tasks.

**3. The editable install was shadowing the worktree.** `cog` is installed editable pointing at the
MAIN checkout, and this repo uses a `src/` layout, so `cd $COG_REPO` does NOT put the worktree's code
on `sys.path`. Every eval so far read *this* tree's checkpoints while executing the *main* tree's
`cog` package. I verified by `diff -rq` that the two `src/cog` trees were identical apart from
`__pycache__`, so **T1's 24 results are unaffected** -- but this was luck, not design, and it would
have silently ignored any src fix made here. Both eval scripts now set `PYTHONPATH` explicitly.

Also generalised `run_local_eval_l3.py` to T1/T2/T3 (verified from the eval sets that all three
expand L3 into exactly 10 variants) and the registry updater to `t1_/t2_/t3_` run ids.

**Regression-checked before launching**, since renaming 27 result files could have quietly corrupted
a finished result: `curves.py` reproduces T1 identically (cost ratios 8.31x and 9.38x at 90%, same
N* everywhere) and the registry updater finds all 24 results and changes **0** fields.

**Launched:** 48 cells, jobs 53195xxx, all **RUNNING** immediately. Datasets verified first: all
eight T2/T3 levels present on `$FAST` with 400 episodes each. Expect ~2-3 h per cell and ~105 GPU-h,
which would take the study to ~160 GPU-h total (~7% of the 2,200 ceiling).

## 2026-08-20 -- "why does L3 not reach high SR?" turns out to be a seeding bug in datagen (D27)

The user asked whether L3's low success rate could come from the *generation* pipeline: Mimic failing
on configurations that eval nonetheless tests, so the policy is trained on an easier distribution
than it is scored on. Chasing that produced three separable findings, only one of which is the
originally suspected mechanism.

**1. For T1 the suspected mechanism is ruled out, quantitatively.** New tool `cog.analysis.gen_bias`
compares the retained demos against the *rejected* attempts, which `generate_level` happens to keep
in a parallel `<level>_failed.hdf5`. T1 generation SR is flat across the ladder -- 86.4 / 85.8 / 85.1
/ 87.9% for L0/L1/L2/L3 -- and retained vs rejected initial poses are statistically
indistinguishable at every level (KS p 0.18-0.99 on x, y, yaw). So the filter is an unbiased ~13%
thinning, and it applies equally at L0 (SR 1.00) and L3 (SR 0.45); it cannot explain a gap that
appears only at L3. The bound that settles it: a selection filter can only remove what it rejects,
so at 88% gen SR the maximum attributable deficit is ~12 points against an observed ~55.

**2. The actual cause: L3 has ~9x redundant initial poses.** See D27. All ten variant generation runs
were seeded identically, so the nominal 400-demo L3 datasets hold **43 / 45 / 48 unique initial
poses** (T1/T2/T3) instead of 400, while L1/L2 hold 400. v00-v04 are byte-identical in pose and
v05-v09 likewise. L3's demo axis therefore measures "5 -> 40 unique poses" where L0-L2 measure "10 ->
400", and the plateau at ~0.45-0.53 is a pose-coverage ceiling. This retracts the earlier
"L3 has a ceiling, not a data cost" reading, and with it the argument that the N=800 arm is
unwarranted -- that conclusion rested on the artifact.

Independent corroboration from the loss curves (read from the offline `.wandb` datastores): L2@400
plateaus at 0.0724 by step 40k, whereas L3@400 descends monotonically to 0.0187 and is still falling
8.8% per 20k at step 80k, with an identical annealed LR schedule. A model fitting 400 distinct
scenes hits an aleatoric floor; one memorising ~40 trajectories does not. Normalization stats and
`train_config.json` are identical across levels, so neither scaling nor config explains it.

**3. A separate, real confound: the L3 colour set aliases with the goal marker.** Cross-evaluating
the **L2**-trained policy (which only ever saw `cyl_m_red`) on the L3 diagonal gives, per variant:
red 0.95/0.90, blue 0.65, purple 0.80 -- but **green 0.10/0.10 and yellow 0.15**. The goal marker is
green `(0.10,0.70,0.10)` and yellow carries a high green channel, so two of the five L3 colours are
confusable with the target the policy must place onto. That is a property of the task design, not of
the data volume, and it will survive regeneration; it needs to be reported (and probably the marker
recoloured to something outside the object palette).

Notable: the L2-trained single-object policy pools to roughly the same success rate on the
ten-object benchmark as the L3-trained policy does (~0.5 vs 0.45). A policy that saw one object
matching one that saw ten is itself evidence that the L3 training set, not the task, is the
limitation.

**Also corrected here.** (a) `gen_bias` initially reported T1_L3 as significantly skewed at p=0.000;
that was my own error -- the KS test ran over 400 rows that are ~9x duplicates, inflating n. It now
deduplicates poses first, and the verdict becomes "same" at p=0.70-0.99. (b) The same tool flagged
T3's yaw as the most significant skew in its table when yaw is a constant 0 in both populations: the
asymptotic p-value formula returns ~0 rather than 1 at D=0. Guarded. (c) `docs/timings.md` and D26(b)
still carried the "~16% throughput penalty" for separate encoders, read at step 7,200 during
warm-up. Settled over all 24 T1 cells: median **2.01 h** vs the 2.00 h shared-encoder baseline, mean
2.14 h (+6.8%, carried by a few slow cells). Claim withdrawn, planning rule now 2.2 h/cell.

**T2 carries a genuine version of the user's hypothesis.** T2's generation SR falls 54.9 -> 44.2 ->
30.6 -> 32.7% across L0-L3, and there the retained-vs-rejected comparison *is* significantly skewed:
at T2_L2, yaw KS D=0.232 (p<0.001, 400 vs 906 unique poses, no redundancy), plus x and y. Up to ~69
points of eval deficit is attributable to that filter. So T2's cost-of-generality numbers will
conflate "the policy needs more data" with "the demonstrator itself degrades with generality", and
the honest normalisation is policy SR against the generator's own SR on the same distribution.
T2_L0 also fails 45% of attempts on a *fixed* scene, so the T2 expert is intrinsically brittle
rather than merely pose-sensitive. T3 is clean (gen SR 88-98.5%, <=11 points attributable).

**Status.** Seed fix landed (`--seed` on the vendored generator, per-variant offsets in all three
wave scripts, effective seed printed for provenance). L3 regeneration + retrain pending: 18 cells,
~40 GPU-h. Existing L3 runs are kept as a pose-diversity ablation at fixed N, with results renamed
`*_poseredundant` so they cannot be confused with the corrected arm. The 48 T2/T3 training cells
launched yesterday are unaffected for L0-L2 (36 cells); their 12 L3 cells become the ablation arm.

## 2026-08-20 (later) -- L3 rerun with fixed poses + recoloured objects (D27/D28), and gen_bias figures

**User asks, this session:** rerun the L3 pipeline with the pose fix; change the OBJECT colours
rather than the goal marker; and produce plots/statistics for the generation-bias analysis across all
tasks, especially T2 where the bias may be real.

**Cross-evaluation completed first** (it was still running when the seeding bug was found), and it
is worth recording because it is what makes the diagnosis airtight:

| run | result |
|---|---|
| L2-trained policy on the **L3** diagonal (200 eps) | **0.54** |
| L3-trained policy on the **L2** eval set (100 eps) | **0.76** |
| L2-trained policy on L2 eval **batches 5-9** (100 eps) | **0.99** |

The first row is the headline: a policy trained on 400 demos of ONE red cylinder beats the
ten-object-trained policy (0.45) on the ten-object benchmark. The second shows the L3 policy is also
degraded on the single default object it did see (0.76 vs L2's 1.00) -- consistent with training on
43 unique poses. The third closes a protocol gap I had flagged: our L0-L2 cells used eval batches
0-4 while the L3 diagonal spans 0-9, so the headline comparison mixed pose sets; batches 5-9 score
0.99 against batches 0-4's 1.00, so the two halves are equally hard and the comparison stands.

**D28, the colour fix.** Per-variant results from that same cross-eval isolate the mechanism: the
L2-trained policy scores red 0.95/0.90, purple 0.80, blue 0.65/0.80, but green 0.10/0.10 and yellow
0.15. Failure tracks the object's GREEN CHANNEL (green G=0.60, yellow G=0.80) and not distance from
the training colour -- blue sits much further from red in RGB than yellow does, yet blue works. The
goal marker is green (0.10,0.70,0.10), so a greenish object is a second marker-coloured blob. Per the
user's instruction the marker keeps its colour and the objects move: green -> orange
(0.90,0.30,0.02), yellow -> magenta (0.90,0.10,0.55), replaced IN PLACE so every L3 variant index is
stable. T3's palette already excluded pure green but still had yellow, whose G=0.80 exceeds the
marker's own 0.70; replaced there too, keeping palette index 4 pinned because that is DEFAULT_PUCK
and T3's L0-L2 datasets depict it. Verified PALETTE_CHECK_PASS: names resolve, all G within limits,
all three defaults byte-identical. L0-L2 are therefore NOT regenerated, and the frozen eval poses are
untouched (colour is a static material, it does not enter reset sampling).

Because red/blue/purple are unchanged, comparing those variants between the old and new L3 arms
isolates the pose fix from the colour fix at zero extra training cost.

**The rerun.** L3 regenerates for all three tasks as level key `L3b`, with per-variant seeds
(T1 1000+v, T2 2000+v, T3 3000+v; clear of the eval seeds 5000-5009). A distinct key is not
cosmetic: `train.sbatch` auto-resumes when checkpoints exist, so reusing `t*_L3_*` would have
resumed the pose-redundant runs, found them at 80k and exited having trained nothing.
`launch_matrix`/`train.sbatch` already derive run id and dataset name from the level string, so L3b
needed no changes there; `curves.py` and the registry updater now accept `L\d[a-z]?` so L3b is its
own cell rather than being folded into L3. Regression-checked: T1 still reproduces 8.31x/9.38x at
90% with every N* unchanged.

**Cancelled the 12 in-flight t{2,3}_L3 cells** and marked them superseded. They train on the
redundant data, and with the new palette in place evaluating them would require restoring the old
colours, so they cannot contribute; that frees ~28 GPU-h. The 36 T2/T3 L0-L2 cells are unaffected by
both bugs and keep running. T1's L3 arm is already fully measured and stays as the ablation.

**gen_bias figures + statistics** (`paper/figures/gen_bias_{T1,T2,T3,summary}.png`,
`experiments/gen_bias.csv`, 12 level rows). Per task: generator SR with attempt counts, unique-pose
count against the 400 reference, the rejected-fraction upper bound, KS D per axis with significance
marks, and for the most-skewed level an ECDF plus a workspace scatter of where generation succeeded
versus failed. Both the CLI table and the figures now read `gen_bias.level_stats`, so the paper's
numbers cannot drift from the terminal's.

What the figures show, by task:
* **T1** -- SR flat 86.4/85.8/85.1/87.9%, no axis skewed once poses are deduplicated, bound ~12-15
  points. Generation is an unbiased thinning; not a confound.
* **T2** -- the real case. SR falls 54.9 -> 44.2 -> 30.6 -> 32.7%; yaw is skewed from L1 onward
  (D=0.167, 0.232, 0.352; p<=0.001) with x and y joining at L2, and the bound reaches 67-69 points.
  The retained demos sit at negative yaw while the rejected ones sit positive, so the policy trains
  on a rotationally lopsided slice of a symmetric eval distribution. **T2's cost-of-generality
  numbers will therefore conflate "the policy needs more data" with "the demonstrator degrades", and
  the honest normalisation is policy SR against generator SR on the same distribution.** Also
  T2_L0 fails 45% of attempts on a *fixed* scene, so that expert is brittle in itself rather than
  merely pose-sensitive.
* **T3** -- SR 88.5-98.5%. y is statistically significant at L1 (p<0.001) but the rejected fraction
  caps it at 5 points, which is exactly why the bound is plotted beside the test: significance
  without magnitude would have read as a problem.

One presentational correction: at L0 there is a single initial pose, so a filter cannot shift the
pose distribution at all and those rejections are demonstrator stochasticity, not selection. Those
bars are drawn hollow and annotated "n/a: 1 pose" so they are not read as a bias bound.

Also hardened: the HDF5 loader raises `FileLockedError` rather than reading a file a generation job
is still writing, and the caller drops the whole level -- a partially generated level must never
enter a figure. And `gen_L3_wave.sh` is now task-parameterised and L3-only, because the T2/T3
full-wave scripts also regenerate L0-L2 and have no clobber guard, so reusing them to redo L3 would
have destroyed four valid datasets.

**Pending:** T1/T3/T2 L3b datagen (~5 h local, zero grant cost), then 18 L3b cells (~40 GPU-h),
then the L3b diagonal evals. Prediction on record (D27 VERIFY b): corrected L3 keeps rising past
N=50 instead of flattening at ~0.47.

### 2026-08-20 addendum -- the regenerated L3b arm needs `--max_episodes 400`

T1's L3b conversion produced **401** episodes and the pre-launch verify caught it, so no GPU-h was
spent on an unbalanced dataset. Cause: generation runs in `num_envs=8` batches, so a variant can
overshoot its 40-success target -- `L3bv01` produced 41. The original L3 arm hit exactly 40 in every
variant only because all ten runs shared one seed (D27) and therefore overshot identically; with
independent seeds the overshoot is per-variant and irregular.

Fix: convert with `--max_episodes 400`. The converter interleaves the variants round-robin *before*
truncating, so the first 400 episodes are exactly 40 per variant and any overshoot lands at the tail.
The 401-episode root was moved aside as `L3b_401ep_unbalanced` rather than deleted.

Worth noting the verify gate is what made this a five-minute correction instead of a silently
lopsided L3 arm -- the same class of check whose absence produced D27 in the first place. It now
asserts 400 episodes AND that every nested prefix (N=10..400) covers all ten variants within one
episode of balance.

`gen_L3_wave.sh` also became resumable while fixing an interrupted T3 wave: a variant holding exactly
40 demos is skipped, anything else aborts and asks for the partial to be moved aside. T3 resumed at
v03 keeping its three finished variants.

**Local GPU is now the bottleneck, not the cluster.** 31 of the 36 T2/T3 L0-L2 cells are at 80k and
their checkpoints sync down fine (GPU-free), but each eval waits for 14 GB free VRAM and the L3b
datagen leaves 13.6 GB, so evaluation queues behind datagen and starts when it ends (~15:30),
finishing ~20:30. The threshold is deliberately left alone: an eval's own footprint is ~7 GB so
concurrency would fit, but that 14 GB margin is what protects the foreign job on this card (rule 2),
and trading it away is the user's call, not a unilateral optimisation.

### 2026-08-20 -- T2/T3 evaluation was impossible: rollout_eval registered only cup_place

The first 34 T2/T3 evals all "failed" in ~11 s each. Not a headroom wait and not a result:
`rollout_eval.py` had `import cog.tasks.cup_place` hardcoded, and the gym ids are registered as an
import SIDE EFFECT of each task package, so no `Cog-DrawerStow-*` or `Cog-PushTarget-*` id ever
existed in the registry. The error reads

    NameNotFound: Environment `Cog-DrawerStow-L0-IK-Rel-Visuomotor` doesn't exist.
                  Did you mean: `Cog-CupPlace-L0-IK-Rel-Visuomotor`?

which looks like a typo in the task name rather than a missing import, and Kit still exits 0 (D6),
so the only signal was the missing artifact.

Now imports the package matching the `--task` prefix, falling back to importing all three for an
unrecognised prefix. Verified: the same cell boots the DrawerStow env and proceeds into rollouts.

**Why the earlier T2/T3 readiness audit missed it.** That audit fixed three real things --
`--max_steps` per task, task-tagged result filenames, PYTHONPATH shadowing -- by *reading* the eval
path, and it regression-tested the T1 numbers. It never actually ran a T2 or T3 eval, because at
that point no T2/T3 cell had finished training. Reading the code found the bugs that were visible in
the code; this one only appears at import time in a process that reports success by absence.
Lesson worth keeping: for a path that has never once executed, the audit is not the check -- the
first real run is, and it should be run on a single cell before a 36-cell driver is turned loose.

No results were affected: the 34 failures produced no artifacts, so nothing wrong was ever written
to results/ or the registry.

### 2026-08-20 -- D27 also corrupted the reported GENERATION SR for L3, and only where it matters

The regenerated L3b waves give per-variant generation SRs that differ from the old L3 figures in a
revealing pattern:

| task | old L3 gen SR (shared seed) | new L3b gen SR (per-variant seeds) |
|---|---|---|
| T1 cup_place | 87.9 % | ~86.9 % (range 78.4-95.2 across variants) |
| T3 push_target | 88.5 % | ~91.5 % (range 88.9-100) |
| T2 drawer_stow | 32.7 % | **~27 %** (v00 26.8, v01 27.0, v03 running at 21.4) |

T1 and T3 barely move; T2 drops by about six points. That is exactly what the bias analysis
predicts. Generation success on T1/T3 is essentially pose-independent (KS D small, no axis skewed),
so estimating gen SR from one pose set repeated ten times still gave the right answer by luck. On T2
generation success is strongly pose-dependent -- the retained-vs-rejected yaw skew is D=0.35,
p<0.001 -- so a single pose set is a badly biased sample of the L3 distribution's difficulty.

So D27's damage was wider than the training data: **the "generation SR per level" number the plan
asks us to report as a finding was also wrong for L3, in all three tasks, and materially wrong for
T2.** The `gen_stats.csv` L3 rows were computed over 43-48 unique poses, not 400. The L3b rows are
the first honest estimate. The L0-L2 rows were always fine (single generation run, 400 unique poses).

Corollary for the write-up: the T2 demonstrator-SR curve is steeper than we reported. It now reads
roughly 54.9 / 44.2 / 30.6 / ~27 % across L0-L3 rather than bottoming out at 32.7, which strengthens
rather than weakens the point that a large share of T2's apparent cost of generality belongs to the
data collector.

Also measured while checking: T2 datagen is running ~37 min per variant rather than the 23 min in
`timings.md`, because it is sharing the 4090 with the eval driver. Both keep progressing; the
timings table's ~40-50 % mutual slowdown under GPU sharing holds. Disk went 297 -> 242 GB free in
~2.5 h, mostly T2's `_failed.hdf5` files (4.9 GB for three variants, ~16 GB projected for ten,
because a rejected T2 attempt still stores ~680 frames of video). Projected floor ~200 GB free --
comfortable, but this is the first leg where the failed-attempt archive is a material disk cost.

### 2026-08-20 -- local eval is a single-slot resource; L3b given priority

All cluster training is now complete: 72 COMPLETED, 12 CANCELLED (the superseded T2/T3 L3 cells).
That is 24 T1 + 36 T2/T3 L0-L2 + 12 T1/T3 L3b; only T2's 6 L3b cells remain, waiting on datagen.

A scheduling fact worth recording, because it was not obvious: **the local 4090 supports exactly one
eval at a time** while L3b datagen holds ~6 GB. An eval needs ~7 GB and the headroom gate wants
14 GB free, so two eval drivers do not double throughput -- they race for one slot. Having started a
second driver for the L3b diagonals, I found it starved for 30 min behind `evalt23`, which begins its
next cell immediately on finishing the last.

Resolved by explicit priority rather than by lowering the gate (that 14 GB margin is what protects
the foreign eval job, rule 2) and without discarding work: a handoff script waits for the in-flight
cell's artifact, stops `evalt23` between cells, lets the 12 L3b diagonals run, then restarts it --
`evalt23` is idempotent and skips cells whose results already exist. Priority goes to L3b because it
is the arm that tests D27/D28 and the study's headline prediction; it should not sit behind ~12 h of
L0-L2 queue.

Remaining local eval, at the measured rates: ~34 T2/T3 L0-L2 cells and 12 L3b diagonals, on the
order of **20-25 h serial**. Unchanged conclusion: eval wall-clock, not GPU-hours, is what this study
is short of.

### 2026-08-20 -- a watcher that gives up must exit non-zero (rule 10 refinement)

A background watcher reported **"completed (exit code 0)"** for a chain that had been superseded
hours earlier and whose marker it therefore never saw. Cause is a shape I used repeatedly today:

    for i in $(seq 1 N); do <check> && exit 0; sleep 60; done
    echo "WATCHER_TIMEOUT"          # <-- sets the exit code to 0

The trailing `echo` succeeds, so giving up looks identical to succeeding. That is worse than having
no watcher, because it actively asserts a job is fine. Rule 10 exists because each monitoring layer
has failed alone; this is a new way for the event layer to fail -- not by staying silent, but by
lying. Correct shape ends `echo "WATCHER_TIMEOUT" >&2; exit 1`.

Second half of the lesson: **kill watchers whose target is replaced.** Restructuring the L3b
pipeline mid-flight left two watchers polling for markers that would never appear; both eventually
fired spurious completions competing for attention with real ones. The obsolete one still running
has been stopped.

Neither watcher's failure cost anything here -- the hourly cron and the tmux sessions are what
actually caught every real event today, which is precisely the argument for keeping all three layers.

### 2026-08-20 -- first three corrected-L3 points: ahead at small N, level at N=50, verdict pending

| N | L3b (400 unique poses) | L3 (43 unique poses) |
|---|---|---|
| 10 | 0.090 [0.058,0.138] | 0.030 [0.014,0.064] |
| 25 | 0.235 [0.182,0.298] | 0.150 [0.107,0.206] |
| 50 | **0.465** [0.397,0.534] | **0.480** [0.412,0.549] |
| 100 | pending | 0.470 |
| 200 | pending | 0.530 |
| 400 | pending | 0.445 |

The corrected arm is clearly ahead at N=10 and N=25 -- which is where the old arm was most starved of
distinct poses (1 and 2-3 unique poses in total, against 10 and 25). **At N=50 the two arms are
identical within noise**, 0.465 vs 0.480, even though L3b has ten times the pose diversity there
(50 unique poses vs 5).

That is worth stating plainly because it puts my own stated mechanism at risk. I claimed the L3
plateau at ~0.47 was a pose-coverage ceiling created by D27. If N=100/200/400 also come in at ~0.47
on the corrected arm, that claim is wrong: the plateau would be a real property of L3 as posed, and
D27 -- while still a genuine data bug that inflated the demo axis ~9x and invalidated the L3
generation-SR figures -- would not be its cause. The three remaining diagonals decide it, and I will
report the outcome either way rather than waiting to be contradicted.

The alternative reading, if the plateau survives: at N=50 the binding constraint is per-object data
(5 demos per object) rather than pose diversity, and something else caps L3 above that. The L2-policy
cross-eval already hints the ceiling is not purely about data -- a policy trained on ONE object
scored 0.54 on the ten-object diagonal, above the ten-object policy's 0.45.

D28 sanity check at N=10 (not a verification -- 20 episodes per variant): orange 0.15/0.25 and
magenta 0.15/0.10 sit at or above red 0.00/0.00, blue 0.05/0.05, purple 0.15/0.00. No sign of the
green/yellow cliff they replaced, but the N>=100 diagonals are what will settle it.

### 2026-08-20 -- D27 VERIFY (b) CONFIRMED: the L3 plateau was the pose-redundancy artifact

| N | L3b (400 unique poses) | L3 (43 unique poses) |
|---|---|---|
| 10 | 0.090 [0.058,0.138] | 0.030 [0.014,0.064] |
| 25 | 0.235 [0.182,0.298] | 0.150 [0.107,0.206] |
| 50 | 0.465 [0.397,0.534] | 0.480 [0.412,0.549] |
| **100** | **0.700 [0.633,0.759]** | **0.470 [0.402,0.539]** |
| 200 | pending | 0.530 [0.461,0.598] |
| 400 | pending | 0.445 [0.378,0.514] |

The corrected arm rises 0.465 -> 0.700 between N=50 and N=100, with non-overlapping Wilson
intervals against the old arm at N=100. The old arm was flat across exactly that interval
(0.480 -> 0.470) and stayed flat to N=400. **The plateau was an artifact of pose redundancy, not a
property of L3.** The prediction recorded before the measurement holds, and the retraction stands:
"L3 has a ceiling, not a data cost" was wrong, and the N=800 question is genuinely open again --
still not to be launched without the user's say-so.

Note the N=50 coincidence (0.465 vs 0.480) was real and is now explained: at 5 demos per object the
binding constraint is per-object data, not pose diversity, so both arms sit at the same place. Pose
diversity only becomes the binding constraint above that, which is exactly where the old arm stopped
improving and the corrected one did not. I flagged at N=50 that my mechanism claim was at risk; the
risk resolved in its favour, but the flag was the right call on the evidence available then.

**D28: consistent, but not isolated.** At N=100 every variant lands in 0.45-0.80 with no low
outlier, and the replacement colours (orange 0.80/0.65, magenta 0.75/0.70) sit at or above red
(0.70/0.45), blue (0.80/0.75) and purple (0.70/0.70). But L3b changed poses AND colours at once, so
this is not a controlled test of the recolouring. Using the colours that did NOT change as the
control: red/blue/purple improved by ~0.15-0.30 (pose fix alone), and the changed colours improved
by ~0.25-0.30 against the green/yellow they replaced -- comparable, i.e. **no large additional colour
effect for a policy trained on all ten colours.** That is consistent with the original evidence,
which came from the L2-policy cross-eval (green 0.10/0.10, yellow 0.15 for a policy that had never
seen them): the aliasing mainly damaged *generalization to unseen* greenish objects, not the
in-distribution case. The recolouring remains justified -- it removes a perceptual confound from the
level's design -- but it should be reported as a design correction, not as the cause of the plateau.

## 2026-08-20 -- T1's L3 arm is complete on corrected data, and L3 HAS a data cost: 11.56x L0

The corrected T1 L3 arm finished. It does not plateau -- it saturates like every other level, just
later:

| N | L3b (400 unique poses) | L3 (43 unique poses, ablation) |
|---|---|---|
| 10 | 0.090 | 0.030 |
| 25 | 0.235 | 0.150 |
| 50 | 0.465 | 0.480 |
| 100 | 0.700 | 0.470 |
| 200 | 0.935 | 0.530 |
| **400** | **0.945** [0.904,0.969] | **0.445** [0.378,0.514] |

**The headline table for Task 1, now that every level has a measurable data cost:**

| level | N*(50%) | N*(80%) | N*(90%) | cost vs L0 @90% | logistic slope b |
|---|---|---|---|---|---|
| L0 | <=10 | <=10 | 16 | 1.00x | +2.31 |
| L1 | 15 | 45 | 133 | 8.31x | +3.11 |
| L2 | 20 | 86 | 150 | 9.38x | +4.44 |
| **L3b** | **57** | **143** | **185** | **11.56x** | **+3.47** |
| L3 (ablation) | 150 | >400 | >400 | not reached | +2.03 |

So the study's central quantity for T1 is: **reaching 90 % success costs 8.3x more demonstrations
under pose randomization (L1), 9.4x under pose + goal randomization (L2), and 11.6x once the object
itself varies (L3)** -- against a fixed-scene baseline that needs 16.

Two things this changes.

1. **L3's cost was previously "not reached" and is now the most interesting number in the table.**
   The old arm's `>400` was an artifact of 43 unique poses; with 400 it lands at 185. The old arm's
   logistic slope (+2.03) was also artifactually shallow -- the corrected slope (+3.47) sits between
   L1 and L2, i.e. **L3 behaves like the other levels, it is simply shifted right.** That is a much
   stronger and cleaner result than "L3 has a ceiling", and it is the opposite conclusion.
2. **The levels are not additive, and the increments are small.** 8.31 -> 9.38 -> 11.56 means adding
   goal randomization to pose randomization costs ~13 % more data, and adding ten-way object
   variation on top costs a further ~23 %. Object identity -- the axis one might expect to be most
   expensive -- is the cheapest increment per unit of apparent scene diversity. Worth stating
   carefully in the paper: it is measured at 90 %, one seed, and on a cylinder whose two sizes differ
   by 4 mm of radius, so "object variation" here is mostly appearance, not geometry.

The N=800 extension question is now genuinely live and genuinely unnecessary for T1: L3b reaches
0.945 at N=400 and is saturating (0.935 -> 0.945 from N=200), so the 90 % crossing is measured, not
extrapolated. Still not to be launched without the user's say-so.

Also regenerated `paper/figures/gen_bias_*.png` and `experiments/gen_bias.csv` (14 level rows) with
the T1/T3 L3b arms included; T2_L3b was correctly skipped as its last variant is still generating.

### 2026-08-20 -- all three L3b datasets regenerated; D27 VERIFY (a) passes everywhere

T2's L3b wave finished: **400 retained demos over 400 unique initial poses** (the old T2_L3 had 45),
947 rejected, pooled generation SR **29.7 %**. So the seed fix is confirmed on every task:

| task | old L3 unique poses | L3b unique poses | old gen SR | L3b gen SR |
|---|---|---|---|---|
| T1 cup_place | 43 | 401 | 87.9 % | ~86.9 % |
| T2 drawer_stow | 45 | 400 | 32.7 % | **29.7 %** |
| T3 push_target | 48 | 400 | 88.5 % | ~91.5 % |

T2's honest L3 generation SR is 29.7 %, not 32.7 %, which tightens the demonstrator curve to
**54.9 / 44.2 / 30.6 / 29.7 %** across L0-L3 -- now monotone, where the artifactual figure had L3
appearing *easier* to generate than L2. That non-monotonicity should have been a clue; it was sitting
in `gen_stats.csv` unexamined.

`gen_stats.csv` and the four `paper/figures/gen_bias_*.png` regenerated (15 level rows), now with all
three corrected arms included. T2_L3b conversion is running (~2 h at ~680 frames/episode), after
which its 6 cells launch.

T3's corrected L3 arm so far: N=10 -> 0.305, N=25 -> 0.510. For comparison the *old* T3 L3 arm was
never evaluated (its cells were cancelled), so T3 has no redundant-pose baseline -- the T1 comparison
is the only within-study evidence for the artifact, which is worth stating plainly in the write-up
rather than implying all three tasks demonstrated it.

### 2026-08-20 22:30-23:40 -- two concurrent evals are NOT delivering the predicted speedup

The user authorised two parallel evals on the strength of my estimate that it would roughly halve the
~30 h eval queue. First measurements say otherwise, and the reason is partly outside our control.

* `t2_L0_n200` reached only **batch 2 of 5 in 62 min** under concurrency. Solo it was 43 min for all
  five batches (~8.6 min/batch); concurrent it is ~31 min/batch, i.e. **~3.6x slower per batch**, far
  worse than the 40-50 % mutual slowdown the datagen-sharing measurement suggested.
* **The foreign eval job became active in the same window.** It was 16 % of SM time earlier today and
  is now 34 %, with memory unchanged at 1.6 GB. So the card is in three-way contention: foreign 34 %,
  our two evals 35 % and 21 % -- our aggregate share is ~56 %, against ~66 % when only one of ours
  runs alongside the foreign job. On that arithmetic two slots make our *aggregate* throughput
  slightly WORSE, not better, because two Isaac processes thrash more than one and the foreign job
  takes a larger slice under heavier contention.

Two things worth separating. **Safety is unchanged**: the threshold protects against OOM and the
foreign job's memory is still 1.6 GB, so the ~8.6 GB free with two evals running remains ample --
lowering 14000 -> 10000 did not put the foreign job at risk (rule 2 respected). **The benefit is
what failed to materialise.** I am not reverting unilaterally on one sample against an explicit
authorisation; the next completed cell gives a clean duration to compare against the 43 min solo
baseline, and I will report the aggregate either way.

If it confirms, the honest conclusion is that this GPU is SM-bound, not memory-bound, for Isaac
rollouts -- so the single-slot arrangement was already optimal and the only real lever on eval
wall-clock is moving evaluation off this card entirely (the CINECA Vulkan question).

### 2026-08-20 23:20-23:55 -- CORRECTION: concurrency DOES work (1.48x); and boost_usr_prod has no nodes

**Correction to the previous entry.** I reported that two concurrent evals were failing to deliver,
based on `t2_L0_n200` sitting at batch 2/5 after ~62 min. That was wrong -- I compared a mid-run
batch count against a wall-clock estimate I had misjudged. The completion timestamps settle it:

| cell | window | duration | solo baseline | penalty |
|---|---|---|---|---|
| `t2_L0_n200` (flat, 100 eps) | 22:33:09 -> 23:19:30 | **46 min** | 43 min | +7 % |
| `t3_L3b_n200` (diagonal, 200 eps) | 22:43:48 -> 23:38:32 | **55 min** | ~53 min | +4 % |

Both ran concurrently the whole time. In 65 minutes the pair completed 96 minutes of serial work, so
**concurrency gives ~1.48x aggregate throughput at a 4-7 % per-cell penalty** -- close to the ideal
2x minus the third-party contention. The foreign job's rise from 16 % to 34 % of SM time is real but
did not cost us what I feared. Lesson: judge eval throughput from artifact timestamps, never from
mid-run batch counters, which are not evenly spaced across batches.

Remaining queue: 38 cells, ~27.5 h serial -> **~19 h at the measured 1.48x**, so completion around
Friday evening.

**Separate and cluster-side: `boost_usr_prod` currently has ZERO nodes.** T2's six L3b cells were
rejected at 23:17 with

    sbatch: error: Batch job submission failed: More processors requested than permitted

That message is misleading. It is not our request: a 1-CPU submission is rejected too, and so is a T3
submission byte-identical to one that succeeded at 12:24 today. `sinfo -p boost_usr_prod` reports
`NODES(A/I/O/T) = 0/0/0/0`, STATE `n/a` -- the partition has no nodes assigned, so any request
exceeds the zero permitted. Account, QOS and budget are all fine (`euhpc_b38_106` active to
2026-10-29, `normal` QOS has no CPU cap, `saldo` shows the account healthy; its `totConsumed=0` is
just saldo's nightly lag, not a billing failure).

Impact is contained: **all training is complete except T2's L3b arm**, and the eval queue -- the
actual bottleneck -- runs locally and is unaffected. A retry watcher polls every 10 min and submits
the six cells as soon as nodes return; `launch_matrix` skips run_ids already in the registry, so it
cannot double-submit. If nodes return within ~19 h, the outage costs the study nothing at all.

Worth adding to the cluster playbook: **"More processors requested than permitted" on Leonardo can
mean the partition is empty, not that the job asked for too much.** Check `sinfo -p <partition>` node
counts before touching the job's resource request.

### 2026-08-21 00:00-00:55 -- partition recovered; T3's L3b arm complete, with a non-monotone tail

**The `boost_usr_prod` outage lasted ~45 min.** Nodes returned and the retry watcher submitted T2's
six L3b cells at 00:04:35, so the outage cost the study nothing -- training resumed well inside the
~19 h the local eval queue still needs.

**T3's corrected L3 arm is complete**, and shows the same rising shape as T1's up to N=200, then dips:

| N | T3 L3b | Wilson 95 % |
|---|---|---|
| 10 | 0.305 | [0.245,0.372] |
| 25 | 0.510 | [0.441,0.578] |
| 50 | 0.605 | [0.536,0.670] |
| 100 | 0.780 | [0.718,0.832] |
| 200 | **0.920** | [0.874,0.950] |
| 400 | **0.840** | [0.783,0.884] |

The N=400 point sits *below* N=200 with intervals that overlap only in [0.874, 0.884] -- marginal,
but it should not be smoothed over. Three readings, in order of prior plausibility:

1. **Single-seed training variance.** The protocol is one seed per cell (user directive), so each
   point is one draw from the training distribution and an 8-point swing between adjacent N is well
   within what a different seed can produce. This is the limitation the plan already lists, and it is
   the first place in the study where it visibly bites.
2. A genuine effect (more data changing which modes the diffusion policy commits to). Possible but
   unsupported by anything else we have measured.
3. An eval or checkpoint artifact. Ruled out as far as it can be cheaply: the diagonal is the frozen
   protocol, all ten variants completed, and the pooled count is a clean 168/200.

For the write-up the honest treatment is to report both points with intervals, note that the logistic
fit (which pools all six cells) is the primary estimator precisely because it is robust to one noisy
cell, and cite this as the concrete cost of the one-seed design rather than burying it. T1's
equivalent tail was monotone (0.935 -> 0.945), so this is not systematic.

**Gap closed:** `eval_l3b.sh` only ever covered T1 and T3 -- T2's dataset did not exist when it was
written -- and that session has now exited. T2's six L3b diagonals therefore had no driver at all,
which would have been invisible because nothing errors when a queue is simply absent. A dedicated
driver is now running; at ~113 min per T2 diagonal it is ~11 h of work and is the single largest
remaining block.

### 2026-08-21 01:55 -- latent bash bug silently killed a driver's polling loop

`evalt2l3b` exited 7 seconds after launch having logged only its final line. Root cause, reproduced
in isolation (`bash 5.2.21`):

    set -u
    declare -A FAILED          # empty associative array
    echo "${#FAILED[@]}"       # -> "FAILED: unbound variable"

Under `set -u`, `${#arr[@]}` on an **empty** associative array is an unbound-variable error. It does
not merely return 0, and it does not abort the script -- it **aborts the enclosing loop** and
execution resumes after it. So the driver ran one pass, hit the summary line
`say "round ... ${#FAILED[@]} failed"`, silently abandoned its polling loop, printed its DONE line
and exited looking like a clean completion.

Nasty because the failure mode is invisible three ways over: it needs the array to be empty (i.e.
nothing has failed yet -- the *healthy* case), it produces a success-looking final log line, and the
consequence is not an error but an absence, namely a queue that stops polling for checkpoints that
have not finished training yet. Same family as the watcher-exit-code bug recorded yesterday: **the
monitoring layer reporting success while doing nothing.**

Affects all three eval drivers written today. Actual damage: only `evalt2l3b`, which had 6 cells to
wait for and stopped waiting. `evall3b` hit it too but had already finished all 12 of its cells on the
one pass it completed, so nothing was lost -- which is exactly why it went unnoticed. `evalt23`
carries the bug but is mid-pass over 30+ cells at ~46 min each, so it will not reach the summary line
for many hours; it is left running rather than restarted, since a restart would discard the in-flight
cell, and a fixed copy is staged to swap in when it stops.

Fix: replace `${#FAILED[@]}` with a plain integer counter incremented alongside the array. Verified
the fixed driver now logs `round 1: 0/6 T2 L3b cells evaluated, 0 failed` and keeps polling.

### 2026-08-21 04:40 -- T2's L1 plateau is NOT a coverage problem; but the fixed step budget is not equal across tasks

**T2 L1 is complete and plateaus: 0.12 / 0.22 / 0.19 / 0.30 / 0.25 / 0.23** across N=10-400. Flat from
N=25, so 8x more data buys nothing, and it never reaches 50 % -- meaning every T2 L1 cost ratio would
be "not reached". Given the shape is exactly what the D27 artifact produced on T1, it needed testing
rather than reporting.

**Not D27:** T2_L1 holds 400 unique poses, redundancy 1.0.

**Not generation coverage either.** `gen_bias` flags T2_L1's yaw as skewed, and the direction is
informative: retained attempts have yaw sd 0.392 against the rejected attempts' 0.509, i.e.
generation succeeded preferentially at SMALL |yaw|. That predicts the policy should fail at large
|yaw|. New tool `cog.analysis.success_vs_pose` joins the frozen eval-set snapshot (which commits
per-episode initial poses) to the per-episode outcomes and tests it directly:

| \|yaw\| bin (equal-count over eval) | n | SR |
|---|---|---|
| [0.001, 0.166) | 25 | 0.280 |
| [0.166, 0.365) | 25 | 0.120 |
| [0.365, 0.582) | 25 | 0.280 |
| [0.582, 0.781) | 25 | 0.240 |

Flat. And the demos cover the tested range (demo \|yaw\| max 0.784 vs eval max 0.781; demo mean 0.339
vs eval 0.386). **The prediction is refuted: T2 L1's failures are spread uniformly across yaw, so the
generation skew is real but immaterial to the policy.** Worth recording as a negative result -- it
closes off the explanation the whole gen_bias apparatus was built to test, for this level.

**What it does look like: the fixed 80k-step budget is not equal across tasks.** At the same 80k steps
and 5.12M samples (L1, N=400):

| task | epochs over its own data | mean episode length | final train loss |
|---|---|---|---|
| T1 cup_place | **67.9** | ~188 frames | 0.0748 |
| T3 push_target | **40.1** | ~310 | 0.0655 |
| T2 drawer_stow | **18.4** | ~680 | 0.0412 |

T2 gets **3.7x fewer passes over its demonstrations than T1** at identical step count, purely because
its episodes are 3.6x longer. The protocol (fixed 80k steps, one seed, identical hyperparameters --
user directive, frozen after G5a) equalises *gradient steps*, not *epochs*, and the tasks differ ~3.7x
in frames per demo.

Scope of the damage, stated precisely:
* **Within-task cost ratios are unaffected.** Episode length is essentially constant across a task's
  levels, so L0/L1/L2/L3b for a given task all get the same epochs at the same N. T1's headline table
  (8.31x / 9.38x / 11.56x) stands, and T2's and T3's will be internally valid too.
* **Cross-task SR comparisons are confounded.** "T2 is harder than T1" currently conflates task
  difficulty with 3.7x less training per demo. Any statement comparing absolute SR between tasks must
  either report epochs alongside or be dropped.

Two possible responses, both the user's call: report it as a stated limitation with the epoch table,
or retrain T2 at matched epochs (~300k steps, ~3.7x, roughly 100 GPU-h for its 24 cells -- affordable
against the ~190 GPU-h spent and the 2,200 ceiling) which would break the frozen-protocol rule that
every cell trains for exactly 80k steps. I am not doing the latter unasked.

Note also that final train loss runs *opposite* to epochs (T2 lowest at 0.041 despite fewest passes),
which is the third time this session that loss magnitude has proved incomparable across datasets --
it tracks the conditional entropy of the action given the observation, not fit quality.

## 2026-08-21 -- TASK 2 COMPLETE: generality on a long-horizon task costs more than 400 demos

T2 (drawer_stow) finished all 24 cells. Its table is qualitatively unlike T1's:

| level | N*(50%) | N*(80%) | N*(90%) | SR at N=400 | fit b |
|---|---|---|---|---|---|
| L0 | <=10 | 18 | 23 | 0.96 | +1.20 |
| L1 | >400 | >400 | >400 | 0.23 | +0.47 |
| L2 | >400 | >400 | >400 | 0.29 | +0.86 |
| L3b | >400 | >400 | >400 | 0.14 | +0.82 |

**No T2 cost ratio is computable**: only L0 crosses any target. This is a cliff, not a cost curve --
0.96 at a fixed scene, then 0.14-0.29 the moment anything is randomised, with 40x more data (N=10 ->
400) buying almost nothing.

**Undertraining is ruled out.** The obvious suspicion was the step-budget confound recorded earlier
(T2 gets 18.4 epochs at 80k steps against T1's 67.9, because its episodes are 3.6x longer). Two
pieces of evidence say that is not the explanation:

1. **L0 reaches 0.96 on the same 18.4 epochs.** The budget is sufficient for a fixed-pose long-horizon
   task; it fails only once poses randomise.
2. **All three T2 cells are converged at 80k**: final-20k loss drift is +0.07 % (L0), -0.13 % (L1),
   +0.01 % (L2). Compare T1's pose-redundant L3, which was still descending 8.8 % per 20k steps -- the
   signature this same diagnostic caught for D27. T2's optimisation has finished; the policies simply
   do not succeed.

**What the evidence does point to: compounding error over a long horizon.** T2's training loss is the
*lowest* in the study (0.042 at L1 against T1's 0.075), so the policies fit their demonstrations well
and still fail at rollout. That combination -- low imitation loss, low closed-loop success -- is
distribution shift, not underfitting. T2 is a four-phase task (grasp handle, pull open, grasp object,
stow) over ~680 steps; an early deviation such as an incompletely opened drawer makes everything
after it unrecoverable, and there is no recovery behaviour in the demonstrations because Mimic only
keeps successes. L0 escapes this because a fixed scene lets the policy replay essentially one
trajectory.

**The finding, stated carefully:** on a long-horizon multi-phase task, the demonstration cost of even
modest generality (randomising one object's pose) exceeds 400 demos -- so the cost of generality is
not merely *larger* for harder tasks, it can be prohibitive inside a fixed budget. That is a stronger
claim than T1 alone supports and it is the main reason having three tasks was worth the cost.

**Caveats to carry into the write-up.** (a) One seed. (b) The 80k-step budget gives T2 3.7x fewer
epochs than T1, which does not explain the cliff (see above) but does mean absolute SR is not
comparable across tasks -- report epochs beside any cross-task statement. (c) T2's demonstrator is
itself weak: gen SR 54.9/44.2/30.6/29.7 % across L0-L3, so the demos are a filtered subset of a
brittle expert, and at L1 the retained attempts are yaw-narrower than the rejected ones -- though
`success_vs_pose` showed the policy's failures are NOT concentrated where that coverage thins, so the
skew is measurable but not the mechanism. (d) Whether a longer schedule would clear the cliff is
untested; the honest statement is that it is unresolved at the frozen budget, and testing it needs a
protocol exception the user has not authorised.

# 2026-08-21 -- STUDY COMPLETE: all 78 cells evaluated, cross-task synthesis

Every cell of all three tasks is trained and evaluated: 78 result files (T1 30 including its
pose-redundant L3 ablation, T2 24, T3 24). Total cost **188.1 GPU-h over 97 cells**, about 8.5 % of
the 2,200 h ceiling, of which 13.3 h was the waste from D27's cancelled cells.

## The headline table

`experiments/cost_of_generality_summary.csv`; per-task detail in `experiments/curves_T{1,2,3}.csv`.

| task | level | SR@10 | SR@100 | SR@400 | N*(50%) | N*(80%) | N*(90%) | cost vs L0 @90% |
|---|---|---|---|---|---|---|---|---|
| cup_place | L0 | 0.85 | 1.00 | 0.99 | <=10 | <=10 | 16 | 1.00x |
| cup_place | L1 | 0.42 | 0.86 | 0.99 | 15 | 45 | 133 | 8.31x |
| cup_place | L2 | 0.31 | 0.82 | 1.00 | 20 | 86 | 150 | 9.38x |
| cup_place | **L3b** | 0.09 | 0.70 | 0.94 | 57 | 143 | **185** | **11.56x** |
| drawer_stow | L0 | 0.65 | 0.90 | 0.96 | <=10 | 18 | 23 | 1.00x |
| drawer_stow | L1 | 0.12 | 0.30 | 0.23 | >400 | >400 | >400 | >=17.4x |
| drawer_stow | L2 | 0.17 | 0.32 | 0.41 | >400 | >400 | >400 | >=17.4x |
| drawer_stow | **L3b** | 0.04 | 0.11 | 0.14 | >400 | >400 | >400 | >=17.4x |
| push_target | L0 | 0.99 | 1.00 | 1.00 | <=10 | <=10 | <=10 | -- |
| push_target | L1 | 0.58 | 0.97 | 0.96 | <=10 | 24 | 41 | >=4.1x |
| push_target | L2 | 0.41 | 0.94 | 0.95 | 16 | 42 | 67 | >=6.7x |
| push_target | **L3b** | 0.30 | 0.78 | 0.84 | 24 | 114 | **186** | >=18.6x |

## Four findings

**1. Generality has a monotone, large demonstration cost in every task.** Reaching 90 % costs 8.3x
more demos under pose randomisation and 11.6x once the object varies (cup_place); 4.1x and 18.6x
respectively (push_target). No task escapes it and no level is free.

**2. The absolute cost of full generality is nearly identical on the two tractable tasks -- and the
ratio is not.** cup_place needs **185** demos for L3b at 90 %, push_target **186**. Their ratios differ
almost threefold (11.6x vs >=18.6x) purely because push_target's fixed-scene baseline is easier
(<=10 demos vs 16). **So the ratio is a statement about the baseline as much as about generality, and
absolute N* is the more robust primary quantity.** Cost ratios should be reported with their
denominators visible; ours are in the table above for exactly that reason.

**3. On a long-horizon task, modest generality is unaffordable inside 400 demos.** drawer_stow goes
0.96 -> 0.23 the moment one object's pose randomises, and 40x more data barely moves it: every level
above L0 has N* > 400 at every target. Not undertraining (all cells converged at 80k, and L0 succeeds
on the same 18.4 epochs) and not generation coverage (`success_vs_pose` found failures spread
uniformly across the skewed axis). The evidence points to compounding error over ~680 steps in a
four-phase task, with no recovery behaviour in the demos because Mimic keeps only successes. **The
cost of generality is not merely larger for harder tasks; it can be prohibitive.**

**4. The randomisation axes are not additive, and the object axis dominates.** Adding goal
randomisation on top of pose randomisation costs little (cup_place 8.31x -> 9.38x; push_target
>=4.1x -> >=6.7x), while adding object variation costs the most (-> 11.56x and >=18.6x). Whatever the
policy struggles with, it is not the number of randomised dimensions but which ones.

## What the study also produced, beyond the surface

* A measured demonstrator-degradation curve per task (generation SR by level: cup_place flat at
  85-88 %, push_target 88-98 %, drawer_stow **54.9 / 44.2 / 30.6 / 29.7 %**), which is itself a finding:
  on drawer_stow the *data collector* degrades with generality, so part of any naive cost measurement
  belongs to the pipeline rather than the policy.
* A reusable bias-audit tool (`cog.analysis.gen_bias`) that compares retained against rejected
  generation attempts, plus `cog.analysis.success_vs_pose` which joins frozen eval-set initial states
  to per-episode outcomes. The second refuted the coverage hypothesis for drawer_stow L1 -- a negative
  result that mattered.
* A pose-diversity ablation for free: cup_place's original L3 arm (43 unique poses over 400 demos)
  against its corrected arm (401), same demo count. 0.445 vs 0.945 at N=400. **Pose diversity, not
  demo count, is what the demo axis has to measure** -- the single most expensive lesson of the study.

## Standing caveats, all recorded

One seed per cell (visible as push_target L3's non-monotone tail, 0.920 at N=200 against 0.840 at
N=400). Fixed 80k steps gives drawer_stow 3.7x fewer epochs than cup_place, so absolute SR is not
comparable across tasks even though within-task ratios are unaffected. The L3 object axis is mostly
appearance: the two cylinder sizes differ by 4 mm of radius. And whether a longer schedule would clear
drawer_stow's cliff is untested -- it needs a protocol exception, and is the one experiment worth
~8 GPU-h if that claim is to be airtight.

# 2026-08-21 -- reporting names settled (L3b IS L3), per-task curve figures, and a published figure that had been plotting the deprecated arm

**User directive:** "leave all L3* out of reporting [name as deprecated] and report L3b as L3, because
this is the current version. generate line plots for all three tasks (one per task with 4 lines each)".

**What the rename is and is not.** Files keep their names (`L3b` datasets, `t1_L3b_n100_s0` run ids,
`eval_T1_L3b_*.json`); only the reporting layer renames, via one function --
`cog.analysis.curves.canonical()` -- so a reported number stays traceable to the artifact it came
from. Rationale, the disk/report split, and the two ordering properties that keep it safe are in
**D29**. The registry now says `REPORTED AS L3 (D29)` on all 18 L3b rows and
`DEPRECATED as a reported level` on T1's six original-L3 rows (T2/T3's were already `superseded`).

**Regenerated, all from the eval JSONs, no hand-typed numbers:** `experiments/curves_T{1,2,3}.csv`
(24 cells each, levels L0-L3), `experiments/cost_of_generality_summary.csv` (12 rows), and the
figures. Headline N* values are unchanged by the rename, as they must be: T1 16/133/150/185,
T2 23/>400/>400/>400, T3 <=10/41/67/186.

**New figures `paper/figures/fig_sr_vs_n_T{1,2,3}.png`** -- one per task, four lines, Wilson bars,
log-N, `N*(90 %)` printed in each legend entry. Two things worth keeping:
* the legend sits BELOW the axes. Inside, no corner is free on all three tasks -- matplotlib's
  "best" put T2's legend on top of its own L0 point at N=10, because three of T2's four curves live
  under 45 %. A legend that hides a data point on one task is worse than 15 % of figure height on
  every task.
* Wilson offsets are clamped at zero. At p = 1.0 the interval's upper limit is 0.99987, *below* the
  point estimate, and matplotlib rejects a negative `yerr`. The interval is centred on a shrunk
  estimate; that is not a bug in the data.

**A real bug found while doing this, and it was in a published figure.** `fig_gen_sr` aggregated
`gen_stats.csv` by its `level` column -- but that CSV keys the two arms differently: the original arm
writes `level="L3"` with the variant in the `variant` column, while the regenerated arm encodes the
variant IN the level (`L3bv00`..`L3bv09`). So the figure pooled the ten deprecated `L3` rows and
silently dropped all thirty `L3b` rows for matching no known level name. It had been showing
87.9 / 32.7 / 88.5 %; the truth is **86.6 / 29.7 / 92.2 %**. Fixed by stripping the `v\d\d` suffix
before applying the same mapping, and cross-checked against `experiments/gen_bias.csv`, which counts
retained and rejected attempts directly out of the HDF5s and agrees to the digit.

`paper/limitations.md` carried a third set of values again (86.9 / 29.7 / 91.5) -- typed while the
regeneration was still running. Corrected, with the correction history left visible. The argument
built on those numbers needed weakening too: the artifact's error has no consistent sign or size
(T1 -1.3, T2 -3.0, T3 +3.7 points), so "a single pose set is harmless where generation is
pose-insensitive" is not supported. What survives is monotonicity -- the corrected T2 column falls
54.9/44.2/30.6/29.7 where the old one had L3 easier to generate than L2.

**Gate exceedance recorded:** T2's corrected L3 generation SR is **29.7 %**, just under the G3 floor
of 30 %, and G3 was not re-run against the regenerated arms. The shortfall looks like a demonstrator
genuinely struggling at full breadth on a long-horizon task rather than a mis-specified subtask
offset, but it is an exceedance of a stated gate and is now in `paper/limitations.md` rather than
rounded away.

**Housekeeping.** `cog.analysis.curves --out` now derives `experiments/curves_<TASK>.csv` from
`--task` instead of defaulting to a single `curves.csv` that three tasks overwrote; the stale
`curves.csv` is removed (superseded by `curves_T1.csv`). The cross-task summary generator, which had
been living in a scratch directory, is now `src/cog/analysis/summary.py` -- a committed CSV whose
generator is not in the repo is a number nobody can reproduce.

## Consolidation, and a half-applied rename found on the way (2026-08-21)

The whole study has been living on the branch `worktree-docs-dp-default-diff` in the worktree of the
same name -- created 2026-08-19 22:25 for a small "DP default deltas" doc task, then simply never left.
`main` is 46 commits behind and 0 ahead, i.e. a strict ancestor, so consolidation is a fast-forward.
Pushed to origin; the fast-forward of `main` itself has to run in the main checkout.

**Two run dirs exist under the same name in both checkouts, and they are NOT copies:**

* `t1_L0_n25_s0` -- the main checkout's is 1,067,218,908 bytes and dated 08-19 20:34; this worktree's
  is 1,112,015,188 bytes, dated 08-20 00:41. The difference is 44,796,280 bytes, one resnet18: the
  main-checkout copy is the **pre-D26 SHARED-encoder** run, and the worktree's is the canonical
  two-encoder cell. The registry has called that run `t1_L0_n25_s0_sharedenc` since D26 and records
  the rename as done *on $WORK* -- **the local rename was never applied**, so a stale directory has
  been sitting under a run_id the registry reassigned to a different architecture. Anything that had
  resolved a checkpoint path by run_id in the main checkout would have silently scored the wrong
  model. Renaming the local copy to match (`t1_L0_n25_s0_sharedenc`) is the fix; nothing is deleted.
* `g4_smoke_L0_n25` -- the worktree holds only the 92 KB of git-tracked skeleton files (the weights
  are gitignored and were never there); the main checkout holds the real 3.0 GB. Main's is the one to
  keep.

**Lesson worth generalising:** a rename recorded as done in the registry was only done on one of the
two machines. "Renamed to X" in a note should say WHERE, and a size check is a cheap way to tell two
architectures apart when the config files do not name the flag.

81 GB of checkpoints (79 dirs, ~1.1 GB each, `training_state` already pruned to 32 KB total) are
gitignored and live only in this worktree, so they must move into the main checkout BEFORE the
worktree is removed. Both paths are on `/dev/nvme0n1p5`, so the move is a rename, not a copy.
The only other copy is `$WORK/cog/checkpoints` on Leonardo, which dies with the grant on 2026-10-29.

### 2026-08-21 -- T2 L2>L1 inversion investigated: real at the episode level, best explained by demo-quality selection; eval-horizon confound ruled out

**Question:** T2 L2 (obj pose + cabinet pose) beats T2 L1 (obj pose only) at most N -- pooled
N>=50 L2 137/400 vs L1 97/400 -- although L2 is nominally harder. Investigated in the
results-analysis worktree; scripts under `scripts/dev/t2_*.py`, CSVs under `experiments/t2_*.csv`.

**The two frozen eval sets share bit-identical object poses.** `configs/eval_sets/T2_L1.json` vs
`T2_L2.json`: max |dx|,|dy|,|dyaw| = 0.0 exactly, because `events._sample_pose` draws all 6 dims
from the global RNG even for degenerate (fixed) ranges, so the stream stays aligned across levels.
The sets differ ONLY in cabinet pose (L1 fixed at (0.9, 0.0, yaw=pi); L2 x[0.85,0.95] y[-0.06,0.06]
yaw pi+-0.13). Drawer starts closed (joint 0) in every episode of both. Success criterion
(`object_stowed_in_drawer`) is drawer-BODY-frame relative with identical params; horizon identical
(episode_length_s=60 -> 1200 steps for every T2 level). So "L1's eval set was intrinsically harder"
is dead: same benchmark, easier (nominal, fixed) cabinet for L1.

**Statistics.** Episode-level: pooled N>=50 z=3.11 p=0.0019 (Fisher p=0.0024, OR=1.63); CMH over
all 6 N-strata chi2=5.29 p=0.021; McNemar on pose-paired episodes (N>=50 pooled: only-L2 103 vs
only-L1 63) p=0.0024. Per N, only N=50 (35 vs 19, p=0.016) and N=400 (41 vs 23, p=0.0097) are
individually significant; N=25 reverses (11 vs 22, p=0.056). BUT cell-level, treating each
training run as the unit (n=6 paired diffs +5,-11,+16,+2,+4,+18): paired t p=0.24. And within-level
single-run noise is the same magnitude as the inversion: L2 n25 (0.11) -> n50 (0.35) is Fisher
p=0.0001 on nested subsets. One seed means episode-level significance overstates certainty.

**Where L2 wins: everywhere, not in a subregion.** L2-L1 SR delta >= 0 in 11 of 12 pose-quartile
cells (|yaw|, x, y). Cabinet dims do NOT modulate L2 success (MWU p=0.74/0.22/0.84; carry distance
trend is even positive). So neither gen-bias subregion concentration nor easy-cabinet draws explain it.

**What does differ: demo quality via survivorship.** L2's demonstrator was filtered harder
(gen SR 30.6% vs 44.2%) and its retained demos are systematically more efficient EVEN AT MATCHED
GEOMETRY: L2 demos with cabinet within 2 cm of L1's fixed pose run 672 steps / 3.27 m eef path vs
L1's 694 / 3.40 m (Welch p=0.0002 / 0.0005); cabinet pose explains none of L2's length variance
(r=0.06 n.s.). Cleaner, more direct imitation targets on a compounding-error task. A second,
untested contributor: cabinet randomization as DR-style augmentation (would need L2-policy-on-L1-
eval to disentangle; ~2 GPU-h, not run). The inversion is T2-specific -- T1 and T3 both show
L1 >= L2 at every N.

**Ruled out (negative results):** eval-set difficulty (identical poses), horizon (both 1200 via
`--max_steps 1200`, commit 3370324 predates the T2 evals), epochs (18.44 vs 18.91 at N=400;
+-2.5% at every N), hyperparams (checkpoint config.json identical incl.
use_separate_rgb_encoder_per_camera=true), dead eval batches, registry anomalies, drawer initial
state, success-region size. L2 n25 < n10 is not significant (Fisher p=0.31).

**Q2 horizon check:** eval max_steps 1200 vs demo max 743 (L2) / 724 (L1) -> minimum headroom 457
steps (62% of the longest demo). Timeout censoring cannot explain low T2 SR or the inversion
(and L1's 2.5%-longer demos are immaterial at that slack). Caveat: rollout_eval does not record
episode lengths, so how close successful rollouts run to 1200 is unmeasurable post hoc. Epochs
arithmetic of the earlier entry verified exactly (18.44/18.16/67.94); the final-20k loss-drift
numbers could not be re-verified locally (train logs live only on the cluster).

### 2026-08-21 18:30 -- T2 follow-up evals LAUNCHED (user-authorised): L1<->L2 cross-evals + stage funnels

User authorised local-GPU follow-ups (no cluster): (a) the decisive cross-eval for the L2>L1
inversion -- evaluate the L2/N=400 policy on the frozen T2_L1 eval set (and the reverse) -- and
(b) per-stage SR for T2 to locate where the ~680-step task fails, motivated by "maybe shorten
the task?".

**Stage instrumentation** (`rollout_eval.py --stages`, drawer_stow only, read-only on sim state,
same latching semantics as the official success): `drawer_opened` = drawer_top joint >= 0.15 (the
success criterion's own threshold); `object_lifted` = object 5 cm above its episode-initial
height; `object_over_drawer` = object xy inside the cavity bounds in the drawer_top body frame
while the drawer is open; plus per-episode maxima (max_drawer_open, max_object_lift) and
first-latch step indices (t_open/t_lift/t_over/t_success). Funnel reading: opened -> lifted ->
over -> stowed(official).

**Queue** (`scripts/ops/run_t2_followup.sh`, tmux `cog_t2fu`, one eval at a time, waits for
>=10 GB free VRAM per the 2026-08-20 threshold, checkpoints read from the main checkout,
results into this worktree's `results/`):
1. `eval_T2_xeval_L2n400_onL1_080000.json` -- L2 ckpt on L1 eval set (+stages)
2. `eval_T2_L1_n400_080000_stages.json` -- L1 ckpt, own set, stage-instrumented re-run
3. `eval_T2_L2_n400_080000_stages.json` -- L2 ckpt, own set
4. `eval_T2_xeval_L1n400_onL2_080000.json` -- reverse cross (+stages)
5. `eval_T2_L0_n400_080000_stages.json` -- control
Frozen protocol untouched (same seeds/num_envs); the re-runs double as a determinism check
against the published SRs. Budget ~45 min/eval measured + up to 12 h headroom wait; hourly cron
fallback + watcher armed (rule 10).

Prediction registered up front: if the L2-on-L1 cross-eval SR stays ~0.4 (its own-set level),
the inversion is training-side (cleaner filtered demos / DR-augmentation), as the 2026-08-21
analysis concluded; if it drops to ~0.23 (L1's level), the eval-side explanations were wrongly
excluded and the inversion story needs rework.

# 2026-08-21 (night) -- EVAL HARNESS BUG: batch-boundary success carryover inflates EVERY multi-batch SR

Found via the stage instrumentation's first-latch timestamps: roughly half of all recorded
successes had t_success = 0 -- success on the first policy step after a batch reset, which is
physically impossible (the drawer starts closed; the shortest demo in any task is >= 150 steps).

**Mechanism, established empirically (scripts/dev/t2_t0_artifact_check.py, batch_pattern_check.py):**
on the first `env.step()` after the manual between-batch `env.reset(seed=...)`,
`termination_manager.get_term("success")` still returns the PREVIOUS batch's value. The official
latch `success |= succ_now & ~finished` therefore records a phantom success at t=0 of batch b for
every env that genuinely succeeded in batch b-1. Evidence, all from data already on disk:
1. Zeros never occur in batch 0 (nothing to carry): 0/60+ across five stage-instrumented runs.
2. Exact carryover identity: phantom count in batch b == true-success count in batch b-1 in
   20 of 20 batch transitions (one off by one).
3. Phantom episodes never lift the object or bring it over the drawer (scene-state reads are
   fresh; the stale read is confined to the termination buffer at t=0 -- min genuine
   t_success across runs is 606).
4. The published per-episode outcomes show the predicted signature everywhere: batch-0 SR is the
   lowest batch in 36 of 38 non-saturated flat cells (sign test p = 6e-10; mean rest-minus-b0
   gap +0.20). L3 cells ran ONE batch per variant in fresh processes and are therefore CLEAN.

**Scope:** every multi-batch eval ever run with rollout_eval.py -- all 54 flat cells of the
published study, the D24 checkpoint comparison, and this session's five follow-up evals (whose
stage data exposed it). Recorded SR = P(true(b) or true(b-1)), so inflation is largest exactly
where SR is mid/low -- the scientifically interesting cells.

**Fix:** rollout_eval.py now zeroes succ_now at t == 0 (genuine t=0 success is impossible in all
three tasks). One line; commit in this worktree.

**Post-hoc correction without re-running** (scripts/dev/stale_correction.py,
experiments/stale_corrected_sr.csv): the mechanism gives recorded(b,e) = true(b,e) OR true(b-1,e),
which constraint-propagates per env chain: recorded=0 forces both terms false (and identifies the
previous episode); recorded=1 after a known-false resolves true. Yields exact per-episode truth for
~85-95% of episodes in low/mid-SR cells, plus hard bounds [all-unknowns-false, all-unknowns-true]
everywhere. Validated on the five stage runs: truth (success AND raw object_over_drawer) inside the
bounds 5/5. Point estimates are unreliable for near-saturated cells (few identifiable episodes) --
those cells are barely inflated in absolute terms anyway.

**Corrected headline movements (point estimates, n_identified 82-99 unless noted):**
- T2 cliff DEEPENS: L1 0.08-0.15 across N (published 0.12-0.30); L2 0.06-0.21 (published
  0.11-0.41). L3b unchanged (clean). The "generality cliff" finding strengthens.
- T1/T3 mid-curve cells drop hard: T1_L1_n25 0.67->0.38, T1_L2_n50 0.75->0.43, T3_L2_n25
  0.63->0.34. All N*(50/80/90%) thresholds move right; every cost ratio needs recomputation.
- LEVEL ORDERINGS involving L3 arms flip in places, because flat cells were inflated and L3 cells
  were not: e.g. T1 L3b at N=100 is 0.70 (clean) vs L2 corrected ~0.50 -- published order said L2
  0.82 > L3b 0.70. Finding 4 ("the object axis dominates the cost") is now in doubt and must be
  re-derived from corrected numbers.
- Saturated cells (>=0.95): inflation bounded by (1-p)^2 arithmetic; conclusions unaffected.

**Decision needed (user):** a clean re-eval of the 54 flat cells with the fixed harness costs
roughly 15 h serial on the 4090 (T2 ~23 min/cell measured tonight exclusive, T1 ~9, T3 ~17), ~8-10 h
with two slots. The constraint-propagation numbers are defensible for the paper's qualitative
claims, but exact N*/cost-ratio tables should come from the re-run. Not launched unasked.

### The T2 follow-up results themselves (all five evals completed, ~23 min each exclusive)

Corrected numbers (success AND raw object_over_drawer; raw stage latches are artifact-free):

| policy -> eval set | opened | lifted | over drawer | stowed (corr) | recorded (buggy) |
|---|---|---|---|---|---|
| L0 -> L0 | 0.90 | 0.92 | 0.87 | 0.86 | 0.94 |
| L1 -> L1 | 0.87 | 0.19 | 0.17 | 0.17 | 0.26 |
| L2 -> L1 (cross) | 0.85 | 0.29 | 0.27 | 0.27 | 0.43 |
| L1 -> L2 (cross) | 0.79 | 0.26 | 0.22 | 0.22 | 0.37 |
| L2 -> L2 | 0.87 | 0.33 | 0.29 | 0.29 | 0.47 |

- **Cross-eval verdict (the D-question from this afternoon): the L2>L1 inversion is a POLICY
  effect, confirmed on corrected data.** 2x2 decomposition: policy effect +10/+7 pts at fixed
  set, set effect +5/+2 at fixed policy. Episode-paired McNemar on recorded outcomes (identical
  object poses): pooled policy effect p=0.0045; pooled set effect p=0.11. The
  cleaner-survivor-demos explanation stands; the small ns set effect hints L1's fixed nominal
  cabinet is marginally harder than L2's randomized average.
- **Stage funnel: the drawer phase is essentially solved at every level** (0.79-0.90 opened,
  median t_open ~155 steps regardless of policy), **and stowing is free once the object is over
  the drawer (P(stow|over) = 1.00 in all five runs, 107/107).** The single bottleneck is
  open->lift: 1.00 (L0) vs 0.38 (L2) vs 0.22 (L1). T2's generality cliff is entirely "grasp the
  object mid-rollout after ~300 steps of accumulated drift" -- the same pick that works at ~0.95+
  in T1 at matched level/N. This localises the compounding-error mechanism.
- **"Shorten the task?"** (user question): splitting at the drawer would work as measurement --
  an open-drawer-start variant isolates the true bottleneck and cuts ~25% of horizon -- but the
  funnel predicts its SR at ~P(lift|open) = 0.2-0.4, so shortening alone does not rescue T2;
  the pick-under-drift is the expensive part, not the drawer.
- Determinism note: same-seed re-runs agree with published outcomes on only ~83-84% of episodes
  (recorded-vs-recorded); net SR drift +3/+6 pts. Per-episode nondeterminism is real; McNemar
  pairings are noisy but unbiased.
- Timing: five T2 evals at 22-24 min each with the GPU otherwise free (vs 43 min measured shared
  on 2026-08-20) -- timings.md updated.

### 2026-08-21 21:51 -- CLEAN RE-EVAL SWEEP LAUNCHED (user-authorised): 54 flat cells + D24 re-check, fixed harness

User: "yes, rerun eval. use two slots if gpu permits. check hourly. if gpu later permits increase
parallelism." Launched `scripts/ops/resweep_eval.py` in tmux `cog_resweep`: 56 evals (54 flat
cells at 080000 + t1_L0_n25 at 040000/060000 to re-check D24), frozen protocol, fixed harness
(t==0 phantom guard), --stages kept on for T2 cells. Outputs `results/eval_*_fixed.json`
(originals untouched); resume-safe (skips existing outputs).

Parallelism is admission-controlled rather than a fixed slot count: a new eval starts only when
free VRAM >= 10 GB, starts staggered 5 min apart, hard cap 3 concurrent (3 x ~7 GB + margin;
timings.md forbids more). With the foreign job resident this yields exactly 2 slots; the 3rd
admits itself if the card empties -- which implements the "increase parallelism if gpu permits"
directive without ever squeezing the foreign job (rule 2). Per-job 150-min kill timeout;
per-job logs `ops/resweep/`. Watcher + hourly cron armed (rule 10). Estimate ~7-9 h wall at 2
slots (T2 23 min, T3 ~17, T1 ~9 per cell).

Analysis planned on completion: recompute the corrected surface from the _fixed results, compare
against the constraint-propagation point estimates in `experiments/stale_corrected_sr.csv`
(a direct validation of that estimator), regenerate the corrected-vs-published figure with final
numbers, re-derive N*/cost-ratio tables and the finding-4 (object-axis) verdict, update the
report artifact, and re-judge D24 (last-checkpoint-only) on clean numbers.

### 2026-08-21 22:12 -- PHASE 2 CHAINED: six T2 L3 cells re-run with stage telemetry (user instruction via origin session)

User (relayed verbatim from the origin session): "i want full curves - just also rerun eval for
remaining 6 cells... once scheduled queue is done". Chained `scripts/ops/resweep_l3_phase.py` in
tmux `cog_resweep_l3`: waits for `cog_resweep` to end AND requires RESWEEP_DONE in its log (a
crash without the marker aborts the phase), then runs t2_L3b_n{400,200,100,50,25,10} through
`run_local_eval_l3.py` -- now passing `--stages` per variant (task-conditional, so T1/T3 L3
would be unaffected) -- giving the drawer-open/lift/over funnel for the FULL 24-cell T2 matrix.
N=400 first. Two cell drivers concurrently (each holds <=1 Isaac process at ~7 GB; three L3
drivers would churn too many simultaneous Kit boots), driver-internal VRAM gate lowered to the
standard 10 GB via COG_EVAL_MIN_FREE_MIB. Outputs use the driver's diagnostic path:
`results/diagnostics/eval_T2_L3b_n<N>_080000_fixed.json` -- deliberately outside the
registry/curves globs, exactly because `_fixed` flat files in `results/` DO match the
`eval_T*` glob prefix; any curves/registry regeneration must handle the _fixed convention
explicitly (noted for the merge). Checkpoints reach the worktree via per-run symlinks
`experiments/runs/t2_L3b_n*_s0` -> main checkout (runs/ is gitignored). ~1.5 h/cell, ~5 h at
2 slots after phase 1 drains. Note: the L3 carryover exposure is zero (one batch per process),
so these re-runs exist for the stage funnel and the same-harness consistency check, not for
correction. Watcher + the hourly cron (prompt updated to cover both phases) armed.

### 2026-08-22 00:30 -- Clean-sweep anomaly investigated: old T2_L1 n50/n100 evals are untrustworthy; env-index correlation discovered; guard verified sound

First four clean T2_L1 cells: n400 0.15, n200 0.18, n100 0.31, n50 0.32 -- n50/n100 EXCEED even
their uncorrected published numbers (0.19/0.30), which phantom-removal alone cannot do. Worked
through the hypothesis chain, each with a measurement:

1. **Checkpoint mislabeling in the main checkout: DEAD.** Every run dir's embedded
   `train_config.json` (written by the training job itself) matches its directory: job_name,
   output_dir, dataset repo_id, and the exact episode list [0..n-1]. 0 mismatches across all 80
   run dirs (`scripts/dev/` check, this session).
2. **`terminated` also stale at t=0 (guard would censor carried envs as phantom failures): DEAD.**
   Under censoring no env could be recorded success in consecutive batches; the clean runs show
   consecutive successes in 7/9, 10/15, 16/24, 15/25 of opportunities, and post-success failures
   are not frozen at t=0 (`scripts/dev/censoring_check.py`). The t==0 guard removes phantom
   LABELS only; the fixed harness is sound.
3. **What the censoring check exposed instead: success is strongly ENV-INDEX-correlated.** Envs
   that truly succeeded in batch b-1 succeed again in batch b at 60-78%, against base rates of
   15-32%. The 20 vectorized clones are not statistically identical episodes-across-batches --
   plausible mechanism: per-env-origin differences (RTX lighting/shadows vary across clone
   positions), or physics-state persistence across manual batch resets. CONSEQUENCES:
   (a) the constraint-propagation POINT estimates in stale_corrected_sr.csv are biased LOW --
   they exclude exactly the episodes that follow a success, i.e. the good envs. The hard BOUNDS
   are logic-only and stand. (b) Success-by-env-index uniformity should be tested on the full
   clean sweep (flag for completion analysis).
4. **The old n50/n100 runs contradict their own hard bounds: the original evals of (at least)
   these cells did not measure today's checkpoints.** Original n50's true SR is provably <= 0.14
   (recorded 0.19 with proven phantoms; bound is assumption-free given the carryover identity);
   the clean re-run on identical seeds, verified checkpoints, unchanged task code (last
   drawer_stow change 530df00, before the original evals) measures 0.32. Old-vs-new episode
   agreement is only 0.73-0.75 on these cells. Most plausible: the docs-dp-default-diff
   worktree's checkpoint copies for some cells were stale/corrupt at original eval time (that
   worktree's runs/ is gone -- moved, not copied -- so unverifiable by hash). n400/n200 old runs
   are consistent with clean (within correction + nondeterminism), so the damage is per-cell,
   not global.

**Bottom line: published numbers are not reliably correctable post hoc -- for some cells the
original measurement itself is wrong, beyond the carryover bug. The clean sweep (running) is the
only authoritative surface.** The report's corrected-estimates table is hereby demoted to
"bounds where the original run was sound"; final tables come from the sweep alone.

# 2026-08-22 -- CLEAN SWEEP COMPLETE: 62/62 evals, the final surface, and the warm-up effect that re-opens finding 4

The full re-eval finished: 54 flat cells + 6 T2-L3 cells (stage-instrumented) + the D24
40k/60k pair = **62 evals, zero failures, ~18 h wall** (2026-08-21 21:51 -> 08-22 15:32) on the
shared 4090 with admission-controlled 2-way parallelism. Authoritative tables:
`experiments/clean_surface.csv`, `clean_nstar.csv`, `t2_stage_funnel_full.csv`,
`warmup_matched_f4.csv`; figure `paper/figures/corrected_vs_published_sr.png`.

**The clean surface (SR at N=10..400):** T1 L0 0.89-1.00, L1 0.46->0.99, L2 0.29->1.00, L3
0.09->0.94; T2 L0 0.71->0.95, L1 0.11/0.23/0.32/0.31/0.18/0.15, L2 0.13/0.16/0.37/0.43/0.27/0.37,
L3 0.04-0.14; T3 L0 ~1.0, L1 0.56->0.99, L2 0.38->0.95, L3 0.30->0.84.

**N\*(90%) / cost ratios (clean):** T1: 11 / 126 / 141 / 180 -> L1 11.4x, L2 12.8x, L3 16.3x.
T2: L0 21, everything else >400 (>=18.8x). T3: <=10 / 37 / 59 / 181 -> 3.8x / 6.0x / 18.1x.

**Verdicts, one by one:**
- **Phantom-carryover bug: material only for T2 L1/L2.** Clean T1/T3/T2-L0 match published
  within noise (only 1 of 36 comparable flat cells differs >2 sigma -- T2_L1_n50, the cell whose
  original run was separately proven bad). T2 L1 n400 0.23->0.15, n200 0.25->0.18.
- **D24 re-check (clean): 0.95 / 0.98 / 0.98** -- last checkpoint (joint-)best;
  last-checkpoint-only protocol stands.
- **Batch-0 depression = PROCESS WARM-UP, proven by probe.** A fresh process evaluating seeds
  5001-5004 (skipping 5000) scored [0.35, 0.65, 0.80, 0.85] batch-by-batch, converging on those
  seeds' warm values [0.85, 0.70, 0.60, 0.85]: the depression follows process position, not the
  seed. Cost ~ the first 20 episodes (one batch), then gone. Present in 37/37 non-saturated clean
  cells, mean -0.21 (T1 0.43-vs-0.75, T2 0.33-vs-0.50, T3 0.65-vs-0.85). Artifact:
  `results/diagnostics/eval_T1_L1_n25_080000_warmupprobe.json`. Likely renderer (RTX
  accumulation/shader-cache) or physics warm-up; the exact sub-mechanism is untested.
- **Finding 4 ("object axis costs most") is CONFOUNDED and does not survive as stated.** L3's
  protocol makes all 200 episodes first-batch-cold while flat cells are 80% warm. On the matched
  comparison (flat batch-0-only, n=20/cell, vs L3): T1 L3 at n100 0.70 vs L1/L2 0.50; T3 L3
  tracks L2 within noise from n25 up. A small-N premium survives (L3 worst at N<=25 everywhere).
  The published L3 curves are conservative (true warm L3 performance is HIGHER than reported).
  **Deciding it needs warm L3 evals: prepend one unscored burn-in batch per variant and re-run
  the 18 L3 cells (~10-12 h local). Not launched unasked.**
- **Env-index non-uniformity (new protocol-level finding):** success depends on which of the 20
  clones runs the episode -- pooled chi2 p = 2.4e-6 (T1) / 1.3e-2 (T2) / 6.9e-4 (T3), and
  repeat-success rates of 60-78% vs 15-32% base. The clones are not exchangeable; suspect
  per-clone-origin rendering differences. Affects any per-episode statistics that assume
  episode independence.
- **T2 story final:** cliff confirmed and deepened; L2>L1 inversion holds on clean data (0.37 vs
  0.15 at n400) and is policy-side (cross-evals); the full 24-cell stage funnel pins every SR to
  the mid-rollout grasp (lift: L0 0.84-1.00, L1 0.14-0.37, L2 0.14-0.44, L3 0.07-0.18; drawer
  opening 0.65-0.98 with no level trend; stow-once-over ~1.0 everywhere). NEW: both L1 and L2
  show a mid-N hump (peak n50-100, decline to n400) -- unexplained, one seed, flagged.

**Open decisions for the user:** (1) warm-L3 re-protocol + re-run (~10-12 h) to settle finding 4;
(2) second seed for T2 L1/L2 (hump + inversion fragility); (3) whether the paper reports
batches 1-4 only (n=80, warm) as the primary metric with batch 0 quantified as a limitation.
Registry not modified (these are re-evals of existing runs; clean_surface.csv is the source of
truth until the user merges).

### 2026-08-22 06:15 -- MAJOR REVISION: batch-0 is GENUINELY depressed; the "36/38 cells inflated" inference was two phenomena conflated

Clean T1_L1 cells came in at the PUBLISHED values (n100 0.86=0.86, n50 0.84~0.83, n25 0.68~0.67),
not at the corrected estimates -- and the per-batch decomposition settles why: clean
T1_L1_n100 with the t==0 guard ACTIVE is [0.50, 0.90, 0.95, 0.95, 1.00], identical to the old
run. Batches 1-4 at 0.9-1.0 cannot be phantom-inflated in a guarded run, so **batch 0 is
genuinely worse, not the only honest batch**. Two separate phenomena were conflated in the
2026-08-21 night entry:

1. **Carryover phantoms: real, but only where mid-batch SR is mid/low.** Mechanically proven on
   instrumented T2 runs (20/20 identity, phantoms never lift). Where batches 1-4 run near
   ceiling (T1 L1/L2 at N>=50, T3 high cells, T2 L0), a phantom almost always coincides with a
   genuine success, so net inflation was NEGLIGIBLE -- clean T1/T2_L0 reproduce published. Where
   mid-batch SR is 0.1-0.5 (T2 L1/L2), phantom inflation was real (clean L1 n400 0.15 vs
   published 0.23).
2. **Batch-0 depression: a real, run-independent effect** present in old AND guarded-clean runs,
   across tasks (T1_L1_n100 0.50 vs rest 0.95; clean T2_L1_n50 batch0 0.20 vs rest ~0.35).
   Plausible mechanisms, to be separated in the completion analysis: renderer/physics warm-up in
   a fresh process (first episodes see unconverged RTX lighting), or the first reset after env
   creation sampling off-snapshot states. NOT a scoring bug -- those episodes genuinely fail.

**Consequences:**
- The constraint-propagation correction anchored truth on batch-0 -> systematically UNDERSHOT.
  stale_corrected_sr.csv is fully deprecated (bounds included -- their premise "recorded =
  true OR prev-true" holds, but "batch-0 = representative truth" does not).
- **L3 evals run every variant in a fresh process, so ALL 200 L3 episodes are batch-0-like.** If
  batch-0 depression is process-warm-up, published L3 cells are biased LOW relative to flat
  cells' batches 1-4 -- the opposite direction of the phantom bias. Finding 4 (object axis
  dominates) is now double-contested. Completion analysis must compare: flat batch-0-only SR vs
  L3 SR (same warm-up regime), and phase-2 L3 stage data (t_open distributions) for warm-up
  signatures.
- The old T2_L1 n50/n100 anomaly STANDS (old runs lower than clean across ALL batches, which
  neither phenomenon explains) -- those two original runs remain untrustworthy.
- Protocol implication for the paper: either discard batch 0 (report batches 1-4, n=80/cell,
  and accept L3 needs a warm-up-matched re-protocol) or quantify the warm-up effect explicitly.
  User's call; flagged for the completion report.

### 2026-08-21 -- Generator-filter contamination audit CLOSED for all 12 cells: no cell's SR is credibly inflated by the generator's selection filter

The 2026-08-20 concern -- "demos exist only where generation succeeded, so measured SR partly
reflects the generator's filter, not the policy" -- is now tested end-to-end for every task x level,
not just T2 L1. New tooling joins the frozen eval-set initial states (world->env-local via per-env
origins recovered from the fixed anchor entity and validated against the declared ranges) to
per-episode outcomes, and adds two composite measures per cell: (a) kNN local rejection rate
(fraction rejected among an eval state's 25 nearest generation attempts, z-scored demo-observable
dims) vs success, and (b) nearest-TRAINING-demo distance (actual N-demo subset from
conversion_manifest.json) vs success. N=400 primary, N=100 secondary. Artifacts:
`experiments/genbias_link.csv` (+ `_episodes.csv`, `_cells.json`),
`paper/figures/genbias_link_{T1,T2,T3,summary}.png`,
scripts in `scripts/dev/genbias_link_{stats,figs}.py`.

**Verdicts (N=400):** T1 L1/L2/L3b absent; T2 L1/L2/L3b present-but-immaterial; T3 L1/L2/L3b
present-but-immaterial (T3 L1 link suggestive p=0.09, bounded; T3 L3b shared-difficulty, see below).
L0 cells exempt (single fixed pose).

**The pose-blind baseline decomposition is the headline bound.** L0 rejects with NO pose variation:
T1 13.6%, T2 45.1%, T3 1.5%. Subtracting it, the pose-SELECTIVE rejection mass is T1 <=0.6-1.3%
(T1's filter is essentially pure controller noise), T2 <=10.8% (L1) / ~25% (L2, L3b), T3 <=3.7-6.3%.

**T2: filter large and real, effect on SR absent.** Retained-vs-rejected KS: L1 skewed on |yaw|
(p=8e-15, rejects at large |yaw|); L2/L3b skewed on SIGNED yaw (p=8e-20 / 1e-22 -- retained demos
shifted toward negative yaw) and weakly on y. Downstream: success is flat across every filtered dim
(L1 |yaw| p=0.66; L3b obj_yaw p=0.50), local-rejection link null (p=0.92/0.62/0.44), nearest-demo
distance null, per-variant gen SR vs eval SR uncorrelated (Spearman -0.14, p=0.71). T2's plateaus
(0.23/0.41/0.14) are NOT a data-coverage artifact; the epoch-deficit explanation stands. If
anything the L2/L3b filter starved the +yaw region without hurting success there -- anti-inflation.

**T3 L1, the D=0.68 mystery resolved:** the KS compares 400 unique retained vs only 22 unique
rejected attempts; all 22 rejects sit at y>-0.088 (puck starting nearest the y=-0.04 edge), which
is a real but tiny filter: local rejection in that strip is 12.9% (148 retained demos remain), 0%
elsewhere. Eval SR in the strip 0.947 vs 0.968 outside; local-rejection link rb=-0.41, p=0.09 with
only 4 failures -- suggestive direction, bounded <=~5 SR points worst-case, not material.

**T3 L2/L3b carry a filter dim gen_bias.csv missed: bearing.** Demos DO record target_pos, so
bearing is auditable end-to-end: rejects concentrate at low bearing (KS D=0.485/0.395, p=7e-5/6e-5).
At L3b the only p<0.01 success-geometry result of the audit appears: success rises with obj_x
(rb=+0.31, p=0.006) and suggestively with bearing (p=0.03); local-rejection link suggestive in the
contamination direction (rb=-0.23, p=0.034; gen-easy half SR 0.892 vs gen-hard 0.786, Fisher
p=0.053). BUT nearest-demo distance is dead null (p=0.86) -- failures are not far from training
demos -- and the x-direction is anti-filter (rejects sit at HIGH x where success is highest). Read:
the low-bearing/low-x region is intrinsically hard for scripted generator and learned policy alike
(common cause), not a demo-starvation effect; bounded by 7.8% rejected mass anyway.

**T1: nothing.** All retained-vs-rejected KS ns (filter spatially blind, matching the <=1.3%
pose-selective mass); the weak obj_x success gradient at N=100 (p~0.02) cannot be generator-caused.

**GOTCHA (recorded for reuse): L3b eval JSONs log batch=0 for all 200 outcomes.** The outcomes
list is variant-major (10 blocks of 20, env 0..19 per block; block v = variant v = eval-set batch v,
verified against per_variant successes and seeds). Joining L3b outcomes to eval-set poses on the
recorded (batch, env) silently maps 90% of episodes to variant-0 poses -- this initially produced a
spurious "significant" contamination link in T2 L3b (p=0.003) that vanished (p=0.44) once the join
was fixed. Any future per-episode analysis of L3 evals must reindex batch = index//20.

Negative results worth keeping: success_vs_pose's T2 L1 yaw-bin null replicates under MW/KS/logistic
and extends to all dims and all cells; no cell shows the coverage signature (higher SR where demo
density is high) at p<0.01 anywhere.

## 2026-08-22 -- Candidate B: multi_task_dit backported onto 0.4.4; B1 unit + dry-parse gates green

Branch `lang/cand-b` (worktree lang-cand-b). The 0.5.2 checkout's multi_task_dit policy
(project_repos/lerobot_AICchallange/lerobot @ fc6c94c, READ-ONLY) now runs on the PINNED
lerobot 0.4.4 as the in-repo plugin `src/lerobot_policy_mtdit/`. Module names kept
verbatim (`configuration_/modeling_/processor_multi_task_dit.py`) because 0.4.4's
factory derives the modeling/processor module paths from
`config_cls.__module__.replace("configuration_", ...)` (factory.py:531-590); the package
name itself must not contain those prefixes. Activation:
`PYTHONPATH=<repo>/src` + `--policy.discover_packages_path=lerobot_policy_mtdit`
(the parser wrapper loads + strips the flag BEFORE config parsing, so it is legal — and
required — on the config_path-only resume invocation too).

**API deltas actually hit: exactly the 5 predicted, zero stragglers.**
1. configuration: `lerobot.configs` has no package `__init__` at 0.4.4 →
   `lerobot.configs.policies.PreTrainedConfig` + `lerobot.configs.types.NormalizationMode`.
2. configuration: `lerobot.optim` re-exports gone → `.optimizers.AdamConfig` +
   `.schedulers.DiffuserSchedulerConfig`.
3. modeling: 0.4.4's import_utils lacks `_diffusers_available`/`require_package` → new
   `_compat.py` (12 lines: flag via 0.4.4's own `is_package_available`; `require_package`
   copied verbatim from the 0.5.2 checkout utils/import_utils.py:86-95).
4. modeling: relative `..pretrained`/`..utils` → absolute `lerobot.policies.*`.
5. processor: `policy_action_to_transition`/`transition_to_policy_action` not re-exported
   by `lerobot.processor` at 0.4.4 → import from `lerobot.processor.converters`.
Checked-identical across versions (so NOT deltas): `populate_queues`, `PreTrainedPolicy`'s
five abstract methods, `TokenizerProcessorStep` kwargs, all `lerobot.utils.constants`
names, all eight `lerobot.processor` step classes.

**B1 gate 1 (unit smoke, scripts/dev/smoke_mtdit_unit.py): MTDIT_UNIT_OK.** Registration
via plugin import; config with study features (state (9,), 2x128x128 cams, action (7,));
crop-survival assert (validate_features silently sets image_crop_shape=None when
crop > effective size — with resize [256,256] the [224,224] crop SURVIVES, guarding the
CLIP fixed-224-pos-embed trap); policy builds: 249.0M params, 185.8M trainable (63.2M =
frozen CLIP text tower); decisive conditioning assert
`conditioning_dim == (9 + 768*2 + 512) * 2 = 4114` proves the 512-d text projection is in
the conditioning vector; fwd+bwd on random batch with pre-tokenized language: loss 1.259
finite, grad_norm 3.92, text-tower grads None, projection grads present.

**B1 gate 2 (draccus dry parse + 2 real steps): MTDIT_DRYPARSE_EXIT=0.** Full
COG_DIT_FLAGS set through `lerobot.scripts.lerobot_train` on local/L0 episodes [0,1],
batch 2, pyav: parses, constructs, runs 2 steps (loss 0.993 → 0.943), writes checkpoint
000002 with model.safetensors + config.json + policy_pre/postprocessor jsons. This also
proves the tokenizer pulls the dataset's task string: `task` flows dataloader →
`complementary_data` (converters.py:170) → TokenizerProcessorStep →
observation.language.{tokens,attention_mask}; had it not, the conditioning vector would
be 1024 short and the AdaLN linear errors at first forward. Overrides for smokes must
come AFTER ${COG_DIT_FLAGS} (draccus last-wins), mirror image of the diffusion_base rule.

New files: `configs/train/lang_dit_b.sh` (COG_DIT_FLAGS: resize 256/crop 224, horizon 20 /
n_action_steps 16 / n_obs_steps 2 for 20 Hz, policy preset lr 2e-5 + vision tower 0.1x,
ONE shared CLIP encoder for both cams — deliberate asymmetry vs D26, goes in the report;
COG_DIT_BATCH default 64), `slurm/train_lang_dit.sbatch` (sibling of frozen train.sbatch;
RUN_ID `*_s0_mtdit`; discover flag on BOTH branches; 24h walltime),
`scripts/ops/assert_resume_config_mtdit.py` (COG_DIT_* resume guard — the diffusion one
would refuse or vouch against the wrong flag set), `sync_up.sh hf` mode (stages
models--openai--clip-vit-base-patch16 → $WORK/cog/hf_cache/hub/, no --delete).

## 2026-08-23 -- Candidate B: B1 gates 3-5 (300-step train, DDIM reload, throughput probe)

**Gate 3 (300-step train): SMOKE300_EXIT=0, loss falls 0.504 -> 0.132.** local/L1 eps
0..24, batch 16, pyav, tmux `cog_smoke_mtdit_300` (finished in ~75 s, well under the
10-min rule-10 threshold, so no watcher/cron were needed; launched with all-absolute
paths per running_jobs.md anyway). Loss curve (log_freq 25): 0.504, 0.227, 0.203, 0.176,
0.186, 0.152, 0.181, 0.162, 0.162, 0.140, 0.150, 0.132. Checkpoints 000150 + 000300 both
complete: model.safetensors, config.json, train_config.json, policy_preprocessor.json +
step_4_normalizer safetensors, policy_postprocessor.json + step_0_unnormalizer.
**4.83 steps/s at batch 16** (updt_s 0.205, data_s 0.002 -- COMPUTE-bound; pyav keeps up,
unlike the dataloader-bound ResNet diffusion cells).

**Gate 4 (reload + inference): MTDIT_RELOAD_OK** (scripts/dev/smoke_mtdit_reload.py).
Mirrors rollout_eval.py's exact pattern: `MultiTaskDiTPolicy.from_pretrained(ckpt,
cli_overrides=["--noise_scheduler_type=DDIM","--num_inference_steps=10"])` (asserted:
config flipped, objective.num_inference_steps=10, scheduler isinstance DDIMScheduler) +
`make_pre_post_processors(cfg, pretrained_path=ckpt)` (5 pre / 2 post steps from the
saved jsons). One `select_action` on a dummy 20-env batch -- state (20,9), two
(20,3,128,128) cams, "task"=[canonical]*20 -- returns postprocessed action (20,7) finite,
on cpu. Verified along the way: AddBatchDimensionProcessorStep is a NO-OP on batched
inputs (only unsqueezes dim-1 states / dim-3 images / plain-str task), so the harness's
batched-obs convention passes through the pipeline unchanged, and the tokenizer emits
observation.language.tokens (20,77) from the task list.

**Gate 5 (throughput probe, batch 16/32/64): only batch 16 measurable locally today.**
A foreign process (PID 1526420, 6.7 GiB, untouchable per guardrail 2) occupied the card
throughout: b16 4.83 steps/s uncontended (gate 3) vs 3.67 contended, peak 13.3 GiB;
b32 OOM at 16.2 GiB needing +296 MiB (retried with PYTORCH_CUDA_ALLOC_CONF=
expandable_segments, same) -- would likely fit alone (~16.5-17 GiB) but not beside 6.7
GiB foreign; b64 OOM in warm-up, needs >20 GiB alone -> never local, fine on A100-64GB.
Full numbers + notes in docs/timings.md. The B2 cluster dbg smoke (64/128/192 on A100)
remains the batch-decider, unchanged.

**B1 VERDICT: PASS.** All five B1 stages green (unit MTDIT_UNIT_OK, dry-parse
MTDIT_DRYPARSE_EXIT=0, 300-step loss falling + checkpoint complete, MTDIT_RELOAD_OK,
throughput recorded with the local-VRAM caveat). Deviations from plan: none in substance;
b32/b64 local throughput unmeasurable under foreign GPU occupancy (recorded, cluster
probe covers it).
