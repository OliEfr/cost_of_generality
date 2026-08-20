# Decisions (ADR-style, newest last)

## D1 — 2026-08-16: Default cup = procedural cylinder tumbler; mug USD only as L3 variant after QA
Why: zero cloud-asset risk, robust top-down grasp, no handle-vs-yaw grasp confound
(a mug handle breaks top-down grasps at random yaw, coupling expert quality to the
generality level -- a confound in a data-cost study). Cylinder is yaw-symmetric.
Consequence: L1 yaw randomization is visually inert for the cup body; position is
the operative axis. Mug variants (mug_s/mug_m) join L3_VARIANTS only if grasp+render
QA passes.

## D2 — 2026-08-16: L3 object variation via per-variant sub-environments, not in-scene collections
Why: RigidObjectCollection + in-focus machinery breaks Mimic's get_object_poses
contract and complicates obs/terminations; per-variant envs (L3v00..) keep every
env trivially Mimic-compatible. Datasets merged at the LeRobot level with
variant-interleaved episode order (nested-N prefixes stay variant-balanced);
eval runs per-variant and pools. Distribution semantics identical.

## D3 — 2026-08-16: Env pins beyond PINS.md
numpy==1.26.4 (2.4.6 segfaults Kit: pinocchio compiled vs numpy 1.x, imported via
IsaacLab dex_retargeting at startup); transformers<5 (=4.57.6; transformers 5.15
requires huggingface-hub>=1.5 while lerobot pins <0.36; base lerobot does not need
transformers -- it is an isaaclab dep). ACCEPT_EULA/PRIVACY_CONSENT/OMNI_KIT_ACCEPT_EULA
set as conda env vars.

## D4 — 2026-08-16: One LeRobot dataset per level; N-cells via train-time episode subselection
Why: avoids 6x dataset duplication; nested subsets guaranteed by committed shuffle
order (seed 0) in conversion_manifest.json. Consequence: normalization stats come
from the FULL pool for every N (deliberate: removes normalization as a nuisance
variable across cells; noted in paper methods). VERIFIED 2026-08-16 (source, lerobot 0.4.4): DatasetConfig.episodes
exists in lerobot 0.4.4 train CLI; fallback = per-N dataset copies.

## D5 — 2026-08-16: observation.state = proprio only (eef pose + gripper, 9d)
Privileged object state goes under info.* keys (NOT observation.*) so lerobot's
automatic feature->policy-input mapping cannot wire it into the vision policy.
VERIFIED 2026-08-16 (source): dataset_to_policy_features classifies by prefix; non-observation.*/action keys (our info.*) hit `else: continue` and are dropped. Also verified: LeRobotDataset loads meta/stats.json from the dataset root unconditionally, so normalization stats stay FULL-POOL under episodes= subselection (D4 assumption confirmed).

## D6 — 2026-08-16: Eval protocol frozen
configs/eval_sets/protocol.json: 100 episodes/cell = 5 batches x 20 envs,
env.reset(seed=5000+b); headline cells rerun with batches 0-9 (200 eps).
Determinism of seeded reset sampling VERIFIED 2026-08-16 (scripts/dev/seed_determinism.py): reset(seed=5000) snapshots (cup/goal/joints, 4 envs) identical within-process after interleaved reseeding AND across two separate app launches.

## D7 — 2026-08-16: Demo actions recorded in IK-Rel space end-to-end (spec 04 Option A)
SM emits absolute EE targets, driver converts to IK-Rel deltas with the exact
formula the Mimic env uses; no cross-action-space conversion anywhere.

## D8 — keep Gaussian joint-reset noise at ALL levels, including L0

**Decision (2026-08-16):** keep the stock stack-task reset event
`randomize_joint_by_gaussian_offset` (mean 0, std 0.02 rad) on the Franka joints for
every sub-level, including L0.

**Why:** with a bit-identical initial state, all L0 demos would be the same episode;
SR(N) would be degenerate (step function at N=1) and the L0 baseline meaningless.
The noise gives "fixed task + natural motor noise": the NVIDIA-standard value, small
enough not to change the task, large enough that demos are distinct. Because it is
identical across levels it cancels in the cost ratios N*(L_k)/N*(L0).

**VERIFY:** none — verified visually in frames QA (wrist views vary slightly across
L0 resets; table-cam scene layout identical).

## D9 — 2026-08-16: Source demos recorded on L2, single-env, over-recorded

**Decision:** the per-task source demo set (target 10) is recorded on the **L2** env
(widest pose distribution: cup pose + goal varied, default cup cyl_m_red) with
`--num_envs 1`, over-recording to ~15 and keeping the first 10 that survive
annotate's replay re-check.

**Why:** (a) L2 sources give the Mimic NN-selection (k=3) spatially diverse
references usable for every level; L0-only sources would be 10 near-identical
trajectories. (b) Single-env recording matches annotate_demos.py's hardcoded 1-env
replay — avoids PhysX batch-size divergence (review finding). (c) Over-recording
absorbs the residual replay non-determinism upstream documents even single-env.
Provenance control holds: the SAME surviving sources feed generation for all levels.

**VERIFY at G2/G3:** annotate yield printed as "Exported X (out of Y)"; if <10
survive, record more sources rather than loosening checks.


## D10 — 2026-08-17: Keep table_cam FOV despite corner clipping (~0.25% of episodes)
Full-sweep QA (red-pixel segmentation, frame 0, all 400 eps of L1 and L2): 1/400
episodes start with the cup fully outside the table_cam view, 5-6/400 marginal
(<30 px), all at the far corner x~0.65/y~-0.25 of the cup range. Identical sampler
and camera across L1/L2/L3 => the effect is level-uniform and train/eval-matched
(frozen eval sets sample the same distribution), so cross-level comparisons stay
fair. Wrist cam covers the approach; n_obs_steps=2 policies re-acquire the cup
early in the rollout. Regenerating all visuomotor data with a wider FOV (~3 h +
re-QA) would shrink pixels-per-object for every episode to fix a 1-in-400 corner.
KEEP camera + datasets; report as a known characteristic in the paper's setup
section. Evidence: ops/qa/L1_visibility.png, L2_visibility.png.

## D11 — 2026-08-17: Frozen eval sets = seed protocol + committed state snapshots
Benchmark per level = (env cfg, protocol.json seeds). To make silent env-cfg drift
detectable, we additionally commit per-level initial-state snapshots
(configs/eval_sets/{L}.json): 10 batches x 20 envs of cup/goal poses from
env.reset(seed=5000+b) on the STATE env. L0-L2 standard eval = batches 0-4
(100 eps), headline rerun = 0-9 (200 eps). L3 per D2 pools per-variant runs:
standard = batch 0 on each of the 10 sub-envs (200 eps), rerun adds batch 1
(400 eps). Any future eval run can diff its reset states against the snapshot.

## D12 — 2026-08-17 (FINAL after asset recon): Task 2 design — drawer + stow
Task: Franka opens a closed drawer, then picks a tabletop object and stows it
inside; success = object inside the drawer cavity (pos in drawer-frame box) at
episode end, drawer opening >= 15 cm. Scripted expert = 4-phase SM (grasp handle
-> pull open -> grasp object -> place in drawer); Mimic subtasks object-centric:
handle/cabinet ref for 1-2, object ref for 3, drawer ref for 4.
Generality ladder mirrors Task 1 semantics (same 4-level structure, same N grid,
same 80k-step training, same eval protocol):
- T2-L0: all fixed (cabinet pose, object pose, one object)
- T2-L1: + object XY randomized on table (range set after workspace check)
- T2-L2: + cabinet pose randomized (XY few cm + yaw range — the drawer IS the
  goal, so this is the goal-randomization analog)
- T2-L3: + object variants: 2 box sizes x 5 colors = 10 sub-envs per D2 pattern
Provenance control identical: same sources + generator settings across levels;
gen SR per level reported. Object = procedural box (fits drawer; D1 analog).
User can veto/adjust before datagen starts.

D12 addendum (asset recon, 2026-08-17): base scene = stock cabinet layout (Franka
at origin on ground plane, Sektion cabinet at (0.8,0,0.4) yaw-180, drawer_top as
the target drawer; travel limit ~0.40 m). Stow object rests on the cabinet's top
surface (exact height from empirical inspection). Facts driving implementation:
- Sektion USD is Nucleus-only (Isaac/Props/Sektion_Cabinet/), nothing local ->
  cluster needs the subtree mirrored; add COG_ISAAC_ASSET_ROOT seam rooted at
  ISAAC_NUCLEUS_DIR (cup_place's COG_ASSET_ROOT is rooted at ISAACLAB_NUCLEUS_DIR).
- No isaaclab_mimic env for any articulated object exists -> greenfield subclass
  of FrankaCubeStackIKRelMimicEnv per cup_place pattern.
- VERIFY (d) CLOSED (2026-08-17): the base get_object_poses enumerates RIGID
  objects only — the cabinet Articulation was silently absent from
  datagen_info.object_pose. Fixed by overriding get_object_poses in
  FrankaDrawerStowIKRelMimicEnv to append the cabinet root pose from the
  articulation state (same root_pose layout). Annotation now records all four
  refs; generation with object_ref="cabinet" is unblocked.
- Stock open_cabinet_sm.py: world-frame offsets (break under cabinet yaw ->
  compose in handle frame), single -1.5 cm pull (insufficient -> ramped/segmented
  pull to >=0.2 m), IK-Abs driver conventions match our converter.
- Control at 20 Hz (decimation=5, dt=0.01) like cup_place, NOT the stock 60 Hz.
- ee_frame target order: keep cup_place's (end_effector, right, left).
- Success: drawer_top_joint >= 0.15 AND object inside drawer-frame cavity box
  AND gripper released; timeout ~35 s (longer than cup_place: two grasps).

## D13 — 2026-08-17: T2 box sizes bounded by the stow-corridor feasibility
Empirical (18 debug runs): the drawer-stow descent requires the carried box's
trailing edge to clear the drawer wall from the arm's carry equilibrium
(x~0.32 at the required height) at the reliably achievable opening (~0.30
after post-release drift). This caps the box half-width at ~0.024 m. L3 sizes
set to 4.0/4.8 cm (was 4.5/5.8); DEFAULT_BOX = box_m_red (4.8 cm). The 20%%
size spread keeps the variant axis meaningful. Descent gate now checks the
physical clearance condition (handle-relative, pull-direction projected,
variant-aware) instead of target-distance proxies.

## D14 — 2026-08-17: Franka on a 0.20 m pedestal for drawer_stow
Ground-mounted, the arm's carry envelope tops out at z~0.82 at the radii the
stow needs, while the drawer walls top at 0.785 and the handle-to-wall offset
is asset-fixed at 13.15 cm — the (pull depth, wall clearance, carry height)
window is structurally empty by ~2 cm no matter the pull target (18 debug-run
constraint map). A 0.20 m pedestal moves every cabinet interaction into
mid-workspace: wall crossing gains ~10 cm clearance at any x, the deep pull
becomes unnecessary (target back to 0.28), and the sag/wedge failure class
disappears. Scene change only for Task 2; Task 1 unaffected.

## D14-revised — 2026-08-17: pedestal height 0.08 m, not 0.20
The 0.20 m pedestal mirrored the arm's wrist branch at the handle (j5 sign
flip, j6 pinned) and broke every previously proven phase (runs 22-28: approach
stalls, triple limit pins, violent reconfigurations, paths crossing the open
drawer's volume). The minimal 0.08 m lift raises the carry ceiling past the
stow requirement (~0.82 -> ~0.90) while keeping the arm in the SAME kinematic
branch as the fully-proven ground trajectory. First full success with this
geometry: 650-step episode, traverse tracking 1.2 cm at z 0.879, wrist
mid-range throughout. Key craft rules extracted for the paper's method notes:
ramp every long translation, SLERP orientation only where the branch needs
guiding (obj leg yes, handle approach no), never route paths through the open
drawer's swept volume, and treat wrist-branch selection as set by the FIRST
large motion after reset.

## D15 — Git history rewritten once, before the first push (2026-08-17)

**Decision:** strip blobs >50 MB from the entire history via `git filter-repo`, and
keep training weights out of git permanently (`experiments/runs/**/*.safetensors`).

**Why:** the G4 smoke checkpoint had been committed (3 GB across two safetensors
files). GitHub refuses any blob >100 MB in pushed history, so publishing the repo was
impossible without a rewrite. Weights are regenerable and belong on disk/the cluster;
git holds code, configs, docs, small JSON provenance and eval sets.

**How it was made safe:** full `.git` + checkpoint backup under
`data/_prepush_backup/`; rewrite performed in a scratch bare clone rather than in
place; verified that original and cleaned HEAD differ by exactly the two stripped
paths with every other blob hash identical and all 61 commit subjects preserved; the
pre-rewrite branch is retained locally as `main-prefilter`.

**Cost accepted:** commit hashes cited in journal entries before 2026-08-17 13:20 do
not resolve on `main`. Messages are unchanged, so `git log --grep` still finds them.
This is a one-time cost paid at the first push, when the repo had no other clones.

## D16 — Generation SR is computed from episode counts, never from logs (2026-08-17)

**Decision:** report Mimic generation success rate as
`n(<name>.hdf5) / (n(<name>.hdf5) + n(<name>_failed.hdf5))`, counting HDF5 episode
keys.

**Why:** the generator's progress line is buffered by carb and the final flush is
lost at shutdown, so the last line in the log understates the true count — by 19
demos on T2_L3v00 (21/74 visible vs 40/120 actual). Every gen SR quoted before this
entry was scraped from logs and is therefore ~0.4 points low (L3 variants worse). The
`_failed` companion that has been an operational nuisance all along is in fact the
exact attempt ledger.

**Consequence:** `_failed.hdf5` files must not be deleted until their episode count
is recorded in `docs/timings.md`.

## D17 (RESOLVED 2026-08-17 by user) — L3 has 2 geometries, not the specified mesh set

**Finding:** `L3_VARIANTS` in both tasks is 2 sizes x 5 colours. Colour has no
physical effect, proven by identical generation SR and attempt counts within each size
group across independent runs (T1: 40/45 x5 then 40/46 x5; T2: 40/120 x4 so far). The
plan specifies L3 = 4 mug meshes x 5 colours x scale 0.9-1.1; D1 deferred the meshes
pending grasp/render QA and that QA never ran before P3 closed.

**Why it matters:** L3 is the study's object-generality axis. As built it varies
appearance plus a 10 % (T1) / 20 % (T2) scale step. A small measured data cost at L3
would then be ambiguous between "object generality is cheap" and "this axis barely
varies anything".

**Timing:** decide before P6. No training has run; regenerating L3 costs ~26 min (T1)
and ~4 h (T2) now, and is unaffordable after the matrix.

**Options:** (a) add mug meshes and regenerate L3 + re-freeze its eval sets;
(b) leave and stay silent (rejected — misrepresents the axis); (c) keep the data,
describe L3 as "appearance + mild scale", and add a separate L4 geometry level after
the Task-1 matrix. **Recommended: (c)** — additive, touches no frozen benchmark, and
separates appearance cost from geometry cost, which is a stronger result than either.

**Status:** awaiting user decision. Do not regenerate L3 or edit `L3_VARIANTS` until
it is made.

**RESOLUTION (2026-08-17, user):** keep all existing data and the ladder exactly as
built; record the missing shape/kinematics axis as a **limitation** rather than
retrofitting it. Rationale accepted: the generator's object-centric rigid-transform
assumption is what makes pose generality cheap and geometry generality expensive, so
the gap is a property of the method, not an oversight — reporting it with our measured
per-axis costs is a contribution, while bolting on a weak geometry axis would cost days
and buy an ambiguous result. Drafted text lives in `paper/limitations.md`.

Follow-ups explicitly NOT taken: mug meshes (needs per-shape source demos + expert
retuning, re-imports the yaw/handle grasp confound), randomized drawer starting
position (needs a drawer-frame reference for subtasks 1 and 3 plus a delta-based
`drawer_opened` signal). If L3 is ever regenerated for an unrelated reason, widen the
box edge range within the cube family (3.5-5.5 cm, capped by the D13 stow corridor)
while it is being rebuilt — free at that point.

## D18 — 2026-08-17: L3 standard eval uses the diagonal (variant v <- batch v)

**Problem found while freezing the T2 eval sets.** The L3 protocol as written was
"batch 0 on each of the 10 sub-envs, pooled (200 eps)". But L3 variants differ only in
object size/colour and **share the pose RNG stream**, so batch 0 is the *same* 20 poses
in all ten variants: 200 episodes containing only **20 distinct object poses**. Every
other level's 100-episode standard eval has 100 distinct poses. Verified directly in the
frozen snapshots for both tasks (T1 and T2 identically).

Why it matters: pose is the dominant difficulty axis, so 200 pose-correlated episodes
overstate statistical power badly (naive binomial SE ~3.5 points, but only ~20
independent spatial draws), and it makes L3 non-comparable to L0-L2 on exactly the axis
the study measures — inside the headline cost-of-generality curve.

**Decision:** L3's standard eval pairs **variant v with batch v** (the diagonal). Same
200 episodes, same ten appearance variants, but **200 distinct poses**. The frozen
snapshots already contain 10 batches for every variant, so this is a change to *which
committed rows the protocol reads*, not a regeneration — rule 8 is respected and the
snapshot data is byte-identical (verified). No evaluation had run yet (P6 blocked on
G0), so nothing is invalidated.

Also recorded: 10 batches x 20 envs = 200 distinct poses is the *total* pose supply at
L3, so 200 episodes is its maximum spatial coverage; `headline_rerun` at L3 therefore
equals its standard eval. Cross-level headline comparisons must use each level's
**200-episode** set (L0-L2 batches 0-9, L3 diagonal), which have equal spatial coverage.

**Corrects an earlier error:** the G3 entry claimed "L3 sub-envs draw independent
streams". They do not -- same seed gives the same poses across variants. That claim was
wrong and is retracted here.

## D19 — 2026-08-17: Task 3 = push a puck 20 cm to a target disk, single Mimic subtask, synthetic push frame

**The binding constraint (from recon + `data_generator.py:52-83`):** Mimic expresses each
subtask's EEF trajectory relative to ONE 4x4 reference pose and rigidly re-applies it. A
push is intrinsically a TWO-frame relation — it depends on the object pose AND the goal
pose. Anchoring on the object reproduces the source demo's object->goal *vector*, so a
moved goal is simply not reached; anchoring on the goal loses the approach. Neither body
alone works. Everything below follows from designing around that.

**Design:**

1. **Synthetic push frame as the reference.** `get_object_poses` publishes a derived frame
   `push_frame`: origin at the puck centre, yaw pointing from puck to target, roll/pitch
   stripped. The source stroke is then "advance along +x of this frame", and re-applying it
   rigidly in a new scene pushes along the *new* puck->target direction. Direction adapts
   for free; distance does not, which forces item 2.

2. **Constant stroke length: |puck - target| = 0.20 m at every level.** Rigid transforms
   carry no scale, so the stroke baked into the source demo is the stroke you get. Holding
   the distance fixed makes it exactly right everywhere. The task is therefore honestly
   "push the puck 20 cm to the marker"; the generality axes vary *where* and *which way*,
   not how far. Success radius 5 cm absorbs contact slip.

3. **ONE subtask for the whole episode.** The recon's rules make multi-subtask decomposition
   actively dangerous here: non-final term signals must be latched, monotone and false at
   t=0 (a signal true at t=0 crashes `DataGenInfoPool._add_episode` with a bare IndexError),
   boundaries are taken from the first NONZERO diff so contact/region predicates that chatter
   pick the wrong step, and Mimic inserts a free-space interpolation between subtasks that
   would jump the EEF mid-stroke. A single segment anchored on the push frame has no
   boundaries to get wrong and keeps the stroke intact. Fallback if Mimic rejects a 1-element
   `subtask_configs`: two subtasks sharing the same `object_ref` split at contact.

4. **Gripper closed as a blade, but NOT constant.** Keep the 7-dim action with the gripper
   last (`actions_to_gripper_actions` is a hard-coded `actions[:, -1:]` slice). The expert
   starts open at reset and closes during the approach, then holds closed. Rationale beyond
   realism: LeRobot's diffusion config MIN_MAX-normalizes state and action, so a gripper
   channel that never changes would make those dims degenerate across the pool.

5. **Object = flat puck, not a tall object.** Measured empirically (probe, 2026-08-17): a
   closed-gripper blade pushed the T1 cup 20 cm, but the cup TIPPED to 90 deg partway and
   slid on its side. A low, wide puck cannot tip. Procedural `CylinderCfg`, so no cloud
   assets (D1's reasoning).

6. **Contact height from measured geometry, closed-loop.** Also from the probe: the fingertip
   body sits **+4.5 cm above** the `ee_frame` TCP, and an open-loop descent stalls — the first
   probe attempt pushed air 8 cm above the puck. The SM descends until the *measured*
   fingertip height reaches contact height, exactly as T2's expert gates on physical
   conditions rather than commanded targets.

7. **Success = puck centre within the disk AND settled.** No `released` clause (impossible
   with a closed pusher). `settled` is mandatory, not cosmetic: generation OR-latches success
   across every timestep, so without it a puck sliding *through* the target and out the far
   side would be recorded as a success.

8. **Levels:** L0 everything fixed; L1 puck position randomized (target follows at 20 cm,
   fixed bearing); L2 + target bearing randomized around the puck; L3 + puck geometry
   (radius/height) and colour. **L3 is the point of interest** — with no grasp offset to
   invalidate, this is the one task in the study that may carry a real geometry axis
   (paper/limitations.md entry 2). Radius variation shifts the contact standoff, which the
   5 cm success disk should absorb; to be measured, not assumed.

**Named risks:** single-reference open-loop stroke means contact slip is uncorrected within a
segment (the reference is sampled once at segment start); IK-Rel action scale 0.5 plus one
waypoint per step means the EEF lags under contact friction; MIN_SEPARATION must exceed a
full stroke so the puck cannot start inside the target region.

## D19-addendum — 2026-08-18: Task 3 axis ranges set by measurement; design validated

The D19 design survived contact with the simulator; the two range choices in it did not, and
were replaced with measured ones (evidence in docs/journal.md 2026-08-17/18):

- Bearing range +-40 deg -> **+-25 deg**: expert SR 94-95 % inside 25 deg, 75 % beyond.
- Puck radii (0.032 ... 0.058) -> **(0.032, 0.035, 0.038, 0.042, 0.045)**: expert SR falls
  monotonically with radius above ~0.045.
- Success radius, episode budget and the recording gate also moved: 30 s -> 40 s episodes,
  and source demos are recorded against a **2 cm** success radius while the level keeps its
  5 cm gate, because recording at 5 cm produced templates with a median 5.01 cm placement
  error — a perfect 20/20 expert score that was hiding systematically unusable data.

**Validated:** Mimic accepts a one-element `subtask_configs`, so the single-subtask design is
legal; the synthetic `push_frame` reproduces strokes at new bearings and positions; and the
resulting generation SR (88.5-98.5 %) is the highest of the three tasks. Every load-bearing
assumption in D19 held.

## D22 — 2026-08-19: cluster training env mirrors the local stack exactly, on torch cu128 over a CUDA-12.2 driver

**Decision.** The cluster training env (`$WORK/cog/miniforge3`, env `cog_lerobot`) is pinned
version-for-version to the locally verified stack: python 3.11, **torch 2.7.0+cu128**,
torchvision 0.22.0+cu128, lerobot 0.4.4, numpy 1.26.4, av 15.1.0, `video_backend=pyav`.
Install order is torch-from-cu128-index first (with deps), lerobot second.

**Why mirror rather than take whatever resolves.** Training happens on the cluster and eval
happens locally in `cog_isaac`, so a checkpoint must cross environments. Keeping one set of
pins on both sides removes an entire class of "trained fine, loads wrong" failure, and it is
free: LeRobot 0.4.4 requires only `torch<2.11.0,>=2.2.1`, so nothing forces a divergence.
The numpy 1.26.4 pin is not needed on the cluster (its reason, D3, is an Isaac Kit segfault
and there is no Kit in the training env) but is kept anyway for parity, since it costs
nothing and makes the two envs diffable.

**The open assumption.** Leonardo's A100 nodes run driver **535.274.02, CUDA 12.2** (measured
2026-08-19, node lrdn2752), which is *older* than the cu128 wheels' 12.8. This is expected to
work through CUDA minor version compatibility -- any 12.x runtime on a >=525 driver -- and
sm_80 is compiled into the cu128 binary, so no PTX JIT is involved. But it is an assumption,
not a verified fact, so it is **checked in the G5a smoke** with `torch.cuda.is_available()`
and a real GPU matmul before any 8 h run is submitted.

**Fallback, decided in advance so it is not improvised under time pressure:** if the cu128
wheels fail on this driver, drop the cluster env to **torch 2.7.0+cu126** and leave every
other pin alone. This has no scientific cost: a checkpoint is a build-independent
safetensors directory plus JSON, so a cu126-trained checkpoint loads unchanged in the local
cu128 eval env. Only the wheel's bundled CUDA differs, not the math, the seed, or the data.

**VERIFY:** cu128-on-535 GPU matmul (G5a smoke). If it fails, switch to cu126 and note it here.

## D23 — 2026-08-19: batch 64 / lr 1e-4 frozen by measurement, and the decode bottleneck is fixed by a torchcodec pin, not a LeRobot upgrade

**Context.** P5 assumed training would be GPU-bound and instructed us to "scale batch up as far
as A100-64GB VRAM/throughput allow (e.g. 64->128->256)", scaling LR by sqrt(batch ratio). The
G5a smoke on one A100 (job 52878355) refuted the premise outright.

**Decision 1 — batch 64, lr 1e-4, frozen for every cell.**

| batch | steps/s | samples/s | peak VRAM | median GPU util |
|---|---|---|---|---|
| 64 | **0.962** | 61.5 | 13.5 / 64 GiB | 0 % |
| 128 | 0.862 | 110.3 | 14.5 GiB | 0 % |
| 256 | 0.385 | 98.7 | 17.1 GiB | 0 % |

VRAM never exceeds 27 % of the card, so the question the plan asked ("how large a batch
fits?") has no bearing on anything. Median GPU utilization is 0 % at every batch size and
samples/s is flat, i.e. the dataloader sets throughput. Because the protocol fixes 80k
**steps**, the smallest sensible batch minimises wall-clock; larger batches are strictly worse
(22 h at 128, 58 h at 256, the latter not even fitting the 24 h walltime). lr stays at 1e-4,
which is also `DiffusionConfig.optimizer_lr`'s default, so no sqrt scaling is applied.

Batch is *not* re-tuned per cell: at fixed steps a different batch means a different sample
budget, so comparing cells would confound data-cost with batch size. One value for all 24.

**Decision 2 — fix the decode path via `torchcodec==0.5`, do NOT upgrade LeRobot.**

Root cause, from the 0.4.4 source rather than guesswork: `decode_video_frames_torchvision`
constructs a `torchvision.io.VideoReader` **per call** on a single 82,916-frame / 43 MB mp4 and
closes it again -- ~128 container opens per step at batch 64. Only the torchcodec path has a
decoder cache (`_default_decoder_cache`). Measured cost of that difference: **25.5 ms vs
0.57 ms per frame fetch on a compute node (45x)**; 31.9 ms vs 0.29 ms locally (108x).

Two ideas were killed by measurement before being implemented:
- *Re-encode all-intra so seeks are cheap*: `ffprobe` shows the videos are **already GOP 2**
  (41,058 keyframes in 82,916 frames), so there is nearly nothing to win. The cost is opening
  the container, not seeking within it.
- *Throw CPUs at it*: billing scales linearly with allocated cores (`billing=32` for 32 cores
  vs `billing=8` for 8, confirmed in AllocTRES), so more workers costs proportionally more and
  cannot beat removing the work.

**Why not LeRobot 0.5/0.6** (the user asked, and asked for it to be measured first):
0.5.0's `datasets/video_utils.py` is byte-identical to 0.4.4's -- **zero** gain, since the v0.5
speedups are all record/encode-side. 0.6.x does rewrite the pyav path (PR #3588) and measures
**3.7x** on our own L0 data, but it requires py>=3.12, which breaks the single-env property
Isaac Sim 5.1 imposes (G1b); it silently flips five diffusion defaults (PR #3202); and its
checkpoints need two config keys stripped before 0.4.4 can load them. Fixing one pin is a
larger win (45x) at a fraction of the risk.

**The pin itself is the subtle part.** lerobot 0.4.4 requires `torchcodec>=0.2.1,<0.11.0`,
which resolves to 0.10.0 -- built against torch 2.10 while we pin torch 2.7.0. That fails as
`libtorchcodec_core6.so: undefined symbol: _ZN3c1013MessageLogger6stream...`, a **libtorch ABI**
symbol, not an ffmpeg problem; the `libavutil.so.56` line everyone latches onto is merely the
last of five descending ffmpeg probes. Worse, **0.7.0 imports cleanly and still does not work**
-- it fails on first decode with `no fallback function is registered for schema
torchcodec_ns::_convert_to_tensor`. So "the import succeeded" is not evidence that a native
extension matches your torch; only a decode is.

**Verified, not assumed:** torchcodec 0.5 + conda-forge ffmpeg 6 + `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`
decodes our data and returns frames **bit-identical to pyav** (40 random fetches, max abs pixel
difference 0.0, `BACKENDS_IDENTICAL`). This check was mandatory, not optional: the torchcodec
path selects frames by `round(ts * average_fps)` with `seek_mode="approximate"` while the pyav
path matches by timestamp, so a silent off-by-one would have corrupted every training batch
with nothing in the pipeline complaining. `scripts/dev/decode_bench.py --compare-backends`
keeps the check reproducible.

**Scope guard.** The do-nothing baseline was already acceptable -- 10-12 h per run, ~245 GPU-h
for the 24-cell matrix, inside the plan's envelope -- so this was pursued as a bounded
optimisation, not a blocker, and would have been abandoned in favour of pyav had the frame
check failed. Expected after the fix: `data_s` collapses toward `updt_s` (0.071 s), i.e.
~1.5-2 h per run and roughly 40-60 GPU-h for the matrix.

**Also pinned as a side effect:** `--policy.use_separate_rgb_encoder_per_camera=false` and
`--policy.do_mask_loss_for_padding=false` in `configs/train/diffusion_base.sh`, at 0.4.4's own
defaults. They are inert today; they exist so that the two architecture defaults 0.6 flips can
never change this study's model silently. `pretrained_backbone_weights` is deliberately NOT
passed (risk of draccus decoding the string "null") and is asserted against the calibration
run's saved `config.json` instead.

**VERIFY:** (a) `data_s` with torchcodec in a real training loop -- job 52896093; (b) resume
across requeue -- `slurm/smoke_resume.sbatch`; (c) `pretrained_backbone_weights: null` in the
calibration run's `checkpoints/*/pretrained_model/config.json`.

## D24 — 2026-08-19 (USER DIRECTIVE): full-scale cells evaluate the LAST checkpoint only; the checkpoint comparison is run once

**Decision (user, 2026-08-19).** "I only want to test once how the checkpoints compare during
eval, and for full-scale run always only eval last checkpoint."

So the protocol changes from the plan's *"evaluate last 3 checkpoints (40/60/80k); primary metric
= best-of-last-3"* to:
- **one** cell is evaluated at 40k, 60k **and** 80k, purely to characterise how much the choice of
  checkpoint matters;
- **every other cell** is evaluated at **80k only**, and the headline metric is
  **last-checkpoint SR**.

**Why this is a good trade.** After the G5a decode fix, training a cell costs ~2 h while eval is
~100 rendered rollouts per checkpoint -- so evaluation, not training, had become the binding
wall-clock constraint of the whole study. This cuts the eval workload from **72 checkpoint-evals
to 24** (plus 2 extra for the comparison), i.e. a 3x reduction on the critical path, for the price
of one mitigation.

**What is given up, stated plainly.** `best-of-last-3` was the plan's hedge against single-seed
noise (one seed per cell is itself a user directive). Reporting the last checkpoint instead means
that hedge is gone: if training is non-monotone near the end, a cell can look worse than it is,
and with N varying across the grid that noise lands directly on the demos-vs-success curve we are
trying to measure. `best_of_last_3` is a strictly optimistic statistic, so switching to
last-checkpoint also removes a mild upward bias -- arguably making the numbers *more* honest, just
noisier.

**How the risk is controlled rather than ignored.** The one-time comparison is not a formality --
it is the evidence that decides whether the simplification is safe, and it gets **reported as a
finding**, not just used to justify the choice. Concretely: if the 80k SR sits within the Wilson
interval of the 40k/60k SRs, last-checkpoint-only is sound and we proceed. If 80k is
**systematically worse** than an earlier checkpoint, that is itself a result (late-training
degradation) and it comes back to the user before the matrix is evaluated, because it would mean
the headline metric is measuring an artifact.

**Which cell hosts the comparison:** `t1_L0_n25_s0` -- the G5a calibration cell, which will have
20k/40k/60k/80k checkpoints already on disk at no extra training cost.

**Deliberately NOT changed:** `--save_freq=20000` stays. Intermediate checkpoints cost ~1 GB each
on a 3 TB area, they are what makes requeue/resume work (G5a part 3), and they keep the option of
revisiting any cell's trajectory later. Saving them and choosing not to *evaluate* them are
different decisions.

**Implementation:** `slurm/eval.sbatch TASK LEVEL NDEMOS [STEP...]` now defaults to `080000`;
the comparison is requested explicitly as `... T1 L0 25 040000 060000 080000`.
`experiments/registry.csv` keeps its `sr_40k`/`sr_60k` columns -- they will simply be empty for
every cell except the comparison one, which is honest rather than untidy.

**Budget effect:** T1 evals ~25 -> ~9 GPU-h; all three tasks ~75 -> ~27 GPU-h.

## D25 — 2026-08-19: T1 evaluation runs LOCALLY on the 4090; cluster eval deferred pending CINECA

**Decision.** Evaluate Task 1 on the local RTX 4090 using `scripts/ops/run_local_eval.sh`. Cluster
eval is deferred, not abandoned: the image works and is kept, but Isaac Sim cannot get a GPU inside
Singularity on Leonardo and the remaining question is a site one.

**Why this is now clearly right rather than a concession.** When the plan chose "local eval" as the
*fallback* it was worth ~2-3 days of 4090 wall-clock, which justified building a container to escape
it. Two things changed:
- **D24** cut evaluation from 72 checkpoint-evals to **24+2**.
- The local path is **measured**: ~14 min per checkpoint sharing the GPU with a foreign job, so all of
  T1 is **~5-6 h of local wall-clock and 0 GPU-h of grant**.
So the fallback is now roughly a third of the cost that motivated the container, while the container
still needs an unresolved Vulkan issue solved.

**What blocks the cluster path** (full evidence in the journal, 2026-08-19 22:15): Kit boots inside
`cog-env-5.1.0.sif`, IsaacLab parses our env cfg and the scene builds -- but `vkCreateInstance` fails
with `ERROR_INCOMPATIBLE_DRIVER`, so Kit renders on the CPU (274 s for a ~20 s scene). Five
hypotheses were disproved by measurement, the host is demonstrably Vulkan-capable, and Kit's bundled
loader ignores the Vulkan loader environment variables.

**Consequence for the study.** None for validity: eval placement does not change what is measured.
The frozen eval sets, protocol and metric are identical in both places, and a checkpoint is a
build-independent safetensors directory. It changes only wall-clock and where the artifacts land.

**Revisit if** Tasks 2-3 make eval throughput binding (48 more cells => ~12 h local), or if CINECA
confirms Vulkan works in Singularity there. Restart from the *reference-warning diff* and the ticket,
not from another round of container guesses.

**VERIFY:** none outstanding. The local path is proven end-to-end -- three checkpoints evaluated,
SR 0.97/0.95/0.98, D24 discharged.

---

## D26 -- Separate RGB encoder per camera (2026-08-19)

**Decision.** `use_separate_rgb_encoder_per_camera=true` for the whole study. Each camera
(`table_cam`, `wrist_cam`) gets its own ResNet18 + spatial-softmax encoder instead of one encoder
applied to both. Frozen in `configs/train/diffusion_base.sh`; the T1 matrix was launched under it.

**Why.** User directive. It is also the defensible default for this study: the two views are not
samples from one visual distribution -- a fixed table camera and a wrist camera mounted on the
moving end-effector have systematically different statistics, scale and motion. Sharing one encoder
forces a single feature extractor to serve both, which is a stronger inductive-bias assumption than
this study needs to make. It is additionally what lerobot itself now defaults to from 0.6.0 onward.

**Cost, measured before launch** (`DiffusionPolicy` built both ways, 2 cams @128x128, crop 112,
state 9, action 7): encoder params **11,197,088 -> 22,394,176 (exactly 2.00x)**, total
**266,798,375 -> 277,995,463 (+11.2M, +4.2%)**. The **U-Net is byte-identical** (255,601,287 params
both ways) because `global_cond_dim` adds `feature_dim * num_images` in *both* branches
(`modeling_diffusion.py:176-182`) -- untying the encoders does not widen the conditioning vector.
fwd/bwd verified at batch 64 (peak 3.44 GiB allocated), so VRAM was never a question.

**Consequences accepted.**
1. **The G5a calibration run is off-architecture.** `t1_L0_n25_s0` (job 52899856, SR
   0.97/0.95/0.98) was trained with the shared encoder. It is renamed to
   `t1_L0_n25_s0_sharedenc` in the registry (status `superseded`) and its cluster checkpoints are
   renamed, not deleted -- it remains the evidence that discharged D24 and showed L0 saturating at
   N=25. Those two findings now rest on shared-encoder evidence until re-checked on the new
   architecture, which is free: the matrix saves checkpoints at 20/40/60/80k anyway, so D24 can be
   re-verified with local eval time only, no GPU-h.
2. **The 2.0 GPU-h/cell timing basis is provisional.** It was measured at 11.2 steps/s with one
   encoder. The loop was decode-bound (GPU util ~0% at every batch size, D23), so a 4.2% parameter
   increase is not expected to move wall-clock much -- but that is a prediction, not a measurement.
   Walltime per cell was raised 06:00:00 -> 12:00:00 as insurance; on Leonardo this is free,
   because billing is cores x ELAPSED, not the reservation.
3. **N=25 at L0 is re-run** rather than reused, so all 24 cells share one architecture (rule 7).

**The dangerous part, and the guard added for it.** `train.sbatch` resumes with
`--config_path=<ckpt> --resume=true`, which is mandatory at 0.4.4 (`configs/train.py:89-95`). On
that path the config comes ENTIRELY from the checkpoint and every other CLI flag is ignored. So
flipping this flag while `t1_L0_n25_s0` still had 80k-step shared-encoder checkpoints on `$WORK`
would have made that cell silently keep training the OLD architecture while the registry, the log
line and the frozen config all said otherwise -- a wrong result with no error anywhere. Two things
now prevent it: the stale checkpoints were renamed aside, and
`scripts/ops/assert_resume_config.py` re-checks every `--policy.*` value plus `batch_size` in the
checkpoint against `COG_DP_FLAGS` before any resume, aborting the job (exit 2) on mismatch. Tested
both ways against the real stale checkpoint: 1 mismatch caught with the new flag, 16 values clean
with the old one. This generalises -- it now catches *any* post-hoc edit to the frozen config, not
just this one.

**Also fixed while here:** `sync_up.sh` hardcoded the main checkout as its source, so a sync run
from a git worktree pushed the WRONG (old) config to the cluster while the local branch showed the
new one. It now honours `COG_REPO`, and prints the source repo it is using.

**VERIFY:** (a) **DONE 2026-08-19** -- first checkpoint (`t1_L0_n400_s0/checkpoints/020000`) reports
`sep_enc=True group_norm=True horizon=16 n_act=8 batch=64 steps=80000`, i.e. exactly the frozen
config; no checkpoint anywhere reports `False`. Verified from the saved `train_config.json`
artifact, not from the config we shipped and not from an exit code (D6).
(b) **DONE 2026-08-19, REVISED 2026-08-20** -- the live read at step 7,200 said 9.45 vs 11.2 steps/s
(-16%, 2.35 h/cell), but that was inside warm-up. Settled numbers over all 24 completed cells:
median **2.01 h**, mean 2.14 h, total 51.3 GPU-h -- so the median cell matches the shared-encoder
baseline (2.00 h) and the mean penalty is **+6.8%**, driven by a few slow cells rather than the
architecture. The "-16% throughput" claim is withdrawn; see `docs/timings.md`.
(c) still open -- re-check D24 (last-vs-best checkpoint) and L0 saturation on one new-architecture
cell. Costs no GPU-h: checkpoints at 20/40/60/80k are saved anyway, so this is local eval time only.

## D27 -- 2026-08-20: every L3 dataset in the study has ~9x redundant initial poses (seeding bug); the L3 arm is regenerated

**Trigger.** The user asked whether L3's low success rate could be a *data-generation* artifact --
Mimic failing on cases that eval still tests. Investigating that produced a different and worse
answer: L3's demo axis is not a demo axis.

**Finding.** `scripts/ops/gen_L3_wave.sh` (and the T2/T3 equivalents) call
`datagen/vendored/generate_dataset.py` once per L3 variant, ten times per task. Upstream takes the
generation seed **only** from `env.cfg.datagen_config.seed` and exposes no CLI override, and our task
cfgs never set it -- so all ten calls seeded identically and replayed the same initial-pose stream.
Measured with `cog.analysis.gen_bias` (new):

| level | demos | UNIQUE initial poses |
|---|---|---|
| T1/T2/T3 L0 | 400 | 1 (correct -- fixed by design) |
| T1/T2/T3 L1 | 400 | 400 |
| T1/T2/T3 L2 | 400 | 400 |
| **T1 L3** | 400 | **43** |
| **T2 L3** | 400 | **45** |
| **T3 L3** | 400 | **48** |

The variants are not merely correlated, they are identical: v00-v04 share a byte-identical pose set
(maxdiff 0.0) and v05-v09 share another; the two groups differ only because the s- and m-cylinder
runs rejected a different number of attempts (5 vs 6). Corroborating fingerprint in
`gen_stats.csv`, which we had already committed without noticing: every T1 L3 variant reports the
same 45 attempts, 5 failures, and identical min/mean/max episode length -- ten independent runs do
not coincide like that.

**Consequences.**
1. **L3's N axis is inflated ~9x.** At the nominal N=400, L3 holds 40 unique poses rendered in 10
   appearances. So L3 at N=400 is comparable to L2 at N=40 (SR ~0.68), not L2 at N=400 (SR 1.00).
2. **The "L3 ceiling" finding is void as stated.** The plateau at ~0.45-0.53 across N=50..400 is a
   *pose-coverage* ceiling: the policy never sees more than ~40 initial states no matter how many
   demos it is given, while eval samples 200 fresh ones. The earlier reading -- "L3 has a ceiling,
   not a data cost, so the N=800 arm is unwarranted" -- was drawn from this artifact and is retracted.
3. It also explains L3's training loss sitting **4x below** L0-L2 (0.019 vs 0.096/0.074/0.072) and
   still descending 8.8%/20k steps at 80k while the others are flat: with ~9x redundancy the model
   memorises 40 trajectories instead of fitting 400, so it drops through L2's aleatoric floor.
4. It does **not** affect L0/L1/L2 in any task: those are single generation runs and hold 400 unique
   poses each. Only the 6 L3 cells per task are implicated.

**Decision.** (a) `--seed` added to the vendored generator (default None = upstream behaviour), and
all three wave scripts now pass a per-variant seed offset by task (T1 1000+v, T2 2000+v, T3 3000+v),
kept away from the eval seeds 5000-5009 so training and eval poses stay disjoint. The effective seed
is printed to the log for provenance. (b) L3 is regenerated and retrained for all three tasks (18
cells, ~40 GPU-h). (c) The existing L3 runs are **kept, not discarded**: same demo count, ~9x
redundant poses, they are now a clean pose-diversity ablation at fixed N -- which is worth reporting
in its own right. Their results are renamed `*_poseredundant.json` so they can never be mistaken for
the corrected arm (the same convention already used for `*_sharedenc`).

**Why this went unnoticed.** Every check we ran was a *count* check -- 400 demos present, prefixes
variant-balanced, stats finite, replay-in-sim succeeds. Nothing asserted pose *diversity*, and the
one artifact that showed it (identical attempt counts per variant in `gen_stats.csv`) was committed
without being read as a signal. `gen_bias.py` now reports unique-pose counts and flags any level
above 1.1x redundancy, and it belongs in dataset QA (G3) rather than being run ad hoc.

**VERIFY:** (a) **DONE 2026-08-20** -- T1 L3b reports 401 unique poses (was 43) and T3 L3b 400
(was 48), redundancy 1.0x, no flag; both variant-balanced at exactly 400 episodes after the
`--max_episodes 400` cap. (b) **CONFIRMED 2026-08-20** -- corrected T1 L3 rises 0.465 -> 0.700
between N=50 and N=100 with Wilson intervals that do not overlap the old arm's 0.470 at N=100, where
the old arm was flat (0.480 -> 0.470) and stayed flat to N=400. The plateau was the artifact. Note
the two arms coincide at N=50 (0.465 vs 0.480): at 5 demos per object the binding constraint is
per-object data, and pose diversity only binds above that. N=200/400 still to come. (c) still open --
re-read the corrected arm's training loss; if it still sits ~4x below L2's floor, redundancy was not
the whole story.

## D28 -- 2026-08-20: L3 object palettes move away from the green goal marker (marker colour unchanged)

**Trigger.** Cross-evaluating the L2-trained T1 policy (trained on `cyl_m_red` only) across all ten
L3 variants gave a strikingly bimodal result:

| variant colour | SR (small / medium cylinder) |
|---|---|
| red | 0.95 / 0.90 |
| purple | 0.80 / 0.50 |
| blue | 0.65 / 0.80 |
| **yellow** | **0.15 / --** |
| **green** | **0.10 / 0.10** |

**Mechanism.** The failures track the object's GREEN CHANNEL, not its distance from the training
colour: green G=0.60 and yellow G=0.80 fail, while blue (G=0.10) succeeds despite being far further
from red in RGB than yellow is. The goal marker is green `(0.10, 0.70, 0.10)` and the policy has to
locate it from pixels, so a green-ish object presents a second marker-coloured blob and the policy
places onto the wrong one. Training on the colours does not repair it: the L3-trained policy still
scored green 0.30 and yellow 0.35 against red ~0.48.

**Decision (user instruction 2026-08-20): keep the marker green, move the OBJECTS.** In
`cup_place/assets.py::COLORS` (shared with `drawer_stow`), green -> **orange** `(0.90,0.30,0.02)` and
yellow -> **magenta** `(0.90,0.10,0.55)`. In `push_target/assets.py::PUCK_COLORS`, yellow -> magenta
(that palette already excluded pure green, but excluding green is not sufficient -- yellow's G=0.80
is *higher* than the marker's own 0.70).

**Invariant, now documented at both palettes:** every object colour keeps green channel <= 0.30
(<= 0.40 for push_target, whose default is orange). Any future variant must satisfy this.

**Two things deliberately preserved.**
1. **Positional order.** `levels.py` lists colour names positionally, so replacing in place keeps
   every L3 variant index stable (v01 was `cyl_s_green`, is now `cyl_s_orange`). Comparing a colour
   that did NOT change (red/blue/purple) between the old and new L3 arms therefore isolates the D27
   pose fix from this colour fix, at no extra training cost.
2. **The defaults.** `red` keeps its exact RGB, so `DEFAULT_CUP`/`DEFAULT_BOX` are untouched. For
   push_target, colour is assigned to variants BY INDEX, so palette position 4 is pinned to orange
   because that is `DEFAULT_PUCK` -- had the order shifted, T3's L0-L2 datasets would silently depict
   a differently-coloured puck and would have needed regenerating too. Verified: `PALETTE_CHECK_PASS`
   (names resolve, all G within limits, all three defaults unchanged).

**Scope.** Only L3 is regenerated (with D27's seed fix, as level stem `L3b`). L0-L2 use the default
object in every task and are unaffected, so they are NOT regenerated.

**Eval sets (rule 8) are respected.** Object colour is a static material on a per-variant sub-env; it
does not touch the reset sampling, so the frozen L3 eval poses and the committed state snapshots in
`configs/eval_sets/` remain exactly as they were. What changes is the appearance of the object at
those same poses.

**VERIFY:** (a) the corrected L3 arm's per-variant SRs should no longer show the two-colour cliff --
specifically orange and magenta should behave like red/blue/purple rather than like the green/yellow
they replace. (b) If a cliff persists on some other colour, the invariant is not the whole story and
the marker itself should be reconsidered (the user's preference is to leave it, so that would be a
discussion, not a unilateral change).
