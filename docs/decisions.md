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

**Addendum 2026-08-24 (audit, re-affirmed):** `main-prefilter` is the sole ref pinning
the two stripped blobs (~2.8 GB of a 3.8 GB `.git`); it differs from `main`'s `25c7a0e`
by exactly those two blobs, and all 39 of its commit subjects exist on `main`. The
branch was reviewed for deletion and **deliberately kept**. One correction to "how it
was made safe" above: the `data/_prepush_backup/git_before_filter_repo` fallback is one
commit behind the branch tip -- its HEAD is `ec4dfbc`, and `0f8dfc3` is absent from it,
so that SHA exists only on `main-prefilter`. No content is at risk (`0f8dfc3`'s tree
equals `25c7a0e` apart from the stripped blobs). Details: journal 2026-08-24.

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

**Addendum 2026-09-17 -- the Vulkan blocker is RESOLVED; the decision itself stays until asked.**
CINECA answered the ticket (FoldSpace toolbox); the resulting investigation (journal 2026-09-17)
found the failure was two-factor -- SingularityPRO 4.3.1 itself AND the missing NVIDIA ICD in the
loader path -- plus Kit's documented misparse of driver 535.274.02 ("535.18" < 535.129). Under the
FoldSpace toolbox Apptainer 1.5.3 with the stock host ICD bound to /etc/vulkan/icd.d and
`--/rtx/verifyDriverVersion/enabled=false`, our cog-env-5.1.0.sif renders real RTX pixels on an
A100 (gate FS_G5B_PASSED, job 58038261, `slurm/foldspace_render_g5b.sbatch`). The August
hypotheses were each individually right and individually insufficient, which one-variable testing
could not see. Cluster-side eval remains OFF (this decision unchanged): USD assets still resolve
to the Omniverse S3 (offline nodes -> 300 s timeout, job 58036943) and would need pre-staging +
a local asset root, and the eval pipeline lives locally. Revisit condition from above is now met;
switching is a deliberate follow-up (asset staging ~1 day of ops), worthwhile only if eval
wall-clock becomes binding again (e.g. a language-arm full-study rerun).

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

## D29 -- 2026-08-21 (USER DIRECTIVE): the regenerated arm is reported as "L3"; the original L3 is deprecated and excluded from all reporting

**Directive.** "leave all L3* out of reporting [name as deprecated] and report L3b as L3, because
this is the current version."

**Decision.** There are two naming layers, and they are deliberately not the same:

| layer | name | why |
|---|---|---|
| on disk / run ledger | `L3b` for the regenerated arm (datasets, LeRobot dirs, run_ids `t1_L3b_n100_s0`, `results/eval_T1_L3b_*.json`) and `L3` for the original | provenance. A number must stay traceable to the artifact it came from, and renaming files would break every existing registry row, checkpoint path and eval filename |
| reported | `L3` is the regenerated arm; the original appears nowhere | the design has ONE L3 level. Two names for it in a paper implies two levels |

**Why the original L3 is deprecated rather than reported as a second variant.** Its datasets held
43-48 unique initial poses against a nominal 400 (D27), so its SR(N) curve measures a seeding bug,
not a breadth of distribution. Printing it beside the corrected arm invites exactly the wrong reading
-- that the study measured two kinds of object generality -- when what differs is data correctness.
T2's and T3's original L3 arms were cancelled before evaluation and have no results at all; T1's was
fully evaluated and is kept as a labelled **ablation**: at an identical 400 demos, 43 unique poses
give SR 0.445 and N*(90 %) > 400, where 400 unique poses give 0.945 and 185.

**Implementation -- one function, not a convention.** A convention would have been re-implemented per
script and would have drifted. `cog.analysis.curves.canonical()` owns the mapping
(`REPORT_AS = {"L3b": "L3"}`, `DEPRECATED_LEVELS = {"L3"}`) and every reporting path goes through it:
`curves` (per-task tables), the new `summary` (cross-task table), `figures` (both figure families) and
`gen_bias_plots` (which keys datasets by stem, so it maps the other way: reported `L3` -> stem `L3b`).
Two properties, because both were bugs waiting to happen:

1. **The drop happens before the rename.** Deciding deprecation on the raw name first means
   `L3b -> L3` cannot collide with the `L3` it takes over. Renaming first would have merged two
   different datasets into one cell -- silently, since both have the same six N values.
2. **`--include-deprecated` does not rename at all.** In the ablation view both arms must stay
   distinguishable, so that flag returns raw names. There is no mode in which two datasets share a
   reported name.

**A bug this exposed.** `figures.fig_gen_sr` aggregated `gen_stats.csv` by its `level` column, and
that CSV keys the two arms differently: the original has `level="L3"` with the variant in its own
column, while the regenerated arm encodes the variant in the level (`L3bv07`). So the published
generation-SR figure was plotting the *deprecated* L3 (87.9 / 32.7 / 88.5 %) while silently dropping
all thirty L3b rows for not matching a known level name. Fixed by stripping the `v\d\d` suffix before
applying the same mapping; the figure and `paper/limitations.md` now read 86.6 / 29.7 / 92.2 %, which
agrees with the independent HDF5-level audit in `experiments/gen_bias.csv`. **Lesson: an identifier
that is sometimes a level and sometimes level+variant is not a key.** A future arm must carry its
variant in the `variant` column only.

**Also changed.** `cog.analysis.curves --out` no longer defaults to `experiments/curves.csv`; it
derives `experiments/curves_<TASK>.csv` from `--task`, because the fixed default let three tasks
overwrite one file whose surviving contents named no task. The stale `experiments/curves.csv` (a T1
snapshot predating the regeneration) is removed, superseded by `curves_T1.csv`.

**VERIFY:** any new reporting script must call `canonical()` rather than filtering levels itself. If a
second corrected arm ever appears (an `L2b`, say), the only edit should be to `REPORT_AS` /
`DEPRECATED_LEVELS` -- if it is not, the mapping has leaked out of that module again.

## D30 -- 2026-08-22 (USER DIRECTIVE): language-conditioning INVESTIGATION -- two candidates, shared harness; not a study arm

**Context.** A future full-study rerun will add a language-diversity disturbance
dimension: 20 frozen synonym instructions per task, used identically in training and
eval (paraphrase robustness, not instruction generalization). This session only
investigates and validates the mechanism: build BOTH candidates, measure training
time, smoke + one 80k verify train each + a multi-task probe; the user picks the
setting later. Nothing here enters the surface: all results go to
`results/diagnostics/`, registry rows carry `variant` per D29.

**Decisions.**
1. **No lerobot upgrade.** The pin is py-version-forced (PINS.md rows 9/20: lerobot
   >=0.5 needs py>=3.12, Isaac Sim 5.1 forces py3.11, eval is in-process). Both
   candidates run on pinned 0.4.4 unmodified.
2. **Candidate A = language-as-env-state.** Frozen CLIP ViT-B/16 text embeddings
   (512-d, unit-norm) written per frame as `observation.environment_state`; stock
   DiffusionPolicy conditions on it via the native ENV path
   (modeling_diffusion.py:246-282); ENV gets IDENTITY normalization by default, so
   the embedding reaches global_cond unchanged and a future 1-instruction control
   arm has no zero-variance hazard. This deliberately hijacks a channel meant for
   privileged env state -- the study has never used it, and the converter comment +
   manifest record the reuse. Zero new model code; cluster needs no transformers.
3. **Candidate B = multi_task_dit backport.** The 0.5.2 policy (3 modules) is copied
   into an in-repo plugin `src/lerobot_policy_mtdit/` (module names verbatim = the
   0.4.4 factory contract), 5 import fixes + a ~12-line compat shim, activated via
   PYTHONPATH + `--policy.discover_packages_path`. CLIP@224 vs our 128px frames is
   handled config-side (resize 256 / crop 224 = the baseline's 0.875 crop ratio).
4. **Instructions are a frozen benchmark** (`configs/instructions/`, rule-8
   semantics): index 0 = the canonical string already in every dataset; changes mean
   a new version file + new `*_i<n>` dataset roots, never mutation.
5. **Assignment**: conversion = per-task episode counter % 20 over the interleaved
   order (nested-N prefixes stay balanced; hard-fail on same-task multi-file merges
   -- gcd(20,10) would make instruction predict L3 variant, D18 lesson); eval =
   (batch+env) % 20 layered on the frozen protocol seeds (rotates env columns; poses
   are drawn fresh per reset, so no instruction<->pose confound).
6. **Probe** (proves conditioning is live, because single-task synonym embeddings are
   near-duplicates a policy may legitimately ignore): one policy on merged T1+T3
   (n=100/task), evaled match vs swapped instructions in both envs. PASS = matched
   SR >= 0.50 AND (match - swap) >= 0.30 with non-overlapping Wilson95 CIs. Caveat
   on record: a swap collapse proves causal influence, not semantic parsing. CLIP
   cosine geometry note: cup_place x push_target instruction sets are close (mean
   0.842) -- a conservative pairing; see journal 2026-08-22.

**VERIFY (open).** H4 flag-off regression must reproduce t1_L1_n100_s0 SR=0.86
exactly; A1 +1024 cond-encoder param assert; B throughput on A100 (the 2.2 GPU-h/cell
rule will NOT hold for B); offline CLIP resolution on a compute node before B's 80k.

### D30 addendum -- 2026-08-23: VERIFY items discharged; verdict

All four VERIFY items closed: H4 flag-off reproduced SR 0.860 exactly; A1 +1024
cond-width assert held (402->1426); B throughput measured (1.91 steps/s, 11.71
GPU-h/80k at batch 64 -- b128 OOMs the A100-64GB, so the 2.2 GPU-h/cell rule does NOT
transfer to B); offline CLIP verified on a compute node before the 80k spend.
Investigation verdict (details: docs/lang_conditioning_report.md): **candidate A
(env-state embedding) is the recommended setting for the language-diversity rerun**
-- SR 0.830 vs baseline 0.86 (CIs overlap), probe PASS (match 0.930/0.980 vs swap
0.060/0.040), training cost indistinguishable from baseline. Candidate B validated
end-to-end (SR 0.700 untuned at batch 64) and kept as escape hatch. Open item for
the rerun, deliberately NOT decided here: an L3-lang instruction<->variant
assignment design (converter hard-refuses the confounded case), and whether lang
results enter curves.py's naming contract (D29-style decision needed then).

## D31 -- 2026-09-17 (USER DIRECTIVE): reverse ablation -- two leave-one-out arms per task, on a uniform 200-episode slice protocol, generated and evaluated entirely on Leonardo

**Directive.** "this repo run a study where disturbance dimensions were added to investigate how
performance scales when adding disturbance dimensions. Now, I want the same study but the other way
around: start from all disturbances added, and then always remove a single one to see which
simplification gives the largest performance gain. plan to run everything on cluster (datagen,
train, eval; parallelize where useful)." Scope settled in the same session: leave-one-out only (not
the full 2^3 lattice), all three tasks, baselines re-measured, in-distribution eval only, T2's
existing stage instrumentation recorded and none built for T1/T3.

### The design

The additive ladder walks one ordering of three dimensions; this measures the other end of each
one's marginal. Writing a level as the SET of disturbances it switches on, with

| | A | B | C |
|---|---|---|---|
| T1 cup_place | cup XY 30x40 cm + yaw +-90 deg | goal disk XY 20x20 cm | 10 object variants |
| T2 drawer_stow | stow-object XY + yaw +-45 deg | cabinet XY +-5 cm + yaw +-7.5 deg | 10 box variants |
| T3 push_target | puck XY 12x12 cm | target bearing +-25 deg | 10 puck variants |

the existing ladder is L0={}, L1={A}, L2={A,B}, L3b={A,B,C}, and the new arms are

| arm | set | = | sub-envs | demos |
|---|---|---|---|---|
| `AC` | {A,C} | L3b minus B | `ACv00..ACv09` | 10 x 40 |
| `BC` | {B,C} | L3b minus A | `BCv00..BCv09` | 10 x 40 |

**There is no third arm, because "L3b minus C" is L2 -- field for field, not approximately.**
`_mk(key, level, variant, range_or_None, range_or_None)` in each `levels.py` was already a
per-dimension toggle (`None` collapses an axis to its fixed point), so `SUB_LEVELS["L2"]` and a
hypothetical L3-minus-C are the same object. One third of the leave-one-out grid per task was
therefore already generated, trained and evaluated before this decision was written.

**Vocabulary lives in one place.** `cog.analysis.curves` gains `DISTURBANCE_SETS` (the lattice),
`LEVEL_LABEL` (reverse arms are displayed as what they REMOVE: `L3\B`, `L3\A`) and `LEVEL_TOKEN`
(the regex alternation every level-name parser now imports rather than restating). D29's VERIFY
required exactly this: the mapping must not leak out of that module again. Four parsers were
silently dropping any level they could not name -- `update_registry_from_evals.py`,
`gen_stats.py`, `gen_bias.py`, `merge_eval_sets.py` -- each failing as a no-match rather than an
error. Patching `gen_stats.py` also fixes the D29 residue: `L3bv07` used to parse as
level=`L3bv07`, variant=`-`, and now parses as level=`L3b`, variant=`v07`, which is what D29 said a
future arm must do.

**Seeds** extend the existing per-task blocks and stay disjoint from the eval seeds (5000-5009) and
the new warm-up seeds (4900-4909), so training, eval and warm-up poses can never coincide:

| | T1 | T2 | T3 |
|---|---|---|---|
| `ACv00..09` | 1300-1309 | 2300-2309 | 3300-3309 |
| `BCv00..09` | 1400-1409 | 2400-2409 | 3400-3409 |

Provenance control (D9) is unchanged: the same annotated L2 source demos, the same generator
settings, the same `--num_envs 8`.

### The eval protocol changes, and why that is not optional

The old protocol scores a flat cell as 5 batches in ONE process and a variant cell as one batch per
process. Batch 0 is genuinely depressed -- `t1_L1_n100_s0` scores [0.50, 0.90, 0.95, 0.95, 1.00]
across its five batches **with the t==0 phantom guard active**, so it is a process warm-up effect,
not a scoring bug (journal 2026-08-22). That asymmetry is exactly the confound that put Finding 4
("the axes are not additive and the object axis dominates") in doubt, and a leave-one-out study
whose arms all carry dimension C would inherit it in full.

**New protocol: every cell is ten slices. Slice s is its own process, runs one unscored warm-up
batch (`reset(seed=4900+w)`), then scores batch s (`reset(seed=5000+s)`, 20 envs). 200 scored
episodes per cell.** For a variant arm slice s also selects sub-env `<ARM>v0s`, which is D18's
diagonal -- so D18 is preserved, not overturned. For a flat arm every slice uses the same env.

- **Rule 8 is respected.** The frozen `configs/eval_sets/*.json` already contain batches 0-9 for
  every level. Nothing is regenerated; what changes is which committed rows are read and how they
  are grouped into processes -- the same kind of change D18 itself was. `merge_eval_sets.py` now
  *enforces* rule 8 rather than relying on it: it refuses to overwrite an existing frozen set.
- **Consequences, stated plainly.** Reported episodes for L0-L2 go 100 -> 200. Absolute success
  rates will shift relative to `experiments/clean_surface.csv`, which this sweep supersedes for
  every cell it covers. The re-measured surface is written with the protocol suffix **`_u200`**
  (`eval_T1_L2_n100_080000_u200.json`), beside the originals rather than over them -- the same
  convention as `_fixed` / `_sharedenc` / `_poseredundant`. `curves.load()` and
  `update_registry_from_evals.py` take a `--suffix`, default `""`, so every existing caller reads
  exactly what it read before.
- **The warm-up LENGTH is measured, not chosen.** Gate G6 evaluates one cell at warm-up in
  {0 batches, 20 steps, 100 steps, 1 full batch} and takes the shortest at which the scored SR stops
  rising; a full batch is the conservative default and roughly doubles the eval bill, so this gate
  pays for itself. Every result records its warm-up block, and `pool_variant_eval.py` refuses to
  pool slices that disagree on it.

### Baselines are re-measured on the cluster, not carried over

All 72 baseline cells (L0/L1/L2/**L3b**, 6 N, 3 tasks) are re-evaluated under the new protocol on
the A100s, alongside the 36 new cells. Two confounds go away at once: the batch-0 asymmetry above,
and the machine. The audit that prompted the second: **no L3b cell in any task has a `_fixed`
artifact** -- all 18 are from the original pre-guard sweep, while every L0/L1/L2 cell was re-swept.
That is probably harmless (one batch per process means there is no batch b-1 to carry over from),
but it is unverified, and the published surface therefore mixes guarded flat cells with unguarded
diagonal ones. One code path for all 108 cells removes the question instead of arguing it.

### Everything runs on Leonardo, which inverts D25

D25 kept evaluation local, most recently on throughput grounds: an A100 is ~3.3x slower per
rendered episode than the 4090 (no RT cores). That argument was about a serial sweep. This study is
1,080 eval slices and 60 datagen legs, all independent, and the cluster's parallelism beats a
workstation that can run two evals at a time. **D25 is superseded for this study's waves**; local
eval remains correct for one-off checks.

New infrastructure, all copied from `slurm/bench_eval_a100_dbg.sbatch` (the only verified container
recipe): `slurm/lib_cog_container.sh` holds the recipe ONCE -- FoldSpace apptainer, stock host
Vulkan ICD, `verifyDriverVersion=false`, the cu12-over-cu13 cuDNN bind, no inner `srun` -- because a
drifted second copy fails by rendering on the CPU or dying in a conv, and both look like something
else. On top of it: `slurm/datagen.sbatch` (one job per variant, not one per arm: a camera-enabled
Kit boot is 13 s warm in this container against 3.5-4 min on the workstation, so the batching
rationale behind `gen_L3_wave.sh` is gone), `slurm/convert.sbatch` (no GPU at all -- the converter
imports h5py/numpy/lerobot and encodes through PyAV), `slurm/freeze_eval_sets.sbatch` (state env, no
cameras, but still a GPU because the poses come off the CUDA generator), and a rewritten
`slurm/eval.sbatch` (one job per slice). The pre-FoldSpace eval script is kept as
`slurm/eval_legacy_singularity.sbatch` and must not be copied from: it uses plain `singularity` and
an inner `srun` that silently drops the GPU. Each job gets its own Kit scratch under
`$FAST/cog/jobscratch/$SLURM_JOB_ID`, seeded warm from `$WORK`; ~60 concurrent Isaac jobs against
one shared shader cache and one shared `--home` is a contention mode the single-job 2026-09-17 gate
never exercised.

### Stage instrumentation: T2 only, as it stands

`--stages` is passed on every T2 eval in this study -- both new arms and re-measured baselines, all
six N -- and **no stage instrumentation is built for T1 or T3** (user directive). It matters on T2
because its success rates sit at 0.04-0.43 and are non-monotone in N, so a binary SR cannot resolve
a leave-one-out delta there, while the existing funnel already pins every T2 SR to the mid-rollout
grasp. `pool_variant_eval.py` pools the `stages` block across slices by recounting episodes, which
is new: the three existing `_stages.json` files are flat single-process cells.

### The principal risk, and the gate that settles it

The new arms are generated on A100s but compared against L2 and L3b, whose demos came from the
4090. A renderer difference would confound "which disturbance was removed" with "which GPU rendered
the training images" -- structurally the same class of artefact as D27. `scripts/dev/parity_check.py`
(gate G2) regenerates an existing leg on the cluster with its original seed and compares
attempt count, per-demo `num_samples`, initial states, actions, states and pixels, in that order,
with the first three as hard stops. Pixels are not expected to be bit-identical; the verdict bands
are MAE <= 1 and p99 <= 8 (AGREE, freely mixable), MAE <= 3 with no channel-mean shift > 2
(TOLERABLE, usable because each arm is generated entirely on one machine, but record it), and
anything beyond (DIVERGENT). Unless G2 lands in AGREE, gate G2b retrains a control cell on
cluster-generated data and requires it inside the existing cell's binomial CI: ~4.4 GPU-h total to
de-risk ~350.

**If G2b fails**, the recovery is targeted rather than total: regenerate only L2 and L3b on the
cluster (~20 GPU-h of datagen plus 36 retrained cells, ~79 GPU-h), since those are the two arms the
leave-one-out deltas are actually taken against. Budget: the whole study is ~300-400 GPU-h against
~2,000 remaining of the approved 2,200 ceiling; the binding constraint is the grant calendar
(2026-10-29), not GPU-hours.

**VERIFY (updated 2026-09-18 as the gates landed):** **(c) is DISCHARGED and the warm-up is
decided: one FULL batch.** G6 measured 0.720 / 0.800 / 0.810 / 0.950 for warm-up of
none / 20 steps / 100 steps / a batch run to termination, so the short warm-ups buy nothing and the
eval budget is the conservative ~276 GPU-h. The fully-warmed pooled value, 0.950, equals the mean of
the 4090's four warm batches for the same cell to three decimals -- which also discharges eval-side
machine parity, and shows the published 0.86 to be exactly one cold batch in five. The per-variant
L3 diagonal was 100% cold, so the object-axis arm carried the largest penalty: the confound runs in
the direction that manufactures Finding 4. **(a) is DISCHARGED.** G2 returned TOLERABLE -- pixels within the renderer's own run-to-run noise
(which a local-vs-local rerun showed is not zero), trajectories identical on matched episodes -- and
G2b then settled what that means for a trained policy: 0.940 on A100-generated data against 0.960 on
4090-generated data, paired over identical seeds, mean delta -0.020, t(9) = -1.18. The contingency
branch below (regenerating L2 and L3b on the cluster, ~99 GPU-h) is NOT needed. Remaining: (b) `gen_bias` reports ~400 unique
initial poses per new arm at <=1.1x redundancy -- the D27 check, which must run on every new arm.
(d) `curves.canonical()` maps `AC`/`BC` with no caller filtering levels itself.

### D31 addendum -- 2026-09-18: the L3b checkpoint stem is not the L3b gym key

Caught in an eval dry-run, before the sweep rather than inside it.

D27 regenerated the L3 arm under per-variant seeds; D29 then renamed the **artifacts** -- datasets,
`run_id`s, result filenames, `$FAST` stems -- from `L3` to `L3b`, and taught `curves.canonical()` to
report `L3b` as "L3". What D29 did **not** rename is the gym registrations: `levels.py` in all three
tasks still registers `L3v00..L3v09`, and no `L3bv*` id exists anywhere.

So the re-measurement half of this study has an asymmetry the new-arm half does not:

| | checkpoint | gym key |
|---|---|---|
| `AC` / `BC` | `t2_AC_n400_s0` | `ACv0<slice>` |
| **`L3b`** | `t2_L3b_n400_s0` | **`L3v0<slice>`** |

`slurm/eval.sbatch` derived the gym key from the arm token directly and would have asked for
`Cog-DrawerStow-L3bv00-IK-Rel-Visuomotor-v0` -- a gym registration error, failing all 180 L3b slices
(3 tasks x 6 N x 10) of the 720-slice baseline re-measurement, arriving as 180 separate
`EVAL_FAILED`s some hours into the sweep.

**Decision: map `L3b -> L3` inside `eval.sbatch`, not at the call sites.** `launch_wave.py`,
`resweep_eval.py` and any future caller all name the arm by its checkpoint stem, which is the token
the registry and the result filenames use; the sbatch is the one place that needs the gym key, so it
is the one place that should know the mapping. A convention held only in callers is exactly what
D29 records as drifting.

`scripts/dev/check_levels.py` now asserts both halves -- `L3v00` IS registered, no `L3bv*` is -- so
the invariant the sbatch depends on cannot be silently inverted by a later rename.

**Not affected:** `configs/eval_sets/*.json` are pose *snapshots* consumed by analysis
(`success_vs_pose.py`), not inputs to `rollout_eval.py`, which reproduces the benchmark by seeding
the env from `protocol.json`. The absence of an `L3b.json` is therefore correct, not a second bug.

## D32 -- 2026-09-19: the phantom-success guard was one step too narrow; re-score the sweep

**Status:** accepted. Supersedes the 2026-08-21 fix, which is now known to be incomplete.

### What happened

`rollout_eval.py` reads `env.termination_manager.get_term("success")` after each `env.step()`. After
a manual `env.reset()` that term is STALE: it still reports the previous batch's value, so every env
that succeeded in batch *b−1* latches a phantom success at the head of batch *b*. D-2026-08-21
diagnosed this and zeroed `succ_now` at `t == 0`.

The term is still stale at `t == 1`. The August fix moved the symptom by one step instead of
removing it, and nothing noticed for four weeks because a bare success flag cannot say *when* a
success latched -- only T2's `--stages` path timestamps it, and `--stages` had run on three cells
that nobody re-read.

### The evidence

Two independent measurements, both from data already on disk:

1. **D31 sweep, T2, 2,400 episodes.** `t_success` takes exactly two values: **1** (907 episodes) and
   **>= 590** (395). Nothing in between. A bimodal distribution with a 589-step gap is not a
   behavioural mode.
2. **The three pre-D31 five-batch runs.** Early latches per batch:

   | file | b0 | b1 | b2 | b3 | b4 |
   |---|---|---|---|---|---|
   | `eval_T2_L0_n400_..._stages` | 0/16 | 16/19 | 19/20 | 16/20 | 15/19 |
   | `eval_T2_L1_n400_..._stages` | 0/2 | 2/4 | 2/5 | 4/7 | 3/8 |
   | `eval_T2_L2_n400_..._stages` | 0/5 | 5/10 | 6/12 | 8/11 | 3/9 |

   Batch 0 -- the only batch with no predecessor -- is clean in all three. The first transition
   matches batch *b−1*'s success count exactly (16→16, 2→2, 5→5).

The phantom episodes are *not* the inert ones of the August note: their median `max_object_lift` is
0.450 against 0.460 for genuine successes, and 622 of 907 reach the drawer. They are real attempts
that were credited a step before they began. So the failure is silent in every aggregate: the SR is
wrong and nothing about it looks wrong.

### Consequences, and who is affected

- **Every D31 result under the `u200` suffix is overstated.** The uniform-slice protocol puts a
  warm-up batch before every scored batch, so *every* scored batch is a batch *b>=1*. Observed
  SR ≈ 1 − (1−p_true)(1−p_warmup): T2/BC/n200 reads 0.995 where `object_over_drawer` is 0.925.
- **It cannot be corrected post hoc.** Once success latches at `t == 1` the timestamp is never
  updated, so an env that carried over *and* then genuinely succeeded is indistinguishable from one
  that only carried over. Counting `t_success > 1` undercounts by exactly the amount that matters
  most in high-SR cells.
- **The published additive surface is affected ASYMMETRICALLY, which is worse than uniformly.**
  Flat cells (L0/L1/L2) score batches 0-4 in one process: batch 0 clean, batches 1-4 inflated, so
  ~80 % of their episodes are overstated. The L3b diagonal runs one batch per process -- batch 0,
  with no predecessor -- so it is **clean**. The surface therefore overstates L0/L1/L2 relative to
  L3b, which inflates the apparent cost of the object-variant axis C. **Finding 4 ("the axes are not
  additive and the object axis dominates") rests partly on this artefact** and cannot be restated
  until the re-score lands. It was already marked CONFOUNDED for the warm-up reason; this is a
  second, independent defect pointing the same way.

### Decision

1. Guard a **window**, not a step: `PHANTOM_GUARD_STEPS = 10`. The data bounds the staleness from
   below (>= 2 steps) but *not* from above -- once the latch fires the timestamp stops moving, so a
   stale `t == 2` would be invisible. Over-guarding is free: the shortest source demo is >= 150
   steps and the earliest genuine success ever observed is 590.
2. **Timestamp every success in every task**, not only under `--stages`. One int per episode. This
   is the instrumentation whose absence let the bug survive its own fix.
3. **Fail loudly**: any success before step 100 sets `PHANTOM_SUSPECT` in the result and prints a
   warning. `earliest_success_step` is recorded unconditionally.
4. **Stamp `protocol.phantom_guard_steps` into every result.** A result without the key was scored
   with the broken guard; a corrected number can then never be silently pooled with an old one.
5. **Re-score the full 1,080-slice sweep** under suffix `u200g10`. No retraining -- the bug is in
   scoring only, so the 108 checkpoints stand. ~250 GPU-h.

### VERIFY (open)

- Does the corrected T2 SR land near `object_over_drawer`, as the carryover model predicts? If it
  lands far *below*, the guard is now suppressing something real and 10 is too wide.
- Re-score the pre-D31 flat baselines too, or the additive surface keeps its asymmetry. The
  `u200g10` sweep already covers L0/L1/L2/L3b for all three tasks, so this is satisfied by it.

### D32 amendment -- 2026-09-19, same day: the mechanism above is wrong, and so was the first fix

The evidence in D32 stands; the **explanation does not**, and the 10-step guard it prescribed does
not work. Recorded rather than rewritten, because the wrong model was load-bearing for two fixes and
the way it failed is the useful part.

**What the 10-step guard did.** Four verification slices, suffix `u200g10`:

| cell | SR under `u200` | SR under `u200g10` | earliest success |
|---|---|---|---|
| T2 BC n200 s0 | 1.000 | 1.000 | 10 |
| T2 BC n200 s1 | 1.000 | 1.000 | 10 |
| T1 L1 n100 s0 | 0.900 | 0.900 | 10 |
| T1 L1 n100 s1 | 0.900 | 0.900 | 10 |

`earliest == guard` exactly, at both guard values, with the SR unchanged to three decimals. The
phantom did not shrink; it moved to the far edge of the window. And of the 19 episodes latching at
t=10 in T2/BC/s0, **all 19 had the drawer shut at that instant** (`t_open` 145-259, or -1 for one
episode that never opened the drawer or lifted anything at all: `max_open=0.0`, `max_lift=0.0003`,
and still scored a success).

**The actual mechanism.** `TerminationManager.compute()`:

```python
self._truncated_buf[:] = False      # cleared every call
self._terminated_buf[:] = False     # cleared every call
for i, term_cfg in enumerate(self._term_cfgs):
    value = term_cfg.func(self._env, **term_cfg.params)
    ...
    rows = value.nonzero(as_tuple=True)[0]
    if rows.numel() > 0:
        self._term_dones[rows] = False      # written ONLY for rows that fired
        self._term_dones[rows, i] = True
```

`_term_dones` is written only for envs where some term fired, is never cleared otherwise, and
`TerminationManager.reset()` does not touch it -- it only reads the column means for logging.
So `get_term("success")` is not a per-step signal at all: it is a **sticky, cross-episode record of
the last reason this env's episode ended**. Once an env succeeds it reads True forever, through
resets, in every subsequent batch. `_terminated_buf` is cleared at the top of every `compute()`,
which is why the `terminated` flag returned by `env.step()` was always fresh and why `finished`,
`alive` and the stage timestamps were all correct throughout.

This predicts every observation, including the ones that made the "stale for k steps" model look
right: batch 0 clean (nothing has fired yet), latch at whatever step suppression stops (it is
latched at *all* of them), and genuine late successes still detected (in envs that had not
previously succeeded).

**The fix.** `succ_now = terminated & env.termination_manager.get_term("success")`. Exact, not
defensive: at a step where `terminated` is true, `compute()` has just overwritten `_term_dones`
one-hot for those rows, so `get_term` names the term that fired *this* step; at every other step the
conjunction is false whatever the latch holds. Generic over the three tasks and needs no second copy
of the success predicate. `PHANTOM_GUARD_STEPS` is retired to 0 and `protocol.success_signal` is
stamped into each result, so the three generations -- no key (`u200`, 1-step guard), `guard=10`
(`u200g10`), and `success_signal` present (correct) -- are distinguishable on sight.

**Consequence for the blast radius, which is now smaller and sharper than D32 assumed.** In batch 0
of any process `_term_dones` is all false, so old and new code agree *exactly*: the sticky flag and
the fresh one first become true at the same step. Therefore

- **the L3b diagonal cells are correct as published** -- one batch per process, always batch 0;
- **flat cells' batch 0 is correct** and their batches 1-4 are inflated;
- **every `u200` and `u200g10` slice is inflated**, because the D31 warm-up makes every scored batch
  a batch >= 1.

So D32's asymmetry conclusion holds and tightens: the published surface overstates L0/L1/L2 against
a correct L3b, and Finding 4's "the object axis dominates" is partly an artefact of comparing
inflated flat cells with clean diagonal ones.

**Lesson, superseding the one in the journal.** The August fix and the 10-step fix failed the same
way: both were sized to the *symptom's location* (t == 0, then t < 10) without reading the code that
produced it. Thirty lines of `TerminationManager.compute()` answered in one pass what two rounds of
guard-widening could not. Read the mechanism before sizing a workaround to it.

## D33 -- 2026-09-21: the first-batch depression is a renderer defect, so the warm-up batch is mandatory, not precautionary

**Status:** accepted. Upgrades D31's warm-up batch from an empirical correction to a documented one,
and adds a standing rule.

### Decision

**Every scored episode runs in a process that has already completed at least one unscored batch.**
No result is reported from the first episode of a fresh process. This was already D31's protocol;
what changes is that it is now backed by a mechanism rather than by an observed offset, and that it
applies to any future eval protocol, not only the 108-cell sweep.

### What was measured

Probe `slurm/warmup_probe.sbatch` / `scripts/dev/warmup_frame_probe.py`, job 58342228, 1 m 54 s.
The policy is removed from the loop: one process resets `Cog-CupPlace-L1-IK-Rel-Visuomotor-v0`
(20 envs) three times at seed 5000 and drives every cycle with an identical zero-action sequence,
so nothing can diverge through the policy. Camera tensors and the cup's root pose are captured at
steps 0, 1, 2, 5, 10, 20, 29.

- **Physics is exactly identical.** Cup root pose max |Δ| = `0.000e+00` at every step for all three
  cycle pairs. Equal as float32, not within tolerance.
- **The renderer is not converged in the first cycle.** Cycles 1 vs 2 sit at a flat **0.62 MAE**
  floor (0-255 units) at every step and both cameras -- the renderer's own sampling noise, and the
  yardstick. Cycle 0 against it: table_cam **10×** the floor at step 0, decaying to 1.0× by step 20;
  wrist_cam **112×** at step 0, at the floor from step 1.
- **The wrist camera's first frame is a different image, not a noisy one.** 99.07 % of pixels
  differ, per-channel mean shift −60 to −63, max 231. Most consistent reading: the first render
  after process start uses a camera transform that does not yet reflect the post-reset robot pose.
  The table camera is world-fixed, which is why it shows progressive convergence instead.

So a policy's opening actions in a fresh process are conditioned on a wrist view of the wrong thing
and on table pixels ~20 steps from settled. That is sufficient to produce the measured depression
(one full warm-up batch is worth **+0.13 [+0.02, +0.24]** on `t1_L1_n100`), and it explains why a
20-step warm-up recovers part of it: ~20 steps is where the table camera reaches the floor.

### Consequences

- All 1,080 slices behind the current 108-cell surface ran `warmup={batches: 1, seed: 4900}` --
  verified, zero exceptions. Both published reports rest entirely on warmed evaluations.
- Any protocol that scores the first episode of a process is scoring corrupted observations. The
  pre-D31 per-variant protocol did exactly that for every one of its episodes, which is the
  mechanical reason its diagonal cells read low.
- The warm-up is identical across all cells, so it cannot bias one arm against another. It costs
  one extra batch per slice.

### VERIFY (answered 2026-09-22, partially)

Whether the wrist-camera first-frame defect is a transform-ordering bug that an extra
`sim.render()` before the first observation would remove outright. **It is.** Job 58382793: one
extra render call takes the cold wrist frame from 105x the noise floor to 1.6x, and four calls put
both cameras at the floor. The renderer is one call behind the post-reset transform.

Still open: whether that recovers the success rate. The probe measures observations, not success,
so it does not establish that a few render calls could replace the warm-up batch's
+0.13 [+0.02,+0.24]. Until an eval run says otherwise, the rule above stands unchanged: no result
is reported from the first episode of a fresh process.
