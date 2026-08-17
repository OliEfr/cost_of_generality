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
