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

