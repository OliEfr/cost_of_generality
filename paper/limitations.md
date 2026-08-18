# Limitations (running list for the report)

Kept current as findings land; the report's limitations section is drawn from here.

## 1. Tasks 1 and 2 vary placement and appearance, never shape or mechanism

**Scope note (updated 2026-08-18):** this applies to Tasks 1 and 2. **Task 3 does carry a
real geometry axis** — its L3 varies puck radius across 0.032-0.045 m and height across
0.040-0.055 m, ten physically distinct variants (5 radii x 2 heights), no two differing only
in colour. So the study is not uniformly blind to shape; the blindness is specific to the two
prehensile tasks, and Section 2 explains why.

For Tasks 1 and 2 the ladder adds randomization in this order: object pose, then fixture
pose, then object appearance and a small scale step. What it never varies is **geometry or
kinematics** — the manipulated object keeps its shape (Task 1 a cylinder, Task 2 a
cube), the cabinet is one fixed asset, and the drawer always starts fully closed.

Task 2, concretely:

| Level | robot start | object pose | cabinet pose | object size/colour | drawer geometry | drawer start |
|---|---|---|---|---|---|---|
| L0 | joint noise | fixed | fixed | fixed | same | closed |
| L1 | joint noise | 10x18 cm, yaw +-45 deg | fixed | fixed | same | closed |
| L2 | joint noise | 10x18 cm, yaw +-45 deg | +-5 cm, +-6 cm, +-7.5 deg | fixed | same | closed |
| L3 | joint noise | 10x18 cm, yaw +-45 deg | +-5 cm, +-6 cm, +-7.5 deg | 2 sizes x 5 colours | same | closed |

The colour dimension is **physically inert**, and this is measured rather than assumed:
generation runs that differ only in colour produce byte-identical attempt counts
(Task 1: 40/45 for five variants then 40/46 for the other five; Task 2: 40/120 x5 then
40/125 x5). The size step is real but small — 1.3 points of generation success rate
between the two Task 2 box sizes.

Consequently the top level should be read as **"appearance and mild scale variation"**,
not "object variation", and results at that level do not speak to shape generalization.

## 2. Why that limitation exists: the data generator constrains which axes are affordable

This is a limitation of the *method*, not only of our configuration, and it is worth
stating as a result in its own right.

Isaac Lab Mimic generates a demonstration by expressing each source subtask's
end-effector trajectory relative to a single 4x4 reference pose and rigidly re-applying
it to that reference's pose in the new scene:

    src_eef_rel_obj = src_eef_poses @ inv(src_obj_pose)
    new_eef_poses   = src_eef_rel_obj @ cur_obj_pose

Two assumptions follow. First, the required gripper pose must be a **fixed rigid offset**
from the reference frame, so object geometry must be effectively unchanged — a cube
4.0 -> 4.8 cm preserves the offset onto flat parallel faces, whereas a mug with a handle
requires a different, yaw-dependent grasp and therefore its own source demonstrations.
Second, the reference pose must **capture all relevant state** — our drawer subtasks
reference the cabinet root, which does not move when the drawer slides, so a
partially-open starting drawer is invisible to the transform and the replayed trajectory
misses the handle by the initial opening.

The practical consequence is an asymmetry in what generated data costs:

| Axis added | Generation SR cost (Task 2) |
|---|---|
| object pose | 54.9 % -> 44.2 % (10.7 points) |
| fixture (cabinet) pose | 44.2 % -> 30.6 % (13.6 points) |
| appearance + mild scale | ~1 point (30.6 % -> 32.7 % pooled; 33.3 % vs 32.0 % between sizes) |

Pose generality is expensive to *learn* but cheap to *generate*; geometry generality is
not cheaply generatable at all under this method. This is a plausible explanation for
why MimicGen-style datasets in the literature vary placements rather than shapes, and it
means demonstration-count studies built on such pipelines are systematically better
evidence about spatial generality than about object generality.

### The three-task evidence, and why the constraint is a DESIGN variable (2026-08-18)

With Task 3 complete, generation success rate across the three tasks is:

| level | T1 cup_place | T2 drawer_stow | T3 push_target |
|---|---|---|---|
| L0 | 86.4 % | 54.9 % | 98.5 % |
| L1 | 85.8 % | 44.2 % | 94.8 % |
| L2 | 85.1 % | 30.6 % | 95.0 % |
| L3 | 87.9 % | 32.7 % | 88.5 % |
| pattern | flat | steep collapse | near-flat, high |

Task 3 is the hardest task to CONTROL — its scripted expert needed fifteen debugging cycles
and tops out at 85-94 %, where Task 1's exceeds 98 % — and yet it has the highest generation
success rate of the three, and it is the one task that carries a genuine geometry axis. The
difference is not difficulty. It is that Task 3 was designed around the rigid
single-reference transform: one subtask (no boundary to mis-segment, no interpolation jump
mid-stroke), a synthetic reference frame whose orientation encodes the push direction so
direction adapts for free, a deliberately constant object-to-target distance because a rigid
transform carries no scale, and source demonstrations selected on placement error so no
template hands its own error to its copies.

The sharper claim this supports: **generation success rate measures the fit between the data
pipeline's assumptions and the task's structure, not the task's intrinsic difficulty.** What
costs generation SR is the number of independent pose-dependent relations a task requires
beyond the one that a single rigid reference per subtask can express — Task 2 chains three
subtasks over a cabinet pose, a drawer opening and a box pose, so each added randomization
axis further degrades an already-approximating transform; Task 3 needs exactly one relation
and encodes it. Concretely, that design difference was worth 6.1x in wall-clock: 1600 demos
in 2 h 09 min for Task 3 against 13 h 10 min for Task 2.

The corresponding limitation is therefore narrower and more interesting than "the pipeline
cannot do geometry": a geometry axis is affordable exactly when the grasp or contact it
implies is expressible as a fixed offset in the reference frame. Non-prehensile contact
qualifies; a grasp whose pose depends on the object's shape does not.

## 3. Task 3's generality ranges are capped by the scripted expert, not by the task

Two of Task 3's axes were narrowed to what the expert could service reliably, and both caps
are empirical rather than principled:

- **Push direction** is a 50 deg arc (bearing 90 +- 25 deg). Expert success binned by
  |bearing - 90 deg| was 94 % (0-10 deg), 95 % (10-25 deg), 75 % (25-45 deg): beyond ~25 deg
  the stroke runs toward the edge of the arm's comfortable workspace.
- **Puck radius** spans 0.032-0.045 m (1.4x). Expert success fell monotonically with radius
  — 88 % at 0.032, 92-94 % at 0.045, 73-83 % at 0.052, 63-75 % at 0.058 — because a ~2 cm
  blade cannot keep a wide disc on line: contact is a short chord of a shallow arc, so any
  lateral offset spins the puck instead of translating it.

Both are properties of a Franka gripper used as a pusher, not of pushing as a task. A
purpose-built pusher, or a closed-loop learned expert, would widen both. Any measured data
cost at Task 3's L2 and L3 therefore describes generality over the serviceable range, and
should not be extrapolated to wider ranges.

## 4. The design resolves ~2x data-cost effects, not ~1.3x ones

Quantified by Monte Carlo before the matrix was run
(`scripts/dev/nstar_resolution.py`, 400 trials per condition, true logistic slope
2.2 logit/decade over the 6-cell grid 10..400):

- N* from the **logistic fit** is unbiased with |error| p90 of 15-23 % at 100 eval episodes
  per cell, improving to 12-15 % at 200. N* from an **interpolated crossing** is biased
  +3 to +5 % with |error| p90 of 27-30 %, because one noisy cell moves the crossing far when
  the demo grid doubles between cells. The fit is therefore the primary estimator and the
  crossing a secondary, assumption-free check.
- For a **true 2.0x** cost ratio, the median estimate is 2.03x with 90 % of estimates in
  **[1.57x, 2.65x]**.

Consequently a measured ratio near 1.2-1.3x cannot be distinguished from 1.0x at this
evaluation budget, and results in that range must be reported as *below the resolution of the
design* rather than as evidence that generality is free.

## 5. One seed per cell

Every training cell runs a single seed (0), so per-cell success rates carry run-to-run
variance that is not measured. Mitigations: frozen pre-sampled evaluation sets, nested
demo subsets so curves are monotone in data rather than resampling noise, and
best-of-last-three-checkpoints as the primary metric.

## 6. Fixed step budget, not converged training

All cells train for the same 80k steps rather than to convergence, so results describe
success rate at a fixed compute budget. Large-N cells may be under-trained relative to
small-N cells.

## 7. Evaluation set sizes differ between levels

L0-L2 use a 100-episode standard evaluation (5 batches x 20 environments, 100 distinct
initial states); L3 uses 200 episodes, because it must cover ten appearance variants,
paired diagonally with the ten batches so that it also holds 200 distinct poses. L3
therefore carries a tighter confidence interval than the other levels' standard
evaluation. Cross-level headline comparisons use each level's 200-episode set, which have
equal spatial coverage (see D18).

## 8. In-distribution evaluation, simulation only

Policies are evaluated on held-out initial states drawn from the *same* level they were
trained on, so the study measures data cost of fitting a distribution, not
out-of-distribution transfer. Everything is simulated in Isaac Sim; no real-robot
validation.
