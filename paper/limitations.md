# Limitations (running list for the report)

Kept current as findings land; the report's limitations section is drawn from here.

## 1. The generality axes vary placement and appearance, never shape or mechanism

Each task's ladder adds randomization in this order: object pose, then fixture pose,
then object appearance and a small scale step. What it never varies is **geometry or
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

## 3. One seed per cell

Every training cell runs a single seed (0), so per-cell success rates carry run-to-run
variance that is not measured. Mitigations: frozen pre-sampled evaluation sets, nested
demo subsets so curves are monotone in data rather than resampling noise, and
best-of-last-three-checkpoints as the primary metric.

## 4. Fixed step budget, not converged training

All cells train for the same 80k steps rather than to convergence, so results describe
success rate at a fixed compute budget. Large-N cells may be under-trained relative to
small-N cells.

## 5. In-distribution evaluation, simulation only

Policies are evaluated on held-out initial states drawn from the *same* level they were
trained on, so the study measures data cost of fitting a distribution, not
out-of-distribution transfer. Everything is simulated in Isaac Sim; no real-robot
validation.
