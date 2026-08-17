# Related Work — The Cost of Generality

**Research question of this study.** In manipulation imitation learning, how many
demonstrations does *generality* cost? For each task we define a ladder of generality
levels L0–L3 (progressively broader training *and* evaluation distributions), train a
vision diffusion policy at demo counts N ∈ {10…400} per level under a fixed step budget,
and read off data-cost curves **N\*(s | level)** — the demonstrations needed to reach
success rate *s* — plus cost ratios N\*(L_k)/N\*(L0).

## 1. Data scaling in imitation learning

Neural scaling laws in language and vision motivated the same question for robot
policies. **Tuyls et al. (NeurIPS 2023)** showed IL loss and return follow power laws in
compute for Atari/NetHack, and forecast compute-optimal model/data sizes. The closest
robotics work is **Lin et al., "Data Scaling Laws in Imitation Learning for Robotic
Manipulation" (ICLR 2025 Oral)**: >40k real demonstrations and >15k rollouts showing that
generalization scales roughly as a power law in the *number of training environments and
objects*, that **diversity dominates raw demonstration count**, and that demos *per*
environment saturate quickly (32 environments × 50 demos ⇒ ~90% on a novel
environment-object pair). Crucially, their diversity axis is counted in discrete
environments/objects and their evaluation is zero-shot transfer; they do not measure how
the demonstration requirement grows as a *single* task's initial-state distribution widens.
**Mandlekar et al. (robomimic, CoRL 2021)** ran the earliest systematic dataset-size sweeps
for offline manipulation and established the confounds we control for (demonstration
heterogeneity, observation space, policy selection). **Allshire et al. (ABC, 2026 preprint)**
add a useful sim-real bridge: simulated success rate predicts real success rate at
r ≈ 0.85–0.91, which supports sim-only scaling conclusions.

## 2. Decomposing generality into axes

**Xie et al., "Decomposing the Generalization Gap" (Factor World, ICRA 2024)** is the
methodological ancestor of our level ladder: 19 tasks × 11 factors of variation, yielding an
*ordering* of factors by generalization difficulty (on the real robot: camera pose >
table texture > lighting > distractors > background), and the observation that the gap
shrinks from ~0.4 to <0.1 when training environments grow 5→100. **THE COLOSSEUM (Pumacay
et al., RSS 2024)** perturbs 20 tasks along 14 axes, reporting 30–50% degradation per axis
and >75% when composed (sim-real correlation R² = 0.61). **Gao et al. (RSS 2024)** show
policies compose factors, so data collection can skip factor combinations (77.5% vs 2.5%
on unseen combinations). Recent work extends the axis view: the **★-Gen taxonomy (2026)**
scores generalist policies over 13 axes and 885 real evaluations; **Qi et al., "Scale Up
Strategically" (2026 preprint)** formalize per-factor *bias* (colour ≥ object ≥ spatial ≥
verb ≥ size) and find that under a small budget, concentrating demonstrations on fewer
factors beats spreading them thinly; **Macaluso et al., "Inductive Generalization" (2026
preprint)** build progressive difficulty ladders very much like ours but evaluate
*outside* the training support, where policies collapse (97% → 0–4%) and where extra
in-distribution MimicGen data does not help. All of these hold data roughly fixed and vary
the axis; none price an axis in demonstrations.

## 3. Automated demonstration generation — our data source, and a cost we can measure

**MimicGen (Mandlekar et al., CoRL 2023)** generates 50k+ demonstrations from ~200 human
demos by re-applying object-centric end-effector segments, and defines task variants
**D0/D1/D2** with progressively broader reset distributions — the nearest existing
"generality ladder". But each variant is released at a *fixed* 1000 generated
demonstrations, so it answers "can policies be trained at breadth b?", not "what does
breadth b cost?". Follow-ups scale the generator rather than study the trade-off:
**SkillMimicGen** (24k demos from 60 human demos, +24% SR via cuRobo motion planning),
**DexMimicGen** (bimanual dexterous, ICRA 2025), **IntervenGen**, **DemoGen**, and
**RoboCasa (RSS 2024)**, which reports monotone gains from 100→3000 generated demos per task
and generated data beating human-only data (47.6% vs 28.8%). Our pipeline is the **Isaac
Lab 2.3 / Isaac Sim 5.1** Mimic integration (Isaac Lab; Orbit, RA-L 2023) with the
**LeRobot** (ICLR 2026) implementation of **Diffusion Policy** (Chi et al., RSS 2023 /
IJRR 2025). Architectural routes to lower data cost — e.g. **Equivariant Diffusion Policy**
(CoRL 2024), EquiBot — are complementary: they change the exponent we set out to measure.

## 4. Two adjacent literatures

*Randomization breadth.* Sim-to-real RL has long observed that wider domain randomization
buys robustness but slows learning and lowers per-domain optimality (Tobin et al. 2017 and
successors; see also "How Should a Sim-to-Real Transfer Budget Be Spent?", 2026 preprint).
The demonstration-count analogue of that trade-off is unquantified.

*Evaluation rigour.* Success-rate studies are only as good as their rollout budgets.
**TRI's LBM study (Science Robotics 2026)** used blind randomized A/B testing with 1,800
real and >47,000 simulated rollouts and found no consistent single-task-baseline
outperformance; tightening a binomial CI from ±10 to ±2 points costs ~15× more rollouts
(70 → 1,030). Recent proposals use anytime-valid inference ("Beyond Binary Success", 2026
preprint) or imperfect simulators with guarantees (SureSim, 2025). We adopt frozen
pre-sampled eval sets, Wilson intervals, and best-of-last-3 checkpoints.

## 5. The gap this study occupies

| Study | Varies distribution breadth | Varies demo count N | Reports demos-to-reach-s |
|---|---|---|---|
| Lin et al. 2025 (scaling laws) | # envs/objects (transfer) | yes, saturates | no |
| Factor World / COLOSSEUM | yes, per axis | no | no |
| MimicGen D0/D1/D2 | yes, 3 levels | no (fixed 1000) | no |
| Macaluso et al. 2026 | yes (out-of-support) | no | no |
| **This work** | **yes, 4 nested levels × 3 tasks** | **yes, 10–400 nested** | **yes, N\*(s) + cost ratios** |

---

## 6. Key contributions as they currently stand

**A controlled demos × breadth grid.** We cross a 4-level generality ladder with a
demonstration-count sweep (10–400) at a fixed 80k-step budget, giving data-cost curves
N\*(s | level) and cost ratios N\*(L_k)/N\*(L0) — a *price* in demonstrations, where prior
work gives an ordering of factors (Factor World, COLOSSEUM) or a scaling curve at one
breadth (MimicGen D0–D2, Lin et al.). Confounds are controlled by construction: identical
source demos and generator settings across a task's levels, nested subsets so curves are
monotone in data rather than resampling noise, frozen eval sets, one frozen config per cell.

**Three deliberately different tasks** carry parallel ladders — rigid pick-and-place, an
articulated drawer+stow, and a non-prehensile 20 cm puck push, the last built on a derived
"push frame" because Mimic's single-reference transform cannot express a two-body
(object *and* goal) relation. 3,200 validated demos, frozen benchmarks and per-leg
generation statistics ship with a fully pinned pipeline.

**A generation-side asymmetry, measured.** Generation success rate collapses with pose
breadth but is nearly free for appearance (drawer task 54.9 → 44.2 → 30.6 → 32.7%: object
pose −10.7 pts, fixture pose −13.6 pts, appearance ≈1 pt). Pose generality is expensive to
*learn* but cheap to *generate*; geometry generality is not cheaply generatable at all under
object-centric replay — which explains why MimicGen-style corpora vary placement rather than
shape, and bounds what they can evidence.

**An evaluation-protocol finding of general use.** Appearance variants held in per-variant
sub-environments share the pose RNG stream, so a pooled 200-episode evaluation can hold only
~20 distinct poses; reading the diagonal (variant v ← batch v) restores 200 at zero cost.

*Status:* data is complete for two tasks; the headline curves are unmeasured, with training
blocked on the cluster allocation.

## 7. Contributions worth adding (ranked by lift ÷ cost)

**A predictive breadth→data law (free).** Regress N\*(s) on a *measurable* breadth statistic
— randomization area/volume, or initial-state entropy in nats — not the ordinal level index.
"One nat costs ×k demonstrations" is budgetable before collection; an ordering of factors
never is.

**Two evaluation-only slices that change the story (free).** A 4×4 cross-level matrix turns
"what breadth costs" into what it buys: does an L3 policy match an L0 specialist at equal N?
And one small out-of-support eval per level says whether demonstrations spent on breadth buy
extrapolation or only interpolation — pre-empting the criticism Macaluso et al. (2026) level
at in-distribution scaling.

**Two-sided cost accounting (free).** Fold attempts-per-usable-demo from `gen_stats.csv`
into N\*(s) for a cost per generality level in GPU-hours per success point; given the
asymmetry above, that ledger is not what either side predicts alone.

**Three cheap run-funded extras.** An appearance-only level decomposes axis costs and tests
whether they compose (~4 cells). Re-training anchor cells on the recorded low-dimensional
state localizes the cost — "X% of pose-generality cost is perceptual" (~6 runs). Three seeds
on ~4 anchor cells give a per-cell SD and bootstrap CIs on N\*, repairing the design's
weakest point (~8 runs, ~65 GPU-h).

## References

Allshire et al., *Scalable Behavior Cloning with Open Data, Training, and Evaluation*,
arXiv:2606.27375, 2026 (preprint) · Cadene et al., *LeRobot*, ICLR 2026, arXiv:2602.22818 ·
Chi et al., *Diffusion Policy*, RSS 2023 / IJRR 2025, arXiv:2303.04137 · Etukuru et al.,
*Robot Utility Models*, arXiv:2409.05865, 2024 · Gao, Xie, Xiao, Finn, Sadigh, *Efficient
Data Collection for Robotic Manipulation via Compositional Generalization*, RSS 2024,
arXiv:2403.05110 · Garrett et al., *SkillMimicGen*, CoRL 2024, arXiv:2410.18907 · Jiang et
al., *DexMimicGen*, ICRA 2025, arXiv:2410.24185 · Lin, Hu, Sheng, Wen, You, Gao, *Data
Scaling Laws in Imitation Learning for Robotic Manipulation*, ICLR 2025 (Oral),
arXiv:2410.18647 · Macaluso et al., *Inductive Generalization for Robotic Manipulation*,
arXiv:2606.20999, 2026 (preprint) · Mandlekar et al., *What Matters in Learning from Offline
Human Demonstrations* (robomimic), CoRL 2021, arXiv:2108.03298 · Mandlekar et al.,
*MimicGen*, CoRL 2023, arXiv:2310.17596 · Mittal et al., *Orbit / Isaac Lab*, RA-L 2023;
*Isaac Lab*, arXiv:2511.04831 · Nasiriany et al., *RoboCasa*, RSS 2024, arXiv:2406.02523 ·
Pumacay, Singh, Duan, Krishna, Thomason, Fox, *THE COLOSSEUM*, RSS 2024, arXiv:2402.08191 ·
Qi et al., *Scale Up Strategically*, arXiv:2607.21582, 2026 (preprint) · *A Taxonomy for
Evaluating Generalist Robot Manipulation Policies* (★-Gen), 2026 · *Beyond Binary Success:
Sample-Efficient and Statistically Rigorous Robot Policy Comparison*, arXiv:2603.13616,
2026 (preprint) · *Reliable and Scalable Robot Policy Evaluation with Imperfect Simulators*
(SureSim), arXiv:2510.04354, 2025 · Tobin et al., *Domain Randomization*, IROS 2017 ·
TRI, *A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation*,
Science Robotics 2026, arXiv:2507.05331 · Tuyls et al., *Scaling Laws for Imitation Learning
in Single-Agent Games*, NeurIPS 2023, arXiv:2307.09423 · Wang et al., *Equivariant Diffusion
Policy*, CoRL 2024, arXiv:2407.01812 · Xie, Lee, Xiao, Finn, *Decomposing the Generalization
Gap in Imitation Learning for Visual Robotic Manipulation*, ICRA 2024, arXiv:2307.03659.
