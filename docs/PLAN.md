# The Cost of Generality — Final Study Plan

## Context

Paper-level research study: **what does generality cost in robot manipulation imitation learning, measured in demonstrations?** For each task we define generality levels (breadth of the training/eval distribution) and, per level, train vision-based diffusion policies at multiple demo counts N, measuring rollout success rate. Deliverable: the demos-vs-success surface per level and derived data-cost curves N*(s | level) = demos needed to reach success rate s — plus a paper-level research report.

Constraints (user-set): Isaac Sim/Isaac Lab simulation; demos via Isaac Lab Mimic (MimicGen integration); LeRobot standard diffusion policy, vision obs (state recorded too for a later ablation), 1 seed/run, **training budgeted in fixed steps**; 3 quite-different tasks, **start with Task 1 (cup→target)**; local workstation for debug/sim, CINECA Leonardo for GPU runs; **evals also on cluster if easily possible**; strict no-delete outside project repo (ask user for cleanup instead); don't touch existing code/envs; hourly health watchers for long jobs; rlfv-brain findings are NOT transferable — ignored.

**User decisions (2026-08-16):** Task 2 = open drawer + stow object (articulated); Task 3 = non-prehensile push-to-target. Demo grid capped at 400 (extend to 800 only for unsaturated levels). GPU budget approved: **~2,200 GPU-h** of the shared 14,000. Scope priority: **Task 1 exhaustively first**, then replicate Tasks 2–3 as time allows (grant ends 2026-10-29).

## User actions needed (P0 — please do these now)

1. **Slurm association (BLOCKER):** ohausdoe has no Slurm association with the active grant `euhpc_b38_106` (verified: `sbatch --test-only` fails; only the expired B34 account is associated). PI `aschoell` must add you to the project in CINECA UserDB. Everything cluster-side waits on this; local phases P1–P4 proceed regardless.
2. **Cert renewal Monday:** current Leonardo cert expires **2026-08-18 ~08:33**. Renewal needs your laptop (OAuth tunnel: `ssh st07` with `LocalForward 10000`, then `~/cineca_login.sh`). The watchdog will warn at <12 h remaining.
3. **B34_046 data at purge risk:** 2.6 TB sits in the expired `/leonardo_work/EUHPC_B34_046/max` area (subject to purge 6 months after project end). Not this project's data — decide whether to archive.

## Scientific design

### Task 1: cup/mug placement (Franka, IK-Rel actions)
Scene: table + Franka + mug + **visible flat goal marker** (colored disk, no collision — target must be perceivable by the vision policy). Obs: table cam + wrist cam, 128×128 RGB (identical resolutions — LeRobot requirement) + full state (recorded, unused by the vision policy). Success: mug base within 5 cm XY of goal center AND upright (tilt <30°) at episode end; timeout ~350 steps.

### Generality levels (single source of truth: `levels.py`; user-set granularity 2026-08-16: one init-pose level, one merged object level, no continuous sweep)
| Level | Randomization |
|---|---|
| L0 | mug pose fixed, goal fixed, 1 red mug |
| L1 | + mug XY in 30×40 cm, yaw ±90° |
| L2 | + goal XY in 20×20 cm |
| L3 | + object varied: 4 mug meshes × 5 colors × scale 0.9–1.1 |

Tasks 2 (drawer+stow; Isaac Lab cabinet assets + `open_cabinet_sm.py` expert to adapt) and 3 (push-to-target; no grasp — different contact profile) get analogous 4-level ladders defined after Task 1 (P8).

### Protocol
- **Demo counts** N ∈ {10, 25, 50, 100, 200, 400}, +800 extension only for unsaturated levels. **Nested subsets** (shuffle once with seed 0; dataset N = first N episodes) so curves are monotone in data, not resampling noise.
- **Demo provenance control:** same 10 scripted source demos + same Mimic generator settings for every level of a task (heterogeneity confound control). **Mimic generation success rate per level is logged and reported as a finding.**
- **Training:** fixed **80k steps** (user directive), identical explicit hyperparams for every cell (resnet18, crop 112 from 128, DDPM-100 train, seed 0; `horizon`/`n_obs_steps`/`n_action_steps` set explicitly after reading the pinned DiffusionConfig source — docs conflict on defaults). **Batch size & LR set by a P5 A100 smoke test (user directive: GPU utilization should be high)** — scale batch up as far as A100-64GB VRAM/throughput allow (e.g., 64→128→256), scaling LR by sqrt(batch ratio) from the 1e-4/64 baseline; verify short-run loss parity, then freeze for ALL cells. Checkpoints every 20k.
- **Eval:** frozen pre-sampled eval sets per level (200 configs committed as JSON; standard eval = first 100 episodes; binomial SE ±5 pts at p=0.5), in-distribution. Evaluate last 3 checkpoints (40/60/80k) with DDIM `num_inference_steps=10` set explicitly; primary metric = best-of-last-3 (last-ckpt SR also reported). 200-episode reruns for headline cells.
- **Parallel execution (no adaptive ladder):** with fixed steps, a run costs ~8 GPU-h regardless of N, so serial skip-on-saturation saves almost nothing — submit the **full 4×6 grid (24 runs) as one parallel wave** of independent 1-GPU jobs. One follow-up wave only if needed: N=800 extension for unsaturated levels (+≤4 runs) and reruns. **~24–28 training runs total.**
- **Metrics:** SR(N | level) with Wilson 95% CIs; logistic fit in log N; N*(s) for s ∈ {50, 80, 90}% (report "> N_max" honestly); cost ratios N*(L_k)/N*(L0).

## Infrastructure (verified 2026-08-16)

**Local (tueilsy-st-07):** RTX 4090 24 GB (live foreign eval job PID 1796345 — never kill, coordinate GPU; ~21 GB free), 32 threads, 61 GB RAM, 559 GB free disk (single fs, disk-pressure history → 150 GB project budget, ask user if exceeded). Existing IsaacSim 4.2 / IsaacLab 0.27.15 (`isaaclab` env, dirty checkout) untouched. Docker+NVIDIA toolkit, tmux, empty crontab.

**Leonardo:** `ssh leonardo` works (cert 48 h, helper `~/cineca_login.sh --status`). Grant EUHPC_B38_106 until 2026-10-29: 112,000 core-h = 14,000 A100-h unused. Storage: $WORK 3 TB (empty), $FAST 1 TB NVMe, $SCRATCH (40-day purge), $HOME 50 GB. boost_usr_prod (4×A100-64 per node), 24 h walltime (lprod QOS: 4 days); 1-GPU jobs `--gres=gpu:1 --cpus-per-task=8 --mem=64G` (accounting scales with request). **No compute-node internet** → `WANDB_MODE=offline` + login-node `wandb sync`; `HF_HUB_OFFLINE=1` + pre-staged caches; USD assets pre-staged (no Nucleus). `singularity` (not apptainer) on all nodes. Cluster cron disabled → watchers run from workstation. Module python 3.11 only → own miniforge in $WORK.

**Stack pins:** Isaac Sim 5.1.0 (pip, pypi.nvidia.com) + Isaac Lab v2.3.0 + py3.11 + torch 2.7.0/cu128 (matches local driver 580.173.02; Isaac Lab 3.0 beta avoided). LeRobot: latest v0.6.1 needs py≥3.12 while Isaac needs 3.11 → **G1b decision**: pin the newest LeRobot that installs under py3.11 with LeRobotDataset v3 + diffusion training, used identically for train+eval; if none exists, train on py3.12 env and eval via a small ZMQ policy-server bridge (~150 lines). All pins recorded in `docs/PINS.md`.

**A100 rendering gate (G5b):** Isaac Sim officially unsupported on A100 (no RT cores; issues #3421 crash / #1519 degraded rendering; RenderCfg antialiasing-off workaround). One 30-min `boost_qos_dbg` Singularity test decides: PASS → demo-gen/eval also on cluster; FAIL → sim stages stay on the 4090 (accepted fallback; training unaffected either way).

## Repository & data layout

New git repo `/home/admin_07/cost_of_generality/` (package `cog`):
- `src/cog/tasks/cup_place/`: `cup_place_env_cfg.py` (ManagerBasedRLEnvCfg), `cup_place_mimic_env.py` (ManagerBasedRLMimicEnv subclass, 6 methods; reference `FrankaCubeStackIKRelMimicEnv`), `cup_place_mimic_cfg.py` (SubTaskConfigs: grasp → place, last subtask offset ends at 0), `levels.py`, `assets.py`, `state_machine.py` (scripted expert adapted from `lift_cube_sm.py`, Warp-based)
- `src/cog/datagen/` (`record_source_demos.py`, `generate_level.py` wrapping annotate/generate), `src/cog/convert/` (`hdf5_to_lerobot.py` with nested-subset logic + mandatory `finalize()`; `validate_dataset.py` incl. replay-in-sim spot check), `src/cog/eval/` (`rollout_eval.py` batched fixed-seed loop; `eval_sets.py`), `src/cog/analysis/` (`curves.py`, `figures.py`)
- `configs/` (train/diffusion_base.yaml — all hyperparams explicit; levels/*.yaml; eval_sets/*.json frozen+committed), `slurm/` (`train.sbatch`, `eval.sbatch`, `debug_a100_render.sbatch`), `scripts/ops/` (`sync_up.sh`, `sync_down.sh`, `watchdog.sh`, `launch_matrix.py`), `experiments/` (`registry.csv`, `budget.md`), `docs/` (`journal.md`, `PINS.md`, `decisions.md`), `paper/`, `third_party/IsaacLab` (fresh v2.3.0 checkout, gitignored), `data/` (gitignored)

Cluster: `$WORK/cog/{repo,miniforge3,envs,checkpoints,wandb_offline,results,containers,hf_cache,datasets_backup}`; `$FAST/cog/datasets/` (read-hot LeRobot shards, <20 GB); $SCRATCH ephemeral only. Sync via rsync over `ssh leonardo` (code+datasets up; result CSVs + pruned best+last checkpoints down; wandb synced on login node).

Disk budget: local ≤150 GB (HDF5 ~8 GB/level × 4 datasets/task + LeRobot + scratch); cluster ~60 GB checkpoints after pruning — comfortable.

## Phases & gates

- **P0 user actions** (above) — parallel to P1–P4, cluster work blocked until G0 (`sbatch --test-only` succeeds).
- **P1 local env (~2–4 h, mostly download/first-launch time):** `cog_isaac` conda env (isaacsim pip ~20 GB + first-launch shader/extension caching); Isaac smoke tests (`--headless --enable_cameras`, stock `lift_cube_sm.py`); verify camera tensor dtype/layout empirically; resolve LeRobot pin (G1b). **G1:** headless vision rendering works locally.
- **P2 Task-1 env + expert (~0.5–1.5 d; the genuinely uncertain phase):** env cfg, levels, assets, pick-place state machine, RecorderManager wiring (`concatenate_terms=False`). Coding is hours; the slack is iteration in a slow-booting sim to tune grasp reliability across randomization. **G2:** expert SR ≥90% on L0–L2; 10 source demos replay correctly; camera streams verified (marker visible).
- **P3 Mimic generation + QA (Mimic subclass ~2–4 h; generation 1–3 d wall, compute-bound, day+night parallelized in tmux):** per level: auto-annotate same 10 sources → generate to 400+ successes; QA (visual grids, action ranges, coverage plots); freeze eval sets. Wall time is rendering throughput, not effort — later phases start as soon as L0 data exists. **G3:** ≥400 clean demos/level; generation SR ≥30% everywhere (else fix subtask offsets first); QA pass.
- **P4 converter + pipeline validation (~2–4 h + a short local train; overlaps P3):** converter, dataset validation incl. replay-in-sim; explicit train config drafted (batch/LR finalized in P5); local end-to-end test (L0/N=25, 5k steps, 10 rollouts). **G4:** end-to-end pipeline yields nonzero SR.
- **P5 cluster bring-up (~1 d wall, mostly unattended; needs G0; overlaps P3):** miniforge in $WORK, offline wandb/HF (~1–2 h); **batch-size/LR utilization smoke on one A100** (dbg QOS: sweep batch 64/128/256 with sqrt-scaled LR, measure VRAM, it/s, GPU util, short-run loss parity → freeze the config); then one full 80k run on L0/N=25 (~8 h unattended) → **calibrates real GPU-h/run, updates budget.md**; resume-across-requeue tested (**G5a**). In parallel: Singularity image + `debug_a100_render.sbatch` (**G5b**, user-confirmed: test Isaac on cluster, fall back to local if it fails).
- **P6 matrix execution (~2–4 d wall, queue permitting):** `launch_matrix.py` submits the **entire 24-run grid in parallel** (independent 1-GPU jobs; ~8 h each → training wave completes in ~1 day at reasonable queue depth). Evals: on cluster (G5b pass) all 72 checkpoint-evals also run as parallel jobs (~1 day); local fallback serializes on the 4090 (~2–3 days, interleaved with Task-2 datagen). Optional second wave (N=800 extensions + reruns) adds ~1 day. Hourly watchdog; `saldo -b` ledger each wave. **G6:** all Task-1 cells green; budget within ask.
- **P7 analysis + report (1–2 wk, overlaps P6):** curves, N*(s), scaling fits; paper-level report (abstract/intro/related work/method/setup/results/discussion/limitations; Figs: SR-vs-N per level, N* vs level, cost ratios, generation-SR vs level) as repo `paper/` + published artifact.
- **P8 Tasks 2–3:** reuse pipeline; per task the serial work is implementation (env cfg + expert + Mimic subclass + levels, ~1–2 d incl. tuning) and local datagen (~1–3 d wall, compute-bound); training + eval is again **one parallel wave (~1–2 d)**. Effective ~3–5 d per task, heavily overlapped: Task-2 implementation/datagen starts while the Task-1 wave trains. Scope per remaining budget/calendar (Task-1-deep-first directive).

**Overall timeline (wall-clock, phases overlapped):** P1+P2 ≈ 1–2 d → P3 datagen 1–3 d (P4 + P5 run inside it) → Task-1 wave + evals ≈ 1–2 d (+1 d if an extension wave is needed) → analysis/report ≈ 2–3 d (starts as results land). **Task 1 fully analyzed ≈ 1–1.5 weeks from start; all three tasks ≈ 2.5–3.5 weeks** — the binding constraints are demo-generation rendering throughput, one 8 h training wave per task, cluster queue times, and P0 (Slurm association). Debugging surprises are the main upside risk on these numbers.

## Budget (estimate ~1,000 GPU-h; approved ceiling 2,200 ≈ 16% of grant)

| Stage | GPU-h |
|---|---|
| Task 1 training (24-run grid + ext./reruns, ~8 h @80k steps, larger batch) | ~200 |
| Task 1 evals (if cluster; else free locally) | ~25 |
| Bring-up/debug/A100 gate/batch-LR smoke | ~45 |
| Tasks 2–3 (extrapolated) | ~450 |
| Margin ~25% + demo-gen contingency | ~180 |
| **Total** | **≈900** (approved ceiling 2,200 stands — headroom for surprises) |

Demo generation runs locally on the 4090 **day and night, parallelized** (user directive): vectorized `--num_envs` within each generation job plus concurrent per-level jobs up to VRAM headroom (watchdog keeps them alive and respects the foreign eval job's memory). Zero grant cost; ~2–4 days wall per task (4 datasets per task — one per level).

## Orchestration & monitoring

- `train.sbatch TASK LEVEL NDEMOS`: `-A euhpc_b38_106 -p boost_usr_prod --gres=gpu:1 --cpus-per-task=8 --mem=64G -t 24:00:00`, auto-`--resume` if checkpoints exist; output `$WORK/cog/checkpoints/t1_L3_n100_s0/`.
- `experiments/registry.csv` (committed): run_id, task, level, variant, n_demos, dataset_id, slurm_jobid, status, sr_40k/60k/80k/best, eval_set, eval_n, gpu_h, notes.
- **Watchdog** (hourly, workstation cron + agent-side monitors): cert check (`cineca_login.sh --status`; WARN <12 h, CRITICAL <4 h → alert user; Slurm jobs unaffected, only sync/monitoring stalls); `squeue/sacct` → registry updates; login-node `wandb sync`; local `nvidia-smi` (PID 1796345 untouched; our tmux datagen sessions alive), `df -h` local + `cindata` cluster; weekly `saldo -b`.
- **Failure playbook:** TIMEOUT/NODE_FAIL → resubmit (auto-resume). FAILED ×2 → mark blocked, human look. Never auto-delete.

## Risk register (top items)

| Risk | Mitigation |
|---|---|
| Isaac won't render on A100 | 30-min G5b gate + RenderCfg workaround; default posture hybrid (sim local/train cluster) so only eval placement changes; robosuite/MuJoCo fallback exists but off-default |
| LeRobot py3.11/3.12 split | G1b: single pin for train+eval, else ZMQ policy-server bridge; decided before any dataset is written |
| GPU contention with live local job | check VRAM headroom before/while launching, cap num_envs + concurrent jobs accordingly, never kill PID 1796345 |
| Local disk | 150 GB budget, watchdog df alerts, no-delete rule, ask user for cleanup |
| Mimic gen SR too low at L3 (full variation) | cap attempts 3× target, tune offsets at G3, report gen-SR as finding |
| Demo heterogeneity confound | same sources + generator per task, nested subsets, provenance reported |
| Run-cost uncertainty (6–16 h) | G5a calibration before matrix; adaptive skipping; weekly ledger |
| Grant deadline 2026-10-29 | Task-1 matrix launched by ~mid-Sept; Tasks 2–3 scoped by remaining time (user chose Task-1-deep-first) |
| 1 seed/run (user directive) | best-of-last-3 ckpts, frozen eval sets, nested datasets; noted in limitations |

## Verification

- Each phase has an explicit gate (G0–G6) with a concrete pass command/criterion; no phase proceeds on an unverified assumption.
- Dataset integrity: `validate_dataset.py` (stats, alignment, replay-in-sim reproduces success).
- Pipeline: G4 end-to-end local test before any cluster hours are spent; G5a calibration run before the matrix.
- Results: fixed committed eval sets; Wilson CIs; registry + budget ledger committed to git; report numbers traceable to registry rows.
