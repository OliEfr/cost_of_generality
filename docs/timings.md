# Measured timings (RTX 4090 workstation, tueilsy-st-07)

All values MEASURED on 2026-08-16 unless marked (est). Update this file whenever a
new operation type is timed, or a measured value shifts >30% — future sessions plan
against these numbers (CLAUDE.md rule 5 applies).

## Fixed overheads

| What | Time | Notes |
|---|---|---|
| Isaac Kit app startup (headless, `--enable_cameras`) | ~3.5–4 min | dominates any job < 15 min; batch work per app launch |
| Isaac Kit app shutdown after `env.close()` | often HANGS | benign: outputs are written first; wrap in `timeout -s KILL` and treat exit 137 as OK after verifying the artifact |
| One `gym.make` per Kit session | hard limit | a second make in the same app hangs — loop levels via separate app launches |

## Simulation / data generation (8 parallel envs, 128px dual-cam unless noted)

| Operation | Measured | Rate |
|---|---|---|
| Expert episode (scripted SM, success) | ~190 env steps = 9.5 s sim | |
| Source recording, 1 env, state-only (15 demos) | < 10 min incl. startup | |
| annotate --auto, 15 eps | ~8 min incl. startup | |
| Mimic gen, visuomotor, 400 successes (L0) | **22 min 09 s** | ~18/min after startup |
| Mimic gen, visuomotor, 400 successes (L1) | **22 min 11 s** | GPU exclusive |
| Mimic gen, 400 successes, GPU shared with a training run (L2) | **32 min 08 s** | ~45% slower under sharing — acceptable |
| L3 wave: 10 × (Kit start + gen 40 successes @ 8 envs) | **25 min 34 s** (~2.6 min/variant) | Measured 2026-08-16, GPU otherwise idle. The prior ~65–70 min estimate assumed a cold ~4 min Kit start per variant; with warm shader/extension caches a Kit start is only ~1.5–2 min. Use the warm number for back-to-back sim jobs, the cold number for first launch after reboot/env change. |
| Full 4-level datagen for one task (L0+L1+L2+L3 wave) | **~1 h 42 min** measured total (22+22+32+26) | L2 leg was GPU-shared; ~1.5 h if exclusive |
| frames_qa per level | ~7–8 min | PNG written before the shutdown hang |
| Kit start, STATE-only headless (no cameras/RTX) | **~3 s** (!) | The ~3.5–4 min figure is for camera-enabled visuomotor sessions only — do not budget minutes for state-env legs |
| Eval-set freeze, one sub-level (state env, 20 envs, 10 seeded resets) | **~6.5 s** incl. Kit start | file-mtime cadence, 2026-08-17 |
| Eval-set freeze wave, 13 sub-levels serial | **84 s** total | 00:42:01→00:43:25 |
| T2 Mimic gen (state, 12 successes @ ~30% SR, 4 envs) | ~10 min | long episodes ~650 steps |
| T2 datagen, 400 successes visuomotor (L0) | **2 h 17 min** @ ~54.5% gen SR | 8.1 GB; wave SR ≈ 1.8× the state-env smoke |
| T2 datagen full wave (3×400 + 10×40 visuomotor) | est ~8-10 h (measured L0 leg × SR-adjusted) | was est 13-16 h pre-measurement |

## GPU utilization (RTX 4090, measured 2026-08-16/17)

| Workload | GPU util | VRAM (process) | Note |
|---|---|---|---|
| Mimic visuomotor generation, 8 envs (T2 wave) | **56-65%** | ~6.4 GB | steady across L0/L1 legs; room for a co-located job |
| DP training, bs 64 (G4 smoke, shared with gen) | data-bound | ~8 GB | updt_s 0.085 vs data_s 0.26: pyav decode is the bottleneck, GPU mostly idle-waiting |
| State-env expert/eval runs | <20% | ~5-6 GB | negligible load; safe to run beside generation |
| Idle overheads observed | — | foreign eval job ~1.6 GB; orphaned frames_qa PID 2083049 ~4.8 GB | orphan killable by user |

Practical: generation + a light state-env job co-exist comfortably; generation +
training both fit in VRAM (~21/24 GB with the orphan still resident) at ~40-50%
mutual slowdown (measured on the T1 L2 leg).

## Conversion / validation (CPU, nice -n 10, sharing box with sim)

| Operation | Measured |
|---|---|
| HDF5 → LeRobotDataset v3, 400 eps (h264) | ~20 min |
| validate_dataset (counts + 20-frame pixel check) | ~1 min |
| Resulting dataset size | 3.5 GB HDF5 → **81 MB** LeRobot (~40×) |

## Training (lerobot diffusion, bs 64, 128px, pyav backend)

| Operation | Measured | Notes |
|---|---|---|
| 5k steps @4090 SHARED with gen job | 29.5 min = **2.8 steps/s** | data-bound: updt_s 0.085, data_s 0.26 |
| Pure GPU step time | 0.085 s/step (=11.7 steps/s ceiling) | throughput is dataloader-bound (pyav decode); more workers / faster decode is the lever, not the GPU |
| 80k steps @4090 exclusive (est) | ~4.5–6.5 h | measure once before relying on it |
| 80k steps @A100 (est) | G5a will measure | budget placeholder ~8 GPU-h/run |

## Planning rules of thumb

1. Anything Isaac-based costs ~4 min before the first useful step; batch accordingly.
2. Sub-hour jobs: schedule generously (`timeout` 1.5–2× the estimate) — startup
   variance and shutdown hangs, not the workload, cause most timeouts.
3. GPU sharing (gen + train) works on the 4090 (~21/24 GB) and costs each side
   ~40–50%; fine for smokes, avoid for benchmark runs whose wall time is reported.
4. Generation SR ~85–86% at L0–L2 → plan ~1.17× attempts per target success.

### T2 (drawer + stow) datagen wave — measured 2026-08-17

Visuomotor Mimic, `--num_envs 8`, `--enable_cameras`, 4090 shared with a foreign
eval job. Episodes are 618-743 steps (31-37 s at 20 Hz) vs ~350 for T1, so a T2
attempt costs roughly 2x a T1 attempt before the SR difference is even counted.

| Leg | demos | gen SR | wall time | notes |
|---|---|---|---|---|
| T2 L0 | 400 | 54.9 % | 2 h 16 min | fixed cabinet + fixed object pose |
| T2 L1 | 400 | 44.2 % | 2 h 50 min | + object pose randomized |
| T2 L2 | 400 | 30.6 % | 4 h 08 min |
| **T2 wave total** | **1600** | — | **13 h 10 min** (03:32-16:42, unattended) | + cabinet pose randomized |
| T2 L3 x10 variants | 10 x 40 | 32.7 % pooled (33.3 % size s / 32.0 % size m) | 3 h 54 min (23-24 min each) | + object size/colour; adds a ~3.5 min Kit boot per variant |

Planning rule of thumb that falls out of this: on T2, budget **~35 min of 4090 wall
time per 100 demos at L0 and ~62 min per 100 at L2**. Camera-enabled Kit boot is
~3.5-4 min and is paid once per generate_dataset.py invocation, so many small
per-variant jobs (L3) lose ~35 min of pure startup across ten launches.

SR figures above are exact (success + `_failed` episode counts per D16), not scraped
from generator logs — log tails understate by up to 19 demos because the final
progress flush is lost at shutdown.

**Source of truth for generation SR and demo counts: `experiments/gen_stats.csv`**
(regenerate with `python scripts/dev/gen_stats.py`). The tables in this file are a
human-readable digest; if they ever disagree with the CSV, the CSV is right because it
is recomputed from the HDF5 pairs.

### T2 HDF5 -> LeRobot conversion — measured 2026-08-17

Four levels run **concurrently** as separate tmux sessions (`convert_t2_all.sh <KEY>`).
h264 encode is single-core, so one level per core; the box has 32 threads and load
stayed ~4-5.

| | episodes | frames | output | wall (parallel) |
|---|---|---|---|---|
| T2_L0 | 400 | 281,987 | 356 MB | 2 h 02 min |
| T2_L1 | 400 | 277,661 | 351 MB | 1 h 58 min |
| T2_L2 | 400 | 270,744 | 343 MB | 1 h 53 min |
| T2_L3 | 400 (10 x 40) | 270,745 | 344 MB | 1 h 53 min |

All four `VALIDATE_OK` with `--expect_episodes 400`.

**Planning numbers:** ~3.5 episodes/min or ~2,500 frames/min per core for T2-length
episodes (~680 frames). One level is ~2 h; running the four in parallel costs the same
2 h instead of 8 h, so **always convert levels concurrently**. T1's shorter episodes
(~190 frames) convert roughly 3.5x faster per episode.

### T3 (push-to-target) pipeline — measured 2026-08-17

| step | cost |
|---|---|
| source recording, 20 demos, 1 env, state env | ~13,600 steps, ~12 min (expert SR 0.69 vs the strict 2 cm recording gate) |
| annotation (`--auto`), 20 -> 17 demos | ~2 min |
| generation smoke, state env, 12 demos, 4 envs | ~2 min, 100 % gen SR |
| generation smoke, VISUOMOTOR, 40 demos, 8 envs | ~5 min, **93.0 % gen SR** (40/43) |

Episodes are 265-342 steps (13-17 s at 20 Hz) -- roughly half T1's step count and a
**quarter** of T2's. Combined with a 93 % generation SR versus T2's 31 %, a T3 level of 400
demos should cost well under an hour, against T2's 2-4 h per level.

### T3 (push-to-target) datagen wave — measured 2026-08-18

Visuomotor Mimic, `--num_envs 8`, `--enable_cameras`, 4090 shared with the foreign eval job.

| Leg | demos | gen SR | wall |
|---|---|---|---|
| T3 L0 | 400 | 98.5 % | 29 min |
| T3 L1 | 400 | 94.8 % | 32 min |
| T3 L2 | 400 | 95.0 % | 31 min |
| T3 L3 x10 variants | 10 x 40 | 88.5 % pooled | 3-4 min each |
| **T3 wave total** | **1600** | — | **2 h 09 min** |

Compare T2: same 1600 demos, **13 h 10 min**. The 6.1x difference is ~2.2x from episode
length (311-319 steps vs 677-705) and ~2.8x from generation SR (95 % vs 31-55 %).

**Planning rule:** budget ~30 min of 4090 wall time per 400-demo T3 level, against ~2-4 h
for T2 and ~25 min for T1. A camera-enabled Kit boot is ~3-4 min and is paid once per
`generate_dataset.py` invocation, so the ten L3 variants spend ~35 min of their ~70 min in
startup.

---

## Cluster (Leonardo, A100-SXM-64GB, driver 535.274.02) — measured 2026-08-19

### Queue latency

| Measurement | Value |
|---|---|
| `boost_qos_dbg` submit -> start | **4 s** (submitted 17:30:12, started 17:30:16) |
| Queue depth at the time | 1,575 pending / 2,050 running |
| `sbatch --test-only` prediction | 6 days out -- **wrong by five orders of magnitude** |

**Planning rule: treat the queue as empty.** `--test-only` answers "when could this start if
nothing ahead finished early and no backfill happened", which on a machine full of short jobs
bears no relation to reality. Never plan from it; submit a 2-second probe instead.

### Training (diffusion policy, batch 64, 2x128x128 cams, L0/N=25)

| Configuration | data_s | updt_s | steps/s | 80k steps |
|---|---|---|---|---|
| pyav backend, 8 workers | 0.388 | 0.071 | 2.18 | ~10.2 h |
| **torchcodec backend, 8 workers, 32 cores allocated** | **0.003** | 0.069 | **13.89** | ~1.6 h |
| **torchcodec, 8 workers, 8 cores allocated (a REAL cell)** | -- | -- | **11.2** | **~2.0 h** |

**Planning rule: budget ~2.0 h and ~2.0 GPU-h per cell** at `--cpus-per-task=8`. The 13.89
figure came from a smoke job that happened to request 32 cores; a real cell gets 8 and runs
~20 % slower. Measured end-to-end from checkpoint timestamps, and stable: 20k steps in 1,779 s
then the next 20k in 1,774 s. Do not use the 1.6 h number.

First interval is NOT representative: step 25 reports `updt_s 1.152, data_s 0.480` while the
decoder cache is cold and CUDA autotunes; by step 50 it is at the steady 0.069/0.003. Any
throughput measurement must discard the first logging interval.

Startup overhead per job: ~2.3 min (torch import from Lustre + dataset creation + wandb init).

### Container operations

| Operation | Time | Notes |
|---|---|---|
| `docker pull` isaac-sim:5.1.0 (local) | ~10 min | 15.1 GB |
| `docker save` -> tar | ~4 min | 15,123,856,384 B |
| `rsync -z` tar to `$WORK` | ~7 min | 7.54 GB on the wire, 2.01x compression |
| `singularity build` from docker-archive, **compute node** | **13 min 55 s** | ~19.4 GB rootfs unpack (~10 min) + squashfs to 7.1 GB (~6 min), 16 cores, 0.46 GPU-h |
| same conversion on a **login node** | **FAILED** | shared by ~100 users, capped memory; also floods with Lustre xattr warnings |

**Planning rule: convert images in a compute job, never on a login node.** A compute node's own
`/tmp` is only ~10 GB (against ~35 GB needed for a 15 GB archive), so scratch still falls back to
Lustre -- the win is RAM and dedicated cores, not local disk.

### Gate / smoke costs (all `boost_qos_dbg`, billing = allocated cores)

| Job | Elapsed | GPU-h |
|---|---|---|
| node/queue probe | 0:02 | 0.001 |
| batch/LR sweep (3 arms x 200 steps) | 17:34 | 0.29 |
| dataloader A/B arm | 3:40 | 0.24 (32 cores) |
| frame-equality check (pyav vs torchcodec) | 0:10 | 0.003 |
| resume-across-requeue test | 4:07 | 0.07 |
| sif conversion | 13:55 | 0.46 |

**Planning rule:** a dbg-QOS smoke costs ~0.1-0.5 GPU-h. At that price, measure rather than
reason -- every wrong assumption found today (queue depth, batch scaling, decode cost, resume,
container permissions) was found by a job costing less than half a GPU-hour.

## Per-camera RGB encoders (D26) -- measured 2026-08-19, T1 matrix in flight

| architecture | steps/s | 80k projection | source |
|---|---|---|---|
| one shared RGB encoder | 11.2 | 2.0 h (measured exactly 02:00:04) | G5a calibration, job 52899856 |
| separate encoder per camera | 9.45 | 2.35 h | live read of job 53008600 at step 7,200 |
| **separate, settled** | -- | **median 2.01 h, mean 2.14 h** | all 24 T1 cells, sacct elapsed |

**Use the last row.** The 2.35 h projection was read at step 7,200, inside warm-up, and it
overstated the cost. Across all 24 completed T1 cells: total 51.3 GPU-h, mean 2.14 h, median
**2.01 h**, range 1.84-2.83 h. So the median cell is indistinguishable from the shared-encoder
baseline (2.00 h) and the mean penalty is **+6.8%**, carried by a few slow cells rather than by the
architecture -- consistent with the loop being decode-bound (GPU util ~0% at every batch size, D23)
and with 24 jobs contending for the same filesystem. The earlier "~16% throughput penalty" claim was
an artifact of the warm-up reading and is withdrawn.

Planning rule: **2.2 h and 2.2 GPU-h per cell** (median plus headroom); a 24-cell wave is ~52
GPU-h. The 12 h walltime per cell keeps ~5x margin even on the slowest cell observed.

**Method note, reusable:** throughput of a RUNNING job is readable without waiting for a checkpoint
and without syncing wandb. Offline mode writes no `files/config.yaml`, and `_redirect()` leaves the
Slurm `.out` nearly empty, but the `.wandb` datastore has `history` records with `_step` and
`_timestamp`. `scripts`-side reader: `read_wandb_run.py` (job tmp), using
`wandb.sdk.internal.datastore` + `wandb.proto.wandb_internal_pb2`. This is the only live progress
signal for these jobs -- do not judge liveness from log growth.

### Correction (2026-08-20): the -16% throughput penalty was a warm-up artefact

The 9.45 steps/s figure above was read at **step 7,200**, i.e. during warm-up. Settled per-cell
wall-clock tells a different story:

| | elapsed | gpu_h |
|---|---|---|
| shared encoder (calibration, job 52899856) | 02:00:04 | 2.0 |
| separate encoders (`t1_L0_n25_s0`, job 53008611) | ~02:06 | 2.1 |
| all 24 matrix cells | 01:50:24 - 02:05:43 | 1.84 - 2.1 |

So the real cost of per-camera encoders is **~5%, not 16%** -- the shared-encoder run at 2:00:04 sits
*inside* the matrix's own spread. The early reading was measured before the page cache warmed and
overstated the penalty by 3x. Recorded because the 16% number was already written down here, and
because it is a reminder that a rate sampled in the first few minutes of a decode-bound job is not
that job's rate.

Matrix total to date: **~50 GPU-h** across 31 registry rows (includes partial elapsed for the 5
cells still running, so expect it to settle near 52). Still under the original 48-GPU-h estimate's
order of magnitude and far under the 2,200 ceiling.

## L3 regeneration + T2/T3 evaluation -- measured 2026-08-20

**Datagen, per L3 variant (local 4090, `--num_envs 8`, 40 successes each):** mean episode length is
the driver, since generation cost scales with simulated steps.

| task | mean ep len (steps) | per variant | 10 variants | size |
|---|---|---|---|---|
| T1 cup_place | ~188 | ~2.8 min | ~28 min | 3.1 GB |
| T3 push_target | ~310 | ~3.5 min | ~35 min | 5.1 GB |
| T2 drawer_stow | ~675 | ~23 min | **~3.9 h** | 7.3 GB |

**LeRobot conversion (400 episodes, h264, both cameras):** ~14 min for T1 L3b, consistent with the
~20 min recorded for T1 in 2026-08-16. For T2 do NOT extrapolate from that -- the measured T2 figure
is **~1 h 53 min per level** (see the T2 conversion table above), because T2 episodes are ~680
frames against T1's ~190 and h264 encode is single-core. T2_L3b will take about two hours.

**Checkpoint sync-down ($WORK -> local, 080000 only):** ~65 s per cell (1.1 GB).

**Rollout eval, 100 episodes (5 batches x 20 envs) -- REVISED, and much worse than the T1 number:**

| task | max_steps | measured | note |
|---|---|---|---|
| T1 cup_place | 600 | ~8 min | the figure earlier planning used |
| T2 drawer_stow | 1200 | **43 min** | `t2_L0_n10`, 12:42:28 -> 13:25:27, GPU shared with L3b datagen |
| T2 drawer_stow, GPU otherwise free | 1200 | **22-24 min** | five consecutive evals 2026-08-21 (t2fu queue); use this for exclusive-GPU planning |
| T3 push_target | 800 | not yet measured | expect ~15-25 min |

T2's episode limit is 2x T1's and its policy must be queried at every step of it, so the eval is
~5x slower, not 2x -- part of that is sharing the card with L3b datagen, but the step count is the
dominant term. **This invalidates the "~5 h to work through the 36 T2/T3 evals" estimate.** Serial
local projection:

* 18 T2 L0-L2/L3b cells x ~43 min (L3b's 200-episode diagonal is ~2x again) -> **~13-20 h**
* 18 T3 cells -> ~6-10 h
* T1's 6 L3b diagonal evals -> ~2.5 h

i.e. **on the order of 25-35 h of serial local eval**, against ~190 GPU-h of cluster training that
completes in a couple of wall-clock hours. Planning rule: **budget local eval at 45 min/cell for T2,
20 min for T3, 10 min for T1**, and double it for an L3 diagonal (200 episodes, 10 Isaac boots).

This is now by far the study's binding constraint and it is the strongest argument yet for the
CINECA Vulkan question (`docs/cineca_ticket_vulkan.md`, drafted and unsent): getting Isaac to render
on A100s would move ~30 h of serial wall-clock onto a cluster that runs cells in parallel. It buys
wall-clock, not GPU-hours.

First real T2 number, for the record: **T2 L0 N=10 -> SR 0.65 (65/100)**, versus T1 L0 N=10's 0.85.

### Two concurrent evals (user-authorised 2026-08-20)

With L3b datagen finished, the 4090 has room for two simultaneous evals: each rollout process holds
~7 GB and the foreign job ~1.6 GB, so two run at ~16 GB of 24.5 GB. `COG_EVAL_MIN_FREE_MIB` lowered
**14000 -> 10000**, which admits exactly two and still blocks a third (two running leaves ~8.5 GB
free, under the threshold). Verified live: both drivers reported `headroom OK` within a minute and
ran side by side.

Why 14000 was right before and is not now: it was set while datagen held ~6 GB, where it correctly
prevented a second eval from squeezing the foreign job. The margin protecting that job is now the
~8.5 GB left with two evals running, which is ample.

Effect on the remaining queue: the serial estimate was **~30 h** for 41 outstanding cells (T2 flat
43 min/cell measured, T2 diagonal ~113 min derived, T3 flat ~20 min estimated, T3/T1 diagonal 53 min
measured). Two slots should give **~15-20 h** rather than exactly half, because the two processes
contend for SM time as well as memory -- the earlier datagen-sharing measurement showed ~40-50 %
mutual slowdown, so treat 15 h as the optimistic end.

Do not raise this to three concurrent evals: 3 x 7 + 1.6 = 22.6 GB of 24.5 leaves under 2 GB, and
Isaac's allocation spikes during scene load.
