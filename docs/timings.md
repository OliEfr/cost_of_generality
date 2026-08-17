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
| T2 L2 | 400 | 30.6 % | 4 h 08 min | + cabinet pose randomized |
| T2 L3 x10 variants | 10 x 40 | 33.3 % (v00, exact) | ~4 h projected (v00 = 24 min) | + object size/colour; adds a ~3.5 min Kit boot per variant |

Planning rule of thumb that falls out of this: on T2, budget **~35 min of 4090 wall
time per 100 demos at L0 and ~62 min per 100 at L2**. Camera-enabled Kit boot is
~3.5-4 min and is paid once per generate_dataset.py invocation, so many small
per-variant jobs (L3) lose ~35 min of pure startup across ten launches.

SR figures above are exact (success + `_failed` episode counts per D16), not scraped
from generator logs — log tails understate by up to 19 demos because the final
progress flush is lost at shutdown.
