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
