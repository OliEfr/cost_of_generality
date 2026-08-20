# GPU-hour ledger (grant EUHPC_B38_106; ceiling for this project: 2,200 GPU-h)

**Accounting unit.** Leonardo bills `AllocTRES` **billing units x hours**, and billing equals the
number of allocated CPU cores (verified 2026-08-19: `billing=8` for `cpu=8,gres/gpu=1`,
`billing=32` for `cpu=32`). The grant is 112,000 "local h" = 14,000 A100-h, i.e. **8 billing-h =
1 GPU-h**. So a standard 1-GPU/8-core cell costs 1 GPU-h per wall-clock hour, and asking for more
cores costs proportionally more -- which is why the decode bottleneck was fixed rather than
thrown cores at (D23).

| Date | Item | Job | Elapsed | billing | GPU-h | Cum. |
|---|---|---|---|---|---|---|
| 2026-08-16 | (nothing consumed -- no Slurm association) | - | - | - | 0 | 0 |
| 2026-08-19 | G0 probe (queue latency + A100 node facts) | 52869585 | 0:00:02 | 8 | 0.001 | 0.001 |
| 2026-08-19 | G5a batch/LR sweep (64/128/256) | 52878355 | 0:17:34 | 8 | 0.293 | 0.294 |
| 2026-08-19 | G5a num_workers A/B -- INVALID, cancelled | 52885440 | 0:07:23 | 32 | 0.492 | 0.786 |
| 2026-08-19 | G5a torchcodec arm -- failed (torchcodec 0.7.0 ABI) | 52894292 | 0:02:49 | 32 | 0.188 | 0.974 |
| 2026-08-19 | frame-equality check -- failed (same cause) | 52894432 | 0:02:03 | 8 | 0.034 | 1.008 |
| 2026-08-19 | frame-equality check -- BACKENDS_IDENTICAL | 52896091 | 0:00:10 | 8 | 0.003 | 1.011 |
| 2026-08-19 | G5a torchcodec arm -- 13.89 steps/s | 52896093 | 0:03:40 | 32 | 0.244 | 1.255 |
| 2026-08-19 | G5a resume test -- cancelled (bug in my test) | 52897255 | 0:04:11 | 8 | 0.070 | 1.325 |
| 2026-08-19 | G5a resume test -- PASSED | 52899246 | 0:04:07 | 8 | 0.069 | 1.394 |
| 2026-08-19 | 80k calibration = cell t1_L0_n25_s0 | 52899856 | running | 8 | ~1.6 (est) | ~3.0 |

**Bring-up total so far: ~1.4 GPU-h consumed, ~3.0 including the calibration in flight.** The plan
budgeted ~45 GPU-h for "bring-up/debug/A100 gate/batch-LR smoke"; actual is ~7% of that. Note
~0.8 GPU-h of the 1.4 was spent on runs that produced no usable number (an invalid measurement
and two ABI failures) -- cheap at this scale, and each one bought a finding.

## Re-forecast after G5a (supersedes the plan's estimate v1 of ~900 GPU-h)

The decode fix (D23) took a cell from ~10.2 h to **~2.0 h**, a 5x reduction.

**Per-cell cost is MEASURED, not extrapolated:** the calibration run wrote its 20k checkpoint
1,779 s after the training loop started = **11.2 steps/s**, so 80k steps = ~1.98 h + ~2.3 min
startup -> **~2.0 h and ~2.0 GPU-h per cell** at `--cpus-per-task=8`.

Note this is 20 % worse than the 13.89 steps/s the throughput smoke reported, because that smoke
ran in a job with `--cpus-per-task=32` while a real cell gets 8. **8 cores is still correct**:
billing is linear in cores, so 16 cores would buy ~18 % wall-clock for ~2x the cost.

| Stage | Plan v1 | Re-forecast | Basis |
|---|---|---|---|
| T1 training (23 cells remaining) | ~200 | **~46** | 2.0 GPU-h/cell measured; L0/N=25 already done as the calibration |
| T1 evals (24 last-checkpoint evals + 2 for the comparison, if on cluster) | ~25 | **~9** | D24: last checkpoint only, 3x fewer evals |
| Bring-up / gates / smokes | ~45 | **~3** | measured above |
| Tasks 2-3 training (48 cells) | ~450 | **~96** | same measured per-cell cost |
| Tasks 2-3 evals | (in the 75 above) | **~18** | D24, same basis |
| Margin ~25% + contingency | ~180 | ~43 | |
| **Total** | **~900** | **~215** | ceiling 2,200 |

D24 (evaluate only the last checkpoint at full scale) cuts evaluation from 72 checkpoint-evals to
24+2. That matters less for GPU-hours than for **wall-clock**: after the decode fix, eval had
become the binding constraint on the study's critical path, and this removes two thirds of it.

Wall-clock matters more than cost now: with a ~4 s queue latency and independent cells, the
23-cell T1 wave is a **~2 h block**, not a day. The binding constraint on the study has shifted
from GPU-hours to evaluation throughput -- which is what makes the G5b cluster-eval question
worth settling.

## ACTUALS -- T1 training wave complete (2026-08-20)

The forecast above is superseded for T1 training: it is now measured, not estimated.

| Stage | Re-forecast | **ACTUAL** | Note |
|---|---|---|---|
| T1 training | ~46 (23 cells) | **51.3 (24 cells)** | 24 not 23: D26 re-ran L0/N=25 on the new architecture |
| Bring-up / gates / smokes / calibration / superseded run | ~3 | **4.3** | includes the 2.0 GPU-h shared-encoder run kept as D24 evidence |
| T1 evals | ~9 | **0** | D25: eval runs LOCALLY on the 4090, so it costs 0 grant GPU-h |
| **Spent to date** | | **55.6** | 2.5% of the 2,200 ceiling |

Per-cell actual: 1.84-2.83 h, median 2.03 h (vs 2.0 h forecast from the shared-encoder calibration
-- so the per-camera-encoder architecture cost ~5%, not the 16% an early warm-up reading suggested).

**Revised Tasks 2-3 projection.** At the measured 2.14 GPU-h mean per cell, 48 cells = **~103
GPU-h**, plus a little for datagen debug. Total study projection: **~160 GPU-h**, i.e. ~7% of the
ceiling. The grant is not a constraint on this study at any plausible scope, including the N=800
extension arms.

**The binding constraint is eval wall-clock, not GPU-hours.** Training all 24 cells took ~2 h of
wall-clock (fully parallel, ~4 s queue latency). Evaluating them takes ~14 min each SERIALLY on the
one local 4090 -- **~5.6 h**, i.e. nearly 3x the training time, and it cannot be parallelised
because the GPU is shared with a foreign job that must not be disturbed (rule 2). For Tasks 2-3
that becomes ~11 h of local eval. This is what makes the CINECA Vulkan question (docs/
cineca_ticket_vulkan.md) worth pursuing: it is worth ~11 h of wall-clock, not any GPU-hours.

## 2026-08-20 -- actuals after the L3 rerun (D27/D28)

Measured from `experiments/registry.csv` (`gpu_h` filled from sacct elapsed; on Leonardo a 1-GPU
cell allocates `billing=8` and 8 billing-h = 1 A100-h, so elapsed hours == GPU-h exactly).

| arm | cells | GPU-h |
|---|---|---|
| T2/T3 L0-L2 | 36 | 78.6 |
| T1 (complete, incl. the pose-redundant L3 ablation) | 25 | 53.3 |
| L3b corrected arm (T1 + T3, still running) | 12 | 15.9 |
| **T2/T3 L3 -- cancelled, wasted** | 12 | **13.3** |
| bring-up / smoke | 6 | 2.3 |
| **spent so far** | **91** | **163.4** |

**Projection to completion:** the 12 L3b cells in flight finish around 2.2 h each (~26 GPU-h total,
of which 15.9 is already counted) and T2's 6 L3b cells add ~13, so the study lands at roughly
**190 GPU-h -- about 9% of the 2,200 ceiling.** The grant remains a non-constraint at any plausible
scope, including N=800 extension arms if they turn out to be warranted.

**The 13.3 GPU-h wasted** is the honest cost of D27: 12 T2/T3 L3 cells trained for ~1:06 each on the
pose-redundant datasets before the bug was found, and were cancelled rather than run to 80k (which
would have cost ~28 GPU-h more for numbers that could not be used). T1's equivalent arm is NOT
counted as waste -- it completed, it is retained as the pose-diversity ablation, and its flat
plateau plus 4x-low training loss are what exposed the bug.

**The binding constraint is still local eval wall-clock, not GPU-hours**, and it got worse: 36 T2/T3
L0-L2 evals plus 18 L3b diagonal evals (200 episodes each) on one shared 4090. Eval also has to
interleave with L3b datagen on the same card. This is the case for pursuing the CINECA Vulkan
question (`docs/cineca_ticket_vulkan.md`, drafted, unsent): it buys wall-clock, not GPU-hours.
