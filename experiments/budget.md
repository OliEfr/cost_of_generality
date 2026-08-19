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
