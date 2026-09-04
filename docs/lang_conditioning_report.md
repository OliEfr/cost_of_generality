# Language-conditioned policy setting — investigation report (D30)

**Goal** (user directive, 2026-08-22): find and validate the best way to run a
language-conditioned policy in this repo, for a later full-study rerun with a
language-diversity disturbance dimension (20 frozen synonym instructions per task,
identical at train and eval). Not a study; every result lives in
`results/diagnostics/`. Both candidates were built, smoke-tested, and full-train
verified (80k steps, frozen 100-demo T1-L1 cell, baseline `t1_L1_n100_s0` SR 0.86
Wilson95 [0.779, 0.915]).

---

# FINAL DECISION — Candidate A (user-confirmed 2026-09-04)

**The full-study rerun uses candidate A: the lightweight env-state-embedding setting
(frozen CLIP text embedding conditioning the unchanged, frozen Diffusion Policy).**
Candidate B was ruled out on training cost (11.71 vs 2.21 GPU-h per cell — 5.3×).
There is no single-paper citation for a policy this lightweight; A is cited as a
composition of first-class references — see **"How to cite the setting"** below.
B stays fully provisioned in-repo as the escape hatch (directly citable as
Barreiros et al., Science Robotics 2026).

---

## Verdict at a glance

| | Candidate A — env-state embedding | Candidate B — multi_task_dit backport |
|---|---|---|
| Mechanism | frozen CLIP ViT-B/16 text embedding (512-d, unit-norm) written per frame as `observation.environment_state`; stock pinned DiffusionPolicy conditions on it natively (FiLM/global-cond) | TRI-LBM-style DiT (lerobot 0.5.2 policy) run on pinned 0.4.4 as in-repo plugin `src/lerobot_policy_mtdit`; CLIP text+vision towers, tokenizes the dataset `task` string |
| New model code | **zero** (dataset-driven; config = frozen flags verbatim) | 3 copied modules + 12-line shim + 5 import fixes |
| **Train time / cell (80k, A100)** | **2:12:21 = 2.21 GPU-h** (10.4 steps/s; baseline 2.0–2.2 h — no measurable cost) | **11:42:35 = 11.71 GPU-h** (1.91 steps/s, batch 64) — **5.3× A** |
| VRAM (train) | ~13.5 GiB (baseline regime) | 41.5 GiB at batch 64; **batch 128 OOMs the A100-64GB** |
| Verify-cell SR vs baseline 0.86 [0.779,0.915] | **0.830 [0.745, 0.891] — PASS** (CIs overlap) | 0.700 [0.604, 0.781] — below A; CI barely touches baseline's. Untuned: batch 64 < the policy's recommended 192–320 (128 OOMs the A100-64), policy-default lr/horizon; treat as a lower bound, not B's ceiling |
| Multi-task probe (language steers?) | **PASS decisively**: T1 match 0.930 / swap 0.060; T3 match 0.980 / swap 0.040 | not run (optional; A's pass covers the mechanism question for A) |
| Cluster deps | none (embeddings baked into datasets; offline-safe by construction) | transformers 4.57.6 in `cog_lerobot` + CLIP staged in `$WORK/cog/hf_cache/hub` (done, offline load verified) |
| Eval integration | one extra batch key, in-process (~8 min/T1 eval) | same in-process path (plugin import + factory dispatch); **10.3 min/T1 eval** — inference is not a blocker |
| Hyperparams vs frozen study config | **identical** (rule-7 flags untouched) | policy's own preset (lr 2e-5, DiT 512/6/8, RoPE, resize 256/crop 224, horizon 20 / act 16 @20 Hz, ONE shared CLIP encoder per camera — deliberate asymmetry vs D26) |
| Instruction-set semantics | needs re-conversion to change instructions (embeddings baked into dataset) | reads strings at train time (new instructions = same dataset, new tokenization) |

## Recommendation (user-confirmed — see Final decision above)

**Candidate A is the setting to use for the full-study rerun**, unless the rerun's
scientific framing requires the policy to *read raw text at train time*:

1. **Training cost**: A is free (2.21 vs 2.2 h baseline — the study's 2.2 GPU-h/cell
   planning rule survives verbatim). B costs ~5.3× per cell; a 72-cell rerun goes
   from ~160 GPU-h to ~850 GPU-h (both affordable within the 112k budget, but B also
   moves every wall-clock estimate).
2. **Comparability**: A keeps the frozen architecture and hyperparams (rule 7); the
   language dimension is then the ONLY change vs the existing surface. B changes
   policy family, LR, horizon, crop, encoders — a language cost estimated under B is
   confounded with the architecture swap unless the whole surface is re-run under B.
3. **Conditioning is proven live under A**: the probe's +0.87/+0.94 match−swap deltas
   with non-overlapping CIs eliminate the "policy silently ignores the embedding"
   risk — the one real scientific objection to A.
4. B remains valuable as the escape hatch (stronger language grounding, upstream
   architecture) and is fully provisioned: plugin, configs, sbatch, cluster deps,
   throughput measured. Switching later costs nothing but the GPU hours.

Caveat for A on the record: a frozen text embedding is representation-level
conditioning — the policy never sees tokens, so paraphrase robustness is bounded by
CLIP's text-encoder geometry (unit-norm 512-d; within-task cosine 0.89–0.94). For
the *disturbance* framing (same 20 instructions at train and eval) this is exactly
sufficient; for held-out-instruction generalization claims, B (or a trainable text
projection) would be the better instrument.

## How to cite the setting

**Candidate A (the chosen setting) has no single proposing paper — it is a standard
composition, cited by its parts.** Quotable template for the paper:

> We use a language-conditioned variant of Diffusion Policy (Chi et al., RSS 2023):
> a frozen CLIP text embedding (Radford et al., ICML 2021) of the instruction is
> appended to the policy's observation conditioning, following the
> frozen-language-embedding paradigm of language-conditioned imitation learning
> (Stepputtis et al., NeurIPS 2020; Lynch & Sermanet, RSS 2021; BC-Z, Jang et al.,
> CoRL 2021) and RT-1's FiLM-based conditioning (Brohan et al., RSS 2023).

Architecture sentence (mechanism verified in the installed lerobot 0.4.4 source,
`modeling_diffusion.py:761-784`): the global conditioning vector — proprioception
(9) + visual features (64/camera × 2 cameras) + frozen CLIP instruction embedding
(512), over 2 observation steps, plus a 128-d diffusion-timestep embedding = 1426-d
— **FiLM-modulates every one of the U-Net's 14 residual blocks**, each block owning
its own `Linear(1426 → 2×channels)` scale-and-bias projection
(`use_film_scale_modulation=true`). Cite FiLM as **Perez et al., AAAI 2018**.
Composition precedent (diffusion × language-conditioning as a standard design
point): X-IL, arXiv:2502.12330.

**Candidate B (escape hatch) is directly citable as a named policy**: TRI LBM Team
(Barreiros et al.), *A Careful Examination of Large Behavior Models for Multitask
Dexterous Manipulation*, **Science Robotics 2026**, arXiv:2507.05331 — already in
`docs/literature_review.md`; the LeRobot `multi_task_dit` implements that
architecture. Transparency caveats if ever used: it is a community reimplementation
(not TRI's code), and our config adapts it (20 Hz horizons, 128→224 px upscale,
batch 64 forced by A100-64GB VRAM vs the recommended 192–320).

Considered and rejected for citability-at-low-cost: **SmolVLA** (Shukor et al.,
2025) ships in the pinned lerobot and is citable, but it is a *pretrained* VLA —
fine-tuning it breaks the study's from-scratch fixed-budget protocol. RT-1 or the
LIBERO baseline policies (Liu et al., NeurIPS 2023) are lightweight and named, but
neither exists in lerobot 0.4.4 — adopting one means more new engineering than the
B backport was, plus a policy-family change.

Literature caveat worth one sentence in the paper: frozen language embeddings can
underperform fine-tuned ones on LIBERO-style multi-task splits (wrong-task
completions reported, e.g. X-IL). Our swap probe is the direct empirical cover:
with frozen embeddings, matched instructions score 0.930/0.980 while swapped
instructions collapse to 0.060/0.040 — the conditioning demonstrably selects the
task in this setting.

## The frozen instruction benchmark

`configs/instructions/instructions_v1.json` + `embeddings_clip_vit_b16_v1.npz`
(CLIP ViT-B/16 @ 57c2164, 512-d, unit-norm) + `SHA256SUMS`; validated by
`scripts/dev/validate_instructions.py` (INSTRUCTIONS_OK). Rule-8 semantics: never
mutate — new version file + new `*_i<n>` dataset roots. Index 0 = the canonical
string every pre-i20 dataset carries. Cosine geometry: within-task means
0.89/0.94/0.91 (T1/T2/T3); closest cross-task pair is cup_place×push_target (mean
0.842) — the probe deliberately used that pair and still separated fully.

## Gates run (all green)

| Gate | Result |
|---|---|
| H1 instructions frozen | INSTRUCTIONS_OK; checksums committed |
| H2/H3 datasets | L1_i20 / T3_L1_i20 / probe_T1T3_L1_i20 all VALIDATE_OK; **episode order identical to baseline roots** (makes SR comparisons valid); nested-N instruction balance exact |
| H4 eval-harness regression | flag-off reproduces baseline SR 0.860 exactly (4/100 episode flips = GPU nondeterminism, both directions); flag-on inert on a language-less ckpt (0.870, within CI) |
| A1 smoke | loss 0.87→0.10 @1k; saved config carries ENV[512]; **U-Net cond width 402→1426 = exactly +1024** (the anti-silent-failure assert); smoke eval end-to-end with env_state injection |
| B1 smoke (local) | registration + synthetic forward (conditioning dim 4114 includes the 512-d text branch), draccus dry-parse, 300-step train (loss 0.50→0.13), DDIM-10 reload + 20-env select_action, throughput/VRAM probe |
| B provisioning | transformers 4.57.6 on cluster (PINS row), CLIP staged, HF_HUB_OFFLINE dress rehearsal OFFLINE_CLIP_OK, dbg smoke on compute node (plugin + offline CLIP + torchcodec under slurm) |
| A2/A3 verify | 2.21 GPU-h; SR 0.830 [0.745,0.891] vs 0.86 [0.779,0.915] — PASS |
| A4/A5 probe | PASS both envs (see table); matched SR ≥ single-task baselines |
| B2/B3 verify | B2 COMPLETED 11:42:35 (11.71 GPU-h); B3 SR 0.700 [0.604,0.781], per-instruction 0.40/0.70/1.00 |

## Probe detail (candidate A, one policy on T1+T3, n=100/task, 2.08 GPU-h)

| env | instructions | SR | Wilson95 |
|---|---|---|---|
| T1 cup_place | match | 0.930 | [0.863, 0.966] |
| T1 cup_place | swap (push_target) | 0.060 | [0.028, 0.125] |
| T3 push_target | match | 0.980 | [0.930, 0.994] |
| T3 push_target | swap (cup_place) | 0.040 | [0.016, 0.098] |

PASS criteria (matched ≥0.50, delta ≥0.30, disjoint CIs) exceeded by ~3×. Caveat: a
swap collapse proves causal influence, not semantic parsing; per-instruction matched
spread 0.80–1.00 shows no paraphrase is broken. Multi-task training showed no
penalty vs single-task baselines (possibly mild positive transfer on T1).

## How to run the rerun with candidate A (operational recipe)

1. Convert each level with `--instructions configs/instructions/instructions_v1.json`
   → `<LEVEL>_i20` roots (existing HDF5 reused, no Isaac time; ~35–75 min/level on
   local CPU, 3-wide). Validate with `validate_dataset --instructions
   --match_order <baseline root>`. NOTE: L3-style same-task multi-file merges are
   hard-refused (instruction↔variant confound) — an L3 lang arm needs its own
   assignment design first.
2. Train via `slurm/train_lang_a.sbatch TASK LEVEL NDEMOS` (sources the frozen flags;
   `COG_RUN_ID`/`COG_DS_NAME` overrides for special cells). 2.2 GPU-h/cell holds.
3. Eval via `scripts/ops/run_lang_eval.sh RUN_ID t1|t2|t3 LEVEL OUT.json [STEP]`
   (assignment (batch+env)%20 on the frozen protocol; per-instruction SR recorded).
   For study cells, move results out of diagnostics/ and extend the curves.py
   naming contract deliberately (D29-style decision) — NOT done in this
   investigation on purpose.
4. A 1-instruction control arm is safe by construction (ENV features bypass
   normalization → no zero-variance hazard).

## Artifacts

- Code: converter/validator `--instructions`, eval injection + type dispatch,
  `run_lang_eval.sh`, `lang_report.py`, `read_wandb_run.py`, plugin
  `src/lerobot_policy_mtdit/`, configs `lang_diffusion_a.sh` / `lang_dit_b.sh`,
  sbatches `train_lang_a` / `train_lang_dit` / `smoke_mtdit`.
- Datasets: `data/lerobot/{L1_i20,T3_L1_i20,probe_T1T3_L1_i20}` (+ cluster copies).
- Checkpoints: `t1_L1_i20_n100_s0`, `t1t3_probe_i20_n200_s0`,
  `t1_L1_i20_n100_s0_mtdit` (all on `$WORK/cog/checkpoints`; 080000 synced locally).
- Results: `results/diagnostics/eval_lang_*`, `probe_candA_*`.
- Registry rows: `lang_a_i20`, `probe_i20a`, `lang_b_mtdit` variants. Decision D30;
  journal 2026-08-22/23; PINS rows (CLIP, cluster transformers); timings sections.

## Final numbers

- Investigation GPU-h spent: A2 2.21 + probe 2.08 + B2 11.71 + dbg smokes ~0.2 ≈
  **16.2 GPU-h** (budget impact negligible).
- B3 note: 0.700 with CIs [0.604, 0.781] means B as-configured is measurably behind
  both A (0.830) and the baseline on this cell. The multi_task_dit docs themselves
  recommend batch ≥192 and >30k–100k steps with tuning; batch 64 was forced by the
  A100-64GB OOM at 128. A fair tuned-B comparison would need larger-memory GPUs or
  gradient accumulation — out of scope here and unnecessary given A's result.
