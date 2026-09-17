# Running IsaacLab eval on Leonardo (A100)

How to run a rollout evaluation on the cluster, why each piece is needed, and what it costs.
Everything here was established on 2026-09-17 (journal entry of that date); the working
reference job is `slurm/bench_eval_a100_dbg.sbatch`.

**Status: works, but is not the default.** D25 keeps evaluation on the local 4090. This page
exists so the cluster path can be picked up without re-deriving it.

---

## 1. Why this was blocked until 2026-09-17

Three independent problems, each of which alone produced "Isaac cannot render on A100":

| # | symptom | cause | fix |
|---|---|---|---|
| 1 | `vkCreateInstance` -> `ERROR_INCOMPATIBLE_DRIVER` | system SingularityPRO 4.3.1 cannot expose the NVIDIA Vulkan driver here **and** the container has no NVIDIA ICD on a loader search path | run under the FoldSpace toolbox **Apptainer 1.5.3** and bind the host ICD dir |
| 2 | Kit: "installed driver 535.18 is unsupported" | Kit misparses driver `535.274.02` as `535.18` (NVIDIA's documented misreport for 535.255+) | `--/rtx/verifyDriverVersion/enabled=false` |
| 3 | every `conv2d` -> `CUDNN_STATUS_NOT_INITIALIZED` | the image was built from `cog_isaac`, which has **both** `nvidia-cudnn-cu12 9.7.1.26` and `nvidia-cudnn-cu13 9.20.0.48` sharing one lib dir, so the **CUDA-13** build wins; it cannot initialise on Leonardo's CUDA-12 driver 535 | bind the cluster training env's cu12 cuDNN over the container's |

Plus one environmental constraint that is not a bug: **compute nodes have no internet**, so any
USD asset referenced from the Omniverse S3 stalls 300 s and then fails.

> Note on #3: this also means the **local** `cog_isaac` env is one driver change away from the
> same breakage. It works today only because the workstation runs driver 580 (CUDA 13 capable).
> Cleanup if wanted: `pip uninstall nvidia-cudnn-cu13` inside `cog_isaac`.

---

## 2. Prerequisites (one-time, already done)

- **Assets staged.** `$WORK/cog/isaac_assets/Assets/Isaac/5.1` holds the four subtrees T1
  references (Franka, SeattleLabTable, Mug, Grid; 118 MB). Re-run from a **login node** (compute
  nodes are offline) with `scripts/dev/stage_isaac_assets.py`; pass extra S3 prefixes as argv to
  add assets for other tasks. **T2/T3 have not been checked for additional assets** -- expect to
  stage more before evaluating them.
- **Container.** `$WORK/cog/containers/cog-env-5.1.0.sif`.
- **Writable Kit dirs.** `$WORK/cog/kit_rw/{logs,data,cache}`, seeded from the image on first use
  (the sbatch does this automatically). Kit writes into its own install tree, which is read-only
  inside a `.sif`.
- **libgomp shim.** `$WORK/cog/extralibs` on `LD_LIBRARY_PATH`; the image lacks `libgomp1`, which
  alone caused 99 of 101 startup errors.

---

## 3. The invocation

```bash
FS=/leonardo/prod/opt/tools/foldspace/1.0
APPTAINER=$FS/apptainer/bin/apptainer          # NOT the system `singularity`
COG=$WORK/cog
ENV_IN=/home/admin_07/miniconda3/envs/cog_isaac/lib/python3.11/site-packages/isaacsim
NV_LIBS=/home/admin_07/miniconda3/envs/cog_isaac/lib/python3.11/site-packages/nvidia
CUDNN_OK=$COG/miniforge3/envs/cog_lerobot/lib/python3.11/site-packages/nvidia/cudnn/lib
ASSET_ROOT=$COG/isaac_assets/Assets/Isaac/5.1

$APPTAINER exec --nv \
  -B $COG:$COG -B /leonardo_work:/leonardo_work \
  -B /usr/share/vulkan/icd.d:/etc/vulkan/icd.d \        # fix 1: NVIDIA ICD on a search path
  -B $CUDNN_OK:$NV_LIBS/cudnn/lib \                     # fix 3: CUDA-12 cuDNN over the CUDA-13 one
  -B $COG/kit_rw/logs:$ENV_IN/kit/logs \
  -B $COG/kit_rw/data:$ENV_IN/kit/data \
  -B $COG/kit_rw/cache:$ENV_IN/kit/cache \
  -B $COG/extralibs:/opt/extralibs \
  --home $COG/isaac_home \                              # NOT --env HOME=, apptainer refuses that
  --env OMNI_KIT_ACCEPT_EULA=YES --env HF_HUB_OFFLINE=1 \
  --env PYTHONPATH=<repo>/src \
  $COG/containers/cog-env-5.1.0.sif bash -c '
    export LD_LIBRARY_PATH=/opt/extralibs:'"$NV_LIBS"'/cudnn/lib:$LD_LIBRARY_PATH
    python -u -m cog.eval.rollout_eval \
      --task Cog-CupPlace-L1-IK-Rel-Visuomotor-v0 \
      --checkpoint <ckpt>/pretrained_model \
      --num_inference_steps 10 --max_steps 600 \
      --out <out>.json --headless --enable_cameras \
      --kit_args="--/rtx/verifyDriverVersion/enabled=false \
                  --/persistent/isaac/asset_root/default='"$ASSET_ROOT"' \
                  --/persistent/isaac/asset_root/cloud='"$ASSET_ROOT"'"'
```

Use `slurm/bench_eval_a100_dbg.sbatch` rather than retyping this. It also runs a conv2d
pre-flight and refuses to benchmark a cuDNN-disabled fallback.

---

## 4. Gotchas that cost real time here

- **No inner `srun`.** A step inside the allocation does not inherit `--gres`, so torch reports
  "No CUDA GPUs are available". Run the container directly from the batch shell.
- **`--kit_args` needs the `=` form.** `--kit_args "--/rtx/..."` makes argparse read the value as
  the next option and exit 2 before Kit starts.
- **Kit can hang after a Python exception**, holding the Slurm slot until walltime (observed: 30
  min). Budget for it when the dbg QOS 2-job limit matters.
- **Do not rewrite the ICD JSON.** August's absolute-path rewrite makes the Apptainer loader fail
  `vk_icdGetInstanceProcAddr` ("Found no drivers"). The stock host ICD, bind-mounted, works.
- **Kit clears Vulkan loader env vars** (`VK_ICD_FILENAMES` etc.), so only the bind works.
  `VK_LOADER_DEBUG` does still produce output under Apptainer, which is how #1 was cracked.
- **Judge artifacts, not exit codes** (D6): Kit exits 0 after fatal errors, and `fastShutdown`
  can swallow buffered stdout -- write results to a file.

---

## 5. QOS and scheduling

| QOS | walltime | concurrent jobs | observed start latency (2026-09-17) |
|---|---|---|---|
| `boost_qos_dbg` | 30 min | 2 | **~2 min** |
| normal | up to 24 h | -- | **never started** while 3402 ran / 11524 queued |

A T1 cell needs ~26 min, so it fits `boost_qos_dbg` only marginally. Longer cells (T2, and any
200-episode L3 diagonal) require normal QOS and are therefore exposed to the queue.

---

## 6. Measured timings

One standard cell, T1 `t1_L1_n100_s0` @80k, frozen protocol (20 envs/batch), DDIM-10,
`max_steps 600`. Cluster figures from job 58049698 (4 of 5 batches completed inside the 30-min
walltime); local figures from `docs/timings.md`.

| | per episode | per 20-episode batch |
|---|---|---|
| local RTX 4090, card shared | ~4.8 s | -- |
| local RTX 4090, card empty | ~2.4-3.6 s | -- |
| **cluster A100-SXM-64GB** | **15.9 s** | **319 s** steady, 262 s first (cold) |

Startup, cluster: Kit app-ready **13 s** warm cache / **36 s** cold; scene creation **0.6-1.7 s**
with assets staged (**300 s then failure** without).

**Per episode the A100 is ~3.3x slower than the shared 4090 and ~5x slower than an empty one.**
The A100 has no RT cores, so Isaac's RTX renderer dominates; policy inference is milliseconds on
either card. Correctness is unaffected: running SR was 0.812 after 80 episodes against 0.86
recorded locally for the same cell.

Cost of the whole enabling investigation: **~0.7 GPU-h** across 7 dbg jobs.

---

## 7. When to use this

Per cell the cluster is strictly slower, so it can only win through running many cells at once,
against a queue that was badly congested when this was measured. Treat it as an option for a
large rerun when eval wall-clock is binding and the queue is quiet -- not as the default. D25
(eval local, zero grant GPU-h) stands.
