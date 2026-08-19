# Environment reproducibility — audited 2026-08-19

Honest status, per environment. `PARTIAL` and `NO` are stated as such deliberately: the software
stack is part of the method for the paper, so overstating this would be a defect in the write-up.

| Environment | Where | Recipe | Frozen record | Status |
|---|---|---|---|---|
| `cog_lerobot` (cluster **training**) | `$WORK/cog/miniforge3/envs` | `scripts/ops/build_cluster_env.sh` | `cluster_cog_lerobot.freeze.txt` (111 pkgs) | **YES**, with one caveat below |
| `cog_isaac` (local **eval / datagen**) | `~/miniconda3/envs` | none — built interactively in P1 | `local_cog_isaac.freeze.txt` (313 pkgs) | **PARTIAL**: exact versions recorded, no script |
| `cog-env-5.1.0.sif` (cluster eval image) | `$WORK/cog/containers` | `docker/Dockerfile.cog_env` | the image itself (9.8 GB) | **PARTIAL**: needs `cog_isaac` to exist |
| `cog_lerobot06` (py3.12 experiment) | `$WORK/cog/miniforge3/envs` | `scripts/ops/build_cluster_env_lr06.sh` | none | throwaway; used once to answer the 0.5/0.6 question |

## What the audit found and fixed

**Gap 1 (fixed).** `build_cluster_env.sh` did **not** install `ffmpeg` or `torchcodec==0.5`. Those
were applied by hand during debugging, so the script did not reproduce the environment that
actually ran the calibration — it would have produced one that trains **5x slower**, because
`COG_VIDEO_BACKEND=torchcodec` (frozen in `configs/train/diffusion_base.sh`) cannot load without
them. Both are now in the script, with a decode check rather than an import check, because
torchcodec 0.7.0 imports fine and still fails on first decode.

**Gap 2 (documented, not fixed).** `cog_isaac` has no build script. It was assembled
interactively in P1 (isaacsim 5.1.0 pip + IsaacLab v2.3.0 editable + lerobot 0.4.4). It can be
reconstructed from `local_cog_isaac.freeze.txt`, but that reconstruction has **never been
executed**, so calling it reproducible would be a claim without evidence. Two mitigations exist:
the freeze file pins all 313 packages, and `cog-env-5.1.0.sif` is a byte-level snapshot of the
whole env, already on the cluster.

## Known reproducibility limits (stated, not hidden)

- **miniforge is fetched from `/latest`** in the build script, so a rebuild months later gets a
  different conda/python base. Observed at build time: conda 26.3.2, python 3.11.15, constructor
  dated 2026-06-01. Override `COG_MINIFORGE_URL` to pin. The freeze file, not the script, is the
  authoritative record of what ran.
- **`ubuntu:24.04` in `docker/Dockerfile.cog_env` is a moving tag**, so its apt set drifts. Pin by
  digest if an exact image rebuild is ever needed.
- **`Dockerfile.cog_env` copies the live `cog_isaac` env** via `--build-context`. If that env is
  lost the image cannot be rebuilt — only the existing `.sif` survives. This is the strongest
  argument for keeping the sif.
- **The two envs deliberately differ in one package**: cluster `torchcodec==0.5`, local
  `torchcodec==0.10.0` (broken, unused — local eval renders from Isaac and never decodes video).
  The video backend affects training data loading only, never the checkpoint, so this divergence
  cannot affect results. See D22/D23.

## Regenerating the freeze records

```bash
# cluster
ssh leonardo 'source $WORK/cog/miniforge3/etc/profile.d/conda.sh && conda activate cog_lerobot \
  && python -c "import sys;print(\"python\",sys.version.split()[0])" && pip list --format=freeze' \
  > docs/env/cluster_cog_lerobot.freeze.txt
# local
~/miniconda3/envs/cog_isaac/bin/python -c "import sys;print('python',sys.version.split()[0])" \
  > docs/env/local_cog_isaac.freeze.txt
~/miniconda3/envs/cog_isaac/bin/pip list --format=freeze >> docs/env/local_cog_isaac.freeze.txt
```
