# SPEC 06 — lerobot==0.4.4 (pinned): dataset creation, diffusion policy training, inference

Source of truth: wheel `lerobot-0.4.4-py3-none-any.whl` unpacked at
`/home/admin_07/.claude/jobs/62ff5703/tmp/lerobot_pkg/src/lerobot/` (all file refs below are relative to that `lerobot/` dir).
Dataset codebase version written by this release: **`CODEBASE_VERSION = "v3.0"`** (`datasets/lerobot_dataset.py:83`) — v3 layout (chunked parquet + concatenated mp4), NOT the old v2.1 per-episode layout.

---

## 1. Dataset creation API (`datasets/lerobot_dataset.py`)

### 1.1 `LeRobotDataset.create()` — exact signature (line 1641)

```python
@classmethod
def create(
    cls,
    repo_id: str,                     # any "namespace/name" string; purely a label if you never push
    fps: int,
    features: dict,
    root: str | Path | None = None,   # -> HF_LEROBOT_HOME / repo_id if None
    robot_type: str | None = None,
    use_videos: bool = True,
    tolerance_s: float = 1e-4,
    image_writer_processes: int = 0,
    image_writer_threads: int = 0,    # >0 spawns AsyncImageWriter (recommended: threads=4*num_cams)
    video_backend: str | None = None, # decode backend; default get_safe_default_codec() = "torchcodec" if importable else "pyav"
    batch_encoding_size: int = 1,     # episodes to accumulate before video encoding (1 = encode each save_episode)
    vcodec: str = "libsvtav1",        # ENCODE codec: "libsvtav1"(default!) | "h264" | "hevc" | "auto" | hw variants
    metadata_buffer_size: int = 10,   # meta/episodes rows buffered before parquet flush
    streaming_encoding: bool = False, # True = encode frames on the fly (no PNG round-trip), save_episode near-instant
    encoder_queue_maxsize: int = 30,
    encoder_threads: int | None = None,
) -> "LeRobotDataset"
```

Where data lands: `root` if given, else `HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", Path(HF_HOME)/"lerobot"))` (`utils/constants.py:66-67`) i.e. default `~/.cache/huggingface/lerobot/<repo_id>`. **`create()` requires the root dir to NOT exist** (`LeRobotDatasetMetadata.create` does `obj.root.mkdir(parents=True, exist_ok=False)`, line 518) — delete or pick a new dir on rerun.

On-disk layout (v3.0):
```
root/
├── meta/info.json                 # fps, features, paths templates, totals, chunks_size=1000,
│                                  # data_files_size_in_mb=100, video_files_size_in_mb=200
├── meta/stats.json                # aggregated global stats (rewritten after EVERY save_episode)
├── meta/tasks.parquet             # task string -> task_index
├── meta/episodes/chunk-000/file-000.parquet   # per-episode metadata + per-episode stats (buffered writes!)
├── data/chunk-000/file-000.parquet            # frames of MANY episodes appended into one file (<=100MB)
└── videos/<video_key>/chunk-000/file-000.mp4  # MANY episodes concatenated per mp4 (<=200MB)
```

### 1.2 `features` dict format

`features` = `{key: {"dtype": ..., "shape": tuple, "names": ...}}`. `create()` merges in `DEFAULT_FEATURES`
(`datasets/utils.py:73`): `timestamp`(float32,(1,)), `frame_index`, `episode_index`, `index`, `task_index` (int64,(1,)) — do NOT define these yourself, and do NOT put them in `add_frame`. Keys must not contain `/`.

For a Franka cup-place task with 2 cameras (example, image size H=W=224 arbitrary):

```python
features = {
    "observation.state": {"dtype": "float32", "shape": (8,),  # exact key REQUIRED by diffusion policy
        "names": ["j1","j2","j3","j4","j5","j6","j7","gripper"]},
    "action": {"dtype": "float32", "shape": (8,),             # exact key "action" REQUIRED
        "names": ["j1","j2","j3","j4","j5","j6","j7","gripper"]},
    "observation.images.table_cam": {"dtype": "video", "shape": (224, 224, 3),
        "names": ["height", "width", "channels"]},
    "observation.images.wrist_cam": {"dtype": "video", "shape": (224, 224, 3),
        "names": ["height", "width", "channels"]},
}
```

- `dtype: "video"` → frames written as temp PNGs (or streamed), encoded to mp4 at `save_episode`; nothing image-related in parquet. `dtype: "image"` → PNGs embedded into the parquet as HF `datasets.Image` (bigger, slower; use "video").
- Image `shape` convention is **(H, W, C)** with `names=["height","width","channels"]` — this is what `hw_to_dataset_features` (`datasets/utils.py:616`) writes and what `dataset_to_policy_features` (`utils.py:700`) converts to CHW for the policy (it checks `names[2] in ["channel","channels"]`). The frame validator (`validate_feature_image_or_video`) accepts an ndarray in either (C,H,W) or (H,W,C).
- Vector features: `dtype` must be a valid numpy dtype string ("float32","int64",...); shape/dtype are STRICTLY validated per frame — passing float64 or shape (8,1) raises.
- `"string"` dtype exists (skipped in stats). Diffusion policy needs exactly: `observation.state` (STATE), ≥1 `observation.images.*` (VISUAL), `action` (ACTION).

### 1.3 add_frame / save_episode / finalize — exact usage

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset.create(repo_id="local/franka_cup_place", fps=30, features=features,
                           root="/data/lerobot/franka_cup_place", robot_type="franka",
                           image_writer_threads=8, vcodec="h264")   # see codec note below
for ep in episodes:
    for t in range(T):
        ds.add_frame({
            "observation.state": state_t,                # np.float32 (8,)
            "action": action_t,                          # np.float32 (8,)
            "observation.images.table_cam": img_hwc_u8,  # np.uint8 (224,224,3) (or float32 in [0,1], or PIL)
            "observation.images.wrist_cam": wrist_u8,
            "task": "place the cup on the saucer",       # REQUIRED str, per frame
            # "timestamp": t / 30.0,                     # optional; defaults to frame_index/fps
        })
    ds.save_episode()          # stats + parquet append + PNG->mp4 encode + meta update
ds.finalize()                  # REQUIRED: closes parquet writers (data + meta/episodes buffer).
                               # Without it the parquet footers are missing and the dataset cannot be loaded.
# ds.push_to_hub(...)          # entirely optional; dataset is complete locally without it
```

Semantics/gotchas:
- `add_frame` (line 1171): converts torch→numpy, runs `validate_frame`, auto-appends `frame_index`, `timestamp`, pops `"task"`. Torch tensors are converted with `.numpy()` — must be on CPU.
- `fps` semantics: pure metadata + timestamp generator (`timestamp = frame_index / fps` if not provided). At **load** time timestamps are checked to be spaced `1/fps ± tolerance_s` (default 1e-4); video frames are fetched by timestamp. Keep timestamps exactly uniform (or omit them) or loading raises `FrameTimestampError`.
- Task strings: any per-frame string; per-episode `set(tasks)` recorded; task→index map in `meta/tasks.parquet`; `__getitem__` returns `item["task"]` (str). One constant string for the whole dataset is fine (single row, task_index 0).
- `save_episode()` (line 1225): computes per-episode stats, appends frames to the current `data/chunk-XXX/file-XXX.parquet` via a persistent `pq.ParquetWriter` (rolls to a new file at 100MB), encodes each camera's PNG dir to a temp mp4 with PyAV and either moves it (first ep / >200MB) or **concatenates onto the existing mp4** (stream copy, no re-encode), then updates `meta/*`. With multiple cameras and `parallel_encoding=True` (default) encoding runs in a `ProcessPoolExecutor`.
- Episode-metadata rows are buffered (`metadata_buffer_size=10`) — `meta/episodes/...parquet` lags until flush; `finalize()` flushes. A `__del__` safety net exists but do not rely on it.
- Resume recording: re-instantiate plain `LeRobotDataset(repo_id, root=...)` on the existing dir — it detects existing episodes and continues indices/files ("resuming" branches in `_save_episode_data` / `_save_episode_metadata`).
- **Codec choice**: default encode codec is `libsvtav1` (AV1). AV1 decode via pyav/torchcodec generally works, but `vcodec="h264"` is the most portable for other tools and older ffmpeg on clusters. Pixel format yuv420p, g=2, crf=30 defaults (`video_utils._get_codec_options`). Encoding uses **PyAV only** (bundled ffmpeg libs — no system `ffmpeg` binary needed for create/encode/concat).

### 1.4 Loading for training / reading

`LeRobotDataset(repo_id, root=..., episodes=None, delta_timestamps=None, tolerance_s=1e-4, video_backend=None, ...)` — with local `root` containing `meta/`, **no network access happens** (metadata loads locally; hub `snapshot_download` only on FileNotFoundError). `__getitem__` returns per-key torch tensors; video frames decoded on the fly as **float32 CHW in [0,1]**.

---

## 2. `DiffusionConfig` — all fields with 0.4.4 defaults (`policies/diffusion/configuration_diffusion.py`)

Registered as `@PreTrainedConfig.register_subclass("diffusion")` → CLI `--policy.type=diffusion`.

Inherited from `PreTrainedConfig` (`configs/policies.py:41`): `n_obs_steps=1` (overridden below), `input_features={}` / `output_features={}` (auto-filled from dataset by `make_policy`), `device=None` (auto-selects cuda), `use_amp=False`, `use_peft=False`, **`push_to_hub=True`** (!), `repo_id=None`, `private=None`, `tags=None`, `license=None`, `pretrained_path=None`.

Diffusion-specific:

| field | default | notes |
|---|---|---|
| `n_obs_steps` | 2 | obs history length |
| `horizon` | 16 | diffusion prediction length |
| `n_action_steps` | 8 | actions executed per inference |
| `normalization_mapping` | `{"VISUAL": MEAN_STD, "STATE": MIN_MAX, "ACTION": MIN_MAX}` | applied by processor pipeline, not inside model |
| `drop_n_last_frames` | 7 | **= horizon − n_action_steps − n_obs_steps + 1; NOT auto-recomputed — update manually if you change the others** (used by `EpisodeAwareSampler`) |
| `vision_backbone` | `"resnet18"` | must start with "resnet" |
| `resize_shape` | `None` | (H,W) pre-resize; None = native resolution |
| `crop_ratio` | `1.0` | in (0,1]; with `resize_shape` set and ratio<1, `crop_shape` is derived = `int(resize*ratio)`; ratio==1.0 with resize_shape set **forces crop_shape=None** |
| `crop_shape` | `None` | (H,W); legacy direct crop (no resize). None = no crop |
| `crop_is_random` | `True` | RandomCrop at train, always CenterCrop at eval |
| `pretrained_backbone_weights` | `None` | e.g. `"ResNet18_Weights.IMAGENET1K_V1"` — **incompatible with use_group_norm=True (raises)** |
| `use_group_norm` | `True` | replaces BatchNorm; groups = feat//16 |
| `spatial_softmax_num_keypoints` | 32 | feature dim per camera = 2*32=64 |
| `use_separate_rgb_encoder_per_camera` | `False` | |
| `down_dims` | `(512, 1024, 2048)` | U-Net stages; downsampling factor = 2^len |
| `kernel_size` | 5 | |
| `n_groups` | 8 | |
| `diffusion_step_embed_dim` | 128 | |
| `use_film_scale_modulation` | `True` | |
| `noise_scheduler_type` | `"DDPM"` | only `"DDPM"` or `"DDIM"` |
| `num_train_timesteps` | 100 | |
| `beta_schedule` | `"squaredcos_cap_v2"` | |
| `beta_start` / `beta_end` | 1e-4 / 0.02 | |
| `prediction_type` | `"epsilon"` | or `"sample"` |
| `clip_sample` / `clip_sample_range` | `True` / 1.0 | actions must be normalized into [-1,1] (MIN_MAX does this) |
| `num_inference_steps` | `None` | None → `num_train_timesteps` (100) at eval |
| `compile_model` / `compile_mode` | `False` / `"reduce-overhead"` | torch.compile of U-Net |
| `do_mask_loss_for_padding` | `False` | |
| `optimizer_lr` | 1e-4 | preset → `AdamConfig` (grad_clip_norm=10.0 from AdamConfig default) |
| `optimizer_betas` | (0.95, 0.999) | |
| `optimizer_eps` | 1e-8 | |
| `optimizer_weight_decay` | 1e-6 | |
| `scheduler_name` | `"cosine"` | diffusers `get_scheduler` |
| `scheduler_warmup_steps` | 500 | |

Constraint formulas (enforced in `__post_init__`/`validate_features`/`select_action` docstring):
- `horizon % 2**len(down_dims) == 0` (16 % 8 ✓).
- `n_action_steps <= horizon − n_obs_steps + 1` (8 ≤ 15 ✓) — action queue slice `actions[:, n_obs_steps-1 : n_obs_steps-1+n_action_steps]`.
- All camera image shapes must be identical (`validate_features` raises otherwise) → **render both Isaac cameras at the same resolution**.
- `crop_shape` must fit within image H,W (checked only when `resize_shape is None`).
- `observation_delta_indices = [1-n_obs_steps .. 0]` (= `[-1, 0]`); `action_delta_indices = [1-n_obs_steps .. 1-n_obs_steps+horizon-1]` (= `[-1..14]`) — these become `delta_timestamps` (indices / fps) in `make_dataset`. Training batch: `observation.*` are (B, 2, ...), `action` is (B, 16, dim), `action_is_pad` (B,16).

---

## 3. Training CLI at 0.4.4

Entry point (from `entry_points.txt`): **`lerobot-train`** = `lerobot.scripts.lerobot_train:main` (equivalently `python -m lerobot.scripts.lerobot_train`). Old `python -m lerobot.scripts.train` does NOT exist at this version. Draccus CLI over `TrainPipelineConfig` (`configs/train.py`).

Local dataset, fully offline, no hub, wandb on:

```bash
HF_HUB_OFFLINE=1 lerobot-train \
  --dataset.repo_id=local/franka_cup_place \
  --dataset.root=/data/lerobot/franka_cup_place \
  --dataset.video_backend=pyav \
  --policy.type=diffusion \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.crop_shape="[196,196]" \
  --output_dir=outputs/train/dp_cup_place \
  --job_name=dp_cup_place \
  --batch_size=64 --steps=100000 \
  --save_freq=20000 --log_freq=200 --eval_freq=0 \
  --num_workers=4 --seed=1000 \
  --wandb.enable=true --wandb.project=cost_of_generality --wandb.entity=<entity> --wandb.mode=offline
```

Key `TrainPipelineConfig` fields/defaults: `steps=100_000`, `batch_size=8`, `num_workers=4`, `save_freq=20_000` (+ always saves at final step), `save_checkpoint=True`, `log_freq=200`, `eval_freq=20_000` (inert when `--env` unset — eval env only built `if cfg.env is not None`), `seed=1000`, `tolerance_s=1e-4`, `use_policy_training_preset=True` (uses `DiffusionConfig.get_optimizer_preset/get_scheduler_preset`; override with `--optimizer.*`/`--scheduler.*` only if you also set `--use_policy_training_preset=false`). `output_dir` must not already exist unless `--resume=true` (raises `FileExistsError`). Default output dir: `outputs/train/{date}/{time}_{job_name}`.

Hub avoidance — **yes, fully avoidable end-to-end**, but you MUST pass `--policy.push_to_hub=false`: the default is `true` and `validate()` raises `ValueError: 'policy.repo_id' argument missing` otherwise. Dataset side: with `--dataset.root` pointing at a local v3.0 dataset, `LeRobotDatasetMetadata`/`LeRobotDataset` load purely from disk. Datasets `push_to_hub` is a separate manual method never called by train. Belt-and-braces: `HF_HUB_OFFLINE=1`.

wandb (`configs/default.py:WandBConfig` + `rl/wandb_utils.WandBLogger`): `--wandb.enable=true`, `--wandb.project` (default "lerobot"), `--wandb.entity`, `--wandb.mode` ∈ online|offline|disabled (default online), `--wandb.notes`, `--wandb.run_id`, `--wandb.disable_artifact` (checkpoints are uploaded as wandb artifacts on each save unless disabled — set `--wandb.disable_artifact=true` on the cluster). NB memory: confirm the entity matches the local `.netrc` account before trusting sync.

Image stats: `--dataset.use_imagenet_stats=true` is the **default** → camera-key mean/std in `dataset.meta.stats` are replaced by ImageNet values before being baked into the normalizer (`datasets/factory.py:128`). Leave as-is (standard for DP); set false to use true dataset image stats.

Resume:
```bash
lerobot-train --config_path=outputs/train/dp_cup_place/checkpoints/last/pretrained_model/train_config.json --resume=true
```
(config is taken from the checkpoint; CLI overrides at resume are ignored for most fields). `last` is a relative symlink to the newest `checkpoints/{step:06d}/`.

Checkpoint layout (`utils/train_utils.save_checkpoint`):
```
outputs/train/dp_cup_place/checkpoints/000020000/
├── pretrained_model/
│   ├── config.json                # policy config (type=diffusion, all fields)
│   ├── model.safetensors          # policy weights
│   ├── train_config.json          # full TrainPipelineConfig
│   ├── policy_preprocessor.json   # processor pipeline config
│   ├── policy_postprocessor.json
│   └── *.safetensors              # per-step processor state (normalizer stats etc.)
└── training_state/{optimizer_state.safetensors, optimizer_param_groups.json, rng_state.safetensors, scheduler_state.json, training_step.json}
```

Training internals worth knowing: HF `Accelerate` wraps everything (single-GPU fine; multi-GPU via `accelerate launch -m lerobot.scripts.lerobot_train ...`); `EpisodeAwareSampler` with `drop_n_last_frames=7`, shuffle inside; normalization happens in `preprocessor(batch)` inside the loop (policy itself consumes normalized tensors); AMP only if `--policy.use_amp=true`.

---

## 4. Inference API

### 4.1 Loading a checkpoint

```python
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

ckpt = "outputs/train/dp_cup_place/checkpoints/last/pretrained_model"   # the pretrained_model dir!
policy = DiffusionPolicy.from_pretrained(ckpt)          # reads config.json + model.safetensors,
                                                        # .to(config.device), .eval() applied
pre, post = make_pre_post_processors(policy.config, pretrained_path=ckpt)
# loads policy_preprocessor.json/policy_postprocessor.json incl. saved normalization stats
# (device override:)
# pre, post = make_pre_post_processors(policy.config, pretrained_path=ckpt,
#         preprocessor_overrides={"device_processor": {"device": "cuda"}})
```
`from_pretrained` accepts a local dir (checks `config.json`, `model.safetensors` = `SAFETENSORS_SINGLE_FILE`) — no hub contact for local paths. Config overrides at load: `DiffusionPolicy.from_pretrained(ckpt, cli_overrides=["--num_inference_steps=10", "--noise_scheduler_type=DDIM"])` (draccus-style args, forwarded to `PreTrainedConfig.from_pretrained`).

### 4.2 Rollout loop, batch keys, dtypes

```python
import torch
policy.reset()                      # REQUIRED at each episode start: clears obs+action deques
for t in range(max_steps):
    obs = {
        "observation.state": torch.from_numpy(qpos).float(),                # (8,) or (B,8) float32
        "observation.images.table_cam": img_chw_float01,                    # (3,H,W) float32 in [0,1]
        "observation.images.wrist_cam": wrist_chw_float01,
    }
    batch = pre(obs)                # Rename -> AddBatchDim (unsqueezes unbatched) -> to device -> Normalize
    action = policy.select_action(batch)     # (B, action_dim), NORMALIZED space
    action = post(action)                    # unnormalize + move to CPU
    env.step(action.numpy())
```

- **`select_action` does NOT accept raw uint8 HWC images.** The processor pipeline does not convert dtype/layout; convert yourself exactly as `envs/utils.preprocess_observation` does: uint8 (B,H,W,C) → `float32 CHW / 255`. State must be float32.
- Normalization (MEAN_STD w/ ImageNet stats for images, MIN_MAX for state/action) is applied by `pre`; unnormalization of actions by `post`. Do NOT skip the processors — the model weights expect normalized inputs, and its output is in [-1,1].
- Action-queue vs chunk semantics (`modeling_diffusion.py:103-139`): `select_action` pushes obs into `n_obs_steps`-deep deques (first call replicates the obs to fill history), and when the action deque (`maxlen=n_action_steps=8`) is empty runs one diffusion sampling producing `horizon=16` actions, keeps 8 starting at the current step, and thereafter pops one per call. So diffusion runs every 8 env steps. If `"action"` is in the batch it is popped (offline-eval safety).
- `policy.predict_action_chunk(batch)` returns the whole (B, n_action_steps, dim) chunk but also reads from the queues (call after feeding obs via the queues; easier to just use `select_action`).
- Batched envs supported (B>1); queues store batched tensors.

### 4.3 num_inference_steps / DDIM at eval — the post-load gotcha

`DiffusionModel.__init__` **copies** `config.num_inference_steps` into `self.num_inference_steps` and **constructs** the scheduler object from config (lines 193-207). Therefore after `from_pretrained`:
- setting `policy.config.num_inference_steps = 10` has **NO effect**; set `policy.diffusion.num_inference_steps = 10` (used at each `conditional_sample` via `noise_scheduler.set_timesteps`), or pass `cli_overrides` at load.
- switching to DDIM post-hoc requires replacing the scheduler object:
  ```python
  from diffusers.schedulers.scheduling_ddim import DDIMScheduler
  c = policy.config
  policy.diffusion.noise_scheduler = DDIMScheduler(
      num_train_timesteps=c.num_train_timesteps, beta_start=c.beta_start, beta_end=c.beta_end,
      beta_schedule=c.beta_schedule, clip_sample=c.clip_sample, clip_sample_range=c.clip_sample_range,
      prediction_type=c.prediction_type)
  policy.diffusion.num_inference_steps = 10
  ```
  or, cleaner, load with `cli_overrides=["--noise_scheduler_type=DDIM","--num_inference_steps=10"]`. DDPM-trained weights + DDIM sampler is the standard fast-eval combo (same beta schedule).

There is also `lerobot-eval` (`scripts/lerobot_eval.py`, `--policy.path=<ckpt> --env.type=...`) but it needs a registered gym env — for a custom Isaac Lab task, write your own rollout with the Python API above.

---

## 5. Dependencies / environment (from wheel METADATA = pyproject)

- `Requires-Python >=3.10` → **Python 3.11 OK**.
- `torch>=2.2.1,<2.11.0` → **torch 2.7.0+cu128 OK**. `torchvision>=0.21.0,<0.26.0` → pair 0.22.0 with torch 2.7.0.
- `torchcodec>=0.2.1,<0.11.0` with env marker → **installed by default on linux x86_64** (excluded only on aarch64/arm/win/mac-x86). Gotcha: with torch 2.7.0 already pinned, latest torchcodec (0.10.x) requires a newer torch — pip must backtrack; **pin explicitly `torchcodec==0.4.0`** (the torch-2.7 pairing per torchcodec's compat matrix — *uncertain of exact patch, verify: 0.4.x ↔ torch 2.7, 0.5 ↔ 2.8*), or omit torchcodec entirely (see below).
- Other notable pins: `datasets>=4.0,<5`, `diffusers>=0.27.2,<0.36`, `accelerate>=1.10,<2`, `draccus==0.10.0`, `wandb>=0.24,<0.25`, `av>=15,<16` (PyAV), `imageio[ffmpeg]`, `gymnasium>=1.1.1,<2.0`, `opencv-python-headless`, `huggingface-hub>=0.34.2,<0.36`, `einops`, `jsonlines`, `deepdiff`, `rerun-sdk`, `pynput`, `pyserial`, `cmake`, `packaging`. Extras: `[pusht]`, `[aloha]`, etc. — none needed.
- **Conflict risk with the Isaac Lab env**: `gymnasium<2` vs Isaac Lab's pinned gymnasium, `opencv-python-headless` vs Isaac's `opencv-python`, wandb/protobuf pins. Strongly prefer a **separate venv for lerobot** (dataset conversion + training + a thin inference wrapper); only the inference loop needs to co-exist with Isaac — either install lerobot into the Isaac env with care (`pip install lerobot==0.4.4 --no-deps` + manually install the few runtime deps: torch/torchvision (already), av, datasets, diffusers, einops, draccus, huggingface-hub, packaging, jsonlines, deepdiff, opencv, pandas, pyarrow, wandb, accelerate, termcolor, tqdm) or run policy inference in-process via safetensors re-implementation (not recommended).

### torchcodec / ffmpeg at train time (cluster-critical)

- Decoding mp4 happens **on every training batch** in the DataLoader workers (`decode_video_frames`, CPU decode).
- Default backend: `get_safe_default_codec()` = `"torchcodec"` **iff `import torchcodec` succeeds**, else `"pyav"` with a warning (`datasets/video_utils.py:117`). So torchcodec is a soft dependency at runtime: **0.4.4 does NOT need torchcodec — `--dataset.video_backend=pyav` (or simply not installing torchcodec) is fully supported.**
- torchcodec offline behavior: no network use at runtime, but it dynamically links **FFmpeg shared libraries (libavcodec/libavformat/libavutil, majors 4–7)** that must exist on the node (`conda install ffmpeg<8` or module load). `imageio-ffmpeg` ships only a static binary — does NOT satisfy torchcodec. If the A100 node lacks ffmpeg libs, torchcodec import fails → automatic pyav fallback (or force pyav).
- pyav path: `decode_video_frames_torchvision` calls `torchvision.set_video_backend("pyav")` + `torchvision.io.VideoReader` — works with torchvision 0.22 but the video API is **deprecated there (warning spam; slated for removal in later torchvision)**; with torchvision pinned ≤0.25 by lerobot it remains functional. PyAV wheels bundle their own ffmpeg libs → zero system deps.
- Recommendation for the cluster: install `torchcodec==0.4.0` + ensure `ffmpeg` libs (fast, sequential-decode optimized); fallback plan `--dataset.video_backend=pyav` (slower, ~always works). Encoding (dataset creation on the Isaac machine) never needs torchcodec or an ffmpeg binary — PyAV only.

---

## 6. Stats computation (`datasets/compute_stats.py`)

- **Computed at create time, incrementally — no separate script.** Every `save_episode()` calls `compute_episode_stats(episode_buffer, features)`: numeric features → full-episode min/max/mean/std/count + quantiles `[0.01,0.10,0.50,0.90,0.99]`; image/video features → sampled subset of the episode's PNG frames (`estimate_num_samples`: `min(max(100, N^0.75), 10_000)`, evenly spaced, auto-downsampled to ~150px) → per-channel stats shaped **(3,1,1), normalized to [0,1]** (divided by 255).
- Per-episode stats are stored in `meta/episodes/*.parquet` (flattened `stats/...` columns) and merged into the global running stats via `aggregate_stats` (count-weighted mean/std merge), rewritten to **`meta/stats.json` after every episode**. `LeRobotDatasetMetadata.stats` serves them at train time.
- With `streaming_encoding=True`, video stats come from the encoder threads (`RunningQuantileStats`) instead of PNG sampling — same format.
- Training then overrides camera-key mean/std with ImageNet values by default (`use_imagenet_stats=true`), and those merged stats are what get frozen into `policy_preprocessor.json`/processor state at checkpoint time → inference reproduces training normalization automatically.

---

## Blockers / uncertainty flags

1. `create()` fails if the dataset root dir already exists — plan dir lifecycle in the Mimic-generation script.
2. `finalize()` is mandatory; a crash mid-generation leaves the last ≤10 episodes' metadata unflushed and parquet footers missing (data effectively unreadable) — wrap generation in try/finally.
3. `--policy.push_to_hub=false` is mandatory offline (default true errors out on missing repo_id).
4. `drop_n_last_frames` is a static default (7) — recompute manually if horizon/n_action_steps/n_obs_steps change.
5. torchcodec↔torch-2.7 exact pin (0.4.x) taken from the torchcodec compat matrix from memory — verify at install; pyav fallback removes the risk entirely.
6. Default AV1 encode (`libsvtav1`): fine within lerobot, but pass `vcodec="h264"` if anything else must read the mp4s. Unverified whether the cluster's torchcodec build decodes AV1 — another reason for h264.
7. Both cameras must share one resolution (DiffusionConfig.validate_features).
8. This wheel contains extra non-upstream-looking features (RA-BC/SARM hooks in train script) — harmless when unused, but note 0.4.4 is a fast-moving 0.4.x line; pin exactly `lerobot==0.4.4`.
9. `validate()` errors if `output_dir` exists and not resuming — generate unique run dirs.
10. Images passed to `add_frame` may be uint8 [0,255] or float [0,1] (range check happens in image writer); uint8 HWC straight from Isaac render buffers is the intended path.
