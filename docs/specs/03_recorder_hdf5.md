# SPEC 03 — Demo recording + HDF5 dataset format (Isaac Lab v2.3.0)

All paths relative to `/home/admin_07/cost_of_generality/third_party/IsaacLab` (read-only checkout).

Key source files studied:
- `source/isaaclab/isaaclab/managers/recorder_manager.py` — `RecorderManager`, `RecorderTerm`, `RecorderManagerBaseCfg`, `DatasetExportMode`
- `source/isaaclab/isaaclab/utils/datasets/hdf5_dataset_file_handler.py` — `HDF5DatasetFileHandler`
- `source/isaaclab/isaaclab/utils/datasets/episode_data.py` — `EpisodeData`
- `source/isaaclab/isaaclab/envs/mdp/recorders/recorders.py` + `recorders_cfg.py` — stock recorder terms, `ActionStateRecorderManagerCfg`
- `source/isaaclab/isaaclab/envs/manager_based_env.py`, `manager_based_rl_env.py` — hook call sites
- `scripts/tools/record_demos.py`, `scripts/imitation_learning/isaaclab_mimic/{annotate_demos.py,generate_dataset.py}`, `source/isaaclab_mimic/isaaclab_mimic/datagen/{generation.py,data_generator.py}`
- `scripts/tools/{hdf5_to_mp4.py,merge_hdf5_datasets.py,mp4_to_hdf5.py}`
- Reference task: `source/isaaclab_tasks/.../stack/config/franka/stack_ik_rel_visuomotor_env_cfg.py` (`Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0`), mimic variant `source/isaaclab_mimic/isaaclab_mimic/envs/franka_stack_ik_rel_visuomotor_mimic_env_cfg.py` (`Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0`).

---

## 1. Resulting HDF5 tree for a recorded episode

Written by `HDF5DatasetFileHandler.write_episode()` (robomimic-compatible layout, `env_args["type"]=2` = gym type). All leaf datasets are created with `create_dataset(key, data=value.cpu().numpy(), compression="gzip")` — dtype is whatever the torch tensor was; **no dtype conversion happens anywhere in the pipeline**.

```
dataset.hdf5
└── data                                (h5py Group)
    ├── attrs:
    │   ├── total     : int   — running sum of num_samples over all demos
    │   └── env_args  : str   — JSON: {"env_name": "<task id>", "type": 2,
    │                                  "sim_args": {"dt", "decimation", "render_interval", "num_envs"}}
    │                          (sim_args merged in at first export via RecorderManager.get_ep_meta();
    │                           override by defining get_ep_meta() on the env cfg)
    ├── demo_0                          (Group; name = export order, NOT env id)
    │   ├── attrs:
    │   │   ├── num_samples : int  — len(actions); 0 if no "actions" key
    │   │   ├── success     : bool — present iff EpisodeData.success was set (always set by RecorderManager)
    │   │   └── seed        : int  — only if EpisodeData.seed was set (stock pipeline NEVER sets it → absent)
    │   ├── initial_state/                       # from InitialStateRecorder (record_post_reset), shape (1, ...)
    │   │   ├── articulation/
    │   │   │   └── robot/
    │   │   │       ├── root_pose      (1, 7)  float32   # pos(3)+quat wxyz(4), env-relative (is_relative=True)
    │   │   │       ├── root_velocity  (1, 6)  float32
    │   │   │       ├── joint_position (1, nJ) float32   # Franka: nJ=9
    │   │   │       └── joint_velocity (1, nJ) float32
    │   │   └── rigid_object/
    │   │       └── <obj_name>/                          # e.g. cube_1 / your "cup"
    │   │           ├── root_pose      (1, 7)  float32
    │   │           └── root_velocity  (1, 6)  float32
    │   ├── actions            (T, A) float32            # PreStepActionsRecorder: action_manager.action
    │   │                                                # (raw policy action, pre-scaling; IK-rel Franka: A=7)
    │   ├── processed_actions  (T, A') float32           # PostStepProcessedActionsRecorder: concat of
    │   │                                                # per-term processed_actions (post scale/clip)
    │   ├── obs/                                         # PreStepFlatPolicyObservationsRecorder:
    │   │   │                                            # env.obs_buf["policy"] — REQUIRES concatenate_terms=False
    │   │   ├── actions            (T, A)   float32      # mdp.last_action obs term (name collision is fine, it's under obs/)
    │   │   ├── joint_pos          (T, 9)   float32
    │   │   ├── joint_vel          (T, 9)   float32
    │   │   ├── eef_pos            (T, 3)   float32
    │   │   ├── eef_quat           (T, 4)   float32
    │   │   ├── gripper_pos        (T, 1)   float32
    │   │   ├── object             (T, K)   float32      # task-specific low-dim
    │   │   ├── table_cam          (T, H, W, 3) uint8    # mdp.image(data_type="rgb", normalize=False)
    │   │   └── wrist_cam          (T, H, W, 3) uint8    # stack task uses H=W=84; NHWC, RGB order, 0..255
    │   └── states/                                      # PostStepStatesRecorder (record_post_step),
    │       ├── articulation/robot/{root_pose (T,7), root_velocity (T,6),
    │       │                       joint_position (T,9), joint_velocity (T,9)}   float32
    │       └── rigid_object/<obj>/{root_pose (T,7), root_velocity (T,6)}         float32
    ├── demo_1 ...
```

- `T` = number of env steps in the episode. `obs` is recorded **pre-step** (the obs the policy acted on, computed at the end of the previous step) and `actions` pre-step too → `obs[t]` ↔ `actions[t]` aligned robomimic-style. `states[t]` is the state **after** applying `actions[t]`.
- RGB dtype: `mdp.image` with `normalize=False` returns the camera buffer untouched → `torch.uint8`, shape per env `(H, W, 3)`; if you set `normalize=True` you get float32 mean-subtracted images — do NOT do that for recording (hdf5_to_mp4 and LeRobot conversion expect uint8 0..255).
- `initial_state` exists once per episode with leading dim 1 (recorded only at `record_post_reset` for the reset env ids). Mimic's `generate_dataset` replays episodes by `env.reset_to(episode.get_initial_state(), env_ids, is_relative=True)` — keep this recorder if you want replay/regeneration.
- Nested keys: `add_to_episodes` recursively splits dicts into `key/sub_key/...` groups. `EpisodeData.add` appends per-step tensors to python lists; `pre_export()` does `torch.stack(list)` at export → leading T dim.

State dict format = `InteractiveScene.get_state(is_relative=True)` (`source/isaaclab/isaaclab/scene/interactive_scene.py:565`): `{"articulation": {...}, "deformable_object": {name: {nodal_position, nodal_velocity}}, "rigid_object": {...}}` — only categories present in the scene appear.

## 2. `concatenate_terms=False`

- Attribute of `ObservationGroupCfg` (`isaaclab.managers.ObservationGroupCfg`). With `True` (default) the group's terms are concatenated into a single tensor → the obs recorder would write ONE dataset `data/demo_N/obs` (breaks robomimic layout) — and concatenation would anyway **fail with a shape error** when mixing (H,W,3) images with 1-D states (observation_manager raises "set 'concatenate_terms' to False").
- With `False`, `env.obs_buf["policy"]` is a `dict[term_name → tensor(num_envs, ...)]` and the recorder produces one dataset per term under `obs/`.
- Where to set it:
  1. In your env cfg (the stack visuomotor task does it in `PolicyCfg.__post_init__`: `self.concatenate_terms = False`, plus `self.enable_corruption = False`). **Do this in our cup-place cfg** — required for vision obs regardless of recording.
  2. Belt-and-suspenders: the workflow scripts also force it: `env_cfg.observations.policy.concatenate_terms = False` (`record_demos.py:227`, `generation.py::setup_env_config`).
- Gotcha: this changes the env's observation space (dict instead of flat) — RL training scripts expecting flat obs will break; keep a separate non-vision cfg for anything needing flat obs.

## 3. Attaching the recorder, flush cadence, success-only export

### Attach
`ManagerBasedEnv.__init__` builds `self.recorder_manager = RecorderManager(self.cfg.recorders, self)` (`manager_based_env.py:291`). Any `ManagerBasedEnvCfg` has a `recorders` field (defaults to empty → no-op manager). Wire-up used by ALL imitation workflows:

```python
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

env_cfg.recorders = ActionStateRecorderManagerCfg()          # 5 terms: initial_state, states, actions,
                                                             # flat policy obs, processed_actions
env_cfg.recorders.dataset_export_dir_path = output_dir       # default "/tmp/isaaclab/logs"
env_cfg.recorders.dataset_filename = output_file_name        # no extension; ".hdf5" appended
env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
env_cfg.env_name = "Your-Task-Id-v0"    # IMPORTANT: written into env_args["env_name"];
                                        # annotate/generate scripts read it back via get_env_name_from_dataset()
```
You can subclass `ActionStateRecorderManagerCfg` to add custom `RecorderTermCfg` terms (any non-reserved attr of the cfg is treated as a term; set an inherited term attr to `None` to disable it, as `annotate_demos.py` does).

`DatasetExportMode` (IntEnum): `EXPORT_NONE=0`, `EXPORT_ALL=1` (default), `EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES=2` (failed go to `<name>_failed.hdf5`), `EXPORT_SUCCEEDED_ONLY=3`.

### Hook/flush cadence
- `record_pre_step()` — in `env.step()` right after `action_manager.process_action` (both `ManagerBasedEnv:445` and `ManagerBasedRLEnv:176`).
- `record_post_physics_decimation_step()` — RL env only, inside decimation loop (`manager_based_rl_env.py:191`); no stock term uses it.
- `record_post_step()` — end of step. RL env recomputes `self.obs_buf = self.observation_manager.compute()` first **iff recorder terms are active** → obs (incl. cameras) computed twice per step while recording; expect ~2x obs cost.
- `record_pre_reset(env_ids)` — at the start of every `env.reset()` / `reset_to()` and, in the RL env, for terminated envs inside `step()`. It (a) runs pre-reset terms, (b) sets `episode.success` from `termination_manager.get_term("success")` if a termination term literally named `"success"` is active, (c) if `export_in_record_pre_reset` (cfg, default True) calls `export_episodes(env_ids)`.
- `export_episodes(env_ids)` → for each non-empty episode: `pre_export()` (stack lists), route by success/export-mode, `write_episode`, then `h5.flush()` once per export batch. Episode buffer is cleared afterwards. So **an episode hits disk at the reset that ends it**; nothing is written mid-episode. Data is safe after each episode boundary even on crash.
- `recorder_manager.reset(env_ids)` clears episode buffers WITHOUT exporting (used to discard the stale trajectory before a scripted reset).

### Success-only export — the pattern used by record_demos.py (single env, teleop) and mimic
The env's success termination term must be **removed from `terminations` and evaluated manually**, otherwise the env auto-resets on success and success bookkeeping is implicit:

```python
success_term = env_cfg.terminations.success      # TerminationTermCfg with func + params
env_cfg.terminations.success = None
env_cfg.terminations.time_out = None             # record_demos; generation.py sets env_cfg.terminations = None

# per step, after env.step():
if bool(success_term.func(env, **success_term.params)[0]):
    success_step_count += 1
    if success_step_count >= num_success_steps:              # default 10 consecutive steps
        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)  # run pre-reset terms, no export yet
        env.recorder_manager.set_success_to_episodes([0], torch.tensor([[True]], dtype=torch.bool, device=env.device))
        env.recorder_manager.export_episodes([0])
else:
    success_step_count = 0
```
Progress via `env.recorder_manager.exported_successful_episode_count` (property, summed over envs; also `exported_failed_episode_count`).

Alternative (policy rollouts, no manual loop): keep a termination term **named exactly `success`** in `TerminationsCfg` and set `dataset_export_mode = EXPORT_SUCCEEDED_ONLY`; auto-reset in `ManagerBasedRLEnv.step` then exports succeeded episodes only. Note success is read at reset time from the termination buffer of the resetting envs — works because the success term is what triggered the reset.

### Existing CLIs that do the wiring for you
- Teleop recording: `./isaaclab.sh -p scripts/tools/record_demos.py --task <ID> --teleop_device keyboard|spacemouse --dataset_file ./datasets/dataset.hdf5 --num_demos 10 --num_success_steps 10 --step_hz 30 [--enable_cameras]` (forces `num_envs=1`).
- Mimic annotation (adds `obs/datagen_info/subtask_term_signals`): `scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --task <Mimic-ID> --input_file src.hdf5 --output_file annotated.hdf5 [--auto]`. Uses `MimicRecorderManagerCfg(ActionStateRecorderManagerCfg)` defined **inline in that script** with `PreStepDatagenInfoRecorderCfg` (`obs/datagen_info/{object_pose,eef_pose,target_eef_pose}`), `PreStepSubtaskStartsObservationsRecorderCfg`, `PreStepSubtaskTermsObservationsRecorderCfg`. `--auto` requires the env to implement `get_subtask_term_signals()` (default impl reads `obs_buf["subtask_terms"]` group).
- Mimic generation: `scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --task Isaac-...-Mimic-v0 --input_file annotated.hdf5 --output_file generated.hdf5 --generation_num_trials 1000 --num_envs 10 [--headless --enable_cameras]`. `setup_env_config` (in `source/isaaclab_mimic/isaaclab_mimic/datagen/generation.py`) attaches `ActionStateRecorderManagerCfg`, sets export mode from `env_cfg.datagen_config.generation_keep_failed` (True → `EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES`, else `EXPORT_SUCCEEDED_ONLY`), sets `env_cfg.terminations = None`. Success is set explicitly by `DataGenerator.generate()` (`data_generator.py:934-939`: `set_success_to_episodes` + `export_episodes` per env id; it calls `recorder_manager.reset(env_ids)` before each new attempt).

## 4. Gotchas

- **`concatenate_terms=False` mandatory** (see §2). Also set `enable_corruption=False` for clean demos.
- **`env_cfg.env_name` must be set manually** before env creation; otherwise `env_args["env_name"]` is `""` and downstream `get_env_name_from_dataset()` fails (annotate/generate then need explicit `--task`).
- **Demo naming/order with num_envs>1**: per-env `EpisodeData` buffers; `demo_N` index = global export order, interleaved across envs. Episode lengths vary. `data.attrs["total"]` is correct. Writes are sequential in one file handler → safe.
- **`success` attr**: episodes exported via the auto path get `success` from a termination term **named `"success"`**; any other name → success stays False → EXPORT_SUCCEEDED_ONLY silently writes nothing. If you never define/keep a success term and never call `set_success_to_episodes`, all episodes count as failed.
- **`seed` attr is never written** by the stock pipeline (only read in `load_episode`). Don't rely on it.
- **Subtask signals are NOT recorded** by `ActionStateRecorderManagerCfg` — the `subtask_terms` obs group is untouched (only `obs_buf["policy"]` is recorded). They appear only after `annotate_demos.py` (under `obs/datagen_info/subtask_term_signals/<name>`, shape (T,) or (T,1) bool) OR if you add a custom recorder term. For Mimic you need: SubtaskCfg obs group (bool terms, `concatenate_terms=False`) + `SubTaskConfig(object_ref=..., subtask_term_signal="<obs term name>")` list in `MimicEnvCfg.subtask_configs["<eef_name>"]` (see `franka_stack_ik_rel_visuomotor_mimic_env_cfg.py`).
- **The mimic-*generated* visuomotor dataset contains full policy obs incl. RGB** (recorder replays through the real env with cameras) — this is the dataset you feed to diffusion-policy training; the source teleop demos can be recorded in the non-vision variant (faster) and only generation uses the visuomotor mimic env (that is exactly the stack workflow: record `Isaac-Stack-Cube-Franka-IK-Rel-v0` → annotate `...-IK-Rel-Mimic-v0` → generate `...-IK-Rel-Visuomotor-Mimic-v0`).
- **Cameras**: pass `--enable_cameras` (AppLauncher flag) for headless camera rendering. Visuomotor cfg sets `self.rerender_on_reset = True` (else first frames after reset are stale) and `self.sim.render.antialiasing_mode = "OFF"`; it also declares `self.image_obs_list = ["table_cam", "wrist_cam"]` (consumed by robomimic training config helper, not by the recorder).
- **Double obs compute per step** while recording in `ManagerBasedRLEnv` (post-step `observation_manager.compute()`), so recording FPS with cameras is roughly halved.
- **gzip compression on every dataset** (incl. images) — slow-ish writes, small files. No chunking control exposed.
- **`export_in_record_pre_reset=True` (default) means pressing "R" (reset) in record_demos would normally export the discarded episode** — record_demos avoids this by calling `env.recorder_manager.reset()` before `env.reset()` (buffer cleared, nothing exported). Mimic instead sets `force_export_or_skip=False` when it wants terms run without export. Replicate one of these in any custom loop with manual resets.
- **`get_ep_meta()`**: if your env cfg defines it, its dict fully replaces the default `{"sim_args": ...}` merged into `env_args`. Optional.
- **Opening an existing file**: `HDF5DatasetFileHandler.open(path, mode="r")` (use `mode="a"` to append; `create()` always truncates with `"w"`). `load_episode(name, device)` returns `EpisodeData` with torch tensors.
- The recorder callbacks return `(key, value)`; key `None` → nothing recorded. Custom term example (add e.g. per-step subtask signals or extra cams):

```python
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

class PreStepSubtaskTermsRecorder(RecorderTerm):
    def record_pre_step(self):
        return "obs/subtask_terms", self._env.obs_buf["subtask_terms"]

@configclass
class PreStepSubtaskTermsRecorderCfg(RecorderTermCfg):
    class_type: type = PreStepSubtaskTermsRecorder

@configclass
class CupPlaceRecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_subtask_terms = PreStepSubtaskTermsRecorderCfg()
```

## 5. Post-processing tools (`scripts/tools/`)

- **`merge_hdf5_datasets.py`** — plain python (no sim): `python scripts/tools/merge_hdf5_datasets.py --input_files a.hdf5 b.hdf5 --output_file merged_dataset.hdf5`. Copies `data/<episode>` groups into sequentially renumbered `data/demo_N`; copies `env_args` attr from the FIRST file only. **Does NOT recompute `data.attrs["total"]`** (it's absent in output since attrs other than env_args aren't copied) — robomimic readers that need `total` must recompute; per-demo `num_samples`/`success` attrs ARE preserved (group copy includes attrs).
- **`hdf5_to_mp4.py`** — plain python: `python scripts/tools/hdf5_to_mp4.py --input_file dataset.hdf5 --output_dir vids/ --input_keys table_cam wrist_cam --video_height 704 --video_width 1280 --framerate 30`. Expects `data/demo_{i}/obs/<key>` with frames `(T, H, W, C)`: RGB keys uint8 HWC RGB (converted to BGR, resized cubic); keys containing `normals` float in [0,1]*255; `<cam>_depth` shape (...,1) normalized with MIN_DEPTH=0.0/MAX_DEPTH=1.5; `<cam>_shaded_segmentation` synthesized from `<cam>_segmentation` (RGBA) + `<cam>_normals`. Default keys are the cosmos-stack set: `["table_cam","wrist_cam","table_cam_segmentation","table_cam_normals","table_cam_shaded_segmentation","table_cam_depth"]` — **pass `--input_keys` explicitly for our task or it will KeyError on missing seg/normals keys**. Output files `demo_{i}_{key}.mp4`, one per demo per key. Iterates demos as `range(len(f["data"]))` assuming contiguous `demo_0..demo_{N-1}` naming.
- **`mp4_to_hdf5.py`** — writes video frames back into an HDF5 (used for the cosmos augmentation round-trip; check its `--input_file/--videos_dir` args if needed).
- **`scripts/tools/replay_demos.py`** — replays a recorded dataset in an env (`--dataset_file`, `--task`); uses `initial_state` + `actions`, good sanity check that recording is replayable.
- Robomimic consumer for reference: `scripts/imitation_learning/robomimic/train.py` / `robomimic/robomimic_data_collector` style loaders read exactly this layout (`data/demo_N/obs/<key>`, `actions`, attrs `num_samples`). For LeRobot conversion (our project) read `obs/table_cam|wrist_cam` (uint8 NHWC), `obs/<state terms>`, `actions`, per-demo `attrs["success"]`.

## Uncertainties / flags
- Camera image dtype asserted as uint8 based on `mdp.image` (`observations.py:373-413`, no cast when `normalize=False`) and `TiledCamera` RGB buffers being uint8; verify once on a produced file (`h5py` dump) before freezing the LeRobot converter.
- `processed_actions` for `DifferentialInverseKinematicsAction` = processed (scaled) 6-DoF delta pose + gripper term output, concatenated in `action_manager.active_terms` order — exact width depends on our action terms; inspect at runtime.
- `merge_hdf5_datasets.py` `total`-attr behavior inferred from code (only `env_args` copied explicitly); double-check if any downstream reader (robomimic `SequenceDataset`) requires `data.attrs["total"]`.
- v2.3.0 `record_demos.py` supports only `num_envs=1`; parallel recording of scripted/policy demos must use the auto-reset + `EXPORT_SUCCEEDED_ONLY` route or a custom loop calling `export_episodes` per env id.
