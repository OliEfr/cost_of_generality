# SPEC: isaaclab_mimic — Mimic support for a custom Franka cup-place env + dataset generation

All paths relative to `/home/admin_07/cost_of_generality/third_party/IsaacLab` (IsaacLab v2.3.0, read-only).
Key sources studied:
- `source/isaaclab/isaaclab/envs/manager_based_rl_mimic_env.py` (base class)
- `source/isaaclab/isaaclab/envs/mimic_env_cfg.py` (`DataGenConfig`, `SubTaskConfig`, `SubTaskConstraintConfig`, `MimicEnvCfg`)
- `source/isaaclab_mimic/isaaclab_mimic/envs/franka_stack_ik_rel_mimic_env.py` + `franka_stack_ik_rel_mimic_env_cfg.py` (+ visuomotor cfg, abs cfg, `pick_place_mimic_env.py`, `agibot_place_upright_mug_mimic_env_cfg.py`)
- `source/isaaclab_mimic/isaaclab_mimic/datagen/{data_generator,datagen_info,datagen_info_pool,selection_strategy,waypoint,generation,utils}.py`
- `scripts/imitation_learning/isaaclab_mimic/{annotate_demos.py,generate_dataset.py}`
- `scripts/tools/record_demos.py`, `scripts/tools/replay_demos.py`
- `source/isaaclab/isaaclab/envs/mdp/recorders/{recorders.py,recorders_cfg.py}`, `source/isaaclab/isaaclab/managers/recorder_manager.py`, `source/isaaclab/isaaclab/utils/datasets/hdf5_dataset_file_handler.py`
- Task cfgs: `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/…` (`stack_env_cfg.py`, `mdp/observations.py`, `config/franka/stack_ik_rel_env_cfg.py`, `config/franka/stack_ik_rel_visuomotor_env_cfg.py`, `config/franka/stack_joint_pos_env_cfg.py`)
- Docs with canonical CLI: `docs/source/overview/imitation-learning/teleop_imitation.rst`

---

## 1. ManagerBasedRLMimicEnv: methods to implement

`class ManagerBasedRLMimicEnv(ManagerBasedRLEnv)` — file `source/isaaclab/isaaclab/envs/manager_based_rl_mimic_env.py`. Your env class subclasses it (or, easier, subclasses `FrankaCubeStackIKRelMimicEnv` / `PickPlaceRelMimicEnv` and overrides only `get_subtask_term_signals`).

Required (raise `NotImplementedError` in base):

```python
def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor
```
Returns 4x4 pose matrices, shape `(len(env_ids), 4, 4)`. MUST be in **the same frame the EEF controller commands** (for the Franka DiffIK rel action this is the frame produced by the `ee_frame` FrameTransformer = world minus env origin; Franka impl reads `self.obs_buf["policy"]["eef_pos"]` and `["eef_quat"]` — i.e. it requires those two term names in the policy obs group, quat in (w,x,y,z)). Build with `PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat))` (`isaaclab.utils.math`).

```python
def target_eef_pose_to_action(self, target_eef_pose_dict: dict, gripper_action_dict: dict,
                              action_noise_dict: dict | None = None, env_id: int = 0) -> torch.Tensor
```
Input: `{eef_name: 4x4 pose}` and `{eef_name: gripper_action tensor}` for ONE env. Output: 1-D action tensor compatible with `env.step` (Franka rel impl: `cat([delta_pos(3), axis_angle_delta_rot(3), gripper(1)]) → shape (7,)`; delta = target vs current `get_robot_eef_pose`, noise added as `noise * randn_like(pose_action)` then clamped to [-1,1], gripper untouched). NOTE: signature must contain the parameter name `action_noise_dict` — `generate_dataset.py` and `MultiWaypoint.execute` inspect the signature and fall back to a deprecated `noise=` call otherwise (the shipped IK-Abs env still uses the deprecated `noise` form and returns shape `(1,8)` pos(3)+quat_wxyz(4)+gripper(1)).

```python
def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]
```
Inverse of the above, batched: input `(num_envs, action_dim)` → `{eef_name: (num_envs, 4, 4)}`. Used by the annotation recorder to compute `target_eef_pose` from recorded actions.

```python
def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]
```
Slice gripper part out of an action trajectory; Franka: `{eef: actions[:, -1:]}` (works on the whole `(T, action_dim)` episode array — called by `DataGenInfoPool._add_episode` with `ep["actions"]`).

Optional-but-needed-for-`--auto` annotation:
```python
def get_subtask_term_signals(self, env_ids=None) -> dict[str, torch.Tensor]   # bool flags per subtask
def get_subtask_start_signals(self, env_ids=None) -> dict[str, torch.Tensor]  # only for skillgen / --annotate_subtask_start_signals
```
`annotate_demos.py --auto` hard-fails with `NotImplementedError` if `get_subtask_term_signals.__func__ is ManagerBasedRLMimicEnv.get_subtask_term_signals` (i.e. you must override it, not rely on inheritance from an env that didn't).

Provided by base (usually keep):
- `get_object_poses(env_ids)` → `{object_name: (N,4,4)}` built from `self.scene.get_state(is_relative=True)["rigid_object"]` — every rigid object in the scene, poses **relative to env origin** (world-ish frame). This is the frame consistency contract: `get_robot_eef_pose`, `get_object_poses`, and `target_eef_pose` must share one frame. The Franka stack env uses env-origin-relative "world" frame throughout; `PickPlaceRelMimicEnv` (`envs/pick_place_mimic_env.py`) instead overrides `get_object_poses` to **robot base frame** (via `subtract_frame_transforms` with robot root pose) for envs whose obs are base-frame. For a table-mounted Franka whose base sits at env origin the two coincide; pick ONE and keep eef obs consistent with it.
- `serialize()` → `dict(env_name=self.spec.id, type=2, env_kwargs={})`.

`self.cfg.subtask_configs` keys define the eef names; every impl uses `list(self.cfg.subtask_configs.keys())[0]` for the single-arm case (name is arbitrary, e.g. `"franka"`).

## 2. MimicEnvCfg / DataGenConfig / SubTaskConfig

`@configclass class MimicEnvCfg` fields: `datagen_config: DataGenConfig`, `subtask_configs: dict[str, list[SubTaskConfig]]`, `task_constraint_configs: list[SubTaskConstraintConfig]` (empty for single-arm).

Cfg class pattern (multiple inheritance; MimicEnvCfg's `__post_init__` is NOT auto-called, fill values in your own `__post_init__` after `super().__post_init__()`):
```python
@configclass
class FrankaCupPlaceIKRelMimicEnvCfg(FrankaCupPlaceEnvCfg, MimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.datagen_config.name = "demo_src_cup_place_task_D0"
        ...
        self.subtask_configs["franka"] = [SubTaskConfig(...), SubTaskConfig(...)]
```

### DataGenConfig fields (defaults) and FrankaCubeStackIKRelMimicEnvCfg values
| field | default | Franka stack value | meaning |
|---|---|---|---|
| `name` | "demo" | "demo_src_stack_isaac_lab_task_D0" | run label only |
| `generation_guarantee` | True | True | retry until `generation_num_trials` **successes** (else stop after N attempts) |
| `generation_keep_failed` | False | True | export failed episodes too (into `<name>_failed.hdf5`) |
| `max_num_failures` | 50 | 25 | declared, but **not read anywhere in the generation loop** — do not rely on it |
| `seed` | 1 | 1 | seeds random/np/torch in `generate_dataset.py` |
| `source_dataset_path`/`generation_path`/`task_name` | None | — | unused placeholders (CLI wins) |
| `generation_num_trials` | 10 | 10 | overridden by CLI `--generation_num_trials` |
| `generation_select_src_per_subtask` | False | True | re-select source demo at each subtask boundary (needs temporally-consistent subtasks; True is what stack/place use) |
| `generation_select_src_per_arm` | False | (default) | multi-arm only |
| `generation_transform_first_robot_pose` | False | False | if True, prepend current robot pose to every subtask segment (not just first) |
| `generation_interpolate_from_last_target_pose` | True | True | interpolate next subtask from last *target* waypoint instead of achieved pose |
| `use_skillgen` | False | False | cuRobo motion-planner transitions; needs `subtask_start_offset_range` + start-signal annotations |

GOTCHA: the shipped cfgs also set `self.datagen_config.generation_relative = True` — this field does **not exist** in `DataGenConfig` and is never read by any datagen code (grep confirms only assignments). Harmless cargo cult; omit it.

### SubTaskConfig fields
- `object_ref: str` — scene asset name the segment is object-centric to (must be a key of `get_object_poses()`, i.e. a rigid object name in the scene). `None` only for free-space segments (then `selection_strategy` must be `"random"` and the segment is replayed untransformed).
- `subtask_term_signal: str | None` — name of the boundary signal; `None` for the last subtask.
- `subtask_term_offset_range: (int,int)` — random extra steps appended past the detected 0→1 boundary. MUST be `(0,0)` for the last subtask (asserted in `DataGenerator.__init__`). Sanity check at pool-load: `end[i-1] + max_offset[i-1] < end[i] + min_offset[i]` — i.e. consecutive boundaries in *every source demo* must be farther apart than the offsets, else assertion error at load time.
- `selection_strategy: str` — `"random" | "nearest_neighbor_object" | "nearest_neighbor_robot_distance"` (registry in `datagen/selection_strategy.py`). NN strategies need `object_ref`. kwargs (`pos_weight=1.0, rot_weight=1.0, nn_k=3`) passed via `selection_strategy_kwargs` (stack uses `{"nn_k": 3}`).
- `first_subtask_start_offset_range: (0,0)` — start offset of subtask 0 (note: implementation bug — `randomize_subtask_boundaries` samples `randint(low, low+1)`, so only the low end is ever used).
- `subtask_start_offset_range: (0,0)` — skillgen only.
- `action_noise: float = 0.03` — Gaussian amplitude applied inside `target_eef_pose_to_action` per waypoint (stack: 0.03; agibot place: 0.01 — for a precise place, prefer 0.01–0.02).
- `num_interpolation_steps: int = 5` — linear-interp steps bridging previous segment → this segment (agibot uses 15 for the place subtask to smooth the long transit).
- `num_fixed_steps: int = 0` — extra steps holding the first target pose.
- `apply_noise_during_interpolation: bool = False`.
- `description` / `next_subtask_description: str` — UI text only (record_demos instruction display).

### Recommended 2-subtask grasp→place (cup onto target) config
```python
self.datagen_config.name = "demo_src_cup_place_D0"
self.datagen_config.generation_guarantee = True
self.datagen_config.generation_keep_failed = True
self.datagen_config.generation_num_trials = 10          # override on CLI
self.datagen_config.generation_select_src_per_subtask = True
self.datagen_config.generation_transform_first_robot_pose = False
self.datagen_config.generation_interpolate_from_last_target_pose = True
self.datagen_config.seed = 1

subtask_configs = [
    SubTaskConfig(                       # subtask 1: reach+grasp cup, motion relative to CUP
        object_ref="cup",                # scene rigid-object name
        subtask_term_signal="grasp_1",   # obs term name in subtask_terms group
        subtask_term_offset_range=(10, 20),
        selection_strategy="nearest_neighbor_object",
        selection_strategy_kwargs={"nn_k": 3},
        action_noise=0.03,               # lower to 0.01 if place precision suffers
        num_interpolation_steps=5,
        num_fixed_steps=0,
        apply_noise_during_interpolation=False,
    ),
    SubTaskConfig(                       # subtask 2 (LAST): transport+place, motion relative to the PLACE TARGET
        object_ref="target",             # e.g. saucer/plate asset name; stack task uses the *lower* cube here
        subtask_term_signal=None,        # last-subtask rule
        subtask_term_offset_range=(0, 0),# last-subtask rule (asserted)
        selection_strategy="nearest_neighbor_object",
        selection_strategy_kwargs={"nn_k": 3},
        action_noise=0.03,
        num_interpolation_steps=5,
        num_fixed_steps=0,
        apply_noise_during_interpolation=False,
    ),
]
self.subtask_configs["franka"] = subtask_configs
```
Precedent: the stack task's stack-subtask references the *destination* object (`cube_1`), so a place subtask should reference the fixed place target, not the carried cup. If the place target never moves across resets, transformation for subtask 2 is identity-ish and NN selection degenerates gracefully (all distances equal → uniform among top-k). (Aside: `agibot_place_upright_mug` uses `object_ref="mug"` for both subtasks because its goal is mug-relative; either is workable, but destination-relative matches MimicGen semantics when the destination is randomized.)

Subtask segmentation math (DataGenInfoPool._add_episode): boundary = first index where signal goes 0→1, `end_index = transition_idx + 1 (+1 for slicing)`; segment i spans `[end_{i-1}, end_i)`; last subtask ends at `len(actions)`. Signals must therefore be monotone 0…0,1…1 within a demo (they are stored per-step booleans; only the *first* transition is used).

## 3. get_subtask_term_signals ↔ observation group contract

Convention used by all shipped envs: an ObservationGroupCfg named **`subtask_terms`** with `concatenate_terms=False`, one boolean ObsTerm per non-final subtask, term name == `subtask_term_signal` string:

```python
@configclass
class SubtaskCfg(ObsGroup):
    grasp_1 = ObsTerm(func=mdp.object_grasped, params={
        "robot_cfg": SceneEntityCfg("robot"),
        "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        "object_cfg": SceneEntityCfg("cup")})
    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False
...
subtask_terms: SubtaskCfg = SubtaskCfg()
```
and in the Mimic env class:
```python
def get_subtask_term_signals(self, env_ids=None):
    if env_ids is None: env_ids = slice(None)
    subtask_terms = self.obs_buf["subtask_terms"]
    return {"grasp_1": subtask_terms["grasp_1"][env_ids]}   # last subtask: no signal needed
```
The group name `subtask_terms` is only a convention between your cfg and your `get_subtask_term_signals` implementation — annotate_demos only calls the method. But `record_demos.py` also reads `obs[0].get("subtask_terms")` for its instruction display, so keep the name. `mdp.object_grasped` (stack/mdp/observations.py) needs cfg attributes `gripper_joint_names` (`["panda_finger_.*"]`), `gripper_open_val` (0.04), `gripper_threshold` (0.005) on the env cfg; grasp condition = ee-to-object distance < `diff_threshold` (default 0.06 m) AND both finger joints away from open by > threshold. For the place-success obs you can mimic `object_stacked` (xy dist < 0.05, height diff vs expected < 0.005, gripper open) — only needed as a *success* termination for a 2-subtask task, not as a subtask signal.

Success termination contract (needed by ALL three scripts): env cfg must have `terminations.success = DoneTerm(func=<your success fn>)` — scripts do `success_term = env_cfg.terminations.success; env_cfg.terminations.success = None` and then call `success_term.func(env, **success_term.params)` manually; annotate/generate then set `env_cfg.terminations = None` entirely (episodes run to fixed length; no time_out during datagen). `annotate_demos.py` raises `NotImplementedError` if no `terminations.success` attr.

Gym registration convention (`source/isaaclab_mimic/isaaclab_mimic/envs/__init__.py`): id `"Isaac-<Task>-Mimic-v0"`, `entry_point="<module>:<YourMimicEnvClass>"`, `kwargs={"env_cfg_entry_point": <YourMimicEnvCfgClass>}`, `disable_env_checker=True`. Both mimic scripts `import isaaclab_mimic.envs` unconditionally to trigger registration — for a custom package, your task id must be resolvable by `gym.make` after `import isaaclab_tasks` + `import isaaclab_mimic.envs`; simplest: make your own package register on import and either add `import your_pkg` to a copied script or install it as an `isaaclab_tasks` external extension (parse_env_cfg only needs the gym id). **Uncertainty:** stock scripts have no `--import-module`-style hook, so plan to copy the two scripts into your project and add one import line.

Also note: annotate/generate resolve the task from `--task` OR from the `env_name` attr stored in the input HDF5 (`data.attrs["env_args"]["env_name"]`, written from `env_cfg.env_name` at recording time). If you recorded with the *non-mimic* task id, you MUST pass `--task <...>-Mimic-v0` to annotate (it does `env_name = args_cli.task.split(":")[-1]`).

## 4. Script CLIs

All scripts append `AppLauncher.add_app_launcher_args`: relevant common flags `--headless`, `--enable_cameras`, `--device {cuda:0|cpu|...}`, `--kit_args`, `--xr`. Invoke via `./isaaclab.sh -p <script> ...`.

### scripts/tools/record_demos.py
- `--task` (required), `--teleop_device` (default `keyboard`; built-ins keyboard/spacemouse/gamepad, or a key from env cfg's `teleop_devices` DevicesCfg, e.g. `handtracking` → forces `--xr`), `--dataset_file` (default `./datasets/dataset.hdf5`), `--step_hz` (default 30, rate-limits the loop), `--num_demos` (default 0 = infinite), `--num_success_steps` (default 10 consecutive success steps → export), `--enable_pinocchio`.
- Behavior: `parse_env_cfg(task, num_envs=1)`; strips success term; `terminations.time_out=None`; `observations.policy.concatenate_terms=False`; `recorders = ActionStateRecorderManagerCfg()` with `dataset_export_mode=EXPORT_SUCCEEDED_ONLY`; loop: `action = teleop_interface.advance()`, `env.step`, manual success check; on success: `recorder_manager.record_pre_reset([0], force_export_or_skip=False)` → `set_success_to_episodes([0], True)` → `export_episodes([0])`. Keyboard `R` resets/discards.
- Docs example: `./isaaclab.sh -p scripts/tools/record_demos.py --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --device cpu --teleop_device spacemouse --dataset_file ./datasets/dataset.hdf5 --num_demos 10` (record on the **non**-Mimic task id; keyboard teleop is clunky — subtask offsets require demos to pause; docs recommend spacemouse).

### scripts/imitation_learning/isaaclab_mimic/annotate_demos.py
- `--task` (default None → env name read from dataset), `--input_file` (default `./datasets/dataset.hdf5`), `--output_file` (default `./datasets/dataset_annotated.hdf5`), `--auto` (use `get_subtask_term_signals`; without it: interactive keyboard marking, needs GUI: N=play, B=pause, S=mark, Q=skip), `--enable_pinocchio`, `--annotate_subtask_start_signals` (skillgen), + AppLauncher flags.
- Always `num_envs=1`. Replays each episode via `env.reset_to(episode.data["initial_state"], None, is_relative=True)` + stepping recorded `actions`; recorder cfg = `MimicRecorderManagerCfg(ActionStateRecorderManagerCfg)` with extra pre-step terms writing `obs/datagen_info/{object_pose,eef_pose,target_eef_pose}` (from your mimic API) and `obs/datagen_info/subtask_term_signals` (from `get_subtask_term_signals`). Episode exported only if replay reaches success AND every subtask signal fired (`torch.any`); success attr forcibly True on exported episodes. Exit code = number annotated. Deterministic-replay caveat: replay must reproduce success on your machine (physics determinism); non-reproducing demos are silently dropped ("The final task was not completed.").
- Visuomotor: add `--enable_cameras` (docs: `--device cpu --enable_cameras --task ...Visuomotor-Mimic-v0 --auto`).

### scripts/imitation_learning/isaaclab_mimic/generate_dataset.py
- `--task` (default None → from dataset attr), `--generation_num_trials` (overrides cfg), `--num_envs` (default 1; parallel async generators sharing one vectorized env), `--input_file` (required; = annotated output), `--output_file` (default `./datasets/output_dataset.hdf5`), `--pause_subtask` (debug, blocks on `input()`), `--enable_pinocchio`, `--use_skillgen`, + AppLauncher flags (`--headless`, `--enable_cameras`, `--device`).
- Flow: `setup_env_config` (datagen/generation.py): `parse_env_cfg(num_envs)`, extract+remove success term, `terminations=None`, `observations.policy.concatenate_terms=False`, `recorders=ActionStateRecorderManagerCfg()` with `dataset_export_dir_path/dataset_filename` from `--output_file`; export mode = `EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES` if `generation_keep_failed` else `EXPORT_SUCCEEDED_ONLY`. Seeds from `datagen_config.seed`. `DataGenInfoPool.load_from_dataset_file(input)` parses subtask boundaries (assertion errors here mean bad annotations/offset ranges). Loop ends when `num_success >= generation_num_trials` (guarantee=True) or `num_attempts >= trials` (False). Prints running `X/Y (Z%) successful demos generated by mimic`.
- Docs examples: state `--device cpu --num_envs 10 --generation_num_trials 10 --input_file ./datasets/annotated_dataset.hdf5 --output_file ./datasets/generated_dataset_small.hdf5`; large vision run: `--device cpu --enable_cameras --headless --num_envs 10 --generation_num_trials 1000 --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0`. (GPU device also works; cluster note: respect `--num_envs` GPU headroom per repo CLAUDE.md.)

Task-id conventions: teleop/record/replay/train use `Isaac-…-v0`; annotate/generate use `Isaac-…-Mimic-v0`; vision variants `Isaac-…-Visuomotor-v0` / `Isaac-…-Visuomotor-Mimic-v0`.

## 5. Success/failure marking & filtering of generated episodes

- During generation each waypoint step evaluates `success_term.func(env,...)[env_id]` (`MultiWaypoint.execute`); trial success = any step succeeded (`generated_success or exec_results["success"]`; sticky once true even if the state later degrades — success is latched, episode still runs to the end of all subtask trajectories).
- End of trial: `recorder_manager.set_success_to_episodes(env_id, success)` then `export_episodes(env_id)`. `HDF5DatasetFileHandler.write_episode` writes each episode as `data/demo_<n>` with **group attr `success`** (bool) plus datasets `initial_state/...`, `obs/...`, `actions`, `states/...`, `processed_actions`; file-level `data.attrs["env_args"]` json contains `env_name`, `type: 2`, and `sim_args` metadata.
- Filtering happens at export time via `DatasetExportMode`: `EXPORT_SUCCEEDED_ONLY` → failures never written; `EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES` (used when `generation_keep_failed=True`) → successes in `<output>.hdf5`, failures in `<output>_failed.hdf5`. So the main output file contains only successful demos either way; no post-hoc filtering needed. `exported_successful_episode_count` on the recorder manager tracks counts.

## 6. record_demos.py: teleop-only? scripted alternative

`record_demos.py` is **teleop-only**: the action comes from `teleop_interface.advance()` (created via `create_teleop_device` from env cfg `teleop_devices` or fallback Se3Keyboard/Se3SpaceMouse) and it `exit(1)`s if no teleop device can be built. There is no policy/scripted hook.

Cleanest scripted-expert path (recorders are engine-level, not script-level):
1. In your env cfg (the *non*-Mimic task is fine, but the recorded `env_name` must be resolvable later or you pass `--task` to annotate): set
   ```python
   env_cfg.terminations.time_out = None
   env_cfg.observations.policy.concatenate_terms = False   # keep obs as dict in hdf5
   env_cfg.recorders = ActionStateRecorderManagerCfg()      # isaaclab.envs.mdp.recorders.recorders_cfg
   env_cfg.recorders.dataset_export_dir_path = out_dir
   env_cfg.recorders.dataset_filename = name               # no extension
   env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY  # isaaclab.managers
   ```
2. Step the env with your scripted policy. Recording is automatic: `ManagerBasedRLEnv.step` calls `recorder_manager.record_pre_step` (actions + flat policy obs) and `record_post_step` (states); reset records `initial_state`.
3. Two export options:
   a. record_demos-style manual export when your own success monitor fires:
      `env.recorder_manager.record_pre_reset([0], force_export_or_skip=False); env.recorder_manager.set_success_to_episodes([0], torch.tensor([[True]], device=env.device)); env.recorder_manager.export_episodes([0])`; then `env.sim.reset(); env.recorder_manager.reset(); env.reset()`.
   b. fully automatic: keep `terminations.success` ACTIVE in the cfg — `RecorderManager.record_pre_reset` (called on env reset) auto-reads `termination_manager.get_term("success")`, sets the success attr, and auto-exports (`export_in_record_pre_reset=True` default) with `EXPORT_SUCCEEDED_ONLY` dropping failures. This works with `num_envs>1` for parallel scripted collection (episode buffers and demo counts are per-env-id).
   Recorder term classes involved: `InitialStateRecorder`, `PostStepStatesRecorder`, `PreStepActionsRecorder`, `PreStepFlatPolicyObservationsRecorder`, `PostStepProcessedActionsRecorder` (`envs/mdp/recorders/recorders.py`).
4. The HDF5 is then identical in schema to teleop output; feed to `annotate_demos.py`. Requirements for annotation to work: `initial_state` + `actions` must deterministically replay (annotate re-executes actions from `reset_to`), and the action space must match the Mimic task's. **The source demos do NOT need `datagen_info` — annotate adds it.** Scripted expert must also behave like the offsets expect (e.g. brief dwell after grasp so `subtask_term_offset_range=(10,20)` doesn't collide with the next boundary — see §2 sanity check; if demos are too tight, shrink offsets to e.g. (5,10)).
   Sanity-check replay first with `scripts/tools/replay_demos.py --task <id> --dataset_file <file>`.

## 7. Does generate_dataset.py record camera obs? — YES, iff cameras are in the policy obs group

Trace: `setup_env_config` sets `env_cfg.observations.policy.concatenate_terms = False` and installs `ActionStateRecorderManagerCfg`. Its term `PreStepFlatPolicyObservationsRecorder.record_pre_step` returns `("obs", self._env.obs_buf["policy"])` — with concatenation off this is a dict `{term_name: tensor}`, and `RecorderManager.add_to_episodes` + `HDF5DatasetFileHandler.write_episode` recurse into dicts, creating `obs/<term_name>` datasets. Therefore every ObsTerm in the **policy** group is written per-step, including image terms. `FrankaCubeStackVisuomotorEnvCfg` puts `table_cam`/`wrist_cam` `ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False})` (84x84 uint8 RGB) directly in `PolicyCfg`, plus `self.rerender_on_reset = True`, `self.sim.render.antialiasing_mode = "OFF"`, and `self.image_obs_list = ["table_cam", "wrist_cam"]` (the latter consumed by robomimic tooling/hdf5-to-mp4, not by the recorder). Output HDF5 gets `obs/table_cam`, `obs/wrist_cam` of shape `(T, 84, 84, 3)`.
- `--enable_cameras` itself only enables the omniverse camera rendering pipeline (AppLauncher flag); it does not add recording. Without it, a camera-bearing env cfg fails at scene construction; with it but with a camera-less env cfg, nothing extra is recorded.
- Base `StackEnvCfg` has an empty `RGBCameraPolicyCfg` group named `rgb_camera` — a non-policy group is NOT recorded (only `obs_buf["policy"]` is). For the cup-place task: put camera ObsTerms in the `policy` group of a dedicated `...VisuomotorMimicEnvCfg`, register it as `Isaac-...-Visuomotor-Mimic-v0`, and run BOTH annotate and generate with `--enable_cameras`. Camera resolution multiplies HDF5 size and datagen time (docs run visuomotor generation at `--num_envs 10`; state-only can go higher).
- GOTCHA: with `--num_envs > 1`, `obs_buf["policy"]["table_cam"]` rows for *other* envs are recorded into each env's episode slice correctly (`add_to_episodes` indexes by env id), but all envs share the render products — keep `--num_envs` moderate for vision (5–10 in docs).
- For LeRobot conversion later: images are stored uint8 HWC per step under `obs/<cam>`; actions under `actions`; episode grouping `data/demo_i` with `success` attr.

## Minimal end-to-end recipe (custom cup-place)
1. Task package: `FrankaCupPlaceEnvCfg` (scene: robot, `ee_frame`, `cup`, `target`; `terminations.success`; `subtask_terms` obs group with `grasp_1`; policy obs incl. `eef_pos`, `eef_quat`; gripper cfg attrs) + gym ids `Isaac-CupPlace-Franka-IK-Rel-v0` and `...-Mimic-v0` (+ `...-Visuomotor-Mimic-v0` with cams in policy group).
2. Mimic classes: `FrankaCupPlaceIKRelMimicEnv(FrankaCubeStackIKRelMimicEnv)` overriding only `get_subtask_term_signals` (return `{"grasp_1": ...}`); cfg as §2.
3. Record ~10 source demos (teleop `record_demos.py`, or scripted per §6) → `datasets/dataset.hdf5`.
4. `annotate_demos.py --task Isaac-CupPlace-Franka-IK-Rel-Mimic-v0 --auto --input_file datasets/dataset.hdf5 --output_file datasets/annotated_dataset.hdf5` (add `--enable_cameras` only if the mimic cfg has cameras; state-only cfg is fine for annotation even if you later generate with the visuomotor cfg — the visuomotor cfg must share the same actions/scene so `initial_state` replays).
5. `generate_dataset.py --task Isaac-CupPlace-Franka-IK-Rel-Visuomotor-Mimic-v0 --enable_cameras --headless --num_envs 10 --generation_num_trials 1000 --input_file datasets/annotated_dataset.hdf5 --output_file datasets/generated_dataset.hdf5`.

## Uncertainties / blockers flagged
- Custom-package import hook: stock annotate/generate scripts only import `isaaclab_mimic.envs` + `isaaclab_tasks`; a custom task package likely needs copied scripts with one added import (verify how project's own tasks get registered — e.g. via `isaaclab_tasks` entry-point discovery in v2.3.0, not confirmed here).
- `max_num_failures` is dead config (never read in `generation.py` loop) — infinite failure loops are possible if generation success rate is ~0; watch the printed success-rate banner.
- `first_subtask_start_offset_range` high end ignored (randint(low, low+1) bug) — don't rely on it.
- Annotation `--auto` requires deterministic replay success; on GPU PhysX small nondeterminism can drop demos — record and annotate on the same device type (docs use `--device cpu` for all Franka stack teleop/annotate examples).
- Cross-cfg annotate→generate (state cfg for annotate, visuomotor cfg for generate) is what NVIDIA's own pipeline does for stack (annotate on `Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0`, generate on `...Visuomotor-Mimic-v0`), so it is supported as long as both cfgs share scene/action/init-state structure.
- `MultiWaypoint.execute` checks success only on the recorded env's obs after the vectorized step; success latching (§5) means a knocked-over cup after success still exports as success — make the success term strict (e.g. require settled cup pose + gripper open) if that matters.
