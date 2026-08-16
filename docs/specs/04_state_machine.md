# SPEC 04 — Scripted expert state machines in Isaac Lab v2.3.0 (for Franka cup-place demo collection)

Checkout studied (read-only): `/home/admin_07/cost_of_generality/third_party/IsaacLab` (v2.3.0).
Primary sources (read in full):
- `scripts/environments/state_machine/lift_cube_sm.py`
- `scripts/environments/state_machine/open_cabinet_sm.py`
- `scripts/tools/record_demos.py`
- `scripts/imitation_learning/isaaclab_mimic/annotate_demos.py`, `generate_dataset.py`
- `source/isaaclab_mimic/isaaclab_mimic/envs/franka_stack_ik_rel_mimic_env{,_cfg}.py`, `franka_stack_ik_abs_mimic_env{,_cfg}.py`, `envs/__init__.py`
- `source/isaaclab_tasks/.../manipulation/lift/config/franka/{joint_pos,ik_abs,ik_rel}_env_cfg.py`, `.../lift/lift_env_cfg.py`
- `source/isaaclab_tasks/.../manipulation/stack/stack_env_cfg.py`, `config/franka/stack_{joint_pos,ik_rel}_env_cfg.py`, `mdp/observations.py`, `mdp/terminations.py`
- `source/isaaclab/isaaclab/managers/recorder_manager.py`, `envs/mdp/recorders/recorders_cfg.py`
- `source/isaaclab/isaaclab/envs/mdp/actions/binary_joint_actions.py`, `task_space_actions.py`, `controllers/differential_ik.py`
- `source/isaaclab/isaaclab/envs/mimic_env_cfg.py`, `manager_based_rl_mimic_env.py`
- `source/isaaclab_mimic/isaaclab_mimic/datagen/{data_generator,datagen_info_pool,waypoint}.py`

---

## 1. Architecture of the shipped state machines

### 1.1 Overall structure (both scripts identical in shape)

```
AppLauncher(headless=...) -> simulation_app
parse_env_cfg("<TASK-IK-Abs-v0>", device, num_envs, use_fabric=not disable_fabric)
env = gym.make(task, cfg=env_cfg); env.reset()
sm = PickAndLiftSm(dt=env_cfg.sim.dt * env_cfg.decimation, num_envs, device, position_threshold=0.01)
loop:
    dones = env.step(actions)[-2]          # NOTE: index -2 == time_out/truncated tensor, see 1.7
    read poses from scene sensors (NOT from obs buffer)
    actions = sm.compute(ee_pose, object_pose, des_object_pose)   # (num_envs, 8)
    if dones.any(): sm.reset_idx(dones.nonzero(...))
```

The SM is a **per-env finite state machine evaluated in a Warp GPU kernel** (`@wp.kernel infer_state_machine`), launched with `dim=num_envs`; each thread `tid` handles one env. The Python class (`PickAndLiftSm` / `OpenDrawerSm`) owns torch buffers and zero-copy warp views created once via `wp.from_torch(...)`:

Per-env persistent state (torch tensors, shape `(num_envs,)` unless noted):
- `sm_dt` (float, filled with `env_cfg.sim.dt * env_cfg.decimation` = control step; lift: 0.01*2=0.02 s, stack: 0.01*5=0.05 s)
- `sm_state` (`torch.int32`) — current state id
- `sm_wait_time` (float) — time accumulated in current state, incremented by `dt` **every kernel call** (last line of kernel), zeroed on transition
- `des_ee_pose` `(num_envs, 7)` — kernel output, wp.transform layout `(px,py,pz,qx,qy,qz,qw)`
- `des_gripper_state` (float) — kernel output, `GripperState.OPEN=wp.constant(1.0)` / `CLOSE=wp.constant(-1.0)`
- constant per-env transforms: `offset` (lift: `offset[:,2]=0.1; offset[:,-1]=1.0` i.e. +10 cm z, identity rot); cabinet adds `handle_approach_offset` (x=-0.1), `handle_grasp_offset` (x=+0.025), `drawer_opening_rate` (x=-0.015)

### 1.2 Quaternion layout gymnastics (gotcha)

Torch side uses Isaac Lab convention `(w,x,y,z)`; warp `wp.transform` is `(px,py,pz,qx,qy,qz,qw)`.
- into kernel: `pose = pose[:, [0,1,2,4,5,6,3]]` (wxyz→xyzw) then `wp.from_torch(pose.contiguous(), wp.transform)`
- out of kernel: `des_ee_pose = self.des_ee_pose[:, [0,1,2,6,3,4,5]]` (xyzw→wxyz)
- constant offsets are stored directly in warp layout, hence `offset[:, -1] = 1.0  # qw`

### 1.3 Kernel logic (lift_cube_sm)

States (`wp.constant(int)`): `REST=0, APPROACH_ABOVE_OBJECT=1, APPROACH_OBJECT=2, GRASP_OBJECT=3, LIFT_OBJECT=4`.
Wait times (s): `REST=0.2, APPROACH_ABOVE_OBJECT=0.5, APPROACH_OBJECT=0.6, GRASP_OBJECT=0.3, LIFT_OBJECT=1.0`.

Transition rule per state: set `des_ee_pose[tid]` + `gripper_state[tid]`; transition when (optionally) `distance_below_threshold(ee_pos, des_pos, position_threshold)` (`wp.length(a-b) < threshold`, position only, threshold 0.01 m) **and** `sm_wait_time >= WaitTime.X`; on transition `sm_wait_time = 0`.

- `REST`: des=current ee pose, gripper OPEN; time-only gate (0.2 s)
- `APPROACH_ABOVE_OBJECT`: `des = wp.transform_multiply(offset, object_pose)`, OPEN; distance gate + wait. **Bug in stock file: the wait check uses `PickSmWaitTime.APPROACH_OBJECT` (0.6) not `APPROACH_ABOVE_OBJECT` (0.5)** — harmless but copy carefully.
- `APPROACH_OBJECT`: `des = object_pose`, OPEN; distance + wait 0.6 s
- `GRASP_OBJECT`: `des = object_pose`, CLOSE; time-only 0.3 s (gripper actually closes during this dwell)
- `LIFT_OBJECT`: `des = des_object_pose` (the goal), CLOSE; distance + wait, then **transitions to itself** (terminal; episode ends by env time_out and the env auto-resets)

`wp.transform_multiply(a, b)` composes as matrix product `a*b` (apply `b` first, then `a`), so with an identity-rotation offset the result is a **world-frame** translation: `p = offset.p + object.p`. To get an **object-frame** offset you must instead do `transform_multiply(object_pose, offset)`. The stock lift offset (+z world) and cabinet offset (-x world) only work because table-top/world alignment makes it equivalent; for a cup with randomized yaw, grasp offsets along the handle must be object-frame (compose in torch before the kernel, or swap the multiply order).

### 1.4 Where poses come from (not the obs manager)

```python
ee_frame_sensor = env.unwrapped.scene["ee_frame"]                    # FrameTransformer
tcp_pos  = ee_frame_sensor.data.target_pos_w[..., 0, :] - env.unwrapped.scene.env_origins
tcp_quat = ee_frame_sensor.data.target_quat_w[..., 0, :]
object_pos = env.unwrapped.scene["object"].data.root_pos_w - env.unwrapped.scene.env_origins
desired_pos = env.unwrapped.command_manager.get_command("object_pose")[..., :3]   # lift goal command
```
All are env-local frames (env_origins subtracted; the robot base sits at each env origin). The **object orientation fed to the kernel is not the real one** — `desired_orientation` with `w=0, x=1` (quat `(0,1,0,0)` = 180° about x = gripper pointing straight down) is passed for both `object_pose` and `des_object_pose`, so the EE always descends top-down with fixed yaw. For randomized-yaw objects you must build the grasp orientation yourself (e.g. compose downward quat with object yaw via `isaaclab.utils.math.quat_mul`).

### 1.5 Action packing and the env it drives

`sm.compute()` returns `torch.cat([des_ee_pose(wxyz), des_gripper_state.unsqueeze(-1)], dim=-1)` → shape `(num_envs, 8)`, fed directly to `env.step()`.

Driven task: **`Isaac-Lift-Cube-Franka-IK-Abs-v0`** (cabinet: `Isaac-Open-Drawer-Franka-IK-Abs-v0`). Action space = `DifferentialInverseKinematicsActionCfg` in **absolute** mode + `BinaryJointPositionActionCfg`:

```python
# lift/config/franka/ik_abs_env_cfg.py
self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # stiff PD for IK tracking
self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
    asset_name="robot", joint_names=["panda_joint.*"], body_name="panda_hand",
    controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
    body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
)
# gripper (inherited from joint_pos_env_cfg.py)
self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
    asset_name="robot", joint_names=["panda_finger.*"],
    open_command_expr={"panda_finger_.*": 0.04}, close_command_expr={"panda_finger_.*": 0.0},
)
```

`DifferentialIKController.action_dim`: `"position"`→3; `"pose"+relative`→6 `(dx,dy,dz,drx,dry,drz)`; `"pose"+absolute`→**7 `(x,y,z,qw,qx,qy,qz)`**. Total env action dim = 7+1=8 (IK-Abs) or 6+1=7 (IK-Rel). Initial action buffer in the scripts: zeros with `actions[:, 3] = 1.0` (identity quat, gripper 0 → open, see §5).

Frame subtlety: IK controls `panda_hand` + `body_offset.pos=[0,0,0.107]`, while the `ee_frame` FrameTransformer used as "current pose" has offset `[0,0,0.1034]` (TCP between fingertips). ~3.6 mm systematic mismatch, absorbed by `position_threshold=0.01`. Keep both numbers identical in the new task if you want exact convergence, or keep the stock pair and the 1 cm threshold.

### 1.6 dt used by the SM

`PickAndLiftSm(env_cfg.sim.dt * env_cfg.decimation, ...)` — the SM ticks once per `env.step()`, so `sm_dt` must be the control dt, not the physics dt. Lift env: `sim.dt=0.01, decimation=2` → 0.02 s. Stack env (basis for our task): `sim.dt=0.01, decimation=5` → 0.05 s (`stack_env_cfg.py __post_init__`). Wait-time constants are in seconds and hence portable.

### 1.7 Known wart: `dones = env.step(actions)[-2]`

`ManagerBasedRLEnv.step` returns `(obs, reward, terminated, truncated, extras)`; index `-2` is **`truncated` (time_out) only**. Episodes that reset due to other termination terms (object dropped, success) would NOT reset the SM in the stock script. For the new task use:

```python
obs, rew, terminated, truncated, info = env.step(actions)
dones = terminated | truncated
```

---

## 2. IK-Abs vs IK-Rel, and reconciling with Mimic

### 2.1 What the shipped Mimic pipeline uses

Registered mimic envs (`source/isaaclab_mimic/isaaclab_mimic/envs/__init__.py`): `Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0` (class `FrankaCubeStackIKRelMimicEnv`, cfg `FrankaCubeStackIKRelMimicEnvCfg(FrankaCubeStackEnvCfg, MimicEnvCfg)`), plus `...IK-Abs-Mimic-v0` (`FrankaCubeStackIKAbsMimicEnv`), visuomotor variants (`Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0` — camera obs, this is the template for our vision pipeline), Blueprint/Skillgen/Bin variants. So **FrankaCubeStack mimic canonically uses IK-Rel** (docs teleop with `Isaac-Stack-Cube-Franka-IK-Rel-v0`), and an IK-Abs mimic env also exists in-tree.

Key mimic env API (must implement in our custom env class deriving `ManagerBasedRLMimicEnv`):
- `get_robot_eef_pose(eef_name, env_ids)` → 4x4 pose, read from `self.obs_buf["policy"]["eef_pos"|"eef_quat"]` → **policy obs group must contain `eef_pos`/`eef_quat` terms and `concatenate_terms=False`** (stack env: `mdp.ee_frame_pos`, `mdp.ee_frame_quat`).
- `target_eef_pose_to_action(target_eef_pose_dict, gripper_action_dict, action_noise_dict, env_id)`:
  - IK-Rel version: `delta_position = target_pos - curr_pos`; `delta_rotation = axis_angle(target_rot @ curr_rot.T)`; noise added then `clamp(-1,1)`; `cat([pose_action, gripper_action])`. **No division by the action `scale=0.5`** — the delta acts as a P-controller and mimic simply re-issues the (moving) target each step.
  - IK-Abs version: `cat([target_pos, quat_from_matrix(target_rot), gripper_action]).unsqueeze(0)` (note: still has deprecated `noise` arg signature in v2.3.0; generate_dataset warns).
- `action_to_target_eef_pose(action)` — inverse; used by annotate to log `target_eef_pose` from recorded raw actions.
- `actions_to_gripper_actions(actions)` → `actions[:, -1:]` (last dim).
- `get_subtask_term_signals(env_ids)` → reads `self.obs_buf["subtask_terms"][...]` (requires a `subtask_terms` ObsGroup, see §6.3).
- `get_object_poses()` (base-class default) → **every rigid object in `scene.get_state(is_relative=True)["rigid_object"]`** as 4x4 poses.

### 2.2 What datagen actually consumes (verified in `datagen_info_pool.py::_add_episode`, `data_generator.py`)

From annotated source demos, only:
`obs/datagen_info/eef_pose`, `obs/datagen_info/object_pose/<name>`, `obs/datagen_info/target_eef_pose`, `obs/datagen_info/subtask_term_signals/<name>` (+optional start signals), and gripper actions extracted as `actions[:, -1:]`. Segments are transformed object-relative (`pose_in_A_to_pose_in_B`) and re-executed by calling **the generation env's** `target_eef_pose_to_action` per waypoint (`waypoint.py` L389-401). **Raw source arm actions are never replayed during generation** — only during *annotation*.

Consequence — three viable options:

- **Option A (recommended, matches shipped pipeline exactly): run the whole pipeline in IK-Rel.** Register one task `Isaac-Place-Cup-Franka-IK-Rel-v0` (+ `-Mimic-v0` + `-Visuomotor-Mimic-v0`). Drive it with the SM by converting the SM's absolute `des_ee_pose` to a per-step delta action in the Python loop (5 lines, code in §4.4). Record → `annotate_demos.py --auto` → `generate_dataset.py` → LeRobot, all on the same 7-dim IK-Rel action space. No cross-space conversion anywhere; everything replayable.
- **Option B: record IK-Abs source demos (stock SM unmodified), annotate with a custom `Isaac-Place-Cup-Franka-IK-Abs-Mimic-v0` (annotation replays raw actions, so it must match the recording action space), then run `generate_dataset.py --task Isaac-Place-Cup-Franka-IK-Rel-Mimic-v0`** on the annotated file. Structurally supported (datagen only needs `datagen_info` + last-dim gripper; the `--task` flag overrides the dataset's `env_name`), and this is the classic MimicGen design. Flag: **untested combination in this repo — no in-tree example annotates in one action space and generates in another; also `env.reset_to(initial_state)` requires both cfgs to share an identical scene.** Treat as fallback.
- **Option C: offline conversion of recorded IK-Abs actions to IK-Rel deltas** using recorded `eef_pos/eef_quat` obs. Fragile (recomputed deltas replay with compounding error through the `scale=0.5` P-controller); not recommended.

Gotcha: `datagen_config.generation_relative` (set `=True` in all IK-Rel mimic cfgs) **is written but never read anywhere in v2.3.0** (`grep` over `source/` + `scripts/` finds only assignments; the field doesn't even exist in `DataGenConfig`). Setting it is inert — do not rely on it to switch behaviors; the action space is determined solely by the mimic env's `target_eef_pose_to_action`/action cfg.

### 2.3 `SubTaskConfig` / `DataGenConfig` exact fields (`isaaclab/envs/mimic_env_cfg.py`)

`DataGenConfig`: `name`, `generation_guarantee=True`, `generation_keep_failed=False`, `max_num_failures=50`, `seed=1`, `source_dataset_path`, `generation_path`, `generation_num_trials=10`, `task_name`, `generation_select_src_per_subtask=False`, `generation_select_src_per_arm=False`, `generation_transform_first_robot_pose=False`, `generation_interpolate_from_last_target_pose=True`, `use_skillgen=False`.
`SubTaskConfig`: `object_ref`, `subtask_term_signal`, `selection_strategy="random"` (stack uses `"nearest_neighbor_object"`, `selection_strategy_kwargs={"nn_k": 3}`), `first_subtask_start_offset_range=(0,0)`, `subtask_start_offset_range=(0,0)`, `subtask_term_offset_range=(0,0)` (stack: `(10,20)`), `action_noise=0.03`, `num_interpolation_steps=5`, `num_fixed_steps=0`, `apply_noise_during_interpolation=False`, `description`, `next_subtask_description`. Attach as `self.subtask_configs["franka"] = [ ... ]`.

For cup-place: 2 subtasks — `SubTaskConfig(object_ref="cup", subtask_term_signal="grasp_cup", subtask_term_offset_range=(10,20), ...)` then `SubTaskConfig(object_ref="goal_pad", subtask_term_signal=None, subtask_term_offset_range=(0,0), ...)`. **The place segment's `object_ref` must be the goal**, so Mimic re-targets the place motion to the randomized goal pose. This forces the goal to be visible in `get_object_poses()` → **make the goal a scene `RigidObjectCfg`** (e.g. a flat pad/saucer, optionally `kinematic_enabled=True` so it never moves) rather than a `UniformPoseCommandCfg` marker. (Alternative: keep a command-based goal and override `get_object_poses()` in the mimic env to inject a virtual "goal" pose built from `command_manager` — works but bespoke; the recorded `datagen_info/object_pose` dict then contains it. Rigid-object goal is the low-risk path and also gives the vision policy something to see.)

---

## 3. Modifying lift_cube_sm into pick→move→place-at-goal→release→retreat

### 3.1 New state/wait tables

```python
class GripperState:
    OPEN = wp.constant(1.0); CLOSE = wp.constant(-1.0)

class PlaceSmState:
    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)          # go up to transport height above grasp point
    APPROACH_ABOVE_GOAL = wp.constant(5)  # translate at height to above-goal
    LOWER_TO_GOAL = wp.constant(6)        # descend to place pose
    RELEASE_OBJECT = wp.constant(7)       # open gripper, hold pose
    RETREAT = wp.constant(8)              # rise above goal, gripper open
    DONE = wp.constant(9)                 # hold retreat pose (terminal; lets success settle)

class PlaceSmWaitTime:                    # seconds, tune on the real dt (0.05 s if stack-style env)
    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.4)       # >= gripper close travel time; 0.3 stock, +margin
    LIFT_OBJECT = wp.constant(0.4)
    APPROACH_ABOVE_GOAL = wp.constant(0.6)
    LOWER_TO_GOAL = wp.constant(0.5)
    RELEASE_OBJECT = wp.constant(0.5)     # open + let cup settle before moving
    RETREAT = wp.constant(0.5)
    DONE = wp.constant(0.0)
```

### 3.2 Kernel (delta vs lift_cube_sm)

New inputs: `goal_pose: wp.array(dtype=wp.transform)`, plus reuse `offset` for both above-object and above-goal hover (or pass two offsets `pre_grasp_offset`, `place_offset` if heights differ; place hover should clear the cup height, e.g. z=0.15). Signature:

```python
@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float), sm_state: wp.array(dtype=int), sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),     # grasp pose (pos=cup grasp point, rot=downward-facing quat composed w/ cup yaw)
    goal_pose: wp.array(dtype=wp.transform),       # place pose (pos=goal xy + place height, rot=downward quat)
    des_ee_pose: wp.array(dtype=wp.transform), gripper_state: wp.array(dtype=float),
    hover_offset: wp.array(dtype=wp.transform),    # (0,0,0.10, identity)
    place_hover_offset: wp.array(dtype=wp.transform),  # (0,0,0.12, identity)
    position_threshold: float,
):
    tid = wp.tid(); state = sm_state[tid]
    if state == PlaceSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]; gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PlaceSmWaitTime.REST:
            sm_state[tid] = PlaceSmState.APPROACH_ABOVE_OBJECT; sm_wait_time[tid] = 0.0
    elif state == PlaceSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(hover_offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(wp.transform_get_translation(ee_pose[tid]),
                                    wp.transform_get_translation(des_ee_pose[tid]), position_threshold):
            if sm_wait_time[tid] >= PlaceSmWaitTime.APPROACH_ABOVE_OBJECT:
                sm_state[tid] = PlaceSmState.APPROACH_OBJECT; sm_wait_time[tid] = 0.0
    elif state == PlaceSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]; gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(...):
            if sm_wait_time[tid] >= PlaceSmWaitTime.APPROACH_OBJECT:
                sm_state[tid] = PlaceSmState.GRASP_OBJECT; sm_wait_time[tid] = 0.0
    elif state == PlaceSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]; gripper_state[tid] = GripperState.CLOSE
        if sm_wait_time[tid] >= PlaceSmWaitTime.GRASP_OBJECT:
            sm_state[tid] = PlaceSmState.LIFT_OBJECT; sm_wait_time[tid] = 0.0
    elif state == PlaceSmState.LIFT_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(hover_offset[tid], object_pose[tid])  # NOTE: object_pose is
        gripper_state[tid] = GripperState.CLOSE            # frozen at grasp (see §3.3) so this is "up from grasp point"
        if distance_below_threshold(...) and sm_wait_time[tid] >= PlaceSmWaitTime.LIFT_OBJECT: -> APPROACH_ABOVE_GOAL
    elif state == PlaceSmState.APPROACH_ABOVE_GOAL:
        des_ee_pose[tid] = wp.transform_multiply(place_hover_offset[tid], goal_pose[tid])
        gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(...) and wait: -> LOWER_TO_GOAL
    elif state == PlaceSmState.LOWER_TO_GOAL:
        des_ee_pose[tid] = goal_pose[tid]; gripper_state[tid] = GripperState.CLOSE
        if distance_below_threshold(...) and wait: -> RELEASE_OBJECT
    elif state == PlaceSmState.RELEASE_OBJECT:
        des_ee_pose[tid] = goal_pose[tid]; gripper_state[tid] = GripperState.OPEN
        if sm_wait_time[tid] >= PlaceSmWaitTime.RELEASE_OBJECT: -> RETREAT
    elif state == PlaceSmState.RETREAT:
        des_ee_pose[tid] = wp.transform_multiply(place_hover_offset[tid], goal_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        if distance_below_threshold(...) and wait: -> DONE
    elif state == PlaceSmState.DONE:
        des_ee_pose[tid] = wp.transform_multiply(place_hover_offset[tid], goal_pose[tid])
        gripper_state[tid] = GripperState.OPEN     # hold; success termination / settling counter ends the episode
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]
```

(Ellipses = same `distance_below_threshold(wp.transform_get_translation(ee_pose[tid]), wp.transform_get_translation(des_ee_pose[tid]), position_threshold)` + wait pattern as stock.)

### 3.3 Python-side changes

- `compute(ee_pose, grasp_pose, goal_place_pose)`; same wxyz↔xyzw shuffles; `wp.launch` with the new arrays.
- **Grasp pose construction (torch, before kernel):** `grasp_pos = cup.data.root_pos_w - env_origins + torch.tensor([0,0,z_grasp])` (grasp below rim / at handle); `grasp_quat = quat_mul(yaw_quat(cup_quat), DOWN_QUAT)` where `DOWN_QUAT=(0,1,0,0)` (w,x,y,z) and `yaw_quat` from `isaaclab.utils.math` extracts cup yaw — for a cylindrically symmetric cup a fixed `DOWN_QUAT` suffices (stock behavior); only needed if grasping a handle.
- **Freeze the grasp target once GRASP begins** (recommended): after grasping, the cup pose moves with the hand; feeding live `object_pose` into `LIFT_OBJECT` makes the target chase the EE (converges anyway, but produces drifting lifts). Keep a `self.grasp_pose_frozen` buffer updated only while `sm_state < GRASP_OBJECT` (mask update in torch: `upd = self.sm_state < PlaceSmState.GRASP_OBJECT.val`... simplest: update where `sm_state <= APPROACH_OBJECT`).
- **Place height:** `goal_place_pose.z = goal_pad_top_z + cup_half_height + 0.005` (drop from ~5 mm, not 0 — pressing the cup into the pad while opening ejects it).
- `reset_idx(env_ids)` unchanged (`sm_state=0, sm_wait_time=0`), plus reset frozen-grasp buffers.
- Per-env **state timeout** (new, robustness): extra buffer `sm_time_in_state`; in torch after `compute`, `stuck = sm_time_in_state > 6.0 s` → mark env failed and force `env.reset()`/skip export (see §6). Without this, an unreachable `des_ee_pose` (IK can't get there) stalls that env forever — stock scripts just rely on episode `time_out`.

### 3.4 Success settling

Follow `record_demos.py`: pop `terminations.success` out of the cfg, evaluate manually each step, require `num_success_steps` (default 10) consecutive `True` before exporting (§6.1). The `DONE` state holds the retreat pose so the cup can wobble/settle; success term (mirror `stack/mdp/terminations.py::cubes_stacked` structure): cup xy within `xy_threshold=0.04` of goal, `|cup_z - goal_top_z - cup_half_h| < 0.01`, cup lin vel < 0.1 m/s (add — cubes_stacked omits velocity), and **gripper open** (`torch.isclose(joint_pos[finger], gripper_open_val=0.04, atol=1e-4)` — exactly like `cubes_stacked`; requires `env.cfg.gripper_joint_names=["panda_finger_.*"]`, `gripper_open_val=0.04` attributes on the cfg as stack does).

### 3.5 IK-Rel drive (Option A wrapper — put in the collection script)

```python
from isaaclab.utils.math import axis_angle_from_quat, quat_mul, quat_conjugate, quat_unique
abs_target = pick_sm.compute(ee_pose_w, grasp_pose, goal_pose)      # (N, 8) pos+quat(wxyz)+grip
delta_pos  = abs_target[:, 0:3] - tcp_pos                            # world/env frame
dquat      = quat_mul(abs_target[:, 3:7], quat_conjugate(tcp_quat))  # target ∘ current⁻¹
delta_rot  = axis_angle_from_quat(quat_unique(dquat))                # 3-vector, radians
delta_pos  = delta_pos.clamp(-0.05, 0.05)                            # keep teleop-like magnitudes
delta_rot  = delta_rot.clamp(-0.2, 0.2)
actions    = torch.cat([delta_pos, delta_rot, abs_target[:, 7:8]], dim=-1)   # (N, 7) → env.step
```
This is byte-for-byte the transform `FrankaCubeStackIKRelMimicEnv.target_eef_pose_to_action` applies, so recorded actions are exactly in-distribution for the Mimic/LeRobot pipeline. Note `DifferentialInverseKinematicsActionCfg(..., scale=0.5)` on IK-Rel: commanded delta is halved per step (exponential approach). Do NOT pre-divide by the scale — mimic doesn't either; just expect ~2x settle time and tune wait times/clamps accordingly. The clamp slows convergence near large errors only.

---

## 4. Robustness across randomized object/goal poses + expected failure modes

Randomization template (`stack_joint_pos_env_cfg.py` EventCfg): `franka_stack_events.randomize_object_pose` with `pose_range={"x": (0.4, 0.6), "y": (-0.10, 0.10), "z": (h,), "yaw": (-1.0, 1.0)}`, `min_separation=0.1`, `asset_cfgs=[SceneEntityCfg("cup"), SceneEntityCfg("goal_pad")]` (put both in one call to enforce cup-goal separation); plus `set_default_joint_pose` and `randomize_joint_by_gaussian_offset(std=0.02)`.

Robustness measures:
1. **Keep the randomization range inside comfortable Franka reach** (x∈[0.35,0.65], y∈[-0.25,0.25] from base; the DLS IK has no joint-limit/collision awareness — far/low corners produce IK stall = state timeout).
2. **Approach always via hover point** (both grasp and place) — prevents side-swiping the cup/pad; hover z ≥ cup height + 5 cm.
3. **Object-frame grasp offsets** when yaw matters: `transform_multiply(object_pose, offset)` order (see §1.3); for symmetric cups use world-frame downward grasp (stock) and ignore yaw.
4. **Gripper timing**: close only after `APPROACH_OBJECT` has both distance AND dwell satisfied (stock 0.6 s); `GRASP_OBJECT` dwell must exceed finger travel (0.04 m at stock gripper PD ≈ 0.2-0.3 s; use 0.4 s). Early close ⇒ cup pushed away; late lift is only slow, never wrong — bias long.
5. **Distance threshold**: 0.01 m stock. With the ee_frame/body_offset mismatch (§1.5) don't go below ~0.007. If demos stall in APPROACH with the IK-Rel P-controller, raise to 0.015 before touching gains.
6. **Per-state timeout → abort + skip export** (§3.3); with `EXPORT_SUCCEEDED_ONLY` an aborted episode is dropped automatically on reset.
7. **Velocity-aware settling** for RELEASE: cup can rock; release dwell 0.5 s and drop height 5 mm keep it tame.

Expected failure modes (accept <100% demo yield; filter by success):
- cup slips out of closed fingers during transport (thin walls: raise gripper effort/stiffness or grasp lower on the body; stock panda_hand effort 200 N is fine for rigid cups)
- grasp hits rim edge on descent due to 1 cm threshold + 3.6 mm frame offset → cup tips: lower approach speed by raising `APPROACH_OBJECT` dwell
- IK singularity/limit stall at workspace edge → state-timeout abort
- cup bounces off pad on release (drop too high / released while still descending)
- success flicker (cup rolls off pad after gripper opens) → settling counter catches it
- env auto-reset mid-demo from a stray termination while SM mid-sequence → always compute `dones = terminated | truncated` and `sm.reset_idx` those envs (§1.7)

---

## 5. Gripper action conventions (Franka, these scripts)

- SM emits scalar last-dim: `OPEN=+1.0`, `CLOSE=-1.0` (`wp.constant`).
- `BinaryJointAction.process_actions` (`binary_joint_actions.py`): float input → `binary_mask = actions < 0` → close; **`>= 0` (including 0.0) = OPEN**. Bool input → `False`=open... (per docstring: 1/positive=open, 0/negative=close). Hence zero-initialized action buffers start "open" — safe.
- Open sets both `panda_finger_.*` joint targets to `0.04` m, close to `0.0` (`open_command_expr` / `close_command_expr`). Action dim contribution = 1 (single scalar for both fingers).
- Mimic keeps the convention: `actions_to_gripper_actions` = `actions[:, -1:]`, replayed verbatim into generated actions. LeRobot training data will therefore carry a ±1-ish binary last channel; success/grasp checks read *joint positions*, not the action (`object_grasped`: `|finger_pos - 0.04| > gripper_threshold=0.005` AND ee-object dist < 0.06; `object_stacked`/`cubes_stacked`: `isclose(finger_pos, 0.04, atol=1e-4)` i.e. fully open).
- Ordering: action vector = `[arm_action (6 rel | 7 abs), gripper_action (1)]` — order follows the ActionsCfg field declaration order (`arm_action` then `gripper_action` in lift/stack cfgs).

---

## 6. Combining the SM loop with demo recording (so annotate_demos.py accepts the output)

### 6.1 Pattern A — record_demos-style (single env, manual export). Mirrors `scripts/tools/record_demos.py`; replace teleop with SM.

```python
env_cfg = parse_env_cfg("Isaac-Place-Cup-Franka-IK-Rel-v0", device=..., num_envs=1)
env_cfg.env_name = "Isaac-Place-Cup-Franka-IK-Rel-v0"      # REQUIRED: stored as data.attrs["env_args"]["env_name"];
                                                            # annotate/generate read it via HDF5DatasetFileHandler.get_env_name()
success_term = env_cfg.terminations.success                 # pop success out; evaluate manually
env_cfg.terminations.success = None
env_cfg.terminations.time_out = None                        # run until success or manual reset
env_cfg.observations.policy.concatenate_terms = False       # REQUIRED for mimic obs_buf["policy"]["eef_pos"] access & per-term obs in hdf5
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
env_cfg.recorders = ActionStateRecorderManagerCfg()         # records: initial_state, states(post-step), actions(pre-step),
                                                            # obs (pre-step, flattened policy group), processed actions
env_cfg.recorders.dataset_export_dir_path = output_dir
env_cfg.recorders.dataset_filename = output_file_name
env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
env = gym.make(task, cfg=env_cfg).unwrapped
env.reset()
# loop:
obs, rew, term, trunc, _ = env.step(actions)                # recorder hooks fire automatically inside step()
...sm.compute -> actions...
if success_now: success_step_count += 1
else: success_step_count = 0
if success_step_count >= 10:                                # settling, as record_demos --num_success_steps
    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
    env.recorder_manager.set_success_to_episodes([0], torch.tensor([[True]], dtype=torch.bool, device=env.device))
    env.recorder_manager.export_episodes([0])
    env.sim.reset(); env.recorder_manager.reset(); env.reset()   # exact reset sequence from record_demos.handle_reset
if aborted (state timeout):
    env.sim.reset(); env.recorder_manager.reset(); env.reset()   # recorder_manager.reset() DISCARDS the buffered episode
until env.recorder_manager.exported_successful_episode_count >= N
```

Success check call: `bool(success_term.func(env, **success_term.params)[0])`.

### 6.2 Pattern B — parallel multi-env, automatic export (faster; verified against `RecorderManager.record_pre_reset` L384-413)

Keep `terminations.success` ACTIVE and let envs auto-reset. On any internal reset, `record_pre_reset` runs automatically inside `env.step()`, reads `termination_manager.get_term("success")` for the resetting envs, calls `set_success_to_episodes`, and exports (default `RecorderManagerBaseCfg.export_in_record_pre_reset=True`); with `EXPORT_SUCCEEDED_ONLY` failures are dropped silently. Requirements: bake settling into the success term itself (e.g. add a `mdp`-style term that latches N consecutive frames, or accept 1-frame success since DONE-state holds pose — cup velocity check covers most flicker), keep `time_out` enabled as the failure path, `sm.reset_idx(( term|trunc ).nonzero())` each step. All exports append to ONE hdf5 (per-env episode counts tracked in `_exported_successful_episode_count[env_id]`). This is how you get 10 source demos in minutes with `--num_envs 32`. Caveat: obs still need `concatenate_terms=False`; num_envs>1 recording is exercised by mimic's own generation pipeline, so it is a supported path.

### 6.3 What annotate_demos.py then does (must-match contract)

- Loads hdf5, reads `env_name` (or `--task` override); `parse_env_cfg(env_name, num_envs=1)`; **requires `terminations.success` to exist** (raises otherwise), pops it, sets `terminations=None`; swaps in `MimicRecorderManagerCfg` (= `ActionStateRecorderManagerCfg` + `PreStepDatagenInfoRecorder` which records `obs/datagen_info/{object_pose (get_object_poses), eef_pose (get_robot_eef_pose), target_eef_pose (action_to_target_eef_pose(action_manager.action))}` + subtask term/start signal recorders).
- `gym.make(--task Isaac-Place-Cup-Franka-IK-Rel-Mimic-v0)` → env MUST be `ManagerBasedRLMimicEnv` subclass; `--auto` additionally requires `get_subtask_term_signals` to be overridden (checked via `__func__ is` comparison).
- Per episode: `env.reset_to(episode.data["initial_state"], None, is_relative=True)` then steps the recorded `actions` one by one → **recorded action space must equal the mimic env's action space, and the scene/randomization cfg must match recording** (replay determinism: same sim params; success is re-verified with `success_term.func` after replay — episodes that don't reproduce success are dropped).
- `--auto`: requires every `subtask_term_signal` (except the last subtask's `None`) to flip 0→1 during replay, read from `obs_buf["subtask_terms"]` → the task cfg needs an ObsGroup named exactly `subtask_terms` with terms named exactly like the `SubTaskConfig.subtask_term_signal` strings (our task: single `grasp_cup = ObsTerm(func=mdp.object_grasped, params={robot, ee_frame, object_cfg=SceneEntityCfg("cup")})`, `concatenate_terms=False`, `enable_corruption=False`).
- Output: same episodes + `obs/datagen_info/...` + `success=True`, consumable by `generate_dataset.py`.

CLI (our task):
```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-Place-Cup-Franka-IK-Rel-Mimic-v0 --auto \
  --input_file datasets/cup_place_src.hdf5 --output_file datasets/cup_place_annotated.hdf5 --headless
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --task Isaac-Place-Cup-Franka-IK-Rel-Mimic-v0 --num_envs 32 --generation_num_trials 1000 \
  --input_file datasets/cup_place_annotated.hdf5 --output_file datasets/cup_place_generated.hdf5 --headless
```
(For vision: regenerate or replay through the `-Visuomotor-` variant with `--enable_cameras`, per the stack visuomotor cfgs — out of scope here.)

### 6.4 Why not record with the plain SM script + separate hdf5 writer

The RecorderManager path is the only writer that produces the exact `EpisodeData` layout (`initial_state` incl. articulation/rigid-object state for `reset_to`, `actions`, `obs/...`, `states/...`, attrs `env_args`) that `annotate_demos.py`/`generate_dataset.py`/`replay_demos.py` consume. Manual hdf5 assembly is strictly worse — don't.

---

## 7. Uncertainties / flags

1. **Option B cross-action-space generation** (annotate IK-Abs → generate IK-Rel): code paths support it (datagen consumes only datagen_info + last-dim gripper) but no in-tree test exercises it; if chosen, validate on 2 demos first. Option A avoids it entirely.
2. **IK-Rel SM convergence constants**: delta clamp (0.05 m / 0.2 rad) and wait times under `scale=0.5` are engineering estimates; tune on first run (watch for APPROACH stalls; loosen `position_threshold` to 0.015 if needed).
3. Warp `transform_multiply` order was reasoned from the stock offsets' observed semantics (world-frame hover) — verified consistent with both scripts, but if grasp yaw composition misbehaves, flip the multiply order first.
4. Pattern B (multi-env auto-export) success latching: 1-frame success at reset time vs 10-step settle — if generated data shows premature-success episodes, add a settle-latching success term or fall back to Pattern A for source demos (only ~10 needed).
5. `parse_env_cfg(..., use_fabric=not args_cli.disable_fabric)` — keep fabric on for speed; `--disable_fabric` only for debugging USD state.
6. Stock lift SM quirks to not copy: wrong wait constant in APPROACH_ABOVE_OBJECT (uses `APPROACH_OBJECT`'s 0.6 s), `[-2]`-index dones, live (unfrozen) object pose during LIFT.
