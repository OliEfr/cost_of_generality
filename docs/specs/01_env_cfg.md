# SPEC 01 — Manager-based env config for Franka `cup_place` (tabletop pick-place with cameras)

Source of truth studied: IsaacLab v2.3.0 checkout at
`/home/admin_07/cost_of_generality/third_party/IsaacLab` (READ-ONLY — our project must be a
separate installable package, e.g. `cup_place_tasks`, that imports `isaaclab`/`isaaclab_tasks`
and registers its own gym IDs; never edit the checkout).

Primary reference task (copy structurally, not literally):
`source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/`
Closest analog to cup-place (target receptacle + randomized objects):
`stack/config/franka/bin_stack_joint_pos_env_cfg.py` (+ `bin_stack_ik_rel_env_cfg.py`)
Best success-check analog: `manipulation/place/mdp/terminations.py::object_a_is_into_b`.

---

## 0. How the stack task is structured (verified layout)

```
stack/
├── __init__.py                      # empty docstring module (needed so parent pkg imports config/)
├── stack_env_cfg.py                 # robot/embodiment-agnostic base: Scene, Obs, Actions(MISSING), Terminations
├── stack_instance_randomize_env_cfg.py
├── mdp/
│   ├── __init__.py                  # `from isaaclab.envs.mdp import *` then task observations/terminations
│   ├── observations.py              # object_obs, ee_frame_pos/quat, gripper_pos, object_grasped, object_stacked...
│   ├── terminations.py              # cubes_stacked (success)
│   └── franka_stack_events.py       # NOT star-imported; imported explicitly by configs
└── config/
    ├── __init__.py                  # empty
    └── franka/
        ├── __init__.py              # ALL gym.register(...) calls live here
        ├── stack_joint_pos_env_cfg.py           # concrete scene: robot, cubes, ee_frame, EventCfg
        ├── stack_ik_rel_env_cfg.py              # subclass: swaps arm_action -> DiffIK rel, HIGH_PD robot
        ├── stack_ik_abs_env_cfg.py              # same but use_relative_mode=False
        ├── stack_ik_rel_visuomotor_env_cfg.py   # subclass of joint_pos cfg: cameras + image obs + IK-Rel
        ├── stack_ik_rel_visuomotor_cosmos_env_cfg.py  # 200x200, seg/normals/depth obs
        ├── stack_ik_rel_blueprint_env_cfg.py    # local image() fn w/ save_image_to_file
        ├── stack_ik_rel_env_cfg_skillgen.py
        ├── stack_ik_rel_instance_randomize_env_cfg.py  # sets scene.num_envs=2 (camera cost)
        ├── bin_stack_joint_pos_env_cfg.py       # bin receptacle + cubes (pick-place-like)
        ├── bin_stack_ik_rel_env_cfg.py
        └── agents/robomimic/*.json              # robomimic hyperparam files, referenced in gym.register kwargs
```

Registered IDs (all in `stack/config/franka/__init__.py`, all with
`entry_point="isaaclab.envs:ManagerBasedRLEnv"`, `disable_env_checker=True`):

- `Isaac-Stack-Cube-Franka-v0` → `stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg`
- `Isaac-Stack-Cube-Franka-IK-Rel-v0` → `stack_ik_rel_env_cfg.FrankaCubeStackEnvCfg` (+ kwarg `robomimic_bc_cfg_entry_point=<agents>/robomimic/bc_rnn_low_dim.json`)
- `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0` → `stack_ik_rel_visuomotor_env_cfg.FrankaCubeStackVisuomotorEnvCfg` (+ `bc_rnn_image_84.json`)
- `Isaac-Stack-Cube-Franka-IK-Abs-v0`, `...-IK-Rel-Blueprint-v0`, `...-IK-Rel-Skillgen-v0`,
  `Isaac-Stack-Cube-Bin-Franka-IK-Rel-Mimic-v0`, instance-randomize variants, cosmos variant.

Mimic wrappers (separate package `source/isaaclab_mimic/isaaclab_mimic/envs/__init__.py`) register
`Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0` and `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0`
whose cfgs subclass the task cfg + `MimicEnvCfg` (covered by the Mimic spec; the env cfg here must
be the *parent* of those).

Registration pattern to replicate in our package (`cup_place/config/franka/__init__.py`):

```python
gym.register(
    id="CupPlace-Franka-IK-Rel-Visuomotor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": cup_place_ik_rel_visuomotor_env_cfg.FrankaCupPlaceVisuomotorEnvCfg},
    disable_env_checker=True,
)
```
(gym IDs resolve as long as the module executing `gym.register` is imported; isaaclab scripts
accept `--task <id>` after the package is imported — our runner scripts must
`import cup_place_tasks`  # noqa before `gym.make`. isaaclab's own tasks are auto-imported by
`isaaclab_tasks/__init__.py`'s recursive import; an external package must do its own import or use
the `ISAACLAB_TASKS_*`-style entry in the run script.)

---

## 1. Minimal file set + class structure for `cup_place`

Recommended minimal set (5 python files + agents dir):

```
cup_place_tasks/
├── __init__.py                      # walks/imports config so gym IDs register
├── cup_place_env_cfg.py             # base env cfg (scene, obs groups, terminations)
├── mdp/
│   ├── __init__.py                  # from isaaclab.envs.mdp import *; from .observations import *; from .terminations import *
│   ├── observations.py              # cup/goal obs terms + object_grasped/object_placed subtask signals
│   ├── terminations.py              # cup_placed_at_goal success fn
│   └── events.py                    # copy of randomize_object_pose (or import it — see §2)
└── config/franka/
    ├── __init__.py                  # gym.register for: -IK-Rel-v0 (state), -IK-Rel-Visuomotor-v0, L0 fixed variants
    ├── cup_place_joint_pos_env_cfg.py   # concrete Franka scene (can be merged into ik_rel file if joint-pos never used)
    ├── cup_place_ik_rel_env_cfg.py
    └── cup_place_ik_rel_visuomotor_env_cfg.py
```

You MAY import stack's mdp helpers directly instead of copying:
`from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events` and
`from isaaclab_tasks.manager_based.manipulation.stack.mdp import object_grasped` — they are generic
(`object_grasped(env, robot_cfg, ee_frame_cfg, object_cfg, diff_threshold=0.06)` works for any
rigid object). This keeps our file count minimal; copy only what we change.

### 1.1 Base cfg (`cup_place_env_cfg.py`) — exact skeleton (mirrors `stack_env_cfg.py`)

```python
from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from . import mdp

@configclass
class CupPlaceSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING           # filled by franka config subclass
    ee_frame: FrameTransformerCfg = MISSING    # filled by franka config subclass
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )
    plane = AssetBaseCfg(prim_path="/World/GroundPlane",
                         init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
                         spawn=GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light",
                         spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))
    # cup + goal marker/receptacle are added in the franka config subclass (they need rigid props)

@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions   = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cup_pos   = ObsTerm(func=mdp.object_position_in_world_frame,     # our own 3-liner, or reuse pattern
                            params={"object_cfg": SceneEntityCfg("cup")})
        cup_quat  = ObsTerm(func=mdp.object_orientation_in_world_frame,
                            params={"object_cfg": SceneEntityCfg("cup")})
        eef_pos   = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat  = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False        # REQUIRED: mixed dims + named HDF5 keys obs/<term>
    @configclass
    class SubtaskCfg(ObsGroup):
        grasp_1 = ObsTerm(func=mdp.object_grasped, params={
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("cup")})
        # final subtask needs no signal (Mimic convention: last subtask's signal is None)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cup_dropping = DoneTerm(func=mdp.root_height_below_minimum,
                            params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("cup")})
    success = DoneTerm(func=mdp.cup_placed_at_goal)   # MUST be attribute-named `success` (see §3)

@configclass
class CupPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: CupPlaceSceneCfg = CupPlaceSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None; rewards = None; events = None; curriculum = None   # stack sets unused managers to None
    def __post_init__(self):
        self.decimation = 5
        self.episode_length_s = 30.0
        self.sim.dt = 0.01              # 100 Hz physics, 20 Hz policy
        self.sim.render_interval = 2    # stack uses 2 (instance-randomize variant uses = decimation)
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
```

Key API facts (verified):
- `ManagerBasedRLEnvCfg` fields: `episode_length_s: float` (MISSING), `rewards`, `terminations`,
  `curriculum=None`, `commands=None`, `is_finite_horizon=False`. Inherited from
  `ManagerBasedEnvCfg`: `decimation: int` (MISSING), `scene`, `observations`, `actions`,
  `events: object = DefaultEmptyEventManagerCfg()` (so setting `events = None` or a custom
  `EventCfg()` both work), `sim: SimulationCfg`, `rerender_on_reset: bool=False`,
  `recorders`, `seed`, `xr`, `teleop_devices`.
  `episode_length_steps = ceil(episode_length_s / (decimation * sim.dt))` → 30/(5*0.01)=600 steps.
- `InteractiveSceneCfg` fields: `num_envs: int` (MISSING), `env_spacing: float` (MISSING),
  `replicate_physics: bool=True`, `filter_collisions=True`, `lazy_sensor_update=True`,
  `clone_in_fabric=False`. Stack sets `replicate_physics=False` (needed for texture/visual
  randomization and per-env USD edits; also needed if we ever use `randomize_visual_texture_material`).
  Scene entity attribute ORDER matters (entities created in declaration order: physics assets
  before sensors before lights).
- Extra plain attributes are allowed on the cfg and read by mdp helpers via `env.cfg.<attr>`:
  stack sets `self.gripper_joint_names = ["panda_finger_.*"]`, `self.gripper_open_val = 0.04`,
  `self.gripper_threshold = 0.005` — REQUIRED by `mdp.gripper_pos`, `mdp.object_grasped`,
  `mdp.object_stacked`, and `cubes_stacked` (they `hasattr(env.cfg, "gripper_joint_names")`).
  We must set the same three in our franka cfg.

### 1.2 Franka concrete cfg (`cup_place_joint_pos_env_cfg.py` pattern)

Mirrors `stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg.__post_init__` exactly:

```python
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG          # joint-pos variant
self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
self.scene.robot.spawn.semantic_tags = [("class", "robot")]
self.actions.arm_action = mdp.JointPositionActionCfg(
    asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True)
self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
    asset_name="robot", joint_names=["panda_finger.*"],
    open_command_expr={"panda_finger_.*": 0.04}, close_command_expr={"panda_finger_.*": 0.0})
```

Cup asset — follow the cube pattern (`RigidObjectCfg` + `UsdFileCfg` + `RigidBodyPropertiesCfg`):

```python
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
cup_properties = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16, solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0, max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0, disable_gravity=False)
self.scene.cup = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cup",
    init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.0203], rot=[1, 0, 0, 0]),
    spawn=UsdFileCfg(usd_path=<CUP_USD>, scale=(1.0, 1.0, 1.0),
                     rigid_props=cup_properties, semantic_tags=[("class", "cup")]))
```
UNCERTAIN: cup USD asset choice. Candidates on nucleus: SM_Mug_*/mug assets used by
`manipulation/place` (AgiBot mug task) and `{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/*`
(bin_stack uses `sorting_bin_blue.usd` from there, `scale=(1.1, 1.6, 3.3)`). Verify the exact cup
USD path at implementation time; a scaled cylinder via `sim_utils.CylinderCfg` +
`rigid_props`+`collision_props`+`mass_props` is the zero-download fallback.
NOTE (bin_stack): it raises `solver_position_iteration_count` to 40 for objects interacting with
the thin-walled bin. If cup-into-saucer/coaster contact chatters, do the same.

EE frame sensor (copy verbatim from stack — Franka TCP):

```python
from isaaclab.markers.config import FRAME_MARKER_CFG
marker_cfg = FRAME_MARKER_CFG.copy(); marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
marker_cfg.prim_path = "/Visuals/FrameTransformer"
self.scene.ee_frame = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/panda_link0", debug_vis=False, visualizer_cfg=marker_cfg,
    target_frames=[
        FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_hand", name="end_effector",
                                     offset=OffsetCfg(pos=[0.0, 0.0, 0.1034])),
        FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger", name="tool_rightfinger",
                                     offset=OffsetCfg(pos=(0.0, 0.0, 0.046))),
        FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger", name="tool_leftfinger",
                                     offset=OffsetCfg(pos=(0.0, 0.0, 0.046))),
    ])
```
(`OffsetCfg` from `isaaclab.sensors.frame_transformer.frame_transformer_cfg`.)

### 1.3 IK-Rel variant (`cup_place_ik_rel_env_cfg.py`)

Subclass the joint-pos cfg, override robot + arm action (verbatim stack pattern):

```python
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
    asset_name="robot", joint_names=["panda_joint.*"], body_name="panda_hand",
    controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    scale=0.5,
    body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]))
```
- Action space: 6-dim relative pose delta (scaled 0.5) + 1-dim binary gripper = 7 total.
- `FRANKA_PANDA_HIGH_PD_CFG` = FRANKA_PANDA_CFG with `spawn.rigid_props.disable_gravity=True`,
  shoulder/forearm stiffness 400, damping 80 — required for decent IK tracking.
- `DifferentialInverseKinematicsActionCfg` fields: `joint_names: list[str]`, `body_name: str`,
  `body_offset: OffsetCfg|None` (pos, rot wxyz), `scale: float|tuple=1.0`,
  `controller: DifferentialIKControllerCfg`.
- `BinaryJointPositionActionCfg` fields (from `BinaryJointActionCfg`): `joint_names`,
  `open_command_expr: dict[str, float]`, `close_command_expr: dict[str, float]`.
- IK-Abs variant differs only in `use_relative_mode=False` and drops `scale`/`body_offset`
  (see `stack_ik_abs_env_cfg.py`). GOTCHA: stack's IK-Rel uses body_offset z=0.107 while ee_frame
  TCP offset is 0.1034 — intentional in stack; keep both values as-is for Mimic compatibility.

---

## 2. Per-episode randomization via events (and fixed-pose L0)

Events manager: set `self.events = EventCfg()` in the franka cfg `__post_init__` (base sets
`events=None`). Each term is `isaaclab.managers.EventTermCfg` (alias `EventTerm`) with
`func`, `mode` ("startup" | "reset" | "interval"; stack uses "reset" for everything), `params`.

Stack's exact EventCfg (from `stack_joint_pos_env_cfg.py`) — reuse all three:

```python
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

@configclass
class EventCfg:
    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose, mode="reset",
        params={"default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400]})
    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset, mode="reset",
        params={"mean": 0.0, "std": 0.02, "asset_cfg": SceneEntityCfg("robot")})
    randomize_cup_pose = EventTerm(
        func=franka_stack_events.randomize_object_pose, mode="reset",
        params={"pose_range": {"x": (0.4, 0.6), "y": (-0.10, 0.10), "z": (0.0203, 0.0203),
                               "yaw": (-1.0, 1.0)},
                "min_separation": 0.1,
                "asset_cfgs": [SceneEntityCfg("cup")]})
    randomize_goal_pose = EventTerm(
        func=franka_stack_events.randomize_object_pose, mode="reset",
        params={"pose_range": {"x": (0.35, 0.55), "y": (0.10, 0.25), "z": (0.0203, 0.0203)},
                "min_separation": 0.15,     # only enforced among asset_cfgs within ONE event term!
                "asset_cfgs": [SceneEntityCfg("goal_marker")]})
```

`franka_stack_events.randomize_object_pose(env, env_ids, asset_cfgs: list[SceneEntityCfg],
min_separation: float = 0.0, pose_range: dict[str, tuple[float,float]] = {}, max_sample_tries=5000)`:
- `pose_range` keys: `"x","y","z","roll","pitch","yaw"`, values `(min, max)`; missing key → 0.0.
- Samples per env (python loop, CPU `random.uniform` — fine at data-gen scale, slow at 4096 envs),
  rejection-samples until all objects in the SAME call are `min_separation` apart (euclidean on xyz),
  then `write_root_pose_to_sim` (pos + env_origin, quat from euler) and zero root velocity.
- GOTCHA: separation is only enforced *within one event term's asset_cfgs list*. To keep the cup
  away from the goal, randomize BOTH in a single term:
  `"asset_cfgs": [SceneEntityCfg("cup"), SceneEntityCfg("goal_marker")]` with one shared pose_range,
  OR write a small custom event (copy of randomize_object_pose taking per-asset ranges) — needed if
  cup and goal have different x/y ranges. RECOMMENDED: custom
  `cup_place/mdp/events.py::randomize_cup_and_goal(env, env_ids, cup_cfg, goal_cfg, cup_pose_range,
  goal_pose_range, min_separation)` cloned from `randomize_object_pose` (~30 lines).
- GOTCHA (do not copy blindly): stack's actual param is `"yaw": (-1.0, 1, 0)` — a buggy 3-tuple;
  `random.uniform(range[0], range[1])` ignores the third element so it silently works. Use 2-tuples.
- Alternative built-in: `isaaclab.envs.mdp.events.reset_root_state_uniform(env, env_ids, pose_range,
  velocity_range, asset_cfg)` — vectorized, offsets from `default_root_state` (i.e., ranges are
  DELTAS from init_state, not absolute), no min-separation. Good for single-object randomization.
- Robot start pose: `set_default_joint_pose` overwrites `data.default_joint_pos` (9 values: 7 arm +
  2 fingers), then `randomize_joint_by_gaussian_offset` adds N(0, 0.02) to arm joints only
  (last 2 gripper joints restored) and writes joint state to sim.

**Fixed-pose L0 variant**: degenerate ranges, exactly like `bin_stack_joint_pos_env_cfg.py` does
for its bin: `"pose_range": {"x": (0.4, 0.4), "y": (0.0, 0.0), "z": (0.0203, 0.0203), "yaw": (0.0, 0.0)}`.
Implement L0 as a subclass overriding only `events.randomize_cup_pose.params["pose_range"]` (and the
goal's) in `__post_init__`, registered under its own gym ID. Keep the event term (don't delete it):
it still resets the pose deterministically each episode, which is required after physics settles/objects
were moved in a previous episode.

---

## 3. Success termination (object within tolerance of target position)

`TerminationTermCfg` (alias `DoneTerm`): `func` returning bool tensor `(num_envs,)`,
`time_out: bool = False`, `params: dict`. Built-in helpers in `isaaclab.envs.mdp.terminations`
(star-imported into task `mdp`): `time_out`, `root_height_below_minimum(minimum_height, asset_cfg)`,
`bad_orientation`, `joint_*_out_of_limit`, `illegal_contact`. There is NO generic
"object near target" in base mdp — every task ships its own:

- stack: `mdp.cubes_stacked(env, robot_cfg, cube_1_cfg, cube_2_cfg, cube_3_cfg, xy_threshold=0.04,
  height_threshold=0.005, height_diff=0.0468, atol=0.0001, rtol=0.0001)` — xy dist + height diff
  checks AND gripper-open check via `env.cfg.gripper_open_val` (`torch.isclose` on both finger joints).
- place (BEST TEMPLATE for cup_place): `manipulation/place/mdp/terminations.py::object_a_is_into_b(
  env, robot_cfg, object_a_cfg, object_b_cfg, xy_threshold=0.03, height_threshold=0.04, height_diff=0.0)`
  — `xy_dist < xy_threshold and (height_dist - height_diff) < height_threshold` between the two
  assets' `root_pos_w`, AND gripper released (`abs(|joint_pos| - gripper_open_val) < gripper_threshold`).
  Also `object_placed_upright(..., target_height, euler_xy_threshold=0.10)` adds an uprightness
  check via `math_utils.euler_xyz_from_quat` — include for a cup that must land upright.
- lift (goal-as-command route): `manipulation/lift/mdp/terminations.py::object_reached_goal(env,
  command_name="object_pose", threshold=0.02, robot_cfg, object_cfg)` — reads
  `env.command_manager.get_command(command_name)[:, :3]` (goal in robot base frame), transforms to
  world via `combine_frame_transforms(robot.root_pos_w, robot.root_quat_w, des_pos_b)`, checks
  `norm < threshold`.

Our `cup_place/mdp/terminations.py::cup_placed_at_goal` = clone of `object_a_is_into_b` with
`object_a_cfg=SceneEntityCfg("cup")`, `object_b_cfg=SceneEntityCfg("goal_marker")`, thresholds
`xy_threshold≈0.05`, `height_threshold≈0.04`, `height_diff=<cup half-height − marker half-height>`,
plus (optional) upright check. The gripper-open condition is IMPORTANT: without it, success fires
while the cup is still grasped in transit above the goal.

**CRITICAL wiring fact (verified in scripts)**: `scripts/tools/record_demos.py` and
`scripts/imitation_learning/isaaclab_mimic/annotate_demos.py` both do
`if hasattr(env_cfg.terminations, "success"): success_term = env_cfg.terminations.success;
env_cfg.terminations.success = None` — i.e. the term must be the attribute literally named
`success` on the terminations cfg; the scripts pop it out (so success doesn't reset the env during
recording) and evaluate `success_term.func(env, **success_term.params)` manually
(`--num_success_steps`, default 10 consecutive steps, marks a demo successful).

---

## 4. Visuomotor (camera) variant — exact wiring

From `stack_ik_rel_visuomotor_env_cfg.py` (class `FrankaCubeStackVisuomotorEnvCfg`, subclass of the
*joint-pos* cfg that re-applies the IK-Rel action + HIGH_PD robot itself, and REPLACES the whole
`observations` with a new `ObservationsCfg` in which images are terms of the `policy` group):

```python
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils

# in ObservationsCfg.PolicyCfg (alongside the state terms):
table_cam = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("table_cam"),
                                            "data_type": "rgb", "normalize": False})
wrist_cam = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam"),
                                            "data_type": "rgb", "normalize": False})

# in __post_init__:
self.scene.wrist_cam = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",   # rigidly parented to the hand link
    update_period=0.0, height=84, width=84,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0,
                                     horizontal_aperture=20.955, clipping_range=(0.1, 2)),
    offset=CameraCfg.OffsetCfg(pos=(0.13, 0.0, -0.15),
                               rot=(-0.70614, 0.03701, 0.03701, -0.70614), convention="ros"))
self.scene.table_cam = CameraCfg(
    prim_path="{ENV_REGEX_NS}/table_cam",                    # per-env static camera (env-ns root)
    update_period=0.0, height=84, width=84,
    data_types=["rgb", "distance_to_image_plane"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0,
                                     horizontal_aperture=20.955, clipping_range=(0.1, 2)),
    offset=CameraCfg.OffsetCfg(pos=(1.0, 0.0, 0.4),
                               rot=(0.35355, -0.61237, -0.61237, 0.35355), convention="ros"))
self.rerender_on_reset = True                    # else first-frame images after reset are stale
self.sim.render.antialiasing_mode = "OFF"        # disable DLSS (blurs/ghosts tiny images)
self.image_obs_list = ["table_cam", "wrist_cam"] # plain attr; consumed by robomimic train/Mimic
                                                 # tooling to know which obs keys are images
```

For our 128x128: identical blocks with `height=128, width=128`. Camera facts (verified in
`sensors/camera/camera_cfg.py`):
- `CameraCfg` fields: `width`, `height` (MISSING/int), `data_types: list[str] = ["rgb"]`
  (options include "rgb", "rgba", "distance_to_image_plane", "distance_to_camera", "normals",
  "semantic_segmentation", "instance_segmentation_fast", ...), `spawn: PinholeCameraCfg|FisheyeCameraCfg|None`
  (None = camera prim already exists in USD), `offset: OffsetCfg(pos, rot wxyz, convention
  "ros"|"opengl"|"world", default "ros")`, `update_period` (0.0 = every sim render step;
  inherited from `SensorBaseCfg`), `depth_clipping_behavior`, `semantic_filter`,
  `colorize_semantic_segmentation`, `semantic_segmentation_mapping: dict` (see cosmos variant for a
  concrete `{"class:cube_1": (120,230,255,255), ...}` example).
- `TiledCameraCfg(CameraCfg)` adds nothing but `class_type=TiledCamera` — single tiled render
  product for all envs; use it if we ever need many envs with cameras. Stack uses plain `CameraCfg`
  (one render product per env) — fine because camera workflows run with ~1-10 envs.
- `mdp.image(env, sensor_cfg, data_type="rgb", convert_perspective_to_orthogonal=False,
  normalize=True)` (in `isaaclab/envs/mdp/observations.py`): returns
  `sensor.data.output[data_type]` cloned; shape `(num_envs, H, W, 3)` uint8 for "rgb" when
  `normalize=False` (stack uses False → raw uint8 into HDF5 — what robomimic and LeRobot want).
  `normalize=True` would divide by 255 and subtract per-image mean (do NOT use for dataset capture).
- Semantic tags on assets (`spawn.semantic_tags = [("class", "cup")]`) only matter if we record
  segmentation; harmless otherwise.

num_envs / rendering flags:
- CLI MUST include `--enable_cameras` (AppLauncher arg, `source/isaaclab/isaaclab/app/app_launcher.py`)
  for any camera env, else sensor creation fails. Docs
  (`docs/source/overview/imitation-learning/teleop_imitation.rst`) run the visuomotor stack env with
  `--device cpu --enable_cameras --num_envs 10` for Mimic datagen and `--headless` for the big run.
- The cfg itself keeps `num_envs=4096` default and scripts override with `--num_envs`; the
  instance-randomize variant hard-sets `self.scene.num_envs = 2` "due to camera resources" — for our
  visuomotor cfg, hard-set a safe default (e.g. `self.scene.num_envs = 10`) so nobody launches 4096
  cameras by accident (also respects the machine-sharing guardrail: check `nvidia-smi` headroom,
  cap `--num_envs`).
- `sim.render_interval=2` with `decimation=5`: rendering runs every 2 physics steps, policy every
  5 — images are at most 1 physics step stale at policy sampling. Instance-randomize uses
  `render_interval = decimation` (exactly one render per policy step, cheaper). Either works;
  UNCERTAIN which is better for LeRobot temporal consistency — recommend `render_interval = 2`
  to match the reference visuomotor task exactly.
- Registered camera task IDs get their own gym ID (`...-IK-Rel-Visuomotor-v0`); keep the state-only
  ID separate so state pipelines never pay render cost.

---

## 5. Goal exposed to policy ONLY visually, but available to env for termination

How stack-family handles the analogous need:
- bin_stack: the target is a REAL asset in the scene (`blue_sorting_bin` RigidObjectCfg). It is
  visible to cameras automatically; the env reads `env.scene["blue_sorting_bin"].data.root_pos_w`
  in terminations; the *state obs group simply has no term for it* → not exposed to a state policy.
  Exposure is controlled purely by which ObsTerms exist in `PolicyCfg`.
- lift: goal is a `UniformPoseCommandCfg` command (`commands.object_pose`,
  `mdp.generated_commands(command_name="object_pose")` obs + `object_reached_goal` termination).
  Its visualization is a debug marker (`goal_pose_visualizer_cfg`) under `/Visuals` — debug markers
  are NOT reliable/per-env-correct visual features for a camera policy and are meant for humans.
  Do NOT use the command route for a visually-grounded goal.

RECOMMENDED for cup_place (mirrors bin_stack):
1. `self.scene.goal_marker = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/GoalMarker", init_state=...,
   spawn=UsdFileCfg(usd_path=<saucer/coaster/pad USD>, semantic_tags=[("class", "goal")],
   rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True)))` — kinematic rigid body:
   - visible in both cameras (it is real scene geometry, per-env, cloned under `{ENV_REGEX_NS}`);
   - movable per episode by `randomize_object_pose` (which calls `write_root_pose_to_sim` — works
     because RigidObject; a plain `AssetBaseCfg` visual has NO `write_root_pose_to_sim` and cannot
     be repositioned by the stock event, that's why it must be a RigidObjectCfg);
   - `kinematic_enabled=True` (field on `RigidBodyPropertiesCfg`, `sim/schemas/schemas_cfg.py`)
     pins it against physics so the cup resting on it can't nudge it;
   - if it must be non-colliding (a flat decal-like target), add
     `collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False)` on the spawn cfg
     (for UsdFileCfg this disables the colliders baked in the USD). If the cup should physically
     rest ON it, keep collisions on. UNCERTAIN: for primitive spawners (CylinderCfg etc.) omit
     `collision_props` entirely to get no collider — verify at implementation.
2. Termination reads it as a scene entity: `cup_placed_at_goal(..., goal_cfg=SceneEntityCfg("goal_marker"))`
   comparing `cup.data.root_pos_w` vs `goal.data.root_pos_w` (see §3).
3. Policy visibility: DO NOT add any goal_pos ObsTerm to `PolicyCfg` — vision-only. (Optionally add
   `goal_pos` to a separate non-policy group, e.g. an `eval_info` ObsGroup or reuse the
   `subtask_terms` pattern, if the dataset should log ground-truth goal for analysis; groups other
   than `policy` are still recorded by the recorder manager but not fed to the policy.)
4. L0 fixed-goal: degenerate pose_range in the goal event term (§2).

---

## 6. Timing constants used by stack (adopt unchanged)

| Setting | Value | Where |
|---|---|---|
| `sim.dt` | `0.01` (100 Hz) | `StackEnvCfg.__post_init__` |
| `decimation` | `5` (20 Hz policy) | same |
| `episode_length_s` | `30.0` (=600 policy steps) | same |
| `sim.render_interval` | `2` (visuomotor path) / `= decimation` (instance-randomize) | same / `StackInstanceRandomizeEnvCfg` |
| physx | `bounce_threshold_velocity=0.01` (set twice, 0.2 then 0.01 — final 0.01), `gpu_found_lost_aggregate_pairs_capacity=4*1024*1024`, `gpu_total_aggregate_pairs_capacity=16*1024`, `friction_correlation_distance=0.00625` | `StackEnvCfg.__post_init__` |
| scene | `num_envs=4096` (overridden at launch), `env_spacing=2.5`, `replicate_physics=False` | `StackEnvCfg.scene` |

---

## 7. Gotchas / open items (consolidated)

1. `success` MUST be the literal attribute name of the success DoneTerm (record/annotate scripts
   hasattr-pop it). Subtask ObsGroup MUST be named `subtask_terms`; its term names ("grasp_1", ...)
   are the `subtask_term_signal` names referenced by Mimic subtask configs (next spec).
2. All ObsGroups: `concatenate_terms = False` (mixed shapes; named keys in recorded HDF5).
3. Set `gripper_joint_names=["panda_finger_.*"]`, `gripper_open_val=0.04`, `gripper_threshold=0.005`
   as plain attrs on the env cfg or `object_grasped`/success helpers raise.
4. `randomize_object_pose` min_separation only applies within one event term; merge cup+goal into
   one term or write a custom event. Its per-env python loop is slow at high num_envs (irrelevant
   for ≤16-env data generation).
5. Camera runs need `--enable_cameras`; keep visuomotor `num_envs` small (stack docs: 10, CPU);
   `rerender_on_reset=True`; `antialiasing_mode="OFF"`; `normalize=False` on rgb ObsTerms (uint8).
6. `replicate_physics=False` required if we use texture/light randomization events
   (`randomize_visual_texture_material` raises otherwise); stack sets it False unconditionally.
7. Env-cfg register pattern requires our package to be imported before `gym.make`; plan an
   `import cup_place_tasks` in every runner script (or install with an entry point).
8. Do not copy the `"yaw": (-1.0, 1, 0)` 3-tuple typo from stack's EventCfg.
9. Visual-only goal must be a kinematic `RigidObjectCfg` (not `AssetBaseCfg`) so the stock reset
   event can move it; disable collisions only if the cup shouldn't rest on it.
10. UNCERTAIN items to verify at implementation: exact cup USD path; goal marker USD (fallback:
    primitive spawner); collision behavior of primitive spawners without `collision_props`;
    `render_interval` choice (2 vs decimation) for cleanest LeRobot frames.
11. Machine guardrails: never modify the IsaacLab checkout or the `isaaclab` conda env; check
    `nvidia-smi` before GPU launches; cap `--num_envs`.
