# SPEC 05 — Assets for the Franka cup-place task

Studied against IsaacLab v2.3.0 checkout at
`/home/admin_07/cost_of_generality/third_party/IsaacLab` (read-only). All file paths below are exact
paths in that tree; all class/field names verified in source.

---

## 1. Asset inventory: what exists, what the reference tasks use

### 1.1 Nucleus path variables (`isaaclab.utils.assets`)

File: `source/isaaclab/isaaclab/utils/assets.py`

```python
NUCLEUS_ASSET_ROOT_DIR = carb.settings.get_settings().get("/persistent/isaac/asset_root/cloud")
NVIDIA_NUCLEUS_DIR   = f"{NUCLEUS_ASSET_ROOT_DIR}/NVIDIA"
ISAAC_NUCLEUS_DIR    = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac"
ISAACLAB_NUCLEUS_DIR = f"{ISAAC_NUCLEUS_DIR}/IsaacLab"
```

`NUCLEUS_ASSET_ROOT_DIR` is a **carb setting resolved at import time**; for Isaac Sim 5.1 pip
installs it resolves to the NVIDIA AWS S3/CloudFront bucket for the 5.1 asset release (verify
exact URL at runtime with `python -c "from isaaclab.utils.assets import NUCLEUS_ASSET_ROOT_DIR; print(NUCLEUS_ASSET_ROOT_DIR)"`
inside the app — it needs carb, so print it from any running script). **UNCERTAINTY:** exact URL
string not in-tree; do not hardcode it, always go through the variables.

### 1.2 USD props used by in-tree manipulation tasks (grep of `isaaclab_tasks`, counts = #usages)

| Path | Var prefix | Used by |
|---|---|---|
| `Props/Mounts/SeattleLabTable/table_instanceable.usd` | `ISAAC_NUCLEUS_DIR` | stack, lift, place (table) |
| `Props/Blocks/DexCube/dex_cube_instanceable.usd` | `ISAAC_NUCLEUS_DIR` | lift, cabinet variants |
| `Props/Blocks/{red,green,blue,yellow}_block.usd` | `ISAAC_NUCLEUS_DIR` | stack (cubes, ~4.4 cm) |
| `Props/PackingTable/packing_table.usd` | `ISAAC_NUCLEUS_DIR` | GR1 Mimic pick-place |
| `Props/UIElements/frame_prim.usd` | `ISAAC_NUCLEUS_DIR` | FRAME_MARKER_CFG |
| **`Objects/Mug/mug.usd`** | **`ISAACLAB_NUCLEUS_DIR`** | **place task (Agibot "place upright mug")** |
| `Objects/ToyTruck/toy_truck.usd` | `ISAACLAB_NUCLEUS_DIR` | place toy2box |
| `Objects/Box/box.usd` | `ISAACLAB_NUCLEUS_DIR` | place toy2box (goal container) |
| `Objects/Teddy_Bear/teddy_bear.usd` | `ISAACLAB_NUCLEUS_DIR` | lift ik_abs (deformable-ish visual) |
| `Mimic/nut_pour_task/...`, `Mimic/exhaust_pipe_task/...` | `ISAACLAB_NUCLEUS_DIR` | Mimic G1/GR1 tasks (bowls, beakers, bins) |

### 1.3 Recommended mug asset (answer to Q1)

**Primary: `f"{ISAACLAB_NUCLEUS_DIR}/Objects/Mug/mug.usd"`** — the only actual mug in the Isaac
5.1 / IsaacLab asset library referenced anywhere in the v2.3.0 tree. Reference usage:
`source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/config/agibot/place_upright_mug_rmp_rel_env_cfg.py`
(lines 226-234), spawned at `scale=(1.0, 1.0, 1.0)` with a `RigidBodyPropertiesCfg` identical to
the stack cubes' one. That task spawns it at z=0.75 on a table with `roll=-1.57` (lying on its
side), i.e. it is a normal-size mug asset with physics (rigid body + colliders) already authored.

**There is NO YCB asset anywhere in the IsaacLab tree** (no `025_mug`, no YCB folder referenced in
code or docs). The Isaac Sim asset library historically ships YCB under
`{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/025_mug.usd` (and `.../Axis_Aligned_Physics/`), but
**this is not verified in this tree — treat as fallback candidate #2 and stat it at runtime**
with `isaaclab.utils.assets.check_file_path(path)` (returns 0/1/2) before committing to it.

**Graspability constraint (Franka panda, max opening 2×0.04 m = 8 cm):** a real mug body is
~8–9 cm diameter → a full-scale mug body is NOT reliably graspable across the body.
**UNCERTAINTY: the authored dimensions of `Objects/Mug/mug.usd` are not inspectable offline**
(cloud USD). Mitigations, in order:
1. After first download (Section 4), inspect the bbox:
   ```python
   from pxr import Usd, UsdGeom
   stage = Usd.Stage.Open("/path/to/cache/Objects/Mug/mug.usd")
   bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(stage.GetDefaultPrim())
   print(bbox.ComputeAlignedRange().GetSize())
   ```
2. Set `UsdFileCfg.scale` so body outer diameter ≤ ~6.5 cm (leaves ≥1.5 cm finger clearance);
   expected ballpark `scale=(0.7, 0.7, 0.7)` if the asset is ~9 cm — **tune after step 1**.
   `scale` is a plain field on `FileCfg` (`source/isaaclab/isaaclab/sim/spawners/from_files/from_files_cfg.py`,
   line 36: `scale: tuple[float, float, float] | None = None`). Non-uniform scale works but skews
   the handle; keep uniform.
3. Grasp the **rim** (top edge, ~2× wall thickness) instead of the body — this is what makes
   full-scale mugs graspable by pandas in practice; make the demo/Mimic subtask grasp the rim and
   scale becomes uncritical. Recommended: rim grasp + mild scale 0.85 as belt-and-suspenders.

**Fallback #3 (fully offline-safe, zero cloud dependency): procedural "mug" from shapes** — a
cylinder body is sufficient for cup-place (the handle is cosmetic for a place task):
`sim_utils.CylinderCfg(radius=0.03, height=0.08, axis="Z")` with collision + rigid props (see
Section 2 snippet C). If a handle matters visually, author a tiny local USD once (cylinder +
torus section) in Isaac Sim GUI and ship it in the project repo — do NOT depend on Nucleus for it.

---

## 2. Spawner configs (answer to Q2)

All snippets assume the stack-task import block
(`source/isaaclab_tasks/.../stack/config/franka/stack_joint_pos_env_cfg.py` lines 6–24):

```python
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
```

Shared rigid-body properties (copied verbatim from the stack cubes / place mug — both use the
same values):

```python
cup_properties = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)
```

### 2A. Fixed red mug (levels L0–L2)

`UsdFileCfg` supports `visual_material` + `visual_material_path` (in `FileCfg`,
`from_files_cfg.py` lines 58–65): when `visual_material` is set, the spawner **creates the
material at `{prim_path}/{visual_material_path}` and binds it over the asset's authored
materials** — this is the supported way to recolor a USD prop without editing the USD.

```python
self.scene.cup = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cup",
    init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.055], rot=[1, 0, 0, 0]),
    spawn=UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Mug/mug.usd",
        scale=(0.85, 0.85, 0.85),                      # tune after bbox check, see Sec. 1.3
        rigid_props=cup_properties,
        semantic_tags=[("class", "cup")],
        visual_material_path="material",               # default; child prim name for the override
        visual_material=sim_utils.PreviewSurfaceCfg(   # solid red override
            diffuse_color=(0.8, 0.05, 0.05), roughness=0.5, metallic=0.0,
        ),
    ),
)
```

`PreviewSurfaceCfg` fields (verified in
`source/isaaclab/isaaclab/sim/spawners/materials/visual_materials_cfg.py` lines 23–44):
`diffuse_color`, `emissive_color`, `roughness`, `metallic`, `opacity`.

**Gotcha:** the mug USD's meshes may have their own bound materials at stronger binding strength;
the IsaacLab spawner binds with default strength on the prim root. If the red does not show in
RGB renders, use approach 2B-i (5 pre-colored collection entries via `visual_material` on
*shape*-based cups) or the Replicator color event (2B-ii) which force-rebinds. **Verify visually
once.** (The Blocks cubes have color baked in; the place task never recolors the mug in-tree, so
this exact combination is untested upstream.)

### 2B. Per-episode color variation, 5 colors (L3)

Two mechanisms exist in-tree; pick ONE:

**(i) RECOMMENDED — instance-randomize pattern (pre-spawn 5 colored copies, teleport one into
focus per reset).** Exactly follows
`source/isaaclab_tasks/.../stack/config/franka/stack_joint_pos_instance_randomize_env_cfg.py`
(lines 105–171) + `source/isaaclab_tasks/.../stack/mdp/franka_stack_events.py::randomize_rigid_objects_in_focus`
(lines 197–242). Works with recorded datasets, deterministic replay, and Mimic, because the
choice is ordinary state (a pose), not a USD edit.

```python
COLORS = {
    "red":    (0.8, 0.05, 0.05),
    "green":  (0.05, 0.6, 0.05),
    "blue":   (0.05, 0.1, 0.8),
    "yellow": (0.85, 0.8, 0.05),
    "purple": (0.5, 0.05, 0.6),
}
cup_cfg_dict = {
    f"{name}_cup": RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cup_" + name.capitalize(),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[10.0, 10.0, 10.0]),  # parked out of view
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Mug/mug.usd",
            scale=(0.85, 0.85, 0.85),
            rigid_props=cup_properties,
            semantic_tags=[("class", "cup")],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
        ),
    )
    for name, rgb in COLORS.items()
}
self.scene.cup = RigidObjectCollectionCfg(rigid_objects=cup_cfg_dict)
```

Reset event (exact signature from `franka_stack_events.py` line 197):

```python
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
randomize_cup_in_focus = EventTerm(
    func=franka_stack_events.randomize_rigid_objects_in_focus,
    mode="reset",
    params={
        "asset_cfgs": [SceneEntityCfg("cup")],
        # 13-dim per-object state: pos(3) quat(4) linvel(3) angvel(3)
        "out_focus_state": torch.tensor([10.0, 10.0, 10.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "pose_range": {"x": (0.4, 0.6), "y": (-0.10, 0.10), "z": (0.055, 0.055), "yaw": (-1.0, 1.0)},
        "min_separation": 0.1,
    },
)
```

The event stores which variant is active in `env.rigid_objects_in_focus`
(list per env of selected object indices) — use it for the object-pose observation exactly like
`stack/mdp/observations.py::instance_randomize_object_obs`. Scene cfg **must** set
`replicate_physics=False` (the instance-randomize base env does:
`stack_instance_randomize_env_cfg.py` line 109:
`ObjectTableSceneCfg(num_envs=..., env_spacing=2.5, replicate_physics=False)`).
Note the out-of-focus copies sit at (10,10,10) **relative env offsets are not applied to the park
position in that function for out-of-focus objects** — they use the raw `out_focus_state`; with
`env_spacing=2.5` a global (10,10,10) may land inside another env's frustum. In-tree code accepts
this (cameras are close-up); if the parked cups show up in your table cam, raise parking z to 100.

**(ii) Replicator color event** — `isaaclab.envs.mdp.events.randomize_visual_color`
(`source/isaaclab/isaaclab/envs/mdp/events.py` line 1545, a `ManagerTermBase` class). Params:
`asset_cfg`, `mesh_name` (prim-path suffix pattern under the asset, e.g. `"geometry/mug"` —
depends on the USD's internal hierarchy), `colors` (list of RGB tuples OR dict
`{"r": (lo,hi), "g": ..., "b": ...}`), `event_name`. Requires `replicate_physics=False`
(hard `RuntimeError` otherwise, line 1589) and rebinds an `OmniPBR.mdl` material per prim.
Cheaper on memory than 5 copies but: mesh_name must match the mug USD's internal prim layout
(**unknown offline — inspect after download**), and color becomes a USD-side effect that is NOT
captured in recorded states (replay/Mimic regeneration will re-randomize differently unless
seeded). Prefer (i) for a Mimic pipeline.

### 2C. Multi-mesh + scale variation (L3), stack_instance_randomize pattern

Same `RigidObjectCollectionCfg` mechanism — entries just differ in `usd_path`/`scale`/spawner
type (they only need "similar prim hierarchy" if you use `MultiUsdFileCfg`; a collection has no
such constraint since each entry is its own asset):

```python
cup_variants = {
    "mug_small":  RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cup_MugSmall",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[10.0, 10.0, 10.0]),
        spawn=UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Mug/mug.usd",
                         scale=(0.7, 0.7, 0.7), rigid_props=cup_properties,
                         semantic_tags=[("class", "cup")]),
    ),
    "mug_large":  RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cup_MugLarge",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[10.0, 10.0, 10.0]),
        spawn=UsdFileCfg(usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Mug/mug.usd",
                         scale=(0.95, 0.95, 0.95), rigid_props=cup_properties,
                         semantic_tags=[("class", "cup")]),
    ),
    "cylinder_cup": RigidObjectCfg(   # procedural, offline-safe shape variant
        prim_path="{ENV_REGEX_NS}/Cup_Cyl",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[10.0, 10.0, 10.0]),
        spawn=sim_utils.CylinderCfg(
            radius=0.032, height=0.09, axis="Z",
            rigid_props=cup_properties,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.05, 0.05)),
            semantic_tags=[("class", "cup")],
        ),
    ),
}
self.scene.cup = RigidObjectCollectionCfg(rigid_objects=cup_variants)
```

Shape spawner fields verified in `source/isaaclab/isaaclab/sim/spawners/shapes/shapes_cfg.py`:
`CylinderCfg(radius, height, axis)`, `CuboidCfg(size)`, `SphereCfg(radius)`; all inherit
`ShapeCfg` → `visual_material`, `physics_material` and `RigidObjectSpawnerCfg` →
`mass_props`, `rigid_props`, `collision_props`, `semantic_tags`, `visible`.

**Alternative NOT recommended for per-episode variation:** `MultiUsdFileCfg` /
`MultiAssetSpawnerCfg` (`source/isaaclab/isaaclab/sim/spawners/wrappers/wrappers_cfg.py`).
Fields: `assets_cfg: list[SpawnerCfg]`, `usd_path: str | list[str]`, `random_choice: bool = True`.
It picks **one variant per env at spawn time (stage construction), fixed for the whole run**
(`wrappers.py::spawn_multi_asset` line 114: `random.choice(proto_prim_paths)` inside the clone
loop) — good for cross-env diversity in RL, wrong for per-episode variation in a 1–few-env
data-generation setting. It also flips the carb flag `/isaaclab/spawn/multi_assets` and requires
`replicate_physics=False`. Demo: `scripts/demos/multi_asset.py` (mixes `ConeCfg`/`CuboidCfg`/
`SphereCfg` with `PreviewSurfaceCfg` colors in one `MultiAssetSpawnerCfg`).

### Cross-cutting L3 requirement

Any of: `RigidObjectCollectionCfg` variants, `MultiAssetSpawnerCfg`, `randomize_visual_color`,
`randomize_visual_texture_material` ⇒ set `replicate_physics=False` on the `InteractiveSceneCfg`.
This slows scene parsing but is what the shipped instance-randomize stack env does.

---

## 3. Goal marker: flat colored disk, camera-visible, non-colliding (answer to Q3)

### 3.1 Recommended: kinematic, collision-free `RigidObject` cylinder ("disk")

A `RigidObject` needs the `RigidBodyAPI` (supplied by `rigid_props`) but does **not** need
colliders — `collision_props: CollisionPropertiesCfg | None = None` on `RigidObjectSpawnerCfg`
(`source/isaaclab/isaaclab/sim/spawners/spawner_cfg.py` line 89); leaving it `None` spawns pure
visual geometry. `kinematic_enabled=True` (`schemas_cfg.py::RigidBodyPropertiesCfg`, line 71)
makes it ignore gravity/dynamics while remaining poseable via `write_root_pose_to_sim` — i.e. an
event-repositionable prop that renders in every RGB camera and never collides:

```python
self.scene.goal_marker = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/GoalMarker",
    init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.2, 0.001]),  # 1 mm above table top
    spawn=sim_utils.CylinderCfg(
        radius=0.06,
        height=0.002,                      # flat disk
        axis="Z",
        rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        collision_props=None,              # NO collider — objects/gripper pass through freely
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.1, 0.7, 0.1), roughness=0.9, metallic=0.0,
        ),
        semantic_tags=[("class", "goal")],
    ),
)
```

Per-episode positioning reuses the shipped event **unchanged**
(`franka_stack_events.randomize_object_pose` works on any scene entity exposing
`write_root_pose_to_sim`; it also calls `write_root_velocity_to_sim` with zeros — fine for a
kinematic body):

```python
randomize_goal_marker = EventTerm(
    func=franka_stack_events.randomize_object_pose,
    mode="reset",
    params={
        "pose_range": {"x": (0.4, 0.6), "y": (0.1, 0.25), "z": (0.001, 0.001)},
        "min_separation": 0.15,           # only enforced among asset_cfgs listed together
        "asset_cfgs": [SceneEntityCfg("goal_marker")],
    },
)
```

To enforce cup-vs-goal separation, put both in ONE event
(`"asset_cfgs": [SceneEntityCfg("cup"), SceneEntityCfg("goal_marker")]` with a shared
`pose_range`) — `min_separation` is only checked within a single call
(`franka_stack_events.py::sample_object_poses`, lines 133–157).

The marker's pose then also drops out as a normal observation
(`mdp.root_pos_w`-style terms or the stack `object_obs` pattern) and is recorded into HDF5 like
any rigid object — required for Mimic to re-randomize goals consistently.

**Gotcha (vision):** with `rerender_on_reset = True` (set in
`stack_ik_rel_visuomotor_env_cfg.py` line 234) the first camera frame after reset already shows
the moved marker; without it the first frame shows the stale position.

### 3.2 Alternative: `VisualizationMarkers` — usable but second choice

`source/isaaclab/isaaclab/markers/visualization_markers.py` —
`VisualizationMarkersCfg(prim_path="/Visuals/...", markers={"goal": sim_utils.CylinderCfg(...)})`,
then `marker.visualize(translations=..., marker_indices=...)` each reset. They are real USD
geometry (a `UsdGeom.PointInstancer`), so they DO render in RTX camera images (this is exactly
why the visuomotor stack env sets `debug_vis=False` on its `FrameTransformerCfg` to keep frame
axes out of the wrist/table cams). Drawbacks: not a scene entity (no event-manager integration,
no auto pos observation/recording — you hand-roll a `visualize()` call and bookkeeping per reset,
and it lives at a global `/Visuals` path, not under `{ENV_REGEX_NS}`). Prototype presets exist in
`source/isaaclab/isaaclab/markers/config/__init__.py`: `CUBOID_MARKER_CFG`, `SPHERE_MARKER_CFG`,
`POSITION_GOAL_MARKER_CFG`, `FRAME_MARKER_CFG` (the latter spawns
`{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd` — cloud dependency).

Do NOT use `AssetBaseCfg` for the marker: it is a static XForm with no
`write_root_pose_to_sim`, so reset events cannot move it.

---

## 4. Offline / cluster asset strategy (answer to Q4)

Facts from the tree:

- Every `{*_NUCLEUS_DIR}` URL is fetched over the network by `omni.client` (backed by the AWS
  S3/CloudFront bucket since Isaac Sim 4.5 — Nucleus server itself is deprecated;
  `docs/source/setup/installation/asset_caching.rst`). CINECA Leonardo compute nodes have **no
  internet** (see CLAUDE.md rule 4) ⇒ any config that still contains a cloud URL will hang and
  then fail at spawn time (`check_usd_path_with_timeout` in `utils/assets.py` warns after 30 s,
  times out at 300 s).
- `isaaclab.utils.assets.retrieve_file_path(path, download_dir=..., force_download=...)`
  downloads a single file via `omni.client.copy` — **it does NOT pull referenced sub-layers,
  payloads, or textures**, so it is insufficient for `mug.usd`-style assets with sibling texture
  files.
- The documented cache mechanism is the OV **Hub** cache
  (`docs/source/setup/installation/asset_caching.rst`) — a login-node-side daemon; not suitable
  for air-gapped compute nodes.

**Recommended procedure (do once on a machine with internet, e.g. login node or workstation):**

1. Download the official **Isaac Sim 5.1 Assets packs** (three zips, from the Isaac Sim
   "Download Assets" docs page) OR mirror just the needed subfolders from the S3 bucket with
   plain HTTP (each file is individually addressable under the printed
   `NUCLEUS_ASSET_ROOT_DIR`; folders needed for this project:
   `Isaac/IsaacLab/Objects/Mug/`, `Isaac/Props/Mounts/SeattleLabTable/`,
   `Isaac/Props/Blocks/` (only if cubes kept for debugging), `Isaac/Props/UIElements/frame_prim.usd`,
   plus the Franka robot USD `Isaac/Robots/FrankaEmika/panda_instanceable.usd` + its `Props/`
   sibling folder of textures/materials — mirror the whole containing directory, not single
   files, to capture relative references).
   Place under e.g. `$WORK/cog/isaac_assets/Assets/Isaac/5.1/` so that
   `<root>/Isaac/...` and `<root>/Isaac/IsaacLab/...` exist.
2. Point Isaac Sim at the local root via carb settings on every launch (works with any
   IsaacLab script because `AppLauncher` forwards `--kit_args`,
   `source/isaaclab/isaaclab/app/app_launcher.py` line 330):
   ```bash
   ./isaaclab.sh -p <script>.py ... --kit_args \
     "--/persistent/isaac/asset_root/default=$WORK/cog/isaac_assets/Assets/Isaac/5.1 \
      --/persistent/isaac/asset_root/cloud=$WORK/cog/isaac_assets/Assets/Isaac/5.1"
   ```
   Overriding `.../asset_root/cloud` is the key one: `NUCLEUS_ASSET_ROOT_DIR` is read from that
   setting **at module import** (`utils/assets.py` line 26), so all `{ISAAC_NUCLEUS_DIR}`
   f-strings in cfgs resolve to the local mirror with zero code changes. **UNCERTAINTY:**
   whether kit applies `--/persistent/...` CLI overrides before the `isaaclab.utils.assets`
   import in all entry points — verified pattern in the community, but test once on the cluster;
   if the ordering bites, fall back to (3).
3. **Belt-and-suspenders (recommended for the project cfgs anyway):** define in the project
   `ASSET_ROOT = os.environ.get("COG_ASSET_ROOT", ISAACLAB_NUCLEUS_DIR)` and build all
   `usd_path`s from that, so cluster jobs set `COG_ASSET_ROOT=$WORK/cog/isaac_assets/.../Isaac/IsaacLab`
   and never touch carb settings. Local absolute paths in `UsdFileCfg.usd_path` are fully
   supported (`check_file_path` returns 1 for local files and short-circuits the cloud).
4. The procedural cylinder-cup + shape goal-marker variants (Sections 2C, 3.1) have **no USD
   dependency at all** — keep them as the guaranteed-offline L0 fallback.
5. Robot/table USDs are also cloud assets — the same mirror covers
   `FRANKA_PANDA_CFG` (`source/isaaclab_assets/isaaclab_assets/robots/franka.py`, uses
   `{ISAAC_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd` — 1 usage found in tasks
   grep) and the SeattleLabTable. Budget ~1–2 GB for the mirrored subtree; full asset packs are
   tens of GB — prefer the subtree mirror.

---

## 5. Misc verified facts / gotchas

- `isaaclab_assets` (`source/isaaclab_assets/`) contains **robots only** (plus
  `pick_and_place.py`, a suction-cup robot) — no prop/object cfgs live there; props are always
  inline `RigidObjectCfg`s in task cfgs.
- Stack cubes are 4.4 cm-ish (`z init 0.0203` ⇒ half-height ~2 cm) — a known-good graspable size
  reference for the panda; keep the scaled cup body in the 4–6.5 cm diameter band.
- `UsdFileCfg` semantic tags (`semantic_tags=[("class", "cup")]`) are needed if any
  segmentation/`semantic` camera modality is used; visuomotor stack env tags robot/table/ground
  and each cube (`stack_joint_pos_env_cfg.py` lines 72–78, 113/123/133).
- `MultiAssetSpawnerCfg` property overrides: cfg-level `rigid_props`/`mass_props`/
  `collision_props` override each entry's; `visible` is ignored (per-entry wins);
  cfg-level `semantic_tags` are appended to entries'.
- `randomize_visual_texture_material` (franka_stack_events.py line 245) exists for texture-level
  domain randomization (Replicator; same `replicate_physics=False` requirement) — optional L3+.
- The place-mug task's success check (`place/mdp/terminations.py::object_placed_upright`) and its
  Mimic wrapper (`source/isaaclab_mimic/isaaclab_mimic/envs/agibot_place_upright_mug_mimic_env_cfg.py`,
  registered as `Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-Rel-Mimic-v0`) are the closest in-tree
  Mimic reference for a cup/mug task (object_ref="mug" subtask config) — reuse its subtask
  structure, swap robot to Franka.
