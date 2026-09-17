"""Asset-free RTX render smoke for offline cluster nodes.

frames_qa.py cannot run on Leonardo compute nodes: the task envs reference Franka USD
assets on the Omniverse S3 and the nodes have no internet (300 s timeout, job 58036943).
This smoke answers only the renderer question -- does Kit's RTX path produce real pixels
on an A100 -- using nothing but locally-generated prims (cubes + dome light + camera),
modelled on NVIDIA's standalone camera.py example.

Writes rtx_smoke.png + prints RTX_SMOKE_{OK,DEGENERATE} judged on pixel variance.
Output dir: $SMOKE_OUT (default ops/qa under the repo).
"""

import os

from isaacsim import SimulationApp

# Leonardo's driver 535.274.02 is misparsed by Kit as "535.18" and rejected as < 535.129
# (job 58037701) -- NVIDIA's documented misreport for 535.255+, with this documented switch.
# Without it the RTX Hydra engine refuses to start even though Vulkan is fully working.
sim_app = SimulationApp(
    {"headless": True, "extra_args": ["--/rtx/verifyDriverVersion/enabled=false"]}
)

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid  # noqa: E402
from isaacsim.core.utils.prims import create_prim  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
import isaacsim.core.utils.numpy.rotations as rot_utils  # noqa: E402

OUT = os.environ.get("SMOKE_OUT", os.path.join(os.path.dirname(__file__), "..", "..", "ops", "qa"))
os.makedirs(OUT, exist_ok=True)

world = World(stage_units_in_meters=1.0)
# Deliberately NO ground plane: add_default_ground_plane() pulls a USD from the cloud
# assets root. A dome light is a plain prim and needs nothing external.
create_prim("/World/light", "DomeLight", attributes={"inputs:intensity": 1000.0})
rng = np.random.default_rng(0)
for i in range(5):
    world.scene.add(
        DynamicCuboid(
            prim_path=f"/World/cube_{i}",
            name=f"cube_{i}",
            position=np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0.5, 2)]),
            scale=np.array([1.0, 1.0, 1.0]),
            color=rng.uniform(0.2, 1.0, size=3),
        )
    )
camera = Camera(
    prim_path="/World/camera",
    position=np.array([0.0, 0.0, 25.0]),
    frequency=20,
    resolution=(256, 256),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
)

world.reset()
camera.initialize()
for _ in range(60):
    world.step(render=True)

import sys

def emit(line, _buf=[]):
    print(line, flush=True)
    _buf.append(line)
    with open(os.path.join(OUT, "rtx_smoke.txt"), "w") as fh:
        fh.write("\n".join(_buf) + "\n")

rgba = camera.get_rgba()
if rgba is None or getattr(rgba, "size", 0) == 0:
    emit("RTX_SMOKE_DEGENERATE: camera returned no data")
else:
    rgb = np.asarray(rgba)[..., :3].astype(np.float32)
    uniq = len(np.unique(rgb.reshape(-1, 3), axis=0))
    emit(
        f"RTX_SMOKE frame: shape {rgb.shape} mean {rgb.mean():.1f} "
        f"std {rgb.std():.1f} unique_colours {uniq}"
    )
    try:
        from PIL import Image

        Image.fromarray(rgb.astype(np.uint8)).save(os.path.join(OUT, "rtx_smoke.png"))
        emit(f"wrote {os.path.join(OUT, 'rtx_smoke.png')}")
    except Exception as e:  # PNG is evidence, not the verdict
        emit(f"(png save failed: {e})")
    # A real render of lit coloured cubes has structure; a black/uniform frame is the
    # degraded/CPU-fallback signature.
    if rgb.std() > 3.0 and uniq > 50:
        emit("RTX_SMOKE_OK")
    else:
        emit("RTX_SMOKE_DEGENERATE: frame is blank/uniform")

sim_app.close()
