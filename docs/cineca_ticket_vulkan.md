# CINECA support ticket — Vulkan in Singularity on boost_usr_prod (DRAFT, not sent)

Context: D25 defers cluster-side eval pending this answer. Local eval is unblocked and is what the
study uses, so this ticket is not on the critical path — it decides whether Tasks 2-3 evals can run
on the cluster instead of serialising on the workstation 4090.

**To:** superc@cineca.it
**Subject:** EUHPC_B38_106 — Vulkan (VK_ERROR_INCOMPATIBLE_DRIVER) inside Singularity on boost_usr_prod

---

Dear CINECA support,

I am trying to run NVIDIA Isaac Sim 5.1 (Omniverse Kit) headless for offscreen rendering inside a
Singularity container on `boost_usr_prod` (project EUHPC_B38_106, SingularityPRO 4.3.1-1.el8,
driver 535.274.02).

Kit starts and the scene loads, but `vkCreateInstance` fails with `VK_ERROR_INCOMPATIBLE_DRIVER`, so
it silently falls back to CPU rendering (~274 s for a scene that takes ~20 s on a GPU).

What I have already verified inside the container, running with `--nv`:

- `libGLX_nvidia.so.0` is injected into `/.singularity.d/libs/` and can be `dlopen`ed;
- the ICD `/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json` is present and valid (`api_version` 1.3.242);
- `/dev/nvidia*` nodes are visible, including `nvidia-modeset`;
- `nvliblist.conf` already lists the graphics libraries;
- `--nvccli` is not usable here (no `nvidia-container-cli` available).

My questions:

1. Is Vulkan **graphics** (as opposed to CUDA compute) expected to work in Singularity on the A100
   boost nodes?
2. If so, is there a recommended configuration or supported recipe?

Thank you very much,
Oliver Hausdörfer

---

## Appendix — additional detail

**Environment.** Node `lrdn2482`, NVIDIA A100-SXM-64GB, driver `535.274.02` (CUDA 12.2),
SingularityPRO `4.3.1-1.el8`. Image built from `ubuntu:24.04` with Isaac Sim 5.1 installed via pip.
Invocation:

```
singularity exec --nv \
  -B $WORK:$WORK -B /leonardo_work:/leonardo_work -B $FAST:$FAST \
  --env HOME=<container home> --env OMNI_KIT_ACCEPT_EULA=YES \
  cog-env-5.1.0.sif python <script>
```

**The failure, verbatim from Kit's own log.** Emitted twice; the second occurrence is Kit's
compatibility-mode retry.

```
[Error] [carb.graphics-vulkan.plugin] VkResult: ERROR_INCOMPATIBLE_DRIVER
[Error] [carb.graphics-vulkan.plugin] vkCreateInstance failed. Vulkan 1.1 is not
        supported, or your driver requires an update.
[Error] [gpu.foundation.plugin] carb::graphics::createInstance failed.
[Error] [omni.gpu_foundation_factory.plugin] Failed to create any GPU devices,
        including an attempt with compatibility mode.
```

**CUDA in the same container on the same node works normally**, which is why I believe the problem is
specific to Vulkan rather than to GPU access in general: `torch 2.7.0+cu128` reports
`is_available=True`, `device_count=1`, `get_device_name(0)='NVIDIA A100-SXM-64GB'`, and a 1024x1024
matmul completes. `nvidia-smi` inside the container reports driver 535.274.02. Visible device nodes:
`/dev/nvidia0..3`, `nvidiactl`, `nvidia-uvm`, `nvidia-uvm-tools`, `nvidia-modeset`,
`/dev/nvidia-caps/nvidia-cap{1,2}`. `libcuda.so.1` is injected at `/.singularity.d/libs/`.

Kit does also log CUDA errors (`no CUDA-capable device is detected`), but these appear only *after*
`Failed to create any GPU devices`, one of them reports an invalid device ordinal
(`device 1294759088`), and `omni.graph.core` reports "unable to get a valid CUDA device id **from the
renderer**". Given the working CUDA test above, I read these as downstream consequences of the
Vulkan failure rather than a separate problem — Omniverse enumerates devices through its own
Vulkan-based GPU foundation.

**All NVIDIA Vulkan support libraries are injected and version-matched** to the host driver, so the
commonly cited cause of this error (a missing `libnvidia-glvkspirv`) does not apply:
`libGLX_nvidia.so.{0,535.274.02}`, `libnvidia-glvkspirv.so.535.274.02`,
`libnvidia-rtcore.so.535.274.02`, `libnvidia-glcore.so.535.274.02`,
`libnvidia-glsi.so.535.274.02`, `libnvoptix.so.{1,535.274.02}`.

**Loader environment variables cannot be used as a workaround**, because Kit clears them itself. Its
log shows `VK_SDK_PATH`, `VULKAN_SDK`, `VK_LAYER_PATH`, `VK_INSTANCE_LAYERS` and
`VULKAN_HEADERS_INSTALL_DIR` each reported as `Environment variable overridden: <name> = ` (empty),
and Kit loads its own bundled `libvulkan` rather than a system one. Setting `VK_ICD_FILENAMES` or
`VK_LOADER_DEBUG` from outside therefore has no observable effect.

**One specific question this raises**, in case it is the quickest thing to check: everything the ICD
*points to* is present and loadable, so the remaining candidate is whether the Vulkan loader running
inside the container can see the ICD **JSON file itself** at its standard search path. Is
`/usr/share/vulkan/icd.d/nvidia_icd*.json` expected to be made visible inside the container by
`--nv`, or should we bind it in ourselves?

I am happy to share the image or cut down a minimal reproducer if that helps.
