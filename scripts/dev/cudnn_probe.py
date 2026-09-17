"""Isolate CUDNN_STATUS_NOT_INITIALIZED seen in the container eval (job 58046658).
Runs a conv2d WITHOUT Kit; prints library/versions so a missing-lib cause is visible."""
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version(), flush=True)
print("cuda avail", torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
free, total = torch.cuda.mem_get_info()
print(f"gpu mem free {free/2**30:.1f} / {total/2**30:.1f} GiB", flush=True)
for lib in ("libcudnn.so.9", "libcudnn.so.8", "libcudnn_ops.so.9"):
    try:
        import ctypes; ctypes.CDLL(lib); print(f"{lib} dlopen OK", flush=True)
    except OSError as e:
        print(f"{lib} dlopen FAILED: {e}", flush=True)
x = torch.randn(8, 3, 112, 112, device="cuda")
w = torch.nn.Conv2d(3, 64, 7, stride=2, padding=3).cuda()
try:
    y = w(x); torch.cuda.synchronize(); print("CONV_CUDNN_OK", tuple(y.shape), flush=True)
except RuntimeError as e:
    print("CONV_CUDNN_FAILED:", e, flush=True)
    torch.backends.cudnn.enabled = False
    try:
        y = w(x); torch.cuda.synchronize(); print("CONV_NOCUDNN_OK", tuple(y.shape), flush=True)
    except RuntimeError as e2:
        print("CONV_NOCUDNN_FAILED:", e2, flush=True)
print("CUDNN_PROBE_DONE", flush=True)
