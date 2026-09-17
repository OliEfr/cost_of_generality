"""Print the training-loss trajectory of an offline wandb run, plus the slope near the end.

Purpose: decide whether a cell is CONVERGED or merely STOPPED at the frozen 80k steps. L3's success
rate is flat in N, which is consistent with a genuine ceiling -- but equally consistent with 80k
steps being too few for a dataset carrying 10x the visual diversity. The loss slope over the last
10k steps discriminates: still descending => the fixed-step protocol, not the data, is binding.

(Loss is a weak proxy for rollout success in diffusion policies -- val loss can even be
anti-correlated with open-loop error -- so this is used only to judge OPTIMIZATION progress, never
to compare quality across cells.)

usage: loss_trace.py <path/to/run-XXXX.wandb>
"""

import sys

from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb


def main() -> int:
    ds = datastore.DataStore()
    ds.open_for_scan(sys.argv[1])
    pts = []
    while True:
        try:
            raw = ds.scan_data()
        except Exception:
            break
        if raw is None:
            break
        rec = pb.Record()
        rec.ParseFromString(raw)
        if rec.WhichOneof("record_type") != "history":
            continue
        step = loss = lr = None
        for it in rec.history.item:
            # This wandb version puts the name in nested_key (a repeated path), leaving .key empty;
            # matching on .key alone silently finds nothing and looks like "no loss logged".
            k = "/".join(it.nested_key) if it.nested_key else it.key
            if k == "_step":
                step = int(float(it.value_json))
            elif k in ("loss", "train/loss"):
                loss = float(it.value_json)
            elif k == "train/lr":
                lr = float(it.value_json)
        if step is not None and loss is not None:
            pts.append((step, loss, lr))

    if not pts:
        print("no loss history found")
        return 1
    pts.sort()
    print(f"{len(pts)} logged points, steps {pts[0][0]}..{pts[-1][0]}, "
          f"final lr {pts[-1][2]:.2e}")
    # trailing-window means: smooth enough to read a trend through diffusion-loss noise
    for lo in range(0, pts[-1][0] + 1, 10000):
        win = [l for _s, l, _lr in pts if lo <= _s < lo + 10000]
        if win:
            print(f"  steps {lo:>6}-{lo + 10000:<6} mean loss {sum(win) / len(win):.5f}")
    last20 = [l for s, l, _lr in pts if s >= pts[-1][0] - 20000]
    first_half = last20[: len(last20) // 2]
    second_half = last20[len(last20) // 2:]
    if first_half and second_half:
        a, b = sum(first_half) / len(first_half), sum(second_half) / len(second_half)
        print(f"  final 20k: {a:.5f} -> {b:.5f}  ({100 * (b - a) / a:+.2f}% over the last 20k steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
