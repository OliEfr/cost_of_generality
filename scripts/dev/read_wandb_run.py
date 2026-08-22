"""Read live progress from an OFFLINE .wandb datastore (the only reliable live signal
for lerobot_train on Leonardo: slurm .out files stay nearly empty -- docs/timings.md:240).

Usage (on the machine holding the run dir, e.g. over ssh on the login node):
    python scripts/dev/read_wandb_run.py <run_dir_or_.wandb_file>

Prints the last history record's _step, _timestamp, loss and derived steps/s over the
tail of the run. Requires the wandb package (present in cog_lerobot and cog_isaac).
"""

import sys
from pathlib import Path

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore


def main():
    arg = Path(sys.argv[1])
    if arg.is_dir():
        cands = sorted(arg.rglob("*.wandb"))
        assert cands, f"no .wandb under {arg}"
        arg = cands[-1]

    ds = datastore.DataStore()
    ds.open_for_scan(str(arg))
    hist = []
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break
        if data is None:
            break
        rec = wandb_internal_pb2.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        if rec.WhichOneof("record_type") == "history":
            row = {}
            for item in rec.history.item:
                # newer wandb stores the key in nested_key (repeated), older in key
                k = "/".join(item.nested_key) if item.nested_key else item.key
                try:
                    row[k] = float(item.value_json)
                except ValueError:
                    pass
            row["_n"] = rec.history.step.num  # record-level step counter fallback
            hist.append(row)

    if not hist:
        print("NO_HISTORY_YET")
        return
    last = hist[-1]
    step = last.get("_step", last.get("train/step", last["_n"]))
    ts = last.get("_timestamp")
    loss = last.get("loss", last.get("train/loss"))
    line = f"records={len(hist)} step={step:.0f}"
    if loss is not None:
        line += f" loss={loss:.4f}"
    if len(hist) >= 2 and ts:
        prev = hist[max(0, len(hist) - 11)]
        dt = ts - prev.get("_timestamp", ts)
        dstep = step - prev.get("_step", prev.get("train/step", prev["_n"]))
        if dt > 0:
            line += f" steps_per_s={dstep / dt:.2f} (over last {dstep:.0f} steps)"
    else:
        line += " (no _timestamp key; use record count x log_freq for step estimate)"
    print(line)


if __name__ == "__main__":
    main()
