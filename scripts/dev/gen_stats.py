#!/usr/bin/env python
"""Collect exact Mimic generation statistics into experiments/gen_stats.csv.

Ground truth is the episode count in each `<name>.hdf5` / `<name>_failed.hdf5` pair
that RecorderManager writes, NOT the generator's stdout: carb buffers the final
progress line and loses it at shutdown, so log tails understate the success count
(observed: 21/74 printed for a leg that actually held 40/120). See docs/decisions.md D16.

Idempotent — recomputes every row from the files, so it is safe to re-run after each
leg of a wave lands.

  python scripts/dev/gen_stats.py                       # refresh the CSV
  python scripts/dev/gen_stats.py --chain-wave T2_ \
      --wave-start "2026-08-17 03:32:16"                # also fill wall_min for a
                                                        # sequentially-run wave
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import pathlib
import re

import h5py

REPO = pathlib.Path(__file__).resolve().parents[2]
HDF5_DIR = REPO / "data" / "hdf5"
OUT = REPO / "experiments" / "gen_stats.csv"

FIELDS = [
    "dataset", "task", "level", "variant", "successes", "failures", "attempts",
    "gen_sr_pct", "mean_ep_len", "min_ep_len", "max_ep_len", "finished_at",
    "wall_min", "size_gb",
]


def parse_name(stem: str) -> tuple[str, str, str]:
    """`T2_L3v07` -> (drawer_stow, L3, v07); `L1` -> (cup_place, L1, -)."""
    task = "drawer_stow" if stem.startswith("T2_") else "cup_place"
    rest = stem[3:] if stem.startswith("T2_") else stem
    m = re.match(r"^(L\d)(v\d+)?$", rest)
    if not m:
        return task, rest, "-"
    return task, m.group(1), m.group(2) or "-"


class StillGenerating(Exception):
    """The writer still holds an HDF5 lock on this file, so it has no final count."""


def episode_stats(path: pathlib.Path) -> tuple[int, list[int]]:
    """Episode count and action-sequence lengths (metadata only, no payload read).

    Raises StillGenerating if the generator process holds the file lock — a leg that
    is mid-flight has no final result, and reading a half-written HDF5 would be junk.
    """
    try:
        handle = h5py.File(path, "r")
    except (BlockingIOError, OSError) as exc:
        raise StillGenerating(str(path)) from exc
    with handle as h:
        if "data" not in h:
            return 0, []
        data = h["data"]
        lengths = []
        for key in data:
            grp = data[key]
            if "actions" in grp:
                lengths.append(int(grp["actions"].shape[0]))
        return len(data.keys()), lengths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-smoke", action="store_true",
                    help="also report smoke/debug datasets (excluded by default)")
    ap.add_argument("--chain-wave", default=None,
                    help="dataset prefix generated sequentially by one script; "
                         "wall_min is then each leg's finish minus the previous one's")
    ap.add_argument("--wave-start", default=None,
                    help="'YYYY-MM-DD HH:MM:SS' start of --chain-wave, for its first leg")
    args = ap.parse_args()

    rows: list[dict] = []
    in_flight: list[str] = []
    for path in sorted(HDF5_DIR.glob("*.hdf5")):
        stem = path.stem
        if stem.endswith("_failed") or "source" in stem:
            continue
        if not args.include_smoke and "smoke" in stem.lower():
            continue
        failed = path.with_name(f"{stem}_failed.hdf5")
        try:
            succ, lengths = episode_stats(path)
            fail = episode_stats(failed)[0] if failed.exists() else 0
        except StillGenerating:
            in_flight.append(stem)
            continue
        attempts = succ + fail
        task, level, variant = parse_name(stem)
        rows.append({
            "dataset": stem,
            "task": task,
            "level": level,
            "variant": variant,
            "successes": succ,
            "failures": fail,
            "attempts": attempts,
            "gen_sr_pct": f"{100 * succ / attempts:.1f}" if attempts else "",
            "mean_ep_len": f"{sum(lengths) / len(lengths):.0f}" if lengths else "",
            "min_ep_len": min(lengths) if lengths else "",
            "max_ep_len": max(lengths) if lengths else "",
            "finished_at": dt.datetime.fromtimestamp(path.stat().st_mtime)
                             .strftime("%Y-%m-%d %H:%M:%S"),
            "wall_min": "",
            "size_gb": f"{path.stat().st_size / 1024**3:.2f}",
        })

    # wall_min only where it is actually derivable: one script running legs back to back.
    if args.chain_wave:
        wave = sorted((r for r in rows if r["dataset"].startswith(args.chain_wave)),
                      key=lambda r: r["finished_at"])
        prev = (dt.datetime.strptime(args.wave_start, "%Y-%m-%d %H:%M:%S")
                if args.wave_start else None)
        for r in wave:
            end = dt.datetime.strptime(r["finished_at"], "%Y-%m-%d %H:%M:%S")
            if prev is not None:
                r["wall_min"] = f"{(end - prev).total_seconds() / 60:.0f}"
            prev = end

    rows.sort(key=lambda r: (r["task"], r["level"], r["variant"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {OUT.relative_to(REPO)} ({len(rows)} finished datasets)")
    if in_flight:
        print(f"still generating, not counted: {', '.join(in_flight)}")
    print(f"{'dataset':12} {'succ':>5} {'att':>6} {'gen SR':>8} {'ep len':>7} {'wall':>6}")
    for r in rows:
        print(f"{r['dataset']:12} {r['successes']:5} {r['attempts']:6} "
              f"{r['gen_sr_pct']:>7}% {r['mean_ep_len']:>7} {r['wall_min']:>6}")

    # Per-level aggregate: an L3 *variant* is narrower than L3 as a condition, so the
    # curve point for L3 is the pooled count over its variants, never a single variant.
    print("\nper-level aggregate (pooled over variants):")
    agg: dict[tuple[str, str], list[int]] = {}
    for r in rows:
        k = (r["task"], r["level"])
        s, a = agg.setdefault(k, [0, 0])
        agg[k] = [s + r["successes"], a + r["attempts"]]
    for (task, level), (s, a) in sorted(agg.items()):
        if a:
            print(f"  {task:12} {level}  {s:4}/{a:5} = {100 * s / a:5.1f}%")


if __name__ == "__main__":
    main()
