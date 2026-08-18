# Running long jobs (read before launching anything in tmux)

Every rule here was paid for. Dates refer to `docs/journal.md` entries with the full trace.

## 1. Three layers of monitoring, always

For anything over ~10 minutes:

1. **A tmux session** with its own log file: `tmux new-session -d -s cog_<what> "bash
   scripts/ops/<script>.sh > ops/<what>.log 2>&1"`. Never launch a long job in the
   foreground -- a dropped connection kills it.
2. **An event watcher** that fires the moment the session ends, so nothing sits finished
   and unnoticed:
   ```bash
   while tmux has-session -t cog_<what> 2>/dev/null; do sleep 120; done
   echo <WHAT>_ENDED; grep -ao "..._EXIT=[0-9]*" ops/<what>.log
   ```
   Run it with `run_in_background: true`.
3. **An hourly cron fallback** (`CronCreate`, e.g. `"13 * * * *"`) that re-checks the same
   things, because watchers do occasionally not fire. Cancel it when the job is done, so it
   does not become noise -- a stale watcher trains you to ignore watchers.

The hourly check should verify: `tmux ls`, the log's progress line AND the output file's
mtime, `nvidia-smi`, `df -h /`, plus whatever gate the job is working toward.

## 2. A tmux shell is NOT your interactive shell

Three differences, each of which has broken a launch here:

| difference | symptom | fix |
|---|---|---|
| no conda shell function | `python: command not found`, all legs exit 1 instantly (T3 wave, 2026-08-18) | `export PATH="/home/admin_07/miniconda3/envs/cog_isaac/bin:$PATH"` inside the script -- and use the absolute interpreter for pure-python drivers. `conda activate` needs a hook that is not there. |
| it HAS a TTY | Kit blocks forever on `Do you accept the EULA? (Yes/No)`; session alive, no GPU use, no output (T3 wave, 2026-08-18) | `export OMNI_KIT_ACCEPT_EULA=YES` **and** `< /dev/null` on every launch |
| clean environment | a missing redirect directory kills the session in <1 s (eval-set freeze, 2026-08-17) | `mkdir -p` the log directory first; use absolute paths |

After launching, always confirm the job is REALLY running -- process present, GPU memory
allocated, output file growing -- not merely that the tmux session exists.

## 3. Exit codes lie; markers do not

An Isaac/Kit script can **exit 0 after a fatal exception** (Kit's shutdown path swallows it).
The T3 eval-set freeze reported `EXIT=0` for all 13 legs while all 13 had died on
`NameNotFound` (2026-08-18). Every wave script must print a success MARKER
(`EVALSET_OK`, `VALIDATE_OK`, `..._EXIT=0` plus an artifact check) and the checker must
assert on the marker or the artifact, never on `$?` alone.

## 4. Liveness is an artifact question, not a process question

- A spinning process proves nothing: a `frames_qa.py` run sat at 110 % CPU holding 4.8 GB
  of VRAM for **24 hours** after writing its output in 13 minutes (2026-08-17). Check the
  output file's mtime.
- A quiet log proves nothing either: carb buffers output and flushes at shutdown, so a
  healthy job can look stalled. Again: check the file mtime.
- Hung Kit processes ignore SIGTERM. `kill -9` is required.

## 5. Never `pkill -f <pattern>` from a shell whose command line contains that pattern

It matches your own shell and kills it mid-sequence (exit 144), so the rest of your
command -- the cleanup, the relaunch -- silently never runs. Cost two incidents in one
session. Instead:

```bash
for p in $(pgrep -f "vendored/generate_dataset"); do
  case "$(tr '\0' ' ' < /proc/$p/cmdline)" in *PushTarget*) kill -9 "$p";; esac
done
```

## 6. Never glob HDF5 inputs

`RecorderManager` writes `<name>_failed.hdf5` beside every `<name>.hdf5`, so
`T2_L3v0*.hdf5` silently includes the failures. This produced a bad 455-episode dataset that
still passed validation. List inputs explicitly; the converter also refuses `_failed` names
as a backstop.

## 7. Parallelise across levels, not within

h264 conversion is single-core and the box has 32 threads: running the four levels as four
tmux sessions gave 4x throughput at unchanged per-level speed (2 h instead of 8 h, measured
2026-08-17). Check `nproc` and current load before assuming a job is as parallel as it can be.

## 8. Sizing and measuring

- Plan durations from `docs/timings.md`, which holds measured numbers per stage and task.
- Generation SR and demo counts come from `experiments/gen_stats.csv` (recomputed from the
  HDF5 pairs), never from a generator log -- log tails understate because the final progress
  flush is lost at shutdown.
