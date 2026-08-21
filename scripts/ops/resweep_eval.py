"""Clean re-eval of all 54 flat cells (+ the D24 40k/60k re-check) with the FIXED harness.

User-authorised 2026-08-21 (night): "rerun eval. use two slots if gpu permits. check hourly.
if gpu later permits increase parallelism."

Admission control instead of a fixed slot count: a new eval starts only when free VRAM
>= MIN_FREE_MIB (10 GB: one eval holds ~7 GB, margin covers Isaac's scene-load spike),
starts are staggered STAGGER_S apart so a booting process's allocation is visible before
the next admission, and slots are hard-capped at 3 (3 x 7 GB + margin is the ceiling on a
24.5 GB card; docs/timings.md). With the foreign job resident this self-limits to 2 slots;
if the card empties, the 3rd admits itself. The foreign eval job is never touched (rule 2).

Frozen protocol untouched (rule 8): same eval sets, same seeds, num_envs=20 x 5. The only
change vs the published runs is the t==0 phantom-success guard (bug of 2026-08-21) and
--stages telemetry on T2. Outputs: results/eval_<TAG>_<level>_n<N>_<step>_fixed.json.
Resume-safe: existing outputs are skipped. Markers: RESWEEP_ADMIT/EVAL_OK/EVAL_FAILED/DONE.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/admin_07/cost_of_generality/.claude/worktrees/results-analysis")
CKPT_ROOT = Path("/home/admin_07/cost_of_generality/experiments/runs")  # read-only
PY = "/home/admin_07/miniconda3/envs/cog_isaac/bin/python"
OUTDIR = REPO / "results"
JOBLOGS = REPO / "ops" / "resweep"
JOBLOGS.mkdir(parents=True, exist_ok=True)

MIN_FREE_MIB = 10000
MAX_SLOTS = 3
STAGGER_S = 300
TIMEOUT_S = 150 * 60
POLL_S = 60

TASKS = {  # tag -> (gym prefix, max_steps, stages)
    "t2": ("Cog-DrawerStow", 1200, True),
    "t1": ("Cog-CupPlace", 600, False),
    "t3": ("Cog-PushTarget", 800, False),
}

queue = []
for tag in ("t2", "t1", "t3"):  # T2 first: scientifically hottest, longest cells
    for level in ("L1", "L2", "L0"):
        for n in (400, 200, 100, 50, 25, 10):
            queue.append((tag, level, n, "080000"))
queue += [("t1", "L0", 25, "040000"), ("t1", "L0", 25, "060000")]  # D24 re-check

def say(*a):
    print(f"[resweep] {time.strftime('%FT%T')}", *a, flush=True)

def free_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    used, total = (int(x) for x in out.strip().splitlines()[0].split(","))
    return total - used

def spawn(tag, level, n, step):
    prefix, max_steps, stages = TASKS[tag]
    run_id = f"{tag}_{level}_n{n}_s0"
    ckpt = CKPT_ROOT / run_id / "checkpoints" / step / "pretrained_model"
    out = OUTDIR / f"eval_{tag.upper()}_{level}_n{n}_{step}_fixed.json"
    name = out.stem
    if not ckpt.is_dir():
        say(f"RESWEEP_MISSING_CKPT {run_id}/{step}")
        return None
    if out.stat().st_size > 0 if out.exists() else False:
        say(f"RESWEEP_SKIP {name} exists")
        return None
    cmd = [PY, "-m", "cog.eval.rollout_eval", "--task",
           f"{prefix}-{level}-IK-Rel-Visuomotor-v0", "--checkpoint", str(ckpt),
           "--num_inference_steps", "10", "--max_steps", str(max_steps),
           "--out", str(out), "--headless", "--enable_cameras"]
    if stages:
        cmd.insert(-2, "--stages")
    env = dict(os.environ,
               PATH=f"/home/admin_07/miniconda3/envs/cog_isaac/bin:{os.environ['PATH']}",
               OMNI_KIT_ACCEPT_EULA="YES", HF_HUB_OFFLINE="1",
               PYTHONPATH=str(REPO / "src"))
    log = open(JOBLOGS / f"{name}.log", "a")
    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    say(f"RESWEEP_ADMIT {name} pid={proc.pid}")
    return {"proc": proc, "name": name, "out": out, "t0": time.time()}

def reap(job, ok_count, fail_count):
    rc = job["proc"].returncode
    if job["out"].exists() and job["out"].stat().st_size > 0:
        import json
        d = json.load(open(job["out"]))
        say(f"RESWEEP_EVAL_OK {job['name']} SR={d['successes']}/{d['episodes']}"
            f"={d['success_rate']:.3f} (rc={rc}, {int(time.time()-job['t0'])}s)")
        return ok_count + 1, fail_count
    say(f"RESWEEP_EVAL_FAILED {job['name']} (rc={rc}, no artifact)")
    return ok_count, fail_count + 1

say(f"RESWEEP_START queue={len(queue)} min_free={MIN_FREE_MIB} max_slots={MAX_SLOTS}")
running, ok, fail, last_admit = [], 0, 0, 0.0
while queue or running:
    for job in running[:]:
        if job["proc"].poll() is not None:
            running.remove(job)
            ok, fail = reap(job, ok, fail)
        elif time.time() - job["t0"] > TIMEOUT_S:
            say(f"RESWEEP_TIMEOUT {job['name']} — killing process group")
            try:
                os.killpg(job["proc"].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            job["proc"].wait()
            running.remove(job)
            ok, fail = reap(job, ok, fail)
    if queue and len(running) < MAX_SLOTS and time.time() - last_admit >= STAGGER_S:
        free = free_mib()
        if free >= MIN_FREE_MIB:
            job = spawn(*queue.pop(0))
            if job:
                running.append(job)
                last_admit = time.time()
            # None (skip/missing) -> loop immediately consumes the next item
            else:
                last_admit = 0.0
        else:
            say(f"waiting for headroom: {free} MiB free, {len(running)} running, "
                f"{len(queue)} queued")
    time.sleep(POLL_S)
say(f"RESWEEP_DONE ok={ok} failed={fail}")
