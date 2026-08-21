"""Phase 2 of the clean re-eval (user instruction relayed 2026-08-21 night: "i want full
curves - just also rerun eval for remaining 6 cells... once scheduled queue is done"):
re-run the six T2 L3 cells (L3b arm, reported as L3) WITH stage telemetry, so the
drawer-open/lift/over funnel exists for the full 24-cell T2 matrix.

Chains behind the flat-cell sweep: waits for tmux session `cog_resweep` to end, then
requires RESWEEP_DONE in its log (a crash without the marker aborts this phase so a human
looks first). Each cell runs via run_local_eval_l3.py (10 Isaac boots, ~1.5 h/cell), which
self-gates on VRAM per variant; two cell drivers run concurrently (2 x ~7 GB peak + the
foreign job fits; three L3 drivers churn too many simultaneous Kit boots to be safe).
Outputs land in results/diagnostics/eval_T2_L3b_n<N>_080000_fixed.json (the driver's
out_tag path — deliberately outside the registry/curves globs; completion analysis reads
them explicitly). N=400 first so the headline row lands early.
"""
import os
import signal
import subprocess
import time
from pathlib import Path

REPO = Path("/home/admin_07/cost_of_generality/.claude/worktrees/results-analysis")
PY = "/home/admin_07/miniconda3/envs/cog_isaac/bin/python"
SWEEP_LOG = REPO / "ops" / "resweep_orchestrator.log"
MAX_SLOTS = 2
STAGGER_S = 300
TIMEOUT_S = 3 * 3600
POLL_S = 120

def say(*a):
    print(f"[l3phase] {time.strftime('%FT%T')}", *a, flush=True)

def sweep_running():
    return subprocess.run(["tmux", "has-session", "-t", "cog_resweep"],
                          capture_output=True).returncode == 0

say("L3PHASE_ARMED waiting for cog_resweep to finish")
while sweep_running():
    time.sleep(POLL_S)
if "RESWEEP_DONE" not in SWEEP_LOG.read_text():
    say("L3PHASE_ABORT_NO_DONE — flat sweep ended without RESWEEP_DONE; not starting")
    raise SystemExit(2)
say("L3PHASE_START flat sweep complete")

queue = [400, 200, 100, 50, 25, 10]
env_base = dict(os.environ,
                COG_REPO=str(REPO), COG_L3_LEVEL="L3b", COG_L3_OUT_TAG="fixed",
                COG_EVAL_MIN_FREE_MIB="10000",
                PATH=f"/home/admin_07/miniconda3/envs/cog_isaac/bin:{os.environ['PATH']}")

running, ok, fail, last_admit = [], 0, 0, 0.0
def reap(job):
    global ok, fail
    out = REPO / "results" / "diagnostics" / f"eval_T2_L3b_n{job['n']}_080000_fixed.json"
    if out.exists() and out.stat().st_size > 0:
        import json
        d = json.load(open(out))
        say(f"L3PHASE_CELL_OK n{job['n']} SR={d['successes']}/{d['episodes']}"
            f"={d['success_rate']} ({int(time.time()-job['t0'])}s)")
        ok += 1
    else:
        say(f"L3PHASE_CELL_FAILED n{job['n']} (rc={job['proc'].returncode}, no artifact)")
        fail += 1

while queue or running:
    for job in running[:]:
        if job["proc"].poll() is not None:
            running.remove(job)
            reap(job)
        elif time.time() - job["t0"] > TIMEOUT_S:
            say(f"L3PHASE_TIMEOUT n{job['n']} — killing process group")
            try:
                os.killpg(job["proc"].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            job["proc"].wait()
            running.remove(job)
            reap(job)
    if queue and len(running) < MAX_SLOTS and time.time() - last_admit >= STAGGER_S:
        n = queue.pop(0)
        log = open(REPO / "ops" / "resweep" / f"l3phase_n{n}.log", "a")
        proc = subprocess.Popen(
            [PY, str(REPO / "scripts" / "ops" / "run_local_eval_l3.py"), "T2", str(n), "080000"],
            cwd=REPO, env=env_base, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        say(f"L3PHASE_ADMIT n{n} pid={proc.pid}")
        running.append({"proc": proc, "n": n, "t0": time.time()})
        last_admit = time.time()
    time.sleep(POLL_S)
say(f"L3PHASE_DONE ok={ok} failed={fail}")
