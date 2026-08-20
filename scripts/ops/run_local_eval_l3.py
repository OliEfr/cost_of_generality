#!/usr/bin/env python3
"""Evaluate an L3 cell under the frozen variant-diagonal protocol, and pool to one result JSON.

WHY THIS EXISTS. L3's generality axis IS the object, so L3 is not one env: `levels.py` expands it
into 10 sub-levels L3v00..L3v09 (2 cylinder sizes x 5 colours). There is deliberately no
`Cog-CupPlace-L3-IK-Rel-Visuomotor-v0`, so `run_local_eval.sh`, which builds its task id as
f"Cog-CupPlace-{LEVEL}-...", fails all six L3 cells with gymnasium NameNotFound. Training was
unaffected (it only reads the dataset), so the gap stayed invisible until eval.

THE PROTOCOL (protocol.json / configs/eval_sets/L3.json, decided as D18): **variant v uses batch v**
-- the diagonal -- giving 10 x 20 = 200 episodes with 200 DISTINCT poses. Running batch 0 on every
variant would also total 200 episodes but only 20 distinct poses, because the variants share the
pose RNG stream. That distinction is the whole point of the diagonal and must not be "simplified".
Note L3's standard eval is therefore 200 episodes, where L0-L2 use 100; that is the frozen design
(rule 8), not an inconsistency, and it gives L3 a tighter CI.

Each variant runs in its OWN Isaac process: creating a second gym env inside one live Kit instance
is unreliable, and a per-variant process also means one crashed variant cannot poison the rest.

Output is the SAME schema as the L0-L2 results plus a `per_variant` breakdown, so
update_registry_from_evals.py consumes it with no special-casing.

usage: run_local_eval_l3.py NDEMOS [STEP]
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("COG_REPO", "/home/admin_07/cost_of_generality"))
PY = "/home/admin_07/miniconda3/envs/cog_isaac/bin/python"
N_VARIANTS = 10
NUM_ENVS = 20
BASE_SEED = 5000
MIN_FREE_MIB = int(os.environ.get("COG_EVAL_MIN_FREE_MIB", "14000"))
MAX_WAIT_MIN = int(os.environ.get("COG_EVAL_MAX_WAIT_MIN", "720"))


def log(msg: str, logfile: Path) -> None:
    line = f"[eval-l3] {time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with logfile.open("a") as f:
        f.write(line + "\n")


def free_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=60).stdout.strip().splitlines()[0]
    used, total = (int(x) for x in out.split(","))
    return total - used


def wait_for_gpu(logfile: Path) -> bool:
    """A foreign eval job shares this GPU and must never be disturbed (rule 2): wait, never shrink
    our footprint -- num_envs is part of the frozen benchmark and changing it changes the seeds."""
    waited = 0
    while True:
        f = free_mib()
        if f >= MIN_FREE_MIB:
            log(f"headroom OK: {f} MiB free", logfile)
            return True
        if waited >= MAX_WAIT_MIN:
            log(f"GAVE UP after {MAX_WAIT_MIN} min; only {f} MiB free", logfile)
            return False
        if waited % 30 == 0:
            log(f"waiting: {f} MiB free, need {MIN_FREE_MIB} ({waited} min)", logfile)
        time.sleep(120)
        waited += 2


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    ndemos = sys.argv[1]
    step = sys.argv[2] if len(sys.argv) > 2 else "080000"
    run_id = f"t1_L3_n{ndemos}_s0"

    results_dir = REPO / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = results_dir / "_l3_partials"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    logfile = REPO / "ops" / f"local_eval_{run_id}.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)

    final_out = results_dir / f"eval_L3_n{ndemos}_{step}.json"
    if final_out.exists() and final_out.stat().st_size > 0:
        log(f"{final_out.name} already exists, nothing to do", logfile)
        return 0

    ckpt = REPO / "experiments" / "runs" / run_id / "checkpoints" / step / "pretrained_model"
    weights = ckpt / "model.safetensors"
    if not weights.exists() or weights.stat().st_size == 0:
        log(f"MISSING or incomplete weights: {weights}", logfile)
        return 3

    log(f"start {run_id} step={step} -- {N_VARIANTS} variants x {NUM_ENVS} envs (diagonal)", logfile)

    outcomes: list = []
    per_variant: dict[str, dict] = {}
    failed: list[str] = []

    for v in range(N_VARIANTS):
        vkey = f"L3v{v:02d}"
        task = f"Cog-CupPlace-{vkey}-IK-Rel-Visuomotor-v0"
        part = tmp_dir / f"{run_id}_{step}_{vkey}.json"

        if not part.exists() or part.stat().st_size == 0:
            if not wait_for_gpu(logfile):
                log("ABORT: no GPU headroom", logfile)
                return 4
            # batch v on variant v: batches=1 with base_seed shifted by v is exactly the diagonal.
            proto = tmp_dir / f"proto_seed{BASE_SEED + v}.json"
            proto.write_text(json.dumps({
                "num_envs": NUM_ENVS,
                "batches": 1,
                "base_seed": BASE_SEED + v,
                "comment": (f"L3 diagonal slice: variant {vkey} evaluated on batch {v} only "
                            f"(seed {BASE_SEED + v}). Generated by run_local_eval_l3.py; the pooled "
                            f"result is the frozen 200-episode L3 benchmark (D18)."),
            }, indent=1))

            log(f"variant {vkey} -> {task} (seed {BASE_SEED + v})", logfile)
            cmd = [PY, "-m", "cog.eval.rollout_eval",
                   "--task", task,
                   "--checkpoint", str(ckpt),
                   "--protocol", str(proto),
                   "--num_inference_steps", "10",
                   "--out", str(part),
                   "--headless", "--enable_cameras"]
            # Isaac prompts "Do you accept the EULA? (Yes/No)" on a non-tty and then dies with
            # "Unable to bootstrap inner kit kernel: EOF when reading a line" unless this is set.
            # run_local_eval.sh exports it; a python subprocess does not inherit that, and the
            # failure looks nothing like a licence problem, so it is worth naming here.
            env = dict(os.environ)
            env["OMNI_KIT_ACCEPT_EULA"] = "YES"
            env["HF_HUB_OFFLINE"] = "1"
            with logfile.open("a") as lf:
                rc = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=lf, env=env).returncode
            # Kit exits 0 after fatal errors (D6): the artifact is the verdict, not the return code.
            if not part.exists() or part.stat().st_size == 0:
                log(f"VARIANT_FAILED {vkey} (rc={rc}, no artifact)", logfile)
                failed.append(vkey)
                continue

        d = json.loads(part.read_text())
        oc = d.get("outcomes", [])
        outcomes.extend(oc)
        per_variant[vkey] = {"successes": d.get("successes"), "episodes": d.get("episodes"),
                             "success_rate": d.get("success_rate"), "seed": BASE_SEED + v}
        log(f"  {vkey}: {d.get('successes')}/{d.get('episodes')} SR={d.get('success_rate')}", logfile)

    if failed:
        log(f"INCOMPLETE: {len(failed)} variant(s) failed: {failed} -- refusing to write a pooled "
            f"result from a partial diagonal (it would understate coverage silently)", logfile)
        return 5

    successes = sum(1 for o in outcomes if o is True or o == 1 or o == "success")
    if not outcomes:
        log("no outcomes collected", logfile)
        return 6
    # Prefer summing the per-variant success counts: robust to the outcome encoding.
    successes = sum(pv["successes"] for pv in per_variant.values())
    episodes = sum(pv["episodes"] for pv in per_variant.values())

    payload = {
        "task": "Cog-CupPlace-L3-IK-Rel-Visuomotor-v0 (pooled over 10 variants)",
        "checkpoint": str(ckpt),
        "num_inference_steps": 10,
        "success_rate": round(successes / episodes, 4),
        "successes": successes,
        "episodes": episodes,
        "outcomes": outcomes,
        "protocol": {
            "num_envs": NUM_ENVS,
            "variants": N_VARIANTS,
            "base_seed": BASE_SEED,
            "scheme": "diagonal: variant v evaluated on batch v",
            "comment": ("FROZEN L3 protocol (D18): 10 variants x 20 envs = 200 episodes with 200 "
                        "DISTINCT poses. Batch 0 on every variant would also total 200 episodes but "
                        "only 20 distinct poses, since variants share the pose RNG stream. L0-L2 "
                        "use 100 episodes (5 batches x 20); L3's 200 is the frozen design, so L3 "
                        "carries a tighter CI than L0-L2 by construction."),
        },
        "per_variant": per_variant,
    }
    final_out.write_text(json.dumps(payload, indent=1))
    log(f"POOLED_OK {final_out.name}: SR={payload['success_rate']} "
        f"({successes}/{episodes})", logfile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
