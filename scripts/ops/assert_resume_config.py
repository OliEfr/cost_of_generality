#!/usr/bin/env python3
"""Assert a checkpoint's saved config still matches the frozen flags before resuming.

WHY THIS EXISTS (2026-08-19). `train.sbatch` resumes with `--config_path=<ckpt>` and
`--resume=true`, because lerobot 0.4.4 raises "A config_path is expected when resuming a
run" otherwise (configs/train.py:89-95). On that path the config comes ENTIRELY from the
checkpoint and every other CLI flag is ignored. Consequence: if `configs/train/diffusion_base.sh`
is edited after a cell has checkpoints, the edit is SILENTLY IGNORED for that cell -- it keeps
training under the old config while the registry and logs suggest it used the new one.

That is exactly the failure mode CLAUDE.md rule 7 (identical hyperparameters for every cell)
exists to prevent, and it is invisible in exit codes and in wandb. It nearly bit us when
`use_separate_rgb_encoder_per_camera` was flipped false->true while t1_L0_n25_s0 already had
80k-step shared-encoder checkpoints on $WORK.

Usage:  assert_resume_config.py <path/to/train_config.json>
        (reads COG_DP_FLAGS + COG_DP_BATCH from the environment)
Exit 0 = safe to resume. Exit 1 = mismatch, ABORT the job (do not resume, do not requeue).
"""

import json
import os
import sys

# Values are compared after normalisation, so "true"/True and "[112,112]"/[112,112] match.
_TRUE = {"true", "True", "1"}
_FALSE = {"false", "False", "0"}


def norm(v):
    if isinstance(v, str):
        s = v.strip()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
        if s.startswith("[") and s.endswith("]"):
            try:
                return [norm(x) for x in json.loads(s)]
            except json.JSONDecodeError:
                return s
        try:
            return float(s) if ("." in s or "e" in s.lower()) else int(s)
        except ValueError:
            return s
    if isinstance(v, (list, tuple)):
        return [norm(x) for x in v]
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v) if isinstance(v, float) else v
    return v


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: assert_resume_config.py <train_config.json>", file=sys.stderr)
        return 2
    cfg_path = sys.argv[1]
    with open(cfg_path) as f:
        saved = json.load(f)
    saved_pol = saved.get("policy", {})

    flags = os.environ.get("COG_DP_FLAGS", "")
    if not flags.strip():
        print("ASSERT_RESUME: COG_DP_FLAGS is empty -- refusing to vouch for this resume",
              file=sys.stderr)
        return 1

    mismatches, checked = [], 0
    for tok in flags.split():
        if not tok.startswith("--policy.") or "=" not in tok:
            continue
        key, want = tok[len("--policy."):].split("=", 1)
        # `type` and `device` are not architecture and may legitimately differ across machines.
        if key in ("type", "device", "push_to_hub"):
            continue
        if key not in saved_pol:
            mismatches.append(f"  {key}: absent from checkpoint config, frozen wants {want!r}")
            continue
        checked += 1
        if norm(saved_pol[key]) != norm(want):
            mismatches.append(f"  {key}: checkpoint={saved_pol[key]!r} frozen={want!r}")

    want_batch = os.environ.get("COG_DP_BATCH")
    if want_batch and "batch_size" in saved:
        checked += 1
        if norm(saved["batch_size"]) != norm(want_batch):
            mismatches.append(f"  batch_size: checkpoint={saved['batch_size']!r} frozen={want_batch!r}")

    if mismatches:
        print("=" * 78, file=sys.stderr)
        print("ASSERT_RESUME FAILED -- refusing to resume: the checkpoint was trained with a", file=sys.stderr)
        print("DIFFERENT config than the currently frozen one. Resuming would silently keep the", file=sys.stderr)
        print("checkpoint's config (config_path-only resume) and break hyperparameter identity", file=sys.stderr)
        print(f"across cells (CLAUDE.md rule 7).\n  checkpoint: {cfg_path}", file=sys.stderr)
        print("\n".join(mismatches), file=sys.stderr)
        print("\nFix: move that checkpoint dir aside (rename, never delete) so the cell trains", file=sys.stderr)
        print("from scratch under the frozen config -- or revert the frozen config.", file=sys.stderr)
        print("=" * 78, file=sys.stderr)
        return 1

    print(f"[assert_resume] OK: {checked} frozen values match the checkpoint config")
    return 0


if __name__ == "__main__":
    sys.exit(main())
