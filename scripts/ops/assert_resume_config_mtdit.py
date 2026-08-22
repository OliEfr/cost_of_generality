#!/usr/bin/env python3
"""Assert a multi_task_dit checkpoint's saved config still matches COG_DIT_FLAGS before resuming.

Sibling of assert_resume_config.py, which is bound to the diffusion cells' COG_DP_FLAGS /
COG_DP_BATCH and therefore CANNOT vouch for a candidate-B resume (it would refuse with
"COG_DP_FLAGS is empty" or, worse, compare against the wrong frozen set). Same failure
mode being guarded: lerobot 0.4.4's resume is config_path-only, so every CLI flag is
silently ignored on resume and an edited configs/train/lang_dit_b.sh would NOT reach a
resumed run — the checkpoint keeps training under its old config with no warning.

Usage:  assert_resume_config_mtdit.py <path/to/train_config.json>
        (reads COG_DIT_FLAGS + COG_DIT_BATCH from the environment)
Exit 0 = safe to resume. Exit 1 = mismatch, ABORT the job (do not resume, do not requeue).
"""

import json
import os
import sys

# Values are compared after normalisation, so "true"/True and "[224,224]"/[224,224] match.
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
        print("usage: assert_resume_config_mtdit.py <train_config.json>", file=sys.stderr)
        return 2
    cfg_path = sys.argv[1]
    with open(cfg_path) as f:
        saved = json.load(f)
    saved_pol = saved.get("policy", {})

    flags = os.environ.get("COG_DIT_FLAGS", "")
    if not flags.strip():
        print("ASSERT_RESUME_MTDIT: COG_DIT_FLAGS is empty -- refusing to vouch for this resume",
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

    want_batch = os.environ.get("COG_DIT_BATCH")
    if want_batch and "batch_size" in saved:
        checked += 1
        if norm(saved["batch_size"]) != norm(want_batch):
            mismatches.append(f"  batch_size: checkpoint={saved['batch_size']!r} frozen={want_batch!r}")

    if mismatches:
        print("=" * 78, file=sys.stderr)
        print("ASSERT_RESUME_MTDIT FAILED -- refusing to resume: the checkpoint was trained", file=sys.stderr)
        print("with a DIFFERENT config than the current lang_dit_b.sh. Resuming would silently", file=sys.stderr)
        print("keep the checkpoint's config (config_path-only resume) and invalidate the", file=sys.stderr)
        print(f"candidate-B measurement.\n  checkpoint: {cfg_path}", file=sys.stderr)
        print("\n".join(mismatches), file=sys.stderr)
        print("\nFix: move that checkpoint dir aside (rename, never delete) so the cell trains", file=sys.stderr)
        print("from scratch under the current config -- or revert the config edit.", file=sys.stderr)
        print("=" * 78, file=sys.stderr)
        return 1

    print(f"[assert_resume_mtdit] OK: {checked} frozen values match the checkpoint config")
    return 0


if __name__ == "__main__":
    sys.exit(main())
