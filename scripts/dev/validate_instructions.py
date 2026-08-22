"""Offline integrity gate for the frozen instruction sets (H1, lang dimension).

Checks (no network, no transformers):
  - sha256 of the JSON and npz against configs/instructions/SHA256SUMS
  - JSON <-> npz consistency: task keys, 20 strings each, no duplicates,
    <= 14 words each, embeddings (20, dim) float32 unit-norm, string arrays match
  - cosine separability: within-task vs cross-task stats (probe precheck)

Prints INSTRUCTIONS_OK on success (markers, not exit codes -- running_jobs.md).
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instructions", required=True)
    args = ap.parse_args()

    spec_path = Path(args.instructions)
    spec = json.loads(spec_path.read_text())
    npz_path = spec_path.parent / spec["embeddings_file"]
    sums_path = spec_path.parent / "SHA256SUMS"

    sums = dict(
        line.split()[::-1]
        for line in sums_path.read_text().splitlines()
        if line.strip()
    )
    for p in (spec_path, npz_path):
        expect = sums.get(p.name)
        assert expect, f"{p.name} missing from SHA256SUMS"
        actual = sha256(p)
        assert actual == expect, f"sha256 mismatch for {p.name}: {actual} != {expect}"

    npz = np.load(npz_path)
    dim = spec["embedding"]["dim"]
    embs = {}
    for task, strings in spec["tasks"].items():
        assert len(strings) == 20, f"{task}: {len(strings)} strings, expected 20"
        assert len(set(strings)) == 20, f"{task}: duplicate strings"
        long = [s for s in strings if len(s.split()) > 14]
        assert not long, f"{task}: overlong instructions {long}"
        e = npz[task]
        assert e.shape == (20, dim) and e.dtype == np.float32, f"{task}: bad embedding array {e.shape} {e.dtype}"
        assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-4), f"{task}: not unit-norm"
        stored = [s for s in npz[f"{task}__instructions"]]
        assert [str(s) for s in stored] == strings, f"{task}: npz strings differ from JSON"
        embs[task] = e

    tasks = list(embs)
    for t in tasks:
        cos = embs[t] @ embs[t].T
        within = cos[~np.eye(20, dtype=bool)]
        print(f"[validate] {t}: within-task cosine min={within.min():.4f} mean={within.mean():.4f}")
    for i, a in enumerate(tasks):
        for b in tasks[i + 1:]:
            cross = embs[a] @ embs[b].T
            print(f"[validate] {a} x {b}: cross-task cosine mean={cross.mean():.4f} max={cross.max():.4f}")
            wa = (embs[a] @ embs[a].T)[~np.eye(20, dtype=bool)].mean()
            assert cross.mean() < wa, f"{a} vs {b}: cross-task cosine not below within-task"

    print(f"[validate] model={npz['model']} revision={npz['revision']}")
    print("INSTRUCTIONS_OK")


if __name__ == "__main__":
    main()
