"""One-time embedding of the frozen instruction sets (H1, lang dimension).

Runs ONLINE on the workstation in cog_isaac (transformers 4.57.6). The npz it
writes is the frozen artifact consumed by conversion (candidate A env_state
channel) and by eval injection; the cluster never needs transformers or CLIP
weights for candidate A.

Usage:
    python scripts/dev/embed_instructions.py \
        --instructions configs/instructions/instructions_v1.json

Writes <embeddings_file> next to the JSON and prints sha256 lines for
SHA256SUMS. Refuses to overwrite an existing npz (freeze rule: new version,
never mutation).
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instructions", required=True)
    args = ap.parse_args()

    spec_path = Path(args.instructions)
    spec = json.loads(spec_path.read_text())
    model_id = spec["model"]
    out_path = spec_path.parent / spec["embeddings_file"]
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite frozen artifact {out_path} (bump the version instead)")

    from transformers import AutoTokenizer, CLIPTextModelWithProjection

    tok = AutoTokenizer.from_pretrained(model_id)
    model = CLIPTextModelWithProjection.from_pretrained(model_id).eval()
    try:
        from huggingface_hub import snapshot_download

        revision = Path(snapshot_download(model_id, allow_patterns=["config.json"])).name
    except Exception:
        revision = "unknown"

    arrays: dict[str, np.ndarray] = {
        "model": np.array(model_id),
        "revision": np.array(revision),
    }
    for task, strings in spec["tasks"].items():
        with torch.no_grad():
            inputs = tok(strings, padding=True, return_tensors="pt")
            feats = model(**inputs).text_embeds  # (N, 512)
        feats = torch.nn.functional.normalize(feats, dim=-1).float().numpy()
        arrays[task] = feats
        arrays[f"{task}__instructions"] = np.array(strings)

        norms = np.linalg.norm(feats, axis=1)
        cos = feats @ feats.T
        off = cos[~np.eye(len(strings), dtype=bool)]
        assert np.allclose(norms, 1.0, atol=1e-5), f"{task}: not unit-norm"
        assert off.max() < 1.0 - 1e-6, f"{task}: duplicate embeddings"
        print(
            f"[embed] {task}: {feats.shape} unit-norm ok; within-task cosine "
            f"min={off.min():.4f} mean={off.mean():.4f} max={off.max():.4f}"
        )

    np.savez(out_path, **arrays)
    print(f"[embed] wrote {out_path} (model {model_id} @ {revision})")
    for p in (spec_path, out_path):
        print(f"{sha256(p)}  {p.name}")
    print("EMBED_OK")


if __name__ == "__main__":
    main()
