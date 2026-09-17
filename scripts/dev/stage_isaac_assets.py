"""Mirror the USD asset subtrees an Isaac task needs from the public Omniverse S3 bucket.

Leonardo's COMPUTE nodes have no internet, so any asset referenced through
ISAAC_NUCLEUS_DIR / ISAACLAB_NUCLEUS_DIR stalls for 300 s and then fails the run. Staging
them once and pointing Kit at the local copy fixes that:

    --kit_args="--/persistent/isaac/asset_root/default=<DEST>/Assets/Isaac/5.1
                --/persistent/isaac/asset_root/cloud=<DEST>/Assets/Isaac/5.1"

Run on a LOGIN node (or locally, then rsync). stdlib only, so no env activation needed.

    python scripts/dev/stage_isaac_assets.py [extra/s3/prefix/ ...]

STAGE_DEST overrides the destination. The default prefixes cover Task 1 (cup_place);
Tasks 2-3 have NOT been audited -- add their prefixes as arguments when they are needed.
See docs/cluster_eval.md.
"""

import os
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
DEST = os.environ.get("STAGE_DEST", "/leonardo_work/EUHPC_B38_106/cog/isaac_assets")
PREFIXES = [
    "Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika/",
    "Assets/Isaac/5.1/Isaac/Props/Mounts/SeattleLabTable/",
    "Assets/Isaac/5.1/Isaac/IsaacLab/Objects/Mug/",
    "Assets/Isaac/5.1/Isaac/Environments/Grid/",
] + [p.rstrip("/") + "/" for p in sys.argv[1:]]

NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def list_keys(prefix):
    """All (key, size) under prefix, following continuation tokens."""
    token, keys = None, []
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={urllib.parse.quote(prefix)}"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
        for c in root.iter(f"{NS}Contents"):
            keys.append((c.find(f"{NS}Key").text, int(c.find(f"{NS}Size").text)))
        if root.findtext(f"{NS}IsTruncated") == "true":
            token = root.findtext(f"{NS}NextContinuationToken")
        else:
            return keys


def main():
    total = 0
    for prefix in PREFIXES:
        keys = list_keys(prefix)
        print(f"{prefix}: {len(keys)} objects, {sum(s for _, s in keys)/1e6:.1f} MB", flush=True)
        for key, size in keys:
            path = os.path.join(DEST, key)
            if os.path.exists(path) and os.path.getsize(path) == size:
                continue  # resumable: same name and size is treated as already staged
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(f"{BUCKET}/{urllib.parse.quote(key)}", path)
            total += size
    print(f"downloaded {total/1e6:.1f} MB new; asset root = {DEST}/Assets/Isaac/5.1", flush=True)
    print("STAGE_ASSETS_DONE")


if __name__ == "__main__":
    main()
