#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-2.6.0-cuda12.4-cudnn9-devel}"
IMAGE="pytorch/pytorch:${TAG}"

DIGEST=$(python - "$IMAGE" <<'PY'
import json
import sys
import urllib.request

image = sys.argv[1]
repo, tag = image.split(":", 1)

auth_url = (
    f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
)
with urllib.request.urlopen(auth_url) as resp:
    token = json.load(resp)["token"]

request = urllib.request.Request(
    f"https://registry-1.docker.io/v2/{repo}/manifests/{tag}",
    headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.docker.distribution.manifest.v2+json",
    },
)
with urllib.request.urlopen(request) as resp:
    digest = resp.headers["Docker-Content-Digest"]

print(digest)
PY
)

python - "$IMAGE" "$DIGEST" <<'PY'
import pathlib
import re
import sys

image = sys.argv[1]
digest = sys.argv[2]

dockerfile = pathlib.Path("Dockerfile")
text = dockerfile.read_text()
pattern = re.compile(rf"(FROM\s+{re.escape(image)})(?:@[^\s]+)?")
updated, count = pattern.subn(lambda match: f"{match.group(1)}@{digest}", text)
if count == 0:
    raise SystemExit(f"No FROM lines updated for {image}")

dockerfile.write_text(updated)
PY

printf 'Pinned %s to digest %s\n' "$IMAGE" "$DIGEST"
