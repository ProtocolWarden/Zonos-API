"""Lightweight integrity checks for cached/deployed model files.

Cheap, load-free structural validation so a sync-conflict, aborted download, or
zeroed file is rejected before it's deployed into the synced ``models/`` tree
(the full Dac/backbone load is the final backstop).
"""
from __future__ import annotations

import json
from pathlib import Path


def validate_cached_file(path: Path) -> bool:
    """True if ``path`` looks structurally intact.

    - any file: must exist and be non-empty
    - ``.json``: must parse as JSON
    - ``.safetensors``: must carry a sane little-endian header-length prefix
      (first 8 bytes) that fits within the file
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    suffix = path.suffix
    if suffix == ".json":
        try:
            with path.open("r", encoding="utf-8") as handle:
                json.load(handle)
            return True
        except (OSError, ValueError):
            return False
    if suffix == ".safetensors":
        try:
            size = path.stat().st_size
            with path.open("rb") as handle:
                head = handle.read(8)
            if len(head) < 8:
                return False
            header_len = int.from_bytes(head, "little")
            return 0 < header_len <= size - 8
        except OSError:
            return False
    return True
