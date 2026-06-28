"""Post-load snapshot-request sentinels for the VideoFoundry backup responder.

When Zonos actually (re)downloads a model file and deploys it into the synced
``models/tts/zonos`` tree, it drops a tiny sentinel here. A host-side responder
(SyncControl ``snapshot watch``) consumes the sentinels and runs a change-gated,
hardlink-deduped snapshot of ``models/``.

The request dir lives at ``models/.snapshot-requests`` — the same location every
other VideoFoundry model loader uses, and inside the one path this container
bind-mounts from the host (``application-models`` → host ``models/``). It is
excluded from the snapshot manifest and ignored by Syncthing.
"""
from __future__ import annotations

import os
from pathlib import Path


def _deployment_base() -> Path:
    return Path(os.environ.get("ZONOS_DEPLOYMENT_DIR", "/app/models/tts/zonos"))


def request_dir() -> Path:
    """Resolve the request dir; ``VF_SNAPSHOT_REQUEST_DIR`` overrides the default."""
    override = os.environ.get("VF_SNAPSHOT_REQUEST_DIR")
    if override:
        return Path(override)
    # The deployment base is ``<models>/tts/zonos``; the models root is its
    # grandparent, and the request dir is ``<models>/.snapshot-requests``.
    return _deployment_base().parent.parent / ".snapshot-requests"


def request_snapshot(tag: str) -> bool:
    """Best-effort: signal that ``models/`` gained content. Never raises."""
    try:
        target = request_dir()
        target.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if (c.isalnum() or c in "-._") else "_" for c in tag) or "model"
        (target / f"{safe}.req").write_text("", encoding="utf-8")
        return True
    except OSError:
        return False
