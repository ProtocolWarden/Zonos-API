from __future__ import annotations

import logging
import os

import torch

BACKBONES = {}
logger = logging.getLogger(__name__)

if not torch.cuda.is_available():
    os.environ.setdefault("MAMBA_TRITON_DISABLED", "1")

try:
    from ._mamba_ssm import MambaSSMZonosBackbone  # type: ignore[attr-defined]

    BACKBONES["mamba_ssm"] = MambaSSMZonosBackbone
except Exception as exc:  # pragma: no cover - defensive (GPU-only dependency)
    logger.warning(
        "MambaSSM backend unavailable; falling back to torch-only backbone: %s", exc
    )

from ._torch import TorchZonosBackbone

BACKBONES["torch"] = TorchZonosBackbone
