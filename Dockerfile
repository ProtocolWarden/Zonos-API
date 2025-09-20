# ========================================================
# syntax=docker/dockerfile:1.4
# ========================================================

# ========================================================
# Stage 0 — Build CUDA wheels against pinned torch
# MUST use the *devel* variant: compiles CUDA/C++ extensions (nvcc, headers).
# Update digest with tools/docker/update_pytorch_digest.sh when refreshing base image
# ========================================================
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56 AS mamba-builder
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124 \
    PYPI_INDEX_URL=https://pypi.org/simple \
    TORCH_CUDA_ARCH_LIST="8.6"

WORKDIR /tmp/mamba

COPY constraints/torch-cu124-mamba.txt ./constraints/torch-cu124-mamba.txt

# --- APT bootstrap (build essentials, ninja, curl for uv installer) -----------
RUN --mount=type=cache,target=/var/cache/apt,id=apt-cache-zonos-builder-01-apt \
    \
    # Remove any stale APT/DPKG locks (cache mount can retain these).
    rm -f \
      /var/cache/apt/archives/lock \
      /var/lib/dpkg/lock-frontend \
      /var/lib/dpkg/lock \
      /var/lib/apt/lists/lock; \
    \
    # Heal any interrupted dpkg operations (ignore errors).
    dpkg --configure -a || true; \
    \
    # Ensure required dir exists (can be missing in slim images).
    mkdir -p /var/lib/apt/lists/partial; \
    \
    # Update with retries (quiet mode) for transient network issues.
    apt-get -o Acquire::Retries=3 -y -q update; \
    \
    # Install minimal deps quietly without recommendations.
    apt-get install -y -q --no-install-recommends \
      build-essential \
      ninja-build \
      git \
      curl; \
    \
    # Clean package lists and cache to minimize layers.
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# --- Toolchain smoke test (nvcc/ninja present) --------------------------------
RUN --mount=type=cache,target=/root/.cache/toolchain-probe,id=toolchain-cache-zonos-builder-02-probe \
    python - <<'PY'
import shutil, sys

checks = {
    'nvcc': shutil.which('nvcc') is not None,
    'ninja': shutil.which('ninja') is not None,
}

print('Toolchain:', checks)
if not all(checks.values()):
    sys.exit(1)
PY

# --- uv install (fast prebuilt installs). Wheels will be built with pip -------
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-builder-03-install \
    UV_INSTALLER_VERSION=0.4.0 \
    curl -LsSf https://astral.sh/uv/install.sh | sh; \
    ln -sf /root/.local/bin/uv /usr/local/bin/uv; \
    \
    # Remove curl now that uv is available in the builder.
    apt-get purge -y curl; \
    apt-get autoremove -y; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# --- pip bootstrap (for building wheels) --------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip,id=pip-cache-zonos-builder-04-bootstrap \
    python -m pip install --upgrade pip setuptools wheel

# --- Install pinned torch/torchaudio (prebuilt wheels via uv) -----------------
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-builder-05-torch \
    uv pip install --system --no-cache-dir \
      -c constraints/torch-cu124-mamba.txt \
      --index-url ${TORCH_CUDA_INDEX_URL} \
      --extra-index-url ${PYPI_INDEX_URL} \
      torch==2.6.0+cu124 \
      torchaudio==2.6.0+cu124

RUN python - <<'PY'
import json
import pathlib
import torch

pathlib.Path('/tmp/torch_build.json').write_text(
    json.dumps({
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
    })
)
print('Recorded torch build info for base stage guard')
PY

# --- Build wheels (source builds with pip wheel) -------------------------------
# mamba-ssm from source; reuse pinned Torch in this stage.
RUN --mount=type=cache,target=/root/.cache/pip,id=pip-cache-zonos-builder-06-mamba \
    PIP_NO_BUILD_ISOLATION=1 MAMBA_FORCE_BUILD=TRUE \
    python -m pip wheel \
      --no-deps \
      -c constraints/torch-cu124-mamba.txt \
      --index-url ${TORCH_CUDA_INDEX_URL} \
      --extra-index-url ${PYPI_INDEX_URL} \
      --no-binary=mamba-ssm \
      --wheel-dir /tmp/wheels \
      mamba-ssm==2.2.5

# --- Build selective-scan from GitHub (matches Torch in this stage) ----------
ARG SELECTIVE_SCAN_GIT_REF=main
RUN --mount=type=cache,target=/root/.cache/pip,id=pip-cache-sscan \
    --mount=type=cache,target=/root/.cache/git,id=git-cache-sscan \
    PIP_NO_BUILD_ISOLATION=1 \
    python -m pip wheel \
      --no-deps \
      -c constraints/torch-cu124-mamba.txt \
      --index-url ${TORCH_CUDA_INDEX_URL} \
      --extra-index-url ${PYPI_INDEX_URL} \
      --no-binary=:all: \
      --wheel-dir /tmp/wheels \
      git+https://github.com/state-spaces/selective_scan.git@${SELECTIVE_SCAN_GIT_REF}

RUN --mount=type=cache,target=/root/.cache/pip,id=pip-cache-zonos-builder-07-flashcausal \
    PIP_NO_BUILD_ISOLATION=1 \
    python -m pip wheel \
      --no-deps \
      -c constraints/torch-cu124-mamba.txt \
      --index-url ${TORCH_CUDA_INDEX_URL} \
      --extra-index-url ${PYPI_INDEX_URL} \
      --no-binary=:all: \
      --wheel-dir /tmp/wheels \
      flash-attn==2.7.3 \
      causal-conv1d==1.5.0.post8

# ========================================================
# Stage 1 — Base layer with Python and system deps (slimmer runtime)
# MUST use the *runtime* variant: only needs shared libs to import wheels.
# ========================================================
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee AS base
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124 \
    PYPI_INDEX_URL=https://pypi.org/simple

LABEL built-by="Ctrl+C Ctrl+V DevOps - Thanks Chat" \
      purpose="API container that yells in beautiful voices"

WORKDIR /app

COPY constraints/torch-cu124-mamba.txt ./constraints/torch-cu124-mamba.txt
COPY requirements ./requirements

RUN --mount=type=cache,target=/root/.cache/req-scan,id=req-sanity-zonos-base-01-scan \
    python - <<'PY'
from pathlib import Path
import re
import sys

text = Path('requirements/runtime.txt').read_text()
pattern = re.compile(r"(git\+|git@|vcs)")

suspicious = [line.strip() for line in text.splitlines() if pattern.search(line.lower())]

if suspicious:
    print('VCS-like deps in requirements/runtime.txt:', suspicious)
    sys.exit(1)

print('Requirements sanity: OK')
PY

RUN --mount=type=cache,target=/var/cache/apt,id=apt-cache-zonos-base-02-apt \
    \
    # Remove any stale APT/DPKG locks (cache mount can retain these).
    rm -f \
      /var/cache/apt/archives/lock \
      /var/lib/dpkg/lock-frontend \
      /var/lib/dpkg/lock \
      /var/lib/apt/lists/lock; \
    \
    # Heal any interrupted dpkg operations (ignore errors).
    dpkg --configure -a || true; \
    \
    # Ensure required dir exists (can be missing in slim images).
    mkdir -p /var/lib/apt/lists/partial; \
    \
    # Update with retries (quiet mode) for transient network issues.
    apt-get -o Acquire::Retries=3 -y -q update; \
    \
    # Install runtime deps quietly without recommendations.
    apt-get install -y -q --no-install-recommends \
      build-essential \
      espeak-ng \
      ffmpeg \
      libsndfile1; \
    \
    # Clean package lists and cache to minimize layers.
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/var/cache/apt,id=apt-cache-zonos-base-03-uv-install \
    --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-base-04-uv-install \
    \
    # Remove any stale APT/DPKG locks (cache mount can retain these).
    rm -f \
      /var/cache/apt/archives/lock \
      /var/lib/dpkg/lock-frontend \
      /var/lib/dpkg/lock \
      /var/lib/apt/lists/lock; \
    \
    # Heal any interrupted dpkg operations (ignore errors).
    dpkg --configure -a || true; \
    \
    # Ensure required dir exists (can be missing in slim images).
    mkdir -p /var/lib/apt/lists/partial; \
    \
    # Update with retries (quiet mode) for transient network issues.
    apt-get -o Acquire::Retries=3 -y -q update; \
    \
    # Install curl and certificates temporarily for the uv installer.
    apt-get install -y -q --no-install-recommends \
      curl \
      ca-certificates; \
    \
    # Pin uv installer via environment variable (no --version flag).
    UV_INSTALLER_VERSION=0.4.0 \
    curl -LsSf https://astral.sh/uv/install.sh | sh; \
    ln -sf /root/.local/bin/uv /usr/local/bin/uv; \
    \
    # Remove curl (keep ca-certificates for TLS).
    apt-get purge -y curl; \
    apt-get autoremove -y; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-base-05-torch \
    uv pip install --system --no-cache-dir \
      -c constraints/torch-cu124-mamba.txt \
      --index-url ${TORCH_CUDA_INDEX_URL} \
      --extra-index-url ${PYPI_INDEX_URL} \
      torch==2.6.0+cu124 \
      torchaudio==2.6.0+cu124; \
    python -m pip uninstall -y torchvision || true

COPY --from=mamba-builder /tmp/torch_build.json /tmp/torch_build.json
RUN python - <<'PY'
import json, torch, pathlib

b = json.loads(pathlib.Path('/tmp/torch_build.json').read_text())
cur = {"torch": torch.__version__, "cuda": torch.version.cuda}
print("Runtime Torch:", cur)
assert cur == b, f"Builder/runtime Torch mismatch: {b} != {cur}"
PY

RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-base-06-reqs \
    uv pip install --system --no-cache-dir -r requirements/runtime.txt

COPY --from=mamba-builder /tmp/wheels /tmp/wheels
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-base-07-localwheels \
    uv pip install --system --no-cache-dir \
      -c constraints/torch-cu124-mamba.txt \
      --index-url ${TORCH_CUDA_INDEX_URL} \
      --extra-index-url ${PYPI_INDEX_URL} \
      --find-links=/tmp/wheels \
      selective-scan \
      mamba-ssm==2.2.5 \
      flash-attn==2.7.3 \
      causal-conv1d==1.5.0.post8 \
  && python -m pip check

RUN rm -rf /tmp/wheels || true

COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/vcs-scan,id=vcs-sanity-zonos-base-08-scan \
    python - <<'PY'
from pathlib import Path
import re
import sys

text = Path('pyproject.toml').read_text()

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    tomllib = None

suspicious = []
pattern = re.compile(r'(^|[^a-z0-9])vcs([^a-z0-9]|$)')

if tomllib is not None:
    data = tomllib.loads(text)
    project = data.get('project', {})

    def _collect(value):
        collected = []
        if isinstance(value, dict):
            for item in value.values():
                collected.extend(item)
        elif isinstance(value, (list, tuple, set)):
            collected.extend(value)
        elif value:
            collected.append(value)
        return collected

    deps = _collect(project.get('dependencies', []))
    deps += _collect(project.get('optional-dependencies', {}))

    for dep in deps:
        lower = str(dep).lower()
        if 'git+' in lower or 'git@' in lower or pattern.search(lower):
            suspicious.append(dep)
else:
    for line in text.splitlines():
        lower = line.lower()
        if 'git+' in lower or 'git@' in lower or pattern.search(lower):
            suspicious.append(line.strip())

if suspicious:
    print('VCS-like deps found:', suspicious)
    sys.exit(1)

print('Editable install sanity: OK')
PY
COPY zonos ./zonos
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-base-09-editable \
    uv pip install --system --no-cache-dir --no-deps -e .

RUN python - <<'PY'
import importlib.util, torch

print('Torch:', torch.__version__, torch.version.cuda)
for mod in ("selective_scan_cuda", "mamba_ssm"):
    spec = importlib.util.find_spec(mod)
    print(mod, "->", getattr(spec, "origin", None))

import mamba_ssm
print('mamba-ssm OK')
PY

# ========================================================
# Stage 2 — Runtime layer
# ========================================================
FROM base AS runtime

WORKDIR /app
COPY . ./

EXPOSE 8000

CMD ["python3", "main_zonos_tts_api.py"]
