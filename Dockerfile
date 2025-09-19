# ========================================================
# syntax=docker/dockerfile:1.4
# ========================================================

# ========================================================
# Stage 0 — Build CUDA wheels against pinned torch
# MUST use the *devel* variant: this stage compiles CUDA/C++ extensions (nvcc, headers).
# Update the digest with tools/docker/update_pytorch_digest.sh when refreshing the base image
# ========================================================
ARG WITH_TORCHVISION=0
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56 AS mamba-builder
ARG WITH_TORCHVISION
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124 \
    PYPI_INDEX_URL=https://pypi.org/simple

WORKDIR /tmp/mamba

COPY constraints/torch-cu124-mamba.txt ./constraints/torch-cu124-mamba.txt

RUN --mount=type=cache,target=/var/cache/apt,id=apt-cache-zonos-builder-00-apt \
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

RUN --mount=type=cache,target=/root/.cache/toolchain-probe,id=uv-toolchain-probe \
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

RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-builder-00-uv-install \
    # Pin uv installer to a known-good release for reproducibility.
    export UV_INSTALLER_VERSION="0.4.0"; \
    curl -LsSf https://astral.sh/uv/install.sh | sh; \
    ln -sf /root/.local/bin/uv /usr/local/bin/uv; \
    \
    # Remove curl now that uv is available in the builder.
    apt-get purge -y curl; \
    apt-get autoremove -y; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip,id=pip-cache-zonos-builder-00-bootstrap \
    python -m pip install --upgrade pip setuptools wheel

RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-builder-01-torch \
    uv pip install --system --no-cache-dir \
      -c constraints/torch-cu124-mamba.txt \
      --index-url ${TORCH_CUDA_INDEX_URL} \
      --extra-index-url ${PYPI_INDEX_URL} \
      torch==2.6.0+cu124 \
      torchaudio==2.6.0+cu124

# Deterministic local build of mamba-ssm. We disable build isolation so the
# build uses the Torch we pinned earlier in this stage, and we force source
# build to avoid any network wheel guessing.
RUN --mount=type=cache,target=/root/.cache/pip,id=pip-cache-zonos-builder-02-mamba \
    PIP_NO_BUILD_ISOLATION=1 MAMBA_FORCE_BUILD=TRUE \
    python -m pip wheel \
      -c constraints/torch-cu124-mamba.txt \
      --index-url ${TORCH_CUDA_INDEX_URL} \
      --extra-index-url ${PYPI_INDEX_URL} \
      --no-binary=mamba-ssm \
      --wheel-dir /tmp/wheels \
      mamba-ssm==2.2.5

RUN --mount=type=cache,target=/root/.cache/pip,id=pip-cache-zonos-builder-03-flashcausal \
    python -m pip wheel \
      -c constraints/torch-cu124-mamba.txt \
      --index-url ${TORCH_CUDA_INDEX_URL} \
      --extra-index-url ${PYPI_INDEX_URL} \
      --no-binary=:all: \
      --wheel-dir /tmp/wheels \
      flash-attn==2.7.3 \
      causal-conv1d==1.5.0.post8

RUN --mount=type=cache,target=/root/.cache/pip,id=pip-cache-zonos-builder-04-vision \
    if [ "$WITH_TORCHVISION" = "1" ]; then \
      python -m pip wheel \
        -c constraints/torch-cu124-mamba.txt \
        --index-url ${TORCH_CUDA_INDEX_URL} \
        --extra-index-url ${PYPI_INDEX_URL} \
        --no-binary=:all: \
        --wheel-dir /tmp/wheels \
        torchvision==0.21.0+cu124 ; \
    fi

# ========================================================
# Stage 1 — Base layer with Python and system deps (slimmer runtime)
# MUST use the *runtime* variant: only needs shared libs to import wheels.
# ========================================================
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee AS base
ARG WITH_TORCHVISION
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124 \
    PYPI_INDEX_URL=https://pypi.org/simple

LABEL built-by="Ctrl+C Ctrl+V DevOps - Thanks Chat" \
      purpose="API container that yells in beautiful voices"

WORKDIR /app

COPY constraints/torch-cu124-mamba.txt ./constraints/torch-cu124-mamba.txt
COPY requirements ./requirements

RUN --mount=type=cache,target=/root/.cache/req-scan,id=uv-req-sanity \
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

RUN --mount=type=cache,target=/var/cache/apt,id=apt-cache-zonos-base-00-apt \
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
      espeak-ng \
      ffmpeg \
      libsndfile1; \
    \
    # Clean package lists and cache to minimize layers.
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/var/cache/apt,id=apt-cache-zonos-base-01-uv-install \
    --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-base-00-uv-install \
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
    # Pin uv installer to a known-good release for reproducibility.
    export UV_INSTALLER_VERSION="0.4.0"; \
    curl -LsSf https://astral.sh/uv/install.sh | sh; \
    ln -sf /root/.local/bin/uv /usr/local/bin/uv; \
    \
    # Remove curl (keep ca-certificates for TLS).
    apt-get purge -y curl; \
    apt-get autoremove -y; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-base-01-torch \
    uv pip install --system --no-cache-dir \
      -c constraints/torch-cu124-mamba.txt \
      --index-url ${TORCH_CUDA_INDEX_URL} \
      --extra-index-url ${PYPI_INDEX_URL} \
      torch==2.6.0+cu124 \
      torchaudio==2.6.0+cu124

RUN python - <<'PY'
import sys
import torch

print('Torch file:', getattr(torch, '__file__', '<missing>'))
print('Python prefix:', sys.prefix)
PY

RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-base-02-reqs \
    uv pip install --system --no-cache-dir -r requirements/runtime.txt

COPY --from=mamba-builder /tmp/wheels /tmp/wheels
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-base-03-localwheels \
    uv pip install --system --no-cache-dir --no-index --find-links=/tmp/wheels \
      mamba-ssm==2.2.5 \
      flash-attn==2.7.3 \
      causal-conv1d==1.5.0.post8

RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-base-04-vision \
    if [ "$WITH_TORCHVISION" = "1" ]; then \
      uv pip install --system --no-cache-dir --no-index --find-links=/tmp/wheels \
        torchvision==0.21.0+cu124; \
    fi
RUN rm -rf /tmp/wheels || true

COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/vcs-scan,id=uv-vcs-sanity \
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
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-base-05-editable \
    uv pip install --system --no-cache-dir --no-deps -e .

RUN python - <<'PY'
import sys
import torch

print('Torch', torch.__version__, 'CUDA', torch.version.cuda, 'available', torch.cuda.is_available())
print('Torch file:', getattr(torch, '__file__', '<missing>'))
print('Python prefix:', sys.prefix)

try:
    import mamba_ssm
    print('mamba-ssm', getattr(mamba_ssm, '__version__', 'unknown'))
except Exception as exc:  # pragma: no cover - smoke check
    print('mamba-ssm import failed:', exc)

for name in ('flash_attn', 'causal_conv1d'):
    try:
        module = __import__(name.replace('-', '_'))
    except Exception as exc:  # pragma: no cover - smoke check
        print(f'{name} import failed:', exc)
    else:
        print(f'{name}', getattr(module, '__version__', 'unknown'))
PY

# ========================================================
# Stage 2 — Runtime layer
# ========================================================
FROM base AS runtime

WORKDIR /app
COPY . ./

EXPOSE 8000

CMD ["python3", "main_zonos_tts_api.py"]
