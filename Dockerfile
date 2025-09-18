# ========================================================
# syntax=docker/dockerfile:1.4
# ========================================================

# ========================================================
# Stage 0 — Build CUDA wheels against pinned torch
# Update the digest with tools/docker/update_pytorch_digest.sh when refreshing the base image
# ========================================================
ARG WITH_TORCHVISION=0
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56 AS mamba-builder
ARG WITH_TORCHVISION

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_COLOR=1 \
    PIP_DEFAULT_TIMEOUT=60

WORKDIR /tmp/mamba

COPY constraints/torch-cu124-mamba.txt ./constraints/torch-cu124-mamba.txt

RUN --mount=type=cache,target=/var/cache/apt,id=apt-cache-zonos-builder \
    set -eux; \
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
    # Update with retries for transient network issues.
    apt-get -o Acquire::Retries=3 update; \
    \
    # Install minimal deps without recommendations.
    apt-get install -y --no-install-recommends \
      build-essential \
      ninja-build \
      git; \
    \
    # Clean package lists and cache to minimize layers.
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN PIP_INDEX_URL=${TORCH_CUDA_INDEX_URL} \
    pip install --no-cache-dir \
    -c constraints/torch-cu124-mamba.txt \
    torch==2.6.0+cu124 \
    torchaudio==2.6.0+cu124

RUN PIP_INDEX_URL=${TORCH_CUDA_INDEX_URL} \
    pip wheel --no-cache-dir --no-binary=:all: \
    -c constraints/torch-cu124-mamba.txt \
    mamba-ssm==2.2.5 \
    -w /tmp/wheels

RUN PIP_INDEX_URL=${TORCH_CUDA_INDEX_URL} \
    pip wheel --no-cache-dir --no-binary=:all: \
    -c constraints/torch-cu124-mamba.txt \
    flash-attn==2.7.3 \
    causal-conv1d==1.5.0.post8 \
    -w /tmp/wheels

RUN if [ "$WITH_TORCHVISION" = "1" ]; then \
      PIP_INDEX_URL=${TORCH_CUDA_INDEX_URL} \
      pip wheel --no-cache-dir --no-binary=:all: \
        -c constraints/torch-cu124-mamba.txt \
        torchvision==0.21.0+cu124 \
        -w /tmp/wheels ; \
    fi

# ========================================================
# Stage 1 — Base layer with Python and system deps
# ========================================================
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56 AS base
ARG WITH_TORCHVISION

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_COLOR=1 \
    PIP_DEFAULT_TIMEOUT=60

LABEL built-by="Ctrl+C Ctrl+V DevOps - Thanks Chat" \
      purpose="API container that yells in beautiful voices"

WORKDIR /app

COPY constraints/torch-cu124-mamba.txt ./constraints/torch-cu124-mamba.txt
COPY requirements ./requirements

RUN --mount=type=cache,target=/var/cache/apt,id=apt-cache-zonos-base \
    set -eux; \
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
    # Update with retries for transient network issues.
    apt-get -o Acquire::Retries=3 update; \
    \
    # Install runtime deps without recommendations.
    apt-get install -y --no-install-recommends \
      espeak-ng \
      ffmpeg \
      libsndfile1 \
      curl \
      git; \
    \
    # Clean package lists and cache to minimize layers.
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN PIP_INDEX_URL=${TORCH_CUDA_INDEX_URL} \
    pip install --no-cache-dir \
    -c constraints/torch-cu124-mamba.txt \
    torch==2.6.0+cu124 \
    torchaudio==2.6.0+cu124

RUN python - <<'PY'
import sys
import torch

print('Torch file:', getattr(torch, '__file__', '<missing>'))
print('Python prefix:', sys.prefix)
PY

RUN pip install --no-cache-dir -r requirements/runtime.txt

COPY --from=mamba-builder /tmp/wheels /tmp/wheels
RUN pip install --no-cache-dir --no-index --find-links=/tmp/wheels \
    mamba-ssm==2.2.5 \
    flash-attn==2.7.3 \
    causal-conv1d==1.5.0.post8

RUN if [ "$WITH_TORCHVISION" = "1" ]; then \
      pip install --no-cache-dir --no-index --find-links=/tmp/wheels torchvision==0.21.0+cu124 ; \
    fi
RUN rm -rf /tmp/wheels || true

COPY pyproject.toml ./
COPY zonos ./zonos
RUN pip install --no-cache-dir --no-deps -e .

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
