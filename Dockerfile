# ========================================================
# syntax=docker/dockerfile:1.4
# ========================================================

# ========================================================
# Stage 0 — Build mamba-ssm wheel against pinned torch
# ========================================================
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel AS mamba-builder

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124

WORKDIR /tmp/mamba

COPY constraints/torch-cu124-mamba.txt ./constraints/torch-cu124-mamba.txt

RUN --mount=type=cache,target=/var/cache/apt,id=apt-cache-zonos \
    apt update && \
    apt install -y --no-install-recommends \
        build-essential \
        ninja-build \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN PIP_INDEX_URL=${TORCH_CUDA_INDEX_URL} \
    pip install --no-cache-dir \
    -c constraints/torch-cu124-mamba.txt \
    torch==2.6.0+cu124 \
    torchaudio==2.6.0+cu124

RUN PIP_INDEX_URL=${TORCH_CUDA_INDEX_URL} \
    pip wheel --no-cache-dir --no-binary=:all: \
    -c constraints/torch-cu124-mamba.txt \
    mamba-ssm \
    -w /tmp/wheels

# ========================================================
# Stage 1 — Base layer with Python and system deps
# ========================================================
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel AS base

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124

LABEL built-by="Ctrl+C Ctrl+V DevOps - Thanks Chat" \
      purpose="API container that yells in beautiful voices"

WORKDIR /app

COPY constraints/torch-cu124-mamba.txt ./constraints/torch-cu124-mamba.txt
COPY requirements ./requirements

RUN --mount=type=cache,target=/var/cache/apt,id=apt-cache-zonos \
    apt update && \
    apt install -y --no-install-recommends \
        build-essential \
        ninja-build \
        espeak-ng \
        ffmpeg \
        libsndfile1 \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN PIP_INDEX_URL=${TORCH_CUDA_INDEX_URL} \
    pip install --no-cache-dir \
    -c constraints/torch-cu124-mamba.txt \
    torch==2.6.0+cu124 \
    torchaudio==2.6.0+cu124

RUN pip install --no-cache-dir -r requirements/runtime.txt
RUN pip install --no-cache-dir --no-build-isolation -r requirements/compile.txt

COPY --from=mamba-builder /tmp/wheels /tmp/wheels
RUN pip install --no-cache-dir --no-index --find-links=/tmp/wheels mamba-ssm

COPY pyproject.toml ./
COPY zonos ./zonos
RUN pip install --no-cache-dir --no-deps -e .

RUN python - <<'PY'
import torch, mamba_ssm
print('Torch', torch.__version__, 'CUDA', torch.version.cuda, 'available', torch.cuda.is_available())
print('mamba-ssm', getattr(mamba_ssm, '__version__', 'unknown'))
PY

# ========================================================
# Stage 2 — Runtime layer
# ========================================================
FROM base AS runtime

WORKDIR /app
COPY . ./

EXPOSE 8000

CMD ["python3", "main_zonos_tts_api.py"]
