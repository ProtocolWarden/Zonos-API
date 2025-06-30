# ========================================================
# syntax=docker/dockerfile:1.4
# ========================================================

# ========================================================
# Stage 1 — Base Layer with System and Python Deps
# ========================================================
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel AS base

# Non-Interactive Mode
ENV DEBIAN_FRONTEND=noninteractive

# Metadata for traceability
LABEL built-by="Ctrl+C Ctrl+V DevOps - Thanks Chat" \
      purpose="API container that yells in beautiful voices"

WORKDIR /app

# ========================================================
# System Dependencies
# ========================================================
RUN --mount=type=cache,target=/var/cache/apt,id=apt-cache-zonos \
    apt update && \
    apt install -y \
        espeak-ng \
        ffmpeg \
        libsndfile1 \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# ========================================================
# Python Packaging Tools and UV Installer
# ========================================================
RUN --mount=type=cache,target=/root/.cache/pip,id=pip-cache-setup-zonos \
    pip install --upgrade pip setuptools wheel && \
    pip install uv==0.7.9

# ========================================================
# Install Python Dependencies with uv + Editable Install
# ========================================================
COPY pyproject.toml ./
COPY zonos ./zonos
RUN --mount=type=cache,target=/root/.cache/pip,id=pip-cache-zonos \
    uv pip install --system -e . && \
    uv pip install --system -e .[compile]

# ========================================================
# Stage 2 — Runtime Layer
# ========================================================
FROM base AS runtime

WORKDIR /app
COPY . ./
EXPOSE 8000

# ========================================================
# Entrypoint
# ========================================================
CMD ["python3", "main_zonos_tts_api.py"]
