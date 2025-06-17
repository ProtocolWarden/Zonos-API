# ========================================================
# syntax=docker/dockerfile:1.4
# ========================================================

# ========================================================
# Stage 1 — Base with System and Python Dependencies
# ========================================================
FROM pytorch/pytorch:2.5.2-cuda12.4-cudnn9-devel AS base

# Non-Interactive Mode
ENV DEBIAN_FRONTEND=noninteractive

# Metadata for traceability
LABEL built-by="Ctrl+C Ctrl+V DevOps - Thanks Chat" \
      purpose="API container that yells in beautiful voices"

WORKDIR /app

# ========================================================
# Install System Dependencies with Build Cache
# ========================================================
RUN --mount=type=cache,target=/var/cache/apt \
    apt update && \
    apt install -y \
        espeak-ng \
        ffmpeg \
        libsndfile1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ========================================================
# Install Python Dependencies with uv + Editable Install
# ========================================================
COPY pyproject.toml ./
COPY zonos ./zonos

RUN pip install uv==0.7.9 && \
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
