# ========================================================
# syntax=docker/dockerfile:1.4
# ========================================================


ARG UV_VERSION=0.8.19


# ========================================================
# Stage 0 — Build CUDA wheels against pinned torch (devel)
# ========================================================
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56 AS builder
ARG UV_VERSION
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124 \
    PYPI_INDEX_URL=https://pypi.org/simple \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="8.6;8.9+PTX"

WORKDIR /work

# Build toolchain
RUN --mount=type=cache,target=/var/cache/apt,id=apt-cache-zonos-builder \
    rm -f /var/lib/apt/lists/lock /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock || true; \
    apt-get update -q; \
    apt-get install -y -q --no-install-recommends build-essential ninja-build curl git; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# uv
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-builder-install \
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh && \
    ln -sf /root/.local/bin/uv /usr/local/bin/uv && uv --version

# keep torch visible to builds (already true), but upgrade build tools
RUN python -m pip install -U pip setuptools wheel packaging

# Bring manifests only (no source yet)
COPY pyproject.toml uv.lock ./

# 2) Record torch build pair for runtime assert
RUN python - <<'PY'
import json, torch, pathlib
pathlib.Path('/torch_build.json').write_text(
    json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda})
)
PY

# 4) Export the compile extra and build wheels for CUDA extensions (no isolation!)
# list only the CUDA extension packages we need wheels for
RUN printf "%s\n%s\n%s\n" \
    "causal-conv1d==1.5.0.post8" \
    "flash-attn==2.7.4.post1" \
    "mamba-ssm==2.2.4" > /compile.pkgs.txt

# before the wheel step (builder stage)
ENV PIP_INDEX_URL=${TORCH_CUDA_INDEX_URL}
ENV PIP_EXTRA_INDEX_URL=${PYPI_INDEX_URL}
ENV PIP_CACHE_DIR=/root/.cache/pip \
    TMPDIR=/root/.cache/pip/tmp

# build wheels ONLY for those (torch already present in the PyTorch devel image)
RUN --mount=type=cache,target=/root/.cache/pip,id=pip-cache-zonos-builder-wheels \
    mkdir -p "$TMPDIR" && \
    python -m pip wheel --no-deps --no-binary=:all: --no-build-isolation \
      -r /compile.pkgs.txt \
      -w /wheels

# (Optional) sanity: ensure wheels have .so files (don’t import them)
RUN python - <<'PY'
import glob, zipfile, sys
wheels = sorted(glob.glob('/wheels/*.whl'))
print('Built wheels:', wheels)
if not wheels: sys.exit('No wheels built')
bad=[]
for w in wheels:
    with zipfile.ZipFile(w) as zf:
        if not any(n.endswith('.so') for n in zf.namelist()):
            bad.append(w)
if bad: sys.exit(f"Wheels missing .so: {bad}")
print("Wheel contents sanity: OK")
PY


# ========================================================
# Stage 1 — Dependency Layer
# ========================================================
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee AS base
ARG UV_VERSION
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124 \
    PYPI_INDEX_URL=https://pypi.org/simple

WORKDIR /app

# Runtime libs (with compiler for Triton JIT)
RUN --mount=type=cache,target=/var/cache/apt,id=apt-cache-zonos-runtime \
    rm -f /var/lib/apt/lists/lock /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock || true; \
    apt-get update -q; \
    apt-get install -y -q --no-install-recommends \
        espeak-ng ffmpeg libsndfile1 curl ca-certificates \
        gcc g++ make \
    ; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

# Tell Triton which compiler to use
ENV CC=/usr/bin/gcc CXX=/usr/bin/g++

# uv
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-runtime-install \
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh && \
    ln -sf /root/.local/bin/uv /usr/local/bin/uv && uv --version

# Bring manifests + prebuilt wheels + torch-build marker
COPY --from=builder /torch_build.json /torch_build.json
COPY --from=builder /compile.pkgs.txt /compile.pkgs.txt
COPY --from=builder /wheels /wheels
COPY pyproject.toml uv.lock ./

# 2) Install it
ENV PIP_ONLY_BINARY=:all:

RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-runtime-deps \
    uv pip install --system --no-cache-dir \
      --index-url ${TORCH_CUDA_INDEX_URL} \
      --extra-index-url ${PYPI_INDEX_URL} \
      '.[ui]' && \
    python -m pip check

# Ensure libtorch & vendored CUDA libs are discoverable via dynamic linker
ENV TORCH_SITE=/opt/conda/lib/python3.11/site-packages/torch \
    TORCH_NV=/opt/conda/lib/python3.11/site-packages/torch/nvidia

# Optional: keep LD_LIBRARY_PATH for extra safety
ENV LD_LIBRARY_PATH=${TORCH_SITE}/lib:\
${TORCH_NV}/cuda_runtime/lib:\
${TORCH_NV}/cudnn/lib:\
${TORCH_NV}/cublas/lib:\
${TORCH_NV}/cusparse/lib:\
${TORCH_NV}/cufft/lib:\
${TORCH_NV}/curand/lib:\
${TORCH_NV}/nccl/lib:\
${TORCH_NV}/nvjitlink/lib:\
${TORCH_SITE}/cusparselt/lib:\
/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}

# Register all relevant library paths with ld.so
RUN printf "%s\n" \
  "${TORCH_SITE}/lib" \
  "${TORCH_NV}/cuda_runtime/lib" \
  "${TORCH_NV}/cudnn/lib" \
  "${TORCH_NV}/cublas/lib" \
  "${TORCH_NV}/cusparse/lib" \
  "${TORCH_NV}/cufft/lib" \
  "${TORCH_NV}/curand/lib" \
  "${TORCH_NV}/nccl/lib" \
  "${TORCH_NV}/nvjitlink/lib" \
  "${TORCH_SITE}/cusparselt/lib" \
  "/usr/local/cuda/lib64" \
  "/usr/local/cuda/targets/x86_64-linux/lib" \
  > /etc/ld.so.conf.d/torch_cuda.conf && ldconfig

# 3) Install the CUDA extensions from prebuilt wheels
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-runtime-wheels \
    uv pip install --system --no-cache-dir \
      --find-links=/wheels \
      -r /compile.pkgs.txt && \
    python -m pip check

# 4) Add project source and install editable without re-resolving deps
COPY zonos ./zonos
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-runtime-editable \
    uv pip install --system --no-cache-dir --no-deps -e .

# 5) ldd smoke test (no hard import)
RUN python - <<'PY'
import importlib.util, ctypes, sys, os

# 1) Ensure the extension is import-discoverable
spec = importlib.util.find_spec("selective_scan_cuda")
if not spec or not getattr(spec, "origin", None):
    print("NOTE: selective_scan_cuda not found (hybrid disabled?)")
    sys.exit(0)

# 2) Try to dlopen the critical libs directly
required = [
    "libtorch.so",
    "libtorch_cuda.so",
    "libc10.so",
    "libcudart.so.12",
]

missing = []
for lib in required:
    try:
        ctypes.CDLL(lib)
        print(f"OK: {lib} loaded")
    except OSError as e:
        print(f"MISS: {lib} -> {e}")
        missing.append(lib)

# libcuda is provided by the host driver at runtime; don't enforce here.
if missing:
    raise SystemExit(f"Missing required dependency(s): {missing}")
print("CDLL sanity OK")
PY


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
