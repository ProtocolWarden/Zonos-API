# ========================================================
# syntax=docker/dockerfile:1.4
# ========================================================

ARG UV_VERSION=0.8.19

# ========================================================
# Stage 0 — Build CUDA wheels against pinned torch (devel)
# ========================================================
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel AS builder
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
RUN apt-get update -q && \
    apt-get install -y -q --no-install-recommends build-essential ninja-build curl git && \
    rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh && \
    ln -sf /root/.local/bin/uv /usr/local/bin/uv && uv --version

# Bring manifests only (no source yet)
COPY pyproject.toml uv.lock ./

# 1) Install the EXACT torch/torchaudio first (ABI anchor)
RUN uv pip install --system --no-cache-dir \
      --index-url ${TORCH_CUDA_INDEX_URL} \
      --extra-index-url ${PYPI_INDEX_URL} \
      torch==2.6.0+cu124 \
      torchaudio==2.6.0+cu124

# 2) Record torch build pair for runtime assert
RUN python - <<'PY'
import json, torch, pathlib
pathlib.Path('/torch_build.json').write_text(
    json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda})
)
PY

# 3) Export the compile extra and build wheels for CUDA extensions (no isolation!)
RUN uv export --locked -E compile --format requirements-txt > /compile.lock.txt
RUN PIP_NO_BUILD_ISOLATION=1 UV_NO_BUILD_ISOLATION=1 \
    python -m pip wheel --no-deps --no-binary=:all: \
      --index-url ${TORCH_CUDA_INDEX_URL} \
      --extra-index-url ${PYPI_INDEX_URL} \
      -r /compile.lock.txt \
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
# Stage 1 — Runtime (slim) layer
# ========================================================
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS runtime
ARG UV_VERSION
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124 \
    PYPI_INDEX_URL=https://pypi.org/simple

WORKDIR /app

# Runtime libs (no compiler)
RUN apt-get update -q && \
    apt-get install -y -q --no-install-recommends espeak-ng ffmpeg libsndfile1 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh && \
    ln -sf /root/.local/bin/uv /usr/local/bin/uv && uv --version

# Bring manifests + prebuilt wheels + torch-build marker
COPY --from=builder /torch_build.json /torch_build.json
COPY --from=builder /compile.lock.txt /compile.lock.txt
COPY --from=builder /wheels /wheels
COPY pyproject.toml uv.lock ./

# 1) Install EXACT torch/torchaudio to match builder ABI
RUN uv pip install --system --no-cache-dir \
      --index-url ${TORCH_CUDA_INDEX_URL} \
      --extra-index-url ${PYPI_INDEX_URL} \
      torch==2.6.0+cu124 \
      torchaudio==2.6.0+cu124 && \
    python - <<'PY'
import json, torch, pathlib
b = json.loads(pathlib.Path('/torch_build.json').read_text())
cur = {"torch": torch.__version__, "cuda": torch.version.cuda}
print("Torch runtime:", cur)
assert cur == b, f"Builder/runtime Torch mismatch: {b} != {cur}"
PY

# 2) Export runtime set from lockfile and install it
RUN uv export --locked --format requirements-txt > /runtime.lock.txt && \
    uv pip install --system --no-cache-dir \
      --index-url ${TORCH_CUDA_INDEX_URL} \
      --extra-index-url ${PYPI_INDEX_URL} \
      -r /runtime.lock.txt && \
    python -m pip check

# 3) Install the CUDA extensions from prebuilt wheels
RUN uv pip install --system --no-cache-dir \
      --find-links=/wheels \
      -r /compile.lock.txt && \
    python -m pip check

# 4) Add project source and install editable without re-resolving deps
COPY zonos ./zonos
RUN uv pip install --system --no-cache-dir --no-deps -e .

# 5) ldd smoke test (no hard import)
RUN python - <<'PY'
import importlib.util, subprocess, shlex, sys
spec = importlib.util.find_spec("selective_scan_cuda")
if not spec or not getattr(spec, "origin", None):
    print("NOTE: selective_scan_cuda not found (hybrid disabled?)")
    sys.exit(0)
so = spec.origin
print("selective_scan_cuda .so:", so)
out = subprocess.check_output(f"ldd {shlex.quote(so)}", shell=True, text=True, stderr=subprocess.STDOUT)
print(out)
# Allow missing libcuda at build time; require libtorch deps
required = ("libtorch_cuda.so", "libtorch.so", "libc10.so", "libcudart.so")
for line in out.splitlines():
    if "=> not found" in line:
        lib = line.strip().split()[0]
        if lib == "libcuda.so.1":  # runtime provided by host
            continue
        if any(lib.startswith(m) for m in required):
            raise SystemExit(f"Missing required dependency: {lib}")
print("ldd sanity OK")
PY

EXPOSE 8000
CMD ["python3", "main_zonos_tts_api.py"]
