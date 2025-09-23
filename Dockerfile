# ========================================================
# syntax=docker/dockerfile:1.4
# ========================================================

# Pin uv version globally (must also be re-declared after each FROM if needed)
ARG UV_VERSION=0.8.19

# ========================================================
# Stage 0 — Build CUDA wheels against pinned torch
# MUST use the *devel* variant: compiles CUDA/C++ extensions (nvcc, headers).
# Update digest with tools/docker/update_pytorch_digest.sh when refreshing base image
# ========================================================
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel AS mamba-builder
ARG UV_VERSION
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124 \
    PYPI_INDEX_URL=https://pypi.org/simple \
    TORCH_CUDA_ARCH_LIST="8.6;8.9+PTX"

# Force CUDA build (no CPU fallback) for native extensions
ENV CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1

WORKDIR /tmp/mamba

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

# --- uv install (pin uv binary version) ---------------------------------------
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-builder-03-install \
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh && \
    ln -sf /root/.local/bin/uv /usr/local/bin/uv && \
    uv --version && \
    apt-get purge -y curl && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- pip bootstrap (for building wheels) --------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip,id=pip-cache-zonos-builder-04-bootstrap \
    python -m pip install --upgrade pip setuptools wheel

# --- Install pinned torch/torchaudio (prebuilt wheels via uv) -----------------
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-builder-05-torch \
    uv pip install --system --no-cache-dir \
      --index-url ${TORCH_CUDA_INDEX_URL} \
      --extra-index-url ${PYPI_INDEX_URL} \
      torch==2.4.1+cu124 \
      torchaudio==2.4.1+cu124

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
      --index-url ${TORCH_CUDA_INDEX_URL} \
      --extra-index-url ${PYPI_INDEX_URL} \
      --no-binary=mamba-ssm \
      --wheel-dir /tmp/wheels \
      mamba-ssm==2.2.4

# Verify the built wheel actually contains the CUDA .so
RUN python - <<'PY'
import glob, zipfile, sys
wheels = sorted(glob.glob('/tmp/wheels/mamba_ssm-*.whl'))
assert wheels, "mamba-ssm wheel not found in /tmp/wheels"
wheel = wheels[-1]
print("Built wheel:", wheel)
with zipfile.ZipFile(wheel) as zf:
    so = [n for n in zf.namelist() if n.endswith('.so') and 'selective_scan_cuda' in n]
    print("selective_scan_cuda entries:", so)
    if not so:
        sys.exit("ERROR: selective_scan_cuda*.so missing from mamba-ssm wheel")
print("OK: mamba-ssm wheel contains selective_scan_cuda")
PY

RUN --mount=type=cache,target=/root/.cache/pip,id=pip-cache-zonos-builder-07-flashcausal \
    PIP_NO_BUILD_ISOLATION=1 \
    python -m pip wheel \
      --no-deps \
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
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime AS base
ARG UV_VERSION
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu124 \
    PYPI_INDEX_URL=https://pypi.org/simple
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9+PTX"

LABEL built-by="Ctrl+C Ctrl+V DevOps - Thanks Chat" \
      purpose="API container that yells in beautiful voices"

WORKDIR /app

# Bring lock + metadata so uv export can materialize requirement sets on demand.
COPY uv.lock pyproject.toml ./

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
    # Install uv at a pinned version
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh; \
    ln -sf /root/.local/bin/uv /usr/local/bin/uv; \
    uv --version; \
    \
    # Remove curl (keep ca-certificates for TLS).
    apt-get purge -y curl; \
    apt-get autoremove -y; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-base-05-torch \
    uv pip install --system --no-cache-dir \
      --index-url ${TORCH_CUDA_INDEX_URL} \
      --extra-index-url ${PYPI_INDEX_URL} \
      torch==2.4.1+cu124 \
      torchaudio==2.4.1+cu124; \
    python -m pip uninstall -y torchvision || true

COPY --from=mamba-builder /tmp/torch_build.json /tmp/torch_build.json
RUN python - <<'PY'
import json, torch, pathlib

b = json.loads(pathlib.Path('/tmp/torch_build.json').read_text())
cur = {"torch": torch.__version__, "cuda": torch.version.cuda}
print("Runtime Torch:", cur)
assert cur == b, f"Builder/runtime Torch mismatch: {b} != {cur}"
PY

# refresh lock for Python 3.11 + Linux
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-base-05a-uv-lock \
    uv lock --python 3.11

# Export the lock to a pip-readable requirements file (from existing uv.lock)
# NOTE: --frozen and --locked are mutually exclusive. Use --locked to rely strictly on uv.lock.
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-base-05b-uv-export \
    uv export --locked --format requirements-txt > /tmp/runtime.lock.txt

# Export compile extra requirements for wheel installs.
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-base-05c-uv-export-compile \
    uv export --locked -E compile --format requirements-txt > /tmp/compile.lock.txt

# Install normal runtime deps pinned by the exported lock file
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-base-06-reqs \
    uv pip install --system --no-cache-dir \
      -r /tmp/runtime.lock.txt \
      --index-url ${TORCH_CUDA_INDEX_URL} \
      --extra-index-url ${PYPI_INDEX_URL} \
  && python -m pip check

COPY --from=mamba-builder /tmp/wheels /tmp/wheels
RUN python - <<'PY'
import glob, zipfile
for name in ("flash_attn-*.whl","causal_conv1d-*.whl"):
    wh = sorted(glob.glob(f"/tmp/wheels/{name}"))
    if not wh:
        raise SystemExit(f"Missing wheel for {name}")
    print(name, "->", wh[-1])
    with zipfile.ZipFile(wh[-1]) as zf:
        sos = [n for n in zf.namelist() if n.endswith('.so')]
        if not sos:
            raise SystemExit(f"No .so files detected in {wh[-1]}")
        print("  .so entries:", sos[:5], "... total:", len(sos))
PY
RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache-zonos-base-07-localwheels \
    uv pip install --system --no-cache-dir \
      -c /tmp/runtime.lock.txt \
      --index-url ${TORCH_CUDA_INDEX_URL} \
      --extra-index-url ${PYPI_INDEX_URL} \
      --find-links=/tmp/wheels \
      -r /tmp/compile.lock.txt \
  && python -m pip check

RUN rm -rf /tmp/wheels || true
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
import importlib.util, importlib.metadata, torch

print('Torch:', torch.__version__, torch.version.cuda)

# Check presence WITHOUT importing (avoids Triton driver init)
for mod in ("selective_scan_cuda", "mamba_ssm"):
    spec = importlib.util.find_spec(mod)
    print(mod, "found:", bool(spec), "origin:", getattr(spec, "origin", None))

# Also sanity print versions from metadata (no import)
try:
    print("mamba-ssm version:", importlib.metadata.version("mamba-ssm"))
except importlib.metadata.PackageNotFoundError:
    print("mamba-ssm not installed?")
PY

# CUDA dependency ldd smoke-test: ensure the extension resolves core deps
RUN python - <<'PY'
import importlib.util, sys, subprocess, shlex


def find_so(modname: str) -> str | None:
    spec = importlib.util.find_spec(modname)
    return getattr(spec, "origin", None) if spec else None


so_path = find_so("selective_scan_cuda")
if not so_path:
    print("FATAL: selective_scan_cuda .so not found on sys.path")
    sys.exit(1)

print("selective_scan_cuda candidate:", so_path)
cmd = f"ldd {shlex.quote(so_path)}"
out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
print(out)

# Allow missing libcuda.so.1 at build-time; require core libs to resolve
required_ok = True
must_have = ("libtorch_cuda.so", "libtorch.so", "libc10.so", "libcudart.so")
for line in out.splitlines():
    # format:    libname => /path (0x...)  OR libname => not found
    if "=> not found" in line:
        lib = line.strip().split()[0]
        if lib == "libcuda.so.1":
            continue  # expected at build-time
        if any(lib.startswith(m) for m in must_have):
            print(f"FATAL: required dependency missing: {lib}")
            required_ok = False

if not required_ok:
    sys.exit(1)
print("ldd sanity: OK (core deps resolved; libcuda will be provided at runtime)")
PY

# ========================================================
# Stage 2 — Runtime layer
# ========================================================
FROM base AS runtime

WORKDIR /app
COPY . ./

EXPOSE 8000

CMD ["python3", "main_zonos_tts_api.py"]
