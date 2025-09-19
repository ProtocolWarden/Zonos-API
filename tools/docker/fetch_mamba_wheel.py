#!/usr/bin/env python3
"""Download the prebuilt mamba-ssm wheel with retry and resume support."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
import urllib.request


def _log(message: str) -> None:
    print(message, flush=True)


def _resolve_asset(
    version: str,
    torch_tag: str,
    cuda_tag: str,
    abi_flag: str,
    py_tag: str,
) -> tuple[str, Optional[int], str]:
    """Return the expected wheel name, size (if known), and download URL."""
    release_tag = f"v{version}"
    wheel_name = (
        f"mamba_ssm-{version}+{cuda_tag}torch{torch_tag}cxx11abi{abi_flag}-"
        f"{py_tag}-{py_tag}-linux_x86_64.whl"
    )

    api_url = (
        "https://api.github.com/repos/state-spaces/mamba/releases/tags/"
        f"{release_tag}"
    )

    try:
        with urllib.request.urlopen(api_url, timeout=30) as response:
            release = json.load(response)
    except Exception as exc:  # pragma: no cover - network fallback
        _log(
            "GitHub API lookup failed; falling back to constructed download URL. "
            f"(error: {exc})"
        )
        return (
            wheel_name,
            None,
            "https://github.com/state-spaces/mamba/releases/download/"
            f"{release_tag}/{wheel_name}",
        )

    for asset in release.get("assets", []):
        if asset.get("name") == wheel_name:
            return wheel_name, asset.get("size"), asset.get("browser_download_url")

    _log(
        "Matching asset metadata not found in release response; "
        "falling back to constructed download URL."
    )
    return (
        wheel_name,
        None,
        "https://github.com/state-spaces/mamba/releases/download/"
        f"{release_tag}/{wheel_name}",
    )


def _download_with_resume(
    url: str,
    destination: Path,
    expected_size: Optional[int],
    attempts: int,
    timeout: int,
) -> None:
    """Download a URL to *destination* with retry and resume support."""
    headers = {"User-Agent": "mamba-wheel-downloader/1.0", "Accept": "application/octet-stream"}
    partial_path = destination.with_suffix(destination.suffix + ".part")

    for attempt in range(1, attempts + 1):
        resume_from = partial_path.stat().st_size if partial_path.exists() else 0
        if expected_size is not None and resume_from > expected_size:
            partial_path.unlink(missing_ok=True)
            resume_from = 0

        request_headers = dict(headers)
        mode = "wb"
        if resume_from:
            request_headers["Range"] = f"bytes={resume_from}-"
            mode = "ab"

        request = urllib.request.Request(url, headers=request_headers)

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response, partial_path.open(mode) as handle:
                shutil.copyfileobj(response, handle, length=1 << 20)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:  # pragma: no cover - network failures
            _log(f"Attempt {attempt} failed: {exc}")
        else:
            if expected_size is not None:
                actual_size = partial_path.stat().st_size
                if actual_size == expected_size:
                    partial_path.rename(destination)
                    _log(f"Downloaded {destination.name} ({actual_size} bytes).")
                    return
                _log(
                    "Size mismatch after download: "
                    f"expected {expected_size} bytes, got {actual_size} bytes."
                )
            else:
                partial_path.rename(destination)
                _log(f"Downloaded {destination.name} ({partial_path.stat().st_size} bytes).")
                return

        time.sleep(min(30, 2 ** attempt))

    partial_path.unlink(missing_ok=True)
    raise RuntimeError("Failed to download mamba-ssm wheel after multiple attempts.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="2.2.5", help="mamba-ssm version to download")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/wheels"),
        help="Directory to store the downloaded wheel.",
    )
    parser.add_argument("--attempts", type=int, default=5, help="Maximum download attempts.")
    parser.add_argument("--timeout", type=int, default=600, help="Per-attempt timeout in seconds.")
    parser.add_argument(
        "--force", action="store_true", help="Always redownload even if the wheel already exists."
    )
    args = parser.parse_args()

    import torch

    torch_version = torch.__version__.split("+", 1)[0]
    torch_tag = ".".join(torch_version.split(".")[:2])

    cuda_version = getattr(torch.version, "cuda", None) or ""
    cuda_major = cuda_version.split(".")[0] if cuda_version else ""
    cuda_tag = f"cu{cuda_major}" if cuda_major else "cu"

    abi_flag = "TRUE" if getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", True) else "FALSE"
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"

    wheel_name, expected_size, url = _resolve_asset(
        args.version,
        torch_tag=torch_tag,
        cuda_tag=cuda_tag,
        abi_flag=abi_flag,
        py_tag=py_tag,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / wheel_name

    if destination.exists() and not args.force:
        if expected_size is None or destination.stat().st_size == expected_size:
            _log(f"Wheel already present at {destination}; skipping download.")
            return
        _log("Existing wheel has unexpected size; re-downloading.")
        destination.unlink()

    _download_with_resume(
        url=url,
        destination=destination,
        expected_size=expected_size,
        attempts=args.attempts,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
