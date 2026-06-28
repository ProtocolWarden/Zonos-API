import errno
import math
import os
import shutil
from pathlib import Path

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from transformers.models.dac import DacModel

_DAC_REPO_ID = "descript/dac_44khz"


def _deployment_base() -> Path:
    return Path(
        os.environ.get("ZONOS_DEPLOYMENT_DIR", "/app/models/tts/zonos")
    )


def _hf_hub_cache() -> Path:
    return Path(
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or (Path.home() / ".cache" / "huggingface" / "hub")
    )


def _cleanup_hf_cache(repo_id: str, source_path: Path) -> None:
    cache_root = _hf_hub_cache()
    try:
        if not cache_root.exists():
            return
        if cache_root in source_path.parents:
            lock_dir = cache_root / ".locks" / f"models--{repo_id.replace('/', '--')}"
            lock_file = lock_dir / f"{source_path.name}.lock"
            lock_file.unlink(missing_ok=True)
    except Exception:
        pass

def _move_to_deployment(repo_id: str, filename: str, source_path: str) -> Path:
    dest_dir = _deployment_base() / repo_id.replace("/", "__")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    source = Path(source_path)
    if dest_path.exists():
        dest_path.unlink()
    if source.is_symlink():
        link_path = source
        source = source.resolve()
    else:
        link_path = None

    try:
        os.replace(source, dest_path)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copy2(source, dest_path)
        os.unlink(source)

    if link_path is not None:
        try:
            link_path.unlink()
        except Exception:
            pass

    _cleanup_hf_cache(repo_id, source)
    # No chown: metadata churn on a Syncthing-synced file is itself a change
    # event that can spawn sync-conflicts. The atomic os.replace above is the
    # publish; ownership is left as-is (matches the echogarden model-publish
    # discipline in VideoFoundry).
    # A real download just landed in models/tts/zonos — signal the host snapshot
    # responder (best-effort; never breaks model loading).
    try:
        from .snapshot_request import request_snapshot

        request_snapshot(f"zonos-{repo_id.replace('/', '__')}-{filename}")
    except Exception:
        pass
    return dest_path


def _validate_cached_file(path: Path) -> bool:
    if not path.exists():
        return False
    if path.suffix == ".json":
        try:
            with path.open("r", encoding="utf-8") as handle:
                handle.read(1)
            return True
        except Exception:
            return False
    if path.suffix == ".safetensors":
        # Let DacModel validate content when loading; presence is enough here.
        return True
    return True


def _download_with_deployment_move(
    repo_id: str, filename: str, revision: str | None = None
) -> str:
    dest_dir = _deployment_base() / repo_id.replace("/", "__")
    dest_path = dest_dir / filename
    if _validate_cached_file(dest_path):
        return str(dest_path)
    if dest_path.exists():
        try:
            dest_path.unlink()
        except Exception:
            pass

    try:
        cache_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_files_only=True,
        )
        if _validate_cached_file(Path(cache_path)):
            return str(_move_to_deployment(repo_id, filename, cache_path))
        try:
            Path(cache_path).unlink()
        except Exception:
            pass
    except Exception:
        pass

    cache_path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
    if not _validate_cached_file(Path(cache_path)):
        raise RuntimeError(f"Downloaded {filename} failed validation.")
    return str(_move_to_deployment(repo_id, filename, cache_path))


class DACAutoencoder:
    def __init__(self):
        super().__init__()
        config_path = _download_with_deployment_move(_DAC_REPO_ID, "config.json")
        model_path = _download_with_deployment_move(_DAC_REPO_ID, "model.safetensors")
        self.dac = DacModel.from_pretrained(Path(config_path).parent)
        self.dac.eval().requires_grad_(False)
        self.codebook_size = self.dac.config.codebook_size
        self.num_codebooks = self.dac.quantizer.n_codebooks
        self.sampling_rate = self.dac.config.sampling_rate

    def preprocess(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        wav = torchaudio.functional.resample(wav, sr, 44_100)
        right_pad = math.ceil(wav.shape[-1] / 512) * 512 - wav.shape[-1]
        return torch.nn.functional.pad(wav, (0, right_pad))

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        return self.dac.encode(wav).audio_codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            self.dac.device.type, torch.float16, enabled=self.dac.device.type != "cpu"
        ):
            return self.dac.decode(audio_codes=codes).audio_values.unsqueeze(1).float()
