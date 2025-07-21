import sys
from pathlib import Path
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Stub heavy optional dependencies before importing the API module
if "huggingface_hub" not in sys.modules:
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda *a, **k: ""
    sys.modules["huggingface_hub"] = hub

for mod in ["torchaudio", "safetensors", "inflect", "kanjize", "phonemizer.backend", "sudachipy"]:
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)

if "torchaudio" in sys.modules:
    sys.modules["torchaudio"].save = lambda *a, **k: None
    sys.modules["torchaudio"].load = lambda *a, **k: (torch.zeros(1, 16000), 16000)

if "kanjize" in sys.modules:
    sys.modules["kanjize"].number2kanji = lambda x: x

if "phonemizer.backend" in sys.modules:
    sys.modules["phonemizer.backend"].EspeakBackend = object

if "sudachipy" in sys.modules:
    sud = sys.modules["sudachipy"]
    class DummyDict:
        def __init__(self, *a, **k): pass
        def create(self):
            class T: pass
            return T()
    sud.Dictionary = DummyDict
    sud.SplitMode = object

if "inflect" in sys.modules:
    sys.modules["inflect"].engine = lambda: None

if "transformers" not in sys.modules:
    tr = types.ModuleType("transformers")
    models = types.ModuleType("transformers.models")
    dac = types.ModuleType("transformers.models.dac")
    dac.DacModel = object
    sys.modules["transformers.models"] = models
    sys.modules["transformers.models.dac"] = dac
    tr.models = models
    sys.modules["transformers"] = tr

import torch
from fastapi.testclient import TestClient
import main_zonos_tts_api as api

class FakeAutoencoder:
    sampling_rate = 22050
    def decode(self, codes):
        return torch.zeros(1, 160)

class FakeModel:
    def __init__(self):
        self.autoencoder = FakeAutoencoder()
        self.last_cond = None
    def prepare_conditioning(self, cond_dict):
        self.last_cond = cond_dict
        return cond_dict
    def generate(self, **kwargs):
        return torch.zeros(1, 1, dtype=torch.long)

def create_client(monkeypatch):
    fake = FakeModel()
    api.MODELS["transformer"] = fake
    api.MODELS["hybrid"] = fake
    monkeypatch.setattr(api, "load_models", lambda: None)
    monkeypatch.setattr(api, "load_voice_embeddings", lambda: None)
    monkeypatch.setattr(api, "make_cond_dict", lambda **kw: kw)
    return TestClient(api.app), fake

def test_advanced_params(monkeypatch):
    client, model = create_client(monkeypatch)
    payload = {
        "input": "hi",
        "pitch_std": 60.0,
        "speaking_rate": 22.0,
        "fmax": 24000,
        "vqscore_8": [0.78]*8,
        "dnsmos_ovrl": 4.0,
        "speaker_noised": True
    }
    response = client.post("/v1/audio/speech", json=payload)
    assert response.status_code == 200
    assert model.last_cond["pitch_std"] == 60.0
    assert model.last_cond["speaking_rate"] == 22.0
    assert model.last_cond["fmax"] == 24000
    assert model.last_cond["speaker_noised"] is True

def test_defaults(monkeypatch):
    client, model = create_client(monkeypatch)
    response = client.post("/v1/audio/speech", json={"input": "hi"})
    assert response.status_code == 200
    assert model.last_cond["speaking_rate"] == 15.0
