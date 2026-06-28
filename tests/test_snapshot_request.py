"""Tests for the post-load snapshot-request helper."""

from __future__ import annotations

import importlib

from zonos import snapshot_request as sr


def test_default_request_dir_is_under_models(monkeypatch):
    monkeypatch.delenv("VF_SNAPSHOT_REQUEST_DIR", raising=False)
    monkeypatch.setenv("ZONOS_DEPLOYMENT_DIR", "/srv/models/tts/zonos")
    importlib.reload(sr)
    d = sr.request_dir()
    assert d.name == ".snapshot-requests"
    # grandparent of the deployment dir == the models root
    assert d.parent.name == "models"


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("VF_SNAPSHOT_REQUEST_DIR", str(tmp_path / "req"))
    importlib.reload(sr)
    assert sr.request_dir() == tmp_path / "req"


def test_request_writes_and_sanitizes(monkeypatch, tmp_path):
    monkeypatch.setenv("VF_SNAPSHOT_REQUEST_DIR", str(tmp_path / "req"))
    importlib.reload(sr)
    assert sr.request_snapshot("zonos/Zyphra__Zonos-v0.1/model.safetensors") is True
    files = list((tmp_path / "req").glob("*.req"))
    assert len(files) == 1
    assert "/" not in files[0].name


def test_failure_is_swallowed(monkeypatch, tmp_path):
    blocker = tmp_path / "afile"
    blocker.write_text("x")
    monkeypatch.setenv("VF_SNAPSHOT_REQUEST_DIR", str(blocker / "req"))
    importlib.reload(sr)
    assert sr.request_snapshot("x") is False
