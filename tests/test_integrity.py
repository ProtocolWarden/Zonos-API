"""Tests for the lightweight cached-file integrity checks."""

from __future__ import annotations

from zonos.integrity import validate_cached_file


def _safetensors(path, header_len: int, total: int):
    """Write a fake .safetensors: 8-byte LE header length + padding to `total`."""
    data = header_len.to_bytes(8, "little") + b"x" * max(0, total - 8)
    path.write_bytes(data)
    return path


def test_empty_file_is_invalid(tmp_path):
    p = tmp_path / "model.safetensors"
    p.write_bytes(b"")
    assert validate_cached_file(p) is False


def test_missing_file_is_invalid(tmp_path):
    assert validate_cached_file(tmp_path / "nope.json") is False


def test_valid_json(tmp_path):
    p = tmp_path / "config.json"
    p.write_text('{"a": 1}')
    assert validate_cached_file(p) is True


def test_corrupt_json(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("{not json")
    assert validate_cached_file(p) is False


def test_valid_safetensors_header(tmp_path):
    # header_len=2 ('{}'), total 10 → 0 < 2 <= 10-8
    p = _safetensors(tmp_path / "model.safetensors", header_len=2, total=10)
    assert validate_cached_file(p) is True


def test_truncated_safetensors_header_overflows_file(tmp_path):
    # header claims 9000 bytes but file is tiny → invalid
    p = _safetensors(tmp_path / "model.safetensors", header_len=9000, total=32)
    assert validate_cached_file(p) is False


def test_zeroed_safetensors_header(tmp_path):
    p = _safetensors(tmp_path / "model.safetensors", header_len=0, total=32)
    assert validate_cached_file(p) is False


def test_other_nonempty_file_ok(tmp_path):
    p = tmp_path / "vocab.txt"
    p.write_text("hello")
    assert validate_cached_file(p) is True
