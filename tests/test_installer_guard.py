from __future__ import annotations

import re
from pathlib import Path

FORBIDDEN = "".join([chr(117), chr(118)])
PATTERN = re.compile(rf"\\b{FORBIDDEN}\\b")
TEXT_SUFFIXES = {
    ".cfg",
    ".ini",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".pyi",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
ALWAYS_TEXT = {"Dockerfile", "docker-compose.yml", "LICENSE"}


def test_installer_word_absent() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    offenders: list[Path] = []

    for path in repo_root.rglob("*"):
        if path.is_dir():
            continue
        if path.name.startswith(".") and path.parent.name == ".git":
            continue
        if path.suffix and path.suffix not in TEXT_SUFFIXES:
            continue
        if not path.suffix and path.name not in ALWAYS_TEXT:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if PATTERN.search(text):
            offenders.append(path.relative_to(repo_root))

    assert not offenders, f"found forbidden installer references: {offenders}"
