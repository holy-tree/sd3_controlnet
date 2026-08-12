"""Checkpoint provenance helpers used to bind candidates to the DPO reference."""

from __future__ import annotations

import hashlib
from pathlib import Path


def checkpoint_checksum(path: str | Path) -> str:
    """Hash checkpoint configs and weight files in deterministic path order."""
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    files = [root] if root.is_file() else sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file()
        and (candidate.name == "config.json" or candidate.suffix in {".safetensors", ".bin"})
    )
    if not files:
        raise FileNotFoundError(f"No checkpoint weights/config found in {root}")
    digest = hashlib.sha256()
    for file_path in files:
        relative = file_path.name if root.is_file() else file_path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()
