"""Private local artifacts; no writes within a Git checkout."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


class InputError(ValueError):
    """An explicitly safe validation message for CLI display."""


def private_path(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    if any((parent / ".git").exists() for parent in (target, *target.parents)):
        raise InputError("Run data must be outside every Git checkout")
    return target


def new_run(path: str | Path) -> Path:
    target = private_path(path)
    target.mkdir(parents=True, exist_ok=False, mode=0o700)
    target.chmod(0o700)
    return target


def write_json(path: Path, value: object) -> None:
    target = private_path(path)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with target.open("x", encoding="utf-8") as handle:
        os.chmod(target, 0o600)
        handle.write(data)


def read_json(path: str | Path) -> dict:
    def reject_constant(_):
        raise InputError("Nonfinite JSON")

    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle, parse_constant=reject_constant)
    if not isinstance(data, dict):
        raise InputError("Expected a JSON object")
    return data


def relative_asset(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise InputError("Assets must use nonempty relative paths")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise InputError("Asset missing or outside the input directory")
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
