"""
file_utils.py
─────────────
Generic file path resolution and atomic JSON persistence helpers.
"""

import json
import os


def resolve_relative_path(path: str, base_file: str | None = None) -> str:
    """Resolve `path` if relative; if `base_file` is given, resolve against its directory."""
    if os.path.isabs(path):
        return os.path.normpath(path)
    base_dir = os.path.dirname(os.path.abspath(base_file)) if base_file else os.getcwd()
    return os.path.normpath(os.path.join(base_dir, path))


def load_json_dict(path: str, base_file: str | None = None) -> dict:
    """Load a JSON object from `path`, returning `{}` if missing, empty, or invalid."""
    full_path = resolve_relative_path(path, base_file=base_file)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_json_atomic(data: dict, path: str, base_file: str | None = None) -> None:
    """Atomically write `data` to `path` via a sibling temp file + `os.replace`."""
    full_path = resolve_relative_path(path, base_file=base_file)
    parent = os.path.dirname(full_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{full_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, full_path)
