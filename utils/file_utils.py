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


def save_json_atomic(data: dict | list, path: str, base_file: str | None = None) -> None:
    """Atomically write `data` (a JSON object or array) to `path` via a sibling temp file + `os.replace`."""
    full_path = resolve_relative_path(path, base_file=base_file)
    parent = os.path.dirname(full_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{full_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, full_path)


def load_json_list(path: str, base_file: str | None = None) -> list:
    """Load a JSON array from `path`, returning `[]` if missing, empty, or invalid."""
    full_path = resolve_relative_path(path, base_file=base_file)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def raise_fd_limit(want: int = 4096) -> None:
    """
    Raise the soft open-file limit (macOS defaults to 256, which a pipeline that reads
    many parquet files can exceed). A no-op on Windows or restricted environments.
    """
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard != resource.RLIM_INFINITY:
            want = min(want, hard)
        if soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    except Exception:
        pass


def find_first_existing(names: list[str], dirs: tuple[str, ...] | list[str] = (".",),
                        base_dir: str | None = None) -> str | None:
    """
    Return the first existing file among `dirs` x `names` (directories searched in order,
    then names), or None. Relative `dirs` resolve against `base_dir` (default: cwd).
    """
    base = base_dir or os.getcwd()
    for d in dirs:
        for n in names:
            candidate = os.path.normpath(os.path.join(base, d, n))
            if os.path.isfile(candidate):
                return candidate
    return None


def read_text_file(path: str, base_file: str | None = None, default: str = "") -> str:
    """Read a UTF-8 text file; return `default` if it is missing or unreadable."""
    full_path = resolve_relative_path(path, base_file=base_file)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return default