"""UI label lookup backed by config/label_utils.csv."""
from functools import lru_cache
from pathlib import Path
from utils.db_utils import load_csv_table

_LABELS_PATH = "config/label_utils.csv"

@lru_cache(maxsize=1)
def _labels() -> dict[str, str]:
    df = load_csv_table(str(_LABELS_PATH))
    if not {"key", "message"}.issubset(df.columns):
        raise ValueError("label_utils.csv must contain key,message columns")
    return dict(zip(df["key"].astype(str), df["message"].astype(str)))

def label(key: str, **values) -> str:
    """Return a configured UI message, falling back to the key if missing."""
    template = _labels().get(key, key)
    try:
        return template.format(**values)
    except (KeyError, ValueError):
        return template
