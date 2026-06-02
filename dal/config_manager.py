"""
dal/config_manager.py
─────────────────────
Read / write config/app_config.py.
Sections detected by:  # ── Section Name  comment lines.
"""

import re
from pathlib import Path
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)

_DEFAULT_CONFIG = """\
# ── Kite API
KITE_BASE_URL = "http://localhost:8080"

# ── File paths
DATA_DIR              = "data"
TICKER_DATA_DIR       = "data/ticker"
FUNDAMENTAL_DATA_DIR  = "data/fundamental"
SIGNAL_DATA_DIR       = "data/signal_momentum"
EXCLUSIONS_FILE       = "data/excluded_symbols.json"
WEIGHTS_FILE          = "data/target_weights.txt"
LIVE_ORDERS_FILE      = "data/live_orders.json"

# ── Runtime flags
MOCK_MODE = "true"

# ── Fee rates (India equity CNC)
STT_RATE   = 0.001
ETC_RATE   = 0.0000325
SEBI_RATE  = 0.000001
GST_RATE   = 0.18
DP_BASE    = 15.34
"""


class ConfigManager:
    """
    Data-Access layer for app_config.py.
    Reads, writes, and parses the key=value config file.
    """

    def __init__(self, path: str = "config/app_config.py"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(_DEFAULT_CONFIG)
            log.info("Created default config at %s", self.path)

    # ── Raw I/O ───────────────────────────────────────────────────────────────

    def read_raw(self) -> str:
        return self.path.read_text()

    def write_raw(self, text: str) -> None:
        self.path.write_text(text)
        log.debug("Raw config written to %s", self.path)

    # ── Structured access ─────────────────────────────────────────────────────

    def parse(self) -> list[dict]:
        """
        Returns a list of typed entries:
          {"type": "section", "label": "Kite API"}
          {"type": "entry",   "key": "KITE_BASE_URL", "value": "http://...", "raw_line": "..."}
          {"type": "blank",   "raw_line": ""}
          {"type": "comment", "raw_line": "# some comment"}
        """
        entries = []
        for line in self.path.read_text().splitlines():
            stripped = line.strip()
            if stripped == "":
                entries.append({"type": "blank", "raw_line": line})
                continue
            if re.match(r"^#\s*[─—–]+\s*\S", stripped):
                label = re.sub(r"^#\s*[─—–]+\s*", "", stripped).strip()
                entries.append({"type": "section", "label": label, "raw_line": line})
                continue
            if stripped.startswith("#"):
                entries.append({"type": "comment", "raw_line": line})
                continue
            m = re.match(r'^([A-Z_][A-Z0-9_]*)\s*=\s*(.+)$', stripped)
            if m:
                entries.append({
                    "type":      "entry",
                    "key":       m.group(1),
                    "value":     m.group(2).strip().strip('"'),
                    "raw_value": m.group(2).strip(),
                    "raw_line":  line,
                })
                continue
            entries.append({"type": "comment", "raw_line": line})
        return entries

    def set_value(self, key: str, value: Any) -> None:
        """Update a single key in-place, preserving everything else."""
        text = self.path.read_text()
        try:
            float(str(value))
            new_val = str(value)
        except ValueError:
            new_val = f'"{value}"'
        pattern = rf'^({re.escape(key)}\s*=\s*)(.+)$'
        new_text, n = re.subn(pattern, rf'\g<1>{new_val}', text, flags=re.MULTILINE)
        if n == 0:
            new_text = text.rstrip() + f'\n{key} = {new_val}\n'
        self.path.write_text(new_text)
        log.debug("Config: set %s = %s", key, new_val)

    def as_dict(self) -> dict:
        return {e["key"]: e["value"] for e in self.parse() if e["type"] == "entry"}
