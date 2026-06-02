"""
utils/logger.py
───────────────
Central logging setup for EquityDesk.
All modules obtain their logger via:  from utils.logger import get_logger
"""

import logging
import sys
from pathlib import Path

_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)

_FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return (or create) a named logger with file + console handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    logger.addHandler(ch)

    # Rotating file handler
    fh = logging.FileHandler(_LOG_DIR / "equitydesk.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    logger.addHandler(fh)

    return logger
