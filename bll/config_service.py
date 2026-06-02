"""
bll/config_service.py
─────────────────────
Business Logic wrapper for configuration management.
Exposes a clean API over ConfigManager and TickerConfigDAL.
"""

from typing import Any

import pandas as pd

from dal.config_manager import ConfigManager
from dal.ticker_config import TickerConfigDAL
from utils.logger import get_logger

log = get_logger(__name__)


class ConfigService:

    def __init__(
        self,
        app_config_path: str = "config/app_config.py",
        ticker_path: str     = "config/tickers.csv",
        periods_path: str    = "config/periods.json",
    ):
        self._cfg     = ConfigManager(path=app_config_path)
        self._ticker  = TickerConfigDAL(ticker_path=ticker_path, periods_path=periods_path)

    # ── App config ────────────────────────────────────────────────────────────

    def get_app_config(self) -> dict:
        return self._cfg.as_dict()

    def get_app_config_entries(self) -> list[dict]:
        return self._cfg.parse()

    def set_app_value(self, key: str, value: Any) -> None:
        self._cfg.set_value(key, value)

    def read_raw_config(self) -> str:
        return self._cfg.read_raw()

    def write_raw_config(self, text: str) -> None:
        self._cfg.write_raw(text)

    # ── Ticker list ───────────────────────────────────────────────────────────

    def get_tickers(self) -> pd.DataFrame:
        return self._ticker.load()

    def save_tickers(self, df: pd.DataFrame) -> None:
        self._ticker.save(df)

    def add_ticker(self, row: dict) -> None:
        self._ticker.add_ticker(row)

    def delete_ticker(self, name: str) -> None:
        self._ticker.delete_ticker(name)

    # ── Periods ───────────────────────────────────────────────────────────────

    def get_periods(self) -> list:
        return self._ticker.load_periods()

    def add_period(self, period: dict) -> None:
        self._ticker.add_period(period)

    def delete_period(self, name: str) -> None:
        self._ticker.delete_period(name)

    def toggle_period(self, name: str) -> None:
        self._ticker.toggle_period(name)
