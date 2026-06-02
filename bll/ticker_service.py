"""
bll/ticker_service.py
─────────────────────
Business Logic for ticker (OHLCV) data management.

The same service class handles both equity tickers and benchmark indices —
they use identical loaders, just different config files and data roots.

  Equity tickers:  TickerService(data_root="data/ticker")
  Benchmarks:      TickerService(data_root="data/benchmark")
"""

from typing import Generator

import pandas as pd

from dal.ticker_dal import TickerDAL
from dal.ticker_config import TickerConfigDAL
from utils.logger import get_logger

log = get_logger(__name__)


class TickerService:

    def __init__(self, data_root: str = "data/ticker"):
        self._dal = TickerDAL(data_root=data_root)
        self._cfg = TickerConfigDAL()

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return self._dal.stats()

    # ── Loaders (streamed) ────────────────────────────────────────────────────

    def run_historical(
            self,
            config_file: str = "config/tickers.csv",
            check_rebase: bool = False,
    ) -> Generator[str, None, None]:
        log.info("Starting historical loader: config=%s rebase=%s", config_file, check_rebase)
        yield from self._dal.run_historical_loader(config_file, check_rebase)

    def run_live(
            self,
            config_file: str = "config/tickers.csv",
    ) -> Generator[str, None, None]:
        log.info("Starting live loader: config=%s", config_file)
        yield from self._dal.run_live_loader(config_file)

    # ── Ticker list helpers ───────────────────────────────────────────────────

    def get_ticker_names(self) -> list[str]:
        return self._cfg.load()["Name"].dropna().tolist()

    def load_ohlcv(self, ticker: str) -> pd.DataFrame | None:
        return self._dal.load_ticker(ticker)