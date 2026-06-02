"""
bll/signal_service.py
─────────────────────
Business Logic for signal generation and retrieval.

Computed signals include:
  • Momentum score
  • Relative-Strength (Ticker2Ticker / Portfolio comparison)
  • Sector context
  • SQL-based screens
  • Genetic algorithm screens (stub → extend)
"""

from typing import Generator

import pandas as pd

from dal.signal_dal import SignalDAL
from utils.logger import get_logger

log = get_logger(__name__)


class SignalService:

    def __init__(self, signal_dir: str = "data/signal_momentum"):
        self._dal = SignalDAL(signal_dir=signal_dir)

    def run_signal_generation(
            self,
            config_file: str = "config/tickers.csv",
    ) -> Generator[str, None, None]:
        log.info("Generating signals from config: %s", config_file)
        yield from self._dal.run_signal_generator(config_file)

    def get_latest_signals(self) -> pd.DataFrame | None:
        return self._dal.load_latest()

    def get_signals_for_ticker(self, ticker: str) -> pd.DataFrame | None:
        return self._dal.load_for_ticker(ticker)

    # ── SQL Screen ────────────────────────────────────────────────────────────

    def run_sql_screen(self, query: str) -> pd.DataFrame:
        """
        Execute an arbitrary DuckDB SQL query against the signal parquets.
        Returns a DataFrame.  Extend with persistence / scheduling as needed.
        """
        import duckdb
        signals = self.get_latest_signals()
        if signals is None:
            return pd.DataFrame()
        try:
            con = duckdb.connect(":memory:")
            con.register("signals", signals)
            result = con.execute(query).df()
            con.close()
            return result
        except Exception as exc:
            log.error("SQL screen failed: %s", exc)
            return pd.DataFrame()