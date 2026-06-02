"""
bll/ticker_eval_service.py
──────────────────────────
Business Logic for Ticker Evaluation:

  • Gates         – pass/fail criteria (fundamental + price filters)
  • Ticker2Ticker – compare two tickers on a shared set of metrics
  • Ticker vs Portfolio – how a candidate fits a live portfolio
"""

from typing import Optional
import pandas as pd

from dal.fundamental_dal import FundamentalDAL, TABLES
from dal.ticker_dal import TickerDAL
from utils.logger import get_logger

log = get_logger(__name__)


class TickerEvalService:

    def __init__(
        self,
        fundamental_dir: str = "data/fundamental",
        ticker_data_root: str = "data/ticker",
    ):
        self._fund_dal   = FundamentalDAL(out_dir=fundamental_dir)
        self._ticker_dal = TickerDAL(data_root=ticker_data_root)

    # ── Gates ─────────────────────────────────────────────────────────────────

    def evaluate_gates(self, ticker: str, gates: dict) -> dict:
        """
        Apply pass/fail gates to a ticker.
        gates = {
            "min_roe": 15.0,
            "min_revenue_growth_3y": 10.0,
            "max_debt_equity": 1.0,
            ...
        }
        Returns {gate_name: {"pass": bool, "value": ...}}
        Stub: populate from fundamental parquets once screener_parser is live.
        """
        log.info("Evaluating gates for %s: %s", ticker, list(gates.keys()))
        results = {}
        for gate, threshold in gates.items():
            # TODO: fetch actual metric values from FundamentalDAL
            results[gate] = {"pass": None, "value": None, "threshold": threshold}
        return results

    # ── Ticker2Ticker ─────────────────────────────────────────────────────────

    def compare_tickers(
        self,
        ticker_a: str,
        ticker_b: str,
        metrics: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Side-by-side comparison of two tickers on fundamental + price metrics.
        Stub: implement with FundamentalDAL.get_coverage() + OHLCV data.
        """
        log.info("Comparing %s vs %s", ticker_a, ticker_b)
        default_metrics = ["roe", "revenue_growth", "debt_equity", "pe", "momentum_1m"]
        cols = metrics or default_metrics
        return pd.DataFrame({
            "metric":  cols,
            ticker_a: [None] * len(cols),
            ticker_b: [None] * len(cols),
        })

    # ── Ticker vs Portfolio ───────────────────────────────────────────────────

    def compare_ticker_to_portfolio(
        self,
        ticker: str,
        portfolio: list[str],
    ) -> pd.DataFrame:
        """
        How does a candidate ticker rank against an existing portfolio?
        Returns ranking table.
        """
        log.info("Comparing %s to portfolio of %d tickers", ticker, len(portfolio))
        # TODO: load signals, compute rank
        return pd.DataFrame({"status": ["stub — implementation pending"]})
