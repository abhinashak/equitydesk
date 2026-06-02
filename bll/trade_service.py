"""
bll/trade_service.py
─────────────────────
Business Logic for Trade operations:

  • Accounts          – read Kite account / balance info
  • Current Positions – live positions from Kite
  • Target Positions  – compute target from weights + capital
  • Buy/Sell          – phased execution plan (lots, slippage, fee estimate)
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.fees import calc_fees, FeeBreakdown
from utils.logger import get_logger

log = get_logger(__name__)


class TradeService:

    def __init__(
        self,
        kite_base_url: str  = "http://localhost:8080",
        live_orders_file: str = "data/live_orders.json",
        mock_mode: bool = True,
    ):
        self._kite_url    = kite_base_url
        self._orders_file = Path(live_orders_file)
        self._mock        = mock_mode

    # ── Accounts ──────────────────────────────────────────────────────────────

    def get_account_summary(self) -> dict:
        """Return available margin / funds from Kite (or mock)."""
        if self._mock:
            return {"available_cash": 1_000_000.0, "used_margin": 0.0, "source": "mock"}
        # TODO: call Kite API via self._kite_url
        raise NotImplementedError("Live Kite integration pending")

    # ── Current positions ─────────────────────────────────────────────────────

    def get_current_positions(self) -> pd.DataFrame:
        """Return current open positions from Kite (or mock)."""
        if self._mock:
            return pd.DataFrame(columns=["ticker", "qty", "avg_price", "ltp", "pnl"])
        raise NotImplementedError("Live Kite integration pending")

    # ── Target positions ──────────────────────────────────────────────────────

    def compute_target_positions(
        self,
        weights: dict[str, float],
        total_capital: float,
        prices: dict[str, float],
    ) -> pd.DataFrame:
        """
        Given target weights (fractions summing to 1), total capital (₹),
        and current prices, return a target qty DataFrame.
        """
        rows = []
        for ticker, weight in weights.items():
            alloc = total_capital * weight
            price = prices.get(ticker, 0.0)
            qty   = int(alloc / price) if price > 0 else 0
            rows.append({
                "ticker":     ticker,
                "weight":     weight,
                "allocation": alloc,
                "price":      price,
                "target_qty": qty,
            })
        return pd.DataFrame(rows)

    # ── Phased execution plan ─────────────────────────────────────────────────

    def build_execution_plan(
        self,
        current: dict[str, int],   # {ticker: current_qty}
        target:  dict[str, int],   # {ticker: target_qty}
        prices:  dict[str, float],
        phases:  int = 3,
        stt_rate: float = 0.001,
        etc_rate: float = 0.0000325,
        gst_rate: float = 0.18,
        dp_base:  float = 15.34,
    ) -> pd.DataFrame:
        """
        Build a phased buy/sell execution plan.
        Each row = one phase order with fee estimate.
        """
        rows = []
        all_tickers = set(current) | set(target)

        for ticker in sorted(all_tickers):
            curr_qty   = current.get(ticker, 0)
            tgt_qty    = target.get(ticker, 0)
            delta      = tgt_qty - curr_qty
            if delta == 0:
                continue

            side       = "buy" if delta > 0 else "sell"
            phase_qty  = abs(delta) // phases
            remainder  = abs(delta) % phases
            price      = prices.get(ticker, 0.0)

            for ph in range(1, phases + 1):
                qty = phase_qty + (remainder if ph == phases else 0)
                if qty == 0:
                    continue
                value = qty * price
                fees  = calc_fees(value, stt_rate=stt_rate, etc_rate=etc_rate,
                                  gst_rate=gst_rate, dp_base=dp_base, side=side)
                rows.append({
                    "phase":         ph,
                    "ticker":        ticker,
                    "side":          side,
                    "qty":           qty,
                    "price":         price,
                    "value":         value,
                    "fees_total":    fees.total,
                    "fees_stt":      fees.stt,
                    "fees_etc":      fees.etc,
                    "net_value":     value + fees.total if side == "buy" else value - fees.total,
                })

        return pd.DataFrame(rows)

    # ── Order persistence ─────────────────────────────────────────────────────

    def save_orders(self, orders: list[dict]) -> None:
        self._orders_file.parent.mkdir(parents=True, exist_ok=True)
        self._orders_file.write_text(json.dumps(orders, indent=2, default=str))
        log.info("Saved %d orders to %s", len(orders), self._orders_file)

    def load_orders(self) -> list[dict]:
        if not self._orders_file.exists():
            return []
        return json.loads(self._orders_file.read_text())
