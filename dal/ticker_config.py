"""
dal/ticker_config.py
─────────────────────
CRUD helpers for config/tickers.csv and config/periods.json.
Pure data-access; no business logic.
"""

import json
import pandas as pd
from pathlib import Path

from utils.logger import get_logger

log = get_logger(__name__)

TICKER_COLS = [
    "Name", "Yahoo Symbol", "Sector", "market_cap",
    "domestic_market_pct", "num_clients", "num_sectors_served",
]


class TickerConfigDAL:
    """Reads and writes the ticker list and period definitions."""

    def __init__(
        self,
        ticker_path: str = "config/tickers.csv",
        periods_path: str = "config/periods.json",
    ):
        self.ticker_path  = Path(ticker_path)
        self.periods_path = Path(periods_path)
        self.ticker_path.parent.mkdir(parents=True, exist_ok=True)

        # Seed defaults if files don't exist
        if not self.ticker_path.exists():
            _seed_tickers(self.ticker_path)
        if not self.periods_path.exists():
            _seed_periods(self.periods_path)

    # ── Tickers ───────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.ticker_path)
        for col in TICKER_COLS:
            if col not in df.columns:
                df[col] = None
        return df[TICKER_COLS]

    def save(self, df: pd.DataFrame) -> None:
        df.to_csv(self.ticker_path, index=False)
        log.debug("Saved %d tickers to %s", len(df), self.ticker_path)

    def add_ticker(self, row: dict) -> None:
        df = self.load()
        new_row = pd.DataFrame([{c: row.get(c, "") for c in TICKER_COLS}])
        df = pd.concat([df, new_row], ignore_index=True)
        self.save(df)
        log.info("Added ticker: %s", row.get("Name"))

    def delete_ticker(self, name: str) -> None:
        df = self.load()
        df = df[df["Name"] != name]
        self.save(df)
        log.info("Deleted ticker: %s", name)

    def update_ticker(self, name: str, col: str, value) -> None:
        df = self.load()
        df.loc[df["Name"] == name, col] = value
        self.save(df)
        log.debug("Updated %s.%s = %s", name, col, value)

    # ── Periods ───────────────────────────────────────────────────────────────

    def load_periods(self) -> list:
        return json.loads(self.periods_path.read_text())

    def save_periods(self, periods: list) -> None:
        self.periods_path.write_text(json.dumps(periods, indent=2))
        log.debug("Saved %d periods to %s", len(periods), self.periods_path)

    def add_period(self, period: dict) -> None:
        periods = self.load_periods()
        periods.append(period)
        self.save_periods(periods)
        log.info("Added period: %s", period.get("name"))

    def delete_period(self, name: str) -> None:
        periods = [p for p in self.load_periods() if p["name"] != name]
        self.save_periods(periods)
        log.info("Deleted period: %s", name)

    def toggle_period(self, name: str) -> None:
        periods = self.load_periods()
        for p in periods:
            if p["name"] == name:
                p["enabled"] = 1 - p.get("enabled", 1)
        self.save_periods(periods)
        log.info("Toggled period: %s", name)


# ── Seed helpers ──────────────────────────────────────────────────────────────

def _seed_tickers(path: Path) -> None:
    sample = [
        ["ASTRAMICRO",  "ASTRAMICRO.NS",  "Defence_ETF", "Small-cap", 95.0, 1, 2],
        ["BDL",         "BDL.NS",         "Defence_ETF", "Small-cap", 95.0, 1, 1],
        ["BEL",         "BEL.NS",         "Defence_ETF", "Large-cap", 95.0, 2, 3],
        ["DATAPATTNS",  "DATAPATTNS.NS",  "Defence_ETF", "Mid-cap",  100.0, 1, 2],
    ]
    pd.DataFrame(sample, columns=TICKER_COLS).to_csv(path, index=False)
    log.info("Seeded default tickers at %s", path)


def _seed_periods(path: Path) -> None:
    sample = [
        {
            "name":    "EVT-Iran-War",
            "enabled": 1,
            "train":   {"start": "2017-04-01", "end": "2025-12-31", "type": "fy"},
            "test":    {"start": "2026-01-01", "end": "2026-05-07", "type": "crash"},
        }
    ]
    path.write_text(json.dumps(sample, indent=2))
    log.info("Seeded default periods at %s", path)
