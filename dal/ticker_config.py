"""
dal/ticker_config.py
─────────────────────
CRUD helpers for config/tickers.csv, config/sectors_hierarchy.csv
and config/periods.json.  Pure data-access; no business logic.

THIS IS THE SINGLE SOURCE OF TRUTH for the ticker schema.
Everything else (BLL, UI) must reference columns through `TickerCol` /
`HierCol` and the lists below – never string literals – so a schema change
fails loudly (AttributeError / ImportError) instead of silently.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ── Schema: tickers.csv ───────────────────────────────────────────────────────
class TickerCol:
    NAME          = "Name"
    YAHOO         = "Yahoo Symbol"
    SECTOR        = "Sector"
    TICKER        = "ticker"
    SECTOR_GROUP  = "sector_group"
    MACRO_SECTOR  = "macro_sector"
    SEGMENT       = "market_cap"
    MCAP_CR       = "market_cap_cr"
    FII_PCT       = "fii_hold_pct"
    PUBLIC_PCT    = "public_hold_pct"


TICKER_COLS = [
    TickerCol.NAME, TickerCol.YAHOO, TickerCol.SECTOR, TickerCol.TICKER,
    TickerCol.SECTOR_GROUP, TickerCol.MACRO_SECTOR, TickerCol.SEGMENT,
    TickerCol.MCAP_CR, TickerCol.FII_PCT, TickerCol.PUBLIC_PCT,
]
NUM_COLS      = [TickerCol.MCAP_CR, TickerCol.FII_PCT, TickerCol.PUBLIC_PCT]
REQUIRED_COLS = [TickerCol.YAHOO, TickerCol.TICKER]
# Columns filled by enrichment (used for "data completeness" checks)
ENRICH_COLS   = [
    TickerCol.SECTOR_GROUP, TickerCol.MACRO_SECTOR, TickerCol.SEGMENT,
    TickerCol.MCAP_CR, TickerCol.FII_PCT, TickerCol.PUBLIC_PCT,
]

# Allowed market-cap segments (blank / NULL is also allowed)
SEGMENTS = ["smallcap250", "midcap150", "nifty50"]


# ── Schema: sectors_hierarchy.csv ─────────────────────────────────────────────
class HierCol:
    MACRO  = "Macro Sector"
    GROUP  = "Sector Group"
    SECTOR = "Sector"


HIER_COLS = [HierCol.MACRO, HierCol.GROUP, HierCol.SECTOR]


def normalise_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee every schema column exists (in order), numeric cols are floats,
    and blank strings become None. Unknown extra columns are kept at the end."""
    df = df.copy()
    for c in TICKER_COLS:
        if c not in df.columns:
            df[c] = None
    extra = [c for c in df.columns if c not in TICKER_COLS]
    df = df[TICKER_COLS + extra].reset_index(drop=True)
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in TICKER_COLS:
        if c in NUM_COLS:
            continue
        s = df[c].astype("object")
        blank = s.isna() | (s.astype(str).str.strip() == "")
        df[c] = s.where(~blank, None)
    return df


# ── Sector hierarchy (Macro Sector → Sector Group → Sector) ───────────────────
class SectorHierarchy:
    """In-memory view of sectors_hierarchy.csv with lookup helpers."""

    def __init__(self, df: Optional[pd.DataFrame] = None):
        if df is None or df.empty:
            df = pd.DataFrame(columns=HIER_COLS)
        df = df[HIER_COLS].copy()
        for c in HIER_COLS:
            df[c] = df[c].astype("string").str.strip()
        self.df = df.dropna().drop_duplicates().reset_index(drop=True)

    @property
    def empty(self) -> bool:
        return self.df.empty

    @staticmethod
    def _sorted(s: pd.Series) -> list[str]:
        return sorted({str(v) for v in s.dropna()})

    def macros(self) -> list[str]:
        return self._sorted(self.df[HierCol.MACRO])

    def groups(self, macro: Optional[str] = None) -> list[str]:
        d = self.df if not macro else self.df[self.df[HierCol.MACRO] == macro]
        return self._sorted(d[HierCol.GROUP])

    def sectors(self, group: Optional[str] = None, macro: Optional[str] = None) -> list[str]:
        d = self.df
        if macro:
            d = d[d[HierCol.MACRO] == macro]
        if group:
            d = d[d[HierCol.GROUP] == group]
        return self._sorted(d[HierCol.SECTOR])

    def macro_for_group(self, group: Optional[str],
                        prefer: Optional[str] = None) -> Optional[str]:
        """Macro sector that owns `group`. If several do, keep `prefer` when valid."""
        if not group:
            return None
        found = self._sorted(self.df.loc[self.df[HierCol.GROUP] == group, HierCol.MACRO])
        if not found:
            return None
        return prefer if prefer in found else found[0]

    def group_macro_for_sector(self, sector: Optional[str],
                               prefer_group: Optional[str] = None
                               ) -> tuple[Optional[str], Optional[str]]:
        """(sector_group, macro_sector) for `sector`; (None, None) if unknown.
        If the sector appears under several groups, keep `prefer_group` when valid."""
        if not sector:
            return None, None
        d = self.df[self.df[HierCol.SECTOR] == sector]
        if d.empty:
            return None, None
        if prefer_group and (d[HierCol.GROUP] == prefer_group).any():
            d = d[d[HierCol.GROUP] == prefer_group]
        row = d.iloc[0]
        return str(row[HierCol.GROUP]), str(row[HierCol.MACRO])


# ── DAL ───────────────────────────────────────────────────────────────────────
class TickerConfigDAL:
    """Reads and writes the ticker list, sector hierarchy and period definitions."""

    def __init__(
            self,
            ticker_path: str = "config/tickers.csv",
            periods_path: str = "config/periods.json",
            hierarchy_path: str = "config/sectors_hierarchy.csv",
    ):
        self.ticker_path    = Path(ticker_path)
        self.periods_path   = Path(periods_path)
        self.hierarchy_path = Path(hierarchy_path)
        self.ticker_path.parent.mkdir(parents=True, exist_ok=True)

        # Seed defaults if files don't exist (hierarchy is user-supplied, never seeded)
        if not self.ticker_path.exists():
            _seed_tickers(self.ticker_path)
        if not self.periods_path.exists():
            _seed_periods(self.periods_path)

    # ── Sector hierarchy ──────────────────────────────────────────────────────

    def load_sector_hierarchy(self) -> SectorHierarchy:
        if not self.hierarchy_path.exists():
            log.warning("Sector hierarchy file not found: %s", self.hierarchy_path)
            return SectorHierarchy()
        df = pd.read_csv(self.hierarchy_path)
        missing = [c for c in HIER_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{self.hierarchy_path} is missing column(s): {', '.join(missing)}"
            )
        return SectorHierarchy(df)

    # ── Tickers ───────────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self.ticker_path)
        return normalise_tickers(df)[TICKER_COLS]

    def save(self, df: pd.DataFrame) -> None:
        normalise_tickers(df)[TICKER_COLS].to_csv(self.ticker_path, index=False)
        log.debug("Saved %d tickers to %s", len(df), self.ticker_path)

    def add_ticker(self, row: dict) -> None:
        unknown = set(row) - set(TICKER_COLS)
        if unknown:
            raise ValueError(f"Unknown ticker column(s): {sorted(unknown)}")
        df = self.load()
        sym = str(row.get(TickerCol.TICKER, "")).strip().upper()
        if not sym:
            raise ValueError("ticker is required")
        if sym in set(df[TickerCol.TICKER].dropna().astype(str).str.upper()):
            raise ValueError(f"ticker {sym} already exists")
        new_row = pd.DataFrame([{c: row.get(c) for c in TICKER_COLS}])
        df = pd.concat([df, new_row], ignore_index=True)
        self.save(df)
        log.info("Added ticker: %s", sym)

    def delete_ticker(self, ticker: str) -> None:
        df = self.load()
        df = df[df[TickerCol.TICKER] != ticker]
        self.save(df)
        log.info("Deleted ticker: %s", ticker)

    def update_ticker(self, ticker: str, col: str, value) -> None:
        if col not in TICKER_COLS:
            raise ValueError(f"Unknown ticker column: {col}")
        df = self.load()
        df.loc[df[TickerCol.TICKER] == ticker, col] = value
        self.save(df)
        log.debug("Updated %s.%s = %s", ticker, col, value)

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
    C = TickerCol
    sample = [
        {C.NAME: "NSEI", C.YAHOO: "^NSEI", C.TICKER: "^NSEI"},
        {C.NAME: "Bharat Electronics", C.YAHOO: "BEL.NS", C.SECTOR: "Aerospace & Defense",
         C.TICKER: "BEL", C.SECTOR_GROUP: "Aerospace & Defense",
         C.MACRO_SECTOR: "Capital Goods & Industrials", C.SEGMENT: "nifty50"},
    ]
    normalise_tickers(pd.DataFrame(sample))[TICKER_COLS].to_csv(path, index=False)
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