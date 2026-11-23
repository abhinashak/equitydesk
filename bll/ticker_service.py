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

import numpy as np
import pandas as pd

from dal.ticker_dal import TickerDAL
from dal.ticker_config import TickerConfigDAL
from utils.logger import get_logger

log = get_logger(__name__)

_DATE_COLS = ("date", "Date", "datetime", "Datetime", "timestamp", "Timestamp")


class TickerService:

    def __init__(self, data_root: str = "data/ticker"):
        self._dal = TickerDAL(data_root=data_root)
        self._cfg_dal: TickerConfigDAL | None = None

    @property
    def _cfg(self) -> TickerConfigDAL:
        # Lazy: constructing TickerConfigDAL seeds config/tickers.csv if missing,
        # which a benchmark service should never trigger just by existing.
        if self._cfg_dal is None:
            self._cfg_dal = TickerConfigDAL()
        return self._cfg_dal

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
        yield from self._preflight(config_file)
        yield from self._dal.run_historical_loader(config_file, check_rebase)

    def run_live(
            self,
            config_file: str = "config/tickers.csv",
    ) -> Generator[str, None, None]:
        log.info("Starting live loader: config=%s", config_file)
        yield from self._preflight(config_file)
        yield from self._dal.run_live_loader(config_file)

    def _preflight(self, config_file: str) -> Generator[str, None, None]:
        """Surface config problems in the loader log before any download starts."""
        for issue in self.get_config_issues(config_file):
            yield f"⚠ CONFIG: {issue}"

    # ── Config checks ─────────────────────────────────────────────────────────

    def get_config_issues(self, config_file: str) -> list[str]:
        """Human-readable problems in a ticker/benchmark config CSV (never raises)."""
        try:
            df = pd.read_csv(config_file)
        except Exception as e:
            return [f"cannot read {config_file}: {type(e).__name__}: {e}"]

        issues: list[str] = []
        if "Name" not in df.columns:
            return [f"'Name' column missing; found columns {list(df.columns)}"]

        names = df["Name"].astype(str).str.strip()
        dup_names = sorted(set(names[names.duplicated(keep=False)]))
        if dup_names:
            issues.append(f"duplicate Name: {dup_names}")

        if "Yahoo Symbol" in df.columns:
            sym = df["Yahoo Symbol"].astype(str).str.strip()
            for s in sorted(set(sym[sym.duplicated(keep=False)])):
                who = names[sym == s].tolist()
                issues.append(f"Yahoo Symbol '{s}' is shared by {who} (same series downloaded twice)")
            if df["Yahoo Symbol"].isna().any():
                issues.append("some rows have an empty Yahoo Symbol")
        else:
            issues.append("'Yahoo Symbol' column missing")
        return issues

    # ── Ticker list helpers ───────────────────────────────────────────────────

    def get_ticker_names(self, config_file: str | None = None) -> list[str]:
        """Symbol names. Pass the benchmark config for benchmark services;
        without it the default ticker config is used (old behaviour)."""
        df = pd.read_csv(config_file) if config_file else self._cfg.load()
        if "Name" not in df.columns:
            raise ValueError(
                f"'Name' column missing in {config_file or 'default ticker config'}; "
                f"found columns: {list(df.columns)}"
            )
        names = [n for n in df["Name"].dropna().astype(str).str.strip() if n]
        names = list(dict.fromkeys(names))          # de-duplicate, keep order
        if not names:
            raise ValueError(f"No symbols found in {config_file or 'default ticker config'}")
        log.info("%d symbols loaded from %s", len(names), config_file or "default config")
        return names

    def load_ohlcv(self, ticker: str) -> pd.DataFrame | None:
        return self._dal.load_ticker(ticker)

    # ── Quality ───────────────────────────────────────────────────────────────

    def get_last_dates(self, config_file: str | None = None) -> dict:
        """{symbol: last candle Timestamp (normalized) or None if missing/unreadable}."""
        out: dict = {}
        for name in self.get_ticker_names(config_file):
            try:
                df = self.load_ohlcv(name)
                if df is None or len(df) == 0:
                    out[name] = None
                    continue
                col = next((c for c in _DATE_COLS if c in df.columns), None)
                s = df[col] if col else df.index.to_series()
                s = pd.to_datetime(s, errors="coerce").dropna()
                out[name] = s.max().normalize() if len(s) else None
            except Exception as e:
                log.warning("last-date read failed for %s: %s", name, e)
                out[name] = None
        return out

    def get_stale_report(self, config_file: str | None = None,
                         tolerance_bdays: int = 0) -> dict:
        """Symbols whose last candle is behind the newest candle in the set.

        Returns {"reference": Timestamp|None, "total": int, "stale": DataFrame}.
        """
        last = self.get_last_dates(config_file)
        valid = {k: v for k, v in last.items() if v is not None}
        if not valid:
            return {"reference": None, "total": len(last), "stale": pd.DataFrame()}

        ref = max(valid.values())
        rows = []
        for sym, d in last.items():
            if d is None:
                rows.append({"Symbol": sym, "Last candle": "missing / unreadable",
                             "Business days behind": None, "Calendar days behind": None})
                continue
            bd = int(np.busday_count(d.date(), ref.date()))
            if bd > tolerance_bdays:
                rows.append({"Symbol": sym, "Last candle": d.date(),
                             "Business days behind": bd,
                             "Calendar days behind": (ref - d).days})
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("Business days behind", ascending=False,
                                na_position="first").reset_index(drop=True)
        return {"reference": ref, "total": len(last), "stale": df}