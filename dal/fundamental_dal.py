"""
dal/fundamental_dal.py
──────────────────────
Raw data-access for fundamental data:
  • load_html_from_upload() – accept a caller-supplied HTML path (browser download)
  • parse_html()            – invoke screener_parser.py → staging parquet
  • incremental_merge()     – period-union merge into per-ticker subfolders

Storage layout:
  data/fundamental/
    <TICKER>/
      general_info.parquet        ← always overwritten
      quarterly_results.parquet   ← period-union (old ∪ new periods)
      profit_loss.parquet
      ... (all other tables)
"""

import shutil
from pathlib import Path

import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


TABLES = [
    "general_info", "pros", "cons",
    "quarterly_results", "profit_loss", "balance_sheet", "cash_flows",
    "ratios", "compounded_sales_growth", "compounded_profit_growth",
    "stock_price_cagr", "return_on_equity",
    "shareholding_quarterly", "shareholding_yearly",
]

# Keys used to identify a unique row per table.
TABLE_KEYS = {
    "general_info":              ["ticker"],
    "pros":                      ["ticker", "item"],
    "cons":                      ["ticker", "item"],
    "quarterly_results":         ["ticker", "period", "metric"],
    "profit_loss":               ["ticker", "period", "metric"],
    "balance_sheet":             ["ticker", "period", "metric"],
    "cash_flows":                ["ticker", "period", "metric"],
    "ratios":                    ["ticker", "period", "metric"],
    "compounded_sales_growth":   ["ticker", "horizon"],
    "compounded_profit_growth":  ["ticker", "horizon"],
    "stock_price_cagr":          ["ticker", "horizon"],
    "return_on_equity":          ["ticker", "horizon"],
    "shareholding_quarterly":    ["ticker", "period", "category"],
    "shareholding_yearly":       ["ticker", "period", "category"],
}

# Tables whose rows slide forward each quarter.
PERIOD_UNION_TABLES = {
    "quarterly_results", "profit_loss", "balance_sheet", "cash_flows",
    "ratios", "shareholding_quarterly", "shareholding_yearly",
}

# Tables that are simply overwritten each run
OVERWRITE_TABLES = {"general_info"}


class FundamentalDAL:
    """Low-level fundamental data I/O — per-ticker subfolder layout."""

    def __init__(
            self,
            out_dir: str = "data/fundamental",
            parser_script: str = "loader/screener_parser.py",
            sleep_secs: int = 0,          # no longer used for downloads; kept for API compat
            cookies: dict[str, str] | None = None,  # kept for API compat; unused
    ):
        self.out_dir       = Path(out_dir)
        self.parser_script = parser_script
        self.sleep_secs    = sleep_secs
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _ticker_dir(self, ticker: str) -> Path:
        """Return (and create) the per-ticker subfolder."""
        d = self.out_dir / ticker
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── Step 1: Accept uploaded HTML ──────────────────────────────────────────

    def load_html_from_upload(
            self,
            ticker: str,
            uploaded_path: Path,
    ) -> tuple[Path, int] | tuple[None, int]:
        """
        Copy a caller-supplied HTML file (e.g. a Streamlit UploadedFile saved
        to a temp path) into /tmp/<TICKER>.html and validate it's large enough
        to be a real Screener page.

        Returns (html_path, bytes) on success or (None, bytes) on failure.
        """
        html_path = Path(f"/tmp/{ticker}.html")
        try:
            shutil.copy2(uploaded_path, html_path)
        except Exception as exc:
            log.warning("Failed to copy upload for %s: %s", ticker, exc)
            return None, 0

        size = html_path.stat().st_size
        if size < 5_000:
            log.warning(
                "Suspiciously small HTML for %s (%d bytes) — likely not a valid page",
                ticker, size,
            )
            return None, size

        log.info("Accepted uploaded HTML for %s (%d bytes)", ticker, size)
        return html_path, size

    # ── Step 2: Parse ─────────────────────────────────────────────────────────

    def parse_html(self, ticker: str, html_path: Path) -> Path | None:
        """Run screener_parser.py → staging parquet dir. Returns dir or None."""
        import subprocess
        staging_dir = Path(f"/tmp/fundamental_staging/{ticker}")
        staging_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["python", self.parser_script, str(html_path), "--out-dir", str(staging_dir)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            log.error("screener_parser failed for %s:\n%s", ticker, result.stderr)
            return None
        return staging_dir

    # ── Step 3: Incremental merge ─────────────────────────────────────────────

    def incremental_merge(self, ticker: str, staging_dir: Path) -> dict[str, tuple[int, int]]:
        """
        Merge staging parquets into the per-ticker subfolder.

        Strategies by table type:
          OVERWRITE_TABLES    → replace master entirely with new data
          PERIOD_UNION_TABLES → keep old rows for periods not in new data,
                                then append all new rows  (sliding-window safe)
          everything else     → LEFT JOIN dedup on composite key (append missing keys)

        Returns {table: (inserted, skipped)}.
        """
        ticker_dir = self._ticker_dir(ticker)
        stats: dict[str, tuple[int, int]] = {}

        for table in TABLES:
            staging_file = staging_dir / f"{table}.parquet"
            if not staging_file.exists():
                continue

            new_df = pd.read_parquet(staging_file)
            if new_df.empty:
                continue

            master_file = ticker_dir / f"{table}.parquet"

            # ── Overwrite (general_info) ──────────────────────────────────────
            if table in OVERWRITE_TABLES:
                keys = TABLE_KEYS.get(table, ["ticker"])
                new_df = new_df.drop_duplicates(subset=keys, keep="last")
                new_df["last_updated"] = pd.Timestamp.utcnow().floor("s")
                new_df.to_parquet(master_file, index=False)
                stats[table] = (len(new_df), 0)
                log.debug("%s/%s: overwritten (%d rows)", ticker, table, len(new_df))
                continue

            # ── First write ───────────────────────────────────────────────────
            if not master_file.exists():
                new_df.to_parquet(master_file, index=False)
                stats[table] = (len(new_df), 0)
                log.debug("%s/%s: new file, wrote %d rows", ticker, table, len(new_df))
                continue

            old_df = pd.read_parquet(master_file)

            # ── Period-union (sliding time-series tables) ─────────────────────
            if table in PERIOD_UNION_TABLES:
                keys = TABLE_KEYS.get(table, ["ticker"])
                if "period" in new_df.columns and "period" in old_df.columns:
                    new_periods = set(new_df["period"].unique())
                    old_unique  = old_df[~old_df["period"].isin(new_periods)]
                    merged      = pd.concat([old_unique, new_df], ignore_index=True)
                    merged      = merged.drop_duplicates(subset=keys, keep="last")
                    kept_periods = len(old_unique)
                else:
                    merged       = pd.concat([old_df, new_df], ignore_index=True)
                    merged       = merged.drop_duplicates(subset=keys, keep="last")
                    old_unique   = old_df
                    kept_periods = len(old_unique)

                inserted = len(merged) - len(old_df)
                skipped  = len(new_df) - max(inserted, 0)
                merged.to_parquet(master_file, index=False)
                stats[table] = (max(inserted, 0), max(skipped, 0))
                log.debug(
                    "%s/%s: period-union → %d rows total (+%d, %d old-period rows retained)",
                    ticker, table, len(merged), max(inserted, 0), kept_periods,
                )
                continue

            # ── Key dedup (pros, cons, growth tables, etc.) ───────────────────
            keys = TABLE_KEYS.get(table, ["ticker"])
            merged_check = new_df.merge(
                old_df[keys].drop_duplicates(),
                on=keys,
                how="left",
                indicator=True,
            )
            delta = new_df[merged_check["_merge"] == "left_only"].reset_index(drop=True)

            if delta.empty:
                stats[table] = (0, len(new_df))
                log.debug("%s/%s: key-dedup — no new rows", ticker, table)
                continue

            merged = pd.concat([old_df, delta], ignore_index=True)
            merged.to_parquet(master_file, index=False)
            stats[table] = (len(delta), len(new_df) - len(delta))
            log.debug("%s/%s: key-dedup +%d new, %d skipped",
                      ticker, table, len(delta), len(new_df) - len(delta))

        return stats

    # ── Coverage query ────────────────────────────────────────────────────────

    def get_coverage(self) -> pd.DataFrame:
        """Walk per-ticker subfolders and summarise coverage per table."""
        from collections import defaultdict
        table_tickers: dict[str, int] = defaultdict(int)
        table_rows:    dict[str, int] = defaultdict(int)

        for ticker_dir in sorted(self.out_dir.iterdir()):
            if not ticker_dir.is_dir():
                continue
            for table in TABLES:
                f = ticker_dir / f"{table}.parquet"
                if f.exists():
                    try:
                        df = pd.read_parquet(f, columns=["ticker"])
                        table_tickers[table] += 1
                        table_rows[table]    += len(df)
                    except Exception:
                        pass

        return pd.DataFrame([
            {"table": t, "tickers": table_tickers[t], "rows": table_rows[t]}
            for t in TABLES
        ])

    # ── Stale / missing query ─────────────────────────────────────────────────

    REQUIRED_TABLES = [
        "quarterly_results", "profit_loss", "balance_sheet", "cash_flows",
        "ratios", "compounded_sales_growth", "compounded_profit_growth",
        "stock_price_cagr", "return_on_equity",
        "shareholding_quarterly", "shareholding_yearly",
    ]

    def get_stale_or_missing(
            self,
            all_tickers: list[str],
            stale_hours: float | None = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of tickers that are stale or have missing tables.

        Columns:
          ticker        – screener symbol
          last_updated  – UTC timestamp from general_info (NaT if never loaded)
          hours_ago     – float, hours since last update  (NaN if never loaded)
          missing       – comma-separated list of absent required tables ('' if none)
          reason        – 'never loaded' | 'stale' | 'missing tables' | 'stale + missing tables'
        """
        rows = []
        for ticker in all_tickers:
            ticker_dir = self.out_dir / ticker

            gi_file = ticker_dir / "general_info.parquet"
            last_updated = pd.NaT
            if gi_file.exists():
                try:
                    gi = pd.read_parquet(gi_file, columns=["last_updated"])
                    if "last_updated" in gi.columns and not gi.empty:
                        last_updated = pd.to_datetime(gi["last_updated"].iloc[0], utc=True)
                except Exception:
                    pass

            now = pd.Timestamp.utcnow()
            hours_ago = (
                (now - last_updated).total_seconds() / 3600
                if pd.notna(last_updated) else float("nan")
            )

            missing = [
                t for t in self.REQUIRED_TABLES
                if not (ticker_dir / f"{t}.parquet").exists()
            ]

            is_never    = pd.isna(last_updated)
            is_stale    = (not is_never) and (stale_hours is not None) and (hours_ago > stale_hours)
            has_missing = bool(missing)

            if not (is_never or is_stale or has_missing):
                continue

            if is_never:
                reason = "never loaded"
            elif is_stale and has_missing:
                reason = "stale + missing tables"
            elif is_stale:
                reason = "stale"
            else:
                reason = "missing tables"

            rows.append({
                "ticker":       ticker,
                "last_updated": last_updated,
                "hours_ago":    round(hours_ago, 1) if not is_never else None,
                "missing":      ", ".join(missing),
                "reason":       reason,
            })

        return pd.DataFrame(rows, columns=["ticker", "last_updated", "hours_ago", "missing", "reason"])