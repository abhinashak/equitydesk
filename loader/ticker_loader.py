import polars as pl
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import os
import time
import argparse


class NSEInstitutionalPipeline:
    BATCH_SIZE = 10          # Max tickers per yf.download() call
    STALE_DAYS = 30          # Gap threshold: >30 days → needs full refresh via batch

    def __init__(self, config_file="./config/tickers.csv", data_root="./data_ticker", check_rebase=False):
        self.config_file = config_file
        self.data_root = data_root
        self.tz = pytz.timezone("Asia/Kolkata")
        self.max_years = 10
        self.failed_downloads = []
        self.check_rebase = check_rebase

        cfg_dir = os.path.dirname(self.config_file)
        if cfg_dir:
            os.makedirs(cfg_dir, exist_ok=True)
        os.makedirs(self.data_root, exist_ok=True)
        self._output_handler = None   # optional extra sink (e.g. queue.Queue.put)

    # ── Output tee ────────────────────────────────────────────────────────────

    def set_output_handler(self, fn):
        """
        Register a callable that receives every log line in addition to stdout.
        Useful for streaming output into a Streamlit queue without subprocess.
        Call with fn=None to remove the handler.
        """
        self._output_handler = fn

    def _emit(self, *args, **kwargs):
        """Drop-in replacement for print() that tees to the registered handler."""
        import io
        buf = io.StringIO()
        print(*args, file=buf, **kwargs)
        line = buf.getvalue().rstrip("\n")
        print(line)                          # always goes to real stdout
        if self._output_handler is not None:
            try:
                self._output_handler(line)
            except Exception:
                pass

    # ─────────────────────────────────────────────
    # Config & date helpers
    # ─────────────────────────────────────────────

    def load_and_dedup_tickers(self):
        """Loads tickers from config, resolves symbols, and removes duplicates."""
        if not os.path.exists(self.config_file):
            self._emit(f"Config file not found. Creating sample at {self.config_file}")
            sample_df = pl.DataFrame({
                "Name": ["RELIANCE", "INFY", "TCS"],
                "Yahoo Symbol": ["RELIANCE.NS", "INFY.NS", None]
            })
            sample_df.write_csv(self.config_file)

        df = pl.read_csv(self.config_file)
        df = df.with_columns(
            pl.coalesce([
                pl.col("Yahoo Symbol"),
                pl.col("Name") + ".NS"
            ]).alias("resolved_ticker")
        )
        return df.select("resolved_ticker").unique().to_series().to_list()

    def get_market_end_date(self):
        """NSE closes at 3:30 PM IST. Returns the last actual trading date."""
        now = datetime.now(self.tz)
        yesterday = now - timedelta(days=1)
        start = (yesterday - timedelta(days=7)).date()
        end = now.date()
        try:
            df = yf.download("^NSEI", start=start, end=end, interval="1d",
                             auto_adjust=True, progress=False)
            if df.empty:
                self._emit("⚠️ ^NSEI returned empty, falling back to yesterday")
                return yesterday.date()
            df = df.reset_index()
            last_dt = pd.to_datetime(df["Date"]).max().date()
            self._emit(f"📅 Last NSE trading date (from ^NSEI): {last_dt}")
            return last_dt
        except Exception as e:
            self._emit(f"❌ Error fetching ^NSEI: {e}")
            return yesterday.date()

    # ─────────────────────────────────────────────
    # Classification: batch vs incremental
    # ─────────────────────────────────────────────
    def classify_tickers(self, tickers, end_date):
        from collections import defaultdict

        stale_cutoff = end_date - timedelta(days=self.STALE_DAYS)

        # Only scan the 1-2 year partitions that could contain recent records
        years_to_scan = set()
        years_to_scan.add(end_date.year)
        years_to_scan.add(stale_cutoff.year)

        # Build glob patterns for only those partitions
        partition_paths = [
            f"{self.data_root}/year={yr}/data.parquet"
            for yr in years_to_scan
            if os.path.exists(f"{self.data_root}/year={yr}/data.parquet")
        ]

        # Single scan: get latest record per ticker within the window
        recent_records = {}  # ticker -> (last_dt, last_close)

        if partition_paths:
            df = (
                pl.scan_parquet(partition_paths)
                .filter(pl.col("Date") > stale_cutoff)
                .filter(pl.col("Ticker").is_in(tickers))
                .sort("Date", descending=True)
                .group_by("Ticker")
                .agg([
                    pl.col("Date").first().alias("last_dt"),
                    pl.col("Close").first().alias("last_close"),
                ])
                .collect()
            )

            for row in df.iter_rows(named=True):
                recent_records[row["Ticker"]] = (row["last_dt"], row["last_close"])

        # Now classify
        batch_needed = []
        incremental_groups = defaultdict(list)

        for ticker in tickers:
            last_record = recent_records.get(ticker)
            if last_record is None:
                batch_needed.append(ticker)
                continue
            last_dt, last_close = last_record
            if (end_date - last_dt).days > self.STALE_DAYS:
                batch_needed.append(ticker)
            else:
                incremental_groups[last_dt].append((ticker, last_close))

        n_incremental = sum(len(v) for v in incremental_groups.values())
        self._emit(f"\n📊 Classification complete → "
                   f"{len(batch_needed)} need full refresh (sequential), "
                   f"{n_incremental} need incremental sync "
                   f"({len(incremental_groups)} distinct last_dt group(s))\n")
        return batch_needed, dict(incremental_groups)


    def _clean_batch_slice(self, df_pd, ticker):
        """
        Cleans a single-ticker slice from a batch download result.
        Handles both plain and MultiIndex columns (single-ticker batch edge case).
        Adds Ticker column and casts Date to pl.Date.
        """
        if df_pd.empty:
            return None

        # Flatten any residual MultiIndex (single-ticker batch edge case)
        if isinstance(df_pd.columns, pd.MultiIndex):
            df_pd.columns = df_pd.columns.get_level_values(0)

        df_pd = df_pd.reset_index()
        df_pd.columns = [str(c) for c in df_pd.columns]

        df_pl = pl.from_pandas(df_pd)
        df_pl = df_pl.drop_nulls(subset=["Open", "High", "Low", "Close"])

        if df_pl.is_empty():
            return None

        return df_pl.with_columns([
            pl.lit(ticker).alias("Ticker"),
            pl.col("Date").cast(pl.Date),
        ])

    # ─────────────────────────────────────────────
    # Incremental sync (per-ticker, small delta)
    # ─────────────────────────────────────────────

    def clean_yf_dataframe(self, df_pd):
        """Standardizes a single-ticker yfinance download and drops Null price rows."""
        if df_pd.empty:
            return None

        if isinstance(df_pd.columns, pd.MultiIndex):
            df_pd.columns = df_pd.columns.get_level_values(0)

        df_pd = df_pd.reset_index()
        df_pd.columns = [str(c) for c in df_pd.columns]

        df_pl = pl.from_pandas(df_pd)
        df_pl = df_pl.drop_nulls(subset=["Open", "High", "Low", "Close"])
        return df_pl

    def get_ticker_last_record(self, ticker):
        """Scans partitioned store to find the latest date and closing price."""
        try:
            df = (
                pl.scan_parquet(f"{self.data_root}/year=*/data.parquet")
                .filter(pl.col("Ticker") == ticker)
                .sort("Date", descending=True)
                .limit(1)
                .collect()
            )
            if df.is_empty():
                return None, None
            return df["Date"][0], df["Close"][0]
        except Exception:
            return None, None

    def full_refresh(self, ticker, end_date):
        """
        Single-ticker full refresh (used during incremental sync when a rebase/split
        is detected on overlap).  Still rate-limited with a 5 s sleep.
        """
        self._emit(f"⚠️  Full Refresh: {ticker}. Downloading {self.max_years} years, cooling off 5s...")
        time.sleep(5)
        start_date = end_date - timedelta(days=365 * self.max_years)

        try:
            df_pd = yf.download(ticker, start=start_date, end=end_date,
                                auto_adjust=True, progress=False)
            new_df = self.clean_yf_dataframe(df_pd)

            if new_df is not None:
                new_df = new_df.with_columns([
                    pl.lit(ticker).alias("Ticker"),
                    pl.col("Date").cast(pl.Date),
                ])
                self._write_to_partitions(new_df, purge_ticker=ticker)
            else:
                self.failed_downloads.append({"Ticker": ticker, "Error": "Empty Dataframe returned"})
        except Exception as e:
            self._emit(f"❌ Error during full refresh for {ticker}: {e}")
            self.failed_downloads.append({"Ticker": ticker, "Error": str(e)})

    def sync_ticker(self, ticker, end_date):
        """
        Single-ticker incremental sync — kept as a fallback/safety net.
        The main pipeline uses batch_incremental_sync instead.
        """
        last_dt, last_close = self.get_ticker_last_record(ticker)

        if last_dt is None:
            self.full_refresh(ticker, end_date)
            return

        self._emit(f"[DEBUG] check_rebase={self.check_rebase} | last_dt={last_dt} | end_date={end_date}")
        if not self.check_rebase and last_dt >= end_date:
            self._emit(f"⏭️  Skipping {ticker} (already up to date, check_rebase=OFF)")
            return

        self._emit(f"⏳ Syncing: {ticker}. Cooling off 5s...")
        time.sleep(5)
        check_start = last_dt - timedelta(days=1)

        try:
            df_pd = yf.download(ticker, start=check_start, end=end_date,
                                auto_adjust=True, progress=False)
            new_data = self.clean_yf_dataframe(df_pd)

            if new_data is None:
                return

            new_data = new_data.with_columns([
                pl.lit(ticker).alias("Ticker"),
                pl.col("Date").cast(pl.Date),
            ])

            overlap_row = new_data.filter(pl.col("Date") == last_dt)
            if not overlap_row.is_empty():
                new_close = overlap_row["Close"][0]
                if abs(new_close - last_close) > 0.01:
                    self._emit(f"🔄 Price mismatch on {last_dt} for {ticker} "
                               f"(stored={last_close:.4f}, fetched={new_close:.4f}) — triggering full refresh")
                    self.full_refresh(ticker, end_date)
                    return

            delta = new_data.filter(pl.col("Date") > last_dt)
            if not delta.is_empty():
                self._emit(f"✅ Appending {len(delta)} new rows for {ticker}")
                self._write_to_partitions(delta)
            else:
                self._emit(f"⏭️  {ticker} already up to date.")

        except Exception as e:
            self._emit(f"❌ Error syncing {ticker}: {e}")
            self.failed_downloads.append({"Ticker": ticker, "Error": str(e)})

    def batch_incremental_sync(self, incremental_groups, end_date):
        """
        Batched incremental sync.  Tickers that share the same last_dt are
        downloaded together in a single yf.download() call, eliminating the
        per-ticker 5 s sleep that made the old sequential loop so slow.

        For each last_dt group
        ──────────────────────
        1. Fetch  (last_dt − 1 day) → end_date  for the whole group at once.
        2. For each ticker in the group:
             a. Extract its slice from the MultiIndex result.
             b. Check the overlap row (last_dt) for a price mismatch → rebase.
             c. If rebase detected  → fall back to single-ticker full_refresh.
             d. Otherwise           → append the delta rows (Date > last_dt).

        Parameters
        ----------
        incremental_groups : dict[date, list[tuple[str, float]]]
            Output of classify_tickers: maps last_dt → [(ticker, last_close), …]
        end_date : date
        """
        if not incremental_groups:
            return

        total_tickers = sum(len(v) for v in incremental_groups.values())
        self._emit(f"🔄 Batch incremental sync — {total_tickers} tickers "
                   f"across {len(incremental_groups)} date group(s)...")

        all_deltas = []   # accumulate every ticker's delta; write once at the end

        for last_dt, ticker_records in sorted(incremental_groups.items()):
            tickers   = [t for t, _ in ticker_records]
            closes    = {t: c for t, c in ticker_records}

            # Skip entire group when already up-to-date and rebase is off
            if not self.check_rebase and last_dt >= end_date:
                self._emit(f"⏭️  Group last_dt={last_dt}: all {len(tickers)} tickers "
                           f"already up to date — skipping.")
                continue

            check_start = last_dt - timedelta(days=1)
            self._emit(f"\n📦 Incremental group last_dt={last_dt} "
                       f"— {len(tickers)} tickers  (fetch {check_start} → {end_date})")

            # ── Download entire group in BATCH_SIZE chunks ────────────────────
            for i in range(0, len(tickers), self.BATCH_SIZE):
                batch   = tickers[i: i + self.BATCH_SIZE]
                self._emit(f"   ↳ sub-batch [{i+1}–{i+len(batch)}/{len(tickers)}]")

                try:
                    df_pd = yf.download(
                        batch,
                        start=check_start,
                        end=end_date,
                        auto_adjust=True,
                        group_by="ticker",
                        threads=False,
                        progress=False,
                    )
                except Exception as e:
                    self._emit(f"❌ Batch incremental download failed: {e}")
                    for t in batch:
                        self.failed_downloads.append({"Ticker": t, "Error": f"Batch failed: {e}"})
                    continue

                if df_pd.empty:
                    self._emit("⚠️ Batch returned empty DataFrame.")
                    for t in batch:
                        self.failed_downloads.append({"Ticker": t, "Error": "Empty batch result"})
                    continue

                # ── Single-ticker edge case ───────────────────────────────────
                # yfinance still returns a MultiIndex (ticker, field) even for a
                # 1-element list with group_by='ticker'.  Extract the slice exactly
                # like the multi-ticker path so _clean_batch_slice sees flat columns.
                if len(batch) == 1:
                    ticker = batch[0]
                    if (isinstance(df_pd.columns, pd.MultiIndex)
                            and ticker in df_pd.columns.get_level_values(0)):
                        slice_pd = df_pd[ticker].copy()
                    else:
                        slice_pd = df_pd   # already flat (older yfinance versions)
                    delta = self._process_incremental_slice(
                        slice_pd, ticker, last_dt, closes[ticker], end_date
                    )
                    if delta is not None:
                        all_deltas.append(delta)
                    continue

                # ── Multi-ticker: extract per-ticker slice and process ────────
                available = set(df_pd.columns.get_level_values(0))
                for ticker in batch:
                    if ticker not in available:
                        self._emit(f"⚠️  {ticker} missing from batch response.")
                        self.failed_downloads.append({"Ticker": ticker, "Error": "Not in batch response"})
                        continue
                    delta = self._process_incremental_slice(
                        df_pd[ticker].copy(), ticker, last_dt, closes[ticker], end_date
                    )
                    if delta is not None:
                        all_deltas.append(delta)

        # ── Single bulk write for all accumulated deltas ──────────────────────
        if all_deltas:
            combined = pl.concat(all_deltas, how="diagonal_relaxed")
            self._emit(f"\n💾 Writing {len(combined)} total new rows across "
                       f"{combined['Ticker'].n_unique()} tickers...")
            self._write_to_partitions(combined)
        else:
            self._emit("\n⏭️  All incremental tickers already up to date — nothing to write.")

    def _process_incremental_slice(self, df_pd, ticker, last_dt, last_close, end_date):
        """
        Validates one ticker's slice from a batch incremental download.

        Performs the overlap/rebase check on last_dt:
          - Rebase detected → triggers full_refresh immediately, returns None.
          - No new rows     → returns None.
          - New rows found  → returns the delta DataFrame (caller accumulates
                              and writes in one shot).
        """
        df_pl = self._clean_batch_slice(df_pd, ticker)
        if df_pl is None:
            self._emit(f"⚠️  No valid rows for {ticker} after cleaning.")
            self.failed_downloads.append({"Ticker": ticker, "Error": "Empty after cleaning"})
            return

        # ── Rebase / split detection ──────────────────────────────────────────
        overlap_row = df_pl.filter(pl.col("Date") == last_dt)
        if not overlap_row.is_empty():
            new_close = overlap_row["Close"][0]
            if abs(new_close - last_close) > 0.01:
                self._emit(f"🔄 Rebase detected for {ticker} on {last_dt} "
                           f"(stored={last_close:.4f}, fetched={new_close:.4f}) — full refresh")
                self.full_refresh(ticker, end_date)
                return

        # ── Return delta for bulk write ───────────────────────────────────────
        delta = df_pl.filter(pl.col("Date") > last_dt)
        if not delta.is_empty():
            self._emit(f"✅ {ticker}: {len(delta)} new row(s) queued")
            return delta

        self._emit(f"⏭️  {ticker}: already up to date.")
        return None

    # ─────────────────────────────────────────────
    # Storage
    # ─────────────────────────────────────────────

    def _write_to_partitions(self, df, purge_ticker=None):
        """Saves data year-partitioned to ./data_ticker/year=YYYY/data.parquet"""
        df = df.with_columns(pl.col("Date").dt.year().alias("Year"))

        for yr in df["Year"].unique().to_list():
            year_dir = os.path.join(self.data_root, f"year={yr}")
            os.makedirs(year_dir, exist_ok=True)
            file_path = os.path.join(year_dir, "data.parquet")

            chunk = df.filter(pl.col("Year") == yr).drop("Year")

            if os.path.exists(file_path):
                existing = pl.read_parquet(file_path)
                if purge_ticker:
                    existing = existing.filter(pl.col("Ticker") != purge_ticker)
                # Align column order then cast to existing schema (e.g. Volume
                # arrives as Float64 from yfinance when batch has any NaNs).
                chunk = chunk.select(existing.columns)
                chunk = chunk.with_columns([
                    pl.col(c).cast(existing[c].dtype, strict=False)
                    for c in existing.columns
                    if chunk[c].dtype != existing[c].dtype
                ])
                final = pl.concat([existing, chunk]).unique(
                    subset=["Ticker", "Date"], keep="last"
                )
            else:
                final = chunk

            final.sort(["Ticker", "Date"]).write_parquet(
                file_path, compression="zstd", compression_level=3
            )

    # ─────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────

    def report_failures(self):
        """Prints a summary table of all failed tickers."""
        self._emit("\n" + "=" * 50)
        self._emit("📊 FINAL DOWNLOAD REPORT")
        self._emit("=" * 50)
        if not self.failed_downloads:
            self._emit("✨ All downloads successful!")
        else:
            self._emit(f"Found {len(self.failed_downloads)} failures:")
            report_df = pl.DataFrame(self.failed_downloads)
            with pl.Config(tbl_rows=-1): self._emit(str(report_df))
        self._emit("=" * 50 + "\n")


# ─────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Yahoo Finance NSE ticker downloader"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="CSV config file (tickers or benchmarks)"
    )

    parser.add_argument(
        "--data-root",
        required=True,
        help="Output parquet root"
    )

    parser.add_argument(
        "--check-rebase",
        action="store_true",
        help="Enable overlap price mismatch rebase detection"
    )

    args = parser.parse_args()

    pipeline = NSEInstitutionalPipeline(
        config_file=args.config,
        data_root=args.data_root,
        check_rebase=args.check_rebase
    )

    try:
        tickers = pipeline.load_and_dedup_tickers()
        print(f"🚀 Starting pipeline for {len(tickers)} tickers...")

        end_dt = pipeline.get_market_end_date()
        end_dt = end_dt + timedelta(days=1) # Make it exclusive

        print(f"📅 Target End Date: {end_dt}")

        # ── Step 1: classify ──────────────────────────────────────────────────
        # Tickers with no data or gap >STALE_DAYS → sequential full_refresh (stable).
        # Tickers with recent data → batch_incremental_sync (grouped by last_dt).
        batch_tickers, incremental_groups = pipeline.classify_tickers(tickers, end_dt)

        # ── Step 2: sequential full refresh for new / stale tickers ──────────
        if batch_tickers:
            print(f"⬇️  Full refresh for {len(batch_tickers)} tickers (one at a time)...")
            for t in batch_tickers:
                try:
                    pipeline.full_refresh(t, end_dt)
                except Exception as e:
                    print(f"❌ Critical error during full refresh for {t}: {e}")

        # ── Step 3: batched incremental sync (grouped by last_dt) ────────────
        if incremental_groups:
            pipeline.batch_incremental_sync(incremental_groups, end_dt)

        pipeline.report_failures()
        print("🏁 Pipeline Finished.")

    except Exception as e:
        print(f"💀 Critical Error: {str(e)}")