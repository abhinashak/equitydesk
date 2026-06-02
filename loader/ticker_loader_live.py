import polars as pl
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import os
import argparse


class NSEInstitutionalPipeline:
    BATCH_SIZE = 10

    def __init__(self, config_file="./config/tickers.csv", data_root="./data_ticker", check_rebase=False):
        self.config_file = config_file
        self.data_root = data_root
        self.failed_downloads = []

        cfg_dir = os.path.dirname(self.config_file)
        if cfg_dir:
            os.makedirs(cfg_dir, exist_ok=True)
        os.makedirs(self.data_root, exist_ok=True)
        self._output_handler = None

    # ── Output tee ────────────────────────────────────────────────────────────

    def set_output_handler(self, fn):
        """Register a callable that receives every log line in addition to stdout."""
        self._output_handler = fn

    def _emit(self, *args, **kwargs):
        """Drop-in for print() — writes to stdout and to the registered handler."""
        import io
        buf = io.StringIO()
        self._emit(*args, file=buf, **kwargs)
        line = buf.getvalue().rstrip("\n")
        self._emit(line)
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
            self._emit(f"Config file not found. Looked at {self.config_file}")

        df = pl.read_csv(self.config_file)
        df = df.with_columns(
            pl.coalesce([
                pl.col("Yahoo Symbol"),
                pl.col("Name") + ".NS"
            ]).alias("resolved_ticker")
        )
        return df.select("resolved_ticker").unique().to_series().to_list()

    # ─────────────────────────────────────────────
    # Download today's data
    # ─────────────────────────────────────────────

    def _clean_batch_slice(self, df_pd, ticker):
        """Cleans a single-ticker slice from a batch download result."""
        if df_pd.empty:
            return None

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

    def download_today(self, tickers):
        """
        Downloads only today's data for all tickers in batches.
        Raises if market is closed / no data returned.
        """
        today      = date.today()
        fetch_start = today - timedelta(days=1)  # yfinance end is exclusive
        fetch_end   = today + timedelta(days=1)

        all_rows = []

        for i in range(0, len(tickers), self.BATCH_SIZE):
            batch = tickers[i: i + self.BATCH_SIZE]
            self._emit(f"   ↳ batch [{i+1}–{i+len(batch)}/{len(tickers)}]")

            try:
                df_pd = yf.download(
                    batch,
                    start=fetch_start,
                    end=fetch_end,
                    auto_adjust=True,
                    group_by="ticker",
                    threads=False,
                    progress=False,
                )
            except Exception as e:
                self._emit(f"❌ Batch download failed: {e}")
                for t in batch:
                    self.failed_downloads.append({"Ticker": t, "Error": str(e)})
                continue

            if df_pd.empty:
                raise RuntimeError("Market closed or no data available for today.")

            # Single-ticker edge case: yfinance may return a flat or MultiIndex df
            if len(batch) == 1:
                ticker = batch[0]
                if (isinstance(df_pd.columns, pd.MultiIndex)
                        and ticker in df_pd.columns.get_level_values(0)):
                    slice_pd = df_pd[ticker].copy()
                else:
                    slice_pd = df_pd
                row = self._clean_batch_slice(slice_pd, ticker)
                if row is not None:
                    all_rows.append(row.filter(pl.col("Date") == today))
                else:
                    self.failed_downloads.append({"Ticker": ticker, "Error": "Empty after cleaning"})
                continue

            # Multi-ticker path
            available = set(df_pd.columns.get_level_values(0))
            for ticker in batch:
                if ticker not in available:
                    self._emit(f"⚠️  {ticker} missing from batch response.")
                    self.failed_downloads.append({"Ticker": ticker, "Error": "Not in batch response"})
                    continue
                row = self._clean_batch_slice(df_pd[ticker].copy(), ticker)
                if row is not None:
                    all_rows.append(row.filter(pl.col("Date") == today))
                else:
                    self.failed_downloads.append({"Ticker": ticker, "Error": "Empty after cleaning"})

        if not all_rows:
            raise RuntimeError("No data returned for any ticker.")

        return pl.concat(all_rows, how="diagonal_relaxed"), today

    # ─────────────────────────────────────────────
    # Storage
    # ─────────────────────────────────────────────

    def write_live(self, df, year):
        """
        Purges data_ticker/year=<year>/live.parquet if it exists, then writes fresh data.
        Adds a 'year' column to match the standard schema.
        """
        year_dir  = os.path.join(self.data_root, f"year={year}")
        os.makedirs(year_dir, exist_ok=True)
        file_path = os.path.join(year_dir, "live.parquet")

        if os.path.exists(file_path):
            os.remove(file_path)
            self._emit(f"🗑️  Purged existing {file_path}")

        df = df.with_columns(pl.lit(year).cast(pl.Int64).alias("year"))
        df.sort(["Ticker", "Date"]).write_parquet(
            file_path, compression="zstd", compression_level=3
        )
        self._emit(f"💾 Written {len(df)} rows → {file_path}")

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
            with pl.Config(tbl_rows=-1):
                self._emit(report_df)
        self._emit("=" * 50 + "\n")


# ─────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Yahoo Finance live downloader"
    )
    parser.add_argument( "--config", required=True, help="CSV config file" )
    parser.add_argument( "--data-root", required=True, help="Output parquet root directory" )
    args = parser.parse_args()

    pipeline = NSEInstitutionalPipeline( config_file=args.config, data_root=args.data_root )
    try:
        tickers = pipeline.load_and_dedup_tickers()
        print(f"🚀 Downloading today's data for {len(tickers)} tickers...")

        df, today = pipeline.download_today(tickers)
        pipeline.write_live(df, year=today.year)

        pipeline.report_failures()
        print("🏁 Done.")

    except Exception as e:
        print(f"💀 Critical Error: {str(e)}")