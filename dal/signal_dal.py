"""
dal/signal_dal.py
─────────────────
Read / write access to computed signal parquets
(momentum, relative-strength, sector context, etc.)
produced by signal_generator.py.
"""

import io
import queue
import sys
import threading
import traceback as tb
from pathlib import Path
from typing import Generator

import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

_SENTINEL = object()


class _QueueWriter(io.TextIOBase):
    """
    A file-like object that forwards every write() call to a queue as a
    line-at-a-time string.  Partial writes (no trailing newline) are buffered
    until a newline arrives or the object is closed/flushed.
    """

    def __init__(self, q: queue.Queue):
        self._q = q
        self._buf = ""

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._q.put(line)
        return len(s)

    def flush(self):
        if self._buf:
            self._q.put(self._buf)
            self._buf = ""

    def close(self):
        self.flush()
        super().close()


class SignalDAL:
    """Data-access for pre-computed signal data."""

    def __init__(self, signal_dir: str = "data/signal_momentum"):
        self.signal_dir = Path(signal_dir)
        self.signal_dir.mkdir(parents=True, exist_ok=True)

    # ── Read ──────────────────────────────────────────────────────────────────

    def load_latest(self) -> pd.DataFrame | None:
        """Load the most recently generated signal snapshot."""
        files = sorted(self.signal_dir.rglob("*.parquet"))
        if not files:
            return None
        return pd.read_parquet(files[-1])

    def load_for_ticker(self, ticker: str) -> pd.DataFrame | None:
        files = list(self.signal_dir.rglob(f"*{ticker}*.parquet"))
        if not files:
            return None
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # ── Generate (in-process, streamed) ───────────────────────────────────────

    def run_signal_generator(
            self,
            config_file: str = "config/tickers.csv",
    ) -> Generator[str, None, None]:
        """
        Run signal_generator logic in a background thread and stream its
        print() output line-by-line, exactly the same way TickerDAL streams
        pipeline output via _run_pipeline / _QueueWriter.

        signal_generator.py uses bare print() calls throughout, so we redirect
        sys.stdout inside the worker thread to a _QueueWriter that forwards
        each line to a queue.  The main thread yields from the queue until the
        sentinel arrives.
        """
        q: queue.Queue = queue.Queue()

        def _worker():
            writer = _QueueWriter(q)
            old_stdout = sys.stdout
            sys.stdout = writer
            try:
                _run_signal_generator_main(
                    config_file=config_file,
                    out_dir=str(self.signal_dir),
                )
            except Exception as exc:
                # Restore stdout first so these lines actually reach the queue
                sys.stdout = old_stdout
                q.put(f"❌ Exception in signal worker: {exc}")
                for line in tb.format_exc().splitlines():
                    q.put(line)
            finally:
                sys.stdout = old_stdout
                writer.close()
                q.put(_SENTINEL)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        log.info("Signal generator worker thread started (id=%s)", t.ident)

        while True:
            item = q.get()          # blocks until a line or sentinel arrives
            if item is _SENTINEL:
                break
            yield item

        t.join()
        log.info("Signal generator worker thread finished")


# ── In-process entry point for signal_generator ───────────────────────────────

def _run_signal_generator_main(config_file: str, out_dir: str) -> None:
    """
    Mirrors the __main__ block of signal_generator.py, but accepts
    config_file / out_dir as parameters instead of argparse args, and
    writes output to the caller-supplied out_dir rather than the hard-coded
    ./data_signal_momentum path.

    All print() calls inside signal_generator.py flow automatically to the
    _QueueWriter installed by the worker thread above.
    """
    import duckdb
    import importlib
    import sys as _sys
    from datetime import timedelta

    # Try to import signal_generator from wherever it lives on sys.path.
    # Emit a clear diagnostic so the UI shows exactly what was tried.
    sg = None
    _import_errors: list[str] = []
    for _mod_name in ("signal_generator", "loader.signal_generator"):
        try:
            sg = importlib.import_module(_mod_name)
            print(f"✅ Imported signal_generator as '{_mod_name}'")
            break
        except ModuleNotFoundError as _e:
            _import_errors.append(f"  tried '{_mod_name}': {_e}")

    if sg is None:
        print("❌ Could not import signal_generator. Searched:")
        for _msg in _import_errors:
            print(_msg)
        print("\nsys.path entries:")
        for _p in _sys.path:
            print(f"  {_p}")
        return

    # ── Respect the caller's out_dir and ticker/benchmark data paths ────────
    original_output_root        = sg.OUTPUT_ROOT
    original_ticker_glob        = sg.DATA_TICKER_GLOB
    original_benchmark_glob     = sg.DATA_BENCHMARK_GLOB
    original_ticker_latest      = sg.DATA_TICKER_LATEST

    sg.OUTPUT_ROOT          = out_dir
    # These are read by fetch_ticker_data() at call-time via the module globals,
    # so they must be patched here even though they don't depend on out_dir.
    sg.DATA_TICKER_GLOB     = "./data/ticker/**/*.parquet"
    sg.DATA_BENCHMARK_GLOB  = "./data/benchmark/**/*.parquet"
    sg.DATA_TICKER_LATEST   = "./data/ticker/**/*.parquet"

    try:
        # ── Load benchmark name map ───────────────────────────────────────────
        sg.BENCHMARK_NAME_MAP = sg.load_benchmark_name_map()

        con = duckdb.connect(database=":memory:")

        # ── Derive anchor date ────────────────────────────────────────────────
        query = f"""
            SELECT min(date) + {sg.ANCHOR_OFFSET_DAYS} AS start_date_derived
            FROM read_parquet('{sg.DATA_TICKER_GLOB}', hive_partitioning = true)
            WHERE ticker != '{sg.INDEX_TICKER}'
        """
        result = con.execute(query).fetchone()[0]
        anchor_date = __import__("pandas").to_datetime(result)

        print(f"Computed Anchor Date: {anchor_date.strftime('%Y-%m-%d')}")
        fetch_start = (anchor_date - timedelta(days=sg.LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        print(f"Fetch start date: {fetch_start}")

        # ── Discover tickers ──────────────────────────────────────────────────
        all_tickers_query = f"""
            SELECT DISTINCT Ticker AS ticker
            FROM read_parquet('{sg.DATA_TICKER_GLOB}', hive_partitioning = true)
        """
        regular_tickers = con.execute(all_tickers_query).df()["ticker"].tolist()
        benchmark_tickers = list(sg.BENCHMARK_NAME_MAP.keys())

        print(f"Found {len(regular_tickers)} equity tickers and {len(benchmark_tickers)} benchmarks.")
        print(f"  Benchmarks : {benchmark_tickers}")

        # ── Benchmark features ────────────────────────────────────────────────
        benchmark_features = []
        for ticker in benchmark_tickers:
            print(f"\n--- [BENCHMARK] Processing {ticker} ---")
            feature_df = sg.generate_features(ticker, start=fetch_start, is_benchmark=True)
            if feature_df.empty:
                print(f"⚠️ Skipping {ticker}: Not enough data.")
                continue
            feature_df["year"] = feature_df["date"].dt.year
            X = sg.create_slabs(feature_df)
            X["ticker"] = ticker
            X["name"]   = sg.BENCHMARK_NAME_MAP.get(ticker, ticker.lstrip("^"))
            benchmark_features.append(X)

        if benchmark_features:
            import pandas as pd
            bench_df = pd.concat(benchmark_features, ignore_index=True)
            sg.save_by_year(bench_df, out_dir, "benchmark_momentum.parquet")

        # ── Equity features ───────────────────────────────────────────────────
        sector_map   = sg.get_sector_mapping(config_file)
        all_features = []
        for ticker in regular_tickers:
            print(f"\n--- Processing {ticker} ---")
            feature_df = sg.generate_features(ticker, start=fetch_start)
            if feature_df.empty:
                print(f"⚠️ Skipping {ticker}: Not enough data to calculate features.")
                continue
            feature_df["year"] = feature_df["date"].dt.year
            X = sg.create_slabs(feature_df)
            X["ticker"] = ticker
            X = X.merge(sector_map, on="ticker", how="left")
            all_features.append(X)

        if all_features:
            import pandas as pd
            final_df = pd.concat(all_features, ignore_index=True)
            sg.save_by_year(final_df, out_dir, "ticker_momentum.parquet")

            # ── Normalised join via DuckDB ────────────────────────────────────
            norm_query = f"""
            WITH ticker AS (
                SELECT * FROM read_parquet('{out_dir}/**/ticker_momentum.parquet',
                                           hive_partitioning = true)
            ),
            bench AS (
                SELECT * FROM read_parquet('{out_dir}/**/benchmark_momentum.parquet',
                                           hive_partitioning = true)
            ),
            nsei_close AS (
                SELECT date, close AS nsei_close, sma200_dist
                FROM bench WHERE ticker = '{sg.INDEX_TICKER}'
            ),
            sector_joined AS (
                SELECT t.*,
                       {", ".join(f"b.{c} AS sector_{c}" for c in sg.AGG_COLS)}
                FROM ticker t
                LEFT JOIN bench b ON t.sector = b.name AND t.date = b.date
            ),
            nsei_agg AS (
                SELECT date,
                       {", ".join(f"AVG({c}) AS nsei_{c}" for c in sg.AGG_COLS)}
                FROM bench GROUP BY date
            ),
            with_rs AS (
                SELECT s.*,
                       {", ".join(f"n.nsei_{c}" for c in sg.AGG_COLS)},
                       s.close / nc.nsei_close AS rs,
                       (s.close / nc.nsei_close)
                           / LAG(s.close / nc.nsei_close, 20)
                               OVER (PARTITION BY s.ticker ORDER BY s.date) - 1 AS rs_momentum,
                       CASE WHEN nc.sma200_dist > 0 THEN 1 ELSE 0 END           AS market_trend
                FROM sector_joined s
                LEFT JOIN nsei_agg   n  ON s.date = n.date
                LEFT JOIN nsei_close nc ON s.date = nc.date
            )
            SELECT *,
                   CASE WHEN rs_momentum >    0 THEN 1 ELSE 0 END AS rs_improving,
                   CASE WHEN rs_momentum > 0.05 THEN 1 ELSE 0 END AS rs_strong
            FROM with_rs
            """

            print("ℹ️  Building ticker_momentum_normalized via DuckDB join...")
            norm_con = duckdb.connect(database=":memory:")
            norm_df  = norm_con.execute(norm_query).df()
            norm_df["year"] = pd.to_datetime(norm_df["date"]).dt.year
            sg.save_by_year(norm_df, out_dir, "ticker_momentum_normalized.parquet")
            print(f"✅ ticker_momentum_normalized written ({len(norm_df)} rows).")

        print("🏁 Signal generation finished.")

    finally:
        # Always restore all module-level constants so repeated calls are safe
        sg.OUTPUT_ROOT          = original_output_root
        sg.DATA_TICKER_GLOB     = original_ticker_glob
        sg.DATA_BENCHMARK_GLOB  = original_benchmark_glob
        sg.DATA_TICKER_LATEST   = original_ticker_latest