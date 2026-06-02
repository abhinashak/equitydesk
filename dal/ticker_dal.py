"""
dal/ticker_dal.py
─────────────────
Read-only access to ticker OHLCV parquet files produced by
ticker_loader.py / ticker_loader_live.py.
Also wraps the subprocess calls that trigger those loaders.
"""

import queue
import threading
from pathlib import Path
from typing import Generator

import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


class TickerDAL:
    """Data-access for historical and live price data."""

    def __init__(self, data_root: str = "data/ticker"):
        self.data_root = Path(data_root)

    # ── Metadata / stats ──────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return basic coverage stats from the parquet store."""
        try:
            import polars as pl
            files = list(self.data_root.rglob("*.parquet"))
            if not files:
                return {"n_tickers": 0, "n_points": 0, "last_date": "—"}
            df_meta = pl.scan_parquet(str(self.data_root / "**" / "*.parquet"),
                                      hive_partitioning=True)
            return {
                "n_tickers":  df_meta.select(pl.col("Ticker").n_unique()).collect().item(),
                "n_points":   df_meta.select(pl.len()).collect().item(),
                "last_date":  str(df_meta.select(pl.col("Date").max()).collect().item()),
            }
        except Exception as exc:
            log.warning("stats() failed: %s", exc)
            return {"n_tickers": "?", "n_points": "?", "last_date": "?"}

    def load_ticker(self, ticker: str) -> pd.DataFrame | None:
        """Load all OHLCV rows for a single ticker symbol."""
        files = list(self.data_root.rglob(f"*{ticker}*.parquet"))
        if not files:
            return None
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    # ── Subprocess launchers ──────────────────────────────────────────────────

    def run_historical_loader(
            self,
            config_file: str = "config/tickers.csv",
            check_rebase: bool = False,
    ) -> Generator[str, None, None]:
        """Run NSEInstitutionalPipeline directly (in-process) and stream its output."""
        from loader.ticker_loader import NSEInstitutionalPipeline
        from datetime import timedelta

        pipeline = NSEInstitutionalPipeline(
            config_file=config_file,
            data_root=str(self.data_root),
            check_rebase=check_rebase,
        )
        yield from _run_pipeline(pipeline, _historical_runner)

    def run_live_loader(
            self,
            config_file: str = "config/tickers.csv",
    ) -> Generator[str, None, None]:
        """Run NSEInstitutionalPipeline (live) directly (in-process) and stream its output."""
        try:
            from loader.ticker_loader_live import NSEInstitutionalPipeline as LivePipeline
        except Exception as exc:
            yield f"❌ Import error: {exc}"
            return

        try:
            pipeline = LivePipeline(
                config_file=config_file,
                data_root=str(self.data_root),
            )
        except Exception as exc:
            yield f"❌ Pipeline init error: {exc}"
            return

        yield from _run_pipeline(pipeline, _live_runner)


# ── Pipeline runner helpers ────────────────────────────────────────────────────

def _historical_runner(pipeline):
    """Execute the full historical pipeline logic (mirrors __main__ in ticker_loader.py)."""
    from datetime import timedelta
    tickers = pipeline.load_and_dedup_tickers()
    pipeline._emit(f"🚀 Starting pipeline for {len(tickers)} tickers...")

    end_dt = pipeline.get_market_end_date()
    end_dt = end_dt + timedelta(days=1)
    pipeline._emit(f"📅 Target End Date: {end_dt}")

    batch_tickers, incremental_groups = pipeline.classify_tickers(tickers, end_dt)

    if batch_tickers:
        pipeline._emit(f"⬇️  Full refresh for {len(batch_tickers)} tickers (one at a time)...")
        for t in batch_tickers:
            try:
                pipeline.full_refresh(t, end_dt)
            except Exception as e:
                pipeline._emit(f"❌ Critical error during full refresh for {t}: {e}")

    if incremental_groups:
        pipeline.batch_incremental_sync(incremental_groups, end_dt)

    pipeline.report_failures()
    pipeline._emit("🏁 Pipeline Finished.")


def _live_runner(pipeline):
    """Execute the live pipeline logic (mirrors __main__ in ticker_loader_live.py)."""
    tickers = pipeline.load_and_dedup_tickers()
    pipeline._emit(f"🚀 Downloading today's data for {len(tickers)} tickers...")
    try:
        df, today = pipeline.download_today(tickers)
        pipeline.write_live(df, year=today.year)
        pipeline.report_failures()
        pipeline._emit("🏁 Done.")
    except Exception as exc:
        pipeline._emit(f"💀 Critical Error: {exc}")


def _run_pipeline(pipeline, runner_fn) -> Generator[str, None, None]:
    """
    Run *runner_fn(pipeline)* in a background thread; yield each line it emits.

    q.get() blocks until a line arrives — that is intentional and correct.
    The background thread is free to run; the main thread simply waits at
    q.get() and is unblocked the moment _emit() puts a line in the queue.
    Streamlit re-renders after each yielded line because _run_blocking()
    calls st.empty().code() before asking for the next line.
    """
    import traceback as tb
    _SENTINEL = object()
    q: queue.Queue = queue.Queue()

    pipeline.set_output_handler(q.put)

    def _worker():
        try:
            runner_fn(pipeline)
        except Exception as exc:
            q.put(f"❌ Exception in pipeline worker: {exc}")
            for line in tb.format_exc().splitlines():
                q.put(line)
        finally:
            q.put(_SENTINEL)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    log.info("Pipeline worker thread started (id=%s)", t.ident)

    while True:
        item = q.get()          # blocks until _emit() or sentinel arrives
        if item is _SENTINEL:
            break
        yield item

    t.join()
    log.info("Pipeline worker thread finished")