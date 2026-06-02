"""
bll/fundamental_service.py
──────────────────────────
Business Logic for the fundamental data pipeline.

Orchestrates: accept uploaded HTML → parse → merge (via FundamentalDAL).
Yields human-readable status lines so both Streamlit and CLI
can display progress without coupling to either.

NOTE: The download step has been removed.  HTML files are now supplied by the
caller (downloaded manually via the DevTools script and uploaded in the UI).
"""

from pathlib import Path
from typing import Generator

import pandas as pd

from dal.fundamental_dal import FundamentalDAL
from utils.logger import get_logger

log = get_logger(__name__)


class FundamentalService:
    """
    Incremental fundamental data pipeline.
    Caller iterates the generator to drive progress display.
    """

    def __init__(
            self,
            out_dir: str = "data/fundamental",
            parser_script: str = "loader/screener_parser.py",
            sleep_secs: int = 0,           # kept for API compat; no longer used
            cookies: dict[str, str] | None = None,  # kept for API compat; unused
    ):
        self._dal = FundamentalDAL(
            out_dir=out_dir,
            parser_script=parser_script,
            sleep_secs=sleep_secs,
            cookies=cookies,
        )

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def run_from_files(
            self,
            ticker_paths: dict[str, Path],
    ) -> Generator[str, None, None]:
        """
        Generator that drives the parse→merge pipeline for pre-downloaded HTML.

        Parameters
        ----------
        ticker_paths : {ticker: path_to_html_file}
            Mapping from screener ticker symbol to the locally saved HTML file
            (e.g. a Streamlit UploadedFile written to /tmp by the UI layer).

        Yields status strings; caller decides how to display them.
        """
        total = len(ticker_paths)
        for idx, (ticker, html_file) in enumerate(ticker_paths.items(), 1):
            yield f"[{idx}/{total}] 📂  Loading {ticker} from {Path(html_file).name}…"

            html_path, html_size = self._dal.load_html_from_upload(ticker, Path(html_file))
            if html_path is None:
                yield f"  ❌ Could not read HTML for {ticker} ({html_size:,} bytes — too small or unreadable)"
                continue

            yield f"  🔍 Parsing HTML → staging parquet… ({html_size:,} bytes)"
            staging_dir = self._dal.parse_html(ticker, html_path)
            if staging_dir is None:
                yield f"  ❌ screener_parser failed for {ticker}"
                continue

            yield f"  🔄 Merging into {self._dal.out_dir}…"
            stats = self._dal.incremental_merge(ticker, staging_dir)

            for table, (inserted, skipped) in stats.items():
                if inserted > 0:
                    yield f"    ✅ {table}: +{inserted} new rows ({skipped} already existed)"

            yield f"  ✅ {ticker} done."

        yield "🏁 Fundamental load complete."

    # ── Coverage ──────────────────────────────────────────────────────────────

    def get_coverage(self) -> pd.DataFrame:
        """Return a summary DataFrame: table / tickers / rows."""
        return self._dal.get_coverage()

    def get_stale_or_missing(
            self,
            all_tickers: list[str],
            stale_hours: float | None = None,
    ) -> pd.DataFrame:
        """
        Return tickers that are never loaded, stale, or missing required tables.
        Pass stale_hours=None to only surface never-loaded and missing-table cases.
        """
        return self._dal.get_stale_or_missing(all_tickers, stale_hours=stale_hours)