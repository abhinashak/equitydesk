"""
algo/ga/ga_service.py
─────────────────────
Business Logic layer for the Genetic Algorithm optimizer.

Wraps GADAL; exposes clean service methods to the UI layer.
"""

from typing import Generator

import pandas as pd

from algo.ga.ga_dal import GADAL
from utils.logger import get_logger

log = get_logger(__name__)


class GAService:

    def __init__(self, run_name: str, base_out: str = "outputs"):
        self._out_dir = f"{base_out}/{run_name}"
        self._dal     = GADAL(out_dir=self._out_dir)
        self.run_name = run_name

    # ── Run management ────────────────────────────────────────────────────────

    @staticmethod
    def list_runs(base_out: str = "outputs") -> list[str]:
        from pathlib import Path
        base = Path(base_out)
        if not base.exists():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir())

    @staticmethod
    def load_tickers_from_config(config_file: str = "config/tickers.csv") -> list[str]:
        """Return the Yahoo Symbol column from the tickers CSV."""
        try:
            import pandas as _pd
            df = _pd.read_csv(config_file)
            df.columns = [c.strip() for c in df.columns]
            col = next((c for c in df.columns if "yahoo" in c.lower()), None)
            if col is None:
                return []
            return df[col].dropna().str.strip().tolist()
        except Exception as exc:
            log.warning("Could not load tickers from %s: %s", config_file, exc)
            return []

    # ── Training ──────────────────────────────────────────────────────────────

    def run_train(
        self,
        config_name: str | None       = None,
        ticker_override: list[str] | None = None,
    ) -> Generator[str, None, None]:
        log.info("GA train: run=%s config=%s tickers=%s",
                 self.run_name, config_name,
                 len(ticker_override) if ticker_override else "all")
        yield from self._dal.run_train(
            config_name     = config_name,
            ticker_override = ticker_override,
        )

    # ── Evaluation ────────────────────────────────────────────────────────────

    def run_eval(
        self,
        config_name: str | None = None,
        skip_walkforward: bool  = False,
    ) -> Generator[str, None, None]:
        log.info("GA eval: run=%s config=%s wf=%s",
                 self.run_name, config_name, not skip_walkforward)
        yield from self._dal.run_eval(
            config_name      = config_name,
            skip_walkforward = skip_walkforward,
        )

    # ── Results ───────────────────────────────────────────────────────────────

    def get_eval_results(self) -> pd.DataFrame | None:
        return self._dal.load_eval_results()

    def get_walk_forward(self) -> pd.DataFrame | None:
        return self._dal.load_walk_forward()

    def get_weights(self, config_name: str) -> pd.DataFrame | None:
        return self._dal.load_weights(config_name)

    def get_train_summary(self) -> pd.DataFrame | None:
        return self._dal.load_train_summary()

    @property
    def out_dir(self) -> str:
        return self._out_dir
