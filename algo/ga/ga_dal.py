"""
algo/ga/ga_dal.py
─────────────────
Data-access layer for the Genetic Algorithm optimizer.

Responsibilities:
  • Run _train_entrypoint / _eval_entrypoint in a background thread.
  • Stream output to the caller via ga_common.set_output_handler() — a
    module-level callback that every _emit() call inside ga_common and
    ga_train routes through, regardless of when those modules were imported.
    (The old sys.stdout redirect broke because ga_common is imported at
    module-load time, before the thread's redirect was in place.)
  • Read output artefacts (weights CSVs, eval_results.csv, walk_forward.csv).
  • No Streamlit imports — pure Python.
"""

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


def _stream_in_thread(fn, *args, **kwargs) -> Generator[str, None, None]:
    """
    Run fn(*args, **kwargs) in a background thread.
    All _emit() calls inside ga_common / ga_train / ga_eval route through
    ga_common.set_output_handler, which we point at a queue here.
    Errors are caught and emitted before the sentinel so they show in the UI.
    """
    q: queue.Queue = queue.Queue()

    def _handler(line: str):
        q.put(line)

    # Resolve algo/ga dir once, at call time, from multiple candidates
    # so it works regardless of how/where Streamlit is launched.
    _candidates = [
        Path(__file__).parent,                        # algo/ga  (installed path)
        Path(__file__).parent.resolve(),              # abs algo/ga
        Path.cwd() / "algo" / "ga",                  # <cwd>/algo/ga (most common)
        Path.cwd() / "ga",                            # <cwd>/ga
    ]
    _ga_dir = next((str(p) for p in _candidates if p.is_dir()), str(Path(__file__).parent))

    def _worker():
        # Register sys.path so bare imports (ga_common, ga_train, ga_eval) resolve.
        if _ga_dir not in sys.path:
            sys.path.insert(0, _ga_dir)

        try:
            import ga_common as _gc
            _gc.set_output_handler(_handler)
        except Exception as exc:
            q.put(f"⚠️  Could not import ga_common to register handler: {exc}")
            q.put(f"    sys.path searched: {_ga_dir}")

        try:
            fn(*args, **kwargs)
        except Exception as exc:
            q.put(f"❌ Exception: {exc}")
            for line in tb.format_exc().splitlines():
                q.put(line)
        finally:
            # Clear the handler so stray background callbacks don't queue forever
            try:
                import ga_common as _gc
                _gc.set_output_handler(None)
            except Exception:
                pass
            q.put(_SENTINEL)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    log.info("GA worker thread started (id=%s fn=%s)", t.ident, fn.__name__)

    while True:
        item = q.get()
        if item is _SENTINEL:
            break
        yield item

    t.join()
    log.info("GA worker thread finished (fn=%s)", fn.__name__)


# ─────────────────────────────────────────────────────────────────────────────

class GADAL:
    """Data-access for GA training and evaluation artefacts."""

    def __init__(self, out_dir: str = "outputs/default"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Runners ───────────────────────────────────────────────────────────────

    def run_train(
            self,
            config_name: str | None = None,
            ticker_override: list[str] | None = None,
    ) -> Generator[str, None, None]:
        """Stream ga_train output."""
        yield from _stream_in_thread(
            _train_entrypoint,
            out_dir         = str(self.out_dir),
            config_name     = config_name,
            ticker_override = ticker_override,
        )

    def run_eval(
            self,
            config_name: str | None = None,
            skip_walkforward: bool  = False,
    ) -> Generator[str, None, None]:
        """Stream ga_eval output."""
        yield from _stream_in_thread(
            _eval_entrypoint,
            out_dir          = str(self.out_dir),
            config_name      = config_name,
            skip_walkforward = skip_walkforward,
        )

    # ── Artefact readers ──────────────────────────────────────────────────────

    def list_runs(self, base_out: str = "outputs") -> list[str]:
        base = Path(base_out)
        if not base.exists():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir())

    def load_eval_results(self) -> pd.DataFrame | None:
        path = self.out_dir / "eval_results.csv"
        if not path.exists():
            return None
        return pd.read_csv(path)

    def load_walk_forward(self) -> pd.DataFrame | None:
        path = self.out_dir / "walk_forward.csv"
        if not path.exists():
            return None
        return pd.read_csv(path)

    def load_weights(self, config_name: str) -> pd.DataFrame | None:
        safe = config_name.replace("/", "-").replace(" ", "_")
        candidates   = list(self.out_dir.glob(f"*{safe}*.csv"))
        weight_files = [f for f in candidates if "weights_train" in f.name]
        if not weight_files:
            return None
        return pd.read_csv(weight_files[0])

    def load_train_summary(self) -> pd.DataFrame | None:
        path = self.out_dir / "train_summary.csv"
        if not path.exists():
            return None
        return pd.read_csv(path)


# ── In-process entrypoints ────────────────────────────────────────────────────

def _train_entrypoint(
        out_dir: str,
        config_name: str | None,
        ticker_override: list[str] | None,
):
    import random as _random
    import numpy as np
    from pathlib import Path as _Path

    # Use the same candidate resolution as _stream_in_thread
    _ga_dir = next(
        (str(p) for p in [_Path(__file__).parent, _Path.cwd() / "algo" / "ga", _Path.cwd() / "ga"]
         if p.is_dir()), str(_Path(__file__).parent)
    )
    if _ga_dir not in sys.path:
        sys.path.insert(0, _ga_dir)

    import ga_common as gc
    import ga_train  as gt

    _random.seed(gc.CFG.random_seed)
    np.random.seed(gc.CFG.random_seed)

    out = _Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    loader      = gc.DataLoader(gc.CFG.data_glob).load()
    all_tickers = [t for t in loader.tickers if t in loader.prices.columns]

    if ticker_override:
        tickers = [t for t in ticker_override if t in all_tickers]
        missing = set(ticker_override) - set(tickers)
        if missing:
            gc._emit(f"⚠️  Ticker override: {len(missing)} symbol(s) not in data: {sorted(missing)}")
        gc._emit(f"ℹ️  Using {len(tickers)} overridden tickers (of {len(all_tickers)} available)")
    else:
        tickers = all_tickers
        gc._emit(f"ℹ️  Using all {len(tickers)} tickers from data")

    prices        = loader.prices
    nsei_features = loader.nsei_features
    signals       = loader.signals

    if not tickers:
        gc._emit("❌ No valid tickers — aborting.")
        return

    configs = gc.PERIOD_CONFIG
    if config_name:
        configs = [c for c in configs if c["name"] == config_name]
        if not configs:
            gc._emit(f"❌ Config '{config_name}' not found or disabled.")
            return

    summaries = []
    for cfg in configs:
        gc.load_sector_momentum_cache(
            gc.CFG.benchmark_momentum_glob,
            start=cfg["train"]["start"],
            end=cfg["train"]["end"],
        )
        s = gt.train_one_config(cfg, prices, tickers, nsei_features, signals, out)
        summaries.append(s)

    valid = [s for s in summaries if not s.get("skipped")]
    if valid:
        import pandas as _pd
        _pd.DataFrame(valid).to_csv(out / "train_summary.csv", index=False)

    gc._emit(f"\n🏁 Training complete. {len(valid)}/{len(summaries)} configs succeeded.")


def _eval_entrypoint(
        out_dir: str,
        config_name: str | None,
        skip_walkforward: bool,
):
    import numpy as np
    from pathlib import Path as _Path

    _ga_dir = next(
        (str(p) for p in [_Path(__file__).parent, _Path.cwd() / "algo" / "ga", _Path.cwd() / "ga"]
         if p.is_dir()), str(_Path(__file__).parent)
    )
    if _ga_dir not in sys.path:
        sys.path.insert(0, _ga_dir)

    import ga_common as gc
    import ga_eval   as ge

    out = _Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    loader        = gc.DataLoader(gc.CFG.data_glob).load()
    tickers       = [t for t in loader.tickers if t in loader.prices.columns]
    prices        = loader.prices
    nsei_features = loader.nsei_features
    signals       = loader.signals

    if not tickers:
        gc._emit("❌ No valid tickers — aborting.")
        return

    configs = gc.PERIOD_CONFIG
    if config_name:
        configs = [c for c in configs if c["name"] == config_name]
        if not configs:
            gc._emit(f"❌ Config '{config_name}' not found or disabled.")
            return

    results = []
    for cfg in configs:
        gc.load_sector_momentum_cache(
            start=cfg["test"]["start"], end=cfg["test"]["end"]
        )
        r = ge.eval_one_config(cfg, prices, tickers, nsei_features, out)
        results.append(r)

    valid = [r for r in results if not r.get("skipped")]
    if valid:
        import pandas as _pd
        results_df   = _pd.DataFrame(valid)
        results_path = out / gc.CFG.eval_results_file
        results_df.to_csv(results_path, index=False)

    if not skip_walkforward and valid:
        ge.run_walk_forward_all_configs(
            configs, prices, tickers, nsei_features, signals, out
        )

    gc._emit(f"\n🏁 Evaluation complete. {len(valid)}/{len(results)} configs succeeded.")