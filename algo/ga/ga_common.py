"""
=============================================================================
ga_common.py  —  Shared infrastructure for the GA Portfolio Optimizer
=============================================================================
Loads all configuration from:
  config/ga_config.csv   →  scalar hyper-parameters
  config/periods.json    →  train/test period definitions

Exports:
  CFG                    —  SimpleNamespace of all scalar config values
  PERIOD_CONFIG_ALL      —  list of all period dicts (raw, incl. disabled)
  PERIOD_CONFIG          —  list of enabled period dicts
  TRAIN_PERIODS          —  list of train-window dicts for enabled configs
  TEST_PERIODS           —  list of test-window  dicts for enabled configs
  SECTOR_MAP             —  {ticker: sector}   (populated at data-load time)
  REGIME_SECTOR_BIAS     —  {regime: {sector: multiplier}}
  REGIME_THRESHOLDS      —  canonical numeric thresholds (single source of truth)
  REGIME_CASE_SQL        —  canonical SQL CASE expression (inject into any query)
  DataLoader, ReturnsEngine, Individual, FitnessEvaluator,
  Initialiser, GeneticOptimizer,
  detect_regime, regime_series,
  tournament_selection, sbx_crossover, gaussian_mutate, regime_bias_mutate
=============================================================================

PERFORMANCE CHANGES vs original
--------------------------------
* iterrows() eliminated everywhere — replaced with vectorised pandas/numpy/polars ops
* compute_exposure_series: pure-Python per-date loop → np.select vectorised cap
    + numpy forward-pass for hysteresis (no more .loc[dt] per day)
* sbx_crossover / gaussian_mutate / regime_bias_mutate: element loops → numpy vectorised
* _sector_penalty: ticker dict loop → np.bincount on pre-built index array
    (array built once in FitnessEvaluator.__init__, not per evaluate() call)
* portfolio_returns EXIT/REENTRY logging: per-dt .loc loops →
    vectorised boolean masks + cumsum spell aggregation (_log_dma50_events)
* _load_scalar_config: iterrows → zip(col arrays) via polars read_csv
* get_dynamic_regime_biases: iterrows → polars from_arrow
* regime_series fallback: apply(detect_regime) → np.select vectorised
"""

import json
import logging
import math
import random
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT HANDLER  — register from ga_dal to stream print() into the UI box
# regardless of when modules were imported.
#
# Usage (ga_dal._train_entrypoint):
#   import ga_common
#   ga_common.set_output_handler(queue_put_fn)
#   ... run training ...
#   ga_common.set_output_handler(None)   # cleanup
# ─────────────────────────────────────────────────────────────────────────────
_output_handler = None  # callable(str) | None

def set_output_handler(fn):
    """Register (or clear) the UI streaming callback."""
    global _output_handler
    _output_handler = fn

def _emit(*args, **kwargs):
    """
    Drop-in for print() inside ga_common / ga_train / ga_eval.
    Writes to real stdout AND calls the registered handler so output
    reaches both the terminal and the Streamlit streaming box.
    """
    import io as _io
    buf = _io.StringIO()
    print(*args, file=buf, **kwargs)
    line = buf.getvalue().rstrip("\n")
    print(line)                                # → real stdout / terminal
    if _output_handler is not None:
        try:
            _output_handler(line)
        except Exception:
            pass

import duckdb
import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0-A.  LOAD SCALAR CONFIG  (config/ga_config.csv)
# ─────────────────────────────────────────────────────────────────────────────
_CONFIG_CSV = Path(__file__).parent.parent.parent / "config" / "ga_config.csv"

def _load_scalar_config(path: Path) -> "SimpleNamespace":
    """Parse ga_config.csv into a flat namespace; cast values by dtype column.
    PERF: replaced iterrows() with polars read_csv + zip(col arrays) (10-50× faster).
    """
    import types
    cfg_pl = pl.read_csv(path)
    ns = types.SimpleNamespace()
    _cast = {"str": str, "int": int, "float": float, "bool": lambda v: v.lower() == "true"}
    keys   = cfg_pl["key"].to_list()
    values = cfg_pl["value"].to_list()
    dtypes = cfg_pl["dtype"].to_list()
    for key, value, dtype in zip(keys, values, dtypes):
        caster = _cast.get(str(dtype).strip(), str)
        setattr(ns, str(key).strip(), caster(str(value).strip()))
    return ns

CFG = _load_scalar_config(_CONFIG_CSV)

# ─────────────────────────────────────────────────────────────────────────────
# 0-B.  LOAD PERIOD CONFIG  (config/periods.json)
# ─────────────────────────────────────────────────────────────────────────────
_PERIODS_JSON = Path(__file__).parent.parent.parent / "config" / "periods.json"

def _load_periods(path: Path):
    with open(path) as f:
        raw = json.load(f)
    all_periods = raw
    enabled, train_list, test_list = [], [], []
    for cfg in all_periods:
        if cfg.get("enabled") == 0:
            print(f"[Config] {cfg['name']} — SKIPPED (disabled)")
            continue
        print(f"[Config] {cfg['name']} — enabled")
        enabled.append(cfg)
        train_list.append({
            "name":  cfg["name"],
            "start": cfg["train"]["start"],
            "end":   cfg["train"]["end"],
            "type":  cfg["train"]["type"],
        })
        test_list.append({
            "name":  cfg["name"],
            "start": cfg["test"]["start"],
            "end":   cfg["test"]["end"],
            "type":  cfg["test"]["type"],
        })
    return all_periods, enabled, train_list, test_list

PERIOD_CONFIG_ALL, PERIOD_CONFIG, TRAIN_PERIODS, TEST_PERIODS = _load_periods(_PERIODS_JSON)

# ─────────────────────────────────────────────────────────────────────────────
# 0-C.  GLOBAL MUTABLE STATE  (populated by DataLoader at runtime)
# ─────────────────────────────────────────────────────────────────────────────
SECTOR_MAP: Dict[str, str] = {}
REGIME_SECTOR_BIAS: Dict[str, Dict[str, float]] = {}

# Manual sector exclusions / zero-weight overrides
MANUAL_SKIP_SECTORS: Dict[str, float] = {}  # e.g. {"NIFTY_IT": 0.0}
MANUAL_SKIP_SECTORS.update({
    key[len("skip_sector_"):]: float(val)
    for key, val in vars(CFG).items()
    if key.startswith("skip_sector_")
})


# market_cap category per ticker: "Large-cap" | "Mid-cap" | "Small-cap"
MARKET_CAP_MAP: Dict[str, str] = {}

# domestic_market_pct per ticker: float 0–100
DOMESTIC_PCT_MAP: Dict[str, float] = {}

# Eval-only: sector-index price cache — loaded ONCE per eval run via
# load_sector_momentum_cache().
# Key   = sector name from the parquet's 'name' column (e.g. "PHARMABEES",
#          "NSEBANK") — must match values in SECTOR_MAP.
# Value = DataFrame[date → {close, sma50}].
# Only ~4 sector-index rows/day; negligible memory footprint.
SECTOR_MOMENTUM_CACHE: Dict[str, "pd.DataFrame"] = {}

# ─────────────────────────────────────────────────────────────────────────────
# 0-D.  LOAD TICKERS METADATA  (Name, Yahoo Symbol, Sector, market_cap,
#                                domestic_market_pct)
# ─────────────────────────────────────────────────────────────────────────────
_TICKERS_CSV_DEFAULT = Path(__file__).parent.parent.parent / "config" / "tickers.csv"


def _load_tickers_metadata(path: Optional[Path] = None) -> None:
    """
    Parse the tickers CSV and populate SECTOR_MAP, MARKET_CAP_MAP,
    DOMESTIC_PCT_MAP globals.

    PERF: replaced iterrows() with polars read_csv (Arrow-backed, zero-copy).
    Populating dicts from column arrays is significantly faster than row iteration.
    """
    global SECTOR_MAP, MARKET_CAP_MAP, DOMESTIC_PCT_MAP
    csv_path = path or _TICKERS_CSV_DEFAULT
    if not csv_path.exists():
        print(f"[Tickers] WARNING: {csv_path} not found — "
              "MARKET_CAP_MAP and DOMESTIC_PCT_MAP will be empty.")
        return

    # Read with polars (Arrow columnar) — all as strings, fast
    df_pl   = pl.read_csv(csv_path, infer_schema_length=0)
    col_map = {c.strip().lower(): c for c in df_pl.columns}
    df_pl   = df_pl.rename({v: k for k, v in col_map.items()})

    col_symbol = next((c for c in df_pl.columns if "yahoo" in c or c == "symbol"), None)
    col_sector = next((c for c in df_pl.columns if c == "sector"), None)
    col_mcap   = next((c for c in df_pl.columns if "market_cap" in c), None)
    col_dom    = next((c for c in df_pl.columns if "domestic" in c), None)

    if col_symbol is None:
        print("[Tickers] WARNING: Cannot find 'Yahoo Symbol' column — skipping metadata load.")
        return

    symbols = [str(s).strip() for s in df_pl[col_symbol].to_list()]

    if col_sector:
        sectors = df_pl[col_sector].to_list()
        SECTOR_MAP.update({
            sym: str(sec).strip()
            for sym, sec in zip(symbols, sectors)
            if sec is not None and str(sec).strip() not in ("", "None", "null")
        })

    if col_mcap:
        mcaps = df_pl[col_mcap].to_list()
        MARKET_CAP_MAP.update({
            sym: str(mc).strip()
            for sym, mc in zip(symbols, mcaps)
            if mc is not None and str(mc).strip() not in ("", "None", "null")
        })

    if col_dom:
        doms = df_pl[col_dom].to_list()
        for sym, dom in zip(symbols, doms):
            if dom is None or str(dom).strip() in ("", "None", "null"):
                continue
            try:
                DOMESTIC_PCT_MAP[sym] = float(dom)
            except (ValueError, TypeError):
                pass

    print(f"[Tickers] Loaded metadata for {len(df_pl)} tickers — "
          f"{len(MARKET_CAP_MAP)} with market_cap, "
          f"{len(DOMESTIC_PCT_MAP)} with domestic_market_pct.")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  REGIME BIAS  (computed from data at load time)
# ─────────────────────────────────────────────────────────────────────────────
def get_dynamic_regime_biases(con) -> Dict[str, Dict[str, float]]:
    skip_sectors = [s.upper() for s, w in MANUAL_SKIP_SECTORS.items() if w == 0.0]
    sector_exclusion_sql = ""
    if skip_sectors:
        excluded = ", ".join(f"'{s}'" for s in skip_sectors)
        sector_exclusion_sql = f" AND UPPER(d.sector) NOT IN ({excluded}) "

    query = f"""
WITH daily_returns AS (
    SELECT
        date, ticker, sector,
        (close - LAG(close) OVER (PARTITION BY ticker ORDER BY date)) /
            NULLIF(LAG(close) OVER (PARTITION BY ticker ORDER BY date), 0) AS daily_ret
    FROM raw
),
regime_labels AS (
    SELECT date,
        {REGIME_CASE_SQL} AS regime
    FROM raw WHERE ticker = '{CFG.index_ticker}'
),
sector_performance AS (
    SELECT r.regime, d.sector, AVG(d.daily_ret) AS avg_ret
    FROM daily_returns d
    JOIN regime_labels r ON d.date = r.date
    WHERE d.sector IS NOT NULL {sector_exclusion_sql}
    GROUP BY 1, 2
),
index_performance AS (
    SELECT r.regime, AVG(d.daily_ret) AS index_ret
    FROM daily_returns d
    JOIN regime_labels r ON d.date = r.date
    WHERE d.ticker = '{CFG.index_ticker}'
    GROUP BY 1
)
SELECT s.regime, s.sector,
       (1 + s.avg_ret) / (1 + i.index_ret) AS bias_ratio
FROM sector_performance s
JOIN index_performance i ON s.regime = i.regime;
"""
    # PERF: fetch as Arrow → polars (avoids pandas overhead for simple grouping)
    arrow_tbl = con.execute(query).arrow()
    df_bias   = pl.from_arrow(arrow_tbl)

    biases: Dict[str, Dict[str, float]] = {}
    for row in df_bias.iter_rows(named=True):
        reg, sec, ratio = row["regime"], row["sector"], row["bias_ratio"]
        biases.setdefault(reg, {})[sec] = max(0.7, min(1.3, ratio))
    return biases


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
class DataLoader:
    """Loads and caches all price/signal data from parquet glob via DuckDB."""

    def __init__(self, glob: str = CFG.data_glob):
        self.glob = glob
        self.con  = duckdb.connect()
        self._price_pivot:   Optional[pd.DataFrame] = None
        self._signals:       Optional[pd.DataFrame] = None
        self._tickers:       Optional[List[str]]    = None
        self._nsei_features: Optional[pd.DataFrame] = None

    def _q(self, sql: str) -> pd.DataFrame:
        return self.con.execute(sql).df()

    def load(self) -> "DataLoader":
        _emit(f"[DataLoader] Reading parquet: {Path(self.glob).resolve()}")
        stmt = f"""CREATE OR REPLACE VIEW raw AS SELECT * FROM read_parquet('{self.glob}')  """
        self.con.execute(stmt)

        cols = self._q("DESCRIBE raw")["column_name"].tolist()
        _emit(f"[DataLoader] Columns ({len(cols)}): {cols[:10]} …")

        date_col = self._infer_date_col(cols)
        _emit(f"[DataLoader] Date column: {date_col}")

        # Populate global SECTOR_MAP and REGIME_SECTOR_BIAS
        sqlstmt = f"SELECT DISTINCT ticker, sector FROM raw WHERE year = year(today())"
        global SECTOR_MAP, REGIME_SECTOR_BIAS
        SECTOR_MAP.clear()
        SECTOR_MAP.update(
            {t: (s if s is not None else "Other")
             for t, s in self.con.execute(sqlstmt).fetchall()}
        )
        REGIME_SECTOR_BIAS.clear()
        REGIME_SECTOR_BIAS.update(get_dynamic_regime_biases(self.con))
        _tickers_csv = getattr(CFG, "tickers_csv", None)
        _load_tickers_metadata(Path(_tickers_csv) if _tickers_csv else None)

        all_tickers = self._q("SELECT DISTINCT ticker FROM raw ORDER BY ticker")["ticker"].tolist()
        self._tickers = [t for t in all_tickers if t != CFG.index_ticker]
        _emit(f"[DataLoader] Tickers found: {len(all_tickers)}  (excl. index: {len(self._tickers)})")

        _emit("[DataLoader] Building price pivot …")
        self._price_pivot = self._build_price_pivot(date_col)

        _emit("[DataLoader] Extracting NSEI features …")
        self._nsei_features = self._build_nsei_features(date_col)

        _emit("[DataLoader] Loading signal table …")
        self._signals = self._build_signals(date_col)

        _emit("[DataLoader] Done.\n")
        return self

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _infer_date_col(cols: List[str]) -> str:
        for candidate in ["date", "Date", "trade_date", "timestamp", "dt"]:
            if candidate in cols:
                return candidate
        raise ValueError(f"Cannot find date column in {cols}")

    def _build_price_pivot(self, date_col: str) -> pd.DataFrame:
        df = self._q(f"""
            SELECT {date_col}::DATE AS date, ticker,
                   LAST(close ORDER BY {date_col}) AS close
            FROM raw WHERE close IS NOT NULL
            GROUP BY {date_col}::DATE, ticker
            ORDER BY 1, 2
        """)
        _emit(f"[DataLoader] Price rows after dedup: {len(df)}")
        pivot = df.pivot(index="date", columns="ticker", values="close")
        pivot.index = pd.to_datetime(pivot.index)
        pivot.sort_index(inplace=True)
        return pivot

    def _build_nsei_features(self, date_col: str) -> pd.DataFrame:
        nsei_cols = [
            "close", "sma50", "sma200", "ulcer", "momentum_accelerating", "near_52w_high"
        ]
        available = self._q("DESCRIBE raw")["column_name"].tolist()
        sel_cols  = ", ".join([c for c in nsei_cols if c in available])
        if not sel_cols:
            sel_cols = ("NULL AS close, NULL AS sma200, NULL AS sma50, "
                        "NULL AS ulcer, NULL AS momentum_accelerating, NULL AS near_52w_high" )
        sqlstmt = f"""
            SELECT {date_col}::DATE AS date, {sel_cols}
            FROM raw WHERE ticker = '{CFG.index_ticker}' ORDER BY date
        """
        print(sqlstmt)
        df = self._q(sqlstmt)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        defaults = { "close": 0.0, "sma50": 0.0, "sma200": 0.0, "ulcer": 3.0,
                     "momentum_accelerating": 0.5, "near_52w_high": 0.0,}
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val
        return df

    def _build_signals(self, date_col: str) -> pd.DataFrame:
        signal_cols = ["1Y", "6M", "3M", "1M", "rsi", "rs", "rs_momentum",
                       "momentum_quality", "vol_ratio", "ulcer", "adx"]
        available = self._q("DESCRIBE raw")["column_name"].tolist()
        sel = ", ".join([f'"{c}"' for c in signal_cols if c in available])
        df  = self._q(f"""
            SELECT {date_col}::DATE AS date, ticker, {sel}
            FROM raw ORDER BY date, ticker
        """)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # ── public accessors ──────────────────────────────────────────────────────
    @property
    def prices(self) -> pd.DataFrame:        return self._price_pivot
    @property
    def tickers(self) -> List[str]:          return self._tickers
    @property
    def nsei_features(self) -> pd.DataFrame: return self._nsei_features
    @property
    def signals(self) -> pd.DataFrame:      return self._signals


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MARKET REGIME DETECTION
# ─────────────────────────────────────────────────────────────────────────────
#
# Single source of truth for all regime thresholds.
# Update numbers here — detect_regime(), regime_series(), get_dynamic_regime_biases(),
# and ga_train._regime_windows() all derive from these values automatically.
#
REGIME_THRESHOLDS: Dict[str, float] = {
    # crash: severe drawdown — index deep below 200-DMA
    "crash_ulcer":       5.0,   # ulcer  >  this
    "crash_sma200_pos":  0.2,   # (close-sma200)/sma200  <  this

    # bull: clean uptrend — low stress, well above 200-DMA, near 52-week high
    "bull_ulcer":        5.0,   # ulcer  <  this
    "bull_sma200_pos":   0.15,   # sma200_pos  >  this
    "bull_near_52w":     0.3,   # near_52w_high  >  this

    # recovery: momentum turning, price back near/above 200-DMA
    "recovery_mom_acc":  0.02,   # momentum_accelerating  >  this
    "recovery_sma200_pos": 0.05, # sma200_pos  >  this
}

# Canonical SQL CASE expression — inject into any query that needs regime labels.
# Assumes the surrounding query exposes columns:
#   ulcer, momentum_accelerating, close, sma200  (all from the NSEI/index rows)
# The computed column "sma200_pos" = (close - sma200) / sma200 is inlined so the
# expression remains self-contained (no CTE dependency).
REGIME_CASE_SQL: str = """\
CASE
    WHEN ulcer > {crash_ulcer}
         AND (close - sma200) / NULLIF(sma200, 0) < {crash_sma200_pos}
        THEN 'crash'
    WHEN ulcer < {bull_ulcer}
         AND (close - sma200) / NULLIF(sma200, 0) > {bull_sma200_pos}
         AND near_52w_high > {bull_near_52w}
        THEN 'bull'
    WHEN momentum_accelerating > {recovery_mom_acc}
         AND (close - sma200) / NULLIF(sma200, 0) > {recovery_sma200_pos}
        THEN 'recovery'
    ELSE 'bear'
END""".format(**REGIME_THRESHOLDS)


def detect_regime(nsei_row: pd.Series) -> str:
    """
    Python mirror of REGIME_CASE_SQL — same thresholds, same priority order.
    Priority: crash → bull → recovery → bear.

    Do NOT hardcode numbers here; read them from REGIME_THRESHOLDS so that
    a single edit propagates to SQL and Python simultaneously.
    """
    t = REGIME_THRESHOLDS
    sma200     = nsei_row.get("sma200", 0.0) or 0.0
    close      = nsei_row.get("close", 0.001) or 0.001
    sma200_pos = (close - sma200) / sma200 if sma200 else 0.0
    ulcer      = nsei_row.get("ulcer", 3.0)
    mom_acc    = nsei_row.get("momentum_accelerating", 0.5)
    near_52w   = nsei_row.get("near_52w_high", 0.3)

    if ulcer > t["crash_ulcer"] and sma200_pos < t["crash_sma200_pos"]:
        return "crash"
    if ulcer < t["bull_ulcer"] and sma200_pos > t["bull_sma200_pos"] and near_52w > t["bull_near_52w"]:
        return "bull"
    if mom_acc > t["recovery_mom_acc"] and sma200_pos > t["recovery_sma200_pos"]:
        return "recovery"
    return "bear"


def regime_series(nsei_features: pd.DataFrame) -> pd.Series:
    """
    Returns pd.Series[date → regime_str] by running REGIME_CASE_SQL via DuckDB
    on the supplied DataFrame.  Falls back to fully vectorised np.select if the
    required columns are absent (e.g. unit tests with sparse fixtures).
    PERF: fallback replaced apply(detect_regime) row-by-row with np.select.
    """
    required = {"ulcer", "momentum_accelerating", "close", "sma200", "near_52w_high"}
    if required.issubset(nsei_features.columns):
        df = nsei_features.reset_index()
        sql = f"SELECT date, {REGIME_CASE_SQL} AS regime FROM df ORDER BY date"
        result = duckdb.query(sql).df()
        result["date"] = pd.to_datetime(result["date"])
        return result.set_index("date")["regime"]

    # ── Vectorised fallback (replaces apply row-by-row) ──────────────────────
    t          = REGIME_THRESHOLDS
    df         = nsei_features
    close      = df.get("close",                 pd.Series(0.001, index=df.index)).fillna(0.001)
    sma200     = df.get("sma200",                pd.Series(0.0,   index=df.index)).fillna(0.0)
    ulcer      = df.get("ulcer",                 pd.Series(3.0,   index=df.index)).fillna(3.0)
    mom_acc    = df.get("momentum_accelerating", pd.Series(0.5,   index=df.index)).fillna(0.5)
    near_52w   = df.get("near_52w_high",         pd.Series(0.3,   index=df.index)).fillna(0.3)

    sma200_safe = sma200.where(sma200 != 0, other=1.0)
    sma200_pos  = (close - sma200) / sma200_safe

    crash_cond    = (ulcer > t["crash_ulcer"])  & (sma200_pos < t["crash_sma200_pos"])
    bull_cond     = (ulcer < t["bull_ulcer"])   & (sma200_pos > t["bull_sma200_pos"]) & (near_52w > t["bull_near_52w"])
    recovery_cond = (mom_acc > t["recovery_mom_acc"]) & (sma200_pos > t["recovery_sma200_pos"])

    regimes = np.select(
        [crash_cond, bull_cond, recovery_cond],
        ["crash",    "bull",    "recovery"],
        default="bear",
    )
    return pd.Series(regimes, index=df.index, name="regime")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  EXPOSURE FILTER  (SMA50 / SMA200 entry + exit)
# ─────────────────────────────────────────────────────────────────────────────
def compute_exposure_series(nsei_features: pd.DataFrame, date_index: pd.DatetimeIndex) -> pd.Series:
    """
    Compute a daily portfolio-exposure scalar in [0.0, 1.0] for every date.
    CRITICAL: entry thresholds MUST be >= exit thresholds to create a buffer
    zone, otherwise exit and re-entry fire on consecutive days and cancel out.
    Thresholds are loaded from ga_config.csv.
    The first date always starts fully invested (exposure = 1.0).

    PERF: The original had a Python for-loop with .loc[dt] per date —
    O(N) interpreter overhead per day.  Replaced with vectorised np.where cap
    + a minimal numpy forward-pass for hysteresis, making it O(N) numpy.
    """
    exit_sma50  = getattr(CFG, "exit_sma50_threshold",  0.0)
    entry_sma50 = getattr(CFG, "entry_sma50_threshold", 1.0)

    feat = nsei_features.reindex(date_index, method="ffill").shift(1)
    feat["sma50_pos"] = (feat["close"] - feat["sma50"]) / feat["sma50"]
    s50 = feat["sma50_pos"].fillna(1.0).to_numpy()

    n = len(s50)
    # ── Step 1: cap level each day imposes independently ─────────────────────
    cap = np.where(s50 < exit_sma50 * 8,  0.00,
                   np.where(s50 < exit_sma50 * 4,  0.25,
                            np.where(s50 < exit_sma50 * 2,  0.50,
                                     np.where(s50 < exit_sma50,       0.75,
                                              1.00))))  # no exit signal

    # ── Step 2: forward-pass — propagate the "min so far" cap ────────────────
    # When s50 >= entry_sma50 AND the previous exposure was < 1.0, reset to 1.0.
    # This mirrors the hysteresis logic exactly without a Python .loc loop.
    exposure = np.empty(n, dtype=np.float64)
    prev = 1.0
    for i in range(n):
        c = cap[i]
        if c < 1.0:
            prev = min(prev, c)           # exit / partial-exit regime: clamp down
        elif prev < 1.0 and s50[i] >= entry_sma50:
            prev = 1.0                    # full re-entry
        # else: hold — buffer zone, neither cap nor re-entry fired
        exposure[i] = prev

    return pd.Series(exposure, index=date_index)


# ─────────────────────────────────────────────────────────────────────────────
# 4-A.  EVAL-ONLY EXIT FILTERS
# ─────────────────────────────────────────────────────────────────────────────
# Three exit types, each controlled by an independent flag in ga_config.csv:
#
#   enable_portfolio_exit  (bool, default True)
#       Scale the whole portfolio's equity exposure via compute_exposure_series()
#       when the NIFTY50 index falls below its SMA50.  Same tiered 0/25/50/75/100%
#       logic as before, but now EVAL-ONLY — training uses raw returns.
#
#   enable_sector_exit  (bool, default True)
#       If a sector-index closes below its SMA50, all stocks belonging to that
#       sector are exited for the day; their weight is redistributed to survivors.
#       Sector names are read from SECTOR_MOMENTUM_CACHE (keyed by the parquet's
#       'name' column) and matched against SECTOR_MAP values — no extra config.
#
#   enable_ticker_exit  (bool, default True)
#       Exit a single stock when its RS (close / rolling_sma50) falls below
#       the sector-index RS × ticker_exit_rs_threshold (default 0.95).
#
# Required ga_config.csv entries (add as needed):
#   portfolio,enable_portfolio_exit,True,bool,...
#   portfolio,enable_sector_exit,True,bool,...
#   portfolio,enable_ticker_exit,True,bool,...
#   portfolio,ticker_exit_rs_threshold,0.95,float,...
#   portfolio,benchmark_momentum_glob,../data_signal_momentum/year=*/benchmark_momentum.parquet,str,...
#
# Call load_sector_momentum_cache() ONCE at eval-script startup before any
# portfolio_returns(mode='eval') calls.
# ─────────────────────────────────────────────────────────────────────────────

def load_sector_momentum_cache(
        benchmark_glob: Optional[str] = None,
        start:          Optional[str] = None,
        end:            Optional[str] = None,
) -> None:
    """
    Load sector-index close + SMA50 into SECTOR_MOMENTUM_CACHE.
    Call ONCE per eval run from the eval script before any
    portfolio_returns(mode='eval') calls.

    Sector names and their tickers are discovered automatically from the
    parquet's 'name' column — no external mapping config required.
    The cache is keyed by 'name' (e.g. "PHARMABEES", "NSEBANK") which must
    match the sector values stored in SECTOR_MAP / tickers.csv.

    Args:
        benchmark_glob: parquet glob, e.g.
            '../data_signal_momentum/year=*/benchmark_momentum.parquet'
            Falls back to CFG.benchmark_momentum_glob if omitted.
        start / end: ISO date strings ('2026-01-01') to bound the load window.
            If omitted, all available dates are loaded.
    """
    global SECTOR_MOMENTUM_CACHE

    glob_path = benchmark_glob or getattr(CFG, "benchmark_momentum_glob", None)
    if not glob_path:
        print("[SectorCache] WARNING: benchmark_momentum_glob not configured — "
              "sector/ticker exits will be disabled.")
        return

    date_filter = (f"AND date BETWEEN '{start}' AND '{end}'"
                   if start and end else "")

    con = duckdb.connect()
    try:
        con.execute(
            f"CREATE OR REPLACE VIEW bm_raw AS "
            f"SELECT * FROM read_parquet('{glob_path}')"
        )
        # Load name + close + sma50 for all sector-index rows in the window.
        # 'name' is the sector label (e.g. "PHARMABEES") that must match
        # the sector values in SECTOR_MAP.
        df = con.execute(f"""
            SELECT date::DATE AS date,
                   name,
                   ticker,
                   close,
                   sma50
            FROM bm_raw
            WHERE name IS NOT NULL
              AND close IS NOT NULL
              AND sma50 IS NOT NULL
              {date_filter}
            ORDER BY date, name
        """).df()
    finally:
        con.close()

    df["date"] = pd.to_datetime(df["date"])
    SECTOR_MOMENTUM_CACHE.clear()
    for sec_name, grp in df.groupby("name"):
        SECTOR_MOMENTUM_CACHE[sec_name] = (
            grp.set_index("date")[["close", "sma50"]].sort_index()
        )
    print( f"[SectorCache] Loaded {len(SECTOR_MOMENTUM_CACHE)} sector series.")

class ReturnsEngine:
    """Pre-computes daily returns; simulates portfolio returns with tx cost
    and dynamic SMA50/SMA200 exposure scaling."""

    def __init__(self, prices: pd.DataFrame, nsei_features: Optional[pd.DataFrame] = None):
        self.prices        = prices
        self.daily_ret     = prices.pct_change().fillna(0)
        self.nsei_features = nsei_features   # None → no exposure filter (index-only calls)
        self.dma50_log: List[Dict] = []      # EXIT / REENTRY events recorded here

    def clear_dma50_log(self):
        """Reset the exit/reentry log between configs or eval runs."""
        self.dma50_log = []

    def portfolio_returns(
            self,
            weights:      np.ndarray,
            tickers:      List[str],
            start:        str,
            end:          str,
            prev_weights: Optional[np.ndarray] = None,
            log_dma50:    bool = True,
            mode:         str  = "train",   # "train" | "eval"
    ) -> pd.Series:
        """
        Simulate daily portfolio returns for the given period.

        mode="train"  — raw weighted returns only; NO exit filters applied.
                        Training fitness is evaluated on clean returns so the GA
                        does not over-fit to exit-trigger timing.

        mode="eval"   — all three exit filters applied in order, each gated by
                        its own CFG flag:
                          1. Sector exit   (enable_sector_exit)
                          2. Ticker exit   (enable_ticker_exit)
                          3. Portfolio SMA50 exposure  (enable_portfolio_exit)
        """
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        mask = (self.daily_ret.index >= s) & (self.daily_ret.index <= e)
        sub  = self.daily_ret.loc[mask, tickers].copy().fillna(0)

        is_index_call = (len(tickers) == 1 and tickers[0] == CFG.index_ticker)

        # ── Raw portfolio return (dot product) ────────────────────────────────
        port_ret = sub @ weights

        # ── Eval-mode: sector / ticker exits ─────────────────────────────────
        # Applied before the portfolio-level SMA50 exposure filter.
        # Freed weight goes to cash earning the daily risk-free rate (not
        # redistributed to surviving stocks).
        invest_mask_2d: Optional[np.ndarray] = None
        if (mode == "eval" and not is_index_call
                and self.nsei_features is not None and len(port_ret) > 0):
            apply_sector_exit = getattr(CFG, "enable_sector_exit", True)
            apply_ticker_exit = getattr(CFG, "enable_ticker_exit", True)
            if (apply_sector_exit or apply_ticker_exit) and SECTOR_MOMENTUM_CACHE:
                invest_mask_2d = np.ones((len(sub), len(tickers)), dtype=np.float64)
                if apply_sector_exit:
                    invest_mask_2d *= self._compute_sector_exit_matrix(sub.index, tickers)
                if apply_ticker_exit:
                    invest_mask_2d *= self._compute_ticker_exit_matrix(sub.index, tickers)
                port_ret = self._apply_stock_exits(
                    sub, weights, apply_sector_exit, apply_ticker_exit
                )

        # ── Exposure filter ───────────────────────────────────────────────────
        # Only applied to real portfolio calls (not the single-ticker index call).
        # Index ticker is passed as a length-1 array; we skip scaling for it.
        is_index_call = (len(tickers) == 1 and tickers[0] == CFG.index_ticker)
        if self.nsei_features is not None and not is_index_call and len(port_ret) > 0:
            apply_portfolio_exit = (mode == "eval"
                                    and getattr(CFG, "enable_portfolio_exit", True))
            exposure   = compute_exposure_series(self.nsei_features, port_ret.index)
            equity_ret = port_ret.copy()

            if apply_portfolio_exit:
                # exposure should only scale the EQUITY portion, not the already-cashed RFR
                # port_ret here = equity_component + cash_component (from _apply_stock_exits)
                # We need to re-separate them before blending.
                invested_sum = pd.Series(
                    (invest_mask_2d * weights).sum(axis=1), index=sub.index
                ) if invest_mask_2d is not None else pd.Series(1.0, index=sub.index)

                equity_ret  = port_ret - invested_sum.rsub(1.0).clip(0, 1) * (CFG.risk_free_rate / 252)
                port_ret    = (equity_ret * exposure
                               + (CFG.risk_free_rate / 252) * (1.0 - exposure * invested_sum))
                # ── Exit cost: one-time event on the day exposure decreases ──────────
                prev_exp      = exposure.shift(1).fillna(1.0)
                exposure_drop = (prev_exp - exposure).clip(lower=0.0)
                exit_events   = exposure_drop > 0
                port_ret[exit_events] -= (
                        exposure_drop[exit_events] * (CFG.transaction_cost_bps / 10_000)
                )

            if log_dma50:
                self._log_dma50_events(
                    exposure, sub, weights, port_ret,
                    invest_mask=invest_mask_2d,
                )

        # ── Transaction cost (both modes) ─────────────────────────────────
        turnover = (np.sum(np.abs(weights - prev_weights))
                    if prev_weights is not None else np.sum(np.abs(weights)))
        tx_cost  = turnover * (CFG.transaction_cost_bps / 10_000)
        if len(port_ret) > 0:
            port_ret.iloc[0] -= tx_cost

        return port_ret

    # ── Exit helpers (eval-only) ──────────────────────────────────────────

    def _apply_stock_exits(
            self,
            sub:               pd.DataFrame,
            weights:           np.ndarray,
            apply_sector_exit: bool,
            apply_ticker_exit: bool,
    ) -> pd.Series:
        """
        Build a per-date effective weight matrix after applying sector and ticker
        exit masks, then return the resulting daily return series.

        Freed weight (zeroed-out stocks) is sent to CASH — it earns the daily
        risk-free rate and does NOT redistribute to surviving stocks.  The
        original configured weights are preserved for those stocks (no row
        renormalisation).  An exit transaction cost (transaction_cost_bps) is
        charged on the day a stock first transitions into the exited state.

        PERF: sector cache is pre-loaded (~4 tickers × n_dates — fits in memory).
        Ticker SMA50 is computed once via rolling mean on the existing price matrix.
        All masking is vectorised; no per-row Python loops.
        """
        dates    = sub.index
        tickers  = list(sub.columns)
        sub_arr  = sub.to_numpy()                            # n_dates × n_tickers
        n_dates  = len(dates)

        # Binary mask: 1.0 = invested, 0.0 = exited to cash
        invest_mask = np.ones((n_dates, len(tickers)), dtype=np.float64)

        if apply_sector_exit and SECTOR_MOMENTUM_CACHE:
            invest_mask *= self._compute_sector_exit_matrix(dates, tickers)

        if apply_ticker_exit and SECTOR_MOMENTUM_CACHE:
            invest_mask *= self._compute_ticker_exit_matrix(dates, tickers)

        # ── Effective weights: base weights × invest_mask (no renormalisation) ─
        # Rows may sum to < 1.0; the remainder earns daily risk-free rate.
        eff_w        = weights * invest_mask               # broadcast weights over dates
        invested_sum = eff_w.sum(axis=1)                   # shape (n_dates,)
        cash_frac    = np.clip(1.0 - invested_sum, 0.0, 1.0)  # weight parked in cash

        # ── Equity component ─────────────────────────────────────────────────
        equity_arr = np.einsum("ij,ij->i", sub_arr, eff_w)

        # ── Cash component: risk-free rate on the cash fraction ──────────────
        daily_rfr  = CFG.risk_free_rate / 252
        cash_arr   = cash_frac * daily_rfr

        # ── Exit transaction cost ────────────────────────────────────────────
        # Charged once on the first day a stock enters the exited state.
        # prev_mask: previous day's invest_mask (shift by 1, default fully invested)
        prev_mask     = np.vstack([np.ones((1, len(tickers))), invest_mask[:-1]])
        newly_exited  = (prev_mask - invest_mask).clip(min=0.0)   # 1 on first exit day
        # Cost = sum of base weights of newly exited stocks × tx_cost_bps
        exit_cost_arr = (newly_exited * weights).sum(axis=1) * (CFG.transaction_cost_bps / 10_000)

        port_arr = equity_arr + cash_arr - exit_cost_arr
        return pd.Series(port_arr, index=sub.index)

    def _compute_sector_exit_matrix(
            self,
            dates:   "pd.DatetimeIndex",
            tickers: List[str],
    ) -> np.ndarray:
        """
        Returns float array (n_dates × n_tickers).
        0.0 on days when the sector-index close is below its SMA50 (1-day lag),
        1.0 otherwise.

        SECTOR_MOMENTUM_CACHE is keyed by the parquet 'name' column, which must
        match SECTOR_MAP values — no external mapping required.

        Flag: enable_sector_exit
        """
        n, m       = len(dates), len(tickers)
        mask       = np.ones((n, m), dtype=np.float64)
        exit_sma50 = getattr(CFG, "exit_sma50_threshold", 0.0)
        entry_sma50  = getattr(CFG, "entry_sma50_threshold", 1.0)

        for sec_name, sec_df in SECTOR_MOMENTUM_CACHE.items():
            sec_aligned = sec_df.reindex(dates, method="ffill").shift(1)
            close_arr   = sec_aligned["close"].to_numpy()
            sma50_arr   = sec_aligned["sma50"].to_numpy()
            sma50_safe  = np.where(sma50_arr != 0, sma50_arr, np.nan)
            sma50_pos   = (close_arr - sma50_safe) / sma50_safe  # pct above/below SMA50

            # ── Step 1: stateless 4-tier cap per day ─────────────────────────
            t1, t2, t4, t8 = exit_sma50, exit_sma50 * 2, exit_sma50 * 4, exit_sma50 * 8
            cap = np.where(sma50_pos < t8,  0.00,
                           np.where(sma50_pos < t4,  0.25,
                                    np.where(sma50_pos < t2,  0.50,
                                             np.where(sma50_pos < t1,  0.75,
                                                      1.00))))

            # ── Step 2: hysteresis forward-pass ──────────────────────────────
            # Mirrors compute_exposure_series exactly:
            #   - while below exit threshold: ratchet down to the lowest cap seen
            #   - only reset to 1.0 when sma50_pos >= entry_sma50_threshold
            sec_exposure = np.empty(n, dtype=np.float64)
            prev = 1.0
            for i in range(n):
                c = cap[i]
                if c < 1.0:
                    prev = min(prev, c)                          # ratchet down
                elif prev < 1.0 and sma50_pos[i] >= entry_sma50:
                    prev = 1.0                                   # confirmed re-entry
                # else: buffer zone — hold current level
                sec_exposure[i] = prev
            col_mask = np.array(
                [SECTOR_MAP.get(t, "Other") == sec_name for t in tickers], dtype=bool
            )
            if col_mask.any():
                mask[:, col_mask] *= sec_exposure[:, np.newaxis]

        return mask

    def _compute_ticker_exit_matrix(
            self,
            dates:   "pd.DatetimeIndex",
            tickers: List[str],
    ) -> np.ndarray:
        """
        Returns float array (n_dates × n_tickers).
        0.0 when ticker RS (close / rolling_sma50) < sector RS × threshold.

        Ticker SMA50 is computed from the existing price matrix (rolling 50-day).
        Sector RS comes from SECTOR_MOMENTUM_CACHE keyed by sector name —
        same key as SECTOR_MAP values, no extra mapping needed.

        Flag:       enable_ticker_exit
        Config key: ticker_exit_rs_threshold (float, default 0.95)
        """
        exit_sma50 = getattr(CFG, "exit_sma50_threshold", 0.0)
        threshold  = getattr(CFG, "ticker_exit_rs_threshold", 0.95)
        n, m       = len(dates), len(tickers)
        mask       = np.ones((n, m), dtype=np.float64)

        # Ticker RS computed once for all tickers from the price matrix
        prices_sub = self.prices.reindex(columns=tickers)
        sma50_full = prices_sub.rolling(50, min_periods=10).mean()
        close_dt   = prices_sub.reindex(dates, method="ffill").shift(1)
        sma50_dt   = sma50_full.reindex(dates, method="ffill").shift(1)
        sma50_safe = sma50_dt.where((sma50_dt != 0) & sma50_dt.notna(), other=np.nan)
        ticker_rs  = (close_dt / sma50_safe).fillna(1.0)   # DataFrame[date × ticker]

        for j, t in enumerate(tickers):
            sec_name = SECTOR_MAP.get(t, "Other")
            sec_df   = SECTOR_MOMENTUM_CACHE.get(sec_name)
            if sec_df is None:
                continue

            sec_aligned = sec_df.reindex(dates, method="ffill").shift(1)
            sma50_s     = sec_aligned["sma50"].where(
                sec_aligned["sma50"] != 0, other=np.nan)
            sec_rs      = (sec_aligned["close"] / sma50_s).fillna(1.0).to_numpy()

            # Relative shortfall: how far below sector_rs * threshold is the ticker RS.
            # Expressed as a fraction of the trip level so the same tier multiples apply.
            # shortfall = 0  → at the trip level exactly
            # shortfall > 0  → below the trip level
            trip        = sec_rs * threshold           # absolute RS trip level
            trip_safe   = np.where(trip != 0, trip, np.nan)
            shortfall   = (trip - ticker_rs[t].to_numpy()) / np.abs(trip_safe)
            shortfall   = np.nan_to_num(shortfall, nan=0.0)

            # Map shortfall to the same 4-tier scalar as the sector/index:
            #   shortfall <= 0           → fully invested (1.00)  — above trip level
            #   shortfall in (0,  |t1|]  → 0.75
            #   shortfall in (|t1|,|t2|] → 0.50
            #   shortfall in (|t2|,|t4|] → 0.25
            #   shortfall >  |t4|        → 0.00
            # exit_sma50 is negative (e.g. -0.05); use its absolute value for spacing.
            band = abs(exit_sma50) if exit_sma50 != 0 else 0.05
            ticker_exposure = np.where(shortfall <= 0,       1.00,
                                       np.where(shortfall <= band,     0.75,
                                                np.where(shortfall <= band * 2, 0.50,
                                                         np.where(shortfall <= band * 4, 0.25,
                                                                  0.00))))
            mask[:, j] *= ticker_exposure

        return mask

    def _log_dma50_events(
            self,
            exposure:  pd.Series,
            sub:       pd.DataFrame,
            weights:   np.ndarray,
            port_ret:  pd.Series,
            invest_mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        Record portfolio-level DMA50 EXIT / REENTRY events into self.dma50_log,
        AND record sector/ticker REENTRY events (re-entry of previously exited
        stocks) alongside them so the full log captures all cash-transition events.

        PERF: vectorised boolean masks + cumsum spell identification.
        """
        n_total   = len(exposure)
        exp_arr   = exposure.to_numpy()
        prev_arr  = exposure.shift(1).fillna(1.0).to_numpy()

        exit_mask    = (exp_arr < 1.0) & (prev_arr == 1.0)
        reentry_mask = (exp_arr == 1.0) & (prev_arr < 1.0)

        idx        = exposure.index
        _daily_rfr = CFG.risk_free_rate / 252

        # Tag every sub-1 day with a spell ID via cumsum on exit events
        spell_id     = np.cumsum(exit_mask)       # increments at each new EXIT
        in_cash      = exp_arr < 1.0              # boolean mask for cash days
        equity_daily = (sub.to_numpy() @ weights) # shape (N,) — pre-compute once

        # ── Portfolio-level EXIT events ───────────────────────────────────────
        for exit_i in np.where(exit_mask)[0]:
            sid           = spell_id[exit_i]
            spell_indices = np.where((spell_id == sid) & in_cash)[0]
            n_cash        = len(spell_indices)
            cash_pct      = round(n_cash / n_total * 100, 1) if n_total > 0 else 0.0
            avoided_ret   = round((np.prod(1.0 + equity_daily[spell_indices]) - 1.0) * 100, 2)
            self.dma50_log.append({
                "date":               idx[exit_i].date(),
                "event":              "EXIT",
                "exposure":           round(float(exp_arr[exit_i]), 2),
                "cash_days":          n_cash,
                "cash_pct_of_period": cash_pct,
                "avoided_equity_ret": avoided_ret,
            })

        # ── Portfolio-level REENTRY events ────────────────────────────────────
        for reentry_i in np.where(reentry_mask)[0]:
            sid           = spell_id[reentry_i - 1] if reentry_i > 0 else 0
            spell_indices = np.where((spell_id == sid) & in_cash)[0]
            n_cash        = len(spell_indices)
            cash_earned   = round(
                (np.prod(1.0 + _daily_rfr * (1.0 - exp_arr[spell_indices])) - 1.0) * 100, 2
            )
            self.dma50_log.append({
                "date":               idx[reentry_i].date(),
                "event":              "REENTRY",
                "exposure":           1.0,
                "cash_days":          n_cash,
                "cash_pct_of_period": round(n_cash / n_total * 100, 1),
                "cash_return_earned": cash_earned,
            })

        # ── Sector / Ticker REENTRY events ────────────────────────────────────
        # invest_mask is the combined sector+ticker binary matrix (n_dates × n_tickers).
        # A re-entry occurs when a stock transitions from 0 → 1 (comes back in from cash).
        if invest_mask is not None and invest_mask.ndim == 2:
            tickers_list = list(sub.columns)
            n_tickers    = len(tickers_list)
            prev_inv     = np.vstack([np.ones((1, n_tickers)), invest_mask[:-1]])
            # newly_reentered[i, j] == 1 when stock j transitions 0→1 on day i
            newly_reentered = ((prev_inv == 0.0) & (invest_mask == 1.0))  # bool array

            for date_i, ticker_j in zip(*np.where(newly_reentered)):
                ticker_name = tickers_list[ticker_j]
                # Count consecutive cash days immediately preceding this re-entry
                d = date_i - 1
                while d >= 0 and invest_mask[d, ticker_j] == 0.0:
                    d -= 1
                cash_days = date_i - d - 1
                self.dma50_log.append({
                    "date":               idx[date_i].date(),
                    "event":              "REENTRY",
                    "source":             "sector_ticker",
                    "ticker":             ticker_name,
                    "sector":             SECTOR_MAP.get(ticker_name, "Other"),
                    "exposure":           1.0,
                    "cash_days":          cash_days,
                    "cash_pct_of_period": round(cash_days / n_total * 100, 1) if n_total > 0 else 0.0,
                    "cash_return_earned": round(
                        ((1 + _daily_rfr) ** cash_days - 1) * weights[ticker_j] * 100, 4
                    ),
                })

    def print_dma50_log(self):
        """Print a formatted table of all recorded EXIT / REENTRY events."""
        if not self.dma50_log:
            print("  [DMA50 Log] No exit/reentry events recorded.")
            return
        events = sorted(self.dma50_log, key=lambda r: r["date"])
        print("\n" + "=" * 76)
        print("DMA-50 / SMA EXIT & REENTRY LOG")
        print("=" * 76)
        print(f"  {'Date':<14} {'Event':<10} {'Exposure':>9} {'Cash Days':>10} {'% Period':>9}  {'Return':>10}")
        print("  " + "-" * 68)
        for row in events:
            if row["event"] == "EXIT":
                ret_val   = row.get("avoided_equity_ret", 0)
                ret_label = f"{ret_val:>+9.2f}%  ← equity avoided"
            else:
                ret_val   = row.get("cash_return_earned", 0)
                ret_label = f"{ret_val:>+9.2f}%  ← cash earned"
            print(
                f"  {str(row['date']):<14} {row['event']:<10} "
                f"{row['exposure']:>9.2f} {row['cash_days']:>10} "
                f"{row['cash_pct_of_period']:>8.1f}%  {ret_label}"
            )
        # Summary
        exits    = [r for r in events if r["event"] == "EXIT"]
        reentries= [r for r in events if r["event"] == "REENTRY"]
        total_cash = sum(r["cash_days"] for r in exits)
        print("  " + "-" * 68)
        print(f"  Exits: {len(exits)}   Reentries: {len(reentries)}   "
              f"Total cash days: {total_cash}")
        print("=" * 76)

    def cagr(self, ret: pd.Series) -> float:
        if len(ret) == 0:
            return np.nan
        cum     = (1 + ret).prod()
        n_years = len(ret) / 252
        return np.nan if n_years <= 0 else cum ** (1 / n_years) - 1

    def simple_return(self, ret: pd.Series) -> float:
        """Simple % change: (Final / Initial) - 1  =  cumulative product - 1.
        This is the total period return with no annualisation, replacing CAGR/XIRR
        for all period-level performance metrics and alpha calculations."""
        if len(ret) == 0:
            return np.nan
        return float((1 + ret).prod() - 1)

    def sharpe(self, ret: pd.Series) -> float:
        if ret.std() == 0 or len(ret) < 10:
            return 0.0
        excess = ret - CFG.risk_free_rate / 252
        return excess.mean() / excess.std() * np.sqrt(252)

    def sortino(self, ret: pd.Series) -> float:
        downside = ret[ret < 0]
        if len(downside) < 2 or downside.std() == 0:
            return 0.0
        return (ret.mean() - CFG.risk_free_rate / 252) / downside.std() * np.sqrt(252)

    def calmar(self, ret: pd.Series) -> float:
        total_ret = self.simple_return(ret)   # using simple % change per global metric policy
        cum       = (1 + ret).cumprod()
        max_dd    = abs(((cum - cum.cummax()) / cum.cummax()).min())
        return 0.0 if max_dd == 0 else (total_ret / max_dd if max_dd != 0 else 0.0)

    def max_drawdown(self, ret: pd.Series) -> float:
        cum = (1 + ret).cumprod()
        return ((cum - cum.cummax()) / cum.cummax()).min()



# ─────────────────────────────────────────────────────────────────────────────
# 5-A.  MAX DRAWDOWN  (standalone helper — used by FitnessEvaluator & ga_eval)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_mdd(ret: pd.Series) -> float:
    """Max Drawdown (%): the largest peak-to-trough decline in the cumulative
    return series.  Returns a negative float, e.g. -0.25 means -25% drawdown.
    Returns NaN for empty series."""
    if len(ret) == 0:
        return float("nan")
    cum    = (1 + ret).cumprod()
    dd     = (cum - cum.cummax()) / cum.cummax()
    return float(dd.min())          # most negative value = worst drawdown


# ─────────────────────────────────────────────────────────────────────────────
# 5.  INDIVIDUAL (CHROMOSOME)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Individual:
    weights: np.ndarray
    fitness: float = -np.inf
    metrics: Dict  = field(default_factory=dict)

    def normalise(self, n: int):
        """Project weights onto the weight-constrained simplex."""
        w = np.clip(self.weights, 0.0, CFG.max_weight)
        if w.sum() == 0:
            w = np.ones(n) / n

        # Sparsity gate: keep only TOP_K_HOLDINGS
        if CFG.top_k_holdings > 0 and n > CFG.top_k_holdings:
            w[np.argsort(w)[:-CFG.top_k_holdings]] = 0.0

        total = w.sum()
        w = w / total if total > 0 else np.ones(n) / n

        # Active-holding floor
        if CFG.min_weight_active > 0:
            active = w > 0
            if active.sum() > 0 and active.sum() * CFG.min_weight_active <= 1.0:
                w[active] = np.clip(w[active], CFG.min_weight_active, CFG.max_weight)

        # Iterative projection to respect MAX_WEIGHT
        for _ in range(50):
            total = w.sum()
            if total == 0:
                break
            w /= total
            over  = w > CFG.max_weight + 1e-9
            under = (w < CFG.min_weight_active) & (w > 0)
            w[over]  = CFG.max_weight
            w[under] = CFG.min_weight_active
            if not over.any() and not under.any():
                break

        total = w.sum()
        if total > 0:
            w /= total
        self.weights = w


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FITNESS EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────
class FitnessEvaluator:
    def __init__(
            self,
            engine:        ReturnsEngine,
            tickers:       List[str],
            nsei_features: pd.DataFrame,
            train_periods: Optional[List[Dict]] = None,
    ):
        self.engine        = engine
        self.tickers       = tickers
        self.nsei_features = nsei_features
        self.regimes       = regime_series(nsei_features)
        self.train_periods = train_periods if train_periods is not None else TRAIN_PERIODS
        self._index_ret_cache: Dict[str, pd.Series] = {}
        # Ensure the engine passed in has nsei_features attached for exposure filtering
        if self.engine.nsei_features is None:
            self.engine.nsei_features = nsei_features

        # ── Pre-build ticker-indexed arrays for vectorised penalty ops ─────────
        # Computed ONCE here instead of inside every evaluate() call.

        # Split tickers into mapped vs unmapped; ignore unmapped and log them.
        unmapped = [t for t in tickers if not SECTOR_MAP.get(t)]
        if unmapped:
            _emit(f"[FitnessEvaluator] ⚠️  {len(unmapped)} ticker(s) have no sector mapping "
                  f"— excluded from sector penalty: {unmapped}")
        tickers = [t for t in tickers if SECTOR_MAP.get(t)]
        self.tickers = tickers   # update so evaluate() uses filtered list

        _sector_list = sorted(set(SECTOR_MAP[t] for t in tickers))
        self._sector_list   = _sector_list
        self._ticker_sector = np.array(
            [_sector_list.index(SECTOR_MAP[t]) for t in tickers], dtype=np.int32
        )


        # cap_ids: 0=Large-cap, 1=Mid-cap, 2=Small-cap, 3=Unknown
        _CAP_ORDER = {"Large-cap": 0, "Mid-cap": 1, "Small-cap": 2}
        self._ticker_cap = np.array(
            [_CAP_ORDER.get(MARKET_CAP_MAP.get(t, "Unknown"), 3) for t in tickers], dtype=np.int32
        )
        self._ticker_has_cap = self._ticker_cap < 3   # boolean mask for known tickers

        # domestic pct array (NaN for unknown)
        self._ticker_dom = np.array(
            [DOMESTIC_PCT_MAP.get(t, np.nan) for t in tickers], dtype=np.float64
        )
        self._ticker_has_dom = ~np.isnan(self._ticker_dom)

    def _index_returns(self, start: str, end: str) -> pd.Series:
        key = f"{start}|{end}"
        if key not in self._index_ret_cache:
            # Index benchmark: no exposure filter (pass nsei_features=None)
            idx_engine = ReturnsEngine(self.engine.prices, nsei_features=None)
            self._index_ret_cache[key] = idx_engine.portfolio_returns(
                np.array([1.0]), [CFG.index_ticker], start, end
            )
        return self._index_ret_cache[key]

    def evaluate(self, ind: Individual) -> Individual:
        w = ind.weights
        e = self.engine
        period_alphas, period_results = [], []

        for p in self.train_periods:
            port_ret = e.portfolio_returns(w, self.tickers, p["start"], p["end"])
            idx_ret  = self._index_returns(p["start"], p["end"])
            if len(port_ret) < 5:
                continue
            port_pct = e.simple_return(port_ret)   # total % change for the period
            idx_pct  = e.simple_return(idx_ret)
            mdd      = calculate_mdd(port_ret)
            alpha = (port_pct - idx_pct
                     if not (np.isnan(port_pct) or np.isnan(idx_pct)) else np.nan)
            period_results.append({
                "name":     p["name"], "type": p["type"],
                "port_ret": port_pct,            # simple % change (replaces port_cagr)
                "idx_ret":  idx_pct,             # simple % change (replaces idx_cagr)
                "alpha":    alpha,               # arithmetic difference in total % return
                "mdd":      mdd,                 # max drawdown for this period
                "beaten":   alpha > 0 if not np.isnan(alpha) else False,
                "ret":      port_ret,
            })
            if not np.isnan(alpha):
                period_alphas.append(alpha)

        if not period_results:
            ind.fitness = -np.inf
            return ind

        all_ret = pd.concat([pr["ret"] for pr in period_results]).sort_index()
        all_ret = all_ret[~all_ret.index.duplicated(keep="first")]

        sharpe_  = e.sharpe(all_ret)
        sortino_ = e.sortino(all_ret)
        calmar_  = e.calmar(all_ret)

        avg_alpha  = np.nanmean(period_alphas) if period_alphas else -0.5
        beats      = sum(pr["beaten"] for pr in period_results)
        beat_frac  = beats / len(period_results)
        avg_mdd    = float(np.nanmean([pr["mdd"] for pr in period_results
                                       if not np.isnan(pr.get("mdd", float("nan")))]))
        crash_alphas    = [pr["alpha"] for pr in period_results
                           if pr["type"] == "crash" and not np.isnan(pr["alpha"])]
        bear_alphas     = [pr["alpha"] for pr in period_results
                           if pr["type"] == "bear" and not np.isnan(pr["alpha"])]
        recovery_alphas = [pr["alpha"] for pr in period_results
                           if pr["type"] == "recovery" and not np.isnan(pr["alpha"])]
        bull_alphas     = [pr["alpha"] for pr in period_results
                           if pr["type"] == "bull" and not np.isnan(pr["alpha"])]
        crash_alpha    = np.nanmean(crash_alphas + bear_alphas) if (crash_alphas or bear_alphas) else 0.0
        recovery_alpha = np.nanmean(recovery_alphas)            if recovery_alphas               else 0.0

        def sigmoid(x, scale=5.0):
            return 1 / (1 + math.exp(-scale * x))

        score_alpha   = sigmoid(avg_alpha, scale=10)
        score_beat    = beat_frac
        score_sharpe  = sigmoid(sharpe_ - 0.5, scale=1.5)
        score_sortino = sigmoid(sortino_ - 0.6, scale=1.5)
        score_calmar  = sigmoid(calmar_ - 0.3, scale=2.0)

        sector_penalty   = self._sector_penalty(w)
        mdd_penalty      = self._mdd_penalty(avg_mdd)
        dominant_regime  = self._dominant_regime()
        cap_tilt_penalty = self._cap_tilt_penalty(w, dominant_regime)
        domestic_penalty = self._domestic_concentration_penalty(w)

        penalty_total = (
                getattr(CFG, "sector_cap_penalty_weight",  0.05) * sector_penalty
                + getattr(CFG, "mdd_penalty_weight",         0.10) * mdd_penalty
                + getattr(CFG, "cap_tilt_penalty_weight",    0.05) * cap_tilt_penalty
                + getattr(CFG, "domestic_penalty_weight",    0.05) * domestic_penalty
        )
        #Regime w_alpha w_sharpe    w_sortino   w_calmar
        # bear      0.25    0.25    0.30    0.20
        # bull      0.35    0.30    0.20    0.15
        # recovery  0.30    0.28    0.25    0.17
        fitness = (
                CFG.w_alpha   * (0.6 * score_alpha + 0.4 * score_beat)
                + CFG.w_sharpe  * score_sharpe
                + CFG.w_sortino * score_sortino
                + CFG.w_calmar  * score_calmar
                - penalty_total
        )
        ind.fitness = float(fitness)
        ind.metrics = {
            "period_results":   period_results,
            "beats":            beats,
            "total_periods":    len(period_results),
            "avg_alpha":        avg_alpha,
            "crash_alpha":      crash_alpha,
            "recovery_alpha":   recovery_alpha,
            "bull_alpha":       np.nanmean(bull_alphas) if bull_alphas else 0.0,
            "sharpe":           sharpe_,
            "sortino":          sortino_,
            "calmar":           calmar_,
            "score_alpha":      score_alpha,
            "score_sharpe":     score_sharpe,
            "score_sortino":    score_sortino,
            "score_calmar":     score_calmar,
            "avg_mdd":          avg_mdd,
            "dominant_regime":  dominant_regime,
            "penalty_sector":   sector_penalty,
            "penalty_cap_tilt": cap_tilt_penalty,
        }
        return ind

    def _sector_penalty(self, weights: np.ndarray) -> float:
        """
        PERF: replaced per-ticker dict loop with numpy bincount (O(N) vectorised).
        _ticker_sector array built once in __init__, not per evaluate() call.
        """
        if not getattr(CFG, "enable_sector_cap", True):
            return 0.0
        cap        = CFG.max_sector_weight
        n_sectors  = len(self._sector_list)
        sector_wts = np.bincount(self._ticker_sector, weights=weights, minlength=n_sectors)
        excess     = np.maximum(sector_wts - cap, 0.0)
        return float(np.sum(excess ** 2))

    # ── Constraint 2: MDD non-linear penalty ─────────────────────────────────
    def _mdd_penalty(self, avg_mdd: float) -> float:
        if not getattr(CFG, "enable_mdd_penalty", True):
            return 0.0
        threshold = getattr(CFG, "mdd_penalty_threshold", -0.15)
        if np.isnan(avg_mdd) or avg_mdd >= threshold:
            return 0.0
        excess = threshold - avg_mdd
        return excess ** 3

    def _dominant_regime(self) -> str:
        if len(self.regimes) == 0:
            return "bull"
        return str(self.regimes.value_counts().idxmax())

    # ── Constraint: Market-cap tilt — vectorised ──────────────────────────────
    def _cap_tilt_penalty(self, weights: np.ndarray, regime: str) -> float:
        """
        Penalises deviations from the target market-cap allocation for the
        current regime.  Targets are read from CFG (cap_tilt_{regime}_{large|mid|small})
        with sensible defaults if not configured.

        PERF: replaced per-ticker dict loop with numpy masking + bincount.
        _ticker_cap / _ticker_has_cap arrays built once in __init__.
        """
        if not getattr(CFG, "enable_cap_tilt_penalty", True):
            return 0.0
        if not MARKET_CAP_MAP:
            return 0.0

        _DEFAULTS: Dict[str, Dict[str, float]] = {
            "bear":     {"Large-cap": 0.85, "Mid-cap": 0.10, "Small-cap": 0.05},
            "crash":    {"Large-cap": 0.85, "Mid-cap": 0.10, "Small-cap": 0.05},
            "bull":     {"Large-cap": 0.85, "Mid-cap": 0.10, "Small-cap": 0.05},
            "recovery": {"Large-cap": 0.85, "Mid-cap": 0.10, "Small-cap": 0.05},
        }
        targets = dict(_DEFAULTS.get(regime, _DEFAULTS["bear"]))
        for bucket, suffix in [("Large-cap", "large"), ("Mid-cap", "mid"), ("Small-cap", "small")]:
            key = f"cap_tilt_{regime}_{suffix}"
            if hasattr(CFG, key):
                targets[bucket] = float(getattr(CFG, key))

        known_w      = weights[self._ticker_has_cap]
        known_weight = known_w.sum()
        if known_weight < 0.01:
            return 0.0

        cap_ids_known = self._ticker_cap[self._ticker_has_cap]
        actuals = np.bincount(cap_ids_known, weights=known_w, minlength=3) / known_weight

        tgt = np.array([targets["Large-cap"], targets["Mid-cap"], targets["Small-cap"]])
        return float(np.sum((actuals - tgt) ** 2))

    # ── Constraint 4: Domestic concentration — vectorised ────────────────────
    def _domestic_concentration_penalty(self, weights: np.ndarray) -> float:
        """
        PERF: replaced per-ticker dict loop with pre-built numpy arrays.
        _ticker_dom / _ticker_has_dom built once in __init__.
        """
        if not getattr(CFG, "enable_domestic_penalty", False):
            return 0.0
        if not DOMESTIC_PCT_MAP:
            return 0.0

        target = getattr(CFG, "target_domestic_pct",  85.0)
        band   = getattr(CFG, "domestic_penalty_band",  5.0)

        w_known       = weights[self._ticker_has_dom]
        dom_known     = self._ticker_dom[self._ticker_has_dom]
        covered_w     = w_known.sum()
        if covered_w < 0.01:
            return 0.0

        actual_pct = float(np.dot(w_known, dom_known) / covered_w)
        excess     = max(0.0, abs(actual_pct - target) - band)
        return (excess / 100.0) ** 2


# ─────────────────────────────────────────────────────────────────────────────
# 7.  INITIALISER
# ─────────────────────────────────────────────────────────────────────────────
class Initialiser:
    def __init__(self, signals: pd.DataFrame, tickers: List[str], nsei_features: pd.DataFrame):
        self.signals       = signals
        self.tickers       = tickers
        self.nsei_features = nsei_features
        self.n             = len(tickers)

    def signal_seed(self, regime: str = "bull", cutoff_date: Optional[pd.Timestamp] = None) -> np.ndarray:
        sig = self.signals[self.signals["ticker"].isin(self.tickers)]
        if cutoff_date is not None:
            sig = sig[sig["date"] <= pd.Timestamp(cutoff_date)]
        latest = sig.sort_values("date").groupby("ticker").last().reindex(self.tickers)
        score_cols = [c for c in ["1Y", "6M", "3M", "rs", "momentum_quality"] if c in latest.columns]
        scores = latest[score_cols].mean(axis=1).fillna(0.0).values if score_cols else np.ones(self.n)
        bias = REGIME_SECTOR_BIAS.get(regime, {})
        multipliers = np.array([bias.get(SECTOR_MAP.get(t, "Other"), 1.0) for t in self.tickers])
        scores = np.clip(scores * multipliers, 0, None)
        if scores.sum() == 0:
            scores = np.ones(self.n)
        return scores / scores.sum()

    def random_individual(self) -> Individual:
        w   = np.random.dirichlet(np.ones(self.n) * 2)
        ind = Individual(weights=w)
        ind.normalise(self.n)
        return ind

    def seeded_individual(self, regime: str, noise: float = 0.3,
                          cutoff_date: Optional[pd.Timestamp] = None) -> Individual:
        base = self.signal_seed(regime, cutoff_date=cutoff_date)
        w    = np.clip(base + np.random.randn(self.n) * noise, 0, None)
        ind  = Individual(weights=w)
        ind.normalise(self.n)
        return ind

    def build_population(self, pop_size: int, dominant_regime: str,
                         cutoff_date: Optional[pd.Timestamp] = None) -> List[Individual]:
        pop = []
        for _ in range(int(pop_size * 0.40)):
            pop.append(self.seeded_individual(dominant_regime, cutoff_date=cutoff_date))
        for reg in ["bull", "bear", "crash", "recovery"]:
            for _ in range(int(pop_size * 0.05)):
                pop.append(self.seeded_individual(reg, noise=0.2, cutoff_date=cutoff_date))
        while len(pop) < pop_size:
            pop.append(self.random_individual())
        return pop[:pop_size]


# ─────────────────────────────────────────────────────────────────────────────
# 8.  GENETIC OPERATORS
# ─────────────────────────────────────────────────────────────────────────────
def tournament_selection(population: List[Individual], k: int = None) -> Individual:
    k = k or CFG.tournament_k
    contestants = random.sample(population, min(k, len(population)))
    return max(contestants, key=lambda x: x.fitness)


def sbx_crossover(p1: Individual, p2: Individual, eta: float = 2.0) -> Tuple[Individual, Individual]:
    """
    PERF: replaced element-wise Python loop with vectorised numpy ops.
    Speedup ~N× for large weight vectors.
    """
    n   = len(p1.weights)
    w1, w2 = p1.weights, p2.weights

    # Draw masks and uniform samples for all positions at once
    cross_mask = np.random.random(n) < 0.5
    u          = np.random.random(n)

    beta = np.where(
        u <= 0.5,
        (2 * u) ** (1.0 / (eta + 1)),
        (1.0 / (2.0 - 2.0 * u)) ** (1.0 / (eta + 1)),
        )

    c1 = np.where(cross_mask, 0.5 * ((1 + beta) * w1 + (1 - beta) * w2), w1)
    c2 = np.where(cross_mask, 0.5 * ((1 - beta) * w1 + (1 + beta) * w2), w2)

    child1 = Individual(weights=np.clip(c1, 0, None))
    child2 = Individual(weights=np.clip(c2, 0, None))
    child1.normalise(n)
    child2.normalise(n)
    return child1, child2


def gaussian_mutate(ind: Individual, sigma: float, prob: float) -> Individual:
    """
    PERF: replaced element-wise Python loop with vectorised numpy ops.
    """
    n    = len(ind.weights)
    mask = np.random.random(n) < prob
    w    = ind.weights.copy()
    w[mask] += np.random.normal(0.0, sigma, size=mask.sum())
    new_ind = Individual(weights=np.clip(w, 0, None))
    new_ind.normalise(n)
    return new_ind


def regime_bias_mutate(ind: Individual, tickers: List[str], regime: str) -> Individual:
    """
    PERF: replaced per-ticker Python loop with vectorised numpy ops.
    """
    bias  = REGIME_SECTOR_BIAS.get(regime, {})
    mults = np.array([bias.get(SECTOR_MAP.get(t, "Other"), 1.0) for t in tickers])

    w = ind.weights.copy()

    # Upward bias: w[i] *= 1 + (mult-1)*0.3*rand
    up      = mults > 1.0
    rand_up = np.random.random(up.sum())
    w[up]  *= 1.0 + (mults[up] - 1.0) * 0.3 * rand_up

    # Downward bias: w[i] *= mult + (1-mult)*0.5*rand
    down    = mults < 1.0
    rand_dn = np.random.random(down.sum())
    w[down] *= mults[down] + (1.0 - mults[down]) * 0.5 * rand_dn

    new_ind = Individual(weights=np.clip(w, 0, None))
    new_ind.normalise(len(tickers))
    return new_ind


# ─────────────────────────────────────────────────────────────────────────────
# 9.  GENETIC OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
class GeneticOptimizer:
    def __init__(
            self,
            tickers:       List[str],
            engine:        ReturnsEngine,
            evaluator:     FitnessEvaluator,
            initialiser:   Initialiser,
            nsei_features: pd.DataFrame,
    ):
        self.tickers       = tickers
        self.n             = len(tickers)
        self.engine        = engine
        self.evaluator     = evaluator
        self.initialiser   = initialiser
        self.nsei_features = nsei_features
        self.regimes       = regime_series(nsei_features)
        self.best_history: List[Individual] = []

    def _current_regime(self) -> str:
        return self.regimes.iloc[-1] if len(self.regimes) > 0 else "bull"

    def _mutation_params(self, gen: int, n_gen: int) -> Tuple[float, float]:
        t     = gen / n_gen
        prob  = (CFG.mutation_prob_final
                 + 0.5 * (CFG.mutation_prob_init - CFG.mutation_prob_final) * (1 + math.cos(math.pi * t)))
        sigma = (CFG.mutation_sigma_final
                 + 0.5 * (CFG.mutation_sigma_init - CFG.mutation_sigma_final) * (1 + math.cos(math.pi * t)))
        return prob, sigma

    def run(
            self,
            pop_size:    int = None,
            n_gen:       int = None,
            cutoff_date: Optional[pd.Timestamp] = None,
    ) -> Individual:
        pop_size = pop_size or CFG.pop_size
        n_gen    = n_gen    or CFG.n_generations
        regime   = self._current_regime()
        _emit(f"[GA] Dominant market regime: {regime.upper()}")
        _emit(f"[GA] Population={pop_size}  Generations={n_gen}  Tickers={self.n}\n")

        population = self.initialiser.build_population(pop_size, regime, cutoff_date=cutoff_date)
        population = [self.evaluator.evaluate(ind) for ind in population]
        population.sort(key=lambda x: x.fitness, reverse=True)
        best = deepcopy(population[0])

        for gen in range(1, n_gen + 1):
            prob_mut, sigma_mut = self._mutation_params(gen, n_gen)
            n_elite = max(2, int(pop_size * CFG.elite_frac))
            new_pop = [deepcopy(ind) for ind in population[:n_elite]]

            while len(new_pop) < pop_size:
                p1 = tournament_selection(population, CFG.tournament_k)
                p2 = tournament_selection(population, CFG.tournament_k)
                if random.random() < CFG.crossover_prob:
                    c1, c2 = sbx_crossover(p1, p2)
                else:
                    c1, c2 = deepcopy(p1), deepcopy(p2)
                c1 = gaussian_mutate(c1, sigma_mut, prob_mut)
                c2 = gaussian_mutate(c2, sigma_mut, prob_mut)
                if random.random() < 0.10:
                    c1 = regime_bias_mutate(c1, self.tickers, regime)
                if random.random() < 0.10:
                    c2 = regime_bias_mutate(c2, self.tickers, regime)
                new_pop.extend([c1, c2])

            new_pop = new_pop[:pop_size]
            for ind in new_pop[n_elite:]:
                self.evaluator.evaluate(ind)

            population = sorted(new_pop, key=lambda x: x.fitness, reverse=True)
            gen_best   = population[0]
            if gen_best.fitness > best.fitness:
                best = deepcopy(gen_best)
            self.best_history.append(deepcopy(gen_best))

            m = gen_best.metrics
            _emit(
                f"Gen {gen:03d} | Score={gen_best.fitness:.4f} | "
                f"Periods Beaten: {m.get('beats',0)}/{m.get('total_periods', len(TEST_PERIODS))} | "
                f"Avg Alpha={m.get('avg_alpha',0)*100:+.1f}% | "
                f"Crash Alpha={m.get('crash_alpha',0)*100:+.1f}% | "
                f"Recovery Alpha={m.get('recovery_alpha',0)*100:+.1f}%"
            )

        _logger.info(
            "[GA] Training complete — fitness=%.4f  beats=%d/%d  "
            "avg_alpha=%.2f%%  dominant_regime=%s",
            best.fitness,
            best.metrics.get("beats", 0),
            best.metrics.get("total_periods", 0),
            best.metrics.get("avg_alpha", 0.0) * 100,
            best.metrics.get("dominant_regime", "N/A"),
            )
        if _logger.isEnabledFor(logging.INFO):
            print_scorecard(best, self.tickers)

        return best


# ─────────────────────────────────────────────────────────────────────────────
# 10.  SCORECARD PRINTER  (shared utility)
# ─────────────────────────────────────────────────────────────────────────────
def print_scorecard(best: Individual, tickers: List[str]):
    _emit("\n" + "=" * 72)
    _emit("FINAL PORTFOLIO SCORECARD")
    _emit("=" * 72)
    m              = best.metrics
    period_results = m.get("period_results", [])

    _emit(f"\n{'Period':<28} {'Portfolio':>10} {'Index':>8} {'Alpha':>8} {'MDD':>8} {'✓/✗':>5}")
    _emit("-" * 74)

    beats, total                         = 0, 0
    crash_alphas, recovery_alphas, all_a = [], [], []

    for pr in period_results:
        pc_s    = f"{pr['port_ret']*100:+.1f}%" if not np.isnan(pr['port_ret']) else "N/A"
        ic_s    = f"{pr['idx_ret']*100:+.1f}%"  if not np.isnan(pr['idx_ret'])  else "N/A"
        alpha_s = f"{pr['alpha']*100:+.1f}%"     if not np.isnan(pr['alpha'])    else "N/A"
        mdd_s   = f"{pr.get('mdd', float('nan'))*100:.1f}%" if not np.isnan(pr.get('mdd', float('nan'))) else "N/A"
        mark    = "✓" if pr["beaten"] else "✗"
        note    = ""
        if any(k in pr["name"] for k in ("Crash", "Tariff", "Iran")):
            note = "  ← survived crash" if pr["beaten"] else "  ← lost in crash"
        elif "Recovery" in pr["name"]:
            note = "  ← recovery period"
        _emit(f"{pr['name']:<28} {pc_s:>10} {ic_s:>8} {alpha_s:>8} {mdd_s:>8}  {mark}{note}")
        if not np.isnan(pr["alpha"]):
            all_a.append(pr["alpha"])
            if pr["type"] in ("crash", "bear"):   crash_alphas.append(pr["alpha"])
            if pr["type"] == "recovery":           recovery_alphas.append(pr["alpha"])
        if pr["beaten"]: beats += 1
        total += 1

    _emit("-" * 74)
    _emit(f"\nPeriods Beaten   : {beats}/{total}")
    _emit(f"Avg Alpha        : {np.mean(all_a)*100:+.2f}%"    if all_a           else "Avg Alpha        : N/A")
    _emit(f"Crash/Bear Alpha : {np.mean(crash_alphas)*100:+.2f}%" if crash_alphas else "Crash/Bear Alpha : N/A")
    _emit(f"Recovery Alpha   : {np.mean(recovery_alphas)*100:+.2f}%" if recovery_alphas else "Recovery Alpha   : N/A")
    avg_mdd = m.get("avg_mdd", float("nan"))
    _emit(f"Avg MDD          : {avg_mdd*100:.2f}%" if not np.isnan(avg_mdd) else "Avg MDD          : N/A")
    _emit(f"Sharpe Ratio     : {m.get('sharpe', 0):.3f}")
    _emit(f"Sortino Ratio    : {m.get('sortino', 0):.3f}  ← primary risk-adjusted metric")
    _emit(f"Calmar Ratio     : {m.get('calmar', 0):.3f}")
    _emit(f"Dominant Regime  : {m.get('dominant_regime', 'N/A')}")
    _emit(f"Fitness Score    : {best.fitness:.4f}")

    _emit("\n" + "─" * 50)
    _emit("CONSTRAINT PENALTIES")
    _emit("─" * 50)
    _emit(f"  Sector Cap      : {m.get('penalty_sector',   0.0):.5f}"
          + ("  [disabled]" if not getattr(CFG, "enable_sector_cap", False) else ""))
    _emit(f"  MDD             : {m.get('penalty_mdd',      0.0):.5f}"
          + ("  [disabled]" if not getattr(CFG, "enable_mdd_penalty", False) else ""))
    _emit(f"  Cap Tilt        : {m.get('penalty_cap_tilt', 0.0):.5f}"
          + ("  [disabled]" if not getattr(CFG, "enable_cap_tilt_penalty", False) else ""))
    _emit(f"  Domestic Conc.  : {m.get('penalty_domestic', 0.0):.5f}"
          + ("  [disabled]" if not getattr(CFG, "enable_domestic_penalty", False) else ""))

    if MARKET_CAP_MAP:
        _emit("\n" + "=" * 72)
        _emit("MARKET-CAP ALLOCATION")
        _emit("=" * 72)
        cap_agg: Dict[str, float] = {}
        for i, t in enumerate(tickers):
            cap = MARKET_CAP_MAP.get(t, "Unknown")
            cap_agg[cap] = cap_agg.get(cap, 0.0) + best.weights[i]
        _emit(f"\n{'Cap Bucket':<16} {'Weight':>8}")
        _emit("-" * 28)
        for cap, wt in sorted(cap_agg.items(), key=lambda x: -x[1]):
            _emit(f"{cap:<16} {wt*100:>6.2f}%  {'█' * int(wt * 200)}")

    if DOMESTIC_PCT_MAP:
        weighted_dom, covered_w = 0.0, 0.0
        for i, t in enumerate(tickers):
            dom = DOMESTIC_PCT_MAP.get(t)
            if dom is not None:
                weighted_dom += best.weights[i] * dom
                covered_w    += best.weights[i]
        if covered_w > 0.01:
            actual_dom_pct = weighted_dom / covered_w
            target_dom_pct = getattr(CFG, "target_domestic_pct", 85.0)
            _emit(f"\nDomestic Concentration : {actual_dom_pct:.1f}%  "
                  f"(target {target_dom_pct:.0f}%)"
                  + ("  ✓" if abs(actual_dom_pct - target_dom_pct) <= getattr(CFG, "domestic_penalty_band", 5.0) else "  ✗ outside band"))

    _emit("\n" + "=" * 72)
    _emit("SECTOR WEIGHTS")
    _emit("=" * 72)
    sector_agg: Dict[str, float] = {}
    for i, t in enumerate(tickers):
        sec = SECTOR_MAP.get(t, "Other")
        sector_agg[sec] = sector_agg.get(sec, 0.0) + best.weights[i]
    _emit(f"\n{'Sector':<24} {'Weight':>8}")
    _emit("-" * 35)
    for sec, wt in sorted(sector_agg.items(), key=lambda x: -x[1]):
        _emit(f"{sec:<24} {wt*100:>6.2f}%  {'█' * int(wt * 200)}")

    _emit("\n" + "=" * 72)
    _emit("TICKER WEIGHTS (top 20)")
    _emit("=" * 72)
    _emit(f"\n{'Ticker':<32} {'Weight':>8} {'Sector':<20}")
    _emit("-" * 64)
    top = sorted(zip(tickers, best.weights), key=lambda x: -x[1])[:20]
    for t, wt in top:
        _emit(f"{t:<32} {wt*100:>6.2f}%  {SECTOR_MAP.get(t,'Other'):<20}  {'█'*int(wt*400)}")



# ─────────────────────────────────────────────────────────────────────────────
# 11.  WALK-FORWARD WEIGHT BLENDING
# ─────────────────────────────────────────────────────────────────────────────
#
# blend_weights()
#   Combines a newly-trained weight vector with the prior period's live weight
#   vector using a configurable alpha.  The result is re-normalised so weights
#   still sum to 1.0 and respect max_weight / min_weight_active constraints.
#
#   alpha = 1.0  → pure new weights   (standard hard replace, no blending)
#   alpha = 0.0  → pure prior weights (no rebalance at all)
#   alpha = 0.7  → 70% new, 30% prior (recommended default)
#
#   Ticker universe changes between periods are handled by zero-padding: a
#   ticker that appears in one vector but not the other is treated as weight=0
#   on the missing side.
#
# run_blended_walk_forward()
#   Drives a full FY simulation with monthly rebalance + blending.
#   Called from ga_train.py --walk-forward.
#
#   For each calendar month in the FY eval window:
#     1. Slice training data up to (but NOT including) the month start → no leakage.
#     2. Run GA → new_weights.
#     3. Blend with prior_weights using blend_alpha.
#     4. Simulate portfolio_returns() for the month slice (eval mode).
#     5. Record blended weights + returns for the month.
#   After all months, stitch returns into a single FY series and report.
#
# Config keys (add to ga_config.csv):
#   portfolio, blend_alpha,          0.7,  float, New-weight fraction in blended rebalance (0-1)
#   portfolio, blend_min_alpha,      0.5,  float, Minimum blend alpha (floors aggressive decay)
#   portfolio, blend_alpha_decay,    0.0,  float, Reduce alpha by this each month (0 = constant)
# ─────────────────────────────────────────────────────────────────────────────

def blend_weights(
        new_weights:   np.ndarray,
        new_tickers:   List[str],
        prior_weights: Optional[np.ndarray],
        prior_tickers: Optional[List[str]],
        alpha:         float = 0.7,
) -> np.ndarray:
    """
    Blend new GA weights with prior period weights.

    Args:
        new_weights:   Weight vector from the latest GA run (sums to 1).
        new_tickers:   Ticker list corresponding to new_weights.
        prior_weights: Weight vector from the previous month (sums to 1).
                       None on the first month → pure new weights returned.
        prior_tickers: Ticker list corresponding to prior_weights.
                       None when prior_weights is None.
        alpha:         Fraction assigned to new_weights (0 < alpha <= 1).
                       (1 - alpha) is assigned to prior_weights.

    Returns:
        Blended weight vector aligned to new_tickers, re-normalised,
        constraints applied via Individual.normalise().
    """
    if prior_weights is None or prior_tickers is None or alpha >= 1.0:
        # First month or full replacement — return new weights as-is
        return new_weights.copy()

    alpha = float(np.clip(alpha, 0.0, 1.0))

    # Build prior weight lookup keyed by ticker
    prior_map: Dict[str, float] = dict(zip(prior_tickers, prior_weights))

    # Align prior weights to the new ticker universe (zero for new/delisted)
    prior_aligned = np.array(
        [prior_map.get(t, 0.0) for t in new_tickers], dtype=np.float64
    )

    # Re-normalise prior_aligned so it sums to 1 over the new universe
    # (handles universe shrink/expand gracefully)
    prior_sum = prior_aligned.sum()
    if prior_sum > 1e-9:
        prior_aligned /= prior_sum
    else:
        # Prior has zero overlap with new universe → fall back to new weights
        return new_weights.copy()

    # Convex combination
    blended = alpha * new_weights + (1.0 - alpha) * prior_aligned
    blended = np.clip(blended, 0.0, None)

    # Re-normalise and apply portfolio constraints via Individual helper
    ind = Individual(weights=blended)
    ind.normalise(len(new_tickers))
    return ind.weights


def run_blended_walk_forward(
        fy_cfg:        dict,
        prices:        "pd.DataFrame",
        all_tickers:   List[str],
        nsei_features: "pd.DataFrame",
        signals:       "pd.DataFrame",
        out_dir:       "Path",
        blend_alpha:   Optional[float]  = None,
        n_gen:         Optional[int]    = None,
        pop_size:      Optional[int]    = None,
) -> "pd.DataFrame":
    """
    Monthly walk-forward simulation for one FY config with blended rebalancing.

    For each calendar month in fy_cfg['test'] window:
    • Train GA on all data up to (but not including) that month — no leakage.
    • Blend resulting weights with prior month's blended weights.
    • Simulate portfolio returns for that month in eval mode.
    • Log per-month stats.

Args:
fy_cfg:       One period dict from PERIOD_CONFIG (must have 'test' window).
prices:       Full price pivot DataFrame.
all_tickers:  Global ticker universe.
nsei_features: NSEI feature DataFrame.
signals:      Signal DataFrame.
out_dir:      Directory for output CSVs.
    blend_alpha:  Override blend alpha (else CFG.blend_alpha, default 0.7).
n_gen:        Override GA generations per month (else CFG.n_generations).
pop_size:     Override GA population size (else CFG.pop_size).

    Returns:
        DataFrame with one row per month: date, weights (dict), returns, metrics.
        Also writes walk_forward_<name>.csv to out_dir.
    """
    name     = fy_cfg["name"]
    test_cfg = fy_cfg["test"]
    fy_start = pd.Timestamp(test_cfg["start"])
    fy_end   = pd.Timestamp(test_cfg["end"])

    # ── Config knobs ──────────────────────────────────────────────────────────
    alpha         = blend_alpha  if blend_alpha is not None  else getattr(CFG, "blend_alpha",       0.7)
    alpha_decay   = getattr(CFG, "blend_alpha_decay",  0.0)
    alpha_min     = getattr(CFG, "blend_min_alpha",    0.5)
    n_gen_        = n_gen        if n_gen is not None         else CFG.n_generations
    pop_size_     = pop_size     if pop_size is not None      else CFG.pop_size

    _emit(f"\n{'='*72}")
    _emit(f"WALK-FORWARD BLEND: {name}")
    _emit(f"  FY window : {fy_start.date()} → {fy_end.date()}")
    _emit(f"  blend_alpha={alpha:.2f}  decay={alpha_decay:.3f}  min_alpha={alpha_min:.2f}")
    _emit(f"  GA: pop={pop_size_}  gen={n_gen_}")
    _emit(f"{'='*72}\n")

    # ── Build monthly rebalance dates (first trading day of each month) ───────
    # Generate month-start dates in the FY window.  We use business-day offsets
    # and snap to the nearest actual trading date in the price index.
    trading_days  = prices.index[(prices.index >= fy_start) & (prices.index <= fy_end)]
    if len(trading_days) == 0:
        _emit(f"  [SKIP] No trading data for {name} in test window.")
        return pd.DataFrame()

    # Month boundaries: first trading day of each calendar month in the window
    month_starts: List[pd.Timestamp] = []
    seen_months = set()
    for dt in trading_days:
        key = (dt.year, dt.month)
        if key not in seen_months:
            seen_months.add(key)
            month_starts.append(dt)

    _emit(f"  Monthly rebalance dates ({len(month_starts)} months):")
    for ms in month_starts:
        _emit(f"    {ms.date()}")
    _emit()

    # ── Period-specific tickers (anti-leakage, same as train_one_config) ─────
    period_tickers_raw: str = fy_cfg.get("tickers", "")
    if period_tickers_raw:
        requested = [t.strip() for t in period_tickers_raw.split(",") if t.strip()]
        tickers   = [t for t in requested if t in prices.columns]
    else:
        tickers = [t for t in all_tickers if t in prices.columns]

    if not tickers:
        _emit(f"  [SKIP] No valid tickers for {name}.")
        return pd.DataFrame()

    # ── Walk-forward loop ─────────────────────────────────────────────────────
    prior_weights: Optional[np.ndarray] = None
    prior_tickers: Optional[List[str]]  = None
    month_records: List[dict]           = []
    fy_returns:    List["pd.Series"]    = []
    current_alpha = float(alpha)

    engine_full = ReturnsEngine(prices, nsei_features=nsei_features)

    for m_idx, month_start in enumerate(month_starts):
        # End of this month slice = day before next month start (or fy_end)
        if m_idx + 1 < len(month_starts):
            month_end = month_starts[m_idx + 1] - pd.Timedelta(days=1)
        else:
            month_end = fy_end
        # Snap month_end to last actual trading day on or before it
        td_in_month = trading_days[(trading_days >= month_start) & (trading_days <= month_end)]
        if len(td_in_month) == 0:
            _emit(f"  [Month {m_idx+1}] No trading days — skipping.")
            continue
        month_end_actual = td_in_month[-1]

        # Training data: everything STRICTLY before this month → no leakage
        train_cutoff = month_start - pd.Timedelta(days=1)
        if train_cutoff < prices.index[0]:
            _emit(f"  [Month {m_idx+1}] {month_start.date()}: insufficient history — using prior weights only.")
            blended = prior_weights.copy() if prior_weights is not None else None
        else:
            # Slice train data
            price_cols   = [CFG.index_ticker] + [t for t in tickers if t != CFG.index_ticker]
            price_cols   = [c for c in price_cols if c in prices.columns]
            train_prices  = prices.loc[:train_cutoff, price_cols]
            train_nsei    = nsei_features.loc[:train_cutoff]
            train_signals = signals[signals["date"] <= train_cutoff]

            if len(train_prices) < 60:
                _emit(f"  [Month {m_idx+1}] {month_start.date()}: <60 training days — using prior weights.")
                blended = prior_weights.copy() if prior_weights is not None else None
            else:
                # Build regime windows for this train slice
                from ga_train import _regime_windows  # local import to avoid circular
                train_period_list = _regime_windows(train_nsei, f"{name}_m{m_idx+1}")
                train_period_list.append({
                    "name":  f"{name}_m{m_idx+1}_full",
                    "start": str(train_prices.index[0].date()),
                    "end":   str(train_cutoff.date()),
                    "type":  "bear",
                })

                # Run GA
                t_engine    = ReturnsEngine(train_prices, nsei_features=train_nsei)
                t_evaluator = FitnessEvaluator(t_engine, tickers, train_nsei,
                                               train_periods=train_period_list)
                t_init      = Initialiser(train_signals, tickers, train_nsei)
                ga          = GeneticOptimizer(tickers, t_engine, t_evaluator, t_init, train_nsei)

                best = ga.run(pop_size=pop_size_, n_gen=n_gen_, cutoff_date=train_cutoff)
                new_weights = best.weights

                # Blend with prior
                blended = blend_weights(
                    new_weights=new_weights,
                    new_tickers=tickers,
                    prior_weights=prior_weights,
                    prior_tickers=prior_tickers,
                    alpha=current_alpha,
                )

                _emit(f"\n  [Month {m_idx+1}] {month_start.date()}→{month_end_actual.date()} "
                      f"| alpha={current_alpha:.2f} "
                      f"| GA fitness={best.fitness:.4f}")

        if blended is None:
            _emit(f"  [Month {m_idx+1}] No weights available — skipping month.")
            continue

        # ── Simulate this month in eval mode ──────────────────────────────────
        prev_w = prior_weights if prior_weights is not None else np.zeros(len(tickers))
        # Align prev_w to current tickers if prior tickers differ
        if prior_tickers is not None and prior_tickers != tickers:
            prior_map = dict(zip(prior_tickers, prev_w))
            prev_w = np.array([prior_map.get(t, 0.0) for t in tickers])

        month_ret = engine_full.portfolio_returns(
            weights      = blended,
            tickers      = tickers,
            start        = str(month_start.date()),
            end          = str(month_end_actual.date()),
            prev_weights = prev_w,
            log_dma50    = False,
            mode         = "eval",
        )

        # ── Per-month metrics ─────────────────────────────────────────────────
        idx_ret_month = engine_full.portfolio_returns(
            weights   = np.array([1.0]),
            tickers   = [CFG.index_ticker],
            start     = str(month_start.date()),
            end       = str(month_end_actual.date()),
            log_dma50 = False,
            mode      = "train",   # raw index, no exposure filter
        )
        port_simple = engine_full.simple_return(month_ret)
        idx_simple  = engine_full.simple_return(idx_ret_month)
        month_alpha = port_simple - idx_simple
        turnover    = float(np.sum(np.abs(blended - prev_w)))

        # Weight snapshot: top-5 by blended weight
        top5 = sorted(zip(tickers, blended), key=lambda x: -x[1])[:5]
        top5_str = "; ".join(f"{t}:{w*100:.1f}%" for t, w in top5)

        record = {
            "month":        month_start.strftime("%Y-%m"),
            "month_start":  str(month_start.date()),
            "month_end":    str(month_end_actual.date()),
            "blend_alpha":  round(current_alpha, 3),
            "port_ret_pct": round(port_simple * 100, 3),
            "idx_ret_pct":  round(idx_simple  * 100, 3),
            "alpha_pct":    round(month_alpha  * 100, 3),
            "turnover_pct": round(turnover     * 100, 2),
            "tx_cost_bps":  round(turnover * CFG.transaction_cost_bps, 1),
            "top5_weights": top5_str,
        }
        # Store full blended weight vector as ticker→weight columns
        for t, w in zip(tickers, blended):
            record[f"w_{t}"] = round(float(w), 6)

        month_records.append(record)
        fy_returns.append(month_ret)

        # Update prior for next iteration
        prior_weights = blended.copy()
        prior_tickers = tickers[:]

        # Decay alpha
        current_alpha = max(alpha_min, current_alpha - alpha_decay)

    if not fy_returns:
        _emit(f"  [WARN] No monthly returns generated for {name}.")
        return pd.DataFrame()

    # ── Stitch FY return series ───────────────────────────────────────────────
    fy_ret = pd.concat(fy_returns).sort_index()
    fy_ret = fy_ret[~fy_ret.index.duplicated(keep="first")]

    fy_port_simple = engine_full.simple_return(fy_ret)
    fy_sharpe      = engine_full.sharpe(fy_ret)
    fy_sortino     = engine_full.sortino(fy_ret)
    fy_calmar      = engine_full.calmar(fy_ret)
    fy_mdd         = engine_full.max_drawdown(fy_ret)

    # FY index benchmark (raw, no exposure filter)
    fy_idx_ret = engine_full.portfolio_returns(
        weights   = np.array([1.0]),
        tickers   = [CFG.index_ticker],
        start     = str(fy_start.date()),
        end       = str(fy_end.date()),
        log_dma50 = False,
        mode      = "train",
    )
    fy_idx_simple = engine_full.simple_return(fy_idx_ret)

    # ── Print FY scorecard ────────────────────────────────────────────────────
    _emit(f"\n{'='*72}")
    _emit(f"WALK-FORWARD BLEND SCORECARD: {name}")
    _emit(f"{'='*72}")
    _emit(f"\n{'Month':<10} {'Port%':>8} {'Idx%':>8} {'Alpha%':>8} "
          f"{'Turnover%':>10} {'TxCost bps':>11} {'Blend α':>8}")
    _emit("─" * 72)
    for r in month_records:
        _emit(f"{r['month']:<10} {r['port_ret_pct']:>7.2f}% {r['idx_ret_pct']:>7.2f}% "
              f"{r['alpha_pct']:>+7.2f}% {r['turnover_pct']:>9.1f}% "
              f"{r['tx_cost_bps']:>10.1f}  {r['blend_alpha']:>7.2f}")
    _emit("─" * 72)
    _emit(f"{'FY TOTAL':<10} {fy_port_simple*100:>7.2f}% {fy_idx_simple*100:>7.2f}% "
          f"{(fy_port_simple-fy_idx_simple)*100:>+7.2f}%")
    _emit(f"\n  Sharpe   : {fy_sharpe:.3f}")
    _emit(f"  Sortino  : {fy_sortino:.3f}")
    _emit(f"  Calmar   : {fy_calmar:.3f}")
    _emit(f"  Max DD   : {fy_mdd*100:.2f}%")
    total_tx = sum(r["tx_cost_bps"] for r in month_records)
    total_to  = sum(r["turnover_pct"] for r in month_records)
    _emit(f"  Total turnover : {total_to:.1f}%  |  Total tx cost : {total_tx:.1f} bps")
    _emit(f"{'='*72}\n")

    # ── Persist per-month records ─────────────────────────────────────────────
    records_df = pd.DataFrame(month_records)
    safe_name  = name.replace("/", "-").replace(" ", "_")
    out_path   = out_dir / f"walk_forward_blend_{safe_name}.csv"
    # Write summary columns first (weight columns are wide — append at end)
    summary_cols = [c for c in records_df.columns if not c.startswith("w_")]
    weight_cols  = [c for c in records_df.columns if c.startswith("w_")]
    records_df[summary_cols + weight_cols].to_csv(out_path, index=False)
    _emit(f"  [Saved] Walk-forward blend → {out_path}")

    return records_df