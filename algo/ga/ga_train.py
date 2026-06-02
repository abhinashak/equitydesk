"""
=============================================================================
ga_train.py  —  Training Phase
=============================================================================
For each enabled period in config/periods.json:
  1. Slices data to the training window only (no leakage).
  2. Runs the GA.
  3. Writes trained weights to:
       outputs/weights_train_<config_name>.csv

Weights files are the hand-off artefacts consumed by ga_eval.py.

Usage:
    python ga_train.py
    python ga_train.py --config FY2024          # train a single config by name
    python ga_train.py --out-dir my_outputs     # override output directory
=============================================================================
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import everything from the shared module
from ga_common import (
    CFG,
    PERIOD_CONFIG,
    REGIME_CASE_SQL,
    SECTOR_MAP,
    DataLoader,
    FitnessEvaluator,
    GeneticOptimizer,
    Initialiser,
    ReturnsEngine,
    load_sector_momentum_cache,
    _emit,
)

import random as _random


def _regime_windows(train_nsei: pd.DataFrame, name: str) -> list:
    """
    Identify stable regime cycles inside the training window using DuckDB.
    Returns a de-duplicated, date-sorted list of period dicts ready for
    FitnessEvaluator (keys: name, start, end, type).
    """
    import duckdb

    # date is the DataFrame index — reset so DuckDB sees it as a named column
    train_nsei = train_nsei.reset_index()   # moves index → column named "date"

    sql = f"""
    WITH classified AS (
        SELECT date,
            {REGIME_CASE_SQL} AS regime
        FROM train_nsei
    ),
    changes AS (
        SELECT *,
            CASE WHEN regime != LAG(regime) OVER (ORDER BY date)
                      OR LAG(regime) OVER (ORDER BY date) IS NULL
                 THEN 1 ELSE 0 END AS new_grp
        FROM classified
    ),
    grouped AS (
        SELECT *, SUM(new_grp) OVER (ORDER BY date) AS grp FROM changes
    ),
    stable_cycles AS (
        SELECT MIN(date) AS start_date, MAX(date) AS end_date,
               ANY_VALUE(regime) AS regime, COUNT(*) AS trading_days
        FROM grouped
        GROUP BY grp
        HAVING COUNT(*) >= 7
    ),
    numbered AS (
        SELECT ROW_NUMBER() OVER (ORDER BY start_date) AS cycle_id,
               start_date, end_date, regime, trading_days,
               ROW_NUMBER() OVER (ORDER BY start_date)  AS seq_num,
               COUNT(*) OVER ()                          AS total_cycles
        FROM stable_cycles
    ),
    longest_per_regime AS (
        SELECT cycle_id, start_date, end_date, regime, trading_days
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY regime ORDER BY trading_days DESC) rn
            FROM numbered
        ) WHERE rn = 1
    )
    SELECT start_date, end_date, regime, trading_days FROM numbered
     WHERE seq_num <= 3
    UNION ALL
    SELECT start_date, end_date, regime, trading_days FROM numbered
     WHERE seq_num BETWEEN GREATEST(1, CAST(FLOOR(total_cycles / 2.0) AS BIGINT))
                       AND LEAST(total_cycles, CAST(FLOOR(total_cycles / 2.0) AS BIGINT) + 2)
    UNION ALL
    SELECT start_date, end_date, regime, trading_days FROM numbered
     WHERE seq_num > total_cycles - 3
    UNION ALL
    SELECT start_date, end_date, regime, trading_days FROM longest_per_regime
    ORDER BY start_date
    """
    print (sql)
    df = duckdb.query(sql).df()
    df = df.drop_duplicates(subset=["start_date", "end_date"]).sort_values("start_date")

    windows = []
    for _, row in df.iterrows():
        sd = pd.Timestamp(row["start_date"]).strftime("%Y-%m-%d")
        ed = pd.Timestamp(row["end_date"]).strftime("%Y-%m-%d")
        windows.append({
            "name":  f"{name}_{row['regime']}_{sd}",
            "start": sd,
            "end":   ed,
            "type":  row["regime"],
        })
    return windows


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ONE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
def train_one_config(
        cfg:           dict,
        prices:        pd.DataFrame,
        tickers:       list,
        nsei_features: pd.DataFrame,
        signals:       pd.DataFrame,
        out_dir:       Path,
) -> dict:
    """
    Train on cfg['train'] window only.
    Returns a summary dict and writes weights CSV to out_dir.
    """
    name        = cfg["name"]
    train_cfg   = cfg["train"]

    # ── Period-specific tickers (anti-leakage) ────────────────────────────────
    # If periods.json includes a "tickers" key for this config, use that explicit
    # universe instead of the global default.  This prevents look-ahead bias from
    # tickers that were only listed / liquid after the training cutoff date.
    period_tickers_raw: str | None = cfg.get("tickers")   # e.g. "RELIANCE,TCS,INFY"
    if period_tickers_raw:
        requested = [t.strip() for t in period_tickers_raw.split(",") if t.strip()]
        # Intersect with tickers that actually have price data in the parquet
        tickers = [t for t in requested if t in prices.columns]
        missing  = set(requested) - set(tickers)
        _emit(f"[Train] {name}: period-specific tickers — "
              f"{len(tickers)} valid, {len(missing)} missing from data"
              + (f": {sorted(missing)}" if missing else ""))
    else:
        # Fall back to the global universe passed in by main()
        _emit(f"[Train] {name}: using global ticker universe ({len(tickers)} tickers)")

    if not tickers:
        _emit(f"  [SKIP] No valid tickers for {name} after period filter — skipping.")
        return {"name": name, "skipped": True}

    # 1. Define the full training range
    start_dt = pd.Timestamp(train_cfg["start"])
    end_dt   = pd.Timestamp(train_cfg["end"])

    # 2. Slice data up front — needed for regime detection in sub-windows
    # Always include the index ticker so FitnessEvaluator._index_returns can benchmark
    # against it even when the period-specific ticker list doesn't contain it.
    price_cols  = [CFG.index_ticker] + [t for t in tickers if t != CFG.index_ticker]
    price_cols  = [c for c in price_cols if c in prices.columns]
    train_prices  = prices.loc[start_dt:end_dt, price_cols]
    train_nsei    = nsei_features.loc[start_dt:end_dt]
    train_signals = signals[(signals["date"] >= start_dt) & (signals["date"] <= end_dt)]

    # 3. Identify stable regime cycles via DuckDB SQL (no arbitrary splitting)
    train_period_list = _regime_windows(train_nsei, name)
    train_period_list.append({
        "name":  f"{name}",
        "start": start_dt,
        "end":   end_dt,
        "type":  "bear", #protect
    })
    _emit(f"\n{'='*72}")
    _emit(f"TRAINING: {name}  |  {start_dt.date()} → {end_dt.date()}")
    _emit(f"{'─'*72}")
    _emit(f"  {'Window':<45} {'Regime':<10} {'Days':>6}")
    for p in train_period_list:
        _emit(f"  {p['name']:<45} {p['type']:<10} {''}")
    _emit(f"{'='*72}\n")

    if train_prices.empty:
        _emit(f"  [SKIP] No price data for train window — skipping {name}")
        return {"name": name, "skipped": True}

    if not train_period_list:
        _emit(f"  [SKIP] No stable regime cycles found in training window — skipping {name}")
        return {"name": name, "skipped": True}
    engine      = ReturnsEngine(train_prices, nsei_features=train_nsei)
    evaluator   = FitnessEvaluator(engine, tickers, train_nsei,
                                   train_periods=train_period_list)
    initialiser = Initialiser(train_signals, tickers, train_nsei)
    ga          = GeneticOptimizer(tickers, engine, evaluator, initialiser, train_nsei)

    best = ga.run(
        pop_size=CFG.pop_size,
        n_gen=CFG.n_generations,
        cutoff_date=end_dt,
    )

    # ── Print trained weights ─────────────────────────────────────────────────
    _emit(f"\n{'='*72}")
    _emit(f"TRAINED WEIGHTS: {name}")
    _emit(f"{'='*72}")
    _emit(f"\n{'Ticker':<32} {'Weight':>10}")
    sorted_w = sorted(zip(tickers, best.weights), key=lambda x: x[1], reverse=True)
    for ticker, weight in sorted_w:
        if weight > 0.001: # 0.1%
            _emit(f"{ticker:<32}\t{weight*100:>8.2f}%")

    # ── Persist weights to CSV ────────────────────────────────────────────────
    safe_name = name.replace("/", "-").replace(" ", "_")
    weights_path = out_dir / f"{CFG.train_weights_prefix}{safe_name}.csv"
    wdf = pd.DataFrame({
        "ticker": tickers,
        "weight": [best.weights[i] for i in range(len(tickers))],
        "sector": [SECTOR_MAP.get(t, "Other") for t in tickers],
        "train_start": str(start_dt.date()),
        "train_end":   str(end_dt.date()),
        "fitness":     best.fitness,
    }).sort_values("weight", ascending=False)
    wdf.to_csv(weights_path, index=False)
    _emit(f"\n  [Saved] Weights → {weights_path}")

    return {
        "name":        name,
        "train_start": str(start_dt.date()),
        "train_end":   str(end_dt.date()),
        "fitness":     best.fitness,
        "weights_file": str(weights_path),
        "skipped":     False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GA Portfolio Optimizer — Training Phase")
    parser.add_argument("--config",  default=None,       help="Train only this named config (default: all enabled)")
    parser.add_argument("--out-dir", default=CFG.out_dir, help="Output directory (default from ga_config.csv)")
    args = parser.parse_args()

    # Seeds
    _random.seed(CFG.random_seed)
    np.random.seed(CFG.random_seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    loader = DataLoader(CFG.data_glob).load()
    tickers        = [t for t in loader.tickers if t in loader.prices.columns]
    prices         = loader.prices
    nsei_features  = loader.nsei_features
    signals        = loader.signals

    _emit(f"[Train] Tickers with price data: {len(tickers)}")
    if not tickers:
        _emit("[ERROR] No tickers with price data. Check data_glob in ga_config.csv.")
        sys.exit(1)

    # ── Filter configs ────────────────────────────────────────────────────────
    configs_to_run = PERIOD_CONFIG
    if args.config:
        configs_to_run = [c for c in PERIOD_CONFIG if c["name"] == args.config]
        if not configs_to_run:
            _emit(f"[ERROR] Config '{args.config}' not found or not enabled.")
            sys.exit(1)

    # ── Run training ──────────────────────────────────────────────────────────
    train_summaries = []
    for cfg in configs_to_run:
        load_sector_momentum_cache(CFG.benchmark_momentum_glob,
                                   start=cfg["train"]["start"], end=cfg["train"]["end"])
        summary = train_one_config(cfg, prices, tickers, nsei_features, signals, out_dir)
        train_summaries.append(summary)

    # ── Write training summary ────────────────────────────────────────────────
    valid = [s for s in train_summaries if not s.get("skipped")]
    summary_df = pd.DataFrame(valid)
    summary_path = out_dir / "train_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    _emit("\n" + "=" * 72)
    _emit("TRAINING COMPLETE")
    _emit("=" * 72)
    _emit(f"\n{'Config':<28} {'Train End':>12} {'Fitness':>10} {'Weights File'}")
    _emit("-" * 80)
    for s in valid:
        _emit(f"{s['name']:<28} {s['train_end']:>12} {s['fitness']:>10.4f}  {s['weights_file']}")

    _emit(f"\n[Train] Summary → {summary_path}")
    _emit(f"[Train] Weights files ready in {out_dir.resolve()}/")
    _emit("Done. ✓")


if __name__ == "__main__":
    main()