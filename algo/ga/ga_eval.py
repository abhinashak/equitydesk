"""
=============================================================================
ga_eval.py  —  Evaluation Phase
=============================================================================
Reads trained weights produced by ga_train.py and evaluates each config
on its STRICTLY ISOLATED test window.

For each enabled period:
  1. Reads outputs/weights_train_<config_name>.csv  (written by ga_train.py)
  2. Re-loads test-window prices.
  3. Computes portfolio vs index CAGR and alpha.
  4. Optionally runs the walk-forward quarterly rebalance.
  5. Writes:
       outputs/eval_results.csv      — per-config scorecard
       outputs/walk_forward.csv      — walk-forward quarterly results

Usage:
    python ga_eval.py
    python ga_eval.py --config FY2024              # eval a single config
    python ga_eval.py --skip-walkforward           # skip walk-forward step
    python ga_eval.py --out-dir my_outputs         # override output directory
=============================================================================
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ga_common import (
    CFG,
    PERIOD_CONFIG,
    TRAIN_PERIODS,
    SECTOR_MAP,
    DataLoader,
    FitnessEvaluator,
    GeneticOptimizer,
    Initialiser,
    ReturnsEngine,
    calculate_mdd,
    regime_series,
    load_sector_momentum_cache,
    run_blended_walk_forward,
    blend_weights,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: load weights file written by ga_train.py
# ─────────────────────────────────────────────────────────────────────────────
def load_trained_weights(name: str, tickers: list, out_dir: Path) -> np.ndarray | None:
    safe_name    = name.replace("/", "-").replace(" ", "_")
    weights_path = out_dir / f"{CFG.train_weights_prefix}{safe_name}.csv"
    if not weights_path.exists():
        print(f"  [MISSING] {weights_path} — run ga_train.py first.")
        return None
    wdf = pd.read_csv(weights_path).set_index("ticker")
    weights = np.array([wdf.loc[t, "weight"] if t in wdf.index else 0.0 for t in tickers])
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# EVAL ONE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
def eval_one_config(
        cfg:           dict,
        prices:        pd.DataFrame,
        tickers:       list,
        nsei_features: pd.DataFrame,
        out_dir:       Path,
) -> dict:
    name      = cfg["name"]
    test_cfg  = cfg["test"]
    train_cfg = cfg["train"]

    test_start   = pd.Timestamp(test_cfg["start"])
    test_end     = pd.Timestamp(test_cfg["end"])
    train_cutoff = pd.Timestamp(train_cfg["end"])

    print(f"\n{'='*72}")
    print(f"EVALUATING: {name}")
    print(f"  Train end : {train_cutoff.date()}")
    print(f"  Test      : {test_start.date()} → {test_end.date()}")
    print(f"{'='*72}\n")

    # ── Load trained weights ──────────────────────────────────────────────────
    weights = load_trained_weights(name, tickers, out_dir)
    if weights is None:
        return {"name": name, "skipped": True, "reason": "weights_missing"}

    # ── Test-window evaluation ────────────────────────────────────────────────
    test_prices = prices.loc[test_start:test_end]
    if test_prices.empty:
        print(f"  [NOTE] No price data for test window of {name} (future period)")
        port_pct = idx_pct = alpha = mdd = float("nan")
        dma50_log = []
    else:
        test_nsei   = nsei_features.loc[test_start:test_end]
        test_engine = ReturnsEngine(test_prices, nsei_features=test_nsei)
        test_engine.clear_dma50_log()                          # ← fresh log for this config
        port_ret = test_engine.portfolio_returns(
            weights, tickers, test_cfg["start"], test_cfg["end"],
            mode="eval", log_dma50=True,
        )
        # Index benchmark runs without exposure filter
        idx_engine = ReturnsEngine(test_prices, nsei_features=None)
        idx_ret = idx_engine.portfolio_returns(
            np.array([1.0]), [CFG.index_ticker], test_cfg["start"], test_cfg["end"]
        )
        # Simple % change: (Final / Initial) - 1  — no annualisation
        port_pct = test_engine.simple_return(port_ret) if len(port_ret) >= 5 else float("nan")
        idx_pct  = test_engine.simple_return(idx_ret)  if len(idx_ret)  >= 5 else float("nan")
        mdd      = calculate_mdd(port_ret) if len(port_ret) >= 5 else float("nan")
        alpha = (port_pct - idx_pct
                 if not (np.isnan(port_pct) or np.isnan(idx_pct)) else float("nan"))
        dma50_log = test_engine.dma50_log

    beaten  = (alpha > 0) if not np.isnan(alpha) else False
    pc_s    = f"{port_pct*100:+.1f}%" if not np.isnan(port_pct) else "N/A"
    ic_s    = f"{idx_pct*100:+.1f}%"  if not np.isnan(idx_pct)  else "N/A"
    alpha_s = f"{alpha*100:+.1f}%"    if not np.isnan(alpha)     else "N/A"
    mdd_s   = f"{mdd*100:.1f}%"       if not np.isnan(mdd)       else "N/A"
    mark    = "✓" if beaten else "✗"
    print(f"  Test result → Portfolio: {pc_s}  Index: {ic_s}  Alpha: {alpha_s}  MDD: {mdd_s}  {mark}")

    # ── Print EXIT / REENTRY log ──────────────────────────────────────────────
    if dma50_log:
        test_engine.print_dma50_log()
        # Save per-config exit log to CSV
        safe_name = name.replace("/", "-").replace(" ", "_")
        log_path  = out_dir / f"dma50_log_{safe_name}.csv"
        pd.DataFrame(dma50_log).to_csv(log_path, index=False)
        print(f"  [Saved] DMA50 log → {log_path}")
    else:
        print("  [DMA50] No exits triggered in this test window.")

    return {
        "name":       name,
        "type":       test_cfg["type"],
        "train_end":  str(train_cutoff.date()),
        "test_start": str(test_start.date()),
        "test_end":   str(test_end.date()),
        "port_ret":   port_pct,     # simple % change (replaces port_cagr + port_actual)
        "idx_ret":    idx_pct,      # simple % change (replaces idx_cagr  + idx_actual)
        "alpha":      alpha,        # arithmetic difference in total % return
        "mdd":        mdd,          # max drawdown (%) for the test window
        "beaten":     beaten,
        "skipped":    False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD  (monthly blended rebalance via run_blended_walk_forward)
# ─────────────────────────────────────────────────────────────────────────────
def run_walk_forward_all_configs(
        configs:       list,
        prices:        pd.DataFrame,
        tickers:       list,
        nsei_features: pd.DataFrame,
        signals:       pd.DataFrame,
        out_dir:       Path,
        n_gen:         int = 20,
) -> pd.DataFrame:
    """
    Run run_blended_walk_forward() for every enabled config and stitch results.

    For each config the test window defines the FY eval period. After all
    months are simulated the full stitched return series is used to compute
    portfolio metrics (CAGR, Sharpe, Sortino, Calmar, MDD) and print an index
    comparison scorecard.

    Returns a combined DataFrame with one row per config containing all metrics.
    """
    all_rows = []

    for cfg in configs:
        name     = cfg["name"]
        test_cfg = cfg["test"]

        # Load sector momentum cache for this config's test window
        load_sector_momentum_cache(start=test_cfg["start"], end=test_cfg["end"])
        wf_df = run_blended_walk_forward(
            fy_cfg        = cfg,
            prices        = prices,
            all_tickers   = tickers,
            nsei_features = nsei_features,
            signals       = signals,
            out_dir       = out_dir,
            n_gen         = n_gen
        )

        if wf_df.empty:
            print(f"  [Walk-Forward] No results for {name} — skipping metrics.")
            continue

        # ── Reconstruct full stitched return series from monthly port_ret_pct ──
        # run_blended_walk_forward already called portfolio_returns() per month
        # and stored the simple monthly return in 'port_ret_pct'.  Re-build the
        # daily series by replaying engine_full for each month so we have the
        # granular series for Sharpe/Sortino/MDD.
        fy_start = pd.Timestamp(test_cfg["start"])
        fy_end   = pd.Timestamp(test_cfg["end"])

        test_prices = prices.loc[fy_start:fy_end]
        test_nsei   = nsei_features.loc[fy_start:fy_end]

        if test_prices.empty:
            print(f"  [Walk-Forward] No price data for {name} test window.")
            continue

        # Replay each month to collect the daily return series
        engine_eval = ReturnsEngine(test_prices, nsei_features=test_nsei)
        engine_eval.clear_dma50_log()

        trading_days = test_prices.index
        month_starts: list = []
        seen_months: set   = set()
        for dt in trading_days:
            key = (dt.year, dt.month)
            if key not in seen_months:
                seen_months.add(key)
                month_starts.append(dt)

        monthly_ret_series = []
        ticker_cols = [c for c in wf_df.columns if c.startswith("w_")]
        month_tickers = [c[2:] for c in ticker_cols]  # strip "w_" prefix

        for m_idx, row in enumerate(wf_df.itertuples()):
            month_start = pd.Timestamp(row.month_start)
            month_end   = pd.Timestamp(row.month_end)

            w_arr = np.array([
                getattr(row, f"w_{t}", 0.0) for t in month_tickers
            ], dtype=np.float64)

            # prev_weights: previous month's blended weights (zero on first month)
            if m_idx == 0:
                prev_w = np.zeros(len(month_tickers))
            else:
                prev_row = wf_df.iloc[m_idx - 1]
                prev_w   = np.array([
                    prev_row.get(f"w_{t}", 0.0) for t in month_tickers
                ], dtype=np.float64)

            valid_tickers = [t for t in month_tickers if t in test_prices.columns]
            valid_idx     = [i for i, t in enumerate(month_tickers) if t in test_prices.columns]
            w_valid       = w_arr[valid_idx]
            p_valid       = prev_w[valid_idx]

            if w_valid.sum() < 1e-9:
                continue

            month_ret = engine_eval.portfolio_returns(
                weights      = w_valid,
                tickers      = valid_tickers,
                start        = str(month_start.date()),
                end          = str(month_end.date()),
                prev_weights = p_valid,
                log_dma50    = True,
                mode         = "eval",
            )
            monthly_ret_series.append(month_ret)

        if not monthly_ret_series:
            print(f"  [Walk-Forward] Could not reconstruct daily series for {name}.")
            continue

        fy_ret = pd.concat(monthly_ret_series).sort_index()
        fy_ret = fy_ret[~fy_ret.index.duplicated(keep="first")]

        # ── Index benchmark (raw, no exposure filter) ─────────────────────────
        idx_engine = ReturnsEngine(test_prices, nsei_features=None)
        idx_ret    = idx_engine.portfolio_returns(
            weights   = np.array([1.0]),
            tickers   = [CFG.index_ticker],
            start     = test_cfg["start"],
            end       = test_cfg["end"],
            log_dma50 = False,
            mode      = "train",
        )

        # ── Metrics ───────────────────────────────────────────────────────────
        port_simple = engine_eval.simple_return(fy_ret)
        idx_simple  = idx_engine.simple_return(idx_ret)
        alpha       = port_simple - idx_simple
        port_cagr   = engine_eval.cagr(fy_ret)
        idx_cagr    = idx_engine.cagr(idx_ret)
        sharpe      = engine_eval.sharpe(fy_ret)
        sortino     = engine_eval.sortino(fy_ret)
        calmar      = engine_eval.calmar(fy_ret)
        mdd         = engine_eval.max_drawdown(fy_ret)
        total_tx    = wf_df["tx_cost_bps"].sum() if "tx_cost_bps" in wf_df.columns else float("nan")
        total_to    = wf_df["turnover_pct"].sum() if "turnover_pct" in wf_df.columns else float("nan")

        beaten = alpha > 0

        # ── Print full scorecard ──────────────────────────────────────────────
        print(f"\n{'='*72}")
        print(f"WALK-FORWARD SCORECARD (blended monthly): {name}")
        print(f"  Test window : {test_cfg['start']} → {test_cfg['end']}")
        print(f"{'='*72}")
        print(f"\n  {'Metric':<28} {'Portfolio':>12} {'Index':>12}")
        print(f"  {'-'*54}")
        print(f"  {'Total Return':<28} {port_simple*100:>+11.2f}% {idx_simple*100:>+11.2f}%")
        print(f"  {'CAGR':<28} {port_cagr*100:>+11.2f}% {idx_cagr*100:>+11.2f}%")
        print(f"  {'Alpha (total ret)':<28} {alpha*100:>+11.2f}%")
        print(f"  {'Sharpe Ratio':<28} {sharpe:>12.3f}")
        print(f"  {'Sortino Ratio':<28} {sortino:>12.3f}")
        print(f"  {'Calmar Ratio':<28} {calmar:>12.3f}")
        print(f"  {'Max Drawdown':<28} {mdd*100:>12.2f}%")
        print(f"  {'Total Turnover':<28} {total_to:>11.1f}%")
        print(f"  {'Total Tx Cost (bps)':<28} {total_tx:>12.1f}")
        print(f"  {'Beat Index':<28} {'YES ✓' if beaten else 'NO  ✗':>12}")
        print(f"{'='*72}\n")

        # ── DMA50 exit/reentry log ────────────────────────────────────────────
        if engine_eval.dma50_log:
            engine_eval.print_dma50_log()
            safe_name = name.replace("/", "-").replace(" ", "_")
            log_path  = out_dir / f"dma50_log_wf_{safe_name}.csv"
            pd.DataFrame(engine_eval.dma50_log).to_csv(log_path, index=False)
            print(f"  [Saved] DMA50 log → {log_path}")
        else:
            print("  [DMA50] No exits triggered in this walk-forward window.")

        # ── Save daily return series ──────────────────────────────────────────
        safe_name  = name.replace("/", "-").replace(" ", "_")
        ret_path   = out_dir / f"wf_daily_returns_{safe_name}.csv"
        fy_ret.to_csv(ret_path, header=["daily_ret"])
        print(f"  [Saved] Daily return series → {ret_path}")

        all_rows.append({
            "name":         name,
            "test_start":   test_cfg["start"],
            "test_end":     test_cfg["end"],
            "port_ret_pct": round(port_simple * 100, 3),
            "idx_ret_pct":  round(idx_simple  * 100, 3),
            "alpha_pct":    round(alpha        * 100, 3),
            "port_cagr_pct":round(port_cagr   * 100, 3),
            "idx_cagr_pct": round(idx_cagr    * 100, 3),
            "sharpe":       round(sharpe,  3),
            "sortino":      round(sortino, 3),
            "calmar":       round(calmar,  3),
            "mdd_pct":      round(mdd      * 100, 3),
            "total_tx_bps": round(total_tx, 1),
            "total_to_pct": round(total_to, 1),
            "beaten":       beaten,
        })

    if not all_rows:
        return pd.DataFrame()

    summary_df = pd.DataFrame(all_rows)

    # ── Aggregate summary across all configs ──────────────────────────────────
    valid      = summary_df.dropna(subset=["alpha_pct"])
    beats      = valid["beaten"].sum()
    avg_alpha  = valid["alpha_pct"].mean()
    avg_mdd    = valid["mdd_pct"].mean()
    avg_sharpe = valid["sharpe"].mean()

    print("\n" + "=" * 80)
    print("AGGREGATE WALK-FORWARD SCORECARD  (blended monthly, all configs)")
    print("=" * 80)
    print(f"\n{'Config':<28} {'TotalRet':>9} {'Idx':>9} {'Alpha':>8} "
          f"{'CAGR':>8} {'Sharpe':>7} {'MDD':>8} {'✓/✗':>4}")
    print("-" * 80)
    for _, r in summary_df.iterrows():
        mark = "✓" if r["beaten"] else "✗"
        print(f"{r['name']:<28} {r['port_ret_pct']:>+8.2f}% {r['idx_ret_pct']:>+8.2f}% "
              f"{r['alpha_pct']:>+7.2f}% {r['port_cagr_pct']:>+7.2f}% "
              f"{r['sharpe']:>7.3f} {r['mdd_pct']:>7.2f}% {mark:>4}")
    print("-" * 80)
    print(f"  Configs beaten : {beats}/{len(valid)}")
    print(f"  Avg alpha      : {avg_alpha:+.2f}%")
    print(f"  Avg MDD        : {avg_mdd:.2f}%")
    print(f"  Avg Sharpe     : {avg_sharpe:.3f}")
    print("=" * 80 + "\n")

    return summary_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GA Portfolio Optimizer — Evaluation Phase")
    parser.add_argument("--config",           default=None,        help="Evaluate only this named config")
    parser.add_argument("--out-dir",          default=CFG.out_dir, help="Output directory")
    parser.add_argument("--skip-walkforward", action="store_true", help="Skip walk-forward rebalance")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    loader = DataLoader(CFG.data_glob).load()
    tickers       = [t for t in loader.tickers if t in loader.prices.columns]
    prices        = loader.prices
    nsei_features = loader.nsei_features
    signals       = loader.signals

    print(f"[Eval] Tickers with price data: {len(tickers)}")
    if not tickers:
        print("[ERROR] No tickers found. Check data_glob in ga_config.csv.")
        sys.exit(1)

    # ── Filter configs ────────────────────────────────────────────────────────
    configs_to_eval = PERIOD_CONFIG
    if args.config:
        configs_to_eval = [c for c in PERIOD_CONFIG if c["name"] == args.config]
        if not configs_to_eval:
            print(f"[ERROR] Config '{args.config}' not found or not enabled.")
            sys.exit(1)

    # ── Evaluate each config ──────────────────────────────────────────────────
    all_results = []
    for cfg in configs_to_eval:
        load_sector_momentum_cache(start=cfg["test"]["start"], end=cfg["test"]["end"])
        result = eval_one_config(cfg, prices, tickers, nsei_features, out_dir)
        all_results.append(result)

    # ── Aggregate scorecard ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("AGGREGATE EVAL SCORECARD  (each row = independent GA run)")
    print("=" * 72)
    print(f"\n{'Config':<28} {'Train end':>12} {'Test start':>12} "
          f"{'Portfolio':>10} {'Index':>8} {'Alpha':>8} {'MDD':>7} {'✓/✗':>4}")
    print("-" * 96)

    valid      = [r for r in all_results if not r.get("skipped")]
    beats      = sum(r["beaten"] for r in valid)
    alphas_all = [r["alpha"] for r in valid if not np.isnan(r["alpha"])]

    for r in valid:
        pc_s    = f"{r['port_ret']*100:+.1f}%" if not np.isnan(r['port_ret']) else "N/A"
        ic_s    = f"{r['idx_ret']*100:+.1f}%"  if not np.isnan(r['idx_ret'])  else "N/A"
        alpha_s = f"{r['alpha']*100:+.1f}%"    if not np.isnan(r['alpha'])    else "N/A"
        mdd_s   = f"{r['mdd']*100:.1f}%"       if not np.isnan(r['mdd'])      else "N/A"
        mark    = "✓" if r["beaten"] else "✗"
        print(f"{r['name']:<28} {r['train_end']:>12} {r['test_start']:>12} "
              f"{pc_s:>10} {ic_s:>8} {alpha_s:>8} {mdd_s:>7} {mark:>4}")

    print("-" * 96)
    avg_alpha = np.mean(alphas_all) if alphas_all else float("nan")
    mdds_all  = [r["mdd"] for r in valid if not np.isnan(r["mdd"])]
    avg_mdd   = np.mean(mdds_all) if mdds_all else float("nan")
    print(f"\nConfigs beaten   : {beats}/{len(valid)}")
    print(f"Avg alpha        : {avg_alpha*100:+.2f}%" if not np.isnan(avg_alpha) else "\nAvg alpha        : N/A")
    print(f"Avg MDD          : {avg_mdd*100:.2f}%"    if not np.isnan(avg_mdd)   else "Avg MDD          : N/A")

    # ── Save eval results CSV ─────────────────────────────────────────────────
    results_df   = pd.DataFrame(valid)
    results_path = out_dir / CFG.eval_results_file
    results_df.to_csv(results_path, index=False)
    print(f"\n[Eval] Scorecard → {results_path}")

    # ── Walk-forward ──────────────────────────────────────────────────────────
    if not args.skip_walkforward:
        print("\n" + "=" * 72)
        print("WALK-FORWARD QUARTERLY REBALANCE (20 gens)")
        print("=" * 72)
        wf_df = run_walk_forward_all_configs(
            configs_to_eval, prices, tickers, nsei_features, signals, out_dir
        )
        if not wf_df.empty:
            print(wf_df[["name", "port_ret_pct", "alpha_pct", "sharpe", "mdd_pct"]].to_string(index=False))
            wf_path = out_dir / CFG.walk_forward_file
            wf_df.to_csv(wf_path, index=False)
            print(f"\n[Eval] Walk-forward → {wf_path}")
    else:
        print("\n[Eval] Walk-forward skipped (--skip-walkforward).")

    print(f"\n[Eval] All outputs in {out_dir.resolve()}/")
    print("Done. ✓")


if __name__ == "__main__":
    main()