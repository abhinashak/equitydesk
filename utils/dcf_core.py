"""
utils/dcf_core.py
─────────────────
Streamlit-free building blocks for the DCF page and the DCF assumption optimizer.

* Every database read goes straight through `utils.db_utils.run_readonly_query`
  (the shared `sqls/all.sql` database) with bound `?` parameters; no connection objects
  are passed around.
* `dcf_multiple` / `run_dcf` hold the two-stage DCF maths in one place.
* `normalize_earnings` is the earnings-normalisation waterfall (moved here unchanged).

The page wraps the `get_*` functions in `st.cache_data`; the optimizer calls them directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.db_utils import run_readonly_query


# ══════════════════════════════════════════════════════════════════════════════
# DATA GETTERS
# ══════════════════════════════════════════════════════════════════════════════

def get_net_profit(tkr: str) -> pd.DataFrame:
    try:
        df = run_readonly_query("""
            SELECT dt, val FROM profit_loss
            WHERE ticker = ? AND metric = 'Net Profit'
            ORDER BY dt
        """, [tkr])
        df["dt"] = pd.to_datetime(df["dt"])
        return df
    except Exception:
        return pd.DataFrame()


def get_eps(tkr: str) -> pd.DataFrame:
    try:
        df = run_readonly_query("""
            SELECT dt, val FROM profit_loss
            WHERE ticker = ? AND metric = 'EPS in Rs'
            ORDER BY dt
        """, [tkr])
        df["dt"] = pd.to_datetime(df["dt"])
        return df
    except Exception:
        return pd.DataFrame()


def get_price_history(tkr: str) -> pd.DataFrame:
    try:
        df = run_readonly_query("""
            SELECT Date AS dt, Close AS price
            FROM ticker_prices
            WHERE REPLACE(Ticker, '.NS', '') = ?
            ORDER BY Date
        """, [tkr])
        df["dt"] = pd.to_datetime(df["dt"])
        return df
    except Exception:
        return pd.DataFrame()


def get_growth_rates(tkr: str) -> dict:
    rates = {}
    try:
        df = run_readonly_query("""
            SELECT horizon, val FROM compounded_profit_growth
            WHERE ticker = ?
        """, [tkr])
        for _, r in df.iterrows():
            rates[r["horizon"]] = float(r["val"])
    except Exception:
        pass
    return rates


def get_general_info(tkr: str) -> dict:
    try:
        row = run_readonly_query("""
            SELECT current_price, stock_p_e, book_value, dividend_yield, roe, roce
            FROM general_info WHERE ticker = ? LIMIT 1
        """, [tkr])
        if not row.empty:
            return row.iloc[0].to_dict()
    except Exception:
        pass
    return {}


def get_historical_pe(tkr: str) -> dict:
    """
    Compute time-decay weighted P/E by stitching two data sources:

    1. ANNUAL segment  — profit_loss EPS × annual avg price (up to ~10 years)
       Each annual point uses the full-year average price vs full-year average EPS.

    2. QUARTERLY segment — quarterly_results TTM EPS × quarter-end close price
       TTM EPS = rolling sum of 4 consecutive quarterly EPS figures.
       Quarter-end price = last Close in that calendar month from ticker_prices.
       Provides ~10 recent quarterly observations with finer granularity.

    Stitching rule
    ──────────────
    Find the earliest quarterly observation date.  Any annual point whose year-end
    falls ON or AFTER that date is dropped — the quarterly data covers that period
    more accurately.  Annual points before that cutoff are kept as-is.

    Time-decay weighting
    ────────────────────
    Because the combined timeline is non-uniform (annual gaps early, quarterly
    gaps recent), weights are based on calendar distance from today rather than
    rank:

        weight[i] = exp(−λ × days_ago[i])      λ = 0.001  (~1-year half-life)

    This gives natural, continuous recency bias regardless of whether a point
    comes from the annual or quarterly segment.

    Returns dict with keys:
        weighted_median_pe  — primary metric used in scan & detail
        weighted_mean_pe
        median_pe, mean_pe  — simple unweighted versions for reference
        min_pe, max_pe      — full historical range
        n_annual            — annual points used
        n_quarterly         — quarterly TTM points used
        pe_series           — DataFrame: dt, pe, source, weight_pct
    """
    try:
        today = pd.Timestamp.today().normalize()
        LAMBDA = 0.001   # decay per day  (half-life ≈ 693 days ≈ 1.9 years)

        # ── 1. ANNUAL segment ─────────────────────────────────────────────────
        price_df = run_readonly_query("""
            SELECT Date AS dt, Close AS price
            FROM ticker_prices
            WHERE REPLACE(Ticker, '.NS', '') = ?
            ORDER BY Date
        """, [tkr])
        eps_ann_df = run_readonly_query("""
            SELECT dt, val FROM profit_loss
            WHERE ticker = ? AND metric = 'EPS in Rs'
            ORDER BY dt
        """, [tkr])

        annual_rows = pd.DataFrame()
        if not price_df.empty and not eps_ann_df.empty:
            price_df["dt"]    = pd.to_datetime(price_df["dt"])
            eps_ann_df["dt"]  = pd.to_datetime(eps_ann_df["dt"])

            price_ann = (price_df.set_index("dt")["price"]
                         .resample("YE").mean()
                         .reset_index()
                         .rename(columns={"dt": "dt", "price": "avg_price"}))
            eps_ann   = (eps_ann_df.set_index("dt")["val"]
                         .resample("YE").sum()          # annual EPS = sum of quarters
                         .reset_index()
                         .rename(columns={"dt": "dt", "val": "eps"}))
            eps_ann   = eps_ann[eps_ann["eps"] > 0]

            ann_merged = pd.merge(price_ann, eps_ann, on="dt", how="inner")
            ann_merged["pe"]     = ann_merged["avg_price"] / ann_merged["eps"]
            ann_merged["source"] = "annual"
            annual_rows = ann_merged[["dt", "pe", "source"]].copy()

        # ── 2. QUARTERLY segment — TTM EPS via rolling 4-quarter sum ─────────
        qtr_df = run_readonly_query("""
            SELECT dt, val AS eps_q
            FROM quarterly_results
            WHERE ticker = ? AND metric = 'EPS in Rs'
            ORDER BY dt
        """, [tkr])

        quarterly_rows = pd.DataFrame()
        if not qtr_df.empty and not price_df.empty:
            qtr_df["dt"] = pd.to_datetime(qtr_df["dt"])
            qtr_df = qtr_df.sort_values("dt").reset_index(drop=True)

            # TTM EPS: rolling 4-quarter sum, only keep rows where we have 4 qtrs
            qtr_df["ttm_eps"] = qtr_df["eps_q"].rolling(4, min_periods=4).sum()
            qtr_df = qtr_df.dropna(subset=["ttm_eps"])
            qtr_df = qtr_df[qtr_df["ttm_eps"] > 0]

            if not qtr_df.empty:
                # Quarter-end price: last Close in the same calendar month as dt
                price_df["ym"] = price_df["dt"].dt.to_period("M")
                qtr_df["ym"]   = qtr_df["dt"].dt.to_period("M")
                qtr_price = (price_df.sort_values("dt")
                             .groupby("ym")["price"]
                             .last()
                             .reset_index()
                             .rename(columns={"price": "close_price"}))
                qtr_merged = pd.merge(qtr_df, qtr_price, on="ym", how="inner")
                qtr_merged["pe"]     = qtr_merged["close_price"] / qtr_merged["ttm_eps"]
                qtr_merged["source"] = "quarterly"
                quarterly_rows = qtr_merged[["dt", "pe", "source"]].copy()

        if annual_rows.empty and quarterly_rows.empty:
            return {}

        # ── 3. Stitch — drop annual points covered by quarterly window ────────
        if not quarterly_rows.empty and not annual_rows.empty:
            qtr_cutoff = quarterly_rows["dt"].min()
            annual_rows = annual_rows[annual_rows["dt"] < qtr_cutoff]

        combined = pd.concat(
            [df for df in [annual_rows, quarterly_rows] if not df.empty],
            ignore_index=True,
        ).sort_values("dt").reset_index(drop=True)

        # Sanity filter
        combined = combined[(combined["pe"] > 3) & (combined["pe"] < 300)]
        if combined.empty:
            return {}

        # ── 4. Time-decay weights (calendar distance from today) ──────────────
        combined["days_ago"] = (today - combined["dt"]).dt.days.clip(lower=0)
        raw_w   = np.exp(-LAMBDA * combined["days_ago"].values)
        weights = raw_w / raw_w.sum()
        combined["weight"]     = weights
        combined["weight_pct"] = (weights * 100).round(2)

        # ── 5. Weighted median ────────────────────────────────────────────────
        pe_sorted = combined.sort_values("pe").reset_index(drop=True)
        pe_sorted["cum_w"] = pe_sorted["weight"].cumsum()
        weighted_median = float(pe_sorted[pe_sorted["cum_w"] >= 0.5]["pe"].iloc[0])
        weighted_mean   = float((combined["pe"] * combined["weight"]).sum())

        n_ann = int((combined["source"] == "annual").sum())
        n_qtr = int((combined["source"] == "quarterly").sum())

        return {
            "weighted_median_pe": weighted_median,
            "weighted_mean_pe":   weighted_mean,
            "median_pe":          float(combined["pe"].median()),
            "mean_pe":            float(combined["pe"].mean()),
            "min_pe":             float(combined["pe"].min()),
            "max_pe":             float(combined["pe"].max()),
            "n_annual":           n_ann,
            "n_quarterly":        n_qtr,
            "lambda":             LAMBDA,
            "pe_series":          combined[["dt", "pe", "source", "weight_pct"]].copy(),
        }
    except Exception:
        return {}


def get_shares_outstanding(tkr: str) -> float | None:
    try:
        eq = run_readonly_query("""
            SELECT val FROM balance_sheet
            WHERE ticker = ? AND metric = 'Equity Capital'
            ORDER BY dt DESC LIMIT 1
        """, [tkr])
        fv = run_readonly_query("""
            SELECT face_value FROM general_info WHERE ticker = ? LIMIT 1
        """, [tkr])
        if not eq.empty and not fv.empty:
            equity_cr = float(eq.iloc[0]["val"])
            face      = float(fv.iloc[0]["face_value"]) or 10.0
            shares    = equity_cr * 1e7 / face
            return shares / 1e7          # in crores
    except Exception:
        pass
    return None




def get_other_income(tkr: str) -> pd.DataFrame:
    """Annual Other Income series."""
    try:
        df = run_readonly_query("""
            SELECT dt, val FROM profit_loss
            WHERE ticker = ? AND metric = 'Other Income'
            ORDER BY dt
        """, [tkr])
        df["dt"] = pd.to_datetime(df["dt"])
        return df
    except Exception:
        return pd.DataFrame()


def get_exceptional_items(tkr: str) -> pd.DataFrame:
    """Annual Exceptional Items series (signed — negative = charge)."""
    try:
        df = run_readonly_query("""
            SELECT dt, val FROM profit_loss
            WHERE ticker = ?
              AND metric IN ('Exceptional Items', 'Exceptional Item', 'Extraordinary Items')
            ORDER BY dt
        """, [tkr])
        df["dt"] = pd.to_datetime(df["dt"])
        return df
    except Exception:
        return pd.DataFrame()


def get_free_cash_flow(tkr: str) -> pd.DataFrame:
    """Annual 'Free Cash Flow' series (same metric dcf.sql uses for fcf_conversion)."""
    try:
        df = run_readonly_query("""
            SELECT dt, val FROM cash_flows
            WHERE ticker = ? AND metric = 'Free Cash Flow'
            ORDER BY dt
        """, [tkr])
        df["dt"] = pd.to_datetime(df["dt"])
        return df
    except Exception:
        return pd.DataFrame()


def get_sector(tkr: str) -> str | None:
    try:
        df = run_readonly_query("SELECT sector FROM general_info WHERE ticker = ? LIMIT 1", [tkr])
        if not df.empty and pd.notna(df.iloc[0]["sector"]):
            return str(df.iloc[0]["sector"])
    except Exception:
        pass
    return None


def get_tax_rate(tkr: str) -> float:
    """Estimate effective tax rate from last 3 years of data (PBT / Tax)."""
    try:
        pbt = run_readonly_query("""
            SELECT dt, val FROM profit_loss
            WHERE ticker = ? AND metric = 'Profit before tax'
            ORDER BY dt DESC LIMIT 3
        """, [tkr])
        tax = run_readonly_query("""
            SELECT dt, val FROM profit_loss
            WHERE ticker = ? AND metric = 'Tax %'
            ORDER BY dt DESC LIMIT 3
        """, [tkr])
        if not tax.empty:
            # Tax % is stored directly
            return float(tax["val"].mean()) / 100.0
        if not pbt.empty:
            # Fallback: assume 25%
            pass
    except Exception:
        pass
    return 0.25          # default effective tax rate




def get_receivables_and_revenue(tkr: str) -> pd.DataFrame:
    """
    Returns annual DataFrame with columns: year, revenue, receivables, debtor_days.

    Schema fix: balance_sheet does NOT contain Trade Receivables / Debtors.
    The ratios view has metric = 'Debtor Days' directly — use that as the
    primary source. Revenue comes from profit_loss metric = 'Sales'.
    Receivables is back-computed as debtor_days / 365 × revenue for display.
    """
    try:
        # Primary: Debtor Days from ratios view (exists in schema)
        dd_df = run_readonly_query("""
            SELECT dt, val AS debtor_days FROM ratios
            WHERE ticker = ? AND metric = 'Debtor Days'
            ORDER BY dt
        """, [tkr])

        # Revenue from profit_loss (metric = 'Sales' confirmed in schema)
        rev_df = run_readonly_query("""
            SELECT dt, val AS revenue FROM profit_loss
            WHERE ticker = ? AND metric = 'Sales'
            ORDER BY dt
        """, [tkr])

        if dd_df.empty or rev_df.empty:
            return pd.DataFrame()

        dd_df["dt"]  = pd.to_datetime(dd_df["dt"])
        rev_df["dt"] = pd.to_datetime(rev_df["dt"])

        dd_ann  = (dd_df.set_index("dt")["debtor_days"]
                   .resample("YE").mean()
                   .reset_index().rename(columns={"dt": "year"}))
        rev_ann = (rev_df.set_index("dt")["revenue"]
                   .resample("YE").sum()
                   .reset_index().rename(columns={"dt": "year"}))

        merged = pd.merge(rev_ann, dd_ann, on="year", how="inner")
        merged = merged[merged["revenue"] > 0]
        # Back-compute receivables for display only
        merged["receivables"] = (merged["debtor_days"] / 365 * merged["revenue"]).round(1)
        merged["debtor_days"]  = merged["debtor_days"].round(1)
        return merged[["year", "revenue", "receivables", "debtor_days"]].tail(7)
    except Exception:
        return pd.DataFrame()


def get_net_margin(tkr: str) -> float | None:
    """
    Average net profit margin over last 3 years.
    Schema note: profit_loss has no 'Net Profit %' metric.
    Computed as Net Profit / Sales using the two metrics that DO exist.
    """
    try:
        np_df = run_readonly_query("""
            SELECT dt, val FROM profit_loss
            WHERE ticker = ? AND metric = 'Net Profit'
            ORDER BY dt DESC LIMIT 3
        """, [tkr])
        # profit_loss schema: metric = 'Sales' is the revenue line
        sal_df = run_readonly_query("""
            SELECT dt, val FROM profit_loss
            WHERE ticker = ? AND metric = 'Sales'
            ORDER BY dt DESC LIMIT 3
        """, [tkr])
        if not np_df.empty and not sal_df.empty:
            # Guard against duplicate dt rows (e.g. standalone + consolidated both
            # tagged with the same date). A non-unique index passed into
            # pd.concat(axis=1) does a cartesian-style alignment and silently
            # pairs net-profit rows against the wrong sales rows, which can
            # produce wildly wrong (even >100%) "margins".
            np_df  = np_df.groupby("dt")["val"].mean()
            sal_df = sal_df.groupby("dt")["val"].mean()
            merged = pd.concat([np_df.rename("np"), sal_df.rename("sal")], axis=1).dropna()
            if not merged.empty and (merged["sal"] > 0).all():
                margin = float((merged["np"] / merged["sal"]).mean())
                # Sanity-clamp: net margin should realistically fall within
                # roughly -100% to 100%. Anything outside that indicates a
                # data issue upstream rather than a genuine margin.
                if -1.0 <= margin <= 1.0:
                    return margin
                return None
    except Exception:
        pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# DCF MATHS
# ══════════════════════════════════════════════════════════════════════════════

def dcf_multiple(phase1: float, phase2: float, terminal: float, discount: float) -> float:
    """
    PV of a 10-year two-stage projection plus terminal value, per ₹1 of base profit.

    Growth is `phase1` % for years 1-5 and `phase2` % for years 6-10; the terminal value
    (Gordon growth at `terminal` %) is only added when discount > terminal.
    The DCF is linear in base profit, so  run_dcf(B, ...) == B * dcf_multiple(...).
    """
    d      = discount / 100
    factor = 1.0
    pv     = 0.0
    for yr in range(1, 11):
        g       = phase1 if yr <= 5 else phase2
        factor *= 1 + g / 100
        pv     += factor / (1 + d) ** yr
    if discount > terminal:
        tv  = factor * (1 + terminal / 100) / (d - terminal / 100)
        pv += tv / (1 + d) ** 10
    return pv


def run_dcf(base_profit: float, phase1: float, phase2: float,
            terminal: float, discount: float) -> float:
    """Two-stage DCF on Net Profit (₹ Cr). Returns total PV in ₹ Cr."""
    return base_profit * dcf_multiple(phase1, phase2, terminal, discount)


# ══════════════════════════════════════════════════════════════════════════════
# EARNINGS NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════

def normalize_earnings(
        np_annual: pd.DataFrame,       # columns: year (Timestamp), net_profit (₹ Cr)
        oi_df: pd.DataFrame,           # raw other_income long-form (dt, val)
        exc_df: pd.DataFrame,          # raw exceptional_items long-form (dt, val)
        tax_rate: float,               # effective tax rate (0–1)
        zscore_threshold: float = 2.0, # flag other-income year as abnormal if |z| > this
        recv_df: pd.DataFrame = None,  # receivables+revenue DataFrame from _get_receivables_and_revenue
        net_margin: float = None,      # net profit margin (0–1) for receivables risk conversion
        recv_haircut_pct: float = 0.0, # % of excess receivables to treat as at-risk (user input)
        order_book_cr: float = 0.0,    # order book value in ₹ Cr (user input)
        ob_executable_pct: float = 50.0,  # % of order book executable in DCF horizon
        ob_margin_pct: float = None,   # margin on order book revenue (defaults to net_margin)
        oi_is_structural: bool = False,   # True → Other Income is core (e.g. Coal India FD interest)
        #         skip z-score removal; only strip exceptional items
) -> tuple[float, dict]:
    """
    Apply the normalisation waterfall to the most-recent annual net profit:

        Reported Net Profit
        − Abnormal Other Income SPIKE  (z-score > threshold, spike only, not full OI)
          [skipped entirely when oi_is_structural=True]
        − Exceptional Items
        + Normalized Other Income (5-yr average)
          [= reported OI when oi_is_structural=True, so no net change from OI steps]
        ± Tax adjustment on the above
        + One-time expenditures (user-supplied via slider, default 0)
        − Receivables Risk Haircut  (excess debtor days × at-risk %)
        + Order Book Profit Uplift  (executable OB × margin)
        ─────────────────────────────────────────────────
        = Normalized Earnings

    Bug fix vs prior version
    ────────────────────────
    Previously when z-score fired: abnormal_oi = reported_oi  (removed EVERYTHING)
    Correct behaviour:             abnormal_oi = reported_oi - norm_oi (remove only the SPIKE)
    The subsequent "+ norm_oi" step then correctly restores the recurring portion.

    Coal India / structural OI flag
    ────────────────────────────────
    Companies like Coal India hold large cash reserves and earn FD interest every year —
    that Other Income is as recurring as operating profit and should NOT be z-score adjusted.
    Set oi_is_structural=True to bypass z-score removal and pass OI through unchanged.

    Returns
    -------
    norm_profit : float   — normalized net profit (₹ Cr)
    detail      : dict    — breakdown dict for display
    """
    if np_annual.empty:
        return 0.0, {}

    reported = float(np_annual["net_profit"].iloc[-1])
    latest_year = np_annual["year"].iloc[-1]

    # ── Other Income: annual resample ─────────────────────────────────────────
    if not oi_df.empty:
        oi_ann = (
            oi_df.set_index("dt")["val"]
            .resample("YE").sum()
            .rename("other_income")
            .reset_index()
            .rename(columns={"dt": "year"})
        )
        oi_ann = oi_ann[oi_ann["other_income"].notna()]
    else:
        oi_ann = pd.DataFrame(columns=["year", "other_income"])

    # 5-year normalised Other Income
    oi_5yr  = oi_ann.tail(5)["other_income"] if len(oi_ann) >= 2 else pd.Series(dtype=float)
    norm_oi = float(oi_5yr.mean()) if not oi_5yr.empty else 0.0

    # Latest year's reported Other Income
    reported_oi = 0.0
    if not oi_ann.empty:
        reported_oi = float(
            oi_ann[oi_ann["year"].dt.year == latest_year.year]["other_income"]
            .sum() if not oi_ann.empty else 0
        )

    # ── Z-score spike detection ───────────────────────────────────────────────
    # BUG FIX: remove only the SPIKE (reported_oi - norm_oi), not the full reported_oi.
    # This means the subsequent "+ norm_oi" step correctly restores the recurring portion.
    # When oi_is_structural=True (Coal India, cash-rich PSUs) skip z-score entirely —
    # their FD/treasury income is as recurring as operating profit.
    abnormal_oi = 0.0
    z_score     = 0.0

    if not oi_is_structural and len(oi_5yr) >= 3:
        mu  = oi_5yr.mean()
        std = oi_5yr.std(ddof=1)
        if std > 0:
            z_score = (reported_oi - mu) / std
            if abs(z_score) > zscore_threshold:
                # FIXED: only strip the spike above the normal level
                abnormal_oi = max(reported_oi - norm_oi, 0.0)
    elif not oi_is_structural and reported_oi > norm_oi * 3:
        # Coarse fallback when < 3 data points: strip only the excess
        abnormal_oi = reported_oi - norm_oi

    # ── Exceptional Items for latest year ─────────────────────────────────────
    # Schema: profit_loss metrics include 'Exceptional Items' — confirmed.
    exc_latest = 0.0
    if not exc_df.empty:
        exc_ann = (
            exc_df.set_index("dt")["val"]
            .resample("YE").sum()
            .reset_index()
            .rename(columns={"dt": "year"})
        )
        yr_exc = exc_ann[exc_ann["year"].dt.year == latest_year.year]["val"]
        exc_latest = float(yr_exc.sum()) if not yr_exc.empty else 0.0

    # ── Tax adjustment on net OI change + exceptional ─────────────────────────
    # income_delta = net pre-tax change from our adjustments
    # When oi_is_structural: abnormal_oi=0 and norm_oi ≈ reported_oi → delta ≈ 0, no tax adj
    income_delta = -abnormal_oi - exc_latest + norm_oi
    tax_adj      = -income_delta * tax_rate

    # ── Receivables Risk Haircut ───────────────────────────────────────────────
    # Schema fix: debtor_days now comes pre-computed from ratios view via _get_receivables_and_revenue
    recv_adj           = 0.0
    debtor_days_latest = None
    debtor_days_avg    = None
    excess_recv_cr     = 0.0

    if recv_df is not None and not recv_df.empty and recv_haircut_pct > 0:
        _eff_margin = net_margin if (net_margin and net_margin > 0) else 0.10
        if len(recv_df) >= 2:
            debtor_days_latest = float(recv_df["debtor_days"].iloc[-1])
            debtor_days_avg    = float(recv_df["debtor_days"].iloc[:-1].mean())
            excess_days        = max(debtor_days_latest - debtor_days_avg, 0)
            latest_revenue     = float(recv_df["revenue"].iloc[-1])
            excess_recv_cr     = excess_days / 365 * latest_revenue
            recv_adj           = -(excess_recv_cr * (recv_haircut_pct / 100) * _eff_margin)

    # ── Order Book Profit Uplift ───────────────────────────────────────────────
    ob_adj = 0.0
    if order_book_cr > 0:
        _ob_margin     = (ob_margin_pct / 100) if ob_margin_pct else (net_margin or 0.10)
        executable_rev = order_book_cr * (ob_executable_pct / 100)
        ob_adj         = executable_rev * _ob_margin

    # ── Assemble ──────────────────────────────────────────────────────────────
    norm_profit = reported - abnormal_oi - exc_latest + norm_oi + tax_adj + recv_adj + ob_adj

    detail = {
        "Reported Net Profit":            reported,
        "Reported Other Income":          reported_oi,
        "Other Income Z-score":           round(z_score, 2),
        "OI is structural (no z-score)":  oi_is_structural,
        "Abnormal Other Income removed":  -abnormal_oi,
        "Exceptional Items removed":      -exc_latest,
        "Normalised Other Income added":  norm_oi,
        "Tax Adjustment":                 tax_adj,
        # Receivables
        "Debtor Days (latest)":           debtor_days_latest,
        "Debtor Days (5yr avg)":          debtor_days_avg,
        "Excess Receivables (₹ Cr)":      round(excess_recv_cr, 1),
        "Receivables Risk Haircut":        recv_adj,
        # Order book
        "Order Book (₹ Cr)":              order_book_cr,
        "Order Book Profit Uplift":        ob_adj,
        "Normalized Earnings":            norm_profit,
    }
    return norm_profit, detail