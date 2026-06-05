"""
dcf_analysis.py
───────────────
Intrinsic Value / DCF Analysis page.

Layout
──────
1. Portfolio DCF Scan   — loops every holding, shows Undervalued / Overvalued cards
                          clicking a card sets the active ticker and scrolls to detail
2. Stock Selector       — portfolio dropdown + all-stocks dropdown
3. DCF Detail           — sliders, verdict, four charts, sensitivity table
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as _st_components

from ui.pages.one_ticker_common import COLORS, load_matrix


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG LOADER  (config/dcf.csv — optional, graceful fallback)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=600)
def _load_dcf_config_raw() -> pd.DataFrame:
    """
    Parse config/dcf.csv once and cache the raw DataFrame.

    Wide format — one row per ticker (or blank for globals), one column per param:
        ticker,g1,g2,tg,dr,mos,zscore_thresh
        ,14.0,10.0,5.0,12.0,25,2.0          ← global fallback (blank ticker)
        RELIANCE,18.0,12.0,5.0,11.0,20,2.0  ← ticker-specific override
        INFY,16.0,11.0,5.0,12.0,25,2.0

    Column names are lowercased before lookup.
    Returns an empty DataFrame on any error so callers degrade gracefully.
    """
    try:
        df = pd.read_csv("config/dcf.csv")
        df.columns = [c.strip().lower() for c in df.columns]
        # Normalise the ticker column: strip whitespace, upper-case, NaN → ""
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].fillna("").astype(str).str.strip().str.upper()
        else:
            # No ticker column → treat every row as the global fallback
            df.insert(0, "ticker", "")
        return df
    except Exception:
        return pd.DataFrame()


def _load_dcf_config(ticker: str = "") -> dict:
    """
    Return the merged assumption dict for *ticker*.

    Resolution order (highest → lowest priority):
        1. Ticker-specific row  (ticker column == TICKER, case-insensitive)
        2. Global fallback row  (ticker column is blank / NaN)
        3. Empty dict           (file absent or no matching rows)

    Callers should always use  .get(key, hard_coded_default)  for safety.
    """
    df = _load_dcf_config_raw()
    if df.empty:
        return {}

    param_cols = [c for c in df.columns if c != "ticker"]

    def _row_to_dict(row) -> dict:
        out = {}
        for col in param_cols:
            try:
                v = float(row[col])
                if not pd.isna(v):
                    out[col] = v
            except (ValueError, TypeError):
                pass
        return out

    # Global fallback (blank ticker)
    global_rows = df[df["ticker"] == ""]
    result = _row_to_dict(global_rows.iloc[0]) if not global_rows.empty else {}

    # Ticker-specific override
    if ticker:
        tkr_rows = df[df["ticker"] == ticker.strip().upper()]
        if not tkr_rows.empty:
            result.update(_row_to_dict(tkr_rows.iloc[0]))

    return result



@st.cache_data(ttl=300)
def _get_net_profit(tkr: str, _conn) -> pd.DataFrame:
    try:
        df = _conn.execute(f"""
            SELECT dt, val FROM profit_loss
            WHERE ticker = '{tkr}' AND metric = 'Net Profit'
            ORDER BY dt
        """).df()
        df["dt"] = pd.to_datetime(df["dt"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def _get_eps(tkr: str, _conn) -> pd.DataFrame:
    try:
        df = _conn.execute(f"""
            SELECT dt, val FROM profit_loss
            WHERE ticker = '{tkr}' AND metric = 'EPS in Rs'
            ORDER BY dt
        """).df()
        df["dt"] = pd.to_datetime(df["dt"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def _get_price_history(tkr: str, _conn) -> pd.DataFrame:
    try:
        df = _conn.execute(f"""
            SELECT Date AS dt, Close AS price
            FROM ticker_prices
            WHERE REPLACE(Ticker, '.NS', '') = '{tkr}'
            ORDER BY Date
        """).df()
        df["dt"] = pd.to_datetime(df["dt"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def _get_growth_rates(tkr: str, _conn) -> dict:
    rates = {}
    try:
        df = _conn.execute(f"""
            SELECT horizon, val FROM compounded_profit_growth
            WHERE ticker = '{tkr}'
        """).df()
        for _, r in df.iterrows():
            rates[r["horizon"]] = float(r["val"])
    except Exception:
        pass
    return rates


@st.cache_data(ttl=300)
def _get_general_info(tkr: str, _conn) -> dict:
    try:
        row = _conn.execute(f"""
            SELECT current_price, stock_p_e, book_value, dividend_yield, roe, roce
            FROM general_info WHERE ticker = '{tkr}' LIMIT 1
        """).df()
        if not row.empty:
            return row.iloc[0].to_dict()
    except Exception:
        pass
    return {}


@st.cache_data(ttl=300)
def _get_historical_pe(tkr: str, _conn) -> dict:
    """
    Compute time-decay weighted P/E from price history and EPS.

    Time decay logic
    ────────────────
    Each year gets an exponential weight = exp(decay_rate × rank) where rank=0
    is the OLDEST year and rank=N-1 is the MOST RECENT year.
    decay_rate = 0.3  →  most-recent year gets ~e^(0.3×N) times the weight of
    the oldest year.  For a 10-year window that gives the last year ~20× the
    weight of year 1, and the last 3 years collectively ~60% of total weight.

    Returns dict with keys:
        weighted_median_pe  — time-decay weighted median (primary metric)
        median_pe           — simple (unweighted) median for reference
        mean_pe             — simple mean
        min_pe, max_pe      — historical range
        years_used          — number of annual data points
        pe_series           — DataFrame with year, pe, weight columns
    """
    try:
        price_df = _conn.execute(f"""
            SELECT Date AS dt, Close AS price
            FROM ticker_prices
            WHERE REPLACE(Ticker, '.NS', '') = '{tkr}'
            ORDER BY Date
        """).df()
        eps_df = _conn.execute(f"""
            SELECT dt, val FROM profit_loss
            WHERE ticker = '{tkr}' AND metric = 'EPS in Rs'
            ORDER BY dt
        """).df()
        if price_df.empty or eps_df.empty:
            return {}
        price_df["dt"] = pd.to_datetime(price_df["dt"])
        eps_df["dt"]   = pd.to_datetime(eps_df["dt"])

        # Annual EPS
        eps_ann = (eps_df.set_index("dt")["val"]
                   .resample("YE").mean()
                   .reset_index()
                   .rename(columns={"dt": "year", "val": "eps"}))
        eps_ann = eps_ann[eps_ann["eps"] > 0]
        if eps_ann.empty:
            return {}

        # Annual avg price
        price_ann = (price_df.set_index("dt")["price"]
                     .resample("YE").mean()
                     .reset_index()
                     .rename(columns={"dt": "year", "price": "avg_price"}))

        merged = pd.merge(price_ann, eps_ann, on="year", how="inner")
        merged["pe"] = merged["avg_price"] / merged["eps"]
        merged = merged[(merged["pe"] > 3) & (merged["pe"] < 300)]  # sanity filter
        if merged.empty:
            return {}

        # ── Time-decay weights ────────────────────────────────────────────────
        # Sort oldest → newest so rank 0 = oldest, rank N-1 = most recent
        merged = merged.sort_values("year").reset_index(drop=True)
        n = len(merged)
        decay_rate = 0.3          # tune here: higher = more recency bias
        raw_w = np.array([np.exp(decay_rate * i) for i in range(n)])
        weights = raw_w / raw_w.sum()          # normalise to sum=1
        merged["weight"] = weights
        merged["weight_pct"] = (weights * 100).round(1)

        # Weighted median: sort by P/E, find the P/E where cumulative weight ≥ 0.5
        pe_sorted = merged.sort_values("pe").reset_index(drop=True)
        pe_sorted["cum_w"] = pe_sorted["weight"].cumsum()
        weighted_median = float(pe_sorted[pe_sorted["cum_w"] >= 0.5]["pe"].iloc[0])

        # Weighted mean
        weighted_mean = float((merged["pe"] * merged["weight"]).sum())

        return {
            "weighted_median_pe": weighted_median,
            "weighted_mean_pe":   weighted_mean,
            "median_pe":          float(merged["pe"].median()),
            "mean_pe":            float(merged["pe"].mean()),
            "min_pe":             float(merged["pe"].min()),
            "max_pe":             float(merged["pe"].max()),
            "years_used":         n,
            "decay_rate":         decay_rate,
            "pe_series":          merged[["year", "pe", "weight_pct"]].copy(),
        }
    except Exception:
        return {}


@st.cache_data(ttl=300)
def _get_shares_outstanding(tkr: str, _conn) -> float | None:
    try:
        eq = _conn.execute(f"""
            SELECT val FROM balance_sheet
            WHERE ticker = '{tkr}' AND metric = 'Equity Capital'
            ORDER BY dt DESC LIMIT 1
        """).df()
        fv = _conn.execute(f"""
            SELECT face_value FROM general_info WHERE ticker = '{tkr}' LIMIT 1
        """).df()
        if not eq.empty and not fv.empty:
            equity_cr = float(eq.iloc[0]["val"])
            face      = float(fv.iloc[0]["face_value"]) or 10.0
            shares    = equity_cr * 1e7 / face
            return shares / 1e7          # in crores
    except Exception:
        pass
    return None




@st.cache_data(ttl=300)
def _get_other_income(tkr: str, _conn) -> pd.DataFrame:
    """Annual Other Income series."""
    try:
        df = _conn.execute(f"""
            SELECT dt, val FROM profit_loss
            WHERE ticker = '{tkr}' AND metric = 'Other Income'
            ORDER BY dt
        """).df()
        df["dt"] = pd.to_datetime(df["dt"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def _get_exceptional_items(tkr: str, _conn) -> pd.DataFrame:
    """Annual Exceptional Items series (signed — negative = charge)."""
    try:
        df = _conn.execute(f"""
            SELECT dt, val FROM profit_loss
            WHERE ticker = '{tkr}'
              AND metric IN ('Exceptional Items', 'Exceptional Item', 'Extraordinary Items')
            ORDER BY dt
        """).df()
        df["dt"] = pd.to_datetime(df["dt"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def _get_tax_rate(tkr: str, _conn) -> float:
    """Estimate effective tax rate from last 3 years of data (PBT / Tax)."""
    try:
        pbt = _conn.execute(f"""
            SELECT dt, val FROM profit_loss
            WHERE ticker = '{tkr}' AND metric = 'Profit before tax'
            ORDER BY dt DESC LIMIT 3
        """).df()
        tax = _conn.execute(f"""
            SELECT dt, val FROM profit_loss
            WHERE ticker = '{tkr}' AND metric = 'Tax %'
            ORDER BY dt DESC LIMIT 3
        """).df()
        if not tax.empty:
            # Tax % is stored directly
            return float(tax["val"].mean()) / 100.0
        if not pbt.empty:
            # Fallback: assume 25%
            pass
    except Exception:
        pass
    return 0.25          # default effective tax rate




@st.cache_data(ttl=300)
def _get_receivables_and_revenue(tkr: str, _conn) -> pd.DataFrame:
    """
    Returns annual DataFrame with columns: year, revenue, receivables, debtor_days.

    Schema fix: balance_sheet does NOT contain Trade Receivables / Debtors.
    The ratios view has metric = 'Debtor Days' directly — use that as the
    primary source. Revenue comes from profit_loss metric = 'Sales'.
    Receivables is back-computed as debtor_days / 365 × revenue for display.
    """
    try:
        # Primary: Debtor Days from ratios view (exists in schema)
        dd_df = _conn.execute(f"""
            SELECT dt, val AS debtor_days FROM ratios
            WHERE ticker = '{tkr}' AND metric = 'Debtor Days'
            ORDER BY dt
        """).df()

        # Revenue from profit_loss (metric = 'Sales' confirmed in schema)
        rev_df = _conn.execute(f"""
            SELECT dt, val AS revenue FROM profit_loss
            WHERE ticker = '{tkr}' AND metric = 'Sales'
            ORDER BY dt
        """).df()

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


@st.cache_data(ttl=300)
def _get_net_margin(tkr: str, _conn) -> float | None:
    """
    Average net profit margin over last 3 years.
    Schema note: profit_loss has no 'Net Profit %' metric.
    Computed as Net Profit / Sales using the two metrics that DO exist.
    """
    try:
        np_df = _conn.execute(f"""
            SELECT dt, val FROM profit_loss
            WHERE ticker = '{tkr}' AND metric = 'Net Profit'
            ORDER BY dt DESC LIMIT 3
        """).df()
        # profit_loss schema: metric = 'Sales' is the revenue line
        sal_df = _conn.execute(f"""
            SELECT dt, val FROM profit_loss
            WHERE ticker = '{tkr}' AND metric = 'Sales'
            ORDER BY dt DESC LIMIT 3
        """).df()
        if not np_df.empty and not sal_df.empty:
            np_df  = np_df.set_index("dt")["val"]
            sal_df = sal_df.set_index("dt")["val"]
            merged = pd.concat([np_df.rename("np"), sal_df.rename("sal")], axis=1).dropna()
            if not merged.empty and (merged["sal"] > 0).all():
                return float((merged["np"] / merged["sal"]).mean())
    except Exception:
        pass
    return None

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

def _run_dcf(base_profit: float, phase1: float, phase2: float,
             terminal: float, discount: float) -> float:
    """Two-stage DCF on Net Profit (₹ Cr). Returns total PV in ₹ Cr."""
    pv   = 0.0
    proj = base_profit
    for yr in range(1, 11):
        g    = phase1 if yr <= 5 else phase2
        proj = proj * (1 + g / 100)
        pv  += proj / (1 + discount / 100) ** yr
    if discount > terminal:
        tv  = proj * (1 + terminal / 100) / (discount / 100 - terminal / 100)
        pv += tv / (1 + discount / 100) ** 10
    return pv


def _quick_dcf(tkr: str, conn, dr: float = 12.0, tg: float = 5.0) -> dict | None:
    """
    Lightweight DCF for portfolio scan — uses historical 3Y growth as Phase 1,
    5Y as Phase 2. Returns dict with iv_base, current_price, premium_pct, or None.
    """
    np_df  = _get_net_profit(tkr, conn)
    if np_df.empty:
        return None
    np_ann = (
        np_df.set_index("dt")["val"]
        .resample("YE").sum()
        .reset_index()
        .rename(columns={"dt": "year", "val": "net_profit"})
    )
    np_ann = np_ann[np_ann["net_profit"] > 0].tail(10)
    if np_ann.empty:
        return None

    # Normalize earnings before DCF
    oi_df  = _get_other_income(tkr, conn)
    exc_df = _get_exceptional_items(tkr, conn)
    tax_r  = _get_tax_rate(tkr, conn)
    latest_profit, _ = normalize_earnings(np_ann, oi_df, exc_df, tax_r)
    if latest_profit <= 0:
        latest_profit = float(np_ann["net_profit"].iloc[-1])  # fallback to reported

    growth = _get_growth_rates(tkr, conn)
    g1 = growth.get("3 Years", growth.get("3Y", 12.0))
    g2 = growth.get("5 Years", growth.get("5Y", 10.0))
    g1 = max(g1, 0); g2 = max(min(g2, g1), 0)

    shares_cr = _get_shares_outstanding(tkr, conn)
    if shares_cr is None:
        eps_df = _get_eps(tkr, conn)
        if not eps_df.empty:
            eps_ann = eps_df.set_index("dt")["val"].resample("YE").mean().reset_index()
            if not eps_ann.empty:
                latest_eps = float(eps_ann["val"].iloc[-1])
                reported_profit = float(np_ann["net_profit"].iloc[-1]) # Use reported profit here
                if latest_eps > 0 and reported_profit > 0:
                    shares_cr = reported_profit / latest_eps

    if not shares_cr or shares_cr <= 0:
        return None

    dcf_val  = _run_dcf(latest_profit, g1, g2, tg, dr)
    iv_base  = dcf_val / shares_cr

    info     = _get_general_info(tkr, conn)
    price_df = _get_price_history(tkr, conn)
    cur_price = info.get("current_price") or (
        float(price_df["price"].iloc[-1]) if not price_df.empty else None
    )
    if not cur_price:
        return None

    premium = (cur_price - iv_base) / iv_base * 100

    # ── Historical P/E verdict (for dual-flag detection) ──────────────────────
    pe_premium = None
    hist_pe = _get_historical_pe(tkr, conn)
    eps_df2  = _get_eps(tkr, conn)
    if hist_pe and not eps_df2.empty:
        eps_ann2 = eps_df2.set_index("dt")["val"].resample("YE").mean().reset_index()
        if not eps_ann2.empty:
            latest_eps2 = float(eps_ann2["val"].iloc[-1])
            if latest_eps2 > 0:
                fv_median = hist_pe["weighted_median_pe"] * latest_eps2
                if fv_median > 0:
                    pe_premium = (cur_price - fv_median) / fv_median * 100

    return {
        "ticker":        tkr,
        "iv_base":       iv_base,
        "current_price": cur_price,
        "premium_pct":   premium,
        "pe_premium":    pe_premium,   # None if P/E data unavailable
        "g1": g1, "g2": g2,
        "company":       tkr,          # fallback; overwritten below if available
    }


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO SCAN SECTION
# ══════════════════════════════════════════════════════════════════════════════

def _render_portfolio_scan(
        portfolio_tickers: list[str],
        conn,
        df_matrix: pd.DataFrame,
        pnl_map: dict | None = None,
):
    """Loops portfolio tickers, runs quick DCF, renders card rows with P&L."""

    if not portfolio_tickers:
        st.info("No portfolio holdings found in session. Connect your broker or add holdings.")
        return

    pnl_map = pnl_map or {}

    # Company name lookup from matrix
    name_map = (
        df_matrix.set_index("Ticker")["Company"].to_dict()
        if "Company" in df_matrix.columns else {}
    )

    with st.spinner(f"Running DCF scan on {len(portfolio_tickers)} holdings…"):
        results = []
        for tkr in portfolio_tickers:
            r = _quick_dcf(tkr, conn)
            if r:
                r["company"]  = name_map.get(tkr, tkr)
                r["pnl_data"] = pnl_map.get(tkr)
                results.append(r)

    if not results:
        st.warning("Could not compute DCF for any portfolio holding (missing profit data).")
        return

    undervalued = sorted([r for r in results if r["premium_pct"] < -20],
                         key=lambda x: x["premium_pct"])
    overvalued  = sorted([r for r in results if r["premium_pct"] >  20],
                         key=lambda x: x["premium_pct"], reverse=True)
    fair        = [r for r in results if -20 <= r["premium_pct"] <= 20]

    # Tickers flagged by BOTH DCF and P/E method (pe_premium not None)
    dual_over  = sorted(
        [r for r in overvalued  if r.get("pe_premium") is not None and r["pe_premium"] >  20],
        key=lambda x: x["premium_pct"], reverse=True,
    )
    dual_under = sorted(
        [r for r in undervalued if r.get("pe_premium") is not None and r["pe_premium"] < -20],
        key=lambda x: x["premium_pct"],
    )

    # ── HTML card renderer (shared) ────────────────────────────────────────────
    def _render_card(r: dict, icon: str, section: str,
                     primary_lbl: str, secondary_lbl: str):
        """
        Renders a styled HTML card with coloured P&L + today's gain,
        plus a small 'Open ↗' button for navigation.
        """
        pd_     = r.get("pnl_data") or {}
        pnl     = pd_.get("pnl", None)
        pnl_pct = pd_.get("pnl_pct", None)
        qty     = pd_.get("qty", None)
        t_gain  = pd_.get("today_gain", None)
        t_pct   = pd_.get("today_pct", None)

        # P&L row
        if pnl is not None:
            pnl_color  = "#16a34a" if pnl >= 0 else "#dc2626"
            pnl_arrow  = "▲" if pnl >= 0 else "▼"
            pnl_sign   = "+" if pnl >= 0 else "-"
            pnl_html   = (
                f"<div style='margin-top:6px;font-size:0.82rem;font-weight:600;"
                f"color:{pnl_color}'>"
                f"{'📈' if pnl >= 0 else '📉'} P&amp;L&nbsp; "
                f"{pnl_arrow} ₹{abs(pnl):,.0f} "
                f"<span style='font-size:0.78rem'>({pnl_sign}{abs(pnl_pct):.1f}%)"
                f"{'  · qty '+str(qty) if qty else ''}</span></div>"
            )
        else:
            pnl_html = ""

        # Today's gain row
        if t_gain is not None:
            tg_color = "#16a34a" if t_gain >= 0 else "#dc2626"
            tg_arrow = "▲" if t_gain >= 0 else "▼"
            tg_sign  = "+" if t_gain >= 0 else "-"
            tg_html  = (
                f"<div style='font-size:0.78rem;font-weight:500;color:{tg_color}'>"
                f"Today&nbsp;{tg_arrow} ₹{abs(t_gain):,.0f} "
                f"<span style='font-size:0.75rem'>({tg_sign}{abs(t_pct):.2f}%)</span></div>"
            )
        else:
            tg_html = ""

        card_html = f"""
        <div style="
            border:1px solid #e2e8f0; border-radius:10px;
            padding:12px 14px 8px 14px; margin-bottom:2px;
            background:#fff;
        ">
            <div style='font-size:1rem;font-weight:700;color:#1e293b'>
                {icon} {r['ticker']}
            </div>
            <div style='font-size:0.85rem;color:#475569;margin-top:2px'>
                ₹{r['current_price']:,.0f}
            </div>
            <div style='font-size:0.78rem;color:#64748b;margin-top:4px'>
                {primary_lbl}
            </div>
            <div style='font-size:0.75rem;color:#94a3b8'>
                {secondary_lbl}
            </div>
            {pnl_html}
            {tg_html}
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        if st.button("Open ↗", key=f"port_card_{r['ticker']}_{section}",
                     use_container_width=True):
            st.session_state["dcf_active_ticker"] = r["ticker"]
            st.session_state["_dcf_source"]       = "portfolio"
            st.session_state["_scroll_to_norm"]   = True
            st.rerun()

    def _card_row_dual(items, icon, section):
        cols = st.columns(min(len(items), 4))
        for i, r in enumerate(items):
            with cols[i % 4]:
                dcf_lbl = (
                    f"DCF: {abs(r['premium_pct']):.1f}% "
                    f"{'below' if r['premium_pct'] < 0 else 'above'}"
                )
                pe_lbl = (
                    f"P/E: {abs(r['pe_premium']):.1f}% "
                    f"{'below' if r['pe_premium'] < 0 else 'above'} hist. median"
                    if r.get("pe_premium") is not None else ""
                )
                _render_card(r, icon, section, dcf_lbl, pe_lbl)

    def _card_row_pe(items, icon, section):
        cols = st.columns(min(len(items), 4))
        for i, r in enumerate(items):
            with cols[i % 4]:
                pe_prem = r.get("pe_premium")
                pe_lbl  = (
                    f"{abs(pe_prem):.1f}% "
                    f"{'below' if pe_prem < 0 else 'above'} P/E fair value"
                    if pe_prem is not None else "P/E data unavailable"
                )
                dcf_lbl = (
                    f"DCF: {abs(r['premium_pct']):.1f}% "
                    f"{'below' if r['premium_pct'] < 0 else 'above'}"
                )
                _render_card(r, icon, section, pe_lbl, dcf_lbl)

    # ── Flagged by BOTH methods ────────────────────────────────────────────────
    if dual_under or dual_over:
        st.markdown(
            "<div style='margin:6px 0 4px;font-size:1rem;font-weight:700;"
            "color:#7c3aed'>🔥 Flagged by Both DCF & Historical P/E  "
            "<span style='font-weight:400;font-size:0.85rem;color:#64748b'>"
            "agreement between both valuation methods</span></div>",
            unsafe_allow_html=True,
        )
        if dual_under:
            st.caption("🟢 Undervalued — below fair value on both DCF and historical P/E")
            _card_row_dual(dual_under, "🔥🟢", "dual_under")
        if dual_over:
            st.caption("🔴 Overvalued — above fair value on both DCF and historical P/E")
            _card_row_dual(dual_over, "🔥🔴", "dual_over")
        st.markdown(
            "<div style='height:6px;border-bottom:1px dashed #ddd8fe;margin-bottom:10px'></div>",
            unsafe_allow_html=True,
        )

    # ── Individual sections — classified by Historical P/E ────────────────────
    # Split results by P/E verdict; fall back to DCF verdict when P/E unavailable
    pe_under = sorted(
        [r for r in results if r.get("pe_premium") is not None and r["pe_premium"] < -20],
        key=lambda x: x["pe_premium"],
    )
    pe_over  = sorted(
        [r for r in results if r.get("pe_premium") is not None and r["pe_premium"] >  20],
        key=lambda x: x["pe_premium"], reverse=True,
    )
    pe_fair  = sorted(
        [r for r in results if r.get("pe_premium") is not None and -20 <= r["pe_premium"] <= 20],
        key=lambda x: x["pe_premium"],
    )
    # Tickers with no P/E data — fall back to DCF classification
    no_pe    = sorted(
        [r for r in results if r.get("pe_premium") is None],
        key=lambda x: x["premium_pct"],
    )

    if pe_under:
        st.markdown(
            "<div style='margin:6px 0 4px;font-size:1rem;font-weight:700;"
            "color:#16a34a'>🟢 Undervalued  "
            "<span style='font-weight:400;font-size:0.85rem;color:#64748b'>"
            "trading below Historical P/E fair value by >20%</span></div>",
            unsafe_allow_html=True,
        )
        _card_row_pe(pe_under, "🟢", "pe_under")

    if pe_fair:
        st.markdown(
            "<div style='margin:14px 0 4px;font-size:1rem;font-weight:700;"
            "color:#ca8a04'>🟡 Fairly Valued  "
            "<span style='font-weight:400;font-size:0.85rem;color:#64748b'>"
            "within ±20% of Historical P/E fair value</span></div>",
            unsafe_allow_html=True,
        )
        _card_row_pe(pe_fair, "🟡", "pe_fair")

    if pe_over:
        st.markdown(
            "<div style='margin:14px 0 4px;font-size:1rem;font-weight:700;"
            "color:#dc2626'>🔴 Overvalued  "
            "<span style='font-weight:400;font-size:0.85rem;color:#64748b'>"
            "trading above Historical P/E fair value by >20%</span></div>",
            unsafe_allow_html=True,
        )
        _card_row_pe(pe_over, "🔴", "pe_over")

    if no_pe:
        st.markdown(
            "<div style='margin:14px 0 4px;font-size:1rem;font-weight:700;"
            "color:#64748b'>⚪ DCF Only  "
            "<span style='font-weight:400;font-size:0.85rem;color:#64748b'>"
            "insufficient P/E history — classified by DCF</span></div>",
            unsafe_allow_html=True,
        )
        _card_row_pe(no_pe, "⚪", "no_pe")

    skipped = len(portfolio_tickers) - len(results)
    if skipped:
        st.caption(f"⚠️ {skipped} holding(s) skipped — no profit history available.")


# ══════════════════════════════════════════════════════════════════════════════
# DCF DETAIL SECTION
# ══════════════════════════════════════════════════════════════════════════════

def _render_dcf_detail(ticker: str, conn, df_matrix: pd.DataFrame):
    np_df    = _get_net_profit(ticker, conn)
    eps_df   = _get_eps(ticker, conn)
    price_df = _get_price_history(ticker, conn)
    growth   = _get_growth_rates(ticker, conn)
    info     = _get_general_info(ticker, conn)

    current_price = info.get("current_price") or (
        float(price_df["price"].iloc[-1]) if not price_df.empty else None
    )

    if np_df.empty:
        st.warning(f"No Net Profit data found for {ticker}.")
        return

    np_annual = (
        np_df.set_index("dt")["val"]
        .resample("YE").sum()
        .reset_index()
        .rename(columns={"dt": "year", "val": "net_profit"})
    )
    np_annual = np_annual[np_annual["net_profit"] > 0].tail(10)
    if np_annual.empty:
        st.warning(f"Insufficient positive profit data for {ticker}.")
        return

    reported_profit = float(np_annual["net_profit"].iloc[-1])

    # ── Earnings Normalisation ────────────────────────────────────────────────
    oi_df     = _get_other_income(ticker, conn)
    exc_df    = _get_exceptional_items(ticker, conn)
    tax_r     = _get_tax_rate(ticker, conn)
    recv_df   = _get_receivables_and_revenue(ticker, conn)
    net_margin = _get_net_margin(ticker, conn)

    # Reset sliders when ticker changes — write config/dcf.csv values directly
    # into session_state so Streamlit's slider 'value=' is actually honoured.
    if st.session_state.get("_dcf_detail_ticker") != ticker:
        for k in ["g1", "g2", "tg", "dr", "mos", "one_time_exp",
                  "zscore_thresh", "oi_structural", "recv_haircut",
                  "ob_cr", "ob_exec_pct", "ob_margin"]:
            st.session_state.pop(k, None)

        # Pre-seed slider keys from config/dcf.csv so the new values are shown
        _reset_cfg = _load_dcf_config(ticker)
        if _reset_cfg:
            _r3y  = growth.get("3 Years", growth.get("3Y", 12.0))
            _r5y  = growth.get("5 Years", growth.get("5Y", 10.0))
            _rg1  = float(round(_reset_cfg.get("g1", _r3y), 1))
            _rg2  = float(round(_reset_cfg.get("g2", min(_r5y, _rg1)), 1))
            st.session_state["g1"]  = _rg1
            st.session_state["g2"]  = float(round(min(_rg2, _rg1), 1))
            st.session_state["tg"]  = float(_reset_cfg.get("tg",  5.0))
            st.session_state["dr"]  = float(_reset_cfg.get("dr",  12.0))
            st.session_state["mos"] = int(_reset_cfg.get("mos", 25))
            if "zscore_thresh" in _reset_cfg:
                st.session_state["zscore_thresh"] = float(_reset_cfg["zscore_thresh"])

        st.session_state["_dcf_detail_ticker"] = ticker

    st.markdown("---")
    # Anchor for scroll-to from portfolio card Open button
    st.markdown(
        "<div id='earnings-normalisation-anchor'></div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"#### 🧮 Earnings Normalisation {ticker}")

    # Scroll to this section if triggered from portfolio card click
    if st.session_state.pop("_scroll_to_norm", False):
        _st_components.html(
            """
            <script>
                window.parent.document
                    .getElementById('earnings-normalisation-anchor')
                    ?.scrollIntoView({behavior: 'smooth', block: 'start'});
            </script>
            """,
            height=0,
        )

    # ── Row 1: Other Income z-score + structural toggle + one-time exp ────────
    _zcol, _oitoggle, _otcol = st.columns([2, 2, 2])
    with _zcol:
        _cfg2 = _load_dcf_config(ticker)
        _def_zthresh = _cfg2.get("zscore_thresh", 2.0)
        zscore_thresh = st.slider(
            "Z-score threshold for abnormal Other Income",
            min_value=1.0, max_value=4.0, value=_def_zthresh, step=0.5, key="zscore_thresh",
            help=(
                "Detects spikes in Other Income using z-score. A threshold of 2.0 flags "
                "any year where OI is >2 std deviations above the 5-yr mean. "
                "Only the SPIKE (excess above normal) is removed — not the full OI. "
                "Disabled when 'Structural OI' is checked."
            ),
        )
    with _oitoggle:
        oi_is_structural = st.checkbox(
            "Other Income is structural (e.g. FD interest, treasury)",
            value=False,
            key="oi_structural",
            help=(
                "Check this for companies like Coal India, ONGC, or any cash-rich PSU "
                "where Other Income is FD/bond interest earned on permanent cash reserves. "
                "This income recurs every year and is as reliable as operating profit — "
                "the z-score filter should NOT strip it. "
                "When checked: z-score removal is skipped entirely; the 5-yr average OI "
                "is used as-is, so normalization has no net effect on Other Income."
            ),
        )
        if oi_is_structural:
            st.caption("ℹ️ Z-score removal disabled — OI treated as recurring.")
    with _otcol:
        one_time_exp = st.number_input(
            "One-time expenditures to add back (₹ Cr)",
            min_value=0.0, value=0.0, step=10.0, key="one_time_exp",
            help="Add back non-recurring expenses (restructuring, legal settlements) "
                 "that artificially depressed reported profit this year.",
        )

    # ── Row 2: Receivables Risk ───────────────────────────────────────────────
    st.markdown("**📦 Receivables Risk**")
    if not recv_df.empty and len(recv_df) >= 2:
        _dd_latest = float(recv_df["debtor_days"].iloc[-1])
        _dd_avg    = float(recv_df["debtor_days"].iloc[:-1].mean())
        _dd_trend  = "🔺 rising" if _dd_latest > _dd_avg * 1.1 else ("🔻 falling" if _dd_latest < _dd_avg * 0.9 else "➡ stable")
        _rc1, _rc2, _rc3 = st.columns(3)
        _rc1.metric("Debtor Days (latest)", f"{_dd_latest:.0f} days")
        _rc2.metric("Debtor Days (5yr avg)", f"{_dd_avg:.0f} days")
        _rc3.metric("Trend", _dd_trend)

        # Debtor days history mini-table
        with st.expander("Debtor Days History", expanded=False):
            _dd_disp = recv_df[["year", "revenue", "receivables", "debtor_days"]].copy()
            _dd_disp["year"] = _dd_disp["year"].dt.year
            _dd_disp.columns = ["Year", "Revenue (₹ Cr)", "Receivables (₹ Cr)", "Debtor Days"]
            st.dataframe(_dd_disp, use_container_width=True, hide_index=True)

        recv_haircut = st.slider(
            "Receivables at-risk haircut %",
            min_value=0, max_value=100, value=0, step=5, key="recv_haircut",
            help=(
                f"% of excess receivables (above {_dd_avg:.0f}-day avg) treated as uncollectable. "
                f"Excess = {max(_dd_latest - _dd_avg, 0):.0f} days of revenue."
            ),
        )
    else:
        st.caption("_No receivables data available in DB for this ticker._")
        recv_haircut = 0

    # ── Row 3: Order Book ─────────────────────────────────────────────────────
    st.markdown("**📋 Order Book / Confirmed Revenue**")
    _ob1, _ob2, _ob3 = st.columns(3)
    with _ob1:
        ob_cr = st.number_input(
            "Order Book value (₹ Cr)",
            min_value=0.0, value=0.0, step=100.0, key="ob_cr",
            help="This allows you to factor a company's unexecuted order backlog directly into its valuation:",
        )
    with _ob2:
        ob_exec_pct = st.slider(
            "Executable in DCF horizon (%)",
            min_value=0, max_value=100, value=50, step=5, key="ob_exec_pct",
            help="What % of the order book converts to revenue in the next 1–2 years",
        )
    with _ob3:
        _default_margin = round((net_margin or 0.10) * 100, 1)
        ob_margin = st.number_input(
            "Order book margin % (net)",
            min_value=0.0, max_value=50.0, value=_default_margin, step=0.5, key="ob_margin",
            help="Expected net profit margin on order book revenue. Defaults to company's historical margin.",
        )

    # ── Run normalisation ────────────────────────────────────────────────────
    st.subheader("Normalized Earning")
    norm_profit_raw, norm_detail = normalize_earnings(
        np_annual, oi_df, exc_df, tax_r,
        zscore_threshold=zscore_thresh,
        recv_df=recv_df,
        net_margin=net_margin,
        recv_haircut_pct=float(recv_haircut),
        order_book_cr=float(ob_cr),
        ob_executable_pct=float(ob_exec_pct),
        ob_margin_pct=float(ob_margin) if ob_cr > 0 else None,
        oi_is_structural=oi_is_structural,
    )
    # Add one-time expenditures (post-tax add-back)
    norm_profit = norm_profit_raw + one_time_exp * (1 - tax_r)
    if norm_profit <= 0:
        norm_profit = reported_profit + one_time_exp * (1 - tax_r)

    if one_time_exp:
        norm_detail["One-time Expenditure add-back (post-tax)"] = one_time_exp * (1 - tax_r)
        norm_detail["Normalized Earnings"] = norm_profit

    # ── Waterfall table ──────────────────────────────────────────────────────
    _wf_rows = [
        ("Reported Net Profit",                norm_detail.get("Reported Net Profit", reported_profit), ""),
        ("− Abnormal Other Income",             norm_detail.get("Abnormal Other Income removed", 0),    "dimmed"),
        ("− Exceptional Items",                 norm_detail.get("Exceptional Items removed", 0),         "dimmed"),
        ("+ Normalised Other Income",           norm_detail.get("Normalised Other Income added", 0),     "dimmed"),
        ("± Tax Adjustment",                   norm_detail.get("Tax Adjustment", 0),                    "dimmed"),
        ("+ One-time Exp. add-back",            norm_detail.get("One-time Expenditure add-back (post-tax)", 0), "dimmed"),
        ("− Receivables Risk Haircut",          norm_detail.get("Receivables Risk Haircut", 0),          "dimmed"),
        ("+ Order Book Profit Uplift",          norm_detail.get("Order Book Profit Uplift", 0),           "dimmed"),
        ("= Normalized Earnings",               norm_profit,                                               "bold"),
    ]

    _wf_html = """
    <table style='width:100%;border-collapse:collapse;font-size:0.85rem;font-family:monospace'>
    <tr style='border-bottom:1px solid #e2e8f0'>
      <th style='text-align:left;padding:4px 8px;color:#64748b'>Line Item</th>
      <th style='text-align:right;padding:4px 8px;color:#64748b'>₹ Cr</th>
      <th style='text-align:right;padding:4px 8px;color:#64748b'>Δ vs Reported</th>
    </tr>
    """
    for label, val, style in _wf_rows:
        if val == 0 and style == "dimmed":
            continue                          # skip zero-impact rows to keep table clean
        clr = "#1e293b" if style == "bold" else ("#94a3b8" if style == "dimmed" else "#334155")
        fw  = "700" if style == "bold" else "400"
        bdr = "border-top:2px solid #cbd5e1;" if style == "bold" else ""
        delta     = val - reported_profit if label == "= Normalized Earnings" else val
        delta_str = "" if label == "Reported Net Profit" else (
            f'<span style="color:{"#16a34a" if delta >= 0 else "#dc2626"}">'
            f'{"+" if delta >= 0 else ""}{delta:,.1f}</span>'
        )
        _wf_html += (
            f"<tr style='{bdr}'>"
            f"<td style='padding:4px 8px;color:{clr};font-weight:{fw}'>{label}</td>"
            f"<td style='text-align:right;padding:4px 8px;color:{clr};font-weight:{fw}'>{val:,.1f}</td>"
            f"<td style='text-align:right;padding:4px 8px'>{delta_str}</td>"
            f"</tr>"
        )
    _wf_html += "</table>"
    st.markdown(_wf_html, unsafe_allow_html=True)

    _oi_z = norm_detail.get("Other Income Z-score", 0)
    if norm_detail.get("OI is structural (no z-score)"):
        st.caption(
            f"ℹ️ Other Income treated as structural/recurring — z-score filter bypassed. "
            f"5-yr avg OI ₹{norm_detail.get('Normalised Other Income added', 0):,.1f} Cr used as base."
        )
    elif abs(_oi_z) > zscore_thresh:
        st.caption(
            f"⚠️ Other Income z-score = **{_oi_z:+.2f}** — spike detected and removed. "
            f"Only the excess above 5-yr avg is stripped. "
            f"5-yr avg Other Income ₹{norm_detail.get('Normalised Other Income added', 0):,.1f} Cr retained."
        )
    _recv_h = norm_detail.get("Receivables Risk Haircut", 0)
    if _recv_h < 0:
        st.caption(
            f"⚠️ Receivables haircut ₹{abs(_recv_h):,.1f} Cr — "
            f"debtor days {norm_detail.get('Debtor Days (latest)', 0):.0f} vs "
            f"{norm_detail.get('Debtor Days (5yr avg)', 0):.0f} avg. "
            f"Excess receivables ₹{norm_detail.get('Excess Receivables (₹ Cr)', 0):,.1f} Cr flagged."
        )
    if norm_detail.get("Order Book Profit Uplift", 0) > 0:
        st.caption(
            f"📋 Order book uplift ₹{norm_detail['Order Book Profit Uplift']:,.1f} Cr — "
            f"₹{ob_cr:,.0f} Cr backlog × {ob_exec_pct}% executable × {ob_margin:.1f}% margin."
        )
    if norm_profit <= 0:
        st.warning("Normalized earnings are ≤ 0. DCF will fall back to reported net profit.")
        norm_profit = reported_profit

    latest_profit = norm_profit

    # ── Assumptions ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⚙️ DCF Assumptions")

    # Load optional config overrides (config/dcf.csv)
    _cfg      = _load_dcf_config(ticker)
    _cfg_raw  = _load_dcf_config_raw()
    _has_ticker_row = (
            not _cfg_raw.empty
            and "ticker" in _cfg_raw.columns
            and _cfg_raw["ticker"].str.upper().eq(ticker.strip().upper()).any()
    )
    _cfg_source = ("ticker-specific" if _has_ticker_row else "global fallback") if _cfg else ""
    _cfg_note   = f" *(config/dcf.csv — {_cfg_source})*" if _cfg_source else ""

    with st.expander("📖 DCF assumptions guide", expanded=False):
        st.markdown(f"""
**What is DCF?**
Discounted Cash Flow values a stock by projecting future profits and discounting them
back to today's money.

**Your current assumptions for {ticker}:**
| Parameter | Value | What it means |
|---|---|---|
| Phase 1 Growth  | Annual profit growth for (yr 1–5) |
| Phase 2 Growth  | Annual profit growth for (yr 6–10) |
| Terminal Growth | Forever growth rate after year 10 |
| Discount Rate | Return (~7% Interest plus a 5% equity risk) |
| Margin of Safety | How much discount you demand before buying |

**Rules of thumb:**
- Terminal Value > 70% → highly speculative; small changes in `tg` swing value wildly
- Discount rate: Large-cap Indian 10–12%, Small/mid-cap 13–15%
- Terminal growth should not exceed India's long-run GDP growth (~6–7%)
        """)

    _3y_raw  = growth.get("3 Years", growth.get("3Y",  None))
    _5y_raw  = growth.get("5 Years", growth.get("5Y",  None))
    hist_3y  = _3y_raw  if _3y_raw  is not None else 12.0
    hist_5y  = _5y_raw  if _5y_raw  is not None else 10.0
    hist_10y = growth.get("10 Years", growth.get("10Y", 8.0))

    # Config overrides: if config/dcf.csv supplies a value, use it as the
    # slider default; otherwise fall back to the historical/hard-coded value.
    _def_g1  = _cfg.get("g1",  float(round(hist_3y, 1)))
    _def_g2  = _cfg.get("g2",  float(round(min(hist_5y, _def_g1), 1)))
    _def_tg  = _cfg.get("tg",  5.0)
    _def_dr  = _cfg.get("dr",  12.0)
    _def_mos = int(_cfg.get("mos", 25))

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown("**Profit Growth** *(Next 1–5 Yrs)*")
        st.caption(
            (f"Historical 3Y: {hist_3y:.1f}%" if _3y_raw else f"Default: {hist_3y:.1f}% (no data)")
            + (f" · config: {_cfg['g1']:.1f}%" if "g1" in _cfg else "")
        )
        g1 = st.slider("Profit Growth", 0.0, 100.0, _def_g1, step=0.5, key="g1", label_visibility="collapsed")
    with c2:
        st.markdown("**Profit Growth** *(Next 6–10 Yrs)*")
        st.caption(
            (f"Historical 5Y: {hist_5y:.1f}%" if _5y_raw else f"Default: {hist_5y:.1f}% (no data)")
            + (f" · config: {_cfg['g2']:.1f}%" if "g2" in _cfg else "")
        )
        g2 = st.slider("Profit Growth", 0.0, 30.0, float(round(min(_def_g2, g1), 1)), step=0.5, key="g2", label_visibility="collapsed")
    with c3:
        st.markdown("**Terminal Growth**")
        st.caption("Long-run perpetuity rate" + (f" · config: {_cfg['tg']:.1f}%" if "tg" in _cfg else ""))
        tg = st.slider("Terminal Growth", 2.0, 8.0, _def_tg, step=0.5, key="tg", label_visibility="collapsed")
    with c4:
        st.markdown("**Discount Rate (WACC)**")
        st.caption("Risk-free ~7% + premium" + (f" · config: {_cfg['dr']:.1f}%" if "dr" in _cfg else ""))
        dr = st.slider("Discount Rate", 8.0, 20.0, _def_dr, step=0.5, key="dr", label_visibility="collapsed")
    with c5:
        st.markdown("**Margin of Safety**")
        st.caption("Buy below intrinsic by X%" + (f" · config: {_cfg['mos']:.0f}%" if "mos" in _cfg else ""))
        mos = st.slider("Margin of Safety", 0, 50, _def_mos, step=5, key="mos", label_visibility="collapsed")

    if _cfg:
        st.caption(f"ℹ️ Assumption defaults loaded from `config/dcf.csv` ({_cfg_source}).")

    # ── DCF calc ─────────────────────────────────────────────────────────────
    shares_cr = _get_shares_outstanding(ticker, conn)
    if shares_cr is None and not eps_df.empty:
        eps_annual = eps_df.set_index("dt")["val"].resample("YE").mean().reset_index()
        if not eps_annual.empty:
            latest_eps = float(eps_annual["val"].iloc[-1])
            if latest_eps > 0 and reported_profit > 0:
                shares_cr = reported_profit / latest_eps

    dcf_base = _run_dcf(latest_profit, g1,       g2,       tg, dr)
    dcf_bull = _run_dcf(latest_profit, g1 * 1.3, g2 * 1.3, tg, max(dr - 1, tg + 0.5))
    dcf_bear = _run_dcf(latest_profit, g1 * 0.6, g2 * 0.6, tg, dr + 1)

    def to_ps(val_cr):
        return (val_cr / shares_cr) if (shares_cr and shares_cr > 0) else None

    iv_base  = to_ps(dcf_base)
    iv_bull  = to_ps(dcf_bull)
    iv_bear  = to_ps(dcf_bear)
    mos_price = iv_base * (1 - mos / 100) if iv_base else None

    # ── Verdict ──────────────────────────────────────────────────────────────
    st.markdown("---")

    # Fetch historical P/E data (used in both columns below)
    hist_pe_data = _get_historical_pe(ticker, conn)
    latest_eps_val = None
    if not eps_df.empty:
        eps_annual_tmp = eps_df.set_index("dt")["val"].resample("YE").mean().reset_index()
        if not eps_annual_tmp.empty:
            latest_eps_val = float(eps_annual_tmp["val"].iloc[-1])

    # ── Side-by-side: DCF | Historical P/E ───────────────────────────────────
    dcf_col, pe_col = st.columns(2)

    with dcf_col:
        st.markdown("##### 🔬 DCF Intrinsic Value")
        if iv_base and current_price:
            premium = (current_price - iv_base) / iv_base * 100
            if premium > 20:
                vcls, vicon = "verdict-over",  "🔴"
                vtxt = f"OVERVALUED by {premium:.1f}%"
                vsub = (f"Current ₹{current_price:,.1f} trades at a {premium:.1f}% premium "
                        f"to base DCF ₹{iv_base:,.1f}. Market pricing in higher growth than assumed.")
            elif premium < -20:
                vcls, vicon = "verdict-under", "🟢"
                vtxt = f"UNDERVALUED by {abs(premium):.1f}%"
                vsub = (f"Current ₹{current_price:,.1f} trades at a {abs(premium):.1f}% discount "
                        f"to base DCF ₹{iv_base:,.1f}. MoS price: ₹{mos_price:,.1f}.")
            else:
                vcls, vicon = "verdict-fair",  "🟡"
                vtxt = f"FAIRLY VALUED ({premium:+.1f}%)"
                vsub = (f"Current ₹{current_price:,.1f} is within ±20% of base DCF ₹{iv_base:,.1f}.")

            v1, v2 = st.columns(2)
            v1.metric("Base DCF / share", f"₹{iv_base:,.1f}" if iv_base else "—")
            v2.metric("Current Price",    f"₹{current_price:,.1f}", f"{premium:+.1f}% vs DCF", delta_color="inverse")
            v3, v4 = st.columns(2)
            v3.metric("Bull DCF / share", f"₹{iv_bull:,.1f}" if iv_bull else "—")
            v4.metric("Bear DCF / share", f"₹{iv_bear:,.1f}" if iv_bear else "—")
            v5, v6 = st.columns(2)
            v5.metric("MoS Price", f"₹{mos_price:,.1f}" if mos_price else "—", f"{mos}% below base")

            st.markdown(f"""
            <div class="{vcls}">
                <div class="verdict-title">{vicon} {vtxt}</div>
                <div class="verdict-sub">{vsub}</div>
            </div>
            """, unsafe_allow_html=True)
        elif not shares_cr:
            st.info("⚠️ Could not derive shares outstanding — showing total DCF value in ₹ Cr.")
            d1, d2, d3 = st.columns(3)
            d1.metric("Base DCF (₹ Cr)", f"₹{dcf_base:,.0f}")
            d2.metric("Bull DCF (₹ Cr)", f"₹{dcf_bull:,.0f}")
            d3.metric("Bear DCF (₹ Cr)", f"₹{dcf_bear:,.0f}")

    with pe_col:
        st.markdown("##### 📊 Historical P/E Fair Value")
        st.caption(
            "Method: Uses median historical P/E as the fair-value anchor — "
            "suitable when growth has been stable and is expected to remain so."
        )
        if hist_pe_data and latest_eps_val and current_price:
            med_pe   = hist_pe_data["weighted_median_pe"]   # time-decay weighted
            mean_pe  = hist_pe_data["weighted_mean_pe"]
            min_pe   = hist_pe_data["min_pe"]
            max_pe   = hist_pe_data["max_pe"]
            yrs      = hist_pe_data["years_used"]
            simple_median = hist_pe_data["median_pe"]
            curr_pe  = info.get("stock_p_e") or (current_price / latest_eps_val if latest_eps_val > 0 else None)

            # Fair value estimates
            fv_median = med_pe  * latest_eps_val
            fv_mean   = mean_pe * latest_eps_val
            fv_low    = min_pe  * latest_eps_val
            fv_high   = max_pe  * latest_eps_val

            pe_premium = (current_price - fv_median) / fv_median * 100 if fv_median else None

            # Verdict
            if pe_premium is not None:
                if pe_premium > 20:
                    pe_vcls, pe_icon = "verdict-over",  "🔴"
                    pe_vtxt = f"OVERVALUED by {pe_premium:.1f}%"
                    pe_vsub = (f"Current ₹{current_price:,.1f} is {pe_premium:.1f}% above "
                               f"decay-weighted median P/E fair value ₹{fv_median:,.1f} "
                               f"(weighted median {med_pe:.1f}x, simple median {simple_median:.1f}x, {yrs} yrs).")
                elif pe_premium < -20:
                    pe_vcls, pe_icon = "verdict-under", "🟢"
                    pe_vtxt = f"UNDERVALUED by {abs(pe_premium):.1f}%"
                    pe_vsub = (f"Current ₹{current_price:,.1f} is {abs(pe_premium):.1f}% below "
                               f"decay-weighted median P/E fair value ₹{fv_median:,.1f} "
                               f"(weighted median {med_pe:.1f}x, simple median {simple_median:.1f}x, {yrs} yrs).")
                else:
                    pe_vcls, pe_icon = "verdict-fair",  "🟡"
                    pe_vtxt = f"FAIRLY VALUED ({pe_premium:+.1f}%)"
                    pe_vsub = (f"Current ₹{current_price:,.1f} is within ±20% of "
                               f"decay-weighted median P/E fair value ₹{fv_median:,.1f} "
                               f"(weighted median {med_pe:.1f}x, simple median {simple_median:.1f}x, {yrs} yrs).")

                p1, p2 = st.columns(2)
                p1.metric("Wtd Median P/E Fair Value", f"₹{fv_median:,.1f}")
                p2.metric("Current Price", f"₹{current_price:,.1f}", f"{pe_premium:+.1f}% vs wtd median", delta_color="inverse")
                p3, p4 = st.columns(2)
                p3.metric("P/E High Fair Value",  f"₹{fv_high:,.1f}", f"at {max_pe:.1f}x P/E")
                p4.metric("P/E Low Fair Value",   f"₹{fv_low:,.1f}",  f"at {min_pe:.1f}x P/E")
                p5, p6 = st.columns(2)
                p5.metric("Wtd Median P/E", f"{med_pe:.1f}x", f"simple median {simple_median:.1f}x")
                p6.metric("Current P/E", f"{curr_pe:.1f}x" if curr_pe else "—")

                st.markdown(f"""
                <div class="{pe_vcls}">
                    <div class="verdict-title">{pe_icon} {pe_vtxt}</div>
                    <div class="verdict-sub">{pe_vsub}</div>
                </div>
                """, unsafe_allow_html=True)

                # Mini P/E history table
                if "pe_series" in hist_pe_data:
                    with st.expander("Historical P/E by Year (with decay weights)", expanded=False):
                        _pe_disp = hist_pe_data["pe_series"].copy()
                        _pe_disp["year"] = _pe_disp["year"].dt.year
                        _pe_disp["pe"]   = _pe_disp["pe"].round(1)
                        # Rename only the columns that actually exist
                        _col_rename = {"year": "Year", "pe": "P/E", "weight_pct": "Weight %"}
                        _pe_disp = _pe_disp.rename(columns=_col_rename)
                        st.dataframe(_pe_disp, use_container_width=True, hide_index=True)
                        st.caption(
                            f"Decay rate = {hist_pe_data.get('decay_rate', 0.3):.1f} — "
                            f"each year's weight grows exponentially toward the present. "
                            f"Recent years carry significantly more influence on the weighted median."
                        )
        else:
            st.info("⚠️ Insufficient price or EPS history to compute historical P/E fair value.")

    # ── Chart 1: Price vs DCF band ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📈 Price vs DCF Intrinsic Value Band")
    if not price_df.empty and iv_base:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[price_df["dt"].iloc[0], price_df["dt"].iloc[-1]], y=[iv_bull, iv_bull],
            mode="lines", name=f"Bull DCF ₹{iv_bull:,.0f}",
            line=dict(color="#16a34a", width=1.5, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=[price_df["dt"].iloc[0], price_df["dt"].iloc[-1]], y=[iv_bear, iv_bear],
            mode="lines", name=f"Bear DCF ₹{iv_bear:,.0f}",
            line=dict(color="#dc2626", width=1.5, dash="dot"),
            fill="tonexty", fillcolor="rgba(22,163,74,0.07)",
        ))
        fig.add_trace(go.Scatter(
            x=[price_df["dt"].iloc[0], price_df["dt"].iloc[-1]], y=[iv_base, iv_base],
            mode="lines", name=f"Base DCF ₹{iv_base:,.0f}",
            line=dict(color="#2563eb", width=2, dash="dash"),
        ))
        if mos_price:
            fig.add_trace(go.Scatter(
                x=[price_df["dt"].iloc[0], price_df["dt"].iloc[-1]], y=[mos_price, mos_price],
                mode="lines", name=f"MoS ₹{mos_price:,.0f}",
                line=dict(color="#d97706", width=1.5, dash="longdash"),
            ))
        fig.add_trace(go.Scatter(
            x=price_df["dt"], y=price_df["price"],
            mode="lines", name=f"{ticker} Price",
            line=dict(color="#1e293b", width=2),
        ))
        fig.update_layout(
            height=400, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.08),
            xaxis=dict(showgrid=False),
            yaxis=dict(title="Price (₹)", showgrid=True, gridcolor="#f0f0f0"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Shaded band = Bear–Bull DCF range. Dashed blue = Base DCF.")

    # ── Chart 2: Value composition ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🧱 Value Composition")
    if iv_base:
        pv1, pv2 = 0.0, 0.0
        proj = latest_profit
        for yr in range(1, 6):
            proj = proj * (1 + g1 / 100)
            pv1 += proj / (1 + dr / 100) ** yr
        for yr in range(6, 11):
            proj = proj * (1 + g2 / 100)
            pv2 += proj / (1 + dr / 100) ** yr
        tv    = proj * (1 + tg / 100) / (dr / 100 - tg / 100) if dr > tg else 0
        pv_tv = tv / (1 + dr / 100) ** 10
        total = pv1 + pv2 + pv_tv or 1
        pcts  = [pv1 / total * 100, pv2 / total * 100, pv_tv / total * 100]
        fig_wf = go.Figure(go.Bar(
            x=["Phase 1 (Yr 1–5)", "Phase 2 (Yr 6–10)", "Terminal Value"],
            y=[round(p, 1) for p in pcts],
            marker_color=["#2563eb", "#0891b2", "#6b7280"],
            text=[f"{p:.1f}%" for p in pcts], textposition="outside",
        ))
        fig_wf.update_layout(
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(title="% of Total DCF Value", showgrid=False, range=[0, 105]),
            xaxis=dict(showgrid=False),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_wf, use_container_width=True)
        st.caption("Terminal Value > 70% → DCF highly sensitive to terminal growth rate.")

    # ── Chart 3: Historical + projected profit ────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 Historical Net Profit + DCF Projection")
    if not np_annual.empty:
        proj_years  = list(range(np_annual["year"].iloc[-1].year + 1,
                                 np_annual["year"].iloc[-1].year + 11))
        proj_profit, p = [], latest_profit
        for i in range(10):
            p = p * (1 + (g1 if i < 5 else g2) / 100)
            proj_profit.append(p)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=[y.year for y in np_annual["year"]], y=np_annual["net_profit"],
            name="Historical", marker_color=COLORS[0],
        ))
        fig2.add_trace(go.Bar(
            x=proj_years, y=proj_profit, name="Projected (Base)",
            marker_color="rgba(37,99,235,0.3)", marker_line=dict(color=COLORS[0], width=1.5),
        ))
        fig2.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0), bargap=0.2,
            yaxis=dict(title="Net Profit (₹ Cr)", showgrid=True, gridcolor="#f0f0f0"),
            xaxis=dict(showgrid=False), legend=dict(orientation="h", y=1.08),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 4: Sensitivity table ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🎯 Sensitivity — Intrinsic Value per Share")
    if iv_base and shares_cr:
        # Use rounded, unique values to avoid duplicate index/column names
        g_vals = sorted(set(round(g, 1) for g in [g1 - 4, g1 - 2, g1, g1 + 2, g1 + 4]))
        d_vals = sorted(set(round(d, 1) for d in [dr - 2, dr - 1, dr, dr + 1, dr + 2]))

        def _cell_colour(v_str: str) -> tuple[str, str]:
            """Returns (background, text-color)."""
            if v_str in ("N/A", "—"):
                return "#f0f4ff", "#94a3b8"
            try:
                v = float(v_str.replace("₹", "").replace(",", ""))
                if current_price and v < current_price * 0.8:
                    return "#dbeafe", "#1e3a8a"   # deeper blue — below price
                elif current_price and v > current_price * 1.2:
                    return "#ffffff", "#1e40af"   # white — well above price
                return "#eff6ff", "#1d4ed8"       # light blue — fair zone
            except Exception:
                return "#f0f4ff", "#334155"

        # Render as HTML table — avoids Styler non-unique index/column restriction
        _th = (
            "padding:8px 14px;text-align:right;font-size:0.78rem;"
            "color:#1e3a8a;background:#1e40af;"
            "color:#ffffff;"
            "border-bottom:2px solid #1e3a8a;white-space:nowrap;font-weight:600"
        )
        _td_base = (
            "padding:7px 14px;text-align:right;font-size:0.83rem;"
            "font-family:monospace;border-bottom:1px solid #bfdbfe"
        )
        _sens_html = f"""
        <table style='width:100%;border-collapse:collapse;border-radius:6px;
                      overflow:hidden;border:1px solid #bfdbfe;background:#eff6ff'>
        <thead><tr>
          <th style='{_th};text-align:left'>Growth ↓ / DR →</th>
          {"".join(f"<th style='{_th}'>{d:.1f}%</th>" for d in d_vals)}
        </tr></thead><tbody>
        """
        for i, g in enumerate(g_vals):
            is_current = abs(g - g1) < 0.05
            row_bg     = "#dbeafe" if is_current else ("#ffffff" if i % 2 == 0 else "#eff6ff")
            g_fw       = "font-weight:700;color:#1e3a8a" if is_current else "font-weight:500;color:#1e40af"
            _sens_html += (
                f"<tr style='background:{row_bg}'>"
                f"<td style='{_td_base};text-align:left;{g_fw}'>"
                f"{'● ' if is_current else ''}{g:.1f}%</td>"
            )
            for d in d_vals:
                if d <= tg:
                    cell, bg, fg = "N/A", "#f0f4ff", "#94a3b8"
                else:
                    v    = _run_dcf(latest_profit, max(g, 0), max(g * 0.7, 0), tg, d)
                    ps   = v / shares_cr
                    cell = f"₹{ps:,.0f}"
                    bg, fg = _cell_colour(cell)
                fw = "font-weight:700" if is_current else ""
                _sens_html += f"<td style='{_td_base};background:{bg};color:{fg};{fw}'>{cell}</td>"
            _sens_html += "</tr>"
        _sens_html += "</tbody></table>"

        st.markdown(_sens_html, unsafe_allow_html=True)
        st.caption(
            f"🟢 Green = DCF > 120% of current price.  "
            f"🔴 Red = DCF < 80% of current price.  "
            f"Current price: ₹{current_price:,.1f}  |  "
            f"Bold row = current Phase 1 growth assumption."
        )

    # ── Key metrics ───────────────────────────────────────────────────────────
    if info:
        st.markdown("---")
        st.markdown("#### 📋 Key Fundamentals")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Current Price", f"₹{info.get('current_price', 0):,.1f}")
        m2.metric("P/E",           f"{info.get('stock_p_e', 0):.1f}x")
        m3.metric("Book Value",    f"₹{info.get('book_value', 0):,.1f}")
        m4.metric("Div Yield",     f"{info.get('dividend_yield', 0):.2f}%")
        m5.metric("ROE",           f"{info.get('roe', 0):.1f}%")
        m6.metric("ROCE",          f"{info.get('roce', 0):.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ══════════════════════════════════════════════════════════════════════════════

def render():
    st.set_page_config(layout="wide", page_title="DCF Analysis", page_icon="🔬")

    st.markdown("""
    <style>
    .block-container { max-width:100%!important; padding:1rem 1.5rem 2rem; }
    .verdict-over  { background:#fef2f2; border-left:4px solid #dc2626; padding:12px 16px; border-radius:4px; }
    .verdict-fair  { background:#fefce8; border-left:4px solid #ca8a04; padding:12px 16px; border-radius:4px; }
    .verdict-under { background:#f0fdf4; border-left:4px solid #16a34a; padding:12px 16px; border-radius:4px; }
    .verdict-title { font-size:1.1rem; font-weight:700; margin-bottom:4px; }
    .verdict-sub   { font-size:0.85rem; color:#475569; }
    div[data-testid="stButton"] button {
        text-align:left; white-space:pre-wrap; height:auto;
        padding:10px 12px; line-height:1.4;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    if "df_matrix" not in st.session_state or "conn" not in st.session_state:
        df_matrix, conn, _ = load_matrix()
        st.session_state["df_matrix"] = df_matrix
        st.session_state["conn"]      = conn
    else:
        df_matrix = st.session_state["df_matrix"]
        conn      = st.session_state["conn"]

    if df_matrix.empty:
        st.error("❌ No data available.")
        st.stop()

    ticker_list = sorted(df_matrix["Ticker"].tolist())

    # ── Resolve portfolio tickers ─────────────────────────────────────────────
    _portfolio_syms = []
    try:
        _all_holdings = st.session_state.get("all_holdings") or {}
        _excl = {s.upper() for s in (st.session_state.get("excluded_symbols") or [])}
        for _holdings in _all_holdings.values():
            for _h in _holdings:
                _sym = _h.get("tradingsymbol", "")
                if _sym and _sym.upper() not in _excl and _sym not in _portfolio_syms:
                    _portfolio_syms.append(_sym)
    except Exception:
        pass
    portfolio_tickers = [s for s in _portfolio_syms if s in ticker_list]

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("### 🔬 DCF — Intrinsic Value Analysis")

    # ── Portfolio scan (collapsible, open by default if holdings exist) ───────
    # Build P&L map {ticker: {qty, avg_cost, pnl, pnl_pct, today_gain, today_pct}}
    _pnl_map: dict = {}
    try:
        _all_h = st.session_state.get("all_holdings") or {}
        for _hlist in _all_h.values():
            for _h in _hlist:
                _sym = _h.get("tradingsymbol", "")
                if not _sym:
                    continue
                _qty      = _h.get("quantity", 0) or 0
                _avg_cost = _h.get("average_price", 0) or 0
                _ltp      = _h.get("last_price", 0) or 0
                _close    = _h.get("close_price", 0) or _ltp
                _pnl_raw  = _h.get("pnl", None)
                if _qty and _avg_cost and _ltp:
                    _invested  = _qty * _avg_cost
                    _cur_val   = _qty * _ltp
                    _pnl_val   = float(_pnl_raw) if _pnl_raw is not None else (_cur_val - _invested)
                    _pnl_pct   = (_pnl_val / _invested * 100) if _invested else 0.0
                    _t_gain    = _qty * (_ltp - _close)
                    _t_pct     = ((_ltp - _close) / _close * 100) if _close else 0.0
                    _pnl_map[_sym] = {
                        "qty":        _qty,
                        "avg_cost":   _avg_cost,
                        "pnl":        round(_pnl_val, 2),
                        "pnl_pct":    round(_pnl_pct, 2),
                        "today_gain": round(_t_gain, 2),
                        "today_pct":  round(_t_pct, 2),
                    }
    except Exception:
        pass

    with st.expander(
            f"📂 Portfolio DCF Scan  —  {len(portfolio_tickers)} holdings",
            expanded=bool(portfolio_tickers),
    ):
        _render_portfolio_scan(portfolio_tickers, conn, df_matrix, pnl_map=_pnl_map)

    # ── Stock selector ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔍 Detailed DCF Analysis")

    saved  = st.session_state.get("dcf_active_ticker") or st.session_state.get("selected_asset")
    _source = st.session_state.get("_dcf_source", "all")

    sel_col1, sel_col2, _ = st.columns([2, 2, 3])

    with sel_col1:
        port_opts     = portfolio_tickers if portfolio_tickers else ["— no portfolio —"]
        port_disabled = not bool(portfolio_tickers)
        port_default  = (portfolio_tickers.index(saved)
                         if saved in portfolio_tickers else 0)
        port_pick = st.selectbox(
            "📂 Portfolio",
            port_opts,
            index=port_default,
            disabled=port_disabled,
            key="dcf_ticker_portfolio",
        )

    with sel_col2:
        all_default = ticker_list.index(saved) if saved in ticker_list else 0
        all_pick = st.selectbox(
            "🔍 All Stocks",
            ticker_list,
            index=all_default,
            key="dcf_ticker_all",
        )

    # Detect which picker changed
    _prev_port = st.session_state.get("_dcf_prev_port")
    _prev_all  = st.session_state.get("_dcf_prev_all")

    if port_pick != _prev_port and not port_disabled and port_pick != "— no portfolio —":
        ticker  = port_pick
        _source = "portfolio"
    elif all_pick != _prev_all:
        ticker  = all_pick
        _source = "all"
    else:
        ticker = port_pick if (_source == "portfolio" and portfolio_tickers) else all_pick

    st.session_state["_dcf_source"]       = _source
    st.session_state["_dcf_prev_port"]    = port_pick
    st.session_state["_dcf_prev_all"]     = all_pick
    st.session_state["dcf_active_ticker"] = ticker

    # ── Detail ────────────────────────────────────────────────────────────────
    _render_dcf_detail(ticker, conn, df_matrix)