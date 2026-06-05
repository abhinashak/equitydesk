"""
portfolio.py
────────────
Full-featured portfolio view.

Features:
  - Portfolio movement across 1D / 7D / 1M / 3M / 6M / 1Y / 2Y / 3Y / 4Y / 5Y
  - Portfolio weight % alongside each ticker
  - Full-width Sector Distribution + Index comparison charts
  - Today's gain % per holding
  - Today's winner / loser highlighted rows
  - Sector orientation based on defined sectors
  - Historical data via DuckDB (one_ticker_common.py connection)
"""

import os
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ui.pages.one_ticker_common import get_duckdb_connection, initialize_views_or_mock, COLORS, COLORS_PALE
from plotly.subplots import make_subplots

# ── Import shared DuckDB bootstrap ──────────────────────────────────────────
try:
    _HAS_COMMON = True
except ImportError:
    _HAS_COMMON = False
    COLORS      = ["#2563eb", "#dc2626", "#16a34a", "#d97706"]
    COLORS_PALE = ["#93c5fd", "#fca5a5", "#86efac", "#fde68a"]


# ==============================================================================
# INDUSTRY DEFINITIONS  ← mapped to NSE industry values from tickers table
# ==============================================================================
DEFINED_INDUSTRY: dict[str, dict] = {
    # IT
    "IT - Software":                                  {"color": "#2563eb", "icon": "💻"},
    "IT - Services":                                  {"color": "#1d4ed8", "icon": "🖥️"},
    # Finance & Banking
    "Banks":                                          {"color": "#16a34a", "icon": "🏦"},
    "Finance":                                        {"color": "#15803d", "icon": "💰"},
    "Capital Markets":                                {"color": "#166534", "icon": "📈"},
    "Insurance":                                      {"color": "#14532d", "icon": "🛡️"},
    "Financial Technology (Fintech)":                 {"color": "#4ade80", "icon": "💳"},
    # Energy & Oil
    "Oil":                                            {"color": "#d97706", "icon": "🛢️"},
    "Gas":                                            {"color": "#b45309", "icon": "🔥"},
    "Petroleum Products":                             {"color": "#92400e", "icon": "⛽"},
    "Power":                                          {"color": "#f59e0b", "icon": "⚡"},
    "Consumable Fuels":                               {"color": "#78350f", "icon": "🪨"},
    # Automobiles
    "Automobiles":                                    {"color": "#7c3aed", "icon": "🚗"},
    "Auto Components":                                {"color": "#6d28d9", "icon": "🔧"},
    "Agricultural, Commercial & Construction Vehicles":{"color": "#5b21b6", "icon": "🚜"},
    # Consumer & FMCG
    "Consumer Durables":                              {"color": "#db2777", "icon": "📺"},
    "Diversified FMCG":                               {"color": "#be185d", "icon": "🛒"},
    "Food Products":                                  {"color": "#9d174d", "icon": "🍱"},
    "Beverages":                                      {"color": "#831843", "icon": "🥤"},
    "Personal Products":                              {"color": "#ec4899", "icon": "🧴"},
    "Household Products":                             {"color": "#f472b6", "icon": "🧹"},
    "Agricultural Food & other Products":             {"color": "#fda4af", "icon": "🌾"},
    # Pharma & Healthcare
    "Pharmaceuticals & Biotechnology":                {"color": "#0891b2", "icon": "💊"},
    "Healthcare Services":                            {"color": "#0e7490", "icon": "🏥"},
    "Healthcare Equipment & Supplies":                {"color": "#155e75", "icon": "🩺"},
    # Metals & Mining
    "Ferrous Metals":                                 {"color": "#92400e", "icon": "⚙️"},
    "Non - Ferrous Metals":                           {"color": "#78350f", "icon": "🔩"},
    "Diversified Metals":                             {"color": "#a16207", "icon": "🪙"},
    "Metals & Minerals Trading":                      {"color": "#854d0e", "icon": "⛏️"},
    "Minerals & Mining":                              {"color": "#713f12", "icon": "🪨"},
    # Infrastructure & Construction
    "Construction":                                   {"color": "#065f46", "icon": "🏗️"},
    "Cement & Cement Products":                       {"color": "#047857", "icon": "🧱"},
    "Realty":                                         {"color": "#059669", "icon": "🏢"},
    "Transport Infrastructure":                       {"color": "#10b981", "icon": "🛤️"},
    # Industrials
    "Industrial Manufacturing":                       {"color": "#0369a1", "icon": "🏭"},
    "Industrial Products":                            {"color": "#0284c7", "icon": "🔨"},
    "Electrical Equipment":                           {"color": "#0ea5e9", "icon": "🔌"},
    "Commercial Services & Supplies":                 {"color": "#38bdf8", "icon": "📦"},
    # Chemicals
    "Chemicals & Petrochemicals":                     {"color": "#4f46e5", "icon": "🧪"},
    "Fertilizers & Agrochemicals":                    {"color": "#4338ca", "icon": "🌱"},
    # Telecom
    "Telecom - Services":                             {"color": "#1e40af", "icon": "📡"},
    # Transport
    "Transport Services":                             {"color": "#1e3a8a", "icon": "🚢"},
    # Entertainment & Leisure
    "Entertainment":                                  {"color": "#c026d3", "icon": "🎬"},
    "Leisure Services":                               {"color": "#a21caf", "icon": "🎭"},
    # Textiles
    "Textiles & Apparels":                            {"color": "#7e22ce", "icon": "👗"},
    # Paper
    "Paper, Forest & Jute Products":                  {"color": "#6b7280", "icon": "📄"},
    # Aerospace
    "Aerospace & Defense":                            {"color": "#374151", "icon": "🚀"},
    # Diversified
    "Diversified":                                    {"color": "#4b5563", "icon": "🔀"},
    # Retailing
    "Retailing":                                      {"color": "#d97706", "icon": "🏪"},
    # Fallback
    "Other":                                          {"color": "#6b7280", "icon": "📦"},
}

def _industry_meta(industry: str) -> dict:
    """Return color+icon for an industry, falling back to Other."""
    return DEFINED_INDUSTRY.get(industry, DEFINED_INDUSTRY["Other"])

# Time-period config: (label, days_back, DuckDB_interval)
PERIODS = [
    ("3D",  3,    "3 day"),
    ("7D",  7,    "7 days"),
    ("30D",  30,   "1 month"),
    ("45D",  45,   "45 Days"),
    ("2M",  60,   "2 month"),
    ("3M",  90,   "3 months"),
    ("6M",  180,  "6 months"),
    ("1Y",  365,  "1 year"),
    ("2Y",  730,  "2 years"),
    ("3Y",  1095, "3 years"),
    ("4Y",  1460, "4 years"),
    ("5Y",  1825, "5 years"),
]

# Benchmark index ticker used for comparison (must exist in price_history)
INDEX_TICKER = "^NSEI"


# ==============================================================================
# HISTORICAL DATA HELPERS
# ==============================================================================

def get_price_history(nse_symbols: list[str], days: int) -> pd.DataFrame:
    """
    Pull close prices from DuckDB `ticker_prices` using nse_symbol column.
    Returns DataFrame with columns: date, ticker (=nse_symbol), close.
    Returns empty DataFrame if DuckDB unavailable or no data found.
    """
    if not _HAS_COMMON or not nse_symbols:
        return pd.DataFrame()

    try:
        conn = get_duckdb_connection()
        initialize_views_or_mock(conn)

        sym_list = ", ".join(f"'{s}'" for s in nse_symbols)
        query = f"""
            SELECT Date       AS date,
                   nse_symbol AS ticker,
                   Close      AS close
            FROM ticker_prices
            WHERE nse_symbol IN ({sym_list})
              AND Date >= CURRENT_DATE - INTERVAL '{days} days'
            ORDER BY nse_symbol, Date
        """
        df = conn.execute(query).df()
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame()


def compute_portfolio_value_series(holdings_qty: dict,
                                   price_df: pd.DataFrame) -> pd.Series:
    """
    Given {ticker: qty} and a long-form price_df (date/ticker/close),
    returns a daily portfolio value series indexed by date.
    """
    if price_df.empty:
        return pd.Series(dtype=float)
    pivot = price_df.pivot_table(index="date", columns="ticker",
                                 values="close", aggfunc="last")
    # ffill fills gaps within a series, but leading NaNs (tickers whose
    # history starts after the window start) need bfill — or better, fill
    # each column's leading NaNs with its first known price so a newly-listed
    # stock doesn't drag the whole portfolio value to NaN.
    pivot = pivot.ffill()                     # forward-fill mid-series gaps
    for col in pivot.columns:
        first_valid = pivot[col].first_valid_index()
        if first_valid is not None and pivot[col].isna().any():
            pivot[col] = pivot[col].fillna(pivot[col].loc[first_valid])

    port_val = pd.Series(0.0, index=pivot.index)
    for tkr, qty in holdings_qty.items():
        if tkr in pivot.columns:
            port_val += pivot[tkr].fillna(0) * qty
    return port_val


def compute_period_return(port_series: pd.Series) -> float:
    """% return over the series."""
    clean = port_series.dropna()
    if len(clean) < 2:
        return 0.0
    start = clean.iloc[0]
    end   = clean.iloc[-1]
    return ((end - start) / start * 100) if start else 0.0


# ==============================================================================
# MAIN RENDER
# ==============================================================================

def render():
    st.set_page_config(page_title="Portfolio", layout="wide", page_icon="📊")

    # ── CSS tweaks ────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .winner-row  { background-color: rgba(22,163,74,0.10) !important; }
    .loser-row   { background-color: rgba(220,38,38,0.10) !important; }
    .sector-pill {
        display: inline-block; padding: 2px 10px;
        border-radius: 9999px; font-size: 0.72rem; font-weight: 600;
    }
    .period-chip {
        display: inline-block; padding: 3px 10px;
        border-radius: 6px; font-size: 0.78rem; font-weight: 600;
        margin: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Guard ─────────────────────────────────────────────────────────────────
    if "all_holdings" not in st.session_state or not st.session_state.all_holdings:
        st.warning("Holdings not loaded. Please go to Setup and click 'Fetch Holdings'.")
        st.stop()

    all_holdings  = st.session_state.all_holdings
    print(all_holdings)
    # Normalise once to uppercase so all comparisons are case-insensitive
    excl_set = {s.upper() for s in (getattr(st.session_state, "excluded_symbols", None) or set())}
    account_names = list(all_holdings.keys())

    # ── Build ticker_data map ─────────────────────────────────────────────────
    ticker_data = {}
    for acc_name, holdings in all_holdings.items():
        for h in holdings:
            sym = h["tradingsymbol"]
            if sym.upper() in excl_set:
                continue
            if sym not in ticker_data:
                ticker_data[sym] = {
                    "last_price":  h.get("last_price",  0),
                    "close_price": h.get("close_price", 0),
                    "day_change_pct": h.get("day_change_percentage", 0),
                    "avg_cost": h.get("average_price", 0),
                    **{an: 0 for an in account_names},
                }
            ticker_data[sym][acc_name] += h.get("quantity", 0)

    # ── Load ticker metadata from DuckDB (industry, market_cap, p_e, dividend_yield) ──
    # Kite tradingsymbol == nse_symbol in tickers table (e.g. "RELIANCE")
    # Also try matching via nse_symbol derived from Yahoo Symbol (split on '.')
    ticker_meta = {}
    if _HAS_COMMON:
        try:
            conn = get_duckdb_connection()
            initialize_views_or_mock(conn)
            sql_stmt = f"""
                SELECT nse_symbol AS matched_symbol,
                    industry, market_cap, market_cap_amt, p_e, dividend_yield
                FROM tickers
                WHERE nse_symbol IN ({", ".join(f"'{s.upper()}'" for s in ticker_data.keys())})
            """
            print(sql_stmt)
            meta_df = conn.execute(sql_stmt).df()
            print ( f"Total found : {len(meta_df)}")
            for _, r in meta_df.iterrows():
                key = str(r["matched_symbol"]).strip().upper()
                ticker_meta[key] = {
                    "industry":      r.get("industry", "Other") or "Other",
                    "market_cap":    r.get("market_cap", "Unknown") or "Unknown",
                    "market_cap_amt":r.get("market_cap_amt", 0) or 0,
                    "p_e":           r.get("p_e", None),
                    "dividend_yield":r.get("dividend_yield", 0) or 0,
                }
        except Exception as e:
            st.caption(f"⚠️ Metadata load error: {e}")

    with st.expander("🐛 Metadata Debug", expanded=False):
        st.write("**Kite symbols (tradingsymbol):**", list(ticker_data.keys()))
        st.write("**Matched from tickers table:**", list(ticker_meta.keys()))
        unmatched = [s for s in ticker_data if s.upper() not in ticker_meta]
        if unmatched:
            st.warning(f"No metadata match for: {unmatched}")
        else:
            st.success("All symbols matched ✓")

    # ── Build base DataFrame ──────────────────────────────────────────────────
    rows = []
    for sym, data in ticker_data.items():
        row = {"Symbol": sym}
        total_qty = sum(data.get(an, 0) for an in account_names)
        for an in account_names:
            row[an] = data.get(an, 0)

        lp = data["last_price"]
        cp = data["close_price"]
        avg_cost = data["avg_cost"]
        invested  = round(total_qty * avg_cost, 2)
        meta = ticker_meta.get(sym.upper(), {})

        row["Total Qty"]     = total_qty
        row["Last Price"]    = lp
        row["Close Price"]   = cp
        row["Curr Value"]    = round(total_qty * lp, 2)
        row["P&L%"]          = round((lp - avg_cost) / avg_cost * 100, 2) if avg_cost else 0.0
        row["Today's Gain"] = round(total_qty * (lp - cp), 2)
        row["Day Chg%"]      = data["day_change_pct"]
        row["Industry"]      = meta.get("industry", "Other")
        row["Market Cap"]    = meta.get("market_cap", "Unknown")
        row["Market Cap ₹"]  = meta.get("market_cap_amt", 0)
        row["P/E"]           = meta.get("p_e", None)
        row["Div Yield%"]    = meta.get("dividend_yield", 0)
        rows.append(row)

    df = (pd.DataFrame(rows)
          .sort_values("Curr Value", ascending=False)
          .reset_index(drop=True))

    # Re-apply exclusions on df (safety net — excl_set is already uppercase)
    if excl_set:
        df = df[~df["Symbol"].str.upper().isin(excl_set)].reset_index(drop=True)



    total_value  = df["Curr Value"].sum()
    todays_gain  = df["Today's Gain"].sum()
    todays_gain_pct = (todays_gain / (total_value - todays_gain) * 100
                       if (total_value - todays_gain) else 0)

    df["Portfolio Wt%"] = (df["Curr Value"] / total_value * 100).round(2) if total_value else 0.0
    df["Today Gain%"]   = df["Day Chg%"].round(2)

    # Winner / loser
    winner_sym = df.loc[df["Day Chg%"].idxmax(), "Symbol"] if not df.empty else ""
    loser_sym  = df.loc[df["Day Chg%"].idxmin(), "Symbol"] if not df.empty else ""

    # ── Holdings qty map for history ──────────────────────────────────────────
    holdings_qty   = dict(zip(df["Symbol"], df["Total Qty"]))
    current_prices = dict(zip(df["Symbol"], df["Last Price"]))
    tickers_all    = list(holdings_qty.keys())



    # ── TOP METRICS ───────────────────────────────────────────────────────────
    st.title("📊 Portfolio Dashboard")

    # ── Pinned ticker config ──────────────────────────────────────────────────
    PINNED_FILE = os.path.join(os.path.dirname(__file__),
                               "..", "..", "outputs", "pinned_ticker.txt")
    PINNED_FILE = os.path.normpath(PINNED_FILE)

    def _load_pinned() -> str:
        try:
            with open(PINNED_FILE) as f:
                return f.read().strip().upper()
        except FileNotFoundError:
            return ""

    def _save_pinned(sym: str):
        os.makedirs(os.path.dirname(PINNED_FILE), exist_ok=True)
        with open(PINNED_FILE, "w") as f:
            f.write(sym.strip().upper())

    if "pinned_ticker" not in st.session_state:
        st.session_state.pinned_ticker = _load_pinned()

    with st.expander("📌 Pinned Ticker", expanded=False):
        pin_col1, pin_col2 = st.columns([3, 1])
        new_pin = pin_col1.selectbox(
            "Choose a stock to always show in the top bar",
            options=[""] + sorted(tickers_all),
            index=([""] + sorted(tickers_all)).index(st.session_state.pinned_ticker)
            if st.session_state.pinned_ticker in tickers_all else 0,
            label_visibility="collapsed",
        )
        if pin_col2.button("Save", use_container_width=True):
            st.session_state.pinned_ticker = new_pin
            _save_pinned(new_pin)
            st.success(f"Pinned: {new_pin or '(none)'}")

    pinned_sym = st.session_state.pinned_ticker

    # ── Metric row ────────────────────────────────────────────────────────────
    cols = st.columns(6 if pinned_sym and pinned_sym in df["Symbol"].values else 5)
    cols[0].metric("Total Value",   f"₹{total_value:,.0f}")
    cols[1].metric("Today's Gain",
                   f"₹{todays_gain:,.0f}",
                   f"{todays_gain_pct:+.2f}%",
                   delta_color="normal")
    cols[2].metric("Holdings", len(df))
    cols[3].metric("🏆 Today's Winner",
                   winner_sym,
                   f"{df.loc[df['Symbol']==winner_sym,'Day Chg%'].values[0]:+.2f}%" if winner_sym else "",
                   delta_color="normal")
    cols[4].metric("📉 Today's Loser",
                   loser_sym,
                   f"{df.loc[df['Symbol']==loser_sym,'Day Chg%'].values[0]:+.2f}%" if loser_sym else "",
                   delta_color="normal")
    if pinned_sym and pinned_sym in df["Symbol"].values:
        pin_row = df[df["Symbol"] == pinned_sym].iloc[0]
        cols[5].metric(f"📌 {pinned_sym}",
                       f"₹{pin_row['Last Price']:,.2f}",
                       f"{pin_row['Day Chg%']:+.2f}%",
                       delta_color="normal")

    st.divider()

    # ── PORTFOLIO MOVEMENT (period selector) ──────────────────────────────────
    st.subheader("Portfolio Movement")
    period_labels = [p[0] for p in PERIODS]
    sel_period    = st.radio("Period", period_labels, index=2,
                             horizontal=True, label_visibility="collapsed")

    days = next(p[1] for p in PERIODS if p[0] == sel_period)

    with st.spinner("Loading price history…"):
        # Fetch all tickers + index together in one call
        ph_df = get_price_history(tickers_all + [INDEX_TICKER], days)

        # Fix 3: Drop newly listed tickers that have insufficient history
        # (fewer than 5 trading days of data in the selected window)
        if not ph_df.empty:
            ticker_counts    = ph_df.groupby("ticker")["date"].count()
            # A ticker is "newly listed" only if it has data for less than 25% of
            # the requested window — this avoids dropping real tickers on short
            min_days         = max(1, days // 4)
            new_listings     = set(ticker_counts[ticker_counts < min_days].index) - {INDEX_TICKER}
            if new_listings:
                ph_df = ph_df[~ph_df["ticker"].isin(new_listings)]
                # Also remove them from the qty map used for value computation
                holdings_qty_hist = {k: v for k, v in holdings_qty.items()
                                     if k not in new_listings}
            else:
                holdings_qty_hist = holdings_qty
        else:
            holdings_qty_hist = holdings_qty

        # Deduplicate & sort before any pivot/reindex operations
        if not ph_df.empty:
            ph_df = (ph_df
                     .sort_values(["ticker", "date"])
                     .drop_duplicates(subset=["date", "ticker"], keep="last")
                     .reset_index(drop=True))

        # Split index series out BEFORE passing to portfolio value computation
        if not ph_df.empty and INDEX_TICKER in ph_df["ticker"].values:
            idx_raw   = ph_df[ph_df["ticker"] == INDEX_TICKER].set_index("date")["close"]
            idx_pivot = idx_raw[~idx_raw.index.duplicated(keep="last")].sort_index()
        else:
            idx_pivot = pd.Series(dtype=float)

        # ── SMA50 for ^NSEI ───────────────────────────────────────────────────
        # Always fetch at least 200 days so SMA50 is well-seeded, regardless of
        # the period the user chose for the portfolio movement chart.
        _sma_days = max(days, 200)
        _sma_ph   = get_price_history([INDEX_TICKER], _sma_days)
        if not _sma_ph.empty:
            _idx_full = (_sma_ph[_sma_ph["ticker"] == INDEX_TICKER]
                         .set_index("date")["close"]
                         .pipe(lambda s: s[~s.index.duplicated(keep="last")])
                         .sort_index())
            _sma50_full = _idx_full.rolling(window=50, min_periods=50).mean()
            # Slice to the same date window as the chart
            _sma50_chart = _sma50_full[_sma50_full.index >= idx_pivot.index[0]] if not idx_pivot.empty else _sma50_full
            _nsei_last   = _idx_full.iloc[-1]  if not _idx_full.empty  else None
            _sma50_last  = _sma50_full.iloc[-1] if not _sma50_full.empty else None
        else:
            _sma50_chart = pd.Series(dtype=float)
            _nsei_last   = None
            _sma50_last  = None

        # Only pass holding tickers to portfolio value (exclude index)
        ph_holdings = ph_df[ph_df["ticker"] != INDEX_TICKER] if not ph_df.empty else ph_df
        port_ser = compute_portfolio_value_series(holdings_qty_hist, ph_holdings) if not ph_holdings.empty else pd.Series(dtype=float)

        # Deduplicate port_ser index as a final safety net
        if not port_ser.empty:
            port_ser = port_ser[~port_ser.index.duplicated(keep="last")].sort_index()

    # ── EXIT PANIC BANNER ─────────────────────────────────────────────────────
    if _nsei_last is not None and _sma50_last is not None:
        _below_sma50 = _nsei_last < _sma50_last
        _gap_pct     = (_nsei_last - _sma50_last) / _sma50_last * 100
        if _below_sma50:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
                    border: 2px solid #ef4444;
                    border-radius: 12px;
                    padding: 18px 24px;
                    margin-bottom: 16px;
                    display: flex;
                    align-items: center;
                    gap: 16px;
                ">
                    <span style="font-size: 2.2rem;">🚨</span>
                    <div>
                        <div style="color: #fca5a5; font-size: 0.78rem; font-weight: 700;
                                    letter-spacing: 0.12em; text-transform: uppercase;">
                            Market Risk Alert
                        </div>
                        <div style="color: #fff; font-size: 1.25rem; font-weight: 800; margin: 2px 0;">
                            EXIT PANIC — ^NSEI is below its SMA50
                        </div>
                        <div style="color: #fca5a5; font-size: 0.9rem;">
                            NSEI: <strong style="color:#fff">{_nsei_last:,.0f}</strong>
                            &nbsp;·&nbsp; SMA50: <strong style="color:#fff">{_sma50_last:,.0f}</strong>
                            &nbsp;·&nbsp; Gap: <strong style="color:#fca5a5">{_gap_pct:+.2f}%</strong>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #14532d 0%, #166534 100%);
                    border: 2px solid #22c55e;
                    border-radius: 12px;
                    padding: 14px 24px;
                    margin-bottom: 16px;
                    display: flex;
                    align-items: center;
                    gap: 16px;
                ">
                    <span style="font-size: 1.8rem;">✅</span>
                    <div>
                        <div style="color: #86efac; font-size: 0.78rem; font-weight: 700;
                                    letter-spacing: 0.12em; text-transform: uppercase;">
                            Market Trend
                        </div>
                        <div style="color: #fff; font-size: 1.1rem; font-weight: 700;">
                            ^NSEI is above its SMA50 — trend intact
                        </div>
                        <div style="color: #86efac; font-size: 0.85rem;">
                            NSEI: <strong style="color:#fff">{_nsei_last:,.0f}</strong>
                            &nbsp;·&nbsp; SMA50: <strong style="color:#fff">{_sma50_last:,.0f}</strong>
                            &nbsp;·&nbsp; Gap: <strong style="color:#86efac">{_gap_pct:+.2f}%</strong>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if not port_ser.empty:
        port_ret = compute_period_return(port_ser)
        idx_ret  = compute_period_return(idx_pivot) if not idx_pivot.empty else 0.0
        alpha    = port_ret - idx_ret

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric(f"Portfolio ({sel_period})", f"{port_ret:+.2f}%")
        mc2.metric(f"NSEI ({sel_period})",  f"{idx_ret:+.2f}%")
        mc3.metric("Alpha", f"{alpha:+.2f}%",
                   delta_color="normal" if alpha >= 0 else "inverse")
        if _sma50_last is not None:
            mc4.metric("^NSEI SMA50",
                       f"{_sma50_last:,.0f}",
                       f"{(_nsei_last - _sma50_last) / _sma50_last * 100:+.2f}% vs price",
                       delta_color="normal" if _nsei_last >= _sma50_last else "inverse")

        # Normalised line chart
        fig_trend = go.Figure()
        port_norm = port_ser / port_ser.iloc[0] * 100 if port_ser.iloc[0] else port_ser
        fig_trend.add_trace(go.Scatter(
            x=port_norm.index, y=port_norm.values,
            name="Portfolio", line=dict(color=COLORS[0], width=2),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.07)"
        ))
        if not idx_pivot.empty:
            idx_aligned = idx_pivot.reindex(port_ser.index, method="ffill")
            idx_norm    = idx_aligned / idx_aligned.iloc[0] * 100 if idx_aligned.iloc[0] else idx_aligned
            fig_trend.add_trace(go.Scatter(
                x=idx_norm.index, y=idx_norm.values,
                name=INDEX_TICKER,
                line=dict(color=COLORS[1], width=1.5, dash="dash")
            ))
        # SMA50 line — normalise to the same base as idx_norm
        if not _sma50_chart.empty and not idx_pivot.empty:
            _sma50_aligned = _sma50_chart.reindex(port_ser.index, method="ffill").dropna()
            _idx_base      = idx_aligned.iloc[0] if not idx_aligned.empty and idx_aligned.iloc[0] else None
            if _idx_base:
                _sma50_norm = _sma50_aligned / _idx_base * 100
                fig_trend.add_trace(go.Scatter(
                    x=_sma50_norm.index, y=_sma50_norm.values,
                    name="NSEI SMA50",
                    line=dict(color="#f59e0b", width=1.5, dash="dot"),
                    opacity=0.85,
                ))
        fig_trend.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.05),
            xaxis=dict(showgrid=False),
            yaxis=dict(title="Indexed (base=100)", showgrid=True, gridcolor="#f0f0f0"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No price history data available for selected period.")

    st.divider()

    # ── PORTFOLIO TABLE (after movement) ─────────────────────────────────────
    st.subheader("Portfolio Holdings")

    _display_cols_quick = (
            ["Symbol", "Industry", "Market Cap", "Portfolio Wt%", "Total Qty"]
            + account_names
            #+ ["Last Price", "Curr Value", "Today's Gain", "Today Gain%", "Day Chg%"]
            + ["Last Price", "Curr Value", "Today's Gain", "Day Chg%", "P&L%"]
    )
    _display_cols_quick = [c for c in _display_cols_quick if c in df.columns]

    def _row_style_quick(row):
        if row["Symbol"] == winner_sym:
            return ["background-color: rgba(22,163,74,0.12)"] * len(row)
        if row["Symbol"] == loser_sym:
            return ["background-color: rgba(220,38,38,0.10)"] * len(row)
        return [""] * len(row)

    _col_cfg_quick = {
        "Last Price":    st.column_config.NumberColumn(format="₹%.2f"),
        "Curr Value":    st.column_config.NumberColumn(format="₹%.0f"),
        "Today's Gain":  st.column_config.NumberColumn(format="₹%.0f"),
        "Today Gain%":   st.column_config.NumberColumn(format="%.2f%%"),
        "Day Chg%":      st.column_config.NumberColumn(format="%.2f%%"),
        "P&L%":      st.column_config.NumberColumn(format="%.2f%%"),
        "Portfolio Wt%": st.column_config.NumberColumn(format="%.2f%%"),
    }
    st.dataframe(
        df[_display_cols_quick].style.apply(_row_style_quick, axis=1),
        column_config=_col_cfg_quick,
        use_container_width=True, hide_index=True,
    )

    st.divider()

    # ── INDUSTRY ORIENTATION TABLE ────────────────────────────────────────────
    st.subheader("Industry Orientation")

    # Pre-fetch price history for the fixed period columns (7D, 1M, 3M, 6M, 1Y)
    _ORIENT_PERIODS = [("7D", 7), ("1M", 30), ("3M", 90), ("6M", 180), ("1Y", 365)]

    def _industry_eq_returns(ph: pd.DataFrame) -> dict[str, float]:
        """Equal-weight return per industry from a long-form price DataFrame."""
        result = {}
        if ph.empty:
            return result
        for industry in df["Industry"].unique():
            ind_tickers = df[df["Industry"] == industry]["Symbol"].tolist()
            series_list = []
            for sym in ind_tickers:
                s = ph[ph["ticker"] == sym].set_index("date")["close"]
                s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
                if len(s) >= 2:
                    series_list.append(s / s.iloc[0] * 100)
            if series_list:
                eq_ser = pd.concat(series_list, axis=1).mean(axis=1)
                result[industry] = round(compute_period_return(eq_ser), 2)
        return result

    with st.spinner("Loading industry period returns…"):
        _orient_returns: dict[str, dict[str, float]] = {}
        for _lbl, _d in _ORIENT_PERIODS:
            _ph = get_price_history(tickers_all, _d)
            if not _ph.empty:
                _ph = (_ph.sort_values(["ticker", "date"])
                       .drop_duplicates(subset=["date", "ticker"], keep="last")
                       .reset_index(drop=True))
            _orient_returns[_lbl] = _industry_eq_returns(_ph)

    ind_orient = []
    for industry in df["Industry"].unique():
        idf = df[df["Industry"] == industry]
        _meta = _industry_meta(industry)
        row = {
            "Industry":       f"{_meta['icon']} {industry}",
            "Holdings":       len(idf),
            "Value (₹)":      idf["Curr Value"].sum(),
            "Wt%":            (idf["Curr Value"].sum() / total_value * 100).round(1),
            "Today's Gain ₹": idf["Today's Gain"].sum().round(0),
            "Avg Day Chg%":   idf["Day Chg%"].mean().round(2),
        }
        for _lbl, _ in _ORIENT_PERIODS:
            row[_lbl] = _orient_returns[_lbl].get(industry, None)
        row["Avg P/E"]        = round(idf["P/E"].dropna().mean(), 1) if idf["P/E"].notna().any() else None
        row["Avg Div Yield%"] = round(idf["Div Yield%"].mean(), 2)
        row["Tickers"]        = ", ".join(idf["Symbol"].tolist())
        ind_orient.append(row)

    orient_df = (pd.DataFrame(ind_orient)
                 .sort_values("Value (₹)", ascending=False)
                 .reset_index(drop=True))

    _period_col_cfg = {
        lbl: st.column_config.NumberColumn(label=lbl, format="%.1f%%")
        for lbl, _ in _ORIENT_PERIODS
    }
    st.dataframe(
        orient_df,
        column_config={
            "Value (₹)":       st.column_config.NumberColumn(format="₹%.0f"),
            "Wt%":             st.column_config.NumberColumn(format="%.1f%%"),
            "Today's Gain ₹": st.column_config.NumberColumn(format="₹%.0f"),
            "Avg Day Chg%":    st.column_config.NumberColumn(format="%.2f%%"),
            **_period_col_cfg,
            "Avg P/E":         st.column_config.NumberColumn(format="%.1f"),
            "Avg Div Yield%":  st.column_config.NumberColumn(format="%.2f%%"),
        },
        use_container_width=True, hide_index=True
    )

    # ── Industry equal-weight return chart ────────────────────────────────────
    st.markdown(f"**Industry Returns ({sel_period}) — Equal Weight**")

    # ph_holdings already fetched above (holdings only, no index); reuse it
    if not ph_holdings.empty:
        # Build equal-weight series per industry: average normalised price of all
        # tickers in that industry, then compute % return over the period
        ind_ret_rows = []
        for industry in df["Industry"].unique():
            ind_tickers = df[df["Industry"] == industry]["Symbol"].tolist()
            series_list = []
            for sym in ind_tickers:
                s = ph_holdings[ph_holdings["ticker"] == sym].set_index("date")["close"]
                s = s[~s.index.duplicated(keep="last")].sort_index()
                if len(s) >= 2:
                    series_list.append(s / s.iloc[0] * 100)   # normalise to 100
            if series_list:
                eq_ser = pd.concat(series_list, axis=1).mean(axis=1)
                ret    = compute_period_return(eq_ser)
                _meta  = _industry_meta(industry)
                ind_ret_rows.append({
                    "Industry": f"{_meta['icon']} {industry}",
                    "Return%":  round(ret, 2),
                    "color":    _meta["color"],
                })

        if ind_ret_rows:
            ind_ret_df = (pd.DataFrame(ind_ret_rows)
                          .sort_values("Return%", ascending=True)
                          .reset_index(drop=True))
            fig_ind_ret = go.Figure(go.Bar(
                x=ind_ret_df["Return%"],
                y=ind_ret_df["Industry"],
                orientation="h",
                marker_color=[
                    c if v >= 0 else "#ef4444"
                    for v, c in zip(ind_ret_df["Return%"], ind_ret_df["color"])
                ],
                text=[f"{v:+.1f}%" for v in ind_ret_df["Return%"]],
                textposition="outside",
                hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>",
            ))
            fig_ind_ret.update_layout(
                height=max(260, len(ind_ret_df) * 34),
                margin=dict(l=0, r=60, t=10, b=0),
                xaxis=dict(title=f"Return % ({sel_period})", zeroline=True,
                           zerolinecolor="#d1d5db", showgrid=False),
                yaxis=dict(autorange="reversed"),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_ind_ret, use_container_width=True)
    else:
        st.caption("_No price history available for selected period._")

    # ── INDUSTRY DISTRIBUTION + MARKET CAP + P/E + DIVIDEND ──────────────────
    st.divider()
    st.subheader("Portfolio Analytics")

    # ── Row 1: Industry pie  +  Market Cap pie ────────────────────────────────
    st.markdown("**Industry Distribution**")
    ind_df = df.groupby("Industry")["Curr Value"].sum().reset_index()
    ind_df["Wt%"] = (ind_df["Curr Value"] / total_value * 100).round(1)
    ind_df = ind_df.sort_values("Curr Value", ascending=False)
    ind_colors = [_industry_meta(ind)["color"] for ind in ind_df["Industry"]]
    fig_ind = go.Figure(go.Pie(
        labels=ind_df["Industry"],
        values=ind_df["Curr Value"],
        marker_colors=ind_colors,
        hole=0.45,
        textinfo="label+percent",
        hovertemplate="%{label}<br>₹%{value:,.0f}<br>%{percent}<extra></extra>"
    ))
    fig_ind.update_layout(
        height=460, margin=dict(l=0, r=0, t=40, b=40),
        showlegend=True, legend=dict(orientation="v", x=1.0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_ind, use_container_width=True)

    st.divider()
    st.markdown("**Market Cap Distribution**")
    mcap_df = df.groupby("Market Cap")["Curr Value"].sum().reset_index()
    mcap_df["Wt%"] = (mcap_df["Curr Value"] / total_value * 100).round(1)
    mcap_colors = {"Large-cap": "#2563eb", "Mid-cap": "#d97706",
                   "Small-cap": "#16a34a", "Unknown": "#6b7280"}
    fig_mcap = go.Figure(go.Pie(
        labels=mcap_df["Market Cap"],
        values=mcap_df["Curr Value"],
        marker_colors=[mcap_colors.get(m, "#6b7280") for m in mcap_df["Market Cap"]],
        hole=0.45,
        textinfo="label+percent",
        hovertemplate="%{label}<br>₹%{value:,.0f}<br>%{percent}<extra></extra>"
    ))
    fig_mcap.update_layout(
        height=460, margin=dict(l=0, r=40, t=40, b=40),
        showlegend=True, legend=dict(orientation="v", x=1.0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_mcap, use_container_width=True)

    # ── Row 2: Portfolio P/E metric  +  Dividend Yield chart ─────────────────
    st.divider()
    st.markdown("**Portfolio P/E**")
    pe_df = df[df["P/E"].notna() & (df["P/E"] > 0)].copy()
    if not pe_df.empty:
        port_pe = round((pe_df["Portfolio Wt%"] * pe_df["P/E"]).sum() / pe_df["Portfolio Wt%"].sum(), 1)
        st.metric("Weighted Avg P/E", port_pe)
    else:
        st.caption("_No P/E data available._")

    st.divider()
    st.markdown("**Dividend Yield — High Yielders**")
    div_df = df[df["Div Yield%"] > 0].copy().sort_values("Div Yield%", ascending=False)
    if not div_df.empty:
        port_div = round((df["Portfolio Wt%"] * df["Div Yield%"]).sum() / 100, 2)
        st.metric("Portfolio Weighted Div Yield", f"{port_div:.2f}%")
        fig_div = go.Figure(go.Bar(
            x=div_df["Symbol"],
            y=div_df["Div Yield%"],
            marker_color=[
                "#16a34a" if v >= 2 else "#d97706" if v >= 1 else "#6b7280"
                for v in div_df["Div Yield%"]
            ],
            text=[f"{v:.2f}%" for v in div_df["Div Yield%"]],
            textposition="outside",
            hovertemplate="%{x}<br>Yield: %{y:.2f}%<extra></extra>"
        ))
        fig_div.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(tickangle=-45),
            yaxis=dict(title="Div Yield %", showgrid=False),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_div, use_container_width=True)
    else:
        st.caption("_No dividend yield data available._")

    # ── Per-ticker movement strip ─────────────────────────────────────────────
    st.divider()
    st.subheader("Per-Ticker Period Returns")

    ph_strip = get_price_history(tickers_all, days)
    strip_rows = []
    for nse_sym in tickers_all:
        if not ph_strip.empty and "ticker" in ph_strip.columns:
            t_raw  = ph_strip[ph_strip["ticker"] == nse_sym].set_index("date")["close"]
            t_ser  = t_raw[~t_raw.index.duplicated(keep="last")].sort_index()
        else:
            t_ser = pd.Series(dtype=float)
        ret   = compute_period_return(t_ser) if len(t_ser) >= 2 else 0.0
        strip_rows.append({
            "Symbol":                 nse_sym,
            f"Return ({sel_period})": round(ret, 2),
            "Curr Value":             holdings_qty[nse_sym] * current_prices.get(nse_sym, 0),
        })

    strip_df = (pd.DataFrame(strip_rows)
                .sort_values(f"Return ({sel_period})", ascending=False)
                .reset_index(drop=True))

    fig_bar = go.Figure(go.Bar(
        x=strip_df["Symbol"],
        y=strip_df[f"Return ({sel_period})"],
        marker_color=[
            COLORS[2] if v >= 0 else COLORS[1]
            for v in strip_df[f"Return ({sel_period})"]
        ],
        text=[f"{v:+.1f}%" for v in strip_df[f"Return ({sel_period})"]],
        textposition="outside",
        hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
    ))
    fig_bar.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(tickangle=-45),
        yaxis=dict(title=f"Return % ({sel_period})", zeroline=True,
                   zerolinecolor="#d1d5db", showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_bar, use_container_width=True)