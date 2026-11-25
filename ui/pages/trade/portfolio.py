"""
portfolio.py
────────────
Full-featured portfolio view.

Features:
  - Portfolio movement across 1D / 7D / 1M / 3M / 6M / 1Y / 2Y / 3Y / 4Y / 5Y / 10Y
    or a custom Start date → End date range
  - Portfolio weight % alongside each ticker
  - Full-width Sector Distribution + Index comparison charts
  - Today's gain % per holding
  - Today's winner / loser highlighted rows
  - Sector orientation based on defined sectors
  - Historical data via DuckDB (one_ticker_common.py connection)
"""

import os
import json
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import time
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
    ("4M",  120,   "4 months"),
    ("5M",  150,   "5 months"),
    ("6M",  180,  "6 months"),
    ("1Y",  365,  "1 year"),
    ("2Y",  730,  "2 years"),
    ("3Y",  1095, "3 years"),
    ("4Y",  1460, "4 years"),
    ("5Y",  1825, "5 years"),
    ("10Y",  3650, "10 years"),
]

# Benchmark index ticker used for comparison (must exist in price_history)
INDEX_TICKER = "NIFTYBEES"

# Extra tickers that get the same SMA50 trend banner as INDEX_TICKER
# {symbol: banner heading}
TREND_BANNER_TICKERS = {
    "GOLDBEES":   "Gold Trend",
    "SILVERBEES": "Silver Trend",
}


# ==============================================================================
# PERSISTENT SETTINGS  (survives app restarts)
# ==============================================================================
# JSON layout:
#   {"portfolio_movement": {"period": "30D", "custom_start": "2026-10-01",
#                           "custom_end": "2026-10-31"}}   # custom_end null = today
# Other sections can be added to the same file without clobbering each other.

SETTINGS_PATH = "../../outputs/trade_portfolio_settings.json"


def _resolve_settings_path(path: str) -> str:
    """Relative paths are resolved against this file's folder (same as PINNED_FILE)."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), path))


def load_portfolio_settings(path: str = SETTINGS_PATH) -> dict:
    """Load saved settings. Returns {} if the file is missing, empty or corrupt."""
    try:
        with open(_resolve_settings_path(path), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_portfolio_settings(settings: dict, path: str = SETTINGS_PATH) -> None:
    """Atomically write settings to disk (temp file + replace, so no half-written JSON)."""
    full = _resolve_settings_path(path)
    try:
        os.makedirs(os.path.dirname(full), exist_ok=True)
        tmp = full + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        os.replace(tmp, full)
    except OSError as e:
        st.warning(f"Could not save portfolio settings: {e}")


# ==============================================================================
# HISTORICAL DATA HELPERS
# ==============================================================================

def _log_sql(label: str, sql: str, df: "pd.DataFrame | None" = None,
             elapsed_ms: float | None = None, error: str | None = None) -> None:
    """Record an executed SQL statement (plus result stats) for the debug panel."""
    entry = {"label": label, "sql": sql.strip(),
             "elapsed_ms": elapsed_ms, "error": error,
             "rows": None, "date_min": None, "date_max": None, "per_ticker_last": None}
    if df is not None:
        entry["rows"] = len(df)
        if "date" in df.columns and len(df):
            entry["date_min"] = str(pd.to_datetime(df["date"]).min().date())
            entry["date_max"] = str(pd.to_datetime(df["date"]).max().date())
            if "ticker" in df.columns:
                last = (df.assign(date=pd.to_datetime(df["date"]))
                        .groupby("ticker")["date"].max().dt.date.astype(str))
                entry["per_ticker_last"] = last.reset_index().rename(
                    columns={"ticker": "ticker", "date": "last_date"})
    st.session_state.setdefault("_sql_log", []).append(entry)


def _render_sql_log(slot) -> None:
    """Draw every SQL executed during this run into the given placeholder."""
    log = st.session_state.get("_sql_log", [])
    with slot.container():
        with st.expander(f"🧾 Executed SQL ({len(log)})", expanded=False):
            if not log:
                st.caption("No SQL executed in this run.")
            for i, e in enumerate(log, 1):
                st.markdown(f"**{i}. {e['label']}**")
                st.code(e["sql"], language="sql")
                bits = []
                if e["rows"] is not None:
                    bits.append(f"rows: **{e['rows']}**")
                if e["date_min"]:
                    bits.append(f"dates: **{e['date_min']} → {e['date_max']}**")
                if e["elapsed_ms"] is not None:
                    bits.append(f"{e['elapsed_ms']:.0f} ms")
                if bits:
                    st.caption("  ·  ".join(bits))
                if e["error"]:
                    st.error(e["error"])
                if e["per_ticker_last"] is not None:
                    # (no nested expander: Streamlit doesn't allow it)
                    st.caption("Last date per ticker")
                    st.dataframe(e["per_ticker_last"], hide_index=True,
                                 use_container_width=True, height=180)


def get_price_history(nse_symbols: list[str], days: int, is_benchmark: bool = False,
                      end_date: "datetime.date | None" = None) -> pd.DataFrame:
    """
    Pull close prices from DuckDB `ticker_prices` using nse_symbol column.
    `days` counts back from today (window start); `end_date` (optional) caps the
    window on the right, so a custom Start → End range can be queried.
    Returns DataFrame with columns: date, ticker (=nse_symbol), close.
    Returns empty DataFrame if DuckDB unavailable or no data found.
    """
    if not _HAS_COMMON or not nse_symbols:
        return pd.DataFrame()

    table_name = "benchmark_prices" if is_benchmark else "ticker_prices"
    query = ""
    t0 = time.perf_counter()
    try:
        conn = get_duckdb_connection()
        initialize_views_or_mock(conn)

        sym_list = ", ".join(f"'{s}'" for s in nse_symbols)
        end_clause = f"AND Date <= DATE '{end_date:%Y-%m-%d}'" if end_date else ""
        query = f"""
            SELECT Date       AS date,
                   nse_symbol AS ticker,
                   Close      AS close
            FROM {table_name}
            WHERE nse_symbol IN ({sym_list})
              AND Date >= CURRENT_DATE - INTERVAL '{days} days'
              {end_clause}
            ORDER BY nse_symbol, Date
        """
        df = conn.execute(query).df()
        df["date"] = pd.to_datetime(df["date"])
        _log_sql(f"get_price_history → {table_name} ({len(nse_symbols)} symbols, {days} days"
                 + (f", to {end_date}" if end_date else "") + ")",
                 query, df, (time.perf_counter() - t0) * 1000)
        return df
    except Exception as exc:
        # Keep the dashboard resilient, but log the underlying query/schema issue.
        print(f"[Portfolio] Price history query failed ({table_name}): {exc}")
        _log_sql(f"get_price_history → {table_name} (FAILED)", query or "-- query not built",
                 None, (time.perf_counter() - t0) * 1000, str(exc))
        return pd.DataFrame()


def _carry_forward(s: pd.Series, target_index) -> pd.Series:
    """Extend series `s` to the dates in `target_index` that are newer than its
    last point, repeating the last known value. Used because benchmark_prices
    (hence the index SMA50/SMA200) lags ~3 days behind ticker_prices."""
    if s is None or s.empty or target_index is None or len(target_index) == 0:
        return s
    extra = target_index[target_index > s.index.max()]
    if len(extra) == 0:
        return s
    return pd.concat([s, pd.Series(s.iloc[-1], index=extra)])


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
            #port_val += pivot[tkr].fillna(0) * qty
            port_val += pivot[tkr].bfill().ffill().fillna(0) * qty
    return port_val


def compute_period_return(port_series: pd.Series) -> float:
    """% return over the series."""
    clean = port_series.dropna()
    if len(clean) < 2:
        return 0.0
    start = clean.iloc[0]
    end   = clean.iloc[-1]
    return ((end - start) / start * 100) if start else 0.0


def _trend_snapshot(ticker: str, lookback_days: int = 150):
    """Return (last_price, sma50) for `ticker`, or (None, None) if unavailable.

    Looks in benchmark_prices first, then falls back to ticker_prices, so it
    works whether the ETF is stored as a benchmark or as a regular ticker.
    benchmark_prices lags ~3 days, so the latest close is topped up from
    ticker_prices when a newer one exists (SMA50 stays as last computed).
    """
    hist = pd.DataFrame()
    for is_bench in (True, False):
        hist = get_price_history([ticker], lookback_days, is_benchmark=is_bench)
        if not hist.empty:
            break
    if hist.empty:
        return None, None

    close = (hist.set_index("date")["close"]
             .pipe(lambda s: s[~s.index.duplicated(keep="last")])
             .sort_index().dropna())
    if len(close) < 50:
        return None, None

    sma50 = close.rolling(window=50, min_periods=50).mean().iloc[-1]
    last  = close.iloc[-1]
    try:
        live = get_price_history([ticker], 15)
        if not live.empty:
            live = live[live["date"] > close.index.max()].sort_values("date")
            if not live.empty:
                last = live["close"].iloc[-1]
    except Exception:
        pass
    return float(last), float(sma50)


def _render_trend_banner(ticker: str, last: float, sma50: float,
                         label: str = "Market Trend", panic: bool = False) -> None:
    """Compact SMA50 trend banner (sized to sit in one of three columns).
    green = above SMA50, red = below. panic=True uses 'EXIT PANIC' wording."""
    gap = (last - sma50) / sma50 * 100
    if last < sma50:
        kicker   = "Market Risk Alert" if panic else f"{label} Alert"
        headline = f"{ticker} is below its SMA50"
        tail     = "— EXIT PANIC" if panic else "— trend weakening"
        icon, icon_sz, pad = "🚨", "2rem", "16px 18px"
        bg, border = "linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%)", "#ef4444"
        kick_c = "#fca5a5"
    else:
        kicker   = label
        headline = f"{ticker} is above its SMA50"
        tail     = "— trend intact"
        icon, icon_sz, pad = "✅", "1.8rem", "16px 18px"
        bg, border = "linear-gradient(135deg, #14532d 0%, #166534 100%)", "#22c55e"
        kick_c = "#86efac"

    def _chip(lbl, val, color="#fff"):
        return (f'<span style="white-space:nowrap;">{lbl}: '
                f'<strong style="color:{color}">{val}</strong></span>')

    html = (
        f'<div style="background:{bg};border:2px solid {border};border-radius:12px;'
        f'padding:{pad};margin-bottom:16px;min-height:150px;'
        f'display:flex;align-items:center;gap:14px;">'
        f'<span style="font-size:{icon_sz};">{icon}</span><div>'
        f'<div style="color:{kick_c};font-size:0.74rem;font-weight:700;'
        f'letter-spacing:0.12em;text-transform:uppercase;">{kicker}</div>'
        f'<div style="color:#fff;font-size:1.05rem;font-weight:800;margin:2px 0 0;">{headline}</div>'
        f'<div style="color:#fff;font-size:1.05rem;font-weight:800;margin:0 0 4px;">{tail}</div>'
        f'<div style="color:{kick_c};font-size:0.82rem;display:flex;flex-wrap:wrap;gap:2px 10px;">'
        f'{_chip(ticker, f"₹{last:,.2f}")}'
        f'{_chip("SMA50", f"₹{sma50:,.2f}")}'
        f'{_chip("Gap", f"{gap:+.2f}%", kick_c)}'
        f'</div></div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ==============================================================================
# MAIN RENDER
# ==============================================================================

def _num(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _fetch_live_holdings(errors: list) -> dict:
    """Fresh Kite holdings per connected account ({account: [holding, ...]}).

    Re-fetched from Kite so day change is real-time; falls back to the copy already in
    session state for any account that cannot be fetched.
    """
    fresh = {}
    accounts = st.session_state.get("accounts") or []
    fetch_holdings = None
    if accounts:
        try:
            from ui.pages.trade.trade_utils import fetch_holdings
        except Exception as e:                               # pragma: no cover
            errors.append(f"fetch_holdings unavailable: {e}")
    if fetch_holdings is not None:
        for acc in accounts:
            tok = acc.get("token")
            if not tok:
                continue
            try:
                fresh[acc.get("name", "?")] = fetch_holdings(tok) or []
            except Exception as e:
                errors.append(f"{acc.get('name', '?')} holdings: {e}")
    for name, hlds in (st.session_state.get("all_holdings") or {}).items():
        fresh.setdefault(name, hlds)                         # session copy as fallback
    return fresh


def _live_price_map(force: bool = False, ttl: int = 60) -> tuple:
    """Live prices + day change → ({SYMBOL: {last_price, close_price, day_change_pct, src}}, errors).

    1. Kite HOLDINGS (current portfolio) — last_price, previous close and day change %
       for every holding                                                  → src "Kite holdings"
    2. Kite POSITIONS LTP for symbols that are not in holdings (net overrides day)
                                                                          → src "Kite positions"
    Cached in session state for `ttl` seconds; `force=True` re-fetches.
    """
    cache = st.session_state.get("pf_live_cache")
    if cache and not force and (time.time() - cache["ts"]) < ttl:
        return cache["map"], cache["errors"]

    out, errors = {}, []

    # 1) holdings — the source of truth for day change
    for hlds in _fetch_live_holdings(errors).values():
        for h in hlds:
            sym = str(h.get("tradingsymbol", "")).strip().upper()
            lp = float(h.get("last_price") or 0)
            if not sym or lp <= 0 or sym in out:
                continue
            cp = float(h.get("close_price") or 0)
            pct = float(h.get("day_change_percentage") or 0)
            if pct == 0 and cp > 0:
                pct = (lp - cp) / cp * 100
            out[sym] = {"last_price": lp, "close_price": cp,
                        "day_change_pct": pct, "src": "Kite holdings"}

    # 2) positions — only for symbols holdings does not have (e.g. bought today, not yet settled)
    try:
        from ui.pages.trade.trade_utils import fetch_positions
    except Exception as e:                                   # pragma: no cover
        fetch_positions = None
        errors.append(f"fetch_positions unavailable: {e}")
    if fetch_positions is not None:
        for acc in (st.session_state.get("accounts") or []):
            tok = acc.get("token")
            if not tok:
                continue
            try:
                pos = fetch_positions(tok) or {}
            except Exception as e:
                errors.append(f"{acc.get('name', '?')}: {e}")
                continue
            for row in (pos.get("day") or []) + (pos.get("net") or []):      # net overrides day
                sym = str(row.get("tradingsymbol", "")).strip().upper()
                lp = float(row.get("last_price") or 0)
                if sym and lp > 0 and (sym not in out or out[sym]["src"] == "Kite positions"):
                    cp = float(row.get("close_price") or 0)
                    out[sym] = {"last_price": lp, "close_price": cp,
                                "day_change_pct": ((lp - cp) / cp * 100) if cp > 0 else 0.0,
                                "src": "Kite positions"}
    st.session_state["pf_live_cache"] = {"ts": time.time(), "map": out, "errors": errors}
    return out, errors


# ── Real-time "today" panel (File Upload mode) ────────────────────────────────
LIVE_REFRESH_OPTIONS = {"Off": 0, "10s": 10, "30s": 30, "60s": 60}


def _live_today_body(qty_map: dict) -> None:
    """Today's status for every uploaded ticker, straight from Kite holdings (re-runs on a timer)."""
    live, errs = _live_price_map(ttl=5)          # ttl dedups bursts; timer is >= 10s
    rows, missing = [], []
    for sym, qty in qty_map.items():
        L = live.get(str(sym).upper())
        if not L or L["close_price"] <= 0:
            missing.append(sym)
            continue
        lp, cp = L["last_price"], L["close_price"]
        rows.append({"Symbol": sym, "Qty": qty, "Prev Close ₹": cp, "LTP ₹": lp,
                     "Day Chg%": L["day_change_pct"], "Today's Gain ₹": qty * (lp - cp),
                     "Source": L["src"]})

    _ts = (st.session_state.get("pf_live_cache") or {}).get("ts")
    st.caption(f"🕒 Last updated: **{datetime.datetime.fromtimestamp(_ts):%H:%M:%S}**"
               if _ts else "🕒 Not updated yet")
    for e in errs or []:
        st.caption(f"⚠️ Kite fetch: {e}")
    if not live:
        st.warning("No Kite account connected — cannot read day change. Connect your account to see live status.")
        return
    if not rows:
        st.info("None of the uploaded tickers were found in your Kite holdings / positions.")
        return

    d = pd.DataFrame(rows).sort_values("Day Chg%", ascending=False).reset_index(drop=True)
    prev_val = float((d["Qty"] * d["Prev Close ₹"]).sum())
    gain = float(d["Today's Gain ₹"].sum())
    up, down = int((d["Day Chg%"] > 0).sum()), int((d["Day Chg%"] < 0).sum())
    best, worst = d.iloc[0], d.iloc[-1]

    m = st.columns(5)
    m[0].metric("Today's Gain", f"₹{gain:,.0f}", f"{gain / prev_val * 100:+.2f}%" if prev_val else "")
    m[1].metric("Advancers / Decliners", f"{up} / {down}", f"{len(d)} tickers", delta_color="off")
    m[2].metric("🏆 Top", best["Symbol"], f"{best['Day Chg%']:+.2f}%")
    m[3].metric("📉 Bottom", worst["Symbol"], f"{worst['Day Chg%']:+.2f}%")
    m[4].metric("Live coverage", f"{len(d)}/{len(qty_map)}", delta_color="off")

    def _clr(col):
        return ["color:#15803d" if v > 0 else "color:#b91c1c" if v < 0 else "" for v in col]
    st.dataframe(
        d.style.apply(_clr, subset=["Day Chg%", "Today's Gain ₹"])
        .format({"Qty": "{:,.0f}", "Prev Close ₹": "{:,.2f}", "LTP ₹": "{:,.2f}",
                 "Day Chg%": "{:+.2f}%", "Today's Gain ₹": "{:+,.0f}"}),
        use_container_width=True, hide_index=True)
    if missing:
        st.caption(f"⚠️ No Kite data for: {', '.join(missing)}")
    st.caption("Outside market hours this shows the last session's move.")


def _render_live_today_status(qty_map: dict) -> None:
    """Header + auto-refresh controls; the table itself refreshes on a timer via st.fragment."""
    st.subheader("⚡ Today's Live Status")
    c1, c2 = st.columns([5, 1])
    choice = c1.radio("Auto-refresh", list(LIVE_REFRESH_OPTIONS), index=1, horizontal=True,
                      key="pf_live_every")
    if c2.button("🔄 Refresh now", key="pf_live_refresh_now", use_container_width=True):
        _live_price_map(force=True)
    every = LIVE_REFRESH_OPTIONS[choice] or None
    frag = getattr(st, "fragment", None)
    if frag is None:                              # old Streamlit: no timer, refresh via button
        _live_today_body(qty_map)
        return
    try:
        frag(run_every=every)(_live_today_body)(qty_map)
    except TypeError:                             # Streamlit < 1.37: fragment without run_every
        frag(_live_today_body)(qty_map)


def _parse_positions_export(udf: "pd.DataFrame", live: dict = None, entry_date=None):
    """Positions export → (ticker_data, tracking).

    Non-exit rows (Side != CLOSED and Qty > 0) are the portfolio holdings.
    CLOSED / Qty == 0 rows are exits; their Chg % is recomputed as 100*P&L/(LTP*Day Buy - P&L).
    Qty and Avg ₹ come from the file; the current price + previous close + day change come from
    Kite HOLDINGS (fallback: Kite positions, then the file's LTP, flagged "File (stale)").
    Exits (CLOSED) use the file's P&L ₹, LTP ₹, Day Buy/Sell; Chg % comes from the formula above.
    """
    live = live or {}
    d = pd.DataFrame({
        "Symbol":   udf["instrument"].astype(str).str.strip().str.upper(),
        "Product":  udf["product"] if "product" in udf.columns else "",
        "Side":     udf["side"].astype(str).str.strip().str.upper(),
        "Qty":      _num(udf["qty"]) if "qty" in udf.columns else 0.0,
        "Avg ₹":    _num(udf["avg ₹"]) if "avg ₹" in udf.columns else 0.0,
        "LTP ₹":    _num(udf["ltp ₹"]) if "ltp ₹" in udf.columns else 0.0,
        "P&L ₹":    _num(udf["p&l ₹"]) if "p&l ₹" in udf.columns else 0.0,
        "Chg %":    _num(udf["chg %"]) if "chg %" in udf.columns else 0.0,
        "Day Buy":  _num(udf["day buy"]).astype(int) if "day buy" in udf.columns else 0,
        "Day Sell": _num(udf["day sell"]).astype(int) if "day sell" in udf.columns else 0,
    })
    d = d[d["Symbol"] != ""]
    is_exit = (d["Side"] == "CLOSED") | (d["Qty"] == 0)
    open_df, exit_df = d[~is_exit].copy(), d[is_exit].copy()

    # one row per symbol (weighted average cost if a symbol repeats across products)
    if not open_df.empty:
        open_df["_cost"] = open_df["Qty"] * open_df["Avg ₹"]
        g = open_df.groupby("Symbol", as_index=False).agg(
            Qty=("Qty", "sum"), _cost=("_cost", "sum"), LTP=("LTP ₹", "last"),
            PnL=("P&L ₹", "sum"), DayBuy=("Day Buy", "sum"), DaySell=("Day Sell", "sum"),
            Side=("Side", "first"), Product=("Product", "first"))
        g["Avg ₹"] = (g["_cost"] / g["Qty"]).where(g["Qty"] > 0, 0.0)
        open_df = pd.DataFrame({
            "Symbol": g["Symbol"], "Product": g["Product"], "Side": g["Side"], "Qty": g["Qty"],
            "Avg ₹": g["Avg ₹"], "LTP ₹": g["LTP"], "P&L ₹": g["PnL"],
            "Chg %": (g["PnL"] / g["_cost"] * 100).where(g["_cost"] > 0, 0.0),
            "Day Buy": g["DayBuy"], "Day Sell": g["DaySell"]})
    open_df["Status"] = ["Partly sold" if r["Day Sell"] > 0 else "Open" for _, r in open_df.iterrows()]
    # live price from Zerodha holdings, else the file's LTP
    has_live = open_df["Symbol"].isin(live.keys())
    open_df["Source"] = [live[k]["src"] if k in live else "File (stale)" for k in open_df["Symbol"]]
    if has_live.any():
        open_df.loc[has_live, "LTP ₹"] = [live[k]["last_price"] for k in open_df.loc[has_live, "Symbol"]]
        _pnl = (open_df["LTP ₹"] - open_df["Avg ₹"]) * open_df["Qty"]
        _chg = ((open_df["LTP ₹"] - open_df["Avg ₹"]) / open_df["Avg ₹"] * 100).where(open_df["Avg ₹"] > 0, 0.0)
        open_df["P&L ₹"] = _pnl.where(has_live, open_df["P&L ₹"])
        open_df["Chg %"] = _chg.where(has_live, open_df["Chg %"])
    exit_df["Status"] = "Exited"
    # Exits: Chg % = 100 × P&L ÷ (LTP × Day Buy − P&L)   (replaces the export's assumed -2%).
    # Falls back to the file's own Chg % only when the formula can't be computed.
    _den = exit_df["LTP ₹"] * exit_df["Day Buy"] - exit_df["P&L ₹"]
    _calc = (100 * exit_df["P&L ₹"] / _den).where((_den > 0) & (exit_df["Day Buy"] > 0))
    exit_df["Chg %"] = _calc.fillna(exit_df["Chg %"])
    exit_df = exit_df[~exit_df["Symbol"].isin(open_df["Symbol"])]   # re-bought names stay open

    ticker_data = {}
    for _, r in open_df.iterrows():
        L = live.get(r["Symbol"])
        if L and L["close_price"] > 0:           # real previous close → true "today" move
            close_p, day_pct = L["close_price"], L["day_change_pct"]
        else:                                     # no live data: treat as entered today
            close_p = float(r["Avg ₹"])
            day_pct = float((r["LTP ₹"] - r["Avg ₹"]) / r["Avg ₹"] * 100) if r["Avg ₹"] else 0.0
        ticker_data[r["Symbol"]] = {
            "last_price": float(r["LTP ₹"]),
            "close_price": float(close_p),
            "day_change_pct": float(day_pct),
            "avg_cost": float(r["Avg ₹"]),
            "Uploaded": float(r["Qty"]),
        }
    stale = open_df.loc[~has_live, "Symbol"].tolist()
    meta = {"live_n": int(has_live.sum()), "total_n": len(open_df), "stale": stale,
            "connected": bool(live), "entry_date": entry_date,
            "by_src": open_df["Source"].value_counts().to_dict()}
    return ticker_data, {"open": open_df.reset_index(drop=True),
                         "exit": exit_df.reset_index(drop=True), "meta": meta}


# ── Estimated Zerodha equity charges (NSE) ────────────────────────────────────
# Same-day buy+sell of a CNC stock is charged as INTRADAY (FIFO): STT 0.025% on the sell side only,
# brokerage 0.03%. Quantity that stays held is DELIVERY: STT 0.1% on the buy (and again on a later sell).
_STT_INTRADAY_SELL = 0.00025
_STT_DELIVERY      = 0.001
_BROKERAGE_INTRA   = 0.0003      # 0.03% (₹20/order cap not applied)
_TXN_NSE           = 0.0000322   # 0.00322%
_SEBI_FEE          = 0.000001    # ₹10 / crore
_STAMP_INTRA_BUY   = 0.00003     # 0.003% on buy side
_STAMP_DELIV_BUY   = 0.00015     # 0.015% on buy side
_GST               = 0.18
_DP_PER_SCRIP      = 13 * 1.18   # ₹13 + GST, only when delivery stock is sold from demat


def _charges(buy_i=0.0, sell_i=0.0, buy_d=0.0, sell_d=0.0, dp_scrips=0):
    """(STT, other charges) for an intraday leg (buy_i/sell_i), a delivery buy (buy_d) and a delivery sell (sell_d)."""
    stt = _STT_INTRADAY_SELL * sell_i + _STT_DELIVERY * (buy_d + sell_d)
    brk = _BROKERAGE_INTRA * (buy_i + sell_i)
    turnover = buy_i + sell_i + buy_d + sell_d
    txn, sebi = _TXN_NSE * turnover, _SEBI_FEE * turnover
    stamp = _STAMP_INTRA_BUY * buy_i + _STAMP_DELIV_BUY * buy_d
    gst = _GST * (brk + txn + sebi)
    return stt, brk + txn + sebi + stamp + gst + dp_scrips * _DP_PER_SCRIP


def _with_charges(df: "pd.DataFrame", kind: str) -> "pd.DataFrame":
    """Add STT ₹ / Charges ₹ / Net P&L ₹ (and, for open rows, 'If sold now ₹') estimates."""
    d = df.copy()
    stt_l, tot_l, now_l = [], [], []
    for _, r in d.iterrows():
        if kind == "exit":                     # bought and sold the same day → intraday
            n = max(int(r["Day Buy"]), int(r["Day Sell"]))
            buy_i = n * float(r["LTP ₹"])      # entry/exit value is not in the file → qty × LTP
            sell_i = buy_i + float(r["P&L ₹"])
            stt, oth = _charges(buy_i=buy_i, sell_i=sell_i)
            now = 0.0
        else:
            p = float(r["Avg ₹"])
            qi = min(int(r["Day Buy"]), int(r["Day Sell"]))        # squared off today → intraday leg
            stt, oth = _charges(buy_i=qi * p, sell_i=qi * p, buy_d=float(r["Qty"]) * p)
            s2, o2 = _charges(sell_d=float(r["Qty"]) * float(r["LTP ₹"]),
                              dp_scrips=1 if r["Qty"] > 0 else 0)   # charges if the rest is sold at LTP now
            now = s2 + o2
        stt_l.append(stt); tot_l.append(stt + oth); now_l.append(now)
    d["STT ₹"] = stt_l
    d["Charges ₹"] = tot_l
    d["Net P&L ₹"] = d["P&L ₹"] - d["Charges ₹"]
    if kind != "exit":
        d["If sold now ₹"] = now_l
    return d


def _render_position_tracking(trk: dict) -> None:
    """Open (in portfolio) vs exited positions from the Positions export."""
    o, x, meta = trk["open"], trk["exit"], trk.get("meta", {})
    o, x = _with_charges(o, "open"), _with_charges(x, "exit")
    st.subheader("🎯 Position Tracking")
    if meta.get("entry_date"):
        _days = (datetime.date.today() - meta["entry_date"]).days
        st.caption(f"📅 Entry file dated **{meta['entry_date']:%d %b %Y}** · tracked for **{_days} day(s)**")
    rc1, rc2 = st.columns([5, 1])
    if rc2.button("🔄 Refresh prices", key="pf_track_refresh", use_container_width=True):
        _live_price_map(force=True)
        st.rerun()
    for _e in meta.get("errors", []) or []:
        rc1.caption(f"⚠️ Kite positions fetch: {_e}")
    _src = ", ".join(f"{v} {k}" for k, v in (meta.get("by_src") or {}).items())
    if not meta.get("connected"):
        rc1.warning("No Kite account connected — prices below are the file's LTP (stale). "
                    "Connect your account to track live.")
    elif meta.get("stale"):
        rc1.warning(f"{len(meta['stale'])} of {meta['total_n']} open symbols are in neither Kite holdings nor "
                    f"holdings and use the file's LTP: {', '.join(meta['stale'])}")
    else:
        rc1.success(f"🟢 Live prices for all {meta.get('live_n', 0)} open positions ({_src}). "
                    "Qty and Avg ₹ from the file.")
    open_pnl, exit_pnl = float(o["P&L ₹"].sum()), float(x["P&L ₹"].sum())
    invested = float((o["Qty"] * o["Avg ₹"]).sum())
    exposure = float((o["Qty"] * o["LTP ₹"]).sum())
    n_all = len(o) + len(x)
    winners = int((o["P&L ₹"] > 0).sum())
    m = st.columns(6)
    m[0].metric("Open (in Portfolio)", len(o), f"{len(o) / n_all * 100:.0f}% of {n_all}" if n_all else "")
    m[1].metric("Exited", len(x), f"{len(x) / n_all * 100:.0f}% of {n_all}" if n_all else "", delta_color="off")
    m[2].metric("Open P&L", f"₹{open_pnl:,.0f}", f"{open_pnl / invested * 100:+.2f}%" if invested else "")
    m[3].metric("Exit P&L (as reported)", f"₹{exit_pnl:,.0f}")
    m[4].metric("Net P&L", f"₹{open_pnl + exit_pnl:,.0f}")
    m[5].metric("Open win rate", f"{winners}/{len(o)}", f"{winners / len(o) * 100:.0f}%" if len(o) else "")
    m2 = st.columns(4)
    m2[0].metric("Capital in open", f"₹{invested:,.0f}")
    m2[1].metric("Open exposure (LTP)", f"₹{exposure:,.0f}")
    m2[2].metric("Avg Chg % — open", f"{o['Chg %'].mean():+.2f}%" if len(o) else "–")
    m2[3].metric("Avg Chg % — exited", f"{x['Chg %'].mean():+.2f}%" if len(x) else "–")

    # ── After STT & charges (estimates) ──────────────────────────────────────
    stt_all = float(o["STT ₹"].sum() + x["STT ₹"].sum())
    chg_all = float(o["Charges ₹"].sum() + x["Charges ₹"].sum())
    net_all = open_pnl + exit_pnl - chg_all
    sell_now = float(o["If sold now ₹"].sum()) if "If sold now ₹" in o.columns else 0.0
    st.markdown("**After STT & other charges (estimated)**")
    m3 = st.columns(5)
    m3[0].metric("STT", f"₹{stt_all:,.2f}")
    m3[1].metric("Other charges", f"₹{chg_all - stt_all:,.2f}", help="Brokerage on same-day square-offs, exchange, SEBI, stamp, GST")
    m3[2].metric("Net P&L after charges", f"₹{net_all:,.0f}", f"{net_all - (open_pnl + exit_pnl):+,.0f} vs gross",
                 delta_color="inverse" if chg_all else "off")
    m3[3].metric("Open P&L after charges", f"₹{open_pnl - float(o['Charges ₹'].sum()):,.0f}")
    m3[4].metric("Net if all open sold now", f"₹{net_all - sell_now:,.0f}", f"{-sell_now:+,.0f} sell-side charges",
                 delta_color="inverse", help="Adds delivery STT 0.1% on the sell, exchange/SEBI/GST and ₹13+GST DP per scrip.")

    def _clr(col):
        return ["color:#15803d" if v > 0 else "color:#b91c1c" if v < 0 else "" for v in col]
    fmt = {"Avg ₹": "{:,.2f}", "LTP ₹": "{:,.2f}", "P&L ₹": "{:+,.2f}", "Chg %": "{:+.2f}%",
           "STT ₹": "{:,.2f}", "Charges ₹": "{:,.2f}", "Net P&L ₹": "{:+,.2f}", "If sold now ₹": "{:,.2f}"}
    cols = ["Symbol", "Status", "Qty", "Avg ₹", "LTP ₹", "P&L ₹", "Chg %", "STT ₹", "Charges ₹",
            "Net P&L ₹", "Day Buy", "Day Sell"]
    ocols = cols + ["If sold now ₹", "Source"]
    xcols = ["Symbol", "Status", "Avg ₹", "LTP ₹", "P&L ₹", "Chg %", "STT ₹", "Charges ₹",
             "Net P&L ₹", "Day Buy", "Day Sell"]
    def _show(df):
        """Styler needs a unique index and unique columns — always reset before styling."""
        df = df.loc[:, ~df.columns.duplicated()].reset_index(drop=True)
        st.dataframe(df.style.apply(_clr, subset=["P&L ₹", "Chg %", "Net P&L ₹"]).format(fmt),
                     use_container_width=True, hide_index=True)
    t_open, t_exit, t_all = st.tabs([f"🟢 Open ({len(o)})", f"🔴 Exited ({len(x)})", f"📋 All ({n_all})"])
    with t_open:
        if o.empty:
            st.caption("No open positions.")
        else:
            _show(o[ocols].sort_values("Chg %", ascending=False))
    with t_exit:
        if x.empty:
            st.caption("No exited positions.")
        else:
            _show(x[xcols])
            st.caption("Exited rows are removed from the portfolio and shown exactly as reported in the file "
                       "(Day Buy / Day Sell = entry / exit qty). Their P&L is included in Net P&L; charges are "
                       "estimated as intraday (value = qty × LTP, since entry/exit values are not in the file).")
    with t_all:
        _show(pd.concat([o, x], ignore_index=True)[cols])
    st.caption("Open positions = Qty and Avg ₹ from the file × live Kite holdings LTP (positions LTP if not in holdings). "
               "All non-exit positions are assumed available in the Portfolio. "
               "Today's Gain / Day Chg% use Kite holdings' previous close when matched.")
    st.caption("💸 Charges are Zerodha equity estimates (NSE, not the contract note): quantity bought and sold the same day "
               "is intraday — STT 0.025% on the sell side only, brokerage 0.03%, no DP charge; quantity still held is "
               "delivery — STT 0.1% on the buy (and 0.1% again when sold), zero brokerage. Also counted: exchange "
               "0.00322%, SEBI ₹10/crore, stamp duty, 18% GST. STT is not rounded and the ₹20/order brokerage cap "
               "is not applied.")


def render():
    st.set_page_config(page_title="Portfolio", layout="wide", page_icon="📊")

    # ── SQL debug log: reset each run, drawn into this slot at the end ───────
    st.session_state["_sql_log"] = []
    _sql_slot = st.empty()

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

    # ── Input-mode tab bar ────────────────────────────────────────────────────
    # All three tabs are always rendered by Streamlit on every run.
    # We resolve ticker_data inside each tab context so only the active
    # tab's logic actually produces output / stops the page.
    ticker_data   = None
    account_names = []
    input_mode    = "current"   # will be overwritten inside the active tab

    # An explicit selector replaces st.tabs: Streamlit cannot tell which tab is open, so an
    # uploaded file used to keep winning even after "Current Holdings" was clicked.
    _SRC_LABELS = {"current": "📋 Current Holdings", "manual": "✏️ Manual", "upload": "📂 File Upload"}
    _src_label = st.radio("Portfolio source", list(_SRC_LABELS.values()), horizontal=True,
                          key="holdings_source_radio", label_visibility="collapsed")
    input_mode = next(k for k, v in _SRC_LABELS.items() if v == _src_label)

    excl_set = {s.upper() for s in (getattr(st.session_state, "excluded_symbols", None) or set())}
    _ticker_data_curr = _ticker_data_manual = _ticker_data_upload = None
    _upload_track = None

    # ── Tab: Current Holdings ─────────────────────────────────────────────────
    if input_mode == "current":
        account_names = []
        _ticker_data_curr = None

        # Switching back from Manual / File Upload (or pressing Refresh) pulls live holdings from Kite
        _prev_src = st.session_state.get("pf_prev_src")
        _force_pull = _prev_src not in (None, "current")
        _hc1, _hc2 = st.columns([5, 1])
        if _hc2.button("🔄 Refresh holdings", key="pf_refresh_holdings", use_container_width=True):
            _force_pull = True
        if _force_pull and (st.session_state.get("accounts") or []):
            try:
                from ui.pages.trade.trade_utils import fetch_holdings
                _fresh, _errs = dict(st.session_state.get("all_holdings") or {}), []
                for _acc in st.session_state.get("accounts") or []:
                    if not _acc.get("token"):
                        continue
                    try:
                        _fresh[_acc["name"]] = fetch_holdings(_acc["token"])
                    except Exception as _e:
                        _errs.append(f"{_acc.get('name', '?')}: {_e}")
                st.session_state["all_holdings"] = _fresh
                if _errs:
                    _hc1.warning("Could not refresh from Kite — showing last known holdings. " + " | ".join(_errs))
                else:
                    _hc1.success("🟢 Live holdings pulled from Kite.")
            except Exception as _e:
                _hc1.warning(f"Could not pull live holdings: {_e}")

        if "all_holdings" not in st.session_state or not st.session_state.all_holdings:
            st.warning(
                "No portfolio holdings matched the matrix. "
                "Connect your broker or add holdings."
            )
        else:
            all_holdings  = st.session_state.all_holdings
            excl_set      = {s.upper() for s in (getattr(st.session_state, "excluded_symbols", None) or set())}
            account_names = list(all_holdings.keys())
            _ticker_data_curr = {}
            for acc_name, holdings in all_holdings.items():
                for h in holdings:
                    sym = h["tradingsymbol"]
                    if sym.upper() in excl_set:
                        print(f"[Ticker] Excluded : {sym.upper()}")   # ← add this
                        continue
                    if sym not in _ticker_data_curr:
                        _ticker_data_curr[sym] = {
                            "last_price":     h.get("last_price",  0),
                            "close_price":    h.get("close_price", 0),
                            "day_change_pct": h.get("day_change_percentage", 0),
                            "avg_cost":       h.get("average_price", 0),
                            **{an: 0 for an in account_names},
                        }
                    _ticker_data_curr[sym][acc_name] += h.get("quantity", 0)

    # ── Tab: Manual ───────────────────────────────────────────────────────────
    if input_mode == "manual":
        _input_mode_manual  = "manual"
        _account_names_manual = ["Manual"]
        st.markdown("Enter NSE symbols separated by commas or newlines (e.g. `RELIANCE, TCS, INFY`).")
        _raw_input = st.text_area(
            "Tickers",
            value=st.session_state.get("manual_tickers_raw", ""),
            height=120,
            placeholder="RELIANCE, TCS, INFY, HDFCBANK",
            label_visibility="collapsed",
            key="manual_tickers_input",
        )
        st.session_state.manual_tickers_raw = _raw_input
        _manual_symbols = [
            s.strip().upper()
            for s in _raw_input.replace("\n", ",").split(",")
            if s.strip()
        ]
        if not _manual_symbols:
            st.info("Enter at least one ticker symbol above to continue.")
        _ticker_data_manual = {
            sym: {
                "last_price": 0, "close_price": 0,
                "day_change_pct": 0, "avg_cost": 0,
                "Manual": 1,
            }
            for sym in _manual_symbols
        } if _manual_symbols else None

    # ── Tab: File Upload ──────────────────────────────────────────────────────
    if input_mode == "upload":
        _input_mode_upload    = "upload"
        _account_names_upload = ["Uploaded"]
        st.markdown(
            "Upload the **Positions export** (`Product, Instrument, Exchange, Side, Qty, Avg ₹, LTP ₹, "
            "P&L ₹, Chg %, Day Buy, Day Sell`). Non-exit rows (LONG) are loaded into the portfolio; "
            "CLOSED rows are treated as exits and shown in Position Tracking. \n\n"
            "A simple CSV with a **`ticker`** column (optional `quantity`, `avg_cost`) also works."
        )
        _uploaded_file = st.file_uploader(
            "Upload CSV", type=["csv"],
            key="holdings_csv_upload",
            label_visibility="collapsed",
        )
        _ticker_data_upload = None
        _upload_track = None          # {"open": df, "exit": df} when a Positions export is uploaded
        if _uploaded_file is not None:      # remember it: the uploader resets when another source is chosen
            st.session_state["pf_upload_bytes"] = _uploaded_file.getvalue()
            st.session_state["pf_upload_name"] = _uploaded_file.name
        elif st.session_state.get("pf_upload_bytes"):
            import io as _io
            _uploaded_file = _io.BytesIO(st.session_state["pf_upload_bytes"])
            _uploaded_file.name = st.session_state.get("pf_upload_name", "uploaded.csv")
            _c1, _c2 = st.columns([5, 1])
            _c1.caption(f"📎 Using previously uploaded file: **{_uploaded_file.name}**")
            if _c2.button("✖ Remove", key="pf_upload_clear", use_container_width=True):
                st.session_state.pop("pf_upload_bytes", None)
                st.session_state.pop("pf_upload_name", None)
                st.rerun()
        if _uploaded_file is not None:
            try:
                udf = pd.read_csv(_uploaded_file, encoding="utf-8-sig")   # utf-8-sig strips the BOM
                udf.columns = [c.strip().lower() for c in udf.columns]
                _ticker_col = next(
                    (c for c in ["ticker", "symbol", "nse_symbol"] if c in udf.columns), None
                )
                if {"instrument", "side"}.issubset(udf.columns):
                    # Positions export: LONG rows → portfolio, CLOSED rows → exits (tracking)
                    import re as _re
                    _m = _re.search(r"(\d{4}-\d{2}-\d{2})", getattr(_uploaded_file, "name", "") or "")
                    _entry = (datetime.datetime.strptime(_m.group(1), "%Y-%m-%d").date() if _m else None)
                    _live, _live_err = _live_price_map()
                    _ticker_data_upload, _upload_track = _parse_positions_export(
                        udf, live=_live, entry_date=_entry)
                    if _upload_track is not None:
                        _upload_track["meta"]["errors"] = _live_err
                    if not _ticker_data_upload:
                        st.warning("No open (non-exit) positions found in the uploaded file.")
                        _ticker_data_upload = None
                elif _ticker_col is None:
                    st.error("CSV must have a column named `ticker`, `symbol`, or `nse_symbol`.")
                else:
                    udf["_sym"] = udf[_ticker_col].str.strip().str.upper()
                    udf["_sym"] = udf["_sym"].str.replace(r"\.(NS|BO)$", "", regex=True)
                    _live, _live_err = _live_price_map()       # day change from Kite holdings
                    _ticker_data_upload = {}
                    for _, row in udf.iterrows():
                        sym = row["_sym"]
                        if not sym:
                            continue
                        qty      = float(row["quantity"]) if "quantity" in row and pd.notna(row["quantity"]) else 1.0
                        avg_cost = float(row["avg_cost"])  if "avg_cost"  in row and pd.notna(row["avg_cost"])  else 0.0
                        L = _live.get(sym)
                        _ticker_data_upload[sym] = {
                            "last_price":     L["last_price"]     if L else 0,
                            "close_price":    L["close_price"]    if L else 0,
                            "day_change_pct": L["day_change_pct"] if L else 0,
                            "avg_cost": avg_cost,
                            "Uploaded": qty,
                        }
                    _no_live = [k for k in _ticker_data_upload if k not in _live]
                    if not _live:
                        st.warning("No Kite account connected — day change will show as 0.")
                    elif _no_live:
                        st.caption(f"⚠️ Not found in Kite holdings/positions: {', '.join(_no_live)}")
                    if not _ticker_data_upload:
                        st.warning("No valid tickers found in the uploaded file.")
                        _ticker_data_upload = None
            except Exception as exc:
                st.error(f"Could not parse CSV: {exc}")

    st.session_state["pf_prev_src"] = input_mode

    # ── Resolve data source: exactly the one selected above ──────────────────
    if input_mode == "upload" and _ticker_data_upload is not None:
        ticker_data   = _ticker_data_upload
        account_names = ["Uploaded"]
    elif input_mode == "manual" and _ticker_data_manual is not None:
        ticker_data   = _ticker_data_manual
        account_names = ["Manual"]
    elif input_mode == "current" and _ticker_data_curr is not None:
        ticker_data   = _ticker_data_curr
        account_names = list(st.session_state.all_holdings.keys())
    else:
        if input_mode == "upload":
            st.info("Upload a CSV above to continue.")
        st.stop()

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
            _t0 = time.perf_counter()
            meta_df = conn.execute(sql_stmt).df()
            _log_sql("ticker metadata → tickers", sql_stmt, meta_df,
                     (time.perf_counter() - _t0) * 1000)
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
            _log_sql("ticker metadata → tickers (FAILED)", locals().get("sql_stmt", "-- query not built"),
                     None, None, str(e))
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
        row["P&L ₹"]         = round(total_qty * (lp - avg_cost), 2) if avg_cost else 0.0
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
    #tickers_all    = list(holdings_qty.keys())
    tickers_all    = [s for s in df["Symbol"].tolist() if s != INDEX_TICKER]



    # ── TOP METRICS ───────────────────────────────────────────────────────────
    col_title, col_btn = st.columns([8, 1])
    col_title.title("📊 Portfolio Dashboard")
    _rebalance_enabled = (input_mode == "current")
    if col_btn.button(
            "⚖️ Rebalance Now",
            type="primary",
            key="dash_rebalance_now",
            use_container_width=True,
            disabled=not _rebalance_enabled,
            help=None if _rebalance_enabled else "Rebalance is only available in Current Holdings mode",
    ):
        with st.spinner("Redirecting to Portfolio Rebalance in 4 seconds..."):
            time.sleep(4)
        st.session_state["active_page"] = "trade_rebalance_planner"
        st.rerun()

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

    if input_mode == "upload":
        _render_live_today_status(holdings_qty)
        st.divider()

    if input_mode == "upload" and _upload_track is not None:
        _render_position_tracking(_upload_track)
        st.divider()

    # ── PORTFOLIO MOVEMENT (period selector) ──────────────────────────────────
    st.subheader("Portfolio Movement")
    _CUSTOM_LABEL = "📅 Custom"
    period_labels = [p[0] for p in PERIODS] + [_CUSTOM_LABEL]
    _today_d = datetime.date.today()
    _min_d   = _today_d - datetime.timedelta(days=3650)
    _max_d   = _today_d - datetime.timedelta(days=1)

    # First run of this session → restore last saved choice from disk.
    if "mv_period" not in st.session_state:
        _saved = load_portfolio_settings().get("portfolio_movement", {})
        _saved_period = _saved.get("period")
        if _saved_period not in period_labels:
            _saved_period = "30D"
        try:
            _saved_start = datetime.date.fromisoformat(str(_saved.get("custom_start", "")))
        except ValueError:
            _saved_start = _today_d - datetime.timedelta(days=30)
        try:
            _saved_end = datetime.date.fromisoformat(str(_saved.get("custom_end", "")))
        except ValueError:
            _saved_end = _today_d          # none saved → follow "today"
        st.session_state["mv_period"]       = _saved_period
        st.session_state["pm_custom_start"] = _saved_start
        st.session_state["pm_custom_end"]   = _saved_end
        st.session_state["_pm_saved"] = {
            "period": _saved_period,
            "custom_start": _saved_start.isoformat(),
            "custom_end": None if _saved_end >= _today_d else _saved_end.isoformat(),
        }

    # Keep the remembered custom dates inside the allowed window (they may have aged out).
    _custom_start = min(max(st.session_state.get("pm_custom_start", _max_d), _min_d), _max_d)
    _custom_end   = min(max(st.session_state.get("pm_custom_end", _today_d),
                            _custom_start + datetime.timedelta(days=1)), _today_d)

    sel_period = st.radio("Period", period_labels, key="mv_period",
                          horizontal=True, label_visibility="collapsed")
    _period_choice = sel_period   # raw choice, kept for saving
    end_date = None               # None = open-ended (up to today)

    if sel_period == _CUSTOM_LABEL:
        # Specific Start → End range. `days` (look-back from today to the start
        # date) keeps every downstream query / SMA lookback working unchanged;
        # `end_date` additionally caps the queries on the right.
        _dc1, _dc2, _dc3 = st.columns([2, 2, 6])
        _start_d = _dc1.date_input(
            "Start date",
            value=_custom_start,
            min_value=_min_d,
            max_value=_max_d,
            format="DD/MM/YYYY",
            key="mv_custom_start",
        )
        # End must be after Start: clamp any stale widget value *before* the
        # widget is created (Streamlit raises if the stored value is out of range).
        _end_min = _start_d + datetime.timedelta(days=1)
        if "mv_custom_end" in st.session_state:
            st.session_state["mv_custom_end"] = min(
                max(st.session_state["mv_custom_end"], _end_min), _today_d)
        _end_d = _dc2.date_input(
            "End date",
            value=min(max(_custom_end, _end_min), _today_d),
            min_value=_end_min,
            max_value=_today_d,
            format="DD/MM/YYYY",
            key="mv_custom_end",
        )
        _custom_start, _custom_end = _start_d, _end_d
        days       = max(1, (_today_d - _start_d).days)
        window_days = max(1, (_end_d - _start_d).days)
        if _end_d < _today_d:
            end_date   = _end_d
            sel_period = f"{_start_d:%d %b %Y} → {_end_d:%d %b %Y}"
        else:
            sel_period = f"since {_start_d:%d %b %Y}"
        _dc3.caption(f"Showing movement from **{_start_d:%d %b %Y}** to "
                     f"**{_end_d:%d %b %Y}** ({window_days} days).")
    else:
        days = next(p[1] for p in PERIODS if p[0] == sel_period)
        window_days = days

    # Remember the custom dates even while another period is selected, and
    # write to disk only when something actually changed.
    st.session_state["pm_custom_start"] = _custom_start
    st.session_state["pm_custom_end"]   = _custom_end
    _pm_now = {
        "period": _period_choice,
        "custom_start": _custom_start.isoformat(),
        # null when the end is "today", so it keeps following today's date
        "custom_end": None if _custom_end >= _today_d else _custom_end.isoformat(),
    }
    if st.session_state.get("_pm_saved") != _pm_now:
        _all_settings = load_portfolio_settings()
        _all_settings["portfolio_movement"] = _pm_now
        save_portfolio_settings(_all_settings)
        st.session_state["_pm_saved"] = _pm_now

    with st.spinner("Loading price history…"):
        # Fetch holdings and benchmark separately: the index history lives in
        # benchmark_prices, while individual holdings live in ticker_prices.
        # Previously NIFTYBEES was requested from ticker_prices, so the index
        # trace was silently missing whenever that table did not contain it.
        _holdings_ph = get_price_history(tickers_all, days, end_date=end_date)
        _benchmark_ph = get_price_history([INDEX_TICKER], days, is_benchmark=True,
                                          end_date=end_date)
        _frames = [frame for frame in (_holdings_ph, _benchmark_ph) if not frame.empty]
        ph_df = pd.concat(_frames, ignore_index=True) if _frames else pd.DataFrame(
            columns=["date", "ticker", "close"]
        )

        # Fix 3: Drop newly listed tickers that have insufficient history
        # (fewer than 5 trading days of data in the selected window)
        if not ph_df.empty:
            ticker_counts    = ph_df.groupby("ticker")["date"].count()
            # A ticker is "newly listed" only if it has data for less than 25% of
            # the requested window — this avoids dropping real tickers on short
            min_days         = max(1, window_days // 4)
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
            #idx_pivot = idx_raw[~idx_raw.index.duplicated(keep="last")].sort_index()
            idx_pivot = idx_raw[~idx_raw.index.duplicated(keep="last")].sort_index().ffill().bfill()
        else:
            idx_pivot = pd.Series(dtype=float)

        # ── SMA50 / SMA200 for INDEX_TICKER (Nifty50) ─────────────────────────
        # SMA200 needs 200 *trading* days before the first plotted date, so fetch
        # the chart window + ~320 calendar days, compute on the full series, then
        # slice to the chart window.
        _sma_lookback = days + 320
        _sma_ph = get_price_history([INDEX_TICKER], _sma_lookback, True)
        _sma50_chart  = pd.Series(dtype=float)
        _sma200_chart = pd.Series(dtype=float)
        _idx_last    = None
        _sma50_last  = None
        if not _sma_ph.empty:
            _idx_full = (_sma_ph[_sma_ph["ticker"] == INDEX_TICKER]
                         .set_index("date")["close"]
                         .pipe(lambda s: s[~s.index.duplicated(keep="last")])
                         .sort_index())
            _sma50_full  = _idx_full.rolling(window=50,  min_periods=50).mean()
            _sma200_full = _idx_full.rolling(window=200, min_periods=200).mean()
            _chart_start = idx_pivot.index[0] if not idx_pivot.empty else _idx_full.index[0]
            _chart_end   = pd.Timestamp(end_date) if end_date else _idx_full.index[-1]
            _sma50_chart  = _sma50_full[(_sma50_full.index   >= _chart_start)
                                        & (_sma50_full.index   <= _chart_end)].dropna()
            _sma200_chart = _sma200_full[(_sma200_full.index >= _chart_start)
                                         & (_sma200_full.index <= _chart_end)].dropna()
            _idx_last    = _idx_full.iloc[-1]   if not _idx_full.empty   else None
            _sma50_last  = _sma50_full.iloc[-1] if not _sma50_full.empty else None
            # benchmark_prices lags ~3 days. Best effort: if ticker_prices has
            # newer closes for the index, use the latest one for the banner
            # (SMA50 itself stays at its last computed value).
            try:
                _live = get_price_history([INDEX_TICKER], 15)
                if not _live.empty and not _idx_full.empty:
                    _live = _live[_live["date"] > _idx_full.index.max()].sort_values("date")
                    if not _live.empty:
                        _idx_last = _live["close"].iloc[-1]
            except Exception:
                pass


        # Only pass holding tickers to portfolio value (exclude index)
        ph_holdings = ph_df[ph_df["ticker"] != INDEX_TICKER] if not ph_df.empty else ph_df
        port_ser = compute_portfolio_value_series(holdings_qty_hist, ph_holdings) if not ph_holdings.empty else pd.Series(dtype=float)

        # Deduplicate port_ser index as a final safety net
        if not port_ser.empty:
            port_ser = port_ser[~port_ser.index.duplicated(keep="last")].sort_index()

        # benchmark table lags ~3 days: skip the missing days by carrying the
        # last known index SMA50/SMA200 forward to today's date
        if not port_ser.empty:
            _today = pd.Timestamp(end_date) if end_date else pd.Timestamp.today().normalize()
            _carry_idx = port_ser.index.union(
                pd.bdate_range(port_ser.index.max(), _today))
            _sma50_chart  = _carry_forward(_sma50_chart,  _carry_idx)
            _sma200_chart = _carry_forward(_sma200_chart, _carry_idx)

        # Align index pivot to port_ser dates to guarantee clean % returns
        if not idx_pivot.empty and not port_ser.empty:
            idx_pivot = idx_pivot.reindex(port_ser.index).ffill().bfill()

        # ── Portfolio SMA50 / SMA200 ───────────────────────────────────────────
        # Same idea as the index: build the portfolio value series over the
        # extended lookback, take rolling means, then slice to the chart window.
        _port_sma50_chart  = pd.Series(dtype=float)
        _port_sma200_chart = pd.Series(dtype=float)
        if not port_ser.empty:
            _full_syms = [s for s in tickers_all if s in holdings_qty_hist]
            _port_full_ph = get_price_history(_full_syms, _sma_lookback, end_date=end_date)
            if not _port_full_ph.empty:
                _port_full_ph = (_port_full_ph
                                 .sort_values(["ticker", "date"])
                                 .drop_duplicates(subset=["date", "ticker"], keep="last"))
                _port_full = compute_portfolio_value_series(holdings_qty_hist, _port_full_ph)
                _port_full = _port_full[~_port_full.index.duplicated(keep="last")].sort_index()
                _p50  = _port_full.rolling(window=50,  min_periods=50).mean()
                _p200 = _port_full.rolling(window=200, min_periods=200).mean()
                _port_sma50_chart  = _p50[_p50.index   >= port_ser.index[0]].dropna()
                _port_sma200_chart = _p200[_p200.index >= port_ser.index[0]].dropna()

    # ── SMA50 TREND BANNERS (NIFTYBEES, GOLDBEES, SILVERBEES) — 3 columns ─────
    _banner_cols = st.columns(1 + len(TREND_BANNER_TICKERS))
    with _banner_cols[0]:
        if _idx_last is not None and _sma50_last is not None:
            _render_trend_banner(INDEX_TICKER, _idx_last, _sma50_last,
                                 "Market Trend", panic=True)
        else:
            st.caption(f"⚠️ {INDEX_TICKER}: not enough price history to compute SMA50.")
    for _col, (_t_sym, _t_label) in zip(_banner_cols[1:], TREND_BANNER_TICKERS.items()):
        with _col:
            _t_last, _t_sma = _trend_snapshot(_t_sym)
            if _t_last is None:
                st.caption(f"⚠️ {_t_sym}: not enough price history to compute SMA50.")
            else:
                _render_trend_banner(_t_sym, _t_last, _t_sma, _t_label)

    if not port_ser.empty:
        port_ret = compute_period_return(port_ser)
        idx_ret  = compute_period_return(idx_pivot) if not idx_pivot.empty else 0.0
        alpha    = port_ret - idx_ret

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric(f"Portfolio ({sel_period})", f"{port_ret:+.2f}%")
        mc2.metric(f"{INDEX_TICKER} ({sel_period})",  f"{idx_ret:+.2f}%")
        mc3.metric("Alpha", f"{alpha:+.2f}%",
                   delta_color="normal" if alpha >= 0 else "inverse")
        if _sma50_last is not None:
            mc4.metric(f"{INDEX_TICKER} SMA50",
                       f"₹{_sma50_last:,.2f}",
                       f"{(_idx_last - _sma50_last) / _sma50_last * 100:+.2f}% vs price",
                       delta_color="normal" if _idx_last >= _sma50_last else "inverse")

        # ── Build normalised series (all rebased to 100 at window start) ──────
        port_norm = port_ser / port_ser.iloc[0] * 100 if port_ser.iloc[0] else port_ser

        idx_norm = pd.Series(dtype=float)
        idx_aligned = pd.Series(dtype=float)
        if not idx_pivot.empty:
            _idx_for_chart = idx_pivot.reindex(port_ser.index).ffill().bfill()
            if not _idx_for_chart.empty and pd.notna(_idx_for_chart.iloc[0]) and _idx_for_chart.iloc[0] != 0:
                idx_aligned = _idx_for_chart
                idx_norm = idx_aligned / idx_aligned.iloc[0] * 100

        # Index SMAs share the index's base value so they sit on the same scale.
        _idx_base = (idx_aligned.iloc[0]
                     if not idx_aligned.empty and pd.notna(idx_aligned.iloc[0]) and idx_aligned.iloc[0] != 0
                     else None)
        def _norm_sma(sma: pd.Series, base) -> pd.Series:
            if sma.empty or not base:
                return pd.Series(dtype=float)
            return sma.reindex(port_ser.index).ffill().dropna() / base * 100

        _port_base = port_ser.iloc[0] if port_ser.iloc[0] else None
        port_sma50_norm  = _norm_sma(_port_sma50_chart,  _port_base)
        port_sma200_norm = _norm_sma(_port_sma200_chart, _port_base)
        idx_sma50_norm   = _norm_sma(_sma50_chart,       _idx_base)
        idx_sma200_norm  = _norm_sma(_sma200_chart,      _idx_base)

        # (label, series, colour, dash, width, checkbox key)
        _series_defs = [
            ("Portfolio",        port_norm,        COLORS[0], "solid",   2.0, "mv_port"),
            ("Portfolio SMA50",  port_sma50_norm,  "#38bdf8", "dot",     1.5, "mv_port_sma50"),
            ("Portfolio SMA200", port_sma200_norm, "#1e3a8a", "dashdot", 1.5, "mv_port_sma200"),
            ("Nifty50",          idx_norm,         COLORS[1], "dash",    1.5, "mv_idx"),
            ("Nifty50 SMA50",    idx_sma50_norm,   "#f59e0b", "dot",     1.5, "mv_idx_sma50"),
            ("Nifty50 SMA200",   idx_sma200_norm,  "#92400e", "dashdot", 1.5, "mv_idx_sma200"),
        ]

        # ── Show / hide checkboxes ─────────────────────────────────────────────
        _chk_cols = st.columns(len(_series_defs))
        _visible = []
        for _col, (_label, _ser, _clr, _dash, _w, _key) in zip(_chk_cols, _series_defs):
            _no_data = _ser.empty
            _on = _col.checkbox(
                _label, value=True, key=_key, disabled=_no_data,
                help="Not enough history in the database for this line." if _no_data else None,
            )
            if _on and not _no_data:
                _visible.append((_label, _ser, _clr, _dash, _w))

        fig_trend = go.Figure()
        for _label, _ser, _clr, _dash, _w in _visible:
            fig_trend.add_trace(go.Scatter(
                x=_ser.index, y=_ser.values, name=_label,
                line=dict(color=_clr, width=_w, dash=_dash),
                hovertemplate=f"{_label}: %{{y:.2f}}<extra></extra>",
            ))

        # Y-axis floor = lowest value among the lines currently shown
        _y_mins = [s.min() for _, s, *_ in _visible if not s.empty]
        _y_floor = min(_y_mins) * 0.995 if _y_mins else 0

        fig_trend.update_layout(
            height=360, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.08),
            hovermode="x unified",
            xaxis=dict(showgrid=False),
            yaxis=dict(title="Indexed (base=100)", showgrid=True, gridcolor="#f0f0f0", range=[_y_floor, None]),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No price history data available for selected period.")

    st.divider()

    # ── PORTFOLIO TABLE (after movement) ─────────────────────────────────────
    st.subheader("Portfolio Holdings")

    # ── Absolute Return: driven by table checkboxes ───────────────────────────
    all_syms = df["Symbol"].tolist()

    # Initialise session state — default all selected
    if "abs_ret_selected" not in st.session_state:
        st.session_state.abs_ret_selected = all_syms.copy()

    # Select All / Clear All buttons
    _btn_col1, _btn_col2, _spacer = st.columns([1, 1, 8])
    with _btn_col1:
        if st.button("✅ Select All", use_container_width=True, key="abs_ret_select_all"):
            st.session_state.abs_ret_selected = all_syms.copy()
            st.rerun()
    with _btn_col2:
        if st.button("☐ Clear All", use_container_width=True, key="abs_ret_clear_all"):
            st.session_state.abs_ret_selected = []
            st.rerun()

    abs_ret_selected = st.session_state.abs_ret_selected

    # ── Compute & display absolute return for selected stocks ─────────────────
    if abs_ret_selected:
        sel_df = df[df["Symbol"].isin(abs_ret_selected)]
        sel_invested   = (sel_df["Total Qty"] * sel_df.apply(
            lambda r: ticker_data.get(r["Symbol"], {}).get("avg_cost", 0), axis=1
        )).sum()
        sel_curr_value = sel_df["Curr Value"].sum()
        sel_abs_pnl    = sel_curr_value - sel_invested
        sel_abs_pct    = (sel_abs_pnl / sel_invested * 100) if sel_invested else 0.0

        # Estimated annual dividend = current value × dividend yield (from tickers table)
        sel_div_amt = (sel_df["Curr Value"] * sel_df["Div Yield%"].fillna(0) / 100).sum()
        sel_div_pct = (sel_div_amt / sel_curr_value * 100) if sel_curr_value else 0.0

        _ar1, _ar2, _ar3, _ar4, _ar5 = st.columns(5)
        _ar1.metric("Selected Holdings", len(abs_ret_selected))
        _ar2.metric("Invested (Selected)", f"₹{sel_invested:,.0f}")
        _ar3.metric("Current Value (Selected)", f"₹{sel_curr_value:,.0f}")
        _ar4.metric(
            "Absolute Return",
            f"₹{sel_abs_pnl:+,.0f}",
            f"{sel_abs_pct:+.2f}%",
            delta_color="normal" if sel_abs_pnl >= 0 else "inverse",
        )
        _ar5.metric(
            "Est. Annual Dividend (Selected)",
            f"₹{sel_div_amt:,.0f}",
            f"{sel_div_pct:.2f}% yield",
            delta_color="off",
            help="Estimated from each holding's current value × its dividend yield. "
                 "Not the dividends actually received.",
        )
    else:
        st.info("Select at least one stock above to see the Absolute Return.")

    st.divider()

    _display_cols_quick = (
            ["Inclusion", "Symbol", "Industry", "Market Cap", "Portfolio Wt%", "Total Qty"]
            + account_names
            #+ ["Last Price", "Curr Value", "Today's Gain", "Today Gain%", "Day Chg%"]
            + ["Last Price", "Curr Value", "Today's Gain", "Day Chg%", "P&L ₹", "P&L%"]
    )

    # Add the Send checkbox column — True if symbol is in abs_ret_selected
    df_display = df.copy()
    df_display["Inclusion"] = df_display["Symbol"].isin(st.session_state.abs_ret_selected)
    _display_cols_quick = [c for c in _display_cols_quick if c in df_display.columns]

    def _row_style_quick(row):
        if row["Symbol"] == winner_sym:
            return ["background-color: rgba(22,163,74,0.12)"] * len(row)
        if row["Symbol"] == loser_sym:
            return ["background-color: rgba(220,38,38,0.10)"] * len(row)
        return [""] * len(row)

    _col_cfg_quick = {
        "Inclusion":          st.column_config.CheckboxColumn(label="Inclusion", help="Include in Absolute Return"),
        "Last Price":    st.column_config.NumberColumn(format="₹%.2f"),
        "Curr Value":    st.column_config.NumberColumn(format="₹%.0f"),
        "Today's Gain":  st.column_config.NumberColumn(format="₹%.0f"),
        "Today Gain%":   st.column_config.NumberColumn(format="%.2f%%"),
        "Day Chg%":      st.column_config.NumberColumn(format="%.2f%%"),
        "P&L ₹":         st.column_config.NumberColumn(format="₹%.0f"),
        "P&L%":          st.column_config.NumberColumn(format="%.2f%%"),
        "Portfolio Wt%": st.column_config.NumberColumn(format="%.2f%%"),
    }
    edited_df = st.data_editor(
        df_display[_display_cols_quick],
        column_config=_col_cfg_quick,
        use_container_width=True,
        hide_index=True,
        disabled=[c for c in _display_cols_quick if c != "Inclusion"],
        key="holdings_table_editor",
    )
    # Sync checkbox selections back to session state
    if edited_df is not None:
        selected_syms = df_display.loc[edited_df["Inclusion"].values, "Symbol"].tolist()
        if selected_syms != st.session_state.abs_ret_selected:
            st.session_state.abs_ret_selected = selected_syms
            st.rerun()

    st.divider()

    # ── INDUSTRY ORIENTATION TABLE ────────────────────────────────────────────
    st.subheader("Industry Orientation")

    # Pre-fetch price history for the fixed period columns (7D, 1M, 3M, 6M, 1Y)
    _ORIENT_PERIODS = [("7D", 7), ("1M", 30), ("3M", 90),  ("4M", 120),  ("5M", 150), ("6M", 180), ("1Y", 365)]

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
            "Wt%":            round((idf["Curr Value"].sum() / total_value * 100),1),
            "Today's Gain ₹": round(idf["Today's Gain"].sum(),0),
            "P&L ₹":          round(idf["P&L ₹"].sum(), 0),
            "Avg Day Chg%":   round(idf["Day Chg%"].mean(), 2),
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
            "P&L ₹":          st.column_config.NumberColumn(format="₹%.0f"),
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

    ph_strip = get_price_history(tickers_all, days, end_date=end_date)
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

    # ── Executed SQL (debug) ──────────────────────────────────────────────────
    _render_sql_log(_sql_slot)