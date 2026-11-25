"""
portfolio.py
"""
import os
import datetime
import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go

from ui.pages.html_utils import render_trend_banner as _render_trend_banner, inject_portfolio_css
from ui.pages.label_utils import label
from ui.pages.trade.portfolio_capsules.kite_portfolio import render_kite_portfolio
from ui.pages.trade.portfolio_capsules.manual_feed import render_manual_feed, render_file_upload
from ui.pages.trade.portfolio_capsules.kite_tradebook import _analyze_tradebook, _render_trade_pnl

from utils.db_utils import (
    get_industry_meta as _industry_meta,
    fetch_price_history,
    fetch_ticker_metadata,
)
from ui.pages.one_ticker_common import COLORS
from utils.time_utils import PERIODS
from utils.file_utils import load_json_dict, save_json_atomic
from utils.debug_utils import init_sql_log, log_sql as _log_sql, render_sql_log as _render_sql_log

num_col = st.column_config.NumberColumn

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

def load_portfolio_settings(path: str = SETTINGS_PATH) -> dict:
    return load_json_dict(path, base_file=__file__)

def save_portfolio_settings(settings: dict, path: str = SETTINGS_PATH) -> None:
    try:
        save_json_atomic(settings, path, base_file=__file__)
    except OSError as e:
        st.warning(f"Could not save portfolio settings: {e}")

# ==============================================================================
# HISTORICAL DATA HELPERS
# ==============================================================================

def get_price_history(nse_symbols: list[str], days: int, is_benchmark: bool = False,
                      end_date: "datetime.date | None" = None) -> pd.DataFrame:
    if not nse_symbols:
        return pd.DataFrame()
    try:
        df = fetch_price_history( nse_symbols, days,
            is_benchmark=is_benchmark, end_date=end_date, log_sql=_log_sql,)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as exc:
        print(f"[Portfolio] Price history query failed  {exc}")
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


def compute_portfolio_value_series(
        holdings_qty: dict, price_df: pd.DataFrame,
) -> pd.Series:
    """
    Given {ticker: qty} and long-form price_df (date/ticker/close),
    return daily portfolio value indexed by date.
    """
    if price_df.empty:
        return pd.Series(dtype=float)

    pivot = price_df.pivot_table(
        index="date", columns="ticker", values="close", aggfunc="last",
    ).sort_index()

    # Forward-fill internal gaps and backward-fill leading gaps.
    prices = pivot.ffill().bfill().fillna(0)

    # Align quantities with available ticker columns.
    quantities = pd.to_numeric(
        pd.Series(holdings_qty),
        errors="coerce",
    ).reindex(prices.columns).fillna(0)

    # Vectorized portfolio valuation.
    return prices.mul(quantities, axis=1).sum(axis=1)

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




# ==============================================================================
# MAIN RENDER
# ==============================================================================



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

def render():
    st.set_page_config(page_title="Portfolio", layout="wide", page_icon="📊")

    # ── SQL debug log: reset each run, drawn into this slot at the end ───────
    st.session_state["_sql_log"] = []
    init_sql_log()
    _sql_slot = st.empty()

    # CSS and HTML presentation are centralized in html_utils.py.
    inject_portfolio_css()

    # ── Portfolio source selector / capsule dispatch ──────────────────────────
    ticker_data = None
    account_names = []
    _SRC_LABELS = {"current": "📋 Current Holdings", "manual": "✏️ Manual", "upload": "📂 File Upload"}
    _src_label = st.radio("Portfolio source", list(_SRC_LABELS.values()), horizontal=True,
                          key="holdings_source_radio", label_visibility="collapsed")
    input_mode = next(k for k, v in _SRC_LABELS.items() if v == _src_label)
    excl_set = {s.upper() for s in (getattr(st.session_state, "excluded_symbols", None) or set())}
    _upload_track = None

    if input_mode == "current":
        ticker_data, account_names = render_kite_portfolio()
    elif input_mode == "manual":
        ticker_data, account_names = render_manual_feed()
    else:
        ticker_data, _upload_track = render_file_upload(_live_price_map, _analyze_tradebook)
        account_names = ["Uploaded"]

    st.session_state["pf_prev_src"] = input_mode

    if ticker_data is None:
        if input_mode == "upload":
            if _upload_track is not None:
                _render_trade_pnl(_upload_track, _live_price_map)
            else:
                st.info(label("upload_no_data"))
        st.stop()

    # ── Load ticker metadata from DuckDB (industry, market_cap, p_e, dividend_yield) ──
    meta_symbols = [s for s in ticker_data.keys() if s.upper() != INDEX_TICKER]
    ticker_meta = {}
    try:
        meta_df = fetch_ticker_metadata(meta_symbols, log_sql=_log_sql)
        for _, r in meta_df.iterrows():
            key = str(r["matched_symbol"]).strip().upper()
            ticker_meta[key] = {
                "industry":       r.get("industry", "Other") or "Other",
                "market_cap":     r.get("market_cap", "Unknown") or "Unknown",
                "market_cap_amt": r.get("market_cap_amt", 0) or 0,
                "p_e":            r.get("p_e", None),
                "dividend_yield": r.get("dividend_yield", 0) or 0,
            }
    except Exception as e:
        st.caption(f"⚠️ Metadata load error: {e}")

    with st.expander("🐛 Metadata Debug", expanded=False):
        st.write("**Kite symbols (tradingsymbol):**", len(ticker_data.keys()))
        st.write("**Matched from tickers table:**", len(ticker_meta.keys()))
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
        _render_trade_pnl(_upload_track, _live_price_map)
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
        avg_cost_map = {
            sym: data.get("avg_cost", 0)
            for sym, data in ticker_data.items()
        }

        sel_invested = (
                sel_df["Total Qty"]
                * sel_df["Symbol"].map(avg_cost_map).fillna(0)
        ).sum()
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
        "Last Price":    num_col(format="₹%.2f"),
        "Curr Value":    num_col(format="₹%.0f"),
        "Today's Gain":  num_col(format="₹%.0f"),
        "Today Gain%":   num_col(format="%.2f%%"),
        "Day Chg%":      num_col(format="%.2f%%"),
        "P&L ₹":         num_col(format="₹%.0f"),
        "P&L%":          num_col(format="%.2f%%"),
        "Portfolio Wt%": num_col(format="%.2f%%"),
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
        lbl: num_col(label=lbl, format="%.1f%%")
        for lbl, _ in _ORIENT_PERIODS
    }
    st.dataframe(
        orient_df,
        column_config={
            "Value (₹)":       num_col(format="₹%.0f"),
            "Wt%":             num_col(format="%.1f%%"),
            "Today's Gain ₹": num_col(format="₹%.0f"),
            "P&L ₹":          num_col(format="₹%.0f"),
            "Avg Day Chg%":    num_col(format="%.2f%%"),
            **_period_col_cfg,
            "Avg P/E":         num_col(format="%.1f"),
            "Avg Div Yield%":  num_col(format="%.2f%%"),
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