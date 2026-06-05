"""
ui/pages/page_portfolio.py  –  Screens › Portfolio Evaluator

Data sources:
  • Local  — DuckDB query over data/ticker/year=YYYY/*.parquet  (partitioned by year)
  • Yahoo  — gentle fetch with 5-s gap between batches to avoid throttling
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ── Inline CSS ────────────────────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');

.input-panel {
    background: #0f0f0f;
    border: 1px solid #222;
    border-radius: 6px;
    padding: 1rem 1.2rem 1.2rem;
    margin-bottom: 0.5rem;
}

.metric-card   { background:#161616;border:1px solid #2a2a2a;border-radius:4px;padding:1.2rem 1.5rem;margin-bottom:.5rem; }
.metric-label  { font-size:.65rem;letter-spacing:.15em;text-transform:uppercase;color:#666;margin-bottom:.25rem; }
.metric-value  { font-size:1.6rem;font-weight:500;color:#e8e0d0;font-family:'DM Serif Display',serif; }
.metric-value.positive { color:#7cfc7c; }
.metric-value.negative { color:#fc7c7c; }

.ticker-badge  { display:inline-block;background:#1e1e1e;border:1px solid #333;border-radius:2px;padding:2px 8px;font-size:.7rem;letter-spacing:.1em;color:#aaa;margin:2px; }
.skipped-list  { font-size:.72rem;color:#555;line-height:1.8; }

/* ── Event table ── */
.evt-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78rem;
    margin-top: 0.5rem;
    background: #ffffff;
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.10);
}
.evt-table th {
    text-align: left;
    border-bottom: 2px solid #e0e0e0;
    padding: 0.55rem 0.9rem;
    color: #1565c0;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
    background: #e3edf9;
    font-size: 0.68rem;
}
.evt-table td {
    padding: 0.38rem 0.9rem;
    border-bottom: 1px solid #f0f0f0;
    color: #222;
    background: #ffffff;
}
.evt-table tr:hover td { background: #fffde7; }
.evt-pos  { color: #1b7e34 !important; font-weight: 600; }
.evt-neg  { color: #c62828 !important; font-weight: 600; }
.evt-na   { color: #bdbdbd !important; }
.evt-section td {
    color: #1565c0 !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.14em;
    padding: 0.5rem 0.9rem 0.2rem !important;
    background: #fff9c4 !important;
    text-transform: uppercase;
    font-weight: 700;
    border-bottom: 1px solid #ffe082 !important;
}
</style>
"""

# Default local parquet path — overridden by app_cfg if present
_DEFAULT_PARQUET_DIR = "data/ticker"

MACRO_OVERLAYS = {
    "NIFTYBEES":   dict(ticker="NIFTYBEES.NS",  color="#f4a26a", dash="dot"),
    "Oil (Brent)": dict(ticker="BZ=F",           color="#f4a26a", dash="dot"),
    "India VIX":   dict(ticker="^INDIAVIX",      color="#c97be8", dash="dashdot"),
    "USD / INR":   dict(ticker="INR=X",           color="#6ae8c9", dash="longdash"),
    "NASDAQ":      dict(ticker="^IXIC",           color="#e86a6a", dash="dash"),
    "S&P 500":     dict(ticker="^GSPC",           color="#6a9fe8", dash="dot"),
    "GOLD":        dict(ticker="GOLDBEES.NS",     color="yellow",  dash="dash"),
}

EVENT_PERIODS = [
    {"name": "FY2018", "section": "Financial Years", "start": "2018-04-01", "end": "2019-03-31"},
    {"name": "FY2019", "section": "Financial Years", "start": "2019-04-01", "end": "2020-03-31"},
    {"name": "FY2020", "section": "Financial Years", "start": "2020-04-01", "end": "2021-03-31"},
    {"name": "FY2021", "section": "Financial Years", "start": "2021-04-01", "end": "2022-03-31"},
    {"name": "FY2022", "section": "Financial Years", "start": "2022-04-01", "end": "2023-03-31"},
    {"name": "FY2023", "section": "Financial Years", "start": "2023-04-01", "end": "2024-03-31"},
    {"name": "FY2024", "section": "Financial Years", "start": "2024-04-01", "end": "2025-03-31"},
    {"name": "FY2025", "section": "Financial Years", "start": "2025-04-01", "end": "2026-03-31"},
    {"name": "Covid Crash",         "section": "Critical Events", "start": "2020-01-17", "end": "2020-03-23"},
    {"name": "Covid V-Recovery",    "section": "Critical Events", "start": "2020-03-23", "end": "2020-08-31"},
    {"name": "Russia-Ukraine War",  "section": "Critical Events", "start": "2022-02-24", "end": "2022-09-30"},
    {"name": "Fed Rate Hike Cycle", "section": "Critical Events", "start": "2022-06-17", "end": "2023-07-31"},
    {"name": "Trump Tariff",        "section": "Critical Events", "start": "2025-01-20", "end": "2025-04-09"},
    {"name": "Iran War",            "section": "Critical Events", "start": "2026-01-01", "end": "2026-05-07"},
    {"name": "Live (YTD FY26)",     "section": "Live",
     "start": "2026-04-01", "end": date.today().strftime("%Y-%m-%d")},
]


# ── Ticker helpers ────────────────────────────────────────────────────────────
def _ensure_ns(ticker: str) -> str:
    """Append .NS to bare NSE tickers; leave indices/FX/already-suffixed alone."""
    if ticker.startswith("^") or "=" in ticker:
        return ticker
    if "." not in ticker:
        return ticker + ".NS"
    return ticker


def _parse_weights(text: str) -> tuple[dict, list[str]]:
    """Parse 'TICKER: weight' lines → auto-add .NS → normalise to sum=100."""
    weights, skipped = {}, []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parts = line.replace(",", "\t").replace(":", " ").split()
            ticker = _ensure_ns(parts[0].rstrip(":"))
            weight = float(parts[1]) if len(parts) > 1 else 1.0
            weights[ticker] = weight
        except Exception:
            skipped.append(line)
    if weights:
        total = sum(weights.values())
        if total > 0:
            weights = {t: w * (100.0 / total) for t, w in weights.items()}
    return weights, skipped


# ── LOCAL fetch via DuckDB ────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_local(
        tickers: tuple[str, ...],
        start: str,
        end: str,
        parquet_dir: str,
) -> pd.DataFrame:
    """
    Query the year-partitioned parquet store with DuckDB.
    Schema: Date (date), Close (double), Ticker (varchar), year (int64).
    Returns a wide Close DataFrame indexed by Date, one column per ticker.
    """
    try:
        import duckdb
    except ImportError:
        st.error("duckdb not installed. Run: pip install duckdb")
        return pd.DataFrame()

    if not Path(parquet_dir).exists():
        st.warning(f"Local data directory not found: `{parquet_dir}`")
        return pd.DataFrame()

    start_year = pd.Timestamp(start).year
    end_year   = pd.Timestamp(end).year

    # Build year-filter glob so DuckDB prunes partitions
    year_parts = " OR ".join(f"year = {y}" for y in range(start_year, end_year + 1))
    glob_path  = f"{parquet_dir}/year=*/*.parquet"

    ticker_list = ", ".join(f"'{t}'" for t in tickers)

    sql = f"""
        SELECT Date, Ticker, Close
        FROM read_parquet('{glob_path}', hive_partitioning = true)
        WHERE ({year_parts})
          AND Ticker IN ({ticker_list})
          AND Date >= '{start}'
          AND Date <= '{end}'
        ORDER BY Date
    """
    try:
        con = duckdb.connect()
        df  = con.execute(sql).df()
        con.close()
    except Exception as e:
        st.error(f"DuckDB query failed: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"])
    wide = df.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="last")
    wide.index.name = None
    wide.columns.name = None
    return wide.sort_index()


# ── Load all available tickers from DuckDB (sqls/init.sql) ───────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _load_all_tickers() -> pd.DataFrame:
    """
    Load ticker mapping from DuckDB using sqls/init.sql.
    Returns DataFrame with columns: Yahoo Symbol, nse_symbol.
    """
    try:
        import duckdb
    except ImportError:
        return pd.DataFrame(columns=["Yahoo Symbol", "nse_symbol"])

    init_sql_path = Path("sqls/init.sql")
    if not init_sql_path.exists():
        return pd.DataFrame(columns=["Yahoo Symbol", "nse_symbol"])

    try:
        con = duckdb.connect()
        init_sql = init_sql_path.read_text()
        con.execute(init_sql)
        df = con.execute(
            'SELECT "Yahoo Symbol", "nse_symbol" FROM tickers ORDER BY "nse_symbol"'
        ).df()
        con.close()
        return df
    except Exception:
        return pd.DataFrame(columns=["Yahoo Symbol", "nse_symbol"])



def _parse_close_yf(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            return raw["Close"].copy()
        return pd.DataFrame()
    # Single-ticker download returns flat columns
    df = raw[["Close"]].copy()
    df.columns = [tickers[0]]
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_yahoo_single_batch(
        tickers: tuple[str, ...],
        start: str,
        end: str,
) -> pd.DataFrame:
    """Download one batch; falls back to period='max' for any missing tickers."""
    import yfinance as yf

    ticker_list = list(tickers)
    start_ts    = pd.Timestamp(start)
    ext_start   = (start_ts - pd.DateOffset(years=2)).strftime("%Y-%m-%d")

    raw      = yf.download(ticker_list, start=ext_start, end=end,
                           auto_adjust=True, progress=False)
    prices   = _parse_close_yf(raw, ticker_list).dropna(how="all")

    missing = (
        [t for t in ticker_list
         if t not in prices.columns or prices[t].isna().all()]
        if not prices.empty else ticker_list
    )

    if missing:
        raw2    = yf.download(missing, period="max", auto_adjust=True, progress=False)
        prices2 = _parse_close_yf(raw2, missing).dropna(how="all")
        if not prices2.empty:
            prices2 = prices2.loc[:end]
            for t in prices2.columns:
                if prices2[t].isna().all():
                    continue
                if prices.empty:
                    prices = prices2[[t]].copy()
                elif t not in prices.columns:
                    prices[t] = prices2[t]
                else:
                    prices[t] = prices[t].combine_first(prices2[t])

    if prices.empty:
        return pd.DataFrame()
    prices = prices.ffill().bfill()
    return prices.loc[start_ts:].dropna(how="all")


@st.cache_data(ttl=3600, show_spinner="Fetching from Yahoo Finance…")
def _fetch_yahoo(
        tickers1: tuple[str, ...],
        tickers2: tuple[str, ...],
        bench_and_macro: tuple[str, ...],
        start: str,
        end: str,
) -> pd.DataFrame:
    """
    Fetch three batches with pauses between them to avoid Yahoo throttling.
    Returns a single merged wide DataFrame.
    """
    import time

    frames = []
    if tickers1:
        frames.append(_fetch_yahoo_single_batch(tickers1, start, end))
    if tickers2:
        time.sleep(5)
        frames.append(_fetch_yahoo_single_batch(tickers2, start, end))
    if bench_and_macro:
        time.sleep(2)
        frames.append(_fetch_yahoo_single_batch(bench_and_macro, start, end))

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for f in frames[1:]:
        new_cols = [c for c in f.columns if c not in merged.columns]
        if new_cols:
            merged = merged.join(f[new_cols], how="outer")
    return merged


# ── Portfolio maths ───────────────────────────────────────────────────────────
_DEBT_YIELD = 0.07  # 7% p.a. proxy for unlisted / pre-listing periods


def _build_series(weights: dict, prices: pd.DataFrame) -> tuple[pd.Series | None, list[str]]:
    missing, valid = [], {}
    for t, w in weights.items():
        if t not in prices.columns or prices[t].dropna().empty:
            missing.append(t)
        else:
            valid[t] = w
    if not valid:
        return None, missing

    total      = sum(valid.values())
    port_start = prices.index[0]
    daily_rate = (1 + _DEBT_YIELD) ** (1 / 252) - 1

    components = []
    for t, w in valid.items():
        col        = prices[t].ffill()
        first_real = col.dropna().index[0]

        if first_real > port_start:
            # Back-fill pre-listing days with 7% p.a. debt proxy.
            # Anchor backwards from the first real price so the series
            # joins seamlessly on listing day.
            pre_idx      = prices.index[prices.index < first_real]
            n_pre        = len(pre_idx)
            anchor       = col.loc[first_real]
            back_values  = anchor / (1 + daily_rate) ** np.arange(n_pre, 0, -1)
            debt_series  = pd.Series(back_values, index=pre_idx)
            col          = pd.concat([debt_series, col.loc[first_real:]])

        components.append(col * (w / total))

    series = sum(components)
    first  = series.dropna().iloc[0] if not series.dropna().empty else 1
    return (series / first * 100).dropna(), missing


def _pct_return(s: pd.Series) -> float:
    return (s.iloc[-1] / s.iloc[0] - 1) * 100


# ═════════════════════════════════════════════════════════════════════════════
# render()
# ═════════════════════════════════════════════════════════════════════════════
def render():
    st.header("📊 Portfolio Evaluator")
    st.markdown(_CSS, unsafe_allow_html=True)

    try:
        import plotly.graph_objects as go
    except ImportError:
        st.error("plotly not installed. Run: pip install plotly")
        return

    app_cfg     = st.session_state.get("app_cfg", {})
    parquet_dir = app_cfg.get("TICKER_PARQUET_DIR", _DEFAULT_PARQUET_DIR)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Portfolio inputs: two equal panels side by side
    # ══════════════════════════════════════════════════════════════════════════
    p1_col, p2_col = st.columns(2, gap="medium")

    with p1_col:
        st.markdown('<div class="input-panel">', unsafe_allow_html=True)
        st.markdown("##### 🟡 Portfolio 1")
        p1_name = st.text_input("Name", "Portfolio 1", key="p1_name")

        # ── Three-way loader for Portfolio 1 ──────────────────────────────────
        load_mode1 = st.radio(
            "Load from",
            ["✏️ Manual", "📂 File", "📋 Current Holdings"],
            horizontal=True,
            key="p1_load_mode",
            label_visibility="collapsed",
        )

        if load_mode1 == "📂 File":
            # Option 1: CSV file (weights_train_*.csv or any uploaded CSV)
            base_out = app_cfg.get("GA_OUT_DIR", "outputs")
            wfiles   = sorted(Path(base_out).rglob("weights_train_*.csv")) if Path(base_out).exists() else []
            uploaded_csv1 = st.file_uploader("Upload weights CSV", type=["csv"], key="p1_csv_upload")
            if uploaded_csv1 is not None:
                # Track file name so we only reload once per new upload
                if st.session_state.get("_p1_csv_name") != uploaded_csv1.name:
                    try:
                        wdf   = pd.read_csv(uploaded_csv1)
                        lines = "\n".join(
                            f"{row['ticker']}: {row['weight']:.6f}"
                            for _, row in wdf.iterrows() if row.get("weight", 0) > 0.001
                        )
                        st.session_state["p1_text"] = lines          # write directly into widget key
                        st.session_state["_p1_csv_name"] = uploaded_csv1.name
                        st.session_state["_p1_load_msg"] = f"✅ Loaded {len(lines.splitlines())} tickers from **{uploaded_csv1.name}**"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not parse CSV: {e}")
                if "_p1_load_msg" in st.session_state:
                    st.success(st.session_state["_p1_load_msg"])
            elif wfiles:
                chosen_wf = st.selectbox(
                    "Or pick a GA run file", ["— none —"] + [str(f) for f in wfiles], key="ga_wf"
                )
                if chosen_wf != "— none —" and st.button("Load weights", key="load_ga"):
                    wdf   = pd.read_csv(chosen_wf)
                    lines = "\n".join(
                        f"{row['ticker']}: {row['weight']:.6f}"
                        for _, row in wdf.iterrows() if row["weight"] > 0.001
                    )
                    st.session_state["p1_text"] = lines
                    st.session_state["_p1_load_msg"] = f"✅ Loaded {len(lines.splitlines())} tickers from GA run"
                    st.rerun()

        elif load_mode1 == "📋 Current Holdings":
            # Option 2: live holdings from session state
            if "all_holdings" not in st.session_state or not st.session_state.all_holdings:
                st.warning("Holdings not loaded. Please go to Setup and click 'Fetch Holdings'.")
            else:
                # Only rebuild when mode is freshly switched or holdings changed
                if st.session_state.get("_p1_holdings_loaded") is not True:
                    all_holdings = st.session_state.all_holdings
                    excl_set     = {s.upper() for s in (getattr(st.session_state, "excluded_symbols", None) or set())}
                    account_names = list(all_holdings.keys())
                    ticker_data   = {}
                    for acc_name, holdings in all_holdings.items():
                        for h in holdings:
                            sym = h["tradingsymbol"]
                            if sym.upper() in excl_set:
                                continue
                            if sym not in ticker_data:
                                ticker_data[sym] = {
                                    "last_price":  h.get("last_price", 0),
                                    "close_price": h.get("close_price", 0),
                                    "avg_cost":    h.get("average_price", 0),
                                    **{an: 0 for an in account_names},
                                }
                            ticker_data[sym][acc_name] += h.get("quantity", 0)

                    holding_lines = []
                    for sym, td in ticker_data.items():
                        total_qty  = sum(td.get(an, 0) for an in account_names)
                        val        = total_qty * (td["last_price"] or td["close_price"] or 1)
                        if val > 0:
                            holding_lines.append((sym, val))
                    if holding_lines:
                        total_val = sum(v for _, v in holding_lines)
                        lines = "\n".join(
                            f"{_ensure_ns(sym)}: {val/total_val*100:.4f}"
                            for sym, val in sorted(holding_lines, key=lambda x: -x[1])
                        )
                        st.session_state["p1_text"] = lines
                        st.session_state["_p1_holdings_loaded"] = True
                        st.session_state["_p1_load_msg"] = f"✅ Loaded {len(holding_lines)} holdings from current portfolio"
                        st.rerun()
                    else:
                        st.warning("No holdings with value found.")
                if "_p1_load_msg" in st.session_state:
                    st.success(st.session_state["_p1_load_msg"])
        else:
            # Manual mode — clear the stale load markers so switching back re-triggers
            st.session_state.pop("_p1_holdings_loaded", None)
            st.session_state.pop("_p1_csv_name", None)
            st.session_state.pop("_p1_load_msg", None)

        # Only pass value= when the widget key isn't already owned by Streamlit.
        # Loaders write st.session_state["p1_text"] then call st.rerun(), which
        # pre-populates the widget; passing value= on top would raise an exception.
        _p1_ta_kwargs = {} if "p1_text" in st.session_state else {
            "value": st.session_state.get("_p1_text_val", "")
        }
        p1_text = st.text_area(
            "Tickers & weights  (TICKER: weight, one per line)",
            height=200, key="p1_text",
            placeholder="RELIANCE: 2.0\nINFY: 1.5\nHDFCBANK: 1.0",
            **_p1_ta_kwargs,
        )
        st.session_state["_p1_text_val"] = p1_text
        st.markdown('</div>', unsafe_allow_html=True)

    with p2_col:
        st.markdown('<div class="input-panel">', unsafe_allow_html=True)
        st.markdown("##### 🟢 Portfolio 2")
        p2_name = st.text_input("Name", "Portfolio 2", key="p2_name")

        # ── Three-way loader for Portfolio 2 ──────────────────────────────────
        load_mode2 = st.radio(
            "Load from",
            ["✏️ Manual", "📂 File", "📋 Current Holdings"],
            horizontal=True,
            key="p2_load_mode",
            label_visibility="collapsed",
        )

        if load_mode2 == "📂 File":
            uploaded_csv2 = st.file_uploader("Upload weights CSV", type=["csv"], key="p2_csv_upload")
            if uploaded_csv2 is not None:
                if st.session_state.get("_p2_csv_name") != uploaded_csv2.name:
                    try:
                        wdf2  = pd.read_csv(uploaded_csv2)
                        lines2 = "\n".join(
                            f"{row['ticker']}: {row['weight']:.6f}"
                            for _, row in wdf2.iterrows() if row.get("weight", 0) > 0.001
                        )
                        st.session_state["p2_text"] = lines2
                        st.session_state["_p2_csv_name"] = uploaded_csv2.name
                        st.session_state["_p2_load_msg"] = f"✅ Loaded {len(lines2.splitlines())} tickers from **{uploaded_csv2.name}**"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not parse CSV: {e}")
                if "_p2_load_msg" in st.session_state:
                    st.success(st.session_state["_p2_load_msg"])

        elif load_mode2 == "📋 Current Holdings":
            if "all_holdings" not in st.session_state or not st.session_state.all_holdings:
                st.warning("Holdings not loaded. Please go to Setup and click 'Fetch Holdings'.")
            else:
                # Only rebuild when mode is freshly switched or holdings changed
                if st.session_state.get("_p2_holdings_loaded") is not True:
                    all_holdings2 = st.session_state.all_holdings
                    excl_set2     = {s.upper() for s in (getattr(st.session_state, "excluded_symbols", None) or set())}
                    account_names2 = list(all_holdings2.keys())
                    ticker_data2   = {}
                    for acc_name, holdings in all_holdings2.items():
                        for h in holdings:
                            sym = h["tradingsymbol"]
                            if sym.upper() in excl_set2:
                                continue
                            if sym not in ticker_data2:
                                ticker_data2[sym] = {
                                    "last_price":  h.get("last_price", 0),
                                    "close_price": h.get("close_price", 0),
                                    "avg_cost":    h.get("average_price", 0),
                                    **{an: 0 for an in account_names2},
                                }
                            ticker_data2[sym][acc_name] += h.get("quantity", 0)

                    holding_lines2 = []
                    for sym, td in ticker_data2.items():
                        total_qty  = sum(td.get(an, 0) for an in account_names2)
                        val        = total_qty * (td["last_price"] or td["close_price"] or 1)
                        if val > 0:
                            holding_lines2.append((sym, val))
                    if holding_lines2:
                        total_val2 = sum(v for _, v in holding_lines2)
                        lines2 = "\n".join(
                            f"{_ensure_ns(sym)}: {val/total_val2*100:.4f}"
                            for sym, val in sorted(holding_lines2, key=lambda x: -x[1])
                        )
                        st.session_state["p2_text"] = lines2
                        st.session_state["_p2_holdings_loaded"] = True
                        st.session_state["_p2_load_msg"] = f"✅ Loaded {len(holding_lines2)} holdings from current portfolio"
                        st.rerun()
                    else:
                        st.warning("No holdings with value found.")
            if "_p2_load_msg" in st.session_state:
                st.success(st.session_state["_p2_load_msg"])

        else:
            # Manual mode — clear stale load markers so switching back re-triggers
            st.session_state.pop("_p2_holdings_loaded", None)
            st.session_state.pop("_p2_load_msg", None)
            st.session_state.pop("_p2_csv_name", None)

        # Only pass value= when the widget key isn't already owned by Streamlit.
        # Loaders write st.session_state["p2_text"] then call st.rerun(), which
        # pre-populates the widget; passing value= on top would raise an exception.
        _p2_ta_kwargs = {} if "p2_text" in st.session_state else {
            "value": st.session_state.get("_p2_text_val", "")
        }
        p2_text = st.text_area(
            "Tickers & weights  (TICKER: weight, one per line)",
            height=200, key="p2_text",
            placeholder="NIFTYBEES: 1.0",
            **_p2_ta_kwargs,
        )
        st.session_state["_p2_text_val"] = p2_text
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Available tickers from DuckDB ─────────────────────────────────────────
    all_tickers_df = _load_all_tickers()
    if not all_tickers_df.empty:
        with st.expander("🔍 Browse available tickers", expanded=False):
            search_q = st.text_input(
                "Search NSE symbol or Yahoo symbol",
                placeholder="e.g. RELIANCE or RELIANCE.NS",
                key="ticker_search",
            )
            disp_df = all_tickers_df.copy()
            if search_q:
                mask = (
                        disp_df["nse_symbol"].str.upper().str.contains(search_q.upper(), na=False) |
                        disp_df["Yahoo Symbol"].str.upper().str.contains(search_q.upper(), na=False)
                )
                disp_df = disp_df[mask]
            st.dataframe(
                disp_df.rename(columns={"Yahoo Symbol": "Yahoo Symbol (use this)", "nse_symbol": "NSE Symbol"}),
                use_container_width=True,
                hide_index=True,
                height=min(300, 35 + len(disp_df) * 35),
            )
            st.caption(f"{len(disp_df)} of {len(all_tickers_df)} tickers shown · Use the Yahoo Symbol column in the text areas above")

    # ── Settings row ──────────────────────────────────────────────────────────
    src_col, d1_col, d2_col, bench_col, macro_col = st.columns(
        [1.1, 1, 1, 1, 2], gap="medium"
    )
    with src_col:
        data_source = st.radio(
            "Data source",
            ["🗄 Local parquet", "🌐 Yahoo Finance"],
            key="data_source",
            help=f"Local reads from {parquet_dir}/year=YYYY/*.parquet",
        )
        use_local = data_source.startswith("🗄")

    with d1_col:
        start_date = st.date_input(
            "Start date",
            value=date(2022, 1, 1),
            min_value=date(2000, 1, 1),
            max_value=date.today() - timedelta(days=2),
            key="start_date",
        )
    with d2_col:
        end_date = st.date_input(
            "End date",
            value=date.today(),
            min_value=start_date + timedelta(days=1),
            max_value=date.today(),
            key="end_date",
        )
    with bench_col:
        benchmark_ticker = st.text_input("Benchmark ticker", "^NSEI", key="bench")
    with macro_col:
        selected_macros = st.multiselect(
            "Macro overlays", list(MACRO_OVERLAYS.keys()), key="macros"
        )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Parse weights
    # ══════════════════════════════════════════════════════════════════════════
    weights1, skipped1 = _parse_weights(p1_text)
    weights2, skipped2 = _parse_weights(p2_text)

    if not weights1 and not weights2:
        st.info("Enter portfolio weights above to get started.")
        return

    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — Fetch prices
    # ══════════════════════════════════════════════════════════════════════════
    macro_tickers  = tuple(v["ticker"] for v in MACRO_OVERLAYS.values())
    bench_and_macro = tuple(dict.fromkeys([benchmark_ticker] + list(macro_tickers)))

    if use_local:
        # Local: single DuckDB query for everything — macros / bench may be
        # absent from local store (indices, FX) so we note that gracefully.
        all_tickers = tuple(dict.fromkeys(
            list(weights1) + list(weights2) + list(bench_and_macro)
        ))
        with st.spinner("Reading local parquet data…"):
            prices = _fetch_local(all_tickers, start_str, end_str, parquet_dir)

        if prices.empty:
            st.error(
                f"No data found in `{parquet_dir}` for the selected tickers / date range. "
                "Try switching to Yahoo Finance or extending the date range."
            )
            return

        # For indices / FX not in local store, fall back silently to Yahoo
        yf_needed = [
            t for t in bench_and_macro
            if t not in prices.columns or prices[t].dropna().empty
        ]
        if yf_needed:
            try:
                import yfinance as yf, time
                with st.spinner(f"Fetching {len(yf_needed)} index/FX tickers from Yahoo…"):
                    extra = _fetch_yahoo_single_batch(tuple(yf_needed), start_str, end_str)
                if not extra.empty:
                    for c in extra.columns:
                        if c not in prices.columns:
                            prices[c] = extra[c]
            except Exception:
                pass  # bench / macros just won't appear

    else:
        # Yahoo: gentle three-batch fetch with sleep between batches
        try:
            import yfinance  # noqa: F401
        except ImportError:
            st.error("yfinance not installed. Run: pip install yfinance")
            return

        p1_tickers = tuple(weights1.keys())
        p2_tickers = tuple(weights2.keys())
        prices = _fetch_yahoo(p1_tickers, p2_tickers, bench_and_macro, start_str, end_str)

        if prices.empty:
            st.error("Could not fetch any price data from Yahoo Finance.")
            return

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — Build series + scorecard
    # ══════════════════════════════════════════════════════════════════════════
    portfolio_series1, missing1 = _build_series(weights1, prices)
    portfolio_series2, missing2 = _build_series(weights2, prices)

    bench_series  = None
    missing_bench = False
    if benchmark_ticker in prices.columns and not prices[benchmark_ticker].dropna().empty:
        bs = prices[benchmark_ticker].ffill().dropna()
        bench_series = bs / bs.iloc[0] * 100
    else:
        missing_bench = True

    p1_ret = _pct_return(portfolio_series1) if portfolio_series1 is not None else None
    p2_ret = _pct_return(portfolio_series2) if portfolio_series2 is not None else None
    delta  = (p1_ret - p2_ret) if (p1_ret is not None and p2_ret is not None) else None

    def _fmt(v):
        return f"{v:+.1f}%" if v is not None else "N/A"

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric(p1_name, _fmt(p1_ret))
    mc2.metric(p2_name, _fmt(p2_ret))
    if delta is not None:
        mc3.metric("Alpha (P1 − P2)", _fmt(delta))
    src_label = "🗄 Local" if use_local else "🌐 Yahoo"
    mc4.metric("Data source", src_label)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — Main chart, full width
    # ══════════════════════════════════════════════════════════════════════════
    fig = go.Figure()
    first_date = None

    if portfolio_series1 is not None and not portfolio_series1.empty:
        first_date = portfolio_series1.index[0]
        fig.add_trace(go.Scatter(
            x=portfolio_series1.index, y=portfolio_series1.values,
            name=p1_name, line=dict(color="#e8c96a", width=2.5),
            hovertemplate=f"%{{x|%d %b %Y}}<br>{p1_name}: <b>%{{y:.1f}}</b><extra></extra>",
        ))

    if portfolio_series2 is not None and not portfolio_series2.empty:
        if first_date is None:
            first_date = portfolio_series2.index[0]
        fig.add_trace(go.Scatter(
            x=portfolio_series2.index, y=portfolio_series2.values,
            name=p2_name, line=dict(color="#6ae8c9", width=2.5),
            hovertemplate=f"%{{x|%d %b %Y}}<br>{p2_name}: <b>%{{y:.1f}}</b><extra></extra>",
        ))

    if bench_series is not None:
        fig.add_trace(go.Scatter(
            x=bench_series.index, y=bench_series.values,
            name=benchmark_ticker, line=dict(color="#6ab4e8", width=2, dash="dot"),
            hovertemplate=f"%{{x|%d %b %Y}}<br>{benchmark_ticker}: <b>%{{y:.1f}}</b><extra></extra>",
        ))

    for lbl in selected_macros:
        cfg_m = MACRO_OVERLAYS[lbl]
        t = cfg_m["ticker"]
        if t in prices.columns and first_date is not None:
            ms = prices.loc[first_date:, t].dropna()
            if not ms.empty:
                ms_idx = ms / ms.iloc[0] * 100
                fig.add_trace(go.Scatter(
                    x=ms_idx.index, y=ms_idx.values, name=lbl, opacity=0.75,
                    line=dict(color=cfg_m["color"], width=1.5, dash=cfg_m["dash"]),
                    hovertemplate=f"%{{x|%d %b %Y}}<br>{lbl}: <b>%{{y:.1f}}</b><extra></extra>",
                ))

    fig.add_hline(y=100, line_dash="dash", line_color="#2a2a2a", line_width=1)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color="#888", size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#aaa"),
                    x=0.01, y=0.99, bordercolor="#2a2a2a", borderwidth=1),
        yaxis=dict(gridcolor="#1a1a1a", showline=True, linecolor="#2a2a2a",
                   title="Indexed Value (base=100)", title_font=dict(size=10, color="#555")),
        xaxis=dict(
            gridcolor="#1a1a1a", showline=True, linecolor="#2a2a2a", tickformat="%b '%y",
            rangeslider=dict(visible=True, bgcolor="#0d0d0d", bordercolor="#2a2a2a",
                             borderwidth=1, thickness=0.10),
            rangeselector=dict(
                buttons=[
                    dict(count=1,  label="1M", step="month", stepmode="backward"),
                    dict(count=3,  label="3M", step="month", stepmode="backward"),
                    dict(count=6,  label="6M", step="month", stepmode="backward"),
                    dict(count=1,  label="1Y", step="year",  stepmode="backward"),
                    dict(step="all", label="All"),
                ],
                bgcolor="#161616", activecolor="#3a3a3a",
                bordercolor="#2a2a2a", borderwidth=1,
                font=dict(color="#aaa", size=10, family="DM Mono, monospace"),
                x=0.0, y=1.02,
            ),
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1a1a1a", bordercolor="#333",
                        font=dict(color="#e8e0d0", family="DM Mono, monospace")),
        height=560, margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — Event period table, full width
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(
        "<div style='font-family:DM Serif Display,serif;font-size:1.2rem;"
        "color:#e8e0d0;margin:1.2rem 0 0.6rem;'>Performance Across Critical Periods</div>",
        unsafe_allow_html=True,
    )

    def _period_ret(series, start, end):
        if series is None or series.empty:
            return None
        sl = series.loc[pd.Timestamp(start):pd.Timestamp(end)].dropna()
        return (sl.iloc[-1] / sl.iloc[0] - 1) * 100 if len(sl) >= 2 else None

    def _fmt_cell(v):
        if v is None:
            return "<td class='evt-na'>—</td>"
        cls  = "evt-pos" if v >= 0 else "evt-neg"
        sign = "+" if v >= 0 else ""
        return f"<td class='{cls}'>{sign}{v:.2f}%</td>"

    rows_html    = ""
    last_section = None
    today_str    = date.today().strftime("%Y-%m-%d")

    for evt in EVENT_PERIODS:
        if evt["start"] > today_str:
            continue
        if evt["section"] != last_section:
            rows_html += (
                f"<tr class='evt-section'>"
                f"<td colspan='6'>{evt['section'].upper()}</td>"
                f"</tr>"
            )
            last_section = evt["section"]
        r1 = _period_ret(portfolio_series1, evt["start"], evt["end"])
        r2 = _period_ret(portfolio_series2, evt["start"], evt["end"])
        rb = _period_ret(bench_series,      evt["start"], evt["end"])
        d  = (r1 - r2) if (r1 is not None and r2 is not None) else None
        rows_html += (
            f"<tr>"
            f"<td style='color:#212121;font-weight:500'>{evt['name']}</td>"
            f"<td style='color:#9e9e9e;font-size:.68rem'>{evt['start']} → {evt['end']}</td>"
            f"{_fmt_cell(r1)}{_fmt_cell(r2)}{_fmt_cell(d)}{_fmt_cell(rb)}"
            f"</tr>"
        )

    bench_th = f"<th>{benchmark_ticker[:12]}</th>" if bench_series is not None else ""
    st.markdown(f"""
    <table class="evt-table">
      <thead><tr>
        <th>Period</th><th>Date Range</th>
        <th style='color:#c77a00'>{p1_name}</th>
        <th style='color:#006064'>{p2_name}</th>
        <th style='color:#1565c0'>Δ (P1−P2)</th>
        {bench_th}
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — Warnings + composition expander
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    warn1, warn2 = st.columns(2)
    with warn1:
        for nm, miss in [(p1_name, missing1), (p2_name, missing2)]:
            if miss:
                st.markdown(f"**{nm} — no price data** (excluded):")
                st.markdown(
                    "<div class='skipped-list'>" +
                    " &nbsp;·&nbsp; ".join(
                        f"<span class='ticker-badge'>{t}</span>" for t in miss
                    ) + "</div>",
                    unsafe_allow_html=True,
                    )
    with warn2:
        for nm, sk in [(p1_name, skipped1), (p2_name, skipped2)]:
            if sk:
                st.warning(f"{nm} — skipped lines: {sk}")
        if missing_bench:
            st.warning(f"Benchmark `{benchmark_ticker}` not found in price data.")

    with st.expander("Portfolio composition"):
        cc1, cc2 = st.columns(2)
        for col, wts, nm in [(cc1, weights1, p1_name), (cc2, weights2, p2_name)]:
            with col:
                valid = {t: w for t, w in wts.items() if t in prices.columns}
                if valid:
                    total = sum(valid.values())
                    df_c  = pd.DataFrame([
                        {"Ticker": t, "Weight": f"{w/total*100:.1f}%"}
                        for t, w in sorted(valid.items(), key=lambda x: -x[1])
                    ])
                    st.markdown(f"**{nm}**")
                    st.dataframe(df_c, use_container_width=True, hide_index=True)