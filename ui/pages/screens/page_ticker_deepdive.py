"""
page_ticker_deepdive.py
────────────────────
Chart renderer for the Single-Ticker Deep Dive.
Standalone — runs its own data pipeline via utils.db_utils.load_ticker_scores().

Run: streamlit run page_ticker_deepdive.py
"""

import json
import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.db_utils import FUNDAMENTAL_VIEWS, load_ticker_scores
from ui.pages.html_utils import COLORS, COLORS_PALE

# Extend the shared 4-colour palette so up to 10 lines stay distinguishable.
COLORS      = list(COLORS)      + ["#7c3aed", "#0891b2", "#db2777", "#65a30d", "#ea580c", "#475569"]
COLORS_PALE = list(COLORS_PALE) + ["#c4b5fd", "#67e8f9", "#f9a8d4", "#bef264", "#fdba74", "#cbd5e1"]

MAX_COMPARE = 8   # tickers that can be overlaid on the selected one

# Keep metric colour consistent across the chart; distinguish tickers by dash style.
METRIC_COLORS = [
    "#2563eb",  # Blue
    "#7c3aed",  # Purple
    "#0d9488",  # Teal
    "#ea580c",  # Orange
    "#db2777",  # Pink
    "#65a30d",  # Green
    "#0891b2",  # Cyan
    "#475569",  # Slate
]
METRIC_COLORS_PALE = [
    "#93c5fd", "#c4b5fd", "#5eead4", "#fdba74",
    "#f9a8d4", "#bef264", "#67e8f9", "#cbd5e1",
]
TICKER_DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]

# ==============================================================================
# 1. PAGE CONFIG & STYLING
# ==============================================================================
def render():

    # ==============================================================================
    # PAGE CONFIG & FULL-WIDTH LAYOUT
    # ==============================================================================
    st.set_page_config(layout="wide")

    st.markdown("""
    <style>
        .block-container {
            max-width: 100% !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            padding-top: 1rem !important;
            padding-bottom: 1.5rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # ==============================================================================
    # BOOTSTRAP
    # ==============================================================================
    if "df_matrix" not in st.session_state or "conn" not in st.session_state:
        df_matrix, conn, _ = load_ticker_scores()
        st.session_state["df_matrix"] = df_matrix
        st.session_state["conn"]      = conn
    else:
        df_matrix = st.session_state["df_matrix"]
        conn      = st.session_state["conn"]

    if df_matrix.empty:
        st.error("❌ No data available. Check your parquet paths.")
        st.stop()

    # ==============================================================================
    # PIN PERSISTENCE — JSON file
    # ==============================================================================
    PINS_FILE = "outputs/chart_pins.json"

    def _load_pins_from_disk() -> dict:
        try:
            if os.path.exists(PINS_FILE):
                with open(PINS_FILE, "r") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_pins_to_disk(pins: dict):
        try:
            with open(PINS_FILE, "w") as f:
                json.dump(pins, f, indent=2)
        except Exception:
            pass

    # Sync disk → session state once per session
    if "chart_pins" not in st.session_state:
        st.session_state["chart_pins"] = _load_pins_from_disk()

    def _pins() -> dict:
        return st.session_state["chart_pins"]

    def _save_pin(name: str, metrics: list):
        _pins()[name] = metrics
        _save_pins_to_disk(_pins())

    def _delete_pin(name: str):
        _pins().pop(name, None)
        _save_pins_to_disk(_pins())

    # ==============================================================================
    # METRIC CATALOGUE
    # ==============================================================================
    REBASED_METRIC = "Price (Rebased=100)"
    TIMESERIES_METRICS = ["Price (Close)", REBASED_METRIC, "Return %"]

    _SNAP_EXCLUDE = {
        'Ticker', 'Company', 'Sector',
        *[c for c in df_matrix.columns if c.startswith('G') or c.startswith('_')]
    }
    SNAPSHOT_METRICS = [
        c for c in df_matrix.columns
        if c not in _SNAP_EXCLUDE and pd.api.types.is_numeric_dtype(df_matrix[c])
    ]

    @st.cache_data(ttl=300)
    def get_momentum_columns():
        skip = {'ticker', 'Ticker', 'date', 'Date', 'nse_symbol'}
        cols: list[str] = []
        for table in ("ticker_momentum", "benchmark_momentum"):
            try:
                for c in conn.execute(f"SELECT * FROM {table} LIMIT 0").df().columns:
                    if c not in skip and c not in cols:
                        cols.append(c)
            except Exception:
                pass
        return [f"[M] {c}" for c in cols]

    MOMENTUM_COLS = get_momentum_columns()

    @st.cache_data(ttl=300)
    def get_fundamental_metrics():
        opts = []
        for view, label in FUNDAMENTAL_VIEWS:
            try:
                rows = conn.execute(f"SELECT DISTINCT metric FROM {view} ORDER BY metric").df()
                for m in rows['metric'].tolist():
                    opts.append(f"[F] {label}::{m}")
            except Exception:
                pass
        return opts

    FUNDAMENTAL_COLS   = get_fundamental_metrics()
    ALL_METRIC_OPTIONS = TIMESERIES_METRICS + SNAPSHOT_METRICS + MOMENTUM_COLS + FUNDAMENTAL_COLS

    # ==============================================================================
    # DATA FETCHERS
    # ==============================================================================
    @st.cache_data(ttl=120)
    def get_price_history(tickers: tuple):
        ticker_sql = ", ".join(f"'{t}'" for t in tickers)
        frames = []
        # Stocks + benchmark ETFs (BANKBEES, GOLDBEES, ...) live in separate tables.
        for table in ("ticker_prices", "benchmark_prices"):
            try:
                frames.append(conn.execute(f"""
                    SELECT REPLACE(Ticker, '.NS', '') AS Ticker, Date, Close
                    FROM {table}
                    WHERE REPLACE(Ticker, '.NS', '') IN ({ticker_sql})
                """).df())
            except Exception:
                pass
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["Ticker", "Date"], keep="first")
        return df.sort_values("Date").reset_index(drop=True)

    @st.cache_data(ttl=300)
    def get_benchmark_symbols():
        try:
            rows = conn.execute("""
                SELECT DISTINCT REPLACE(Ticker, '.NS', '') AS sym
                FROM benchmark_prices
                ORDER BY sym
            """).df()
            return rows['sym'].dropna().tolist()
        except Exception:
            return []

    @st.cache_data(ttl=120)
    def get_momentum_snapshot(tickers: tuple, col_raw: str):
        ticker_sql = ", ".join(f"'{t}'" for t in tickers)
        # Column is double-quoted so names like 1Y / 6M / 3M are valid identifiers.
        # Stocks key on nse_symbol; benchmark ETFs key on ticker (may carry '.NS').
        sources = [
            ("ticker_momentum",    "nse_symbol"),
            ("benchmark_momentum", "REPLACE(ticker, '.NS', '')"),
        ]
        frames = []
        for table, sym_expr in sources:
            try:
                frames.append(conn.execute(f"""
                    SELECT date, {sym_expr} AS ticker, "{col_raw}"
                    FROM {table}
                    WHERE {sym_expr} IN ({ticker_sql})
                """).df())
            except Exception:
                pass
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["date", "ticker"], keep="first")
        return df.sort_values("date").reset_index(drop=True)

    @st.cache_data(ttl=300)
    def get_fundamental_ts(tickers: tuple, view_label: str, metric: str):
        view_name = next((v for v, lbl in FUNDAMENTAL_VIEWS if lbl == view_label), None)
        if view_name is None:
            return pd.DataFrame()
        ticker_sql = ", ".join(f"'{t}'" for t in tickers)
        try:
            df = conn.execute(f"""
                SELECT ticker, dt, val
                FROM {view_name}
                WHERE metric = '{metric}' AND ticker IN ({ticker_sql})
                ORDER BY dt
            """).df()
            df['dt'] = pd.to_datetime(df['dt'])
            return df
        except Exception:
            return pd.DataFrame()

    # ==============================================================================
    # HELPERS
    # ==============================================================================
    def _looks_pct(col_name: str) -> bool:
        return any(kw in col_name.lower() for kw in ['%', 'pct', 'ratio', 'score', 'dist', 'return'])

    def _is_ts(metric) -> bool:
        return (metric in TIMESERIES_METRICS
                or (metric is not None and metric.startswith("[M] "))
                or (metric is not None and metric.startswith("[F] ")))

    # ==============================================================================
    # CHART BUILDER — returns a Plotly figure for any (tickers, metrics, days) combo
    # ==============================================================================
    def build_chart(all_tickers: tuple, primary: str, metrics: list, days: int | None = None,
                    start_date=None, end_date=None, primary_color: str | None = None) -> go.Figure | None:
        """
        Build and return a Plotly figure.
        all_tickers : (selected, *compare_tickers)
        primary     : the first ticker (for snapshot ordering)
        metrics     : up to 4 metric strings; all selected metrics are rendered
        days        : lookback window in days
        Returns None if no data.
        """
        metrics = list(dict.fromkeys(metrics))[:4]
        if not metrics:
            return None
        # A single X-axis cannot combine dated time-series with ticker snapshots.
        # Apply the same rule to saved panels as to the interactive selector.
        first_type = _is_ts(metrics[0])
        metrics = [m for m in metrics if _is_ts(m) == first_type]

        # ── TIME-SERIES PATH: render every selected time-series metric ────────────
        if all(_is_ts(m) for m in metrics):
            price_df = get_price_history(all_tickers)
            if price_df.empty:
                return None
            price_df['Date'] = pd.to_datetime(price_df['Date'])
            if start_date is not None and end_date is not None:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                sliced = price_df[(price_df['Date'] >= start_dt) & (price_df['Date'] <= end_dt)].copy()
                cutoff = start_dt
                end_bound = end_dt
            else:
                cutoff = price_df['Date'].max() - pd.Timedelta(days=days or 3650)
                end_bound = None
                sliced = price_df[price_df['Date'] >= cutoff].copy()
            if sliced.empty:
                return None
            pivot = sliced.pivot_table(index='Date', columns='Ticker', values='Close').ffill()
            ticker_order = [t for t in list(all_tickers) if t in pivot.columns]

            # Plot fundamental series on the right axis and other series on the
            # left. Multiple selected metrics are all added to this same figure.
            has_fundamental = any(m.startswith('[F] ') for m in metrics)
            has_non_fundamental = any(not m.startswith('[F] ') for m in metrics)
            dual = has_fundamental and has_non_fundamental
            fig = make_subplots(specs=[[{"secondary_y": True}]]) if dual else go.Figure()

            def _add_ts(metric, metric_idx=0, secondary=False):
                # Colour by metric so multiple metrics for one ticker are distinguishable.
                metric_color = METRIC_COLORS[metric_idx % len(METRIC_COLORS)]
                metric_pale = METRIC_COLORS_PALE[metric_idx % len(METRIC_COLORS_PALE)]
                metric_dash = st.session_state.get("metric_line_styles", {}).get(metric, TICKER_DASHES[metric_idx % len(TICKER_DASHES)])
                if metric == "Price (Close)":
                    for i, tk in enumerate(ticker_order):
                        s = pivot[tk].dropna()
                        trace = go.Scatter(
                            x=s.index, y=s.values, mode="lines",
                            name=f"{tk} Price",
                            line=dict(color=st.session_state.get("ticker_colors", {}).get(tk, primary_color if tk == primary and primary_color else metric_color),
                                      width=2.8 if i == 0 else 2.0,
                                      dash=metric_dash),
                            hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.2f}<extra>" + tk + "</extra>",
                            yaxis="y2" if (dual and secondary) else "y",
                        )
                        if dual: fig.add_trace(trace, secondary_y=secondary)
                        else:    fig.add_trace(trace)
                    return "Price (₹)", ""

                elif metric.startswith("[M] "):
                    col_raw = metric[4:]
                    mom_df  = get_momentum_snapshot(all_tickers, col_raw)
                    if mom_df.empty: return col_raw, ""
                    mom_df['date'] = pd.to_datetime(mom_df['date'])
                    mom_window = mom_df[mom_df['date'] >= cutoff]
                    if end_bound is not None:
                        mom_window = mom_window[mom_window['date'] <= end_bound]
                    mom_pivot = (
                        mom_window.pivot_table(index='date', columns='ticker', values=col_raw).ffill()
                    )
                    sfx = "%" if _looks_pct(col_raw) else ""
                    for i, tk in enumerate(ticker_order):
                        if tk not in mom_pivot.columns: continue
                        s = mom_pivot[tk].dropna()
                        trace = go.Scatter(
                            x=s.index, y=s.values, mode="lines",
                            name=f"{tk} {col_raw}",
                            line=dict(color=st.session_state.get("ticker_colors", {}).get(tk, primary_color if tk == primary and primary_color else metric_color),
                                      width=2.8 if i == 0 else 2.0,
                                      dash=metric_dash),
                            hovertemplate=f"%{{x|%d %b %Y}}<br>%{{y:,.2f}}{sfx}<extra>{tk}</extra>",
                            yaxis="y2" if (dual and secondary) else "y",
                        )
                        if dual: fig.add_trace(trace, secondary_y=secondary)
                        else:    fig.add_trace(trace)
                    return col_raw, sfx

                elif metric.startswith("[F] "):
                    _, rest = metric.split(" ", 1)
                    view_label, fund_metric = rest.split("::", 1)
                    fund_df = get_fundamental_ts(all_tickers, view_label, fund_metric)
                    if fund_df.empty: return fund_metric, ""
                    fund_df  = fund_df[fund_df['dt'] >= cutoff]
                    if end_bound is not None:
                        fund_df = fund_df[fund_df['dt'] <= end_bound]
                    sfx = "%" if _looks_pct(fund_metric) else ""
                    bw  = 45 * 24 * 3600 * 1000
                    for i, tk in enumerate(list(all_tickers)):
                        tk_data = fund_df[fund_df['ticker'] == tk].dropna(subset=['val'])
                        if tk_data.empty: continue
                        trace = go.Bar(
                            x=tk_data['dt'], y=tk_data['val'],
                            name=f"{tk} {fund_metric}",
                            marker_color=st.session_state.get("ticker_colors", {}).get(tk, metric_pale),
                            marker_line=dict(color=st.session_state.get("ticker_colors", {}).get(tk, primary_color if tk == primary and primary_color else metric_color), width=1),
                            opacity=0.75, width=bw,
                            hovertemplate=f"%{{x|%b %Y}}<br>%{{y:,.2f}}{sfx}<extra>{tk} {fund_metric}</extra>",
                        )
                        if dual:
                            fig.add_trace(trace, secondary_y=secondary)
                        else:
                            fig.add_trace(trace)
                    return fund_metric, sfx

                elif metric == REBASED_METRIC:
                    # Start every ticker at 100 on the first date ALL tickers have
                    # data, so the lines are directly comparable (same start date).
                    complete = pivot.dropna(how="any")
                    if complete.empty:
                        # No common date: fall back to each ticker's own first value
                        base = pivot.apply(lambda c: c.dropna().iloc[0] if c.notna().any() else float("nan"))
                        rebased = pivot / base * 100
                    else:
                        start = complete.index[0]
                        window = pivot.loc[start:]
                        rebased = window / window.iloc[0] * 100
                    for i, tk in enumerate(ticker_order):
                        s = rebased[tk].dropna()
                        trace = go.Scatter(
                            x=s.index, y=s.values, mode="lines",
                            name=f"{tk} (100)",
                            line=dict(color=st.session_state.get("ticker_colors", {}).get(tk, primary_color if tk == primary and primary_color else metric_color),
                                      width=2.8 if i == 0 else 2.0,
                                      dash=metric_dash),
                            hovertemplate="%{x|%d %b %Y}<br>%{y:,.1f}<extra>" + tk + "</extra>",
                            yaxis="y2" if (dual and secondary) else "y",
                        )
                        if dual: fig.add_trace(trace, secondary_y=secondary)
                        else:    fig.add_trace(trace)
                    return "Rebased (start = 100)", ""

                else:  # Return %
                    first_valid = pivot.bfill().iloc[0]
                    perf = ((pivot / first_valid) - 1) * 100
                    for i, tk in enumerate(ticker_order):
                        s = perf[tk].dropna()
                        trace = go.Scatter(
                            x=s.index, y=s.values, mode="lines",
                            name=f"{tk} Ret%",
                            line=dict(color=st.session_state.get("ticker_colors", {}).get(tk, primary_color if tk == primary and primary_color else metric_color),
                                      width=2.8 if i == 0 else 2.0,
                                      dash=metric_dash),
                            hovertemplate="%{x|%d %b %Y}<br>%{y:+.2f}%<extra>" + tk + "</extra>",
                            yaxis="y2" if (dual and secondary) else "y",
                        )
                        if dual: fig.add_trace(trace, secondary_y=secondary)
                        else:    fig.add_trace(trace)
                    return "Return (%)", "%"

            axis_titles = []
            for metric_idx, metric in enumerate(metrics):
                is_fundamental = metric.startswith('[F] ')
                title, suffix = _add_ts(
                    metric, metric_idx=metric_idx,
                    secondary=(dual and is_fundamental)
                )
                axis_titles.append((title, suffix, is_fundamental))
            # Keep axis labels meaningful when several metrics share an axis.
            left_items = [(t, s) for t, s, is_f in axis_titles if not (dual and is_f)]
            right_items = [(t, s) for t, s, is_f in axis_titles if dual and is_f]
            y1_title = ' / '.join(dict.fromkeys(t for t, _ in left_items)) or 'Metric'
            y1_sfx = left_items[0][1] if len(left_items) == 1 else ''
            y2_title = ' / '.join(dict.fromkeys(t for t, _ in right_items)) or 'Fundamentals'
            y2_sfx = right_items[0][1] if len(right_items) == 1 else ''

            layout_kw = dict(
                height=320, margin=dict(l=0, r=0, t=28, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(showgrid=False, zeroline=False),
                hovermode="x unified", barmode="group",
            )
            if dual:
                fig.update_yaxes(title_text=y1_title, ticksuffix=y1_sfx,
                                 showgrid=True, gridcolor="#e2e8f0", secondary_y=False)
                fig.update_yaxes(title_text=y2_title, ticksuffix=y2_sfx,
                                 showgrid=False, secondary_y=True)
                fig.update_layout(**layout_kw)
            else:
                layout_kw["yaxis"] = dict(showgrid=True, gridcolor="#e2e8f0",
                                          zeroline=False, ticksuffix=y1_sfx, title=y1_title)
                fig.update_layout(**layout_kw)

            if any("Return" in m for m in metrics):
                fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", line_width=1)
            if any(m == REBASED_METRIC for m in metrics):
                fig.add_hline(y=100, line_dash="dash", line_color="#94a3b8", line_width=1)
            return fig

        # ── SNAPSHOT PATH: render every selected snapshot metric as grouped bars ──
        snap_metrics = [m for m in metrics if not _is_ts(m)]
        if not snap_metrics:
            return None
        snapshot_tickers = list(all_tickers)
        fig = go.Figure()
        found_any = False

        for metric_idx, metric in enumerate(snap_metrics):
            is_momentum = metric.startswith("[M] ")
            col_raw = metric[4:] if is_momentum else metric

            if is_momentum:
                raw = get_momentum_snapshot(tuple(snapshot_tickers), col_raw)
                if not raw.empty:
                    raw = raw.rename(columns={'ticker': 'Ticker', col_raw: metric})
            elif metric in df_matrix.columns:
                raw = df_matrix[df_matrix['Ticker'].isin(snapshot_tickers)][['Ticker', metric]].copy()
            else:
                continue

            if raw.empty or metric not in raw.columns:
                continue
            raw = raw.dropna(subset=[metric])
            if raw.empty:
                continue
            found_any = True
            ordered = [t for t in snapshot_tickers if t in raw['Ticker'].values]
            raw = raw.set_index('Ticker').reindex(ordered).reset_index()

            for i, row in raw.iterrows():
                tk, val = row['Ticker'], row[metric]
                if pd.isna(val):
                    continue
                is_primary = (tk == primary)
                fig.add_trace(go.Bar(
                    x=[tk], y=[val], name=f"{metric} · {tk}",
                    legendgroup=metric,
                    marker_color=COLORS_PALE[metric_idx % len(COLORS_PALE)],
                    marker_line_color=COLORS[i % len(COLORS)],
                    marker_line_width=2.5 if is_primary else 0.5,
                    text=[f"{val:,.2f}"], textposition="outside",
                    hovertemplate=f"{metric}<br>{tk}<br>%{{y:,.2f}}<extra></extra>",
                ))

        if not found_any:
            return None
        fig.update_layout(
            height=340, margin=dict(l=0, r=0, t=30, b=0),
            title=dict(text=' | '.join(snap_metrics), font=dict(size=13)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=True,
                       zerolinecolor="#94a3b8"),
            barmode='group', bargap=0.25,
        )
        return fig

    # ==============================================================================
    # SELECTORS
    # ==============================================================================
    ticker_list = df_matrix['Ticker'].tolist()
    # Add benchmark ETFs (price-only: no fundamentals) so they can be selected/compared
    _bench_syms  = get_benchmark_symbols()
    _existing    = set(ticker_list)
    _bench_extra = [s for s in _bench_syms if s not in _existing]
    ticker_list  = ticker_list + _bench_extra

    # ── Helper: load portfolio syms from session ───────────────────────────────
    def _get_portfolio_syms() -> list[str]:
        syms: list[str] = []
        try:
            _all_holdings = st.session_state.get("all_holdings") or {}
            _excl = {s.upper() for s in (st.session_state.get("excluded_symbols") or [])}
            for _holdings in _all_holdings.values():
                for _h in _holdings:
                    _sym = _h.get("tradingsymbol", "")
                    if _sym and _sym.upper() not in _excl and _sym not in syms:
                        syms.append(_sym)
        except Exception:
            pass
        return syms

    _port_syms_raw = _get_portfolio_syms()
    _port_in_matrix = [s for s in _port_syms_raw if s in ticker_list]
    _rest = sorted(t for t in ticker_list if t not in _port_in_matrix)
    ticker_list_display = _port_in_matrix + _rest

    st.markdown("### 🔬 Fundamental Analysis")

    sel_col, cmp_col, met_col = st.columns([1, 2, 2])

    with sel_col:
        my_stocks_pill = st.pills(
            "Ticker filter",
            options=["All", "My Stocks only"],
            default="All",
            key="my_stocks_only_pill",
            label_visibility="collapsed",
        )
        restrict_to_my_stocks = (my_stocks_pill == "My Stocks only")

        if restrict_to_my_stocks:
            if _port_in_matrix:
                active_ticker_options = _port_in_matrix
            else:
                st.caption("No holdings found in session; showing all stocks.")
                active_ticker_options = ticker_list_display
        else:
            active_ticker_options = ticker_list_display

        saved = st.session_state.get("selected_asset")
        default_idx = active_ticker_options.index(saved) if saved in active_ticker_options else 0

        all_selected = st.selectbox(
            "All Stocks:", active_ticker_options, index=default_idx,
            key="selected_asset"
        )
        selected = all_selected

    with cmp_col:
        compare_options = [t for t in active_ticker_options if t != selected]
        compare_tickers = st.multiselect(
            "Compare with:", compare_options, default=[],
            max_selections=MAX_COMPARE, placeholder=f"Overlay up to {MAX_COMPARE} tickers…",
            key="compare_tickers"
        )

    with met_col:
        # Seed default from session state but never write back after widget creation
        init_metrics = st.session_state.get("_metrics_init", ["Price (Close)"])
        valid_init   = [m for m in init_metrics if m in ALL_METRIC_OPTIONS] or ["Price (Close)"]

        compare_metrics = st.multiselect(
            "Metrics (up to 4):", ALL_METRIC_OPTIONS,
            default=valid_init, max_selections=4,
            placeholder="Choose metrics…", key="compare_metrics",
            help=(
                "Price (Close) / Return % → time-series line chart.\n"
                "Price (Rebased=100) → all tickers start at 100 for easy comparison.\n"
                "[M] columns → momentum time-series overlay.\n"
                "[F] columns → quarterly fundamentals as bars on secondary Y.\n"
                "All others → snapshot bar chart. Up to 4 metrics are shown together; keep selections within the same chart type (time-series or snapshot)."
            )
        )

    if not compare_metrics:
        compare_metrics = ["Price (Close)"]

    # A single Plotly X-axis cannot meaningfully combine dated series and ticker
    # snapshot bars. Keep the first selected chart type and tell the user why.
    _selected_types = {"time-series" if _is_ts(m) else "snapshot" for m in compare_metrics}
    if len(_selected_types) > 1:
        _keep_type = "time-series" if _is_ts(compare_metrics[0]) else "snapshot"
        compare_metrics = [m for m in compare_metrics if ("time-series" if _is_ts(m) else "snapshot") == _keep_type]
        st.warning("Time-series and snapshot metrics use different X-axes. Showing the selected metrics of the same chart type as your first selection.")

    # Persist for next render (use a shadow key, never the widget key)
    st.session_state["_metrics_init"] = compare_metrics

    all_tickers_to_plot = tuple([selected] + compare_tickers)

    # ── Editable default style configuration: ticker = color, metric = line style ──
    DEFAULT_TICKER_COLORS = ["#2563eb", "#f59e0b", "#16a34a", "#9333ea", "#dc2626",
                             "#0891b2", "#db2777", "#65a30d", "#475569", "#ea580c"]
    DEFAULT_METRIC_DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
    _style_tickers = list(dict.fromkeys(all_tickers_to_plot))
    _style_metrics = [m for m in compare_metrics if _is_ts(m)]
    if "ticker_colors" not in st.session_state:
        st.session_state["ticker_colors"] = {}
    if "metric_line_styles" not in st.session_state:
        st.session_state["metric_line_styles"] = {}
    for _tk in _style_tickers:
        _stable_color_idx = ticker_list.index(_tk) if _tk in ticker_list else 0
        st.session_state["ticker_colors"].setdefault(_tk, DEFAULT_TICKER_COLORS[_stable_color_idx % len(DEFAULT_TICKER_COLORS)])
    def _default_dash_for_metric(metric):
        if metric == "Price (Close)": return "solid"
        if metric == REBASED_METRIC: return "dot"
        if metric == "Return %": return "dash"
        if metric.startswith("[M] "): return "dashdot"
        if metric.startswith("[F] "): return "longdash"
        return "solid"
    for _metric in _style_metrics:
        st.session_state["metric_line_styles"].setdefault(_metric, _default_dash_for_metric(_metric))

    def _sync_ticker_color(ticker, widget_key):
        st.session_state["ticker_colors"][ticker] = st.session_state[widget_key]

    def _sync_metric_dash(metric, widget_key):
        st.session_state["metric_line_styles"][metric] = st.session_state[widget_key]

    with st.expander("🎨 Default configuration grid · Edit colors and line styles", expanded=False):
        st.caption("Ticker color is shared across metrics; metric line style is shared across tickers.")
        _ticker_cols = st.columns(max(1, min(5, len(_style_tickers))))
        for _i, _tk in enumerate(_style_tickers):
            _widget_key = f"ticker_color_widget_{_tk}"
            st.session_state.setdefault(_widget_key, st.session_state["ticker_colors"][_tk])
            with _ticker_cols[_i % len(_ticker_cols)]:
                st.color_picker(f"{_tk} color", key=_widget_key,
                                on_change=_sync_ticker_color, args=(_tk, _widget_key))
        if _style_metrics:
            st.markdown("**Metric line styles**")
            _metric_cols = st.columns(max(1, min(4, len(_style_metrics))))
            for _i, _metric in enumerate(_style_metrics):
                _widget_key = f"metric_dash_widget_{_metric}"
                st.session_state.setdefault(_widget_key, st.session_state["metric_line_styles"][_metric])
                with _metric_cols[_i % len(_metric_cols)]:
                    st.selectbox(_metric, DEFAULT_METRIC_DASHES, key=_widget_key,
                                 on_change=_sync_metric_dash, args=(_metric, _widget_key))
        st.markdown("**Preview grid**")
        _grid_cols = st.columns([2] + [1] * max(1, len(_style_metrics)))
        _grid_cols[0].markdown("**Ticker**")
        for _j, _metric in enumerate(_style_metrics):
            _grid_cols[_j + 1].markdown(f"**{_metric}**")
        for _tk in _style_tickers:
            _row = st.columns([2] + [1] * max(1, len(_style_metrics)))
            _row[0].markdown(f"<span style='color:{st.session_state['ticker_colors'][_tk]}'>●</span> {_tk}", unsafe_allow_html=True)
            for _j, _metric in enumerate(_style_metrics):
                _dash = st.session_state["metric_line_styles"].get(_metric, "solid")
                _samples = {"solid":"━━━━", "dash":"┅┅┅", "dot":"····", "dashdot":"━·━·", "longdash":"━━ ━", "longdashdot":"━━· ━·"}
                _row[_j + 1].markdown(f"<span style='color:{st.session_state['ticker_colors'][_tk]}'> {_samples.get(_dash, '━━━━')} </span>", unsafe_allow_html=True)

    _bench_in_use = [t for t in all_tickers_to_plot if t in _bench_extra]
    if _bench_in_use and any(not (m in TIMESERIES_METRICS or m.startswith("[M] ")) for m in compare_metrics):
        st.caption(f"ℹ️ {', '.join(_bench_in_use)} is a benchmark/ETF — price metrics only "
                   f"(no fundamentals), so it will be blank for fundamental/snapshot metrics.")

    # ==============================================================================
    # PERIOD TABS + MAIN CHART
    # ==============================================================================
    with st.expander("➕ Playground", expanded=True):
        PERIOD_TABS  = ["📅 10Y", "📅 5Y", "📅 3Y", "📅 2Y", "📅 1Y", "📅 6M", "📅 3M", "📅 1M", "📅 7D", "🗓️ Custom"]
        PERIOD_DAYS  = [3650,     1825,    1095,    730,    365,    182,    91,     30,     7]
        PERIOD_LABEL = ["10Y",    "5Y",    "3Y",    "2Y",   "1Y",   "6M",   "3M",   "1M",   "7D"]

        # Custom-range defaults are stored in session state so NEXT can roll the
        # selected window forward without losing the user's chosen duration.
        _today = pd.Timestamp.today().date()
        _custom_start_key = "playground_custom_start"
        _custom_end_key = "playground_custom_end"
        if _custom_start_key not in st.session_state:
            st.session_state[_custom_start_key] = (pd.Timestamp.today() - pd.DateOffset(years=10)).date()
        if _custom_end_key not in st.session_state:
            st.session_state[_custom_end_key] = _today

        def _advance_custom_range():
            start = st.session_state[_custom_start_key]
            end = st.session_state[_custom_end_key]
            duration = end - start
            # As requested, the old End becomes the new Start; advance End by
            # the same selected range length.
            st.session_state[_custom_start_key] = end
            st.session_state[_custom_end_key] = end + duration

        def _retreat_custom_range():
            start = st.session_state[_custom_start_key]
            end = st.session_state[_custom_end_key]
            duration = end - start
            # The old Start becomes the new End; move Start back by the same range.
            st.session_state[_custom_end_key] = start
            st.session_state[_custom_start_key] = start - duration

        period_tabs = st.tabs(PERIOD_TABS)
        for tab_idx, tab in enumerate(period_tabs):
            with tab:
                if tab_idx < len(PERIOD_DAYS):
                    fig = build_chart(all_tickers_to_plot, selected, compare_metrics,
                                      PERIOD_DAYS[tab_idx])
                    period_label = PERIOD_LABEL[tab_idx]
                else:
                    custom_col1, custom_col2, custom_prev, custom_next = st.columns([2, 2, 1, 1], vertical_alignment="top")
                    with custom_col1:
                        custom_start = st.date_input(
                            "Start date", key=_custom_start_key)
                    with custom_col2:
                        custom_end = st.date_input(
                            "End date", key=_custom_end_key)
                    with custom_prev:
                        st.button(
                            "◀ PREVIOUS", key="playground_custom_previous",
                            help="Move backward by the current custom date-range duration.",
                            on_click=_retreat_custom_range,
                            use_container_width=True)
                    with custom_next:
                        st.button(
                            "NEXT ▶", key="playground_custom_next",
                            help="Move the start to the current end date and advance the end by the same duration.",
                            on_click=_advance_custom_range,
                            use_container_width=True)
                    if custom_start > custom_end:
                        st.warning("Start date must be on or before End date.")
                        fig = None
                    else:
                        fig = build_chart(all_tickers_to_plot, selected, compare_metrics,
                                          start_date=custom_start, end_date=custom_end)
                    period_label = f"Custom_{custom_start}_{custom_end}"
                if fig:
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"main_{period_label}_{selected}_{'_'.join(compare_metrics)}")
                else:
                    st.info(f"No data for {period_label}.")

    # ==============================================================================
    # PIN MANAGER — below the chart
    # ==============================================================================
    st.markdown("#### 📌 Saved Views")

    pins = _pins()

    # ── Save current view ────────────────────────────────────────────────────────
    with st.expander("➕ Save current view as a panel", expanded=not bool(pins)):
        save_col1, save_col2 = st.columns([3, 1])
        with save_col1:
            pin_name = st.text_input(
                "Panel name", placeholder="e.g. Price + Net Profit",
                key="pin_name_input", label_visibility="collapsed"
            )
        with save_col2:
            if st.button("📌 Save", use_container_width=True):
                if pin_name.strip():
                    _save_pin(pin_name.strip(), list(compare_metrics))
                    st.success(f"Saved '{pin_name.strip()}'")
                    st.rerun()
                else:
                    st.warning("Enter a name first.")

    # ── Load a pin → update metrics selector ────────────────────────────────────
    if pins:
        load_cols = st.columns(min(len(pins), 4))
        for i, (name, metrics) in enumerate(list(pins.items())):
            col = load_cols[i % len(load_cols)]
            label_str = " + ".join(metrics)
            btn_col, del_col = col.columns([4, 1])
            with btn_col:
                if st.button(f"▶ {name}", key=f"load_pin_{name}",
                             use_container_width=True, help=label_str):
                    valid = [m for m in metrics if m in ALL_METRIC_OPTIONS]
                    st.session_state["_metrics_init"] = valid or ["Price (Close)"]
                    st.rerun()
            with del_col:
                if st.button("✕", key=f"del_pin_{name}", help="Delete"):
                    _delete_pin(name)
                    st.rerun()

    # ==============================================================================
    # GRAFANA-STYLE SAVED PANEL DASHBOARD
    # ==============================================================================
    if pins:
        with st.expander("🗂 Saved Panel Dashboard", expanded=False):
            st.markdown("---")
            st.markdown("#### 🗂 Saved Panel Dashboard")

            dash_period_label = st.radio(
                "Period", PERIOD_LABEL, index=PERIOD_LABEL.index("1Y"),
                horizontal=True, key="dash_period", label_visibility="collapsed"
            )
            dash_days        = PERIOD_DAYS[PERIOD_LABEL.index(dash_period_label)]
            dash_all_tickers = tuple([selected] + compare_tickers)

            panel_names = list(pins.keys())
            for name in panel_names:
                metrics = pins[name]
                with st.expander(f"**{name}** · {'  +  '.join(metrics)} · {'  ·  '.join(dash_all_tickers)}", expanded=False):
                    panel_fig = build_chart(dash_all_tickers, selected, metrics, dash_days)
                    if panel_fig:
                        panel_fig.update_layout(height=320, margin=dict(l=0, r=0, t=8, b=0))
                        st.plotly_chart(panel_fig, use_container_width=True,
                                        key=f"panel_{name}_{'_'.join(dash_all_tickers)}_{dash_days}")
                    else:
                        st.info("No data.")