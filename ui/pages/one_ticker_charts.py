"""
one_ticker_charts.py
────────────────────
Chart renderer for the Single-Ticker Deep Dive.
Standalone — runs its own data pipeline via one_ticker_common.py.

Run: streamlit run one_ticker_charts.py
"""

import json
import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.pages.one_ticker_common import (
    COLORS, COLORS_PALE, FUNDAMENTAL_VIEWS,
    load_matrix,
)

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
        df_matrix, conn, _ = load_matrix()
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
    TIMESERIES_METRICS = ["Price (Close)", "Return %"]

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
        try:
            cols = conn.execute("SELECT * FROM ticker_momentum LIMIT 0").df().columns.tolist()
            skip = {'ticker', 'Ticker', 'date', 'Date', 'nse_symbol'}
            return [f"[M] {c}" for c in cols if c not in skip]
        except Exception:
            return []

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
        try:
            return conn.execute(f"""
                SELECT REPLACE(Ticker, '.NS', '') AS Ticker, Date, Close
                FROM ticker_prices
                WHERE REPLACE(Ticker, '.NS', '') IN ({ticker_sql})
                ORDER BY Date
            """).df()
        except Exception:
            return pd.DataFrame()

    @st.cache_data(ttl=120)
    def get_momentum_snapshot(tickers: tuple, col_raw: str):
        ticker_sql = ", ".join(f"'{t}'" for t in tickers)
        try:
            return conn.execute(f"""
                SELECT date, nse_symbol AS ticker, {col_raw}
                FROM ticker_momentum
                WHERE nse_symbol IN ({ticker_sql})
                ORDER BY date ASC
            """).df()
        except Exception:
            return pd.DataFrame()

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
    def build_chart(all_tickers: tuple, primary: str, metrics: list, days: int) -> go.Figure | None:
        """
        Build and return a Plotly figure.
        all_tickers : (selected, *compare_tickers)
        primary     : the first ticker (for snapshot ordering)
        metrics     : 1 or 2 metric strings
        days        : lookback window in days
        Returns None if no data.
        """
        m1 = metrics[0]
        m2 = metrics[1] if len(metrics) > 1 else None
        m1_is_ts = _is_ts(m1)
        m2_is_ts = _is_ts(m2) if m2 else False

        # ── TIME-SERIES PATH ─────────────────────────────────────────────────────
        if m1_is_ts and (m2 is None or m2_is_ts):
            price_df = get_price_history(all_tickers)
            if price_df.empty:
                return None
            price_df['Date'] = pd.to_datetime(price_df['Date'])
            cutoff = price_df['Date'].max() - pd.Timedelta(days=days)
            sliced = price_df[price_df['Date'] >= cutoff].copy()
            if sliced.empty:
                return None
            pivot = sliced.pivot_table(index='Date', columns='Ticker', values='Close').ffill()
            ticker_order = [t for t in list(all_tickers) if t in pivot.columns]

            dual = (
                    (m2 is not None and m1 != m2)
                    or m1.startswith('[F] ')
                    or (m2 is not None and m2.startswith('[F] '))
            )
            fig = make_subplots(specs=[[{"secondary_y": True}]]) if dual else go.Figure()

            def _add_ts(metric, secondary=False):
                palette = COLORS_PALE if secondary else COLORS
                if metric == "Price (Close)":
                    for i, tk in enumerate(ticker_order):
                        s = pivot[tk].dropna()
                        trace = go.Scatter(
                            x=s.index, y=s.values, mode="lines",
                            name=f"{tk} Price",
                            line=dict(color=COLORS[i % len(COLORS)],
                                      width=2.5 if i == 0 else 1.8,
                                      dash="solid" if i == 0 else "dot"),
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
                    mom_pivot = (
                        mom_df[mom_df['date'] >= cutoff]
                        .pivot_table(index='date', columns='ticker', values=col_raw).ffill()
                    )
                    sfx = "%" if _looks_pct(col_raw) else ""
                    for i, tk in enumerate(ticker_order):
                        if tk not in mom_pivot.columns: continue
                        s = mom_pivot[tk].dropna()
                        trace = go.Scatter(
                            x=s.index, y=s.values, mode="lines",
                            name=f"{tk} {col_raw}",
                            line=dict(color=palette[i % len(palette)],
                                      width=2.5 if i == 0 else 1.8,
                                      dash="dot" if secondary else ("solid" if i == 0 else "dot")),
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
                    sfx = "%" if _looks_pct(fund_metric) else ""
                    bw  = 45 * 24 * 3600 * 1000
                    for i, tk in enumerate(list(all_tickers)):
                        tk_data = fund_df[fund_df['ticker'] == tk].dropna(subset=['val'])
                        if tk_data.empty: continue
                        trace = go.Bar(
                            x=tk_data['dt'], y=tk_data['val'],
                            name=f"{tk} {fund_metric}",
                            marker_color=COLORS_PALE[i % len(COLORS_PALE)],
                            marker_line=dict(color=COLORS[i % len(COLORS)], width=1),
                            opacity=0.75, width=bw,
                            hovertemplate=f"%{{x|%b %Y}}<br>%{{y:,.2f}}{sfx}<extra>{tk} {fund_metric}</extra>",
                        )
                        if dual: fig.add_trace(trace, secondary_y=True)
                        else:    fig.add_trace(trace, secondary_y=True)
                    return fund_metric, sfx

                else:  # Return %
                    first_valid = pivot.bfill().iloc[0]
                    perf = ((pivot / first_valid) - 1) * 100
                    for i, tk in enumerate(ticker_order):
                        s = perf[tk].dropna()
                        trace = go.Scatter(
                            x=s.index, y=s.values, mode="lines",
                            name=f"{tk} Ret%",
                            line=dict(color=palette[i % 4],
                                      width=2.5 if i == 0 else 1.8,
                                      dash="dot" if secondary else ("solid" if i == 0 else "dot")),
                            hovertemplate="%{x|%d %b %Y}<br>%{y:+.2f}%<extra>" + tk + "</extra>",
                            yaxis="y2" if (dual and secondary) else "y",
                        )
                        if dual: fig.add_trace(trace, secondary_y=secondary)
                        else:    fig.add_trace(trace)
                    return "Return (%)", "%"

            if dual and m2 is not None and m2.startswith("[F] "):
                y2_title, y2_sfx = _add_ts(m2, secondary=True)
                y1_title, y1_sfx = _add_ts(m1, secondary=False)
            else:
                y1_title, y1_sfx = _add_ts(m1, secondary=False)
                if dual:
                    y2_title, y2_sfx = _add_ts(m2, secondary=True)

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

            if "Return" in m1:
                fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", line_width=1)
            return fig

        # ── SNAPSHOT PATH ─────────────────────────────────────────────────────────
        snap_metrics = [m for m in metrics if not _is_ts(m)]
        if not snap_metrics:
            return None
        metric = snap_metrics[0]
        is_momentum = metric.startswith("[M] ")
        col_raw = metric[4:] if is_momentum else metric
        snapshot_tickers = list(all_tickers)

        if is_momentum:
            raw = get_momentum_snapshot(tuple(snapshot_tickers), col_raw)
            if not raw.empty:
                raw = raw.rename(columns={'ticker': 'Ticker', col_raw: metric})
        else:
            raw = df_matrix[df_matrix['Ticker'].isin(snapshot_tickers)][['Ticker', metric]].copy()

        raw = raw.dropna(subset=[metric]) if not raw.empty else raw
        if raw.empty:
            return None

        ordered = [t for t in snapshot_tickers if t in raw['Ticker'].values]
        raw = raw.set_index('Ticker').reindex(ordered).reset_index()

        fig = go.Figure()
        for i, row in raw.iterrows():
            tk, val = row['Ticker'], row[metric]
            is_primary = (tk == primary)
            fig.add_trace(go.Bar(
                x=[tk], y=[val], name=tk,
                marker_color=COLORS[i % len(COLORS)],
                marker_line_color="#1e3a8a" if is_primary else "rgba(0,0,0,0)",
                marker_line_width=2.5 if is_primary else 0,
                text=[f"{val:,.2f}"], textposition="outside",
            ))
        sfx = "%" if _looks_pct(metric) else ""
        fig.update_layout(
            height=300, margin=dict(l=0, r=0, t=30, b=0),
            title=dict(text=metric, font=dict(size=13)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=True,
                       zerolinecolor="#94a3b8", ticksuffix=sfx, title=metric),
            bargap=0.35,
        )
        return fig

    # ==============================================================================
    # SELECTORS
    # ==============================================================================
    ticker_list = df_matrix['Ticker'].tolist()

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

    # ── Ticker shortcut pills: My Stocks / File Upload ─────────────────────────
    import io as _io, csv as _csv, re as _re

    _port_syms_raw = _get_portfolio_syms()
    _port_in_matrix = [s for s in _port_syms_raw if s in ticker_list]

    with st.expander("⚡ Quick-select tickers", expanded=True):
        qs_tab_my, qs_tab_file = st.tabs(["📋 My Stocks", "📂 File Upload"])

        # Tab 1 — My Stocks (current holdings)
        with qs_tab_my:
            if _port_in_matrix:
                st.caption("Click a ticker to jump straight to it:")
                cols_per_row = 6
                for row_start in range(0, len(_port_in_matrix), cols_per_row):
                    row_tickers = _port_in_matrix[row_start:row_start + cols_per_row]
                    btn_cols = st.columns(len(row_tickers))
                    for col, tk in zip(btn_cols, row_tickers):
                        with col:
                            if st.button(tk, key=f"qs_my_{tk}", use_container_width=True):
                                st.session_state["selected_asset"] = tk
                                st.rerun()
            else:
                st.info("No holdings found in session. Connect your broker account.")

        # Tab 2 — File Upload
        with qs_tab_file:
            st.caption("Upload a CSV or TXT with a `ticker` column (or one ticker per line).")
            qs_uploaded = st.file_uploader(
                "Drop file", type=["csv", "txt"],
                key="qs_file_upload", label_visibility="collapsed",
            )
            # Parse and persist tickers to session state so they survive rerun
            if qs_uploaded:
                raw = qs_uploaded.read().decode("utf-8", errors="replace")
                if qs_uploaded.name.lower().endswith(".csv"):
                    reader = _csv.DictReader(_io.StringIO(raw))
                    col = next((c for c in (reader.fieldnames or []) if "ticker" in c.lower()), None)
                    col = col or (reader.fieldnames[0] if reader.fieldnames else None)
                    parsed = []
                    if col:
                        for r in reader:
                            val = r[col].strip().upper()
                            # Strip exchange suffix e.g. .NS so it matches ticker_list
                            val = val.split(".")[0]
                            if val and val not in parsed:
                                parsed.append(val)
                else:
                    parsed = list(dict.fromkeys(
                        s.strip().upper().split(".")[0]
                        for s in _re.split(r"[,\n;]+", raw) if s.strip()
                    ))
                st.session_state["qs_file_syms"] = [s for s in parsed if s in ticker_list]

            _file_syms: list[str] = st.session_state.get("qs_file_syms", [])

            if _file_syms:
                st.caption(f"{len(_file_syms)} ticker(s) found. Click to select:")
                cols_per_row = 6
                for row_start in range(0, len(_file_syms), cols_per_row):
                    row_tickers = _file_syms[row_start:row_start + cols_per_row]
                    btn_cols = st.columns(len(row_tickers))
                    for col, tk in zip(btn_cols, row_tickers):
                        with col:
                            if st.button(tk, key=f"qs_file_{tk}", use_container_width=True):
                                st.session_state["selected_asset"] = tk
                                st.rerun()
            elif qs_uploaded:
                st.warning("No matching tickers found in the matrix. Check the file has a `ticker` column.")

    # ── All Stocks dropdown ────────────────────────────────────────────────────
    _rest               = sorted(t for t in ticker_list if t not in _port_in_matrix)
    ticker_list_display = _port_in_matrix + _rest

    saved       = st.session_state.get("selected_asset")
    default_idx = ticker_list_display.index(saved) if saved in ticker_list_display else 0

    st.markdown("### 🔬 Fundamental Analysis")

    sel_col, cmp_col, met_col = st.columns([1, 2, 2])

    with sel_col:
        all_selected = st.selectbox(
            "All Stocks:", ticker_list_display, index=default_idx,
            key="selected_asset"
        )
        selected = all_selected

    with cmp_col:
        compare_options = [t for t in ticker_list_display if t != selected]
        compare_tickers = st.multiselect(
            "Compare with:", compare_options, default=[],
            max_selections=3, placeholder="Overlay up to 3 tickers…",
            key="compare_tickers"
        )

    with met_col:
        # Seed default from session state but never write back after widget creation
        init_metrics = st.session_state.get("_metrics_init", ["Price (Close)"])
        valid_init   = [m for m in init_metrics if m in ALL_METRIC_OPTIONS] or ["Price (Close)"]

        compare_metrics = st.multiselect(
            "Metrics (1 or 2):", ALL_METRIC_OPTIONS,
            default=valid_init, max_selections=2,
            placeholder="Choose metrics…", key="compare_metrics",
            help=(
                "Price (Close) / Return % → time-series line chart.\n"
                "[M] columns → momentum time-series overlay.\n"
                "[F] columns → quarterly fundamentals as bars on secondary Y.\n"
                "All others → snapshot bar chart.  Pick 2 to compare."
            )
        )

    if not compare_metrics:
        compare_metrics = ["Price (Close)"]

    # Persist for next render (use a shadow key, never the widget key)
    st.session_state["_metrics_init"] = compare_metrics

    all_tickers_to_plot = tuple([selected] + compare_tickers)

    # ==============================================================================
    # PERIOD TABS + MAIN CHART
    # ==============================================================================
    with st.expander("➕ Playground"):
        PERIOD_TABS  = ["📅 5Y","📅 3Y","📅 2Y","📅 1Y","📅 6M","📅 3M","📅 1M","📅 7D"]
        PERIOD_DAYS  = [1825,    1095,   730,    365,    182,    91,     30,     7]
        PERIOD_LABEL = ["5Y",    "3Y",   "2Y",   "1Y",   "6M",   "3M",   "1M",   "7D"]

        period_tabs = st.tabs(PERIOD_TABS)
        for tab_idx, tab in enumerate(period_tabs):
            with tab:
                fig = build_chart(all_tickers_to_plot, selected, compare_metrics,
                                  PERIOD_DAYS[tab_idx])
                if fig:
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"main_{PERIOD_LABEL[tab_idx]}_{selected}_{'_'.join(compare_metrics)}")
                else:
                    st.info(f"No data for {PERIOD_LABEL[tab_idx]}.")

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
            with st.expander(f"**{name}** · {'  +  '.join(metrics)} · {'  ·  '.join(dash_all_tickers)}", expanded=True):
                panel_fig = build_chart(dash_all_tickers, selected, metrics, dash_days)
                if panel_fig:
                    panel_fig.update_layout(height=320, margin=dict(l=0, r=0, t=8, b=0))
                    st.plotly_chart(panel_fig, use_container_width=True,
                                    key=f"panel_{name}_{'_'.join(dash_all_tickers)}_{dash_days}")
                else:
                    st.info("No data.")