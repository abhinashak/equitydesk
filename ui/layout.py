"""
ui/layout.py
─────────────
Shared Streamlit layout helpers: CSS injection, sidebar nav, topbar.
"""

import streamlit as st

NAV: dict[str, dict[str, str]] = {
    "CONFIG": {
        "🗂 Tickers":                "config_tickers",
        "⚙️ App Settings":           "config_app",
    },
    "DATA": {
        "📈 Market Data":            "data_ticker",
        "📊 Fundamental":            "data_fundamental",
        "📡 Signals":                "signals",
    },
    "SCREENS": {
        "🏆 Winners ":               "sql_lab",
        "⚛️ Fundamental Analysis":   "fundamental_analysis",
        "🔬 DCF Analysis":           "dcf_analysis",
        "🔬 Quality Gates":          "quality_gates",
        "⚖️ Portfolio Evaluator":    "portfolio_evaluator",
    },
    "ALGOS": {
        "🔮 Genetic Algorithm":      "algo_ga",
        "⚡ Momentum Rules":         "algo_momentum",
        "🧮 Rule Discovery":         "algo_rules",
    },
    "TRADE": {
        "🪁 1. Kite Setup":          "kite_setup",
        "🔴 2. Portfolio":           "trade_portfolio",
        "⚖️ 3. Rebalance Planner":   "trade_rebalance",
        "⚡  4. Live Execution":      "trade_execution",    },
}

_DARK_CSS = """
<style>
html, body, [class*="css"] { font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif; }

[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: transparent; border: none;
    text-align: left; padding: 0.35rem 0.75rem;
    width: 100%; font-size: 0.87rem;
    color: #8b949e !important; border-radius: 6px;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #161b22; color: #e6edf3 !important;
}

[data-testid="metric-container"] {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 8px; padding: 0.75rem 1rem;
}
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; }

.badge-updated { background:#1a4731; color:#3fb950; padding:2px 8px; border-radius:12px; font-size:0.75rem; }
.badge-pending { background:#3d2c08; color:#d29922; padding:2px 8px; border-radius:12px; font-size:0.75rem; }
.badge-bench   { background:#1a2a4a; color:#58a6ff; padding:2px 8px; border-radius:12px; font-size:0.75rem; }
.badge-error   { background:#4d1919; color:#f85149; padding:2px 8px; border-radius:12px; font-size:0.75rem; }

.live-dot {
    display:inline-block; width:8px; height:8px;
    background:#3fb950; border-radius:50%;
    animation:pulse 1.5s infinite; margin-right:6px;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

.topbar {
    display:flex; align-items:center; gap:12px;
    margin-bottom:1rem; padding-bottom:0.5rem;
    border-bottom:1px solid #21262d;
}
.topbar h1 { font-size:1.1rem; font-weight:600; color:#e6edf3; margin:0; }
.topbar .sync-info { font-size:0.75rem; color:#8b949e; }
</style>
"""


def inject_css() -> None:
    st.markdown(_DARK_CSS, unsafe_allow_html=True)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("""
        <div style="padding:1.2rem 0.5rem 0.5rem; display:flex; align-items:center; gap:8px;">
          <span style="font-size:1.3rem;">📈</span>
          <span style="font-size:1rem; font-weight:700; color:#e6edf3; letter-spacing:-0.02em;">
            EquityDesk
          </span>
          <span style="margin-left:auto; font-size:0.65rem; color:#3fb950;">
            <span class="live-dot" style="display:inline-block;width:6px;height:6px;
            background:#3fb950;border-radius:50%;animation:pulse 1.5s infinite;"></span>LIVE
          </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        if "active_page" not in st.session_state:
            st.session_state["active_page"] = "data_ticker"

        for section, pages in NAV.items():
            st.markdown(
                f'<div style="font-size:0.68rem;font-weight:600;letter-spacing:.08em;'
                f'color:#484f58;text-transform:uppercase;padding:0.5rem 0.75rem 0.2rem;">'
                f'{section}</div>',
                unsafe_allow_html=True,
            )
            for label, page_id in pages.items():
                if st.button(label, key=f"nav_{page_id}", use_container_width=True):
                    st.session_state["active_page"] = page_id
                    st.rerun()

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.7rem;color:#484f58;padding:0.5rem;">'
            'NSE · Screener · yFinance · DuckDB</div>',
            unsafe_allow_html=True,
        )
