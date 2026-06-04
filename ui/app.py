"""
ui/app.py  –  EquityDesk  (Streamlit entry point)
──────────────────────────────────────────────────
Run:
    streamlit run ui/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from bll.config_service import ConfigService
from ui.layout import inject_css, render_sidebar
from ui.pages import page_sql_lab
from ui.pages import page_portfolio_evaluator
from ui.pages import page_algo_ga 
from ui.pages import one_ticker_gates
from ui.pages import one_ticker_charts

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EquityDesk",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Bootstrap session state ───────────────────────────────────────────────────
if "app_cfg" not in st.session_state:
    svc = ConfigService()
    st.session_state["app_cfg"] = svc.get_app_config()

inject_css()
render_sidebar()

# ── Page router ───────────────────────────────────────────────────────────────
page = st.session_state.get("active_page", "data_ticker")

if page == "data_ticker":
    from ui.pages.page_data_ticker import render; render()
elif page == "data_fundamental":
    from ui.pages.page_data_fundamental import render; render()
elif page == "data_screener":
    from ui.pages.page_data_screener import render; render()
elif page == "config_tickers":
    from ui.pages.page_config_tickers import render; render()
elif page == "config_app":
    from ui.pages.page_config_app import render; render()
elif page == "signals":
    from ui.pages.page_signals import render; render()
elif page == "trade_account":
    from ui.pages.page_trade import render_account; render_account()
elif page == "trade_positions":
    from ui.pages.page_trade import render_positions; render_positions()
elif page == "trade_plan":
    from ui.pages.page_trade import render_plan; render_plan()
elif page == "sql_lab":
    page_sql_lab.render()
elif page == "portfolio_evaluator":
    page_portfolio_evaluator.render()
elif page == "algo_ga":
    page_algo_ga.render()
elif page == "quality_gates":
    one_ticker_gates.render()
elif page == "fundamental_analysis":
    one_ticker_charts.render()

# ── NEW TRADE ROUTER ──
elif page == "kite_setup":
    from ui.pages.trade.kite_setup import render; render()
elif page == "trade_portfolio":
    from ui.pages.trade.portfolio import render; render()
elif page == "trade_execution":
    from ui.pages.page_3_execution import render; render()
elif page == "dcf_analysis":
    from ui.pages.dcf_analysis import render; render()


