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
from ui.pages.screens import page_portfolio_evaluator
from ui.algos import page_algo_momentum, page_algo_ga
from ui.pages import one_ticker_gates

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
page = st.session_state.get("active_page", "kite_setup")

# ── NEW TRADE ROUTER ──
if page == "kite_setup":
    from ui.pages.trade.kite_setup import render; render()
elif page == "trade_portfolio":
    from ui.pages.trade.portfolio import render; render()
elif page == "trade_rebalance_planner":
    from  ui.pages.trade.rebalance_planner import render; render()
elif page == "trade_live_execution":
    from  ui.pages.trade.live_execution import render; render()
elif page == "data_market":
    from ui.pages.data.page_data_market import render; render()
elif page == "data_fundamental":
    from ui.pages.data.page_data_fundamental import render; render()
elif page == "config_tickers":
    from ui.config.page_config_tickers import render; render()
elif page == "config_app":
    from ui.config.page_config_app import render; render()
elif page == "signals":
    from ui.pages.data.page_signals import render; render()
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
elif page == "buy_candidates":
    from ui.pages.screens.page_buy_candidates import render; render()
elif page == "algo_ga":
    page_algo_ga.render()
elif page == "algo_momentum":
    page_algo_momentum.render()
elif page == "algo_fasta":
    from algo.fasta.fasta_ui import render; render()
elif page == "entry_exit":
    from ui.algos.entry_exit_timing import render; render()
elif page == "quality_gates":
    one_ticker_gates.render()
elif page == "ticker_deepdive":
    from ui.pages.screens.page_ticker_deepdive import render; render()
elif page == "sql_analyst":
    from ui.pages.page_sql_analyst import render; render()

# ── MACRO / NEWS SCREENS (new) ────────────────────────────────────────────────
elif page == "us_economy_pipeline":
    from ui.pages.screens.page_us_economy import render; render()
elif page == "news":
    from ui.pages.screens.page_news import render; render()


elif page == "dcf_analysis":
    from ui.pages.screens.dcf_analysis import render; render()

elif page == "todo":
    from ui.config.page_todo import render; render()


