"""
ui/pages/page_trade.py  –  Trade › Account | Positions | Execution Plan
"""

import streamlit as st
from bll.trade_service import TradeService
from bll.portfolio_service import PortfolioService
from bll.config_service import ConfigService


def _get_svc() -> TradeService:
    cfg = st.session_state.get("app_cfg") or ConfigService().get_app_config()
    return TradeService(
        kite_base_url=cfg.get("KITE_BASE_URL", "http://localhost:8080"),
        live_orders_file=cfg.get("LIVE_ORDERS_FILE", "data/live_orders.json"),
        mock_mode=cfg.get("MOCK_MODE", "true").lower() == "true",
    )


def render_account():
    st.header("🏦 Account")
    svc     = _get_svc()
    summary = svc.get_account_summary()
    for k, v in summary.items():
        st.metric(k, v)


def render_positions():
    st.header("📋 Current Positions")
    svc = _get_svc()
    df  = svc.get_current_positions()
    if df.empty:
        st.info("No open positions (mock mode).")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_plan():
    st.header("🎯 Execution Plan")
    st.caption("Builds a phased buy/sell plan from target weights vs current positions.")

    port_svc = PortfolioService()
    svc      = _get_svc()
    weights  = port_svc.load_weights()

    if not weights:
        st.warning("No weights found in `data/target_weights.txt`. "
                   "Add tickers + weights to generate a plan.")
        return

    col1, col2 = st.columns(2)
    capital = col1.number_input("Total capital (₹)", value=1_000_000.0, step=50_000.0)
    phases  = col2.number_input("Execution phases", min_value=1, max_value=10, value=3)

    if st.button("📐 Build Plan", type="primary"):
        # Mock prices for now; replace with live price lookup
        mock_prices   = {t: 500.0 for t in weights}
        target_df     = svc.compute_target_positions(weights, capital, mock_prices)
        target_qty    = dict(zip(target_df["ticker"], target_df["target_qty"]))
        plan_df       = svc.build_execution_plan(
            current={}, target=target_qty, prices=mock_prices, phases=phases
        )
        if plan_df.empty:
            st.success("No trades needed — already at target.")
        else:
            st.dataframe(plan_df, use_container_width=True, hide_index=True)
            st.caption(f"Total estimated fees: ₹{plan_df['fees_total'].sum():,.2f}")
