"""
ui/pages/page_config_tickers.py  –  Config › Tickers + Periods
"""

import streamlit as st
import pandas as pd
from bll.config_service import ConfigService
from dal.ticker_config import TICKER_COLS


def render():
    st.header("🗂 Ticker Configuration")
    svc = ConfigService()

    tab_list, tab_add, tab_periods = st.tabs(["Ticker List", "Add Ticker", "Periods"])

    # ── Ticker list ───────────────────────────────────────────────────────────
    with tab_list:
        df = svc.get_tickers()
        st.caption(f"{len(df)} tickers")
        edited = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            key="ticker_editor",
            column_config={
                "market_cap": st.column_config.SelectboxColumn(
                    "market_cap",
                    options=["Large-cap", "Mid-cap", "Small-cap", "Micro-cap"]),
                "domestic_market_pct": st.column_config.NumberColumn(
                    "domestic_market_pct", min_value=0, max_value=100, step=0.1),
                "num_clients": st.column_config.NumberColumn("num_clients", min_value=0, step=1),
                "num_sectors_served": st.column_config.NumberColumn("num_sectors_served", min_value=0, step=1),
            },
        )
        c1, c2 = st.columns([1, 4])
        if c1.button("💾 Save changes", type="primary"):
            svc.save_tickers(edited)
            st.success("Saved.")
            st.rerun()
        if c2.button("⬇ Download CSV"):
            st.download_button(
                "Download tickers.csv",
                data=edited.to_csv(index=False),
                file_name="tickers.csv",
                mime="text/csv",
            )
        st.divider()
        uploaded = st.file_uploader("Replace tickers.csv", type=["csv"])
        if uploaded:
            svc.save_tickers(pd.read_csv(uploaded))
            st.success("Uploaded and saved.")
            st.rerun()

    # ── Add ticker ────────────────────────────────────────────────────────────
    with tab_add:
        st.subheader("Add a new ticker")
        with st.form("add_ticker_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            name       = c1.text_input("Name (Screener symbol)", placeholder="INFY")
            yahoo      = c2.text_input("Yahoo Symbol", placeholder="INFY.NS")
            sector     = c1.text_input("Sector / Theme", placeholder="IT_Largecap")
            market_cap = c2.selectbox("Market cap", ["Large-cap", "Mid-cap", "Small-cap", "Micro-cap"])
            dom_pct    = c1.number_input("Domestic market %", 0.0, 100.0, 100.0, step=0.1)
            n_clients  = c2.number_input("# Clients", 0, 100, 1, step=1)
            n_sectors  = c1.number_input("# Sectors served", 0, 50, 1, step=1)
            if st.form_submit_button("➕ Add Ticker", type="primary") and name:
                svc.add_ticker({
                    "Name": name, "Yahoo Symbol": yahoo or f"{name}.NS",
                    "Sector": sector, "market_cap": market_cap,
                    "domestic_market_pct": dom_pct,
                    "num_clients": int(n_clients), "num_sectors_served": int(n_sectors),
                })
                st.success(f"Added {name}.")
                st.rerun()

    # ── Periods ───────────────────────────────────────────────────────────────
    with tab_periods:
        st.subheader("Training / Testing Periods")
        periods = svc.get_periods()
        for i, p in enumerate(periods):
            c1, c2, c3 = st.columns([3, 1, 1])
            badge = "🟢" if p.get("enabled") else "🔴"
            c1.markdown(
                f"**{badge} {p['name']}**  "
                f"`Train {p['train']['start']} → {p['train']['end']}` "
                f"| `Test {p['test']['start']} → {p['test']['end']}`"
            )
            if c2.button("Toggle", key=f"tog_{i}"):
                svc.toggle_period(p["name"]); st.rerun()
            if c3.button("🗑 Delete", key=f"del_{i}"):
                svc.delete_period(p["name"]); st.rerun()

        st.divider()
        st.subheader("Add Period")
        with st.form("add_period_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            p_name    = c1.text_input("Period name", placeholder="EVT-Iran-War")
            p_enabled = c2.checkbox("Enabled", value=True)
            tr_start  = c1.date_input("Train start")
            tr_end    = c2.date_input("Train end")
            tr_type   = c1.text_input("Train type", value="fy")
            te_start  = c1.date_input("Test start")
            te_end    = c2.date_input("Test end")
            te_type   = c1.text_input("Test type", value="crash")
            if st.form_submit_button("➕ Add Period", type="primary") and p_name:
                svc.add_period({
                    "name": p_name, "enabled": int(p_enabled),
                    "train": {"start": str(tr_start), "end": str(tr_end), "type": tr_type},
                    "test":  {"start": str(te_start), "end": str(te_end), "type": te_type},
                })
                st.success(f"Added period '{p_name}'."); st.rerun()
