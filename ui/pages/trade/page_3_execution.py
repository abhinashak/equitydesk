import streamlit as st
import pandas as pd
from ui.pages.trade_utils import (
    init_trade_session, MOCK_MODE, fetch_price_yf, HAS_YF, get_fluctuation_pct,
    evaluate_and_place, fetch_holdings, cooling_off_remaining, now_ist_str,
    save_live_orders, load_live_orders, state_label, fetch_trades
)

def render():
    init_trade_session()
    
    if not st.session_state.staged_orders and not st.session_state.live_orders:
        st.info("No orders staged. Please run the Rebalancer first to generate execution plans.")
        st.stop()

    st.title("⚡ Live Execution Manager")
    account_options = [a["name"] for a in st.session_state.accounts if a["token"]]

    if st.session_state.staged_orders and not st.session_state.live_orders:
        st.subheader("📋 Pre-Execution Review")
        ctl1, ctl2, ctl3 = st.columns([1, 1, 3])
        if ctl1.button("🗑 Clear Orders", use_container_width=True):
            st.session_state.staged_orders = []; st.rerun()
        mock_mode = ctl2.toggle("🧪 Mock execution", value=MOCK_MODE, key="mock_mode_toggle")
        
        bulk_account = ctl3.selectbox("Assign Account to All", [""] + account_options, label_visibility="collapsed")
        if ctl3.button("Apply Selected Account"):
            holding_qty_map = {h["tradingsymbol"]: h.get("quantity", 0) for h in st.session_state.all_holdings.get(bulk_account, [])}
            updated = []
            for o in st.session_state.staged_orders:
                o["Account"] = bulk_account
                if o["Side"] == "SELL" and o["Account"] == bulk_account:
                    available = holding_qty_map.get(o["Symbol"], 0)
                    if available <= 0: o["_sell_cap_warn"] = True
                    else:
                        o["_orig_qty"] = o.get("_orig_qty", o["Qty"]); o["Qty"] = min(o["_orig_qty"], available)
                        o["Value ₹"] = round(o["Qty"] * o["Price"], 2)
                updated.append(o)
            st.session_state.staged_orders = updated; st.rerun()

        orders_df = pd.DataFrame(st.session_state.staged_orders)
        edited_orders = st.data_editor(
            orders_df[["Symbol","Exchange","Tier","Side","Qty","Price","Value ₹","Fluc%","Min Hours","Last Filled","Account","state"]],
            column_config={"Price": st.column_config.NumberColumn("Price ₹", format="₹%.2f"), "Value ₹": st.column_config.NumberColumn("Value ₹", format="₹%.0f"), "Account": st.column_config.SelectboxColumn("Account", options=account_options)},
            disabled=["Symbol","Exchange","Tier","Qty","Fluc%","state"], use_container_width=True, hide_index=True
        )
        for col in ["Price","Min Hours","Last Filled","Account","Side"]:
            if col in edited_orders.columns: orders_df[col] = edited_orders[col].values
        st.session_state.staged_orders = orders_df.to_dict("records")

        if st.button("⚡ Place All Orders", type="primary", use_container_width=True):
            token_map = {a["name"]: a["token"] for a in st.session_state.accounts}
            ltp_map = {h["tradingsymbol"]: h.get("last_price", 0) for hlds in st.session_state.all_holdings.values() for h in hlds}
            max_sell_qty_map = {}
            for hlds in st.session_state.all_holdings.values():
                for h in hlds: max_sell_qty_map[h["tradingsymbol"]] = max_sell_qty_map.get(h["tradingsymbol"], 0) + h.get("quantity", 0)
            
            result = evaluate_and_place(st.session_state.staged_orders, token_map, ltp_map, mock=mock_mode, max_sell_qty_map=max_sell_qty_map)
            st.session_state.live_orders = st.session_state.staged_orders = result
            st.session_state.last_poll_time = now_ist_str()
            st.rerun()

    if st.session_state.live_orders:
        mc1, mc2, mc3, mc4 = st.columns(4)
        if mc1.button("🔄 Poll Market Now", use_container_width=True):
            token_map = {a["name"]: a["token"] for a in st.session_state.accounts}
            fetched = {}
            for acc in st.session_state.accounts:
                if acc["token"]:
                    try: fetched[acc["name"]] = fetch_holdings(acc["token"])
                    except: pass
            if fetched: st.session_state.all_holdings = fetched
            
            ltp_map = {h["tradingsymbol"]: h.get("last_price", 0) for hlds in st.session_state.all_holdings.values() for h in hlds}
            max_sell_qty_map = {}
            for hlds in st.session_state.all_holdings.values():
                for h in hlds: max_sell_qty_map[h["tradingsymbol"]] = max_sell_qty_map.get(h["tradingsymbol"], 0) + h.get("quantity", 0)
                
            pre_processed_orders = []
            for o in st.session_state.live_orders:
                o_copy = dict(o)
                if o_copy.get("state") == "COOLING_OFF" and cooling_off_remaining(o_copy) <= 0: o_copy["state"] = "PENDING"
                pre_processed_orders.append(o_copy)
                
            st.session_state.live_orders = evaluate_and_place(pre_processed_orders, token_map, ltp_map, mock=st.session_state.get("mock_mode_toggle", MOCK_MODE), max_sell_qty_map=max_sell_qty_map)
            st.session_state.last_poll_time = now_ist_str()
            st.rerun()
            
        if mc2.button("🔁 Reconcile Fills", use_container_width=True):
            token_map = {a["name"]: a["token"] for a in st.session_state.accounts}
            filled_ids = set()
            for acc in st.session_state.accounts:
                if acc["token"]:
                    try: filled_ids.update(t["order_id"] for t in fetch_trades(acc["token"], mock=st.session_state.get("mock_mode_toggle", MOCK_MODE), live_orders=st.session_state.live_orders))
                    except: pass
            updated = []
            for o in st.session_state.live_orders:
                if o.get("state") == "PLACED" and o.get("order_id") in filled_ids:
                    o["state"], o["filled_at"] = "FILLED", now_ist_str()
                updated.append(o)
            st.session_state.live_orders = updated; st.rerun()

        if mc3.button("💾 Save to Disk", use_container_width=True): save_live_orders(st.session_state.live_orders); st.toast("Saved!")
        if mc4.button("📂 Load Saved", use_container_width=True):
            saved = load_live_orders()
            if saved: st.session_state.live_orders = saved; st.rerun()

        st.caption(f"Last poll: **{st.session_state.get('last_poll_time') or '—'}**")

        live_df = pd.DataFrame(st.session_state.live_orders)
        live_df["State"] = live_df.apply(lambda r: state_label(r.to_dict()), axis=1)
        
        pending_df = live_df[live_df["state"].isin(["PENDING","COOLING_OFF","SKIPPED"])].copy()
        st.subheader(f"📋 Planned Orders  ({len(pending_df)} remaining)")
        if pending_df.empty: st.success("All orders have been placed or completed.")
        else: st.dataframe(pending_df[["Symbol","Tier","Side","Qty","Price","Value ₹","Account","State"]], use_container_width=True, hide_index=True)

        st.subheader("🟢 Active BUY Orders")
        st.dataframe(live_df[(live_df["Side"] == "BUY") & (live_df["state"].isin(["PLACED","FILLED","CANCELLED","ERROR"]))][["Symbol","Tier","Qty","Price","Account","State","order_id"]], use_container_width=True, hide_index=True)

        st.subheader("🔴 Active SELL Orders")
        st.dataframe(live_df[(live_df["Side"] == "SELL") & (live_df["state"].isin(["PLACED","FILLED","CANCELLED","ERROR"]))][["Symbol","Tier","Qty","Price","Account","State","order_id"]], use_container_width=True, hide_index=True)
