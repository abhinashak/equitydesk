"""Capsule #3 — Manual ticker input and CSV feed upload."""
import io
import pandas as pd
import streamlit as st
from ui.pages.label_utils import label

def render_manual_feed():
    st.markdown(label("manual_help"))
    raw = st.text_area(
        "Tickers", value=st.session_state.get("manual_tickers_raw", ""), height=120,
        placeholder=label("manual_placeholder"), label_visibility="collapsed",
        key="manual_tickers_input",
    )
    st.session_state.manual_tickers_raw = raw
    symbols = [s.strip().upper() for s in raw.replace("\n", ",").split(",") if s.strip()]
    if not symbols:
        st.info(label("manual_ticker_required"))
        return None, ["Manual"]
    data = {sym: {"last_price": 0, "close_price": 0, "day_change_pct": 0,
                  "avg_cost": 0, "Manual": 1} for sym in symbols}
    return data, ["Manual"]

def render_file_upload(live_price_map, analyze_tradebook):
    st.markdown(label("upload_instructions"))
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="holdings_csv_upload",
                                label_visibility="collapsed")
    if uploaded is not None:
        st.session_state["pf_upload_bytes"] = uploaded.getvalue()
        st.session_state["pf_upload_name"] = uploaded.name
    elif st.session_state.get("pf_upload_bytes"):
        uploaded = io.BytesIO(st.session_state["pf_upload_bytes"])
        uploaded.name = st.session_state.get("pf_upload_name", "uploaded.csv")
        c1, c2 = st.columns([5, 1])
        c1.caption("📎 " + label("upload_previous_file", filename=uploaded.name))
        if c2.button("✖ Remove", key="pf_upload_clear", use_container_width=True):
            st.session_state.pop("pf_upload_bytes", None)
            st.session_state.pop("pf_upload_name", None)
            st.rerun()
    if uploaded is None:
        return None, None
    try:
        udf = pd.read_csv(uploaded, encoding="utf-8-sig")
        udf.columns = [str(c).strip().lower() for c in udf.columns]
        ticker_col = next((c for c in ["ticker", "symbol", "nse_symbol"] if c in udf.columns), None)
        if {"symbol", "trade_type", "quantity", "price"}.issubset(udf.columns):
            live, live_errors = live_price_map()
            ticker_data, tracking = analyze_tradebook(udf, live=live)
            if tracking is None:
                st.warning(label("upload_no_valid_trades"))
            else:
                tracking["meta"]["errors"] = live_errors
                if not ticker_data:
                    st.warning(label("upload_no_open_positions"))
            return ticker_data, tracking
        if ticker_col is None:
            st.error(label("upload_csv_column_required"))
            return None, None
        udf["_sym"] = udf[ticker_col].astype(str).str.strip().str.upper()
        udf["_sym"] = udf["_sym"].str.replace(r"\.(NS|BO)$", "", regex=True)
        live, _live_errors = live_price_map()
        ticker_data = {}
        for _, row in udf.iterrows():
            sym = row["_sym"]
            if not sym or sym == "NAN":
                continue
            qty = float(row["quantity"]) if "quantity" in row and pd.notna(row["quantity"]) else 1.0
            avg_cost = float(row["avg_cost"]) if "avg_cost" in row and pd.notna(row["avg_cost"]) else 0.0
            live_row = live.get(sym)
            ticker_data[sym] = {
                "last_price": live_row["last_price"] if live_row else 0,
                "close_price": live_row["close_price"] if live_row else 0,
                "day_change_pct": live_row["day_change_pct"] if live_row else 0,
                "avg_cost": avg_cost, "Uploaded": qty,
            }
        missing = [symbol for symbol in ticker_data if symbol not in live]
        if not live:
            st.warning(label("upload_no_kite_day_change"))
        elif missing:
            st.caption("⚠️ " + label("upload_no_live_symbols", symbols=", ".join(missing)))
        if not ticker_data:
            st.warning(label("upload_no_valid_tickers"))
            return None, None
        return ticker_data, None
    except Exception as exc:
        st.error(label("upload_parse_error", error=str(exc)))
        return None, None
