"""Capsule #1 — Current holdings fetched from connected Kite accounts."""
import streamlit as st
from ui.pages.label_utils import label

def render_kite_portfolio():
    account_names = []
    ticker_data = None
    prev_src = st.session_state.get("pf_prev_src")
    force_pull = prev_src not in (None, "current")
    c1, c2 = st.columns([5, 1])
    if c2.button("🔄 Refresh holdings", key="pf_refresh_holdings", use_container_width=True):
        force_pull = True
    if force_pull and (st.session_state.get("accounts") or []):
        try:
            from ui.pages.trade.trade_utils import fetch_holdings
            fresh, errors = dict(st.session_state.get("all_holdings") or {}), []
            for acc in st.session_state.get("accounts") or []:
                if not acc.get("token"):
                    continue
                try:
                    fresh[acc["name"]] = fetch_holdings(acc["token"])
                except Exception as exc:
                    errors.append(f"{acc.get('name', '?')}: {exc}")
            st.session_state["all_holdings"] = fresh
            if errors:
                c1.warning(label("kite_refresh_failed", errors=" | ".join(errors)))
            else:
                c1.success("🟢 " + label("kite_refresh_success"))
        except Exception as exc:
            c1.warning(label("kite_pull_failed", error=str(exc)))
    all_holdings = st.session_state.get("all_holdings") or {}
    if not all_holdings:
        st.warning(label("kite_no_holdings"))
        return None, []
    excluded = {str(s).upper() for s in (getattr(st.session_state, "excluded_symbols", None) or set())}
    account_names = list(all_holdings.keys())
    ticker_data = {}
    for account_name, holdings in all_holdings.items():
        for holding in holdings:
            symbol = holding["tradingsymbol"]
            if symbol.upper() in excluded:
                continue
            if symbol not in ticker_data:
                ticker_data[symbol] = {
                    "last_price": holding.get("last_price", 0),
                    "close_price": holding.get("close_price", 0),
                    "day_change_pct": holding.get("day_change_percentage", 0),
                    "avg_cost": holding.get("average_price", 0),
                    **{name: 0 for name in account_names},
                }
            ticker_data[symbol][account_name] += holding.get("quantity", 0)
    return ticker_data or None, account_names
