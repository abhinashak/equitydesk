"""
live_execution.py — Live Order Execution
─────────────────────────────────────────
3 sections:
  1. Exchange Orders  — open/executed from Kite today, with refresh + cancel
  2. Order Builder    — MARKET or LIMIT, price tiers per ticker, fee summary
  3. Order Staging    — editable list, send 1-by-1 or send all to exchange
"""
from __future__ import annotations

import json, os, random, time
from datetime import datetime, timezone, timedelta

import pandas as pd
import streamlit as st


# ── import trade_utils — all Kite API calls go through here ──────────────────
from ui.pages.trade.trade_utils import (
    fetch_trades, fetch_orders, place_order, cancel_order,
    get_fluctuation_pct, fetch_price_yf, calc_fees,
    save_live_orders, load_live_orders, kite_headers,
    geo_buy_orders, geo_sell_orders,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants / paths
# ─────────────────────────────────────────────────────────────────────────────
PLAN_FILE        = os.path.join("outputs", "rebalance_plan.json")
LIVE_ORDERS_FILE = os.path.join("outputs", "live_orders.json")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers (used when trade_utils not available)
# ─────────────────────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def _save_orders(orders: list):
    save_live_orders(orders)

def _load_orders() -> list:
    return load_live_orders()

def _load_plan() -> list:
    if not os.path.exists(PLAN_FILE): return []
    try:
        with open(PLAN_FILE) as f:
            p = json.load(f)
        saved = datetime.fromisoformat(p["saved_at"])
        if datetime.now(timezone.utc) - saved > timedelta(hours=24): return []
        return p.get("plan", [])
    except: return []

def _kite_hdr(token: str) -> dict:
    return kite_headers(token)

@st.cache_data(ttl=300, show_spinner=False)
def _ltp(symbol: str, exchange: str = "NSE") -> float:
    return fetch_price_yf(symbol, exchange)

@st.cache_data(ttl=300, show_spinner=False)
def _fluc(symbol: str, exchange: str = "NSE") -> float:
    return get_fluctuation_pct(symbol, exchange)

def _fees(buy_val: float, sell_val: float, sell_scrips: int) -> dict:
    return calc_fees(buy_val, sell_val, sell_scrips)

def _build_orders(sym: str, exch: str, side: str, ltp: float,
                  qty: int, fluc: float, mode: str) -> list[dict]:
    """MARKET = single tier at LTP. LIMIT = geo-progression via trade_utils."""
    if mode == "MARKET":
        return [{"ticker": sym, "exchange": exch, "side": side,
                 "qty": qty, "price": ltp, "value": round(qty * ltp, 2),
                 "order_type": "MARKET", "tier": 1, "fluc_pct": 0.0,
                 "account": "", "state": "PENDING",
                 "order_id": "", "placed_at": "", "filled_at": "",
                 "skip_reason": "", "error": ""}]
    # LIMIT — geo progression from trade_utils
    raw = geo_buy_orders(sym, exch, ltp, qty, fluc) if side == "BUY" \
        else geo_sell_orders(sym, exch, ltp, qty, fluc)
    out = []
    for o in raw:
        out.append({
            "ticker":      o.get("Symbol", sym),
            "exchange":    o.get("Exchange", exch),
            "side":        o.get("Side", side),
            "qty":         o.get("Qty", 0),
            "price":       o.get("Price", ltp),
            "value":       o.get("Value ₹", 0),
            "order_type":  "LIMIT",
            "tier":        o.get("Tier", 1),
            "fluc_pct":    o.get("Fluc%", round(fluc, 3)),
            "account":     "",
            "state":       "PENDING",
            "order_id":    "",
            "placed_at":   "",
            "filled_at":   "",
            "skip_reason": "",
            "error":       "",
        })
    return out

def _do_place(token: str, o: dict, mock: bool) -> dict:
    """Route all order placement through trade_utils → proxy."""
    price = float(o["price"])
    if o.get("order_type", "LIMIT") == "LIMIT":
        price = round(round(price / 0.10) * 0.10, 2)
    print(f"place_order {o} (rounded price={price})")
    return place_order(
        token, o["ticker"], o["exchange"], o["side"],
        int(o["qty"]), price,
        order_type=o.get("order_type", "LIMIT"),
        mock=mock,
    )

def _do_cancel(token: str, order_id: str, mock: bool) -> dict:
    """Route all cancellations through trade_utils → proxy."""
    return cancel_order(token, order_id, mock=mock)

def _normalise_kite_order(o: dict) -> dict:
    """Map Kite API field names to internal names for display."""
    return {
        "Time":       o.get("order_timestamp","") or o.get("fill_timestamp","") or o.get("placed_at",""),
        "Ticker":     o.get("ticker") or o.get("tradingsymbol",""),
        "Type":       o.get("side")   or o.get("transaction_type",""),
        "Product":    o.get("product","CNC"),
        "Qty":        f"{o.get('filled_quantity',0)} / {o.get('qty') or o.get('quantity',0)}",
        "LTP":        o.get("last_price",""),
        "Price":      o.get("price") or o.get("average_price",0),
        "Status":     o.get("status","") or o.get("state",""),
        "Order ID":   o.get("order_id",""),
        "Tier":       o.get("tier",""),
    }

# ─────────────────────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────────────────────
def render() -> None:

    for k, v in [("live_orders",[]), ("staged_orders",[]),
                 ("le_staged_built", False)]:
        if k not in st.session_state: st.session_state[k] = v

    accounts   = st.session_state.get("accounts") or []
    token_map  = {a["name"]: a["token"] for a in accounts if a.get("token")}
    acct_opts  = list(token_map.keys())
    first_tok  = next(iter(token_map.values()), "")

    st.markdown("## ⚡ Live Execution")
    mock_mode = st.toggle("🧪 Mock Mode", value=False, key="le_mock",
                          help="No real orders sent when ON")
    if mock_mode:
        st.caption("⚠️ Mock mode ON — orders are simulated.")

    # ═══════════════════════════════════════════════════════════════════════
    # PART 1 — EXCHANGE ORDERS  (open + executed today from Kite)
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("### 1️⃣ Exchange Orders")

    p1h, p1r = st.columns([5, 1])
    p1h.markdown("Open, executed and cancelled orders for today from the exchange.")
    refresh = p1r.button("🔄 Refresh", key="le_p1_refresh", use_container_width=True)
    if refresh:
        _ltp.clear(); _fluc.clear()
        st.session_state.pop("le_exchange_orders", None)
        st.rerun()

    # Fetch from exchange (or mock)
    if "le_exchange_orders" not in st.session_state or refresh:
        if first_tok:
            try:
                # fetch_orders → proxy /orders  (open + all today's terminal orders)
                raw = fetch_orders(first_tok, mock=mock_mode,
                                   live_orders=st.session_state["live_orders"])
            except Exception as e:
                st.error(f"Exchange error: {e}")
                raw = st.session_state["live_orders"]
        else:
            raw = st.session_state["live_orders"]
        st.session_state["le_exchange_orders"] = raw

    ex_orders = st.session_state.get("le_exchange_orders", [])

    if not ex_orders:
        st.caption("No orders found on exchange today.")
    else:
        norm = [_normalise_kite_order(o) for o in ex_orders]
        ex_df = pd.DataFrame(norm)

        # Colour rows by type
        def _clr(row):
            c = "background-color:#dcfce7;color:#15803d" if str(row.get("Type","")).upper()=="BUY" \
                else "background-color:#fee2e2;color:#b91c1c" if str(row.get("Type","")).upper()=="SELL" \
                else ""
            return [c]*len(row)

        open_df = ex_df[ex_df["Status"].str.upper().isin(["OPEN","TRIGGER PENDING","PENDING","AMO REQ RECEIVED"])] \
            if "Status" in ex_df.columns else pd.DataFrame()
        done_df = ex_df[~ex_df.index.isin(open_df.index)] if not open_df.empty else ex_df

        tab_open, tab_done = st.tabs([
            f"🟡 Open ({len(open_df)})",
            f"✅ Executed / Cancelled ({len(done_df)})"
        ])

        col_cfg = {"Price": st.column_config.NumberColumn(format="₹%.2f"),
                   "LTP":   st.column_config.NumberColumn(format="₹%.2f")}

        with tab_open:
            if open_df.empty:
                st.caption("No open orders.")
            else:
                st.dataframe(open_df.style.apply(_clr, axis=1),
                             use_container_width=True, hide_index=True,
                             column_config=col_cfg)

        with tab_done:
            if done_df.empty:
                st.caption("No executed/cancelled orders.")
            else:
                st.dataframe(done_df.style.apply(_clr, axis=1),
                             use_container_width=True, hide_index=True,
                             column_config=col_cfg)

    # Cancel orders — source of truth is the exchange, not session state
    live_now = st.session_state.get("live_orders", [])
    CANCELLABLE = {"OPEN", "TRIGGER PENDING", "AMO REQ RECEIVED"}
    ex_open  = [o for o in ex_orders
                if str(o.get("status") or "").upper() in CANCELLABLE]
    has_open = bool(ex_open)

    ca1, ca2 = st.columns([1, 3])

    # ── Cancel ALL ─────────────────────────────────────────────────────────
    if ca1.button("🚫 Cancel ALL Open Orders", disabled=not has_open,
                  key="le_cancel_all", type="secondary", use_container_width=True):
        cancelled, failed = 0, 0
        for o in ex_open:
            oid = o.get("order_id", "")
            if not oid:
                continue
            try:
                _do_cancel(first_tok, oid, mock_mode)
                cancelled += 1
            except Exception as e:
                st.warning(f"❌ {o.get('tradingsymbol','')} ({oid}): {e}")
                failed += 1
        st.session_state.pop("le_exchange_orders", None)
        st.success(f"✅ Cancelled {cancelled} order(s)." + (f" {failed} failed." if failed else ""))
        st.rerun()

    # ── Cancel specific order by ID ────────────────────────────────────────
    if ex_open:
        with ca2.expander(f"🎯 Cancel specific order ({len(ex_open)} open)"):
            opts = {
                f"{o.get('tradingsymbol','')} · {o.get('transaction_type','')} · "
                f"{o.get('quantity','')} @ ₹{o.get('price','')} · {o.get('order_id','')}": o.get("order_id","")
                for o in ex_open
            }
            chosen_label = st.selectbox("Select order", list(opts.keys()),
                                        key="le_cancel_specific_sel",
                                        label_visibility="collapsed")
            if st.button("🚫 Cancel Selected", key="le_cancel_specific_btn",
                         type="secondary", use_container_width=True):
                oid = opts.get(chosen_label, "")
                if oid:
                    try:
                        _do_cancel(first_tok, oid, mock_mode)
                        st.session_state.pop("le_exchange_orders", None)
                        st.success(f"✅ Cancelled {oid}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed: {e}")

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════
    # PART 2 — ORDER BUILDER
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("### 2️⃣ Order Builder")

    # ── Source: prefer execution_plan (set by rebalance_planner Execute Plan)
    #           fallback to rebalance_plan (legacy) or disk
    if st.session_state.get("execution_plan_ready"):
        plan = st.session_state.get("execution_plan", [])
        st.success("🔥 Pulling orders from the active Live Execution Queue.", icon="🚀")
    else:
        plan = st.session_state.get("rebalance_plan", [])

    if plan:
        st.info("📋 Pulling orders from the Saved Staging Plan (Requires manual confirmation).")
        n_buy  = sum(1 for o in plan if o.get("action","").upper() == "BUY")
        n_sell = sum(1 for o in plan if o.get("action","").upper() == "SELL")
        st.success(
            f"✅ Execution plan loaded — **{len(plan)} orders** "
            f"({n_buy} BUY · {n_sell} SELL). Review and build below.",
            icon="🚀",
        )

    else :
        st.info("No approved plan in session. Go to the **Rebalancer** page and click **Execute Plan**.")
        return

    # ── Global controls ────────────────────────────────────────────────────
    g1, g2, g3, g4 = st.columns([2, 1, 1, 2])
    order_mode  = g1.radio("Order type", ["LIMIT", "STAGGER", "MARKET"],
                           horizontal=True, key="le_order_mode",
                           help=(
                               "**LIMIT** — single limit order at a % offset from LTP.\n\n"
                               "**STAGGER** — geo-progression tiers (uses trade_utils).\n\n"
                               "**MARKET** — market order at LTP."
                           ))
    bulk_acct   = g2.selectbox(
        "🏦 Target Account",
        [""] + acct_opts,
        format_func=lambda x: "— select account —" if x == "" else x,
        key="le_bulk_acct")
    if not bulk_acct:
        g2.warning("⚠️ Select an account — SELL qty depends on its holdings.")

    # LIMIT: single offset % from LTP  (editable per-row later)
    limit_offset_pct = 0.0
    if order_mode == "LIMIT":
        limit_offset_pct = g3.number_input(
            "Limit offset %", min_value=-5.0, max_value=5.0,
            value=0.25, step=0.05, format="%.2f",
            key="le_limit_offset",
            help="BUY: LTP − offset  |  SELL: LTP + offset. Negative = aggressive.",
        )
    elif order_mode == "STAGGER":
        g3.caption("Tiers built from **Day Fluc%** via trade_utils.")

    # ── Compute already-accounted qty per ticker (FILLED + PLACED/OPEN) ───
    # Both filled and open/placed orders should reduce Send Qty to avoid double-sending.
    filled_map: dict = {}
    for o in live_now:
        if o.get("state") in ("FILLED", "PLACED"):
            k = f"{o.get('ticker','')}_{o.get('side','').upper()}"
            filled_map[k] = filled_map.get(k, 0) + int(o.get("qty", 0))

    # ── Fetch / cache LTP + day-fluc ──────────────────────────────────────
    if "le_prices" not in st.session_state:
        tickers = list({r["ticker"] for r in plan})
        hold_ltp: dict = {}
        for hlds in (st.session_state.get("all_holdings") or {}).values():
            for h in hlds:
                sym = h.get("tradingsymbol", "").upper()
                if sym:
                    hold_ltp[sym] = float(h.get("last_price") or 0)
        prog = st.progress(0, text="Fetching prices…")
        rows = []
        for i, tk in enumerate(tickers):
            prog.progress((i + 1) / max(len(tickers), 1), text=f"{tk}…")
            ltp  = hold_ltp.get(tk) or _ltp(tk)
            fluc = _fluc(tk)
            rows.append({"Ticker": tk, "LTP ₹": round(ltp, 2), "Day Fluc%": round(fluc, 3)})
        prog.empty()
        st.session_state["le_prices"] = rows

    prices = {r["Ticker"]: r for r in st.session_state["le_prices"]}

    # ── Resolve account for this builder pass ─────────────────────────────
    # bulk_acct is chosen in the UI; fall back to the first available account.
    order_account = bulk_acct or (acct_opts[0] if acct_opts else "")

    # ── Portfolio qty map  {ticker: qty_held}  for the selected account ───
    # Source: all_holdings keyed by account name.
    portfolio_qty_map: dict[str, int] = {}
    all_holdings = st.session_state.get("all_holdings") or {}
    # Prefer the selected account; if absent, merge all accounts (conservative).
    holdings_src = (all_holdings.get(order_account) or
                    [h for hlds in all_holdings.values() for h in hlds])
    for h in holdings_src:
        sym = (h.get("tradingsymbol") or h.get("ticker") or "").upper()
        if sym:
            portfolio_qty_map[sym] = portfolio_qty_map.get(sym, 0) + int(
                h.get("quantity") or h.get("qty") or 0
            )

    # ── Open-order qty map from Exchange Orders (both BUY and SELL) ────────
    # Source: exchange orders fetched in Part 1 that are still OPEN/PLACED.
    # These count against Send Qty just like filled orders — to avoid double-sending.
    SENT_STATUSES = {"OPEN", "TRIGGER PENDING", "AMO REQ RECEIVED", "COMPLETE"}
    exch_sent_sell_map: dict[str, int] = {}  # COMPLETE + OPEN on exchange
    exch_sent_buy_map:  dict[str, int] = {}  # COMPLETE + OPEN on exchange
    for o in ex_orders:
        # ex_orders are raw Kite dicts — use lowercase keys; also handle normalised keys
        status = str(o.get("status") or o.get("Status") or o.get("state") or "").upper()
        ttype  = str(o.get("transaction_type") or o.get("Type") or o.get("side") or "").upper()
        if status not in SENT_STATUSES:
            continue
        sym = (o.get("tradingsymbol") or o.get("Ticker") or o.get("ticker") or "").upper()
        if not sym:
            continue
        try:
            # Raw Kite API uses 'quantity' (total ordered); never 'Qty'
            # Fall back to normalised "filled/total" string only if raw key missing
            raw_qty = o.get("quantity") or o.get("Qty") or "0"
            qty_str = str(raw_qty)
            # Handle "filled / total" format from _normalise_kite_order
            qty_val = int(qty_str.split("/")[-1].strip())
        except (ValueError, AttributeError):
            qty_val = 0
        if ttype == "SELL":
            exch_sent_sell_map[sym] = exch_sent_sell_map.get(sym, 0) + qty_val
        elif ttype == "BUY":
            exch_sent_buy_map[sym]  = exch_sent_buy_map.get(sym, 0)  + qty_val

    # ── Build the Order Builder table ──────────────────────────────────────
    builder_rows: list[dict] = []
    for row in plan:
        tk       = row["ticker"]
        side     = (row.get("action") or row.get("side", "")).upper()
        plan_qty = int(row.get("today_qty", row.get("qty", 0)))

        # ask_qty: always a positive number (exchange always receives positive qty)
        ask_qty = abs(plan_qty)

        filled       = filled_map.get(f"{tk}_{side}", 0)

        if side == "SELL":
            # COMPLETE + OPEN on exchange = total already sent
            already_sold  = exch_sent_sell_map.get(tk, 0)
            portfolio_qty = portfolio_qty_map.get(tk, ask_qty)
            rebalance_remaining = max(0, ask_qty - already_sold)
            # Never sell more than what we hold
            residual = max(0, min(rebalance_remaining, portfolio_qty))
        else:  # BUY
            # COMPLETE + OPEN on exchange = total already sent
            already_bought = exch_sent_buy_map.get(tk, 0)
            residual = max(0, ask_qty - already_bought)

        if residual == 0:
            continue

        pi    = prices.get(tk, {})
        ltp   = float(pi.get("LTP ₹") or row.get("price", 0))
        fluc  = float(pi.get("Day Fluc%") or 1.0)
        exch  = row.get("exchange", "NSE").upper()

        # Compute exec price based on mode
        if order_mode == "MARKET":
            exec_price = ltp
            tier_str   = f"MARKET @ ₹{ltp:.2f}"
        elif order_mode == "LIMIT":
            sign       = -1 if side == "BUY" else +1
            exec_price = round(ltp * (1 + sign * limit_offset_pct / 100), 2)
            tier_str   = f"LIMIT 1 order @ ₹{exec_price:.2f}  ({'+' if sign>0 else ''}{sign*limit_offset_pct:.2f}%)"
        else:  # STAGGER
            x    = max(1, residual // 15)
            qtys = [x, 2 * x, 4 * x, 8 * x]
            if side == "BUY":
                tiers = [round(ltp * (1 - 0.0025 - f / 100), 2)
                         for f in [0, fluc / 4, fluc / 2, fluc * 3 / 4]]
            else:
                tiers = [round(ltp * (1 + 0.0025 + f / 100), 2)
                         for f in [0, fluc / 4, fluc / 2, fluc * 3 / 4]]
            exec_price = tiers[0]
            tier_str   = "  |  ".join(
                f"T{i+1}: {q}@₹{p}" for i, (q, p) in enumerate(zip(qtys, tiers))
            )

        exec_val = round(residual * exec_price, 0)

        builder_rows.append({
            "✓":            True,
            "Account":      order_account,
            "Ticker":       tk,
            "Exchange":     exch,
            "Side":         side,
            "Ask Qty":      ask_qty,
            "Plan Qty":     ask_qty,       # kept for display compatibility (always positive)
            "Filled":       already_sold if side == "SELL" else already_bought,
            "Send Qty":     residual,
            "LTP ₹":        ltp,
            "Exec Price ₹": exec_price,
            "Send Value ₹": exec_val,
            "Day Fluc%":    fluc,
            "Order Type":   "MARKET" if order_mode == "MARKET" else "LIMIT",
            "Tiers":        tier_str,
            "_exchange":    exch,
        })

    # ── All filled? Show summary ───────────────────────────────────────────
    if not builder_rows:
        st.success("🎉 All plan orders already filled!")
        plan_keys = {(r["ticker"], (r.get("action") or r.get("side", "")).upper()) for r in plan}
        fill_rows = []
        for o in live_now:
            tk   = o.get("ticker", "")
            side = (o.get("side") or o.get("action", "")).upper()
            if o.get("state") != "FILLED": continue
            if (tk, side) not in plan_keys: continue
            qty   = int(o.get("qty", 0))
            price = float(o.get("price") or o.get("average_price") or 0)
            fill_rows.append({
                "Ticker":       tk, "Side":     side,
                "Qty":          qty, "Fill Price ₹": round(price, 2),
                "Value ₹":      round(qty * price, 2),
                "Tier":         o.get("tier", ""),
                "Account":      o.get("account", ""),
                "Filled At":    o.get("filled_at", "") or o.get("placed_at", ""),
                "Order ID":     o.get("order_id", ""),
            })
        if fill_rows:
            fill_df = pd.DataFrame(fill_rows)
            def _fill_style(row):
                c = "background-color:#dcfce7;color:#15803d" \
                    if str(row.get("Side","")).upper() == "BUY" \
                    else "background-color:#fee2e2;color:#b91c1c"
                return [c] * len(row)
            st.markdown("##### 📑 Fill Details")
            st.dataframe(fill_df.style.apply(_fill_style, axis=1),
                         use_container_width=True, hide_index=True,
                         column_config={
                             "Fill Price ₹": st.column_config.NumberColumn(format="₹%.2f"),
                             "Value ₹":      st.column_config.NumberColumn(format="₹%.0f"),
                         })
            summary = (fill_df.groupby(["Ticker","Side"])
                       .agg(Total_Qty=("Qty","sum"),
                            Avg_Price=("Fill Price ₹", lambda x: round(
                                (x * fill_df.loc[x.index,"Qty"]).sum()
                                / fill_df.loc[x.index,"Qty"].sum(), 2)
                            if fill_df.loc[x.index,"Qty"].sum() else 0),
                            Total_Value=("Value ₹","sum"),
                            Fills=("Order ID","count"))
                       .reset_index()
                       .rename(columns={"Total_Qty":"Total Qty","Avg_Price":"Avg Fill ₹","Total_Value":"Total Value ₹"})
                       )
            st.markdown("##### 🗂 Summary by Ticker")
            st.dataframe(summary.style.apply(_fill_style, axis=1),
                         use_container_width=True, hide_index=True,
                         column_config={"Avg Fill ₹": st.column_config.NumberColumn(format="₹%.2f"),
                                        "Total Value ₹": st.column_config.NumberColumn(format="₹%.0f")})
            bv = fill_df[fill_df["Side"].str.upper()=="BUY"]["Value ₹"].sum()
            sv = fill_df[fill_df["Side"].str.upper()=="SELL"]["Value ₹"].sum()
            m1, m2, m3 = st.columns(3)
            m1.metric("🟢 BUY filled",  f"₹{bv:,.0f}", f"{int((fill_df['Side'].str.upper()=='BUY').sum())} fills")
            m2.metric("🔴 SELL filled", f"₹{sv:,.0f}", f"{int((fill_df['Side'].str.upper()=='SELL').sum())} fills")
            m3.metric("Net deployed",   f"₹{bv-sv:,.0f}")
        else:
            st.caption("No fill detail found in live orders for this plan.")
        return

    # ── Render editable Order Builder table ───────────────────────────────
    builder_df = pd.DataFrame(builder_rows)

    st.markdown("##### 📋 Order Builder — Review & Edit")
    st.caption(
        "Edit **Send Qty**, **Exec Price ₹**, or **Order Type** per row. "
        "Tick ✓ to include a row in the staged batch. "
        "Price tiers are shown for reference in STAGGER mode."
    )

    # ── Select All / Clear All ─────────────────────────────────────────────
    # Build a stable row key per ticker+side to track checkbox state
    row_keys = [f"{r['Ticker']}_{r['Side']}" for r in builder_rows]

    # Initialise state for any new rows (default: selected)
    check_state: dict = st.session_state.setdefault("le_check_state", {})
    for k in row_keys:
        if k not in check_state:
            check_state[k] = True

    sa1, sa2, _ = st.columns([1, 1, 6])
    if sa1.button("✅ Select All", key="le_select_all", use_container_width=True):
        for k in row_keys:
            check_state[k] = True
    if sa2.button("☐ Clear All", key="le_clear_all", use_container_width=True):
        for k in row_keys:
            check_state[k] = False

    # Apply persisted checkbox state to the dataframe before rendering
    builder_df["✓"] = [check_state.get(k, True) for k in row_keys]

    edited_builder = st.data_editor(
        builder_df[[
            "✓", "Account", "Ticker", "Exchange", "Side",
            "Ask Qty", "Plan Qty", "Filled", "Send Qty",
            "LTP ₹", "Exec Price ₹", "Send Value ₹",
            "Day Fluc%", "Order Type", "Tiers",
        ]],
        use_container_width=True, hide_index=True,
        disabled=["Account", "Ticker", "Exchange", "Side",
                  "Ask Qty", "Plan Qty", "Filled", "LTP ₹", "Send Value ₹",
                  "Day Fluc%", "Tiers"],
        column_config={
            "✓":            st.column_config.CheckboxColumn("Send", width="small"),
            "Account":      st.column_config.TextColumn("Account", width="medium"),
            "Ask Qty":      st.column_config.NumberColumn("Ask Qty",       width="small"),
            "LTP ₹":        st.column_config.NumberColumn("LTP ₹",        format="₹%.2f", width="small"),
            "Exec Price ₹": st.column_config.NumberColumn("Exec Price ₹", format="₹%.2f", width="small"),
            "Send Value ₹": st.column_config.NumberColumn("Value ₹",      format="₹%.0f", width="small"),
            "Day Fluc%":    st.column_config.NumberColumn("Fluc%",         format="%.3f%%", width="small"),
            "Send Qty":     st.column_config.NumberColumn("Send Qty",      step=1,          width="small"),
            "Order Type":   st.column_config.SelectboxColumn(
                "Type", options=["LIMIT", "MARKET"], width="small"),
            "Tiers":        st.column_config.TextColumn("Tier Preview", width="large"),
        },
        key="le_builder_editor",
    )

    # Persist any manual checkbox changes back to session state
    for k, checked in zip(row_keys, edited_builder["✓"].tolist()):
        check_state[k] = checked

    # Recompute Send Value ₹ after any qty/price edits
    edited_builder["Send Value ₹"] = (
            edited_builder["Send Qty"].astype(float) *
            edited_builder["Exec Price ₹"].astype(float)
    ).round(0)

    # ── Refresh price controls ─────────────────────────────────────────────
    rc1, _ = st.columns([1, 5])
    if rc1.button("🔄 Re-fetch Prices", key="le_refetch", use_container_width=True):
        _ltp.clear(); _fluc.clear()
        st.session_state.pop("le_prices", None)
        st.session_state["le_staged_built"] = False
        st.rerun()

    # ── Fee / value summary for selected rows ─────────────────────────────
    selected = edited_builder[edited_builder["✓"] == True].copy()
    buy_sel  = selected[selected["Side"] == "BUY"]
    sell_sel = selected[selected["Side"] == "SELL"]
    buy_val  = (buy_sel["Send Qty"].astype(float) * buy_sel["Exec Price ₹"].astype(float)).sum()
    sell_val = (sell_sel["Send Qty"].astype(float) * sell_sel["Exec Price ₹"].astype(float)).sum()
    fees     = _fees(buy_val, sell_val, len(sell_sel))

    st.markdown("##### 💰 Transaction Fee Estimate (selected rows)")
    fc = st.columns(len(fees))
    for col, (label, amt) in zip(fc, fees.items()):
        col.metric(label, f"₹{amt:,.2f}")
    fm1, fm2, fm3, fm4, fm5 = st.columns(5)
    fm1.metric("🟢 BUY",   f"₹{buy_val:,.0f}",           f"{len(buy_sel)} tickers")
    fm2.metric("🔴 SELL",  f"₹{sell_val:,.0f}",           f"{len(sell_sel)} tickers")
    fm3.metric("Net Outflow", f"₹{buy_val - sell_val:,.0f}")
    fm4.metric("Selected", f"{len(selected)} / {len(edited_builder)} rows")
    fm5.metric("🎯 Unique Tickers", selected["Ticker"].nunique())

    st.divider()

    # ── Build Staged Orders ────────────────────────────────────────────────
    bc1, _ = st.columns([1, 3])
    if bc1.button("⚙️ Build Staged Orders →", type="primary",
                  use_container_width=True, disabled=selected.empty,
                  key="le_build_staged"):
        exch_map = builder_df.set_index("Ticker")["_exchange"].to_dict()
        staged   = []
        for _, r in selected.iterrows():
            sym  = r["Ticker"]
            exch = exch_map.get(sym, "NSE")
            side = r["Side"]
            ltp  = float(r["LTP ₹"])
            fluc = float(r["Day Fluc%"])
            qty  = int(r["Send Qty"])
            otype = r["Order Type"]

            if order_mode == "STAGGER" and otype != "MARKET":
                batch = _build_orders(sym, exch, side, ltp, qty, fluc, "LIMIT")
            else:
                # Single order — use edited exec price
                exec_p = float(r["Exec Price ₹"])
                batch  = [{
                    "ticker":      sym,
                    "exchange":    exch,
                    "side":        side,
                    "qty":         qty,
                    "price":       exec_p,
                    "value":       round(qty * exec_p, 2),
                    "order_type":  otype,
                    "tier":        1,
                    "fluc_pct":    fluc,
                    "account":     bulk_acct,
                    "state":       "PENDING",
                    "order_id":    "",
                    "placed_at":   "",
                    "filled_at":   "",
                    "skip_reason": "",
                    "error":       "",
                }]
            row_account = r.get("Account") or bulk_acct
            for o in batch:
                o["account"] = o.get("account") or row_account
            staged.extend(batch)

        st.session_state["staged_orders"]   = staged
        st.session_state["le_staged_built"] = True
        st.success(f"Built **{len(staged)}** staged orders from **{len(selected)}** rows. Scroll to Part 3 ↓")
        st.rerun()

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════
    # PART 3 — ORDER STAGING  (send 1-by-1 or all)
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("### 3️⃣ Staged Orders")

    staged: list = st.session_state.get("staged_orders", [])
    if not staged:
        st.caption("No staged orders yet. Build them in Part 2 above.")
        return

    pending_staged = [o for o in staged if o.get("state") == "PENDING"]
    st.caption(
        f"**{len(pending_staged)}** pending  ·  "
        f"**{sum(1 for o in staged if o.get('state')=='PLACED')}** placed  ·  "
        f"**{sum(1 for o in staged if o.get('state')=='FILLED')}** filled  ·  "
        f"**{sum(1 for o in staged if o.get('state')=='CANCELLED')}** cancelled"
    )

    # ── Editable staged table with inline actions ─────────────────────────
    st.markdown("##### 📋 Review, Edit & Send")
    st.caption("Edit **Price** or **Qty** directly. Use the action buttons on each row to send or remove.")

    # Apply editor delta from session state before rendering
    updated_staged = [dict(o) for o in staged]
    editor_state = st.session_state.get("le_staged_editor", {})
    for row_str, changes in (editor_state.get("edited_rows") or {}).items():
        i = int(row_str)
        if i >= len(updated_staged): continue
        o = updated_staged[i]
        if "qty"        in changes: o["qty"]        = max(1, int(changes["qty"]))
        if "price"      in changes: o["price"]      = float(changes["price"])
        if "order_type" in changes: o["order_type"] = changes["order_type"]
        o["value"] = round(o["qty"] * o["price"], 2)
    st.session_state["staged_orders"] = updated_staged

    pending_indices = [i for i, o in enumerate(updated_staged) if o.get("state") == "PENDING"]

    # ── Compact number_input CSS — remove padding/margins so inputs match cell width ──
    st.markdown("""
    <style>
    div[data-testid="stNumberInput"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stNumberInput"] > div {
        min-height: unset !important;
    }
    div[data-testid="stNumberInput"] input {
        font-size: 0.82rem !important;
        padding: 6px 8px !important;
        height: 38px !important;
        min-height: 38px !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stNumberInput"] button {
        display: none !important;
    }
    /* Match action buttons to same height */
    div[data-testid="stButton"] button {
        height: 38px !important;
        min-height: 38px !important;
        padding: 0 8px !important;
    }
    /* Remove gap between column rows */
    div[data-testid="stHorizontalBlock"] {
        gap: 0 !important;
        margin-bottom: 0 !important;
    }
    div[data-testid="column"] {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header row ────────────────────────────────────────────────────────
    _BG_HEAD = "background:#f0f2f6;padding:0 6px;font-size:0.78rem;font-weight:600;color:#444;height:38px;display:flex;align-items:center;border-bottom:1px dotted #aaa"
    h0,h1,h2,h3,h4,h5,h6,h7,h8,h9 = st.columns([1,2,1,1,1,1,1,1,1,2])
    h0.markdown(f'<div style="{_BG_HEAD}">#</div>',       unsafe_allow_html=True)
    h1.markdown(f'<div style="{_BG_HEAD}">Ticker</div>',  unsafe_allow_html=True)
    h2.markdown(f'<div style="{_BG_HEAD}">Side</div>',    unsafe_allow_html=True)
    h3.markdown(f'<div style="{_BG_HEAD}">Type</div>',    unsafe_allow_html=True)
    h4.markdown(f'<div style="{_BG_HEAD}">Tier</div>',    unsafe_allow_html=True)
    h5.markdown(f'<div style="{_BG_HEAD}">Qty</div>',     unsafe_allow_html=True)
    h6.markdown(f'<div style="{_BG_HEAD}">Price ₹</div>', unsafe_allow_html=True)
    h7.markdown(f'<div style="{_BG_HEAD}">Value ₹</div>', unsafe_allow_html=True)
    h8.markdown(f'<div style="{_BG_HEAD}">State</div>',   unsafe_allow_html=True)
    h9.markdown(f'<div style="{_BG_HEAD}">Actions</div>', unsafe_allow_html=True)

    # ── One row per staged order ───────────────────────────────────────────
    for i, o in enumerate(updated_staged):
        side      = str(o.get("side","")).upper()
        state     = str(o.get("state","PENDING"))
        is_pending = state == "PENDING"
        row_bg    = "#f0fdf4" if side=="BUY" else "#fff1f2" if side=="SELL" else "#fff"
        _CELL     = f"background:{row_bg};padding:0 6px;font-size:0.82rem;height:38px;display:flex;align-items:center;border-bottom:1px dotted #ddd;margin-bottom:0"

        c0,c1,c2,c3,c4,c5,c6,c7,c8,c9 = st.columns([1,2,1,1,1,1,1,1,1,2])
        c0.markdown(f'<div style="{_CELL}">{i}</div>', unsafe_allow_html=True)
        c1.markdown(f'<div style="{_CELL}"><b>{o.get("ticker","")}</b></div>', unsafe_allow_html=True)
        c2.markdown(f'<div style="{_CELL}">{side}</div>', unsafe_allow_html=True)
        c3.markdown(f'<div style="{_CELL}">{o.get("order_type","LIMIT")}</div>', unsafe_allow_html=True)
        c4.markdown(f'<div style="{_CELL}">{o.get("tier","")}</div>', unsafe_allow_html=True)

        # Editable qty & price only for PENDING rows
        if is_pending:
            new_qty = c5.number_input(
                "qty", value=int(o.get("qty",1)), min_value=1, step=1,
                key=f"le_qty_{i}", label_visibility="collapsed"
            )
            new_price = c6.number_input(
                "price", value=float(o.get("price",0.0)), min_value=0.0, step=0.10, format="%.2f",
                key=f"le_price_{i}", label_visibility="collapsed"
            )
            # Persist edits immediately on any change
            if new_qty != int(o.get("qty",1)) or new_price != float(o.get("price",0.0)):
                updated_staged[i]["qty"]   = new_qty
                updated_staged[i]["price"] = new_price
                updated_staged[i]["value"] = round(new_qty * new_price, 2)
                st.session_state["staged_orders"] = updated_staged
            c7.markdown(f'<div style="{_CELL}">₹{round(new_qty*new_price):,}</div>', unsafe_allow_html=True)
        else:
            c5.markdown(f'<div style="{_CELL}">{o.get("qty","")}</div>', unsafe_allow_html=True)
            c6.markdown(f'<div style="{_CELL}">₹{float(o.get("price",0)):.2f}</div>', unsafe_allow_html=True)
            c7.markdown(f'<div style="{_CELL}">₹{float(o.get("value",0)):,.0f}</div>', unsafe_allow_html=True)

        state_icon = {"PENDING":"⏳","PLACED":"📤","FILLED":"✅","ERROR":"❌","CANCELLED":"🚫"}.get(state, state)
        err_tip    = o.get("error","")
        c8.markdown(f'<div style="{_CELL}" title="{err_tip}">{state_icon} {state}</div>', unsafe_allow_html=True)

        if is_pending:
            btn_a, btn_b = c9.columns(2)
            if btn_a.button("📤", key=f"le_send_{i}", use_container_width=True,
                            help="Send To Exchange"):
                if not bulk_acct and not mock_mode:
                    st.error("Select an account before sending live orders.")
                else:
                    # Apply latest qty/price edits before placing
                    o2 = dict(updated_staged[i])
                    o2["qty"]   = int(st.session_state.get(f"le_qty_{i}",   o2["qty"]))
                    o2["price"] = float(st.session_state.get(f"le_price_{i}", o2["price"]))
                    o2["value"] = round(o2["qty"] * o2["price"], 2)
                    tk = token_map.get(o2.get("account","") or bulk_acct, first_tok)
                    try:
                        resp = _do_place(tk, o2, mock_mode)
                        oid  = resp.get("data",{}).get("order_id","?")
                        o2.update({"state":"PLACED","order_id":oid,"placed_at":_now()})
                        st.success(f"✅ {o2['ticker']} sent — order_id {oid}")
                    except Exception as e:
                        o2.update({"state":"ERROR","error":str(e)})
                        st.error(f"❌ {o2['ticker']} failed: {e}")
                    updated_staged[i] = o2
                    st.session_state["staged_orders"] = updated_staged
                    st.session_state["live_orders"]   = updated_staged
                    _save_orders(updated_staged)
                    st.session_state.pop("le_exchange_orders", None)
                    st.rerun()
            if btn_b.button("🗑", key=f"le_remove_{i}", use_container_width=True,
                            help="Remove Trade"):
                updated_staged.pop(i)
                st.session_state["staged_orders"] = updated_staged
                st.rerun()
        else:
            if err_tip:
                c9.caption(err_tip[:40])

    st.divider()

    # ── Bulk send / clear controls ─────────────────────────────────────────
    st.markdown("##### Send to Exchange")
    s1, s2 = st.columns([1, 1])

    # Send ALL pending
    if s1.button("⚡ Send All Pending", type="primary",
                 use_container_width=True, disabled=not pending_indices,
                 key="le_send_all"):
        if not bulk_acct and not mock_mode:
            st.error("Select an account before sending live orders.")
        else:
            for idx, i in enumerate(pending_indices):
                o  = dict(updated_staged[i])
                tk = token_map.get(o.get("account","") or bulk_acct, first_tok)
                try:
                    resp = _do_place(tk, o, mock_mode)
                    oid  = resp.get("data",{}).get("order_id","?")
                    o.update({"state":"PLACED","order_id":oid,"placed_at":_now()})
                except Exception as e:
                    o.update({"state":"ERROR","error":str(e)})
                updated_staged[i] = o
                # Rate limit: 2 orders/sec → 0.5s gap between each
                if idx < len(pending_indices) - 1:
                    time.sleep(0.5)
            st.session_state["staged_orders"] = updated_staged
            st.session_state["live_orders"]   = updated_staged
            _save_orders(updated_staged)
            st.session_state.pop("le_exchange_orders", None)
            placed = sum(1 for o in updated_staged if o.get("state") == "PLACED")
            st.success(f"✅ Sent {placed} orders to exchange.")
            st.rerun()

    if s2.button("🗑 Clear Staged", use_container_width=True, key="le_clear_staged"):
        st.session_state["staged_orders"]   = []
        st.session_state["le_staged_built"] = False
        st.rerun()