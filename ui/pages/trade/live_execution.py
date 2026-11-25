"""
live_execution.py — Live Order Execution
─────────────────────────────────────────
4 sections:
  1. Exchange Orders  — open/executed/cancelled from Kite today, with 3 tabs + cancel controls
  2. Order Builder    — MARKET or LIMIT, price tiers per ticker, fee summary
  3. Order Staging    — editable list, send 1-by-1 or send all to exchange
  +  Positions        — live Kite positions (net/day) with P&L
"""
from __future__ import annotations

import json, os, random, time
from datetime import datetime, timezone, timedelta

import pandas as pd
import streamlit as st

# ── Force IPv4 for outbound connections in this process ───────────────────
import socket
_original_getaddrinfo = socket.getaddrinfo

def _ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

socket.getaddrinfo = _ipv4_only_getaddrinfo

# ── import trade_utils — all Kite API calls go through here ──────────────────
from ui.pages.trade.trade_utils import (
    fetch_trades, fetch_orders, fetch_positions, stagger_qtys, place_order, cancel_order, fetch_gtts, place_gtt_market_sell,
    get_fluctuation_pct, fetch_price_yf, calc_fees,
    save_live_orders, load_live_orders, kite_headers,
    geo_buy_orders, geo_sell_orders,
    load_gtt_exclusions, save_gtt_exclusions,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants / paths
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Helpers (used when trade_utils not available)
# ─────────────────────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def _save_orders(orders: list):
    save_live_orders(orders)

def _load_orders() -> list:
    return load_live_orders()

def _kite_hdr(token: str) -> dict:
    return kite_headers(token)

def _ltp(symbol: str, exchange: str = "NSE") -> float:
    return fetch_price_yf(symbol, exchange)

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

def _round_to_tick(price: float, tick: float = 0.10, mode: str = "nearest") -> float:
    """Round price to a valid exchange tick (Decimal-safe, no float drift).
    Default tick is 0.10: a multiple of 0.10 is also valid for 0.05-tick
    instruments, so it is safe for any stock Kite lists on NSE/BSE."""
    from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN, ROUND_UP
    rounding = {"nearest": ROUND_HALF_UP, "down": ROUND_DOWN, "up": ROUND_UP}[mode]
    t = Decimal(str(tick))
    steps = (Decimal(str(price)) / t).to_integral_value(rounding=rounding)
    return float(steps * t)

def _tick_for_price(price: float) -> float:
    """Tick size by price band: > 10000 -> 1.00, otherwise 0.10 (valid for 0.05-tick stocks too)."""
    return 1.00 if float(price) > 10000 else 0.10

_TIER_EXTRA_FRAC = {1: 0.0, 2: 0.25, 3: 0.50, 4: 0.75}   # × Day Fluc%, same as geo_* tiers

def _stagger_price(side: str, ltp: float, fluc: float, tier: int) -> float:
    """Tier price from a given LTP (same formula as trade_utils.geo_buy/sell_orders)."""
    extra = fluc * _TIER_EXTRA_FRAC.get(int(tier), 0.0)
    if str(side).upper() == "BUY":
        return round(ltp * (1 - 0.0025 - extra / 100), 2)
    return round(ltp * (1 + 0.0025 + extra / 100), 2)

def _build_probe_and_brackets(sym: str, exch: str, side: str, ltp: float,
                              qty: int, fluc: float, with_probe: bool) -> list[dict]:
    """STAGGER staging: a 1-qty PROBE order (tier 0) first, then 4 brackets (tiers 1-4)
    sized by stagger_qtys() on the quantity that is left. Bracket prices staged here are
    placeholders — they are re-priced from the Positions LTP when sent."""
    base = {"ticker": sym, "exchange": exch, "side": side, "order_type": "LIMIT",
            "fluc_pct": round(fluc, 3), "account": "", "state": "PENDING",
            "order_id": "", "placed_at": "", "filled_at": "", "skip_reason": "",
            "error": "", "stagger_id": f"{sym}|{exch}|{side}"}
    out, rest = [], int(qty)
    if with_probe and rest >= 1:
        p = _stagger_price(side, ltp, fluc, 1)
        out.append({**base, "qty": 1, "price": p, "value": round(p, 2), "tier": 0, "probe": True})
        rest -= 1
    if rest > 0:
        for t, q in enumerate(stagger_qtys(rest), start=1):
            if q <= 0:
                continue
            p = _stagger_price(side, ltp, fluc, t)
            out.append({**base, "qty": q, "price": p, "value": round(q * p, 2),
                        "tier": t, "bracket": True})
    return out

def _probe_sent(orders: list, o: dict) -> bool:
    """A bracket may go out only after its 1-qty probe has been sent (if it has one)."""
    gid = o.get("stagger_id")
    probes = [x for x in orders if x.get("probe") and x.get("stagger_id") == gid]
    return (not probes) or any(x.get("state") in ("PLACED", "FILLED") for x in probes)

def _positions_ltp_map(token: str) -> dict:
    """{SYMBOL: last_price} from Kite Positions (net overrides day)."""
    pos = _get_positions(token)
    out = {}
    for row in (pos.get("day") or []) + (pos.get("net") or []):
        sym = str(row.get("tradingsymbol", "")).upper()
        lp = float(row.get("last_price") or 0)
        if sym and lp > 0:
            out[sym] = lp
    return out

def _fallback_ltp_map(token: str = "") -> dict[str, float]:
    """Build {SYMBOL: ltp} from Positions first, then cached builder prices / holdings."""
    out: dict[str, float] = {}
    for hlds in (st.session_state.get("all_holdings") or {}).values():
        for h in hlds:
            sym = str(h.get("tradingsymbol") or h.get("ticker") or "").upper()
            lp = float(h.get("last_price") or 0)
            if sym and lp > 0:
                out[sym] = lp
    for r in (st.session_state.get("le_prices") or []):
        sym = str(r.get("Ticker") or "").upper()
        lp = float(r.get("LTP ₹") or 0)
        if sym and lp > 0:
            out[sym] = lp
    if token:
        try:
            out.update(_positions_ltp_map(token))
        except Exception:
            pass
    return out

def _resolve_order_ltp(o: dict, ltp_map: dict | None = None):
    """Return float LTP from Kite order if available and > 0, else from ltp_map (Positions/fallback)."""
    sym = str(o.get("ticker") or o.get("tradingsymbol") or "").upper()
    raw_ltp = o.get("last_price")
    if raw_ltp not in (None, "", 0, 0.0):
        try:
            val = float(raw_ltp)
            if val > 0:
                return val
        except (TypeError, ValueError):
            pass
    mapped = (ltp_map or {}).get(sym)
    if mapped not in (None, "", 0, 0.0):
        try:
            val = float(mapped)
            if val > 0:
                return val
        except (TypeError, ValueError):
            pass
    return ""

def _reprice_staged(orders: list, idxs: list, get_tok, persist: bool = True):
    """Re-price STAGGER tiers 2-4 (PENDING, LIMIT) from the live LTP in Positions.
    Tier 1 is never touched. Mutates orders[i]; returns (changes, missing)."""
    changes, missing, cache = [], [], {}
    for i in idxs:
        o = orders[i]
        tier = int(o.get("tier", 1))
        is_bracket = bool(o.get("bracket"))
        if o.get("probe") or (not is_bracket and tier < 2) \
                or o.get("state") != "PENDING" or o.get("order_type", "LIMIT") != "LIMIT":
            continue
        tk = get_tok(o)
        if tk not in cache:
            try:
                cache[tk] = _positions_ltp_map(tk)
            except Exception as e:
                cache[tk] = e
        m = cache[tk]
        sym = str(o.get("ticker", "")).upper()
        if isinstance(m, Exception):
            missing.append((sym, tier, f"Positions fetch failed: {m}"))
            o["_unpriced"] = True
            continue
        ltp = m.get(sym)
        if not ltp:
            missing.append((sym, tier, "not in Positions yet"))
            o["_unpriced"] = True
            continue
        o.pop("_unpriced", None)
        fluc = float(o.get("fluc_pct") or 1.0)
        new_p = _stagger_price(o.get("side", ""), ltp, fluc, tier)
        old_p = float(o.get("price", 0))
        if abs(new_p - old_p) >= 0.005:
            changes.append({"ticker": sym, "tier": tier, "old": old_p, "new": new_p, "ltp": ltp})
            o["price"] = new_p
            o["value"] = round(int(o.get("qty", 0)) * new_p, 2)
            if persist:
                st.session_state.pop(f"le_price_{i}", None)   # let the editor pick up the new price
    return changes, missing

def _reprice_summary(changes: list, missing: list) -> str:
    parts = []
    if changes:
        parts.append("Repriced from Positions LTP: " + "; ".join(
            f"{c['ticker']} T{c['tier']} ₹{c['old']:.2f}→₹{c['new']:.2f} (LTP ₹{c['ltp']:.2f})" for c in changes))
    if missing:
        parts.append("Price left as staged: " + "; ".join(f"{t} T{n} ({why})" for t, n, why in missing))
    return "  ·  ".join(parts)

def _dry_run_record(o: dict, note: str = "") -> None:
    """Dry run: log exactly what would be sent, without calling the exchange."""
    price = float(o["price"])
    if o.get("order_type", "LIMIT") == "LIMIT":
        price = _round_to_tick(price, _tick_for_price(price))
    st.session_state.setdefault("le_dry_log", []).append({
        "Time": _now(), "Account": o.get("account", ""), "Ticker": o.get("ticker", ""),
        "Exchange": o.get("exchange", ""), "Side": o.get("side", ""),
        "Type": o.get("order_type", "LIMIT"), "Tier": o.get("tier", 1),
        "Qty": int(o.get("qty", 0)), "Price ₹": price,
        "Value ₹": round(int(o.get("qty", 0)) * price, 2), "Note": note})

def _do_place(token: str, o: dict) -> dict:
    """Route all order placement through trade_utils → proxy."""
    price = float(o["price"])
    if o.get("order_type", "LIMIT") == "LIMIT":
        price = _round_to_tick(price, _tick_for_price(price))
    print(f"place_order {o} (rounded price={price})")
    return place_order(
        token, o["ticker"], o["exchange"], o["side"],
        int(o["qty"]), price,
        order_type=o.get("order_type", "LIMIT"),
        mock=False,
    )

def _is_insufficient_funds(err) -> bool:
    """True for Kite's 'Insufficient funds. Margin required: … Margin available: …' rejections."""
    m = str(err).lower()
    return "insufficient funds" in m or ("margin required" in m and "margin available" in m)


def _do_cancel(token: str, order_id: str) -> dict:
    """Route all cancellations through trade_utils → proxy."""
    return cancel_order(token, order_id, mock=False)

def _normalise_kite_order(o: dict, ltp_map: dict | None = None) -> dict:
    """Map Kite API field names to internal names for display, filling LTP from ltp_map when absent."""
    ltp = _resolve_order_ltp(o, ltp_map)
    return {
        "Time":       o.get("order_timestamp","") or o.get("fill_timestamp","") or o.get("placed_at",""),
        "Ticker":     o.get("ticker") or o.get("tradingsymbol",""),
        "Type":       o.get("side")   or o.get("transaction_type",""),
        "Product":    o.get("product","CNC"),
        "Qty":        f"{o.get('filled_quantity',0)} / {o.get('qty') or o.get('quantity',0)}",
        "LTP":        ltp,
        "Price":      o.get("price") or o.get("average_price",0),
        "Status":     o.get("status","") or o.get("state",""),
        "Order ID":   o.get("order_id",""),
        "Tier":       o.get("tier",""),
    }

# ─────────────────────────────────────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Positions (Kite Connect: GET /portfolio/positions → {"net": [...], "day": [...]})
# ─────────────────────────────────────────────────────────────────────────────
def _get_positions(token: str) -> dict:
    """Return {"net": [...], "day": [...]} via trade_utils.fetch_positions → proxy."""
    return fetch_positions(token)


CLOSED_ASSUMED_CHG_PCT = -2.0   # closed positions are assumed stopped out at −2%

def _positions_df(rows: list) -> pd.DataFrame:
    out = []
    for p in rows:
        qty = int(p.get("quantity", 0) or 0)
        avg = float(p.get("average_price", 0) or 0)
        ltp = float(p.get("last_price", 0) or 0)
        pnl = float(p.get("pnl", 0) or 0)
        basis = abs(qty) * avg
        out.append({
            "Product":    p.get("product", ""),
            "Instrument": p.get("tradingsymbol", ""),
            "Exchange":   p.get("exchange", ""),
            "Side":       "LONG" if qty > 0 else "SHORT" if qty < 0 else "CLOSED",
            "Qty":        qty,
            "Avg ₹":      avg,
            "LTP ₹":      ltp,
            "P&L ₹":      pnl,
            "Chg %":      (pnl / basis * 100) if qty and basis else CLOSED_ASSUMED_CHG_PCT,
            "Day Buy":    int(p.get("day_buy_quantity", 0) or 0),
            "Day Sell":   int(p.get("day_sell_quantity", 0) or 0),
        })
    return pd.DataFrame(out)


def _render_positions(token_map: dict, selected_acct: str = "") -> None:
    st.markdown("### 📊 Positions")
    if not token_map:
        st.warning("Connect a Kite account to view positions.")
        return

    acct = selected_acct if selected_acct in token_map else next(iter(token_map))
    h1, h2, h3 = st.columns([3, 2, 1])
    h1.caption(f"Account: **{acct}**")
    view = h2.radio("View", ["Net", "Day"], horizontal=True, key="le_pos_view")
    refresh = h3.button("🔄 Refresh", key="le_pos_refresh", use_container_width=True)

    t1, t2, t3 = st.columns(3)
    show_long   = t1.toggle("🟢 LONG",   value=True,  key="le_pos_long")
    show_short  = t2.toggle("🔴 SHORT",  value=True,  key="le_pos_short")
    show_closed = t3.toggle("⚪ CLOSED", value=False, key="le_pos_closed")

    cache_key = f"le_positions_{acct}"
    if refresh or cache_key not in st.session_state:
        try:
            st.session_state[cache_key] = _get_positions(token_map[acct])
        except Exception as e:
            st.error(f"Unable to fetch positions: {e}")
            return

    df = _positions_df(st.session_state[cache_key].get(view.lower(), []))
    if df.empty:
        st.caption(f"No {view.lower()} positions.")
        return

    # Everything below (metrics + table) follows the LONG / SHORT / CLOSED toggles
    side_ok = ((df["Side"] == "LONG") & show_long) | \
              ((df["Side"] == "SHORT") & show_short) | \
              ((df["Side"] == "CLOSED") & show_closed)
    shown = df[side_ok]
    if shown.empty:
        st.caption("No positions match the selected LONG / SHORT / CLOSED toggles.")
        return

    open_df = shown[shown["Qty"] != 0]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total P&L", f"₹{shown['P&L ₹'].sum():,.2f}")
    m2.metric("Avg Chg %", f"{shown['Chg %'].mean():+.2f}%")
    m3.metric("Open positions", len(open_df))
    m4.metric("Long / Short", f"{(shown['Side'] == 'LONG').sum()} / {(shown['Side'] == 'SHORT').sum()}")
    m5.metric("Open exposure", f"₹{(open_df['Qty'].abs() * open_df['LTP ₹']).sum():,.0f}")

    def _avg(side):
        sub = shown[shown["Side"] == side]["Chg %"]
        return f"{sub.mean():+.2f}% ({len(sub)})" if len(sub) else "–"
    st.caption(f"Metrics cover the selected rows only · LONG {_avg('LONG')} · "
               f"SHORT {_avg('SHORT')} · CLOSED {_avg('CLOSED')} "
               f"(closed taken as {CLOSED_ASSUMED_CHG_PCT:+.0f}%)")

    def _pnl_clr(col):
        return ["color:#15803d" if v > 0 else "color:#b91c1c" if v < 0 else "" for v in col]

    st.dataframe(
        shown.style.apply(_pnl_clr, subset=["P&L ₹", "Chg %"])
        .format({"Avg ₹": "{:,.2f}", "LTP ₹": "{:,.2f}",
                 "P&L ₹": "{:+,.2f}", "Chg %": "{:+.2f}%"}),
        use_container_width=True, hide_index=True)
    st.caption("Source: Kite positions API (same data as kite.zerodha.com/positions). "
               "Chg % = P&L ÷ (|qty| × avg price); closed positions are set to −2%. Click 🔄 to re-fetch.")


def _classify_order_bucket(status_str: str) -> str:
    """Bucket Kite order status into 'open', 'executed', or 'cancelled'."""
    s = str(status_str or "").strip().upper()
    if s in {"OPEN", "TRIGGER PENDING", "PENDING", "AMO REQ RECEIVED"}:
        return "open"
    if s in {"CANCELLED", "REJECTED", "EXPIRED"}:
        return "cancelled"
    if s in {"COMPLETE", "EXECUTED", "FILLED"}:
        return "executed"
    return "executed"


def _render_unified_orders(ex_orders: list, token: str, selected_acct: str = "") -> None:
    """Merged Orders view with 3 status tabs (Open, Executed, Cancelled/Rejected),
    a red Cancel (<n>) Open Orders button, specific-order cancellation controls,
    and LTP populated from Kite order or Positions/fallback map."""
    st.markdown("### 📋 Orders")
    if not token:
        st.warning("Connect a Kite account to view/cancel orders.")
        return

    ltp_map = _fallback_ltp_map(token)
    raw_list = list(ex_orders or [])

    CANCELLABLE = {"OPEN", "TRIGGER PENDING", "AMO REQ RECEIVED"}
    open_raw = [o for o in raw_list if str(o.get("status") or o.get("state") or "").upper() in CANCELLABLE]
    exec_raw = [o for o in raw_list if _classify_order_bucket(o.get("status") or o.get("state") or "") == "executed"]
    canc_raw = [o for o in raw_list if _classify_order_bucket(o.get("status") or o.get("state") or "") == "cancelled"]

    h1, h2 = st.columns([4, 2])
    acct_label = f" · Account: **{selected_acct}**" if selected_acct else ""
    h1.caption(f"Open orders on exchange: **{len(open_raw)}**{acct_label} · LTP falls back to Positions when omitted by Kite.")
    with h2:
        st.markdown(
            """<style>
            div[data-testid="stButton"] button[kind="primary"] {
                background-color: #dc2626 !important;
                border-color: #dc2626 !important;
                color: #ffffff !important;
            }
            div[data-testid="stButton"] button[kind="primary"]:hover {
                background-color: #b91c1c !important;
                border-color: #b91c1c !important;
                color: #ffffff !important;
            }
            </style>""",
            unsafe_allow_html=True,
        )
        if st.button(f"🚫 Cancel ({len(open_raw)}) Open Orders", disabled=not bool(open_raw),
                     key="le_orders_cancel_all", type="primary", use_container_width=True):
            cancelled, failed = 0, 0
            for o in open_raw:
                oid = str(o.get("order_id") or "")
                if not oid:
                    continue
                try:
                    _do_cancel(token, oid)
                    cancelled += 1
                except Exception as e:
                    st.warning(f"❌ {o.get('tradingsymbol', '')} ({oid}): {e}")
                    failed += 1
            st.session_state.pop("le_exchange_orders", None)
            st.success(f"✅ Cancelled {cancelled} order(s)." + (f" {failed} failed." if failed else ""))
            st.rerun()

    def _clr(row):
        c = "background-color:#dcfce7;color:#15803d" if str(row.get("Type", "")).upper() == "BUY" \
            else "background-color:#fee2e2;color:#b91c1c" if str(row.get("Type", "")).upper() == "SELL" \
            else ""
        return [c] * len(row)

    col_cfg = {
        "Price": st.column_config.NumberColumn(format="₹%.2f"),
        "LTP":   st.column_config.NumberColumn(format="₹%.2f"),
    }

    tab_open, tab_exec, tab_canc = st.tabs([
        f"🟡 Open ({len(open_raw)})",
        f"✅ Executed ({len(exec_raw)})",
        f"🚫 Cancelled/Rejected ({len(canc_raw)})",
    ])

    with tab_open:
        if not open_raw:
            st.caption("No open orders.")
        else:
            open_df = pd.DataFrame([_normalise_kite_order(o, ltp_map) for o in open_raw])
            st.dataframe(open_df.style.apply(_clr, axis=1),
                         use_container_width=True, hide_index=True,
                         column_config=col_cfg)

            st.markdown("##### 🎯 Cancel Specific Open Order")
            for idx, o in enumerate(open_raw):
                oid = str(o.get("order_id") or "")
                sym = str(o.get("tradingsymbol") or o.get("ticker") or "").upper()
                side = str(o.get("transaction_type") or o.get("side") or "").upper()
                qty = o.get("quantity") or o.get("qty") or 0
                filled_qty = o.get("filled_quantity") or 0
                price = float(o.get("price") or o.get("average_price") or 0.0)
                ltp = _resolve_order_ltp(o, ltp_map)
                ltp_text = f"₹{float(ltp):,.2f}" if ltp not in (None, "") else "—"
                status = str(o.get("status") or "")
                exch = str(o.get("exchange") or "NSE")
                with st.container(border=True):
                    r1, r2, r3, r4 = st.columns([3, 2, 2.5, 1.5])
                    r1.markdown(f"**{sym}** · `{exch}` · **{side}** · `{status}`")
                    r2.markdown(f"Qty: **{filled_qty} / {qty}**")
                    r3.markdown(f"LTP: **{ltp_text}** · Price: **₹{price:,.2f}** · ID: `{oid}`")
                    if r4.button("🚫 Cancel", key=f"le_order_cancel_row_{idx}_{oid}",
                                 use_container_width=True):
                        if oid:
                            try:
                                _do_cancel(token, oid)
                                st.session_state.pop("le_exchange_orders", None)
                                st.success(f"✅ Cancelled {sym} ({oid})")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Failed to cancel {sym} ({oid}): {e}")

    with tab_exec:
        if not exec_raw:
            st.caption("No executed orders.")
        else:
            exec_df = pd.DataFrame([_normalise_kite_order(o, ltp_map) for o in exec_raw])
            st.dataframe(exec_df.style.apply(_clr, axis=1),
                         use_container_width=True, hide_index=True,
                         column_config=col_cfg)

    with tab_canc:
        if not canc_raw:
            st.caption("No cancelled or rejected orders.")
        else:
            canc_df = pd.DataFrame([_normalise_kite_order(o, ltp_map) for o in canc_raw])
            st.dataframe(canc_df.style.apply(_clr, axis=1),
                         use_container_width=True, hide_index=True,
                         column_config=col_cfg)


def render() -> None:

    # One-time cleanup of stale Order Builder cache/state from older app runs.
    # Preserve active plans, staged orders, and live orders.
    if not st.session_state.get("le_cache_cleanup_v1_done", False):
        for key in list(st.session_state.keys()):
            if key.startswith("le_builder_editor_"):
                st.session_state.pop(key, None)
        for key in ("le_prices", "le_exchange_orders", "le_check_state",
                    "le_builder_removed_keys"):
            st.session_state.pop(key, None)
        st.session_state["le_cache_cleanup_v1_done"] = True

    for k, v in [("live_orders",[]), ("staged_orders",[]),
                 ("le_staged_built", False)]:
        if k not in st.session_state: st.session_state[k] = v

    accounts   = st.session_state.get("accounts") or []
    token_map  = {a["name"]: a["token"] for a in accounts if a.get("token")}
    acct_opts  = list(token_map.keys())
    first_tok  = next(iter(token_map.values()), "")

    st.markdown("## ⚡ Live Execution")

    # ── Top-level account / profile selector ──────────────────────────────────
    if acct_opts:
        _p_col, _p_info = st.columns([2, 3])
        selected_acct = _p_col.selectbox(
            "🏦 Active Account / Profile",
            acct_opts,
            key="le_active_acct",
        )
        _p_info.caption("Shared across Positions, Orders, Order Builder sellable holdings, and Order Staging.")
    else:
        selected_acct = ""

    active_tok = token_map.get(selected_acct, first_tok)
    bulk_acct = selected_acct

    # ── Top-level safety controls ───────────────────────────────────────────
    # The Dry-run toggle and Max order amount are rendered next to "Send to Exchange" (Part 3);
    # their values are read here from session state so every section above sees the current setting.
    dry_run = bool(st.session_state.get("le_dry_run", False))
    max_order_amt = int(st.session_state.get("le_max_order_amt", 0) or 0)

    _halt = st.session_state.get("le_funds_halt")
    if _halt:
        st.error(f"🛑 **Order sending halted — insufficient funds.** "
                 f"{_halt.get('side','')} {_halt.get('ticker','')} at {_halt.get('at','')}: "
                 f"{_halt.get('msg','')}")
        if st.button("▶️ Resume sending (after adding funds / reviewing orders)",
                     key="le_resume_sending"):
            st.session_state.pop("le_funds_halt", None)
            st.rerun()

    def _exceeds_limit(o: dict) -> bool:
        return bool(max_order_amt) and int(o.get("qty", 0)) * float(o.get("price", 0)) > max_order_amt

    def _halt_on_funds(o: dict, err) -> None:
        st.session_state["le_funds_halt"] = {
            "ticker": o.get("ticker", ""), "side": o.get("side", ""),
            "msg": str(err), "at": _now()}

    # ═══════════════════════════════════════════════════════════════════════
    # PART 1 — EXCHANGE ORDERS & PROTECTIVE GTTs
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("### 1️⃣ Exchange Orders")

    p1h, p1r = st.columns([5, 1])
    p1h.markdown("Open, executed, and cancelled/rejected orders for today from the exchange.")
    refresh = p1r.button("🔄 Refresh", key="le_p1_refresh", use_container_width=True)
    if refresh:
        st.session_state.pop("le_exchange_orders", None)
        st.rerun()

    # Fetch from exchange (or mock)
    if "le_exchange_orders" not in st.session_state or refresh:
        if active_tok:
            try:
                raw = fetch_orders(active_tok, mock=False,
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
        _render_unified_orders(ex_orders, active_tok, selected_acct)

    # ── Protective GTTs for COMPLETE BUY orders ──────────────────────────
    st.markdown("### 🛡 Protective GTT — −2% MARKET SELL")
    all_complete_buys = [o for o in ex_orders
                         if str(o.get("status", "")).upper() == "COMPLETE"
                         and str(o.get("transaction_type", "")).upper() == "BUY"
                         and int(o.get("filled_quantity", 0) or 0) > 0
                         and float(o.get("average_price", 0) or 0) > 0]

    # ── Excluded stocks (saved to disk → survives sessions) ──────────────
    if "gtt_excl_ms" not in st.session_state:
        st.session_state["gtt_excl_ms"] = sorted(load_gtt_exclusions())

    def _gtt_excl_save():
        save_gtt_exclusions(set(st.session_state.get("gtt_excl_ms", [])))

    def _gtt_excl_add():
        new = {s.strip().upper() for s in str(st.session_state.get("gtt_excl_new", "")).replace("\n", ",").split(",")
               if s.strip()}
        if new:
            st.session_state["gtt_excl_ms"] = sorted(set(st.session_state.get("gtt_excl_ms", [])) | new)
            _gtt_excl_save()
        st.session_state["gtt_excl_new"] = ""

    _excl_now = set(st.session_state["gtt_excl_ms"])
    _excl_opts = sorted(_excl_now
                        | {str(o.get("tradingsymbol", "")).upper() for o in all_complete_buys}
                        | {str(h.get("tradingsymbol", "")).upper()
                           for hlds in (st.session_state.get("all_holdings") or {}).values() for h in hlds})
    _excl_opts = [x for x in _excl_opts if x]
    with st.expander(f"🚫 Exclude stocks from protective GTT ({len(_excl_now)})", expanded=False):
        st.multiselect("Excluded symbols", _excl_opts, key="gtt_excl_ms", on_change=_gtt_excl_save,
                       help="No protective GTT is suggested or placed for these stocks. "
                            "Saved to disk, so the list is kept across sessions.")
        ea, eb = st.columns([4, 1])
        ea.text_input("Add symbol(s) not in the list (comma-separated)", key="gtt_excl_new",
                      placeholder="e.g. LIQUIDBEES, GOLDBEES")
        eb.button("➕ Add", key="gtt_excl_add", on_click=_gtt_excl_add, use_container_width=True)
        st.caption("Saved in outputs/gtt_excluded_symbols.json. Existing GTTs in Kite are not touched.")

    _excl = set(st.session_state["gtt_excl_ms"])
    complete_buys = [o for o in all_complete_buys
                     if str(o.get("tradingsymbol", "")).upper() not in _excl]
    _n_excl_hidden = len(all_complete_buys) - len(complete_buys)
    if not active_tok:
        st.warning("Connect a Kite account to view/place protective GTTs.")
    elif not complete_buys:
        st.caption("No COMPLETE BUY orders to protect."
                   + (f" ({_n_excl_hidden} excluded by your GTT exclusion list.)" if _n_excl_hidden else ""))
    else:
        try:
            gtt_existing = fetch_gtts(active_tok)
            gtt_fetch_ok = True
        except Exception as gtt_err:
            gtt_existing = []
            gtt_fetch_ok = False
            st.error(f"Unable to fetch existing GTTs; placement disabled to prevent duplicates: {gtt_err}")

        GTT_TICK_OPTIONS = [0.05, 0.10, 0.50, 1.00]   # default = 0.05

        def _ts(v):
            t = pd.to_datetime(v, errors="coerce")
            if pd.isna(t):
                return None
            return t.tz_localize(None) if getattr(t, "tzinfo", None) else t

        def _gtt_matches(buy_order, exact_trigger, status="active"):
            """Find ALL SELL GTTs with the given status (active / triggered) for the same
            symbol/qty near the -2% level. Compared against the unrounded level (±0.51)
            so it is independent of the tick size picked when that GTT was placed.
            A GTT created before this BUY order cannot be protecting it, so it is skipped."""
            sym = str(buy_order.get("tradingsymbol", "")).upper()
            qty = int(buy_order.get("filled_quantity", 0) or 0)
            buy_ts = _ts(buy_order.get("order_timestamp") or buy_order.get("exchange_timestamp"))
            found = []
            for g in gtt_existing:
                if str(g.get("status", "")).lower() != status:
                    continue
                g_ts = _ts(g.get("created_at"))
                if buy_ts is not None and g_ts is not None and g_ts < buy_ts:
                    continue
                cond = g.get("condition", {}) or {}
                legs = g.get("orders", []) or []
                if str(cond.get("tradingsymbol", "")).upper() != sym:
                    continue
                if str(cond.get("exchange", "")).upper() != str(buy_order.get("exchange", "NSE")).upper():
                    continue
                vals = cond.get("trigger_values", []) or []
                if not vals or abs(float(vals[0]) - exact_trigger) > 0.51:
                    continue
                if any(str(leg.get("transaction_type", "")).upper() == "SELL"
                       and int(leg.get("quantity", 0) or 0) == qty for leg in legs):
                    found.append(g.get("id"))
            return found

        def _gtt_match(buy_order, exact_trigger, status="active"):
            ids = _gtt_matches(buy_order, exact_trigger, status)
            return ids[0] if ids else None

        # Active SELL GTTs configured more than once (same symbol, qty and trigger)
        _grp = {}
        for g in (gtt_existing if gtt_fetch_ok else []):
            if str(g.get("status", "")).lower() != "active":
                continue
            cond = g.get("condition", {}) or {}
            vals = cond.get("trigger_values", []) or []
            for leg in (g.get("orders", []) or []):
                if str(leg.get("transaction_type", "")).upper() == "SELL" and vals:
                    key = (str(cond.get("exchange", "")).upper(),
                           str(cond.get("tradingsymbol", "")).upper(),
                           int(leg.get("quantity", 0) or 0), round(float(vals[0]), 2))
                    _grp.setdefault(key, []).append(g.get("id"))
        dup_groups = {k: v for k, v in _grp.items() if len(v) > 1}

        # Pre-compute rows so we can summarise and filter before rendering
        gtt_rows = []
        for buy in complete_buys:
            order_id = str(buy.get("order_id", ""))
            avg = float(buy.get("average_price", 0) or 0)
            tick = float(st.session_state.get(f"gtt_tick_{order_id}", 0.05))
            gtt_rows.append({
                "buy": buy, "order_id": order_id, "avg": avg, "tick": tick,
                "trigger": _round_to_tick(avg * 0.98, tick),
                "existing_ids": _gtt_matches(buy, avg * 0.98) if gtt_fetch_ok else [],
                "existing_id": _gtt_match(buy, avg * 0.98) if gtt_fetch_ok else None,
                "triggered_id": _gtt_match(buy, avg * 0.98, "triggered") if gtt_fetch_ok else None,
            })

        # A triggered GTT has already fired (position exited): hidden unless ticked below.
        triggered_rows = [r for r in gtt_rows if r["triggered_id"]]
        n_triggered = len(triggered_rows)
        pending_rows = [r for r in gtt_rows if not r["triggered_id"]]

        n_with = sum(1 for r in pending_rows if r["existing_id"])
        n_without = len(pending_rows) - n_with
        tcol1, tcol2, tcol3 = st.columns([2, 2, 2])
        show_existing = tcol1.toggle("Show orders that already have a GTT",
                                     value=False, key="gtt_show_existing")
        show_triggered = tcol2.checkbox(f"Show triggered GTTs ({n_triggered})",
                                        value=False, key="gtt_show_triggered",
                                        disabled=not n_triggered)
        if gtt_fetch_ok:
            tcol3.caption(f"🔴 {n_without} without GTT · 🟢 {n_with} with GTT")
        if n_triggered and not show_triggered:
            st.caption(f"{n_triggered} order(s) hidden — their protective GTT has already triggered.")
        if not show_existing and n_with:
            st.caption(f"{n_with} order(s) with an existing GTT are hidden.")
        if not show_existing and n_without == 0 and gtt_fetch_ok:
            st.success("All completed BUY orders already have a protective GTT.")

        if dup_groups:
            st.warning(f"⚠️ {len(dup_groups)} protective GTT(s) are configured more than once "
                       "(same symbol, quantity and trigger). Delete the extras in Kite to avoid "
                       "selling the same shares twice.")
            st.dataframe(pd.DataFrame([
                {"Symbol": k[1], "Exchange": k[0], "Qty": k[2], "Trigger ₹": k[3],
                 "Times configured": len(v), "GTT IDs": ", ".join(str(i) for i in v)}
                for k, v in dup_groups.items()]),
                use_container_width=True, hide_index=True)

        gtt_rows = [r for r in gtt_rows if show_triggered or not r["triggered_id"]]

        for r in gtt_rows:
            buy, order_id, avg, tick = r["buy"], r["order_id"], r["avg"], r["tick"]
            trigger, existing_id = r["trigger"], r["existing_id"]
            triggered_id = r["triggered_id"]
            if existing_id and not show_existing and not triggered_id:
                continue
            symbol = str(buy.get("tradingsymbol", "")).upper()
            exch = str(buy.get("exchange", "NSE")).upper()
            qty = int(buy.get("filled_quantity", 0) or 0)
            with st.container(border=True):
                c1, c2, c3, c4, c5, c6 = st.columns([2, 1, 1, 1, 1, 1])
                c1.markdown(f"**{symbol}** · BUY order `{order_id}`")
                c2.metric("Filled Qty", qty)
                c3.metric("Avg Buy", f"₹{avg:,.2f}")
                c4.metric("Stop Trigger", f"₹{trigger:,.2f}")
                stop_pct = (trigger - avg) / avg * 100 if avg else 0.0
                stop_amt = (trigger - avg) * qty
                c5.metric("Stop P/L %", f"{stop_pct:+.2f}%", f"₹{stop_amt:+,.2f}")
                if triggered_id:
                    st.info(f"🔔 Protective GTT already triggered (ID: {triggered_id}) — nothing to place.")
                elif existing_id:
                    st.success(f"Protective GTT already exists (ID: {existing_id}) — no duplicate placement.")
                    if len(r["existing_ids"]) > 1:
                        st.warning(f"⚠️ Configured {len(r['existing_ids'])} times "
                                   f"(IDs: {', '.join(str(i) for i in r['existing_ids'])}) — "
                                   "remove the extra GTT(s).")
                elif not gtt_fetch_ok:
                    st.button("Place MARKET SELL GTT", disabled=True, key=f"gtt_disabled_{order_id}")
                else:
                    c6.selectbox("Tick size", GTT_TICK_OPTIONS, index=0,
                                 format_func=lambda v: f"{v:.2f}",
                                 key=f"gtt_tick_{order_id}",
                                 help="Smallest price step for this stock. Trigger is rounded to a multiple of it.")
                    confirm = st.checkbox(
                        f"Confirm protective GTT: {symbol} SELL {qty} at ₹{trigger:.2f} (−2%)",
                        key=f"gtt_confirm_{order_id}")
                    if st.button("🛡 Place MARKET SELL GTT", disabled=not confirm,
                                 key=f"gtt_place_{order_id}", type="primary"):
                        try:
                            result = place_gtt_market_sell(active_tok, symbol, exch, qty, trigger, avg, "CNC")
                            st.success(f"GTT submitted for {symbol}. Trigger ID: {result.get('data', {}).get('trigger_id', 'unknown')}")
                            st.session_state.pop("le_exchange_orders", None)
                            st.rerun()
                        except Exception as gtt_place_err:
                            st.error(f"GTT placement failed for {symbol}: {gtt_place_err}")
        if _n_excl_hidden:
            st.caption(f"🚫 {_n_excl_hidden} COMPLETE BUY order(s) hidden — symbol is in the GTT exclusion list.")
        st.caption("Stop trigger is calculated from the completed BUY average price (−2%); no Kite Quote API call is needed. Active GTTs are checked before placement. API rejection is shown and does not mark the order protected.")

    st.divider()
    _render_positions(token_map, selected_acct)

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════
    # PART 2 — ORDER BUILDER
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("### 2️⃣ Order Builder")

    live_now = st.session_state.get("live_orders", [])

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
                               "**STAGGER** — 1-qty probe order first, then 4 brackets priced from the Positions LTP.\n\n"
                               "**MARKET** — market order at LTP."
                           ))
    if selected_acct:
        g2.caption(f"🏦 Active account:\n**{selected_acct}**")
    else:
        g2.warning("⚠️ Connect an account at the top.")

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
        g3.caption("1-qty probe first, then 4 brackets (qty split + **Day Fluc%** tiers) priced from Positions LTP.")

    # ── Ticker-level reconciliation: actual FILLED + remaining OPEN qty ────
    # Key by ticker + exchange + side so one symbol/order side never offsets another.
    def _order_key(ticker, exchange, side):
        return (str(ticker or "").upper(), str(exchange or "NSE").upper(), str(side or "").upper())

    exch_filled_map: dict[tuple, int] = {}
    exch_open_map: dict[tuple, int] = {}
    exch_order_ids: set[str] = set()
    OPEN_STATUSES = {"OPEN", "TRIGGER PENDING", "PENDING", "AMO REQ RECEIVED"}
    for o in ex_orders:
        status = str(o.get("status") or o.get("Status") or o.get("state") or "").upper()
        side = str(o.get("transaction_type") or o.get("Type") or o.get("side") or "").upper()
        ticker = o.get("tradingsymbol") or o.get("Ticker") or o.get("ticker") or ""
        exchange = o.get("exchange") or o.get("Exchange") or "NSE"
        key = _order_key(ticker, exchange, side)
        filled_qty = int(o.get("filled_quantity") or 0)
        total_qty = int(o.get("quantity") or 0)
        order_id = str(o.get("order_id") or o.get("Order ID") or "")
        if order_id:
            exch_order_ids.add(order_id)
        if ticker and side:
            exch_filled_map[key] = exch_filled_map.get(key, 0) + filled_qty
            if status in OPEN_STATUSES:
                exch_open_map[key] = exch_open_map.get(key, 0) + max(0, total_qty - filled_qty)

    # Local placed/filled records are counted only when their order ID is not
    # present in the exchange response, avoiding duplicate tallying.
    local_filled_map: dict[tuple, int] = {}
    local_open_map: dict[tuple, int] = {}
    for o in live_now:
        state = str(o.get("state", "")).upper()
        oid = str(o.get("order_id") or "")
        if oid and oid in exch_order_ids:
            continue
        key = _order_key(o.get("ticker"), o.get("exchange", "NSE"), o.get("side"))
        qty = int(o.get("qty") or 0)
        if state == "FILLED":
            local_filled_map[key] = local_filled_map.get(key, 0) + qty
        elif state == "PLACED":
            local_open_map[key] = local_open_map.get(key, 0) + qty

    filled_map = {k: exch_filled_map.get(k, 0) + local_filled_map.get(k, 0)
                  for k in set(exch_filled_map) | set(local_filled_map)}
    open_map = {k: exch_open_map.get(k, 0) + local_open_map.get(k, 0)
                for k in set(exch_open_map) | set(local_open_map)}

    # ── Whitelist: ignore FILLED qty for chosen ticker/side (overrides the "all filled" check) ──
    _fill_opts = {}
    for _r in plan:
        _tk = str(_r.get("ticker", "")).upper()
        _sd = str((_r.get("action") or _r.get("side", ""))).upper()
        _ex = str(_r.get("exchange", "NSE")).upper()
        _k = _order_key(_tk, _ex, _sd)
        if filled_map.get(_k, 0) > 0:
            _fill_opts[f"{_tk} · {_sd} · filled {filled_map[_k]}"] = _k
    _fill_saved = st.session_state.get("le_fill_override_labels", [])
    _fill_saved = [l for l in _fill_saved if l in _fill_opts]
    if _fill_opts:
        with st.expander(f"♻️ Whitelist FILLED orders to re-send ({len(_fill_saved)})", expanded=bool(_fill_saved)):
            st.multiselect("Ignore FILLED qty for", list(_fill_opts.keys()), default=_fill_saved,
                           key="le_fill_override_labels",
                           help="Selected ticker/side rows are rebuilt as if nothing was filled: "
                                "Remaining = Ask Qty − Open (filled qty is ignored). "
                                "Open orders are still counted to avoid duplicates.")
            st.caption("⚠️ This can send the full planned qty again. Session only — resets on a new session.")
    fill_override = {_fill_opts[l] for l in st.session_state.get("le_fill_override_labels", []) if l in _fill_opts}

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

    # ── Portfolio sellable qty map  {ticker: sellable_qty} ─────────────────
    portfolio_qty_map: dict[str, int] = {}
    all_holdings = st.session_state.get("all_holdings") or {}
    holdings_src = (all_holdings.get(order_account) or
                    [h for hlds in all_holdings.values() for h in hlds])
    for h in holdings_src:
        sym = (h.get("tradingsymbol") or h.get("ticker") or "").upper()
        if not sym:
            continue
        total_qty      = int(h.get("quantity")             or 0)
        t1_qty         = int(h.get("t1_quantity")          or 0)
        authorised_qty = int(h.get("authorised_quantity")  or 0)
        sellable       = max(0, total_qty - t1_qty - authorised_qty)
        portfolio_qty_map[sym] = portfolio_qty_map.get(sym, 0) + sellable

    # ── Build the Order Builder table ──────────────────────────────────────
    builder_rows: list[dict] = []
    for row in plan:
        tk       = row["ticker"]
        side     = (row.get("action") or row.get("side", "")).upper()
        plan_qty = int(row.get("today_qty", row.get("qty", 0)))

        # ask_qty: always a positive number (exchange always receives positive qty)
        ask_qty = abs(plan_qty)

        exch = str(row.get("exchange", "NSE")).upper()
        tally_key = _order_key(tk, exch, side)
        filled = filled_map.get(tally_key, 0)
        open_qty = open_map.get(tally_key, 0)
        eff_filled = 0 if tally_key in fill_override else filled     # whitelist ignores FILLED qty
        remaining = max(0, ask_qty - eff_filled - open_qty)

        if side == "SELL":
            # Sellable holdings are an execution constraint, not order completion.
            sellable_qty = portfolio_qty_map.get(tk, 0)
            residual = max(0, min(remaining, sellable_qty))
        else:  # BUY
            sellable_qty = ""
            residual = remaining

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
            probe_needed = (filled + open_qty) == 0     # no order today for this ticker/side yet
            rest_qty = residual - 1 if probe_needed else residual
            qtys = stagger_qtys(rest_qty) if rest_qty > 0 else [0, 0, 0, 0]
            if side == "BUY":
                tiers = [round(ltp * (1 - 0.0025 - f / 100), 2)
                         for f in [0, fluc / 4, fluc / 2, fluc * 3 / 4]]
            else:
                tiers = [round(ltp * (1 + 0.0025 + f / 100), 2)
                         for f in [0, fluc / 4, fluc / 2, fluc * 3 / 4]]
            exec_price = tiers[0]
            tier_str   = "  |  ".join(
                f"B{i+1}: {q}@₹{p}" for i, (q, p) in enumerate(zip(qtys, tiers)) if q > 0
            )
            if probe_needed:
                tier_str = f"PROBE: 1@₹{tiers[0]}  |  " + tier_str + "  (brackets re-priced from Positions LTP)"

        exec_val = round(residual * exec_price, 0)

        builder_rows.append({
            "✓":            True,
            "Account":      order_account,
            "Ticker":       tk,
            "Exchange":     exch,
            "Side":         side,
            "Ask Qty":      ask_qty,
            "Plan Qty":     ask_qty,       # kept for display compatibility (always positive)
            "Filled":       filled,
            "Open":          open_qty,
            "Remaining":     remaining,
            "Sellable":     sellable_qty if side == "SELL" else "",
            "Send Qty":     residual,
            "LTP ₹":        ltp,
            "Exec Price ₹": exec_price,
            "Send Value ₹": exec_val,
            "Day Fluc%":    fluc,
            "Order Type":   "MARKET" if order_mode == "MARKET" else "LIMIT",
            "Tiers":        tier_str,
            "_exchange":    exch,
        })

    # ── Ticker reconciliation summary when no executable rows ──────────────
    if not builder_rows:
        recon_rows = []
        all_accounted = True
        for row in plan:
            tk = str(row.get("ticker", "")).upper()
            side = str((row.get("action") or row.get("side", "")).upper())
            exch = str(row.get("exchange", "NSE")).upper()
            plan_qty = abs(int(row.get("today_qty", row.get("qty", 0)) or 0))
            key = _order_key(tk, exch, side)
            filled = filled_map.get(key, 0)
            open_qty = open_map.get(key, 0)
            eff_filled = 0 if key in fill_override else filled
            remaining = max(0, plan_qty - eff_filled - open_qty)
            sellable = portfolio_qty_map.get(tk, 0) if side == "SELL" else "—"
            executable = min(remaining, sellable) if side == "SELL" else remaining
            if remaining > 0:
                all_accounted = False
            recon_rows.append({"Ticker": tk, "Exchange": exch, "Side": side,
                               "Plan Qty": plan_qty, "Filled": filled,
                               "Open": open_qty, "Remaining": remaining,
                               "Sellable": sellable, "Executable": executable})
        if recon_rows:
            st.error(":shield: Remove Existing GTT Orders Before Sell Execution.")
            st.markdown("##### 📊 Plan reconciliation by ticker")
            st.dataframe(pd.DataFrame(recon_rows), use_container_width=True, hide_index=True)
        if all_accounted:
            st.success("🎉 All planned quantities are filled or currently open on the exchange.")
        else:
            st.warning("No executable rows currently. Review ticker reconciliation above—remaining quantity may be blocked by sellable holdings or order state.")
        return

    # ── Render editable Order Builder table ───────────────────────────────
    removed_builder_keys = set(st.session_state.get("le_builder_removed_keys", []))
    builder_rows = [r for r in builder_rows
                    if f"{r['Ticker']}_{r['Side']}" not in removed_builder_keys]
    if not builder_rows:
        st.info("All Order Builder rows were removed for this review pass.")
        return

    builder_df = pd.DataFrame(builder_rows)

    st.markdown("##### 📋 Order Builder — Review & Edit")
    st.caption(
        "Edit **Send Qty**, **Exec Price ₹**, or **Order Type** per row. "
        "Tick ✓ to include a row in the staged batch. "
        "Price tiers are shown for reference in STAGGER mode."
    )

    row_keys = [f"{r['Ticker']}_{r['Side']}" for r in builder_rows]
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

    builder_df["✓"] = [check_state.get(k, True) for k in row_keys]
    builder_df["🗑 Delete"] = False

    edited_builder = st.data_editor(
        builder_df[[
            "✓", "🗑 Delete", "Account", "Ticker", "Exchange", "Side",
            "Ask Qty", "Plan Qty", "Filled", "Open", "Remaining", "Sellable", "Send Qty",
            "LTP ₹", "Exec Price ₹", "Send Value ₹",
            "Day Fluc%", "Order Type", "Tiers",
        ]],
        use_container_width=True, hide_index=True,
        disabled=["Account", "Ticker", "Exchange", "Side",
                  "Ask Qty", "Plan Qty", "Filled", "Open", "Remaining", "Sellable", "LTP ₹", "Send Value ₹",
                  "Day Fluc%", "Tiers"],
        column_config={
            "✓":            st.column_config.CheckboxColumn("Send",     width="small"),
            "🗑 Delete":    st.column_config.CheckboxColumn("Delete row", width="small"),
            "Account":      st.column_config.TextColumn("Account",      width="medium"),
            "Ask Qty":      st.column_config.NumberColumn("Ask Qty",    width="small"),
            "Filled":       st.column_config.NumberColumn("Filled",     width="small",
                                                          help="Actual filled quantity tallied by ticker/exchange/side"),
            "Open":          st.column_config.NumberColumn("Open",        width="small",
                                                           help="Remaining quantity in currently open exchange orders"),
            "Remaining":      st.column_config.NumberColumn("Remaining",   width="small",
                                                            help="Plan qty minus filled and open quantities"),
            "Sellable":     st.column_config.TextColumn("Sellable",     width="small",
                                                        help="Qty available to sell today (excl. T1 & unauthorised)"),
            "LTP ₹":        st.column_config.NumberColumn("LTP ₹",     format="₹%.2f", width="small"),
            "Exec Price ₹": st.column_config.NumberColumn("Exec Price ₹", format="₹%.2f", width="small"),
            "Send Value ₹": st.column_config.NumberColumn("Value ₹",   format="₹%.0f", width="small"),
            "Day Fluc%":    st.column_config.NumberColumn("Fluc%",      format="%.3f%%", width="small"),
            "Send Qty":     st.column_config.NumberColumn("Send Qty",   step=1, width="small"),
            "Order Type":   st.column_config.SelectboxColumn(
                "Type", options=["LIMIT", "MARKET"], width="small"),
            "Tiers":        st.column_config.TextColumn("Tier Preview", width="large"),
        },
        key=f"le_builder_editor_{st.session_state.get('le_builder_editor_version', 0)}",
    )

    if "🗑 Delete" in edited_builder.columns and edited_builder["🗑 Delete"].any():
        delete_mask = edited_builder["🗑 Delete"].fillna(False).astype(bool)
        deleted_keys = [k for k, remove in zip(row_keys, delete_mask.tolist()) if remove]
        removed_keys = set(st.session_state.get("le_builder_removed_keys", []))
        removed_keys.update(deleted_keys)
        st.session_state["le_builder_removed_keys"] = sorted(removed_keys)
        for k in deleted_keys:
            check_state.pop(k, None)
        st.session_state["le_check_state"] = check_state
        st.session_state["le_builder_editor_version"] = st.session_state.get("le_builder_editor_version", 0) + 1
        st.rerun()
    else:
        edited_builder = edited_builder.drop(columns=["🗑 Delete"], errors="ignore")

    for k, checked in zip(row_keys, edited_builder["✓"].tolist()):
        check_state[k] = checked

    edited_builder["Send Value ₹"] = (
            edited_builder["Send Qty"].astype(float) *
            edited_builder["Exec Price ₹"].astype(float)
    ).round(0)

    rc1, _ = st.columns([1, 5])
    if rc1.button("🔄 Re-fetch Prices", key="le_refetch", use_container_width=True):
        st.session_state.pop("le_prices", None)
        st.session_state["le_staged_built"] = False
        st.rerun()

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
                probe_needed = (int(r.get("Filled", 0) or 0) + int(r.get("Open", 0) or 0)) == 0
                batch = _build_probe_and_brackets(sym, exch, side, ltp, qty, fluc, probe_needed)
            else:
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

        staged.sort(key=lambda o: 0 if str(o.get("side", "")).upper() == "SELL" else 1)
        st.session_state["staged_orders"]   = staged
        st.session_state["le_staged_built"] = True
        st.success(f"Built **{len(staged)}** staged orders from **{len(selected)}** rows. Scroll to Part 3 ↓")
        st.rerun()

    st.divider()

    # ═══════════════════════════════════════════════════════════════════════
    # PART 3 — ORDER STAGING  (send 1-by-1 or all)
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("### 3️⃣ Staged Orders")

    _flash = st.session_state.pop("le_send_flash", None)
    if _flash:
        {"error": st.error, "warning": st.warning}.get(_flash[0], st.success)(_flash[1])

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

    st.markdown("##### 📋 Review, Edit & Send")
    st.caption("Edit **Price** or **Qty** directly. Use the action buttons on each row to send or remove.")

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
    div[data-testid="stButton"] button {
        height: 38px !important;
        min-height: 38px !important;
        padding: 0 8px !important;
    }
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
        c4.markdown(f'<div style="{_CELL}">{"Probe" if o.get("probe") else o.get("tier","")}</div>', unsafe_allow_html=True)

        if is_pending:
            new_qty = c5.number_input(
                "qty", value=int(o.get("qty",1)), min_value=1, step=1,
                key=f"le_qty_{i}", label_visibility="collapsed"
            )
            new_price = c6.number_input(
                "price", value=float(o.get("price",0.0)), min_value=0.0, step=0.10, format="%.2f",
                key=f"le_price_{i}", label_visibility="collapsed"
            )
            if new_qty != int(o.get("qty",1)) or new_price != float(o.get("price",0.0)):
                updated_staged[i]["qty"]   = new_qty
                updated_staged[i]["price"] = new_price
                updated_staged[i]["value"] = round(new_qty * new_price, 2)
                st.session_state["staged_orders"] = updated_staged
            _over = _exceeds_limit({"qty": new_qty, "price": new_price})
            c7.markdown(f'<div style="{_CELL}{";color:#b91c1c;font-weight:700" if _over else ""}" '
                        f'title="{"Exceeds max order amount" if _over else ""}">'
                        f'{"⚠️ " if _over else ""}₹{round(new_qty*new_price):,}</div>', unsafe_allow_html=True)
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
                o2 = dict(updated_staged[i])
                o2["qty"]   = int(st.session_state.get(f"le_qty_{i}",   o2["qty"]))
                o2["price"] = float(st.session_state.get(f"le_price_{i}", o2["price"]))
                o2["value"] = round(o2["qty"] * o2["price"], 2)
                if not bulk_acct:
                    st.error("Select an account before sending live orders.")
                elif st.session_state.get("le_funds_halt") and not dry_run:
                    st.error("🛑 Sending is halted after an insufficient-funds rejection. "
                             "Use “Resume sending” at the top.")
                else:
                    tk = token_map.get(o2.get("account","") or bulk_acct, first_tok)
                    note = ""
                    blocked_msg = ""
                    is_br = bool(o2.get("bracket"))
                    if is_br and not dry_run and not _probe_sent(updated_staged, o2):
                        blocked_msg = "Send the 1-qty probe order first — brackets are priced from its Positions LTP."
                    elif is_br or (st.session_state.get("le_reprice_on", True) and int(o2.get("tier", 1)) >= 2):
                        updated_staged[i] = o2
                        work = [dict(x) for x in updated_staged] if dry_run else updated_staged
                        ch, miss = _reprice_staged(work, [i], lambda _o: tk, persist=not dry_run)
                        o2 = dict(work[i])
                        note = _reprice_summary(ch, miss)
                        if is_br and o2.get("_unpriced"):
                            blocked_msg = f"{note} — bracket NOT sent (no Positions LTP)."
                    o2.pop("_unpriced", None)
                    if blocked_msg:
                        st.error(blocked_msg)
                    elif _exceeds_limit(o2):
                        st.error(f"🔒 {o2['ticker']}: ₹{o2['value']:,.0f} exceeds the max order "
                                 f"amount ₹{max_order_amt:,} — reduce the qty or raise the limit.")
                    elif dry_run:
                        _dry_run_record(o2, note)
                        st.session_state["le_send_flash"] = ("success",
                                                             f"🧪 Dry run: would send {o2['side']} {o2['qty']} {o2['ticker']} "
                                                             f"@ ₹{o2['price']:.2f}. {note}")
                        st.rerun()
                    else:
                        try:
                            resp = _do_place(tk, o2)
                            oid  = resp.get("data",{}).get("order_id","?")
                            o2.update({"state":"PLACED","order_id":oid,"placed_at":_now()})
                            st.session_state["le_send_flash"] = ("success",
                                                                 f"✅ {o2['ticker']} sent — order_id {oid}. {note}")
                        except Exception as e:
                            o2.update({"state":"ERROR","error":str(e)})
                            if _is_insufficient_funds(e):
                                _halt_on_funds(o2, e)
                            st.session_state["le_send_flash"] = ("error", f"❌ {o2['ticker']} failed: {e}")
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

    _sh, _sm, _sd = st.columns([2, 2, 1])
    _sh.markdown("##### Send to Exchange")
    _sm.number_input("Max order amount (₹)", min_value=0, value=0, step=10_000, key="le_max_order_amt",
                     help="Safety limit per order (qty × price). Orders above it are NOT sent to the "
                          "exchange — edit the qty or raise the limit. 0 = no limit.")
    _sd.toggle("🧪 Dry run", value=False, key="le_dry_run",
               help="Log what WOULD be sent. Nothing is sent to the exchange.")
    if max_order_amt:
        st.caption(f"🔒 Orders above ₹{max_order_amt:,} will be blocked.")
    if dry_run:
        st.warning("🧪 **DRY RUN is ON** — orders are only logged (see the Dry-run log in Part 3). "
                   "Nothing is sent to the exchange and order states do not change.")
    has_stagger = any(o.get("probe") or o.get("bracket") or int(o.get("tier", 1)) >= 2
                      for o in updated_staged)
    if has_stagger:
        st.toggle("Reprice STAGGER batches 2–4 from Positions LTP when sending",
                  value=True, key="le_reprice_on",
                  help="Applies to legacy tiers 2-4. STAGGER brackets (1-4) are ALWAYS priced from the "
                       "live Positions LTP and are held back until the 1-qty probe is sent and visible there.")
    sb, s1, s3, s2 = st.columns([1, 1, 1, 1])
    if has_stagger:
        batch_sel = sb.selectbox("Batch", ["All pending", "Probe", "Batch 1", "Batch 2", "Batch 3", "Batch 4"],
                                 key="le_batch_sel", label_visibility="collapsed",
                                 help="STAGGER: send the 1-qty Probe first, then Batch 1-4 (priced from Positions LTP).")
    else:
        batch_sel = "All pending"
    reprice_on = has_stagger and st.session_state.get("le_reprice_on", True)

    def _in_batch(i):
        t = int(updated_staged[i].get("tier", 1))
        return batch_sel == "All pending" or (t == 0 if batch_sel == "Probe" else t == int(batch_sel[-1]))
    batch_indices = [i for i in pending_indices if _in_batch(i)]

    if has_stagger and s3.button("🔄 Reprice brackets", use_container_width=True, key="le_reprice_now",
                                 disabled=not pending_indices,
                                 help="Update PENDING bracket prices from the live Positions LTP (nothing is sent)."):
        _tok = lambda _o: token_map.get(_o.get("account", "") or bulk_acct, first_tok)
        ch, miss = _reprice_staged(updated_staged, pending_indices, _tok)
        st.session_state["staged_orders"] = updated_staged
        st.session_state["live_orders"]   = updated_staged
        _save_orders(updated_staged)
        st.session_state["le_send_flash"] = ("success" if ch else "warning",
                                             _reprice_summary(ch, miss) or "Nothing to reprice.")
        st.rerun()

    if s1.button("⚡ Send Pending", type="primary",
                 use_container_width=True, disabled=not batch_indices,
                 key="le_send_all"):
        if not bulk_acct:
            st.error("Select an account before sending live orders.")
        elif st.session_state.get("le_funds_halt") and not dry_run:
            st.error("🛑 Sending is halted after an insufficient-funds rejection. "
                     "Use “Resume sending” at the top.")
        else:
            send_order = sorted(batch_indices, key=lambda i: (
                0 if str(updated_staged[i].get("side", "")).upper() == "SELL" else 1,
                int(updated_staged[i].get("tier", 1))))
            work = [dict(o) for o in updated_staged] if dry_run else updated_staged
            notes = ""
            _tok = lambda _o: token_map.get(_o.get("account", "") or bulk_acct, first_tok)
            rp_idx = send_order if reprice_on else [i for i in send_order if work[i].get("bracket")]
            if rp_idx:
                ch, miss = _reprice_staged(work, rp_idx, _tok, persist=not dry_run)
                notes = _reprice_summary(ch, miss)
            sent = blocked = held = 0
            halted_on = None
            for n, i in enumerate(send_order):
                o = dict(work[i])
                if o.get("bracket") and ((not dry_run and not _probe_sent(updated_staged, o))
                                         or o.get("_unpriced")):
                    held += 1
                    continue
                o.pop("_unpriced", None)
                if _exceeds_limit(o):
                    o["skip_reason"] = f"Exceeds max order amount ₹{max_order_amt:,}"
                    if not dry_run:
                        updated_staged[i] = o
                    blocked += 1
                    continue
                if dry_run:
                    _dry_run_record(o, "")
                    sent += 1
                    continue
                tk = token_map.get(o.get("account","") or bulk_acct, first_tok)
                try:
                    resp = _do_place(tk, o)
                    oid  = resp.get("data",{}).get("order_id","?")
                    o2 = dict(o)
                    o2.update({"state":"PLACED","order_id":oid,"placed_at":_now()})
                    o = o2
                    sent += 1
                except Exception as e:
                    o2 = dict(o)
                    o2.update({"state":"ERROR","error":str(e)})
                    o = o2
                    if _is_insufficient_funds(e):
                        _halt_on_funds(o, e)
                        halted_on = o
                updated_staged[i] = o
                if halted_on:
                    break
                if n < len(send_order) - 1:
                    time.sleep(0.5)
            remaining = sum(1 for o in updated_staged if o.get("state") == "PENDING")
            if held:
                notes += (f"  ·  ⏸ {held} bracket(s) held back — send the 1-qty probe first and wait "
                          f"until the ticker shows in Positions, then send the brackets.")
            st.session_state["staged_orders"] = updated_staged
            st.session_state["live_orders"]   = updated_staged
            _save_orders(updated_staged)
            st.session_state.pop("le_exchange_orders", None)
            if dry_run:
                st.session_state["le_send_flash"] = ("success",
                                                     f"🧪 Dry run ({batch_sel}): logged {sent} order(s), {blocked} blocked by the limit — "
                                                     f"nothing was sent. {notes}")
            elif halted_on:
                st.session_state["le_send_flash"] = ("error",
                                                     f"🛑 Stopped after {sent} order(s): {halted_on.get('side')} {halted_on.get('ticker')} "
                                                     f"rejected — {halted_on.get('error')}. {remaining} order(s) left unsent. {notes}")
            elif blocked:
                st.session_state["le_send_flash"] = ("warning",
                                                     f"Sent {sent} order(s); {blocked} blocked by the ₹{max_order_amt:,} max order amount "
                                                     f"(still pending — edit qty or raise the limit). {notes}")
            else:
                st.session_state["le_send_flash"] = ("success",
                                                     f"✅ Sent {sent} order(s) ({batch_sel}, SELLs first). {notes}")
            st.rerun()

    if s2.button("🗑 Clear Staged", use_container_width=True, key="le_clear_staged"):
        st.session_state["staged_orders"]   = []
        st.session_state["le_staged_built"] = False
        st.rerun()

    _dlog = st.session_state.get("le_dry_log", [])
    if _dlog:
        with st.expander(f"🧪 Dry-run log ({len(_dlog)})", expanded=dry_run):
            _ddf = pd.DataFrame(_dlog)
            st.dataframe(_ddf, use_container_width=True, hide_index=True)
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Download CSV", _ddf.to_csv(index=False).encode(),
                               file_name="dry_run_log.csv", mime="text/csv", key="le_dry_dl")
            if d2.button("🧹 Clear log", key="le_dry_clear"):
                st.session_state["le_dry_log"] = []
                st.rerun()