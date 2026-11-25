"""
page_rebalance.py
─────────────────────────────────────────────────────────────────────────────
Portfolio Rebalancer — integrates with existing session state from rebalance.py

Session state it READS:
  all_holdings   : {account_name: [holding_dict, …]}   (ticker, quantity, last_price, …)
  excluded_symbols: set of symbols to ignore

Session state it WRITES on Approve:
  rebalance_plan  : list of approved order dicts
                    [{ticker, side, qty, value_inr, current_wt, target_wt, action}, …]

Usage (in app.py):
  import ui.pages.page_rebalance as page_rebalance
  page_rebalance.render()
"""

from __future__ import annotations

import re
import json
import os
import time

import pandas as pd
import streamlit as st
from datetime import datetime, timezone, timedelta

PLAN_FILE = "outputs/rebalance_plan.json"
PLAN_TTL_HOURS = 24


def _save_plan_to_disk(plan: list) -> None:
    os.makedirs(os.path.dirname(PLAN_FILE), exist_ok=True)
    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
    }
    with open(PLAN_FILE, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _load_plan_from_disk() -> list:
    if not os.path.exists(PLAN_FILE):
        return []
    try:
        with open(PLAN_FILE) as f:
            payload = json.load(f)
        saved_at = datetime.fromisoformat(payload["saved_at"])
        if datetime.now(timezone.utc) - saved_at > timedelta(hours=PLAN_TTL_HOURS):
            return []   # expired
        return payload.get("plan", [])
    except Exception:
        return []

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

# ─────────────────────────────────────────────────────────────────────────────
# Price helper (yfinance, falls back gracefully)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_price(symbol: str) -> float:
    if not HAS_YF:
        return 0.0
    for suffix in (".NS", ".BO", ""):
        try:
            t    = yf.Ticker(f"{symbol}{suffix}")
            info = t.fast_info
            p    = float(info.get("last_price") or info.get("regularMarketPrice") or 0)
            if p > 0:
                return p
            hist = t.history(period="2d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception:
            continue
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Parse manual orders  (TICKER<TAB>+val = BUY,  TICKER<TAB>-val = SELL;  val = qty or ₹ amount)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_qty(raw: str) -> int | None:
    """Parse a signed share quantity: '+10', '-25', '1,000'.  No sign = positive (BUY).
    Must be a whole number; anything else returns None."""
    t = raw.strip().replace(",", "").replace(" ", "")
    if not re.fullmatch(r"[+-]?\d+", t):
        return None
    return int(t)


def _parse_amount(raw: str) -> float | None:
    """Parse a signed INR amount: '+150000', '-1,50,000', '+₹1.5L', '-2cr', '75k'.
    No sign = positive (BUY)."""
    t = raw.strip().lower().replace("₹", "").replace("rs.", "").replace("rs", "")
    t = t.replace(",", "").replace(" ", "")
    mult = 1.0
    for suffix, m in (("cr", 1e7), ("l", 1e5), ("k", 1e3)):
        if t.endswith(suffix):
            t, mult = t[: -len(suffix)], m
            break
    try:
        return float(t) * mult
    except ValueError:
        return None


def _parse_orders_text(text: str, unit: str = "qty") -> dict[str, float]:
    """Return {TICKER: signed value}.  + = BUY, - = SELL (unsigned = BUY).
    unit="qty"    → value is a whole number of shares
    unit="amount" → value is an INR amount
    Accepts 'TICKER<TAB>val', 'TICKER, val', 'TICKER: val' or 'TICKER val'.
    If a ticker appears twice, the values are netted."""
    parse = _parse_qty if unit == "qty" else _parse_amount
    out: dict[str, float] = {}
    for line in text.strip().splitlines():
        m = re.match(r"^\s*([A-Za-z0-9&_\-\.]+)\s*(?:[\t,:=]|\s)\s*(.+?)\s*$", line)
        if not m:
            continue
        val = parse(m.group(2))
        if val is None or val == 0:
            continue
        tk = m.group(1).upper()
        out[tk] = out.get(tk, 0) + val
    return {k: v for k, v in out.items() if v != 0}


# ─────────────────────────────────────────────────────────────────────────────
# Build current portfolio from session state
# ─────────────────────────────────────────────────────────────────────────────

def _build_current(excl_set: set) -> pd.DataFrame:
    all_holdings: dict = st.session_state.get("all_holdings") or {}
    rows: dict[str, dict] = {}
    for acc_name, holdings in all_holdings.items():
        for h in holdings:
            sym = h.get("tradingsymbol", "").upper()
            if not sym or sym in excl_set:
                continue
            if sym not in rows:
                rows[sym] = {
                    "ticker":      sym,
                    "qty":         0,
                    "last_price":  float(h.get("last_price") or 0),
                    "exchange":    h.get("exchange", "NSE"),
                }
            rows[sym]["qty"] += int(h.get("quantity") or 0)
    df = pd.DataFrame(list(rows.values())) if rows else pd.DataFrame(
        columns=["ticker", "qty", "last_price", "exchange"]
    )
    if not df.empty:
        df["curr_value"] = df["qty"] * df["last_price"]
    else:
        df["curr_value"] = pd.Series(dtype=float)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:

    # ── session defaults ──────────────────────────────────────────────────────
    for k, v in [
        ("rb_manual_vals",     {}),   # {TICKER: signed qty or ₹}  + buy / - sell
        ("rb_manual_unit_applied", "qty"),   # unit the applied orders were entered in
        ("rb_ignored",         set()),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v
    # Load persisted plan on first run (respects 24h TTL)
    if "rebalance_plan" not in st.session_state:
        st.session_state["rebalance_plan"] = _load_plan_from_disk()

    excl_set: set = st.session_state.get("excluded_symbols") or set()

    # ── CSS ───────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .rb-section { font-size:0.82rem; color:#6b7280; font-weight:600;
                  letter-spacing:.06em; text-transform:uppercase; margin-bottom:4px; }
    .rb-pill-buy  { display:inline-block; padding:2px 10px; border-radius:99px;
                    background:#dcfce7; color:#15803d; font-weight:700;
                    font-size:0.78rem; margin:1px 2px; }
    .rb-pill-sell { display:inline-block; padding:2px 10px; border-radius:99px;
                    background:#fee2e2; color:#b91c1c; font-weight:700;
                    font-size:0.78rem; margin:1px 2px; }
    .rb-pill-hold { display:inline-block; padding:2px 10px; border-radius:99px;
                    background:#f3f4f6; color:#6b7280; font-weight:600;
                    font-size:0.78rem; margin:1px 2px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## ⚖️ Portfolio Rebalancer")

    # ══════════════════════════════════════════════════════════════════════════
    # MODE SELECTOR
    # ══════════════════════════════════════════════════════════════════════════
    mode = st.radio(
        "Rebalance mode",
        [ "✏️ Manual (+ Buy / - Sell ₹)", "💰 Scale Capital (+/- %)"],
        horizontal=True,
        key="rb_mode_v2",
        help=(
            "**Scale Capital**: add or remove a % of capital; every holding is bought "
            "or sold proportionally so current weights are preserved.\n\n"
            "**Manual**: enter `TICKER +value` to buy or `TICKER -value` to sell, where value is either a share quantity or a ₹ amount (your choice)."
        ),
    )
    is_scale_mode = mode.startswith("💰")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — CURRENT PORTFOLIO (Scale Capital mode only)
    #   Manual mode never touches existing holdings.
    # ══════════════════════════════════════════════════════════════════════════
    curr_df    = pd.DataFrame()
    total_curr = 0.0

    if is_scale_mode:
        with st.expander("📋 Step 1 — Current Portfolio", expanded=True):
            curr_df = _build_current(excl_set)

            if curr_df.empty:
                st.info("No holdings found in session. "
                        "Go to the Portfolio page, connect your broker and Fetch Holdings.")
                st.stop()

            total_curr = curr_df["curr_value"].sum()
            curr_df["curr_wt%"] = (curr_df["curr_value"] / total_curr * 100).round(2) \
                if total_curr else 0.0

            st.caption(f"**{len(curr_df)} holdings** · Total ₹{total_curr:,.0f}")
            st.dataframe(
                curr_df[["ticker", "curr_wt%", "qty", "last_price", "curr_value"]].rename(columns={
                    "ticker":     "Ticker",
                    "curr_wt%":   "Wt %",
                    "qty":        "Qty",
                    "last_price": "Last Price ₹",
                    "curr_value": "Value ₹",
                }),
                use_container_width=True, hide_index=True,
                column_config={
                    "Last Price ₹": st.column_config.NumberColumn(format="₹%.2f"),
                    "Value ₹":      st.column_config.NumberColumn(format="₹%.0f"),
                    "Wt %":         st.column_config.NumberColumn(format="%.2f%%"),
                },
            )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — INPUT
    #   Scale Capital : +/- %  applied to every holding
    #   Manual        : TICKER +val (BUY) / -val (SELL);  val = Qty or ₹ Amount
    # ══════════════════════════════════════════════════════════════════════════
    REBAL_PCT = 100.0          # always execute 100% of the computed delta
    scale_pct = 0.0
    capital_delta = 0.0

    if is_scale_mode:
        with st.expander("💰 Step 2 — Scale Capital", expanded=True):
            scale_pct = st.number_input(
                "Capital change (%)",
                min_value=-100.0, max_value=1000.0, step=1.0, format="%.2f",
                value=0.0, key="rb_scale_pct",
                help=(
                    "Positive = deploy fresh capital: buy that % more of every holding.\n\n"
                    "Negative = withdraw capital: sell that % of every holding.\n\n"
                    "Current weights are preserved."
                ),
            )
            capital_delta = total_curr * scale_pct / 100
            if scale_pct > 0:
                st.caption(f"➕ Deploying **₹{capital_delta:,.0f}** (+{scale_pct:.2f}% of "
                           f"₹{total_curr:,.0f}) across all holdings proportionally.")
            elif scale_pct < 0:
                st.caption(f"➖ Withdrawing **₹{abs(capital_delta):,.0f}** ({scale_pct:.2f}% of "
                           f"₹{total_curr:,.0f}) by selling every holding proportionally.")
            else:
                st.caption("No change — enter a positive % to buy more or a negative % to sell.")
    else:
        def _on_unit_change() -> None:
            # Applied orders were entered in the other unit — drop them so
            # numbers are never reinterpreted (e.g. 50000 shares vs ₹50,000).
            st.session_state["rb_manual_vals"] = {}

        with st.expander("✏️ Step 1 — Manual Orders", expanded=True):
            unit_label = st.radio(
                "Enter orders as",
                ["Qty (shares)", "Amount (₹)"],
                horizontal=True, key="rb_manual_unit", on_change=_on_unit_change,
            )
            unit = "qty" if unit_label.startswith("Qty") else "amount"

            if unit == "qty":
                st.caption(
                    "One per line:  `TICKER<TAB>+qty` to **BUY**,  `TICKER<TAB>-qty` to **SELL** "
                    "(whole shares; e.g. `10`, `+25`, `-1,000`). No sign = BUY. "
                    "Tickers you don't list are left unchanged."
                )
                _ph = "RELIANCE\t+10\nINFY\t-25\nHDFCBANK\t+100"
            else:
                st.caption(
                    "One per line:  `TICKER<TAB>+amount` to **BUY**,  `TICKER<TAB>-amount` to **SELL** "
                    "(₹; accepts `50000`, `1,50,000`, `75k`, `1.5L`, `2cr`). No sign = BUY. "
                    "Tickers you don't list are left unchanged."
                )
                _ph = "RELIANCE\t+150000\nINFY\t-50000\nHDFCBANK\t+1,20,000"

            manual_txt = st.text_area(
                "manual orders", height=160,
                label_visibility="collapsed",
                key=f"rb_manual_txt_{unit}",      # separate box per unit
                placeholder=_ph,
            )
            if st.button("✅ Apply Orders", key="rb_apply_manual"):
                parsed_orders = _parse_orders_text(manual_txt, unit)
                if parsed_orders:
                    st.session_state["rb_manual_vals"] = parsed_orders
                    st.session_state["rb_manual_unit_applied"] = unit
                    st.success(f"Applied {len(parsed_orders)} orders.")
                else:
                    st.error("Could not parse any orders. Use TICKER<TAB>+"
                             + ("qty (buy) or TICKER<TAB>-qty (sell), whole numbers only."
                                if unit == "qty" else
                                "amount (buy) or TICKER<TAB>-amount (sell)."))

            _orders: dict[str, float] = st.session_state["rb_manual_vals"]
            if _orders:
                st.divider()
                if st.session_state["rb_manual_unit_applied"] == "qty":
                    _n_buy  = sum(1 for v in _orders.values() if v > 0)
                    _n_sell = sum(1 for v in _orders.values() if v < 0)
                    st.markdown(
                        f"**Active orders:** {len(_orders)} tickers · "
                        f"🟢 Buy **{_n_buy}** · 🔴 Sell **{_n_sell}**"
                    )
                    _col, _cfg = "Qty", st.column_config.NumberColumn(format="%d")
                else:
                    _buy_tot  = sum(v for v in _orders.values() if v > 0)
                    _sell_tot = -sum(v for v in _orders.values() if v < 0)
                    st.markdown(
                        f"**Active orders:** {len(_orders)} tickers · "
                        f"🟢 Buy **₹{_buy_tot:,.0f}** · 🔴 Sell **₹{_sell_tot:,.0f}** · "
                        f"Net **₹{_buy_tot - _sell_tot:+,.0f}**"
                    )
                    _col, _cfg = "Amount ₹", st.column_config.NumberColumn(format="₹%.0f")
                st.dataframe(
                    pd.DataFrame({
                        "Ticker": list(_orders.keys()),
                        "Side":   ["BUY" if v > 0 else "SELL" for v in _orders.values()],
                        _col:     [abs(v) for v in _orders.values()],
                    }),
                    use_container_width=True, hide_index=True,
                    column_config={_col: _cfg},
                )

    if not is_scale_mode and not st.session_state["rb_manual_vals"]:
        # Allow bypass if a saved plan was loaded from disk
        if not st.session_state.get("rebalance_plan"):
            st.stop()

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD DIFF TABLE — skip if no manual orders (show saved plan only)
    # ══════════════════════════════════════════════════════════════════════════
    if not is_scale_mode and not st.session_state["rb_manual_vals"]:
        # No weights yet — just show the last approved plan and stop
        with st.expander("📌 Last Approved Plan", expanded=True):
            plan_df = pd.DataFrame(st.session_state["rebalance_plan"])
            st.dataframe(plan_df, use_container_width=True, hide_index=True)

            early_c1, early_c2, _ = st.columns([1, 1, 3])
            if early_c1.button("🗑 Clear Plan", key="rb_clear_plan_early", use_container_width=True):
                st.session_state["rebalance_plan"] = []
                if os.path.exists(PLAN_FILE):
                    os.remove(PLAN_FILE)
                st.rerun()
            if early_c2.button("🚀 Execute Plan", key="rb_execute_plan_early",
                               type="primary", use_container_width=True,
                               help="Loads this plan into session for live order submission to the exchange."):
                plan = st.session_state["rebalance_plan"]
                st.session_state["execution_plan"] = plan
                st.session_state["execution_plan_ready"] = True
                n_buy  = len([o for o in plan if o.get("action") == "BUY"])
                n_sell = len([o for o in plan if o.get("action") == "SELL"])
                st.success(
                    f"✅ Plan loaded for execution — {len(plan)} orders "
                    f"({n_buy} BUY · {n_sell} SELL). "
                    f"Ready to send to the exchange execution program."
                )
                with st.spinner("Redirecting to Live Execution in 4 seconds..."):
                    time.sleep(4)
                    st.session_state["active_page"] = "trade_live_execution"
                    st.rerun()
        st.info("Enter orders in Step 2 above to build a new rebalance plan.")
        return

    if is_scale_mode:
        curr_map = curr_df.set_index("ticker").to_dict("index")

        # Signed ₹ order per ticker  (+ buy / - sell)
        if is_scale_mode:
            order_amt = {tk: float(c["curr_value"]) * scale_pct / 100 for tk, c in curr_map.items()}
        else:
            order_amt = dict(st.session_state["rb_manual_vals"])

        all_tickers = sorted(set(curr_map.keys()) | set(order_amt.keys()))

        diff_rows: list[dict] = []
        capped_sells: list[str] = []
        no_price: list[str] = []
        for tk in all_tickers:
            c         = curr_map.get(tk, {})
            curr_w    = float(curr_df.loc[curr_df["ticker"] == tk, "curr_wt%"].values[0]) \
                if tk in curr_map else 0.0
            price     = float(c.get("last_price", 0))
            if price == 0:
                price = _fetch_price(tk)
            curr_qty  = int(c.get("qty", 0))
            amt       = order_amt.get(tk, 0.0)

            if price > 0:
                delta_qty = int(amt / price)          # truncates toward zero
            else:
                delta_qty = 0
                if amt > 0 or (amt < 0 and tk in curr_map):
                    no_price.append(tk)
            if delta_qty < -curr_qty:                 # can't sell more than held
                if amt < 0:
                    capped_sells.append(tk)
                delta_qty = -curr_qty

            tgt_qty   = curr_qty + delta_qty
            today_qty = int(delta_qty * REBAL_PCT / 100)
            today_val = round(today_qty * price, 2)

            if delta_qty > 0:
                action = "BUY"
            elif delta_qty < 0:
                action = "SELL"
            else:
                action = "HOLD"

            diff_rows.append({
                "Ticker":      tk,
                "Curr Wt%":    round(curr_w, 2),
                "_tgt_val":    tgt_qty * price,
                "Curr Qty":    curr_qty,
                "Target Qty":  tgt_qty,
                "Δ Qty":       delta_qty,
                "Today Qty":   today_qty,
                "Price ₹":     round(price, 2),
                "Today ₹":     today_val,
                "Action":      action,
                "_ignore":     tk in st.session_state["rb_ignored"],
            })

        # Target weight = share of the post-trade portfolio
        _tgt_total = sum(r["_tgt_val"] for r in diff_rows)
        for r in diff_rows:
            tgt_w = (r.pop("_tgt_val") / _tgt_total * 100) if _tgt_total else 0.0
            r["Target Wt%"] = round(tgt_w, 2)
            r["Δ Wt%"]      = round(tgt_w - r["Curr Wt%"], 2) + 0.0   # +0.0 avoids "-0.0"

        if capped_sells:
            st.warning("Sell amount exceeds holding — capped at full position: "
                       + ", ".join(capped_sells))
        if no_price:
            st.warning("No price available, order skipped: " + ", ".join(no_price))

        diff_df = pd.DataFrame(diff_rows)

        # ══════════════════════════════════════════════════════════════════════════
        # STEP 3 — DIFF VIEW + IGNORE
        # ══════════════════════════════════════════════════════════════════════════
        with st.expander("📊 Step 3 — Portfolio Diff", expanded=True):
            st.caption("Review the difference between your current and post-trade portfolio. "
                       "Toggle **Ignore** to skip any ticker from this rebalance.")

            edit_diff = st.data_editor(
                diff_df[["Ticker", "Curr Wt%", "Target Wt%", "Δ Wt%",
                         "Curr Qty", "Target Qty", "Δ Qty",
                         "Today Qty", "Price ₹", "Today ₹", "Action", "_ignore"]].rename(
                    columns={"_ignore": "Ignore"}
                ),
                use_container_width=True, hide_index=True,
                column_config={
                    "Curr Wt%":   st.column_config.NumberColumn(format="%.2f%%"),
                    "Target Wt%": st.column_config.NumberColumn(format="%.2f%%"),
                    "Δ Wt%":      st.column_config.NumberColumn(format="%.2f%%"),
                    "Price ₹":    st.column_config.NumberColumn(format="₹%.2f"),
                    "Today ₹":    st.column_config.NumberColumn(format="₹%.0f"),
                    "Action":     st.column_config.SelectboxColumn(
                        options=["BUY", "SELL", "HOLD"], width="small"
                    ),
                    "Ignore":     st.column_config.CheckboxColumn(
                        "Ignore / Hold", width="small",
                        help="Check to exclude this ticker from the rebalance plan."
                    ),
                },
                disabled=["Ticker", "Curr Wt%", "Target Wt%", "Δ Wt%",
                          "Curr Qty", "Target Qty", "Δ Qty",
                          "Today Qty", "Price ₹", "Today ₹"],
                key="rb_diff_editor",
            )

            # Persist ignore state
            st.session_state["rb_ignored"] = set(
                edit_diff.loc[edit_diff["Ignore"] == True, "Ticker"].tolist()
            )

            # Summary chips — only count rows that pass the ₹5,000 threshold
            MIN_TRADE = 1_000
            _active_mask = (~edit_diff["Ignore"]) & (edit_diff["Today ₹"].abs() >= MIN_TRADE)
            buys    = edit_diff[_active_mask & (edit_diff["Action"] == "BUY")]
            sells   = edit_diff[_active_mask & (edit_diff["Action"] == "SELL")]
            below   = edit_diff[(~edit_diff["Ignore"]) & (edit_diff["Action"].isin(["BUY","SELL"])) & (edit_diff["Today ₹"].abs() < MIN_TRADE)]
            holds   = edit_diff[(edit_diff["Action"] == "HOLD") | edit_diff["Ignore"]]
            ignored = edit_diff[edit_diff["Ignore"] == True]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🟢 BUY",    len(buys),    f"₹{buys['Today ₹'].sum():,.0f}")
            c2.metric("🔴 SELL",   len(sells),   f"₹{sells['Today ₹'].abs().sum():,.0f}")
            c3.metric("⚪ HOLD / < ₹5K",  len(holds) + len(below))
            c4.metric("🚫 Ignored", len(ignored))
            if not below.empty:
                skipped_list = ", ".join(
                    f"{r['Ticker']} (₹{abs(r['Today ₹']):,.0f})" for _, r in below.iterrows()
                )
                st.caption(f"⏭ Below ₹5,000 threshold (skipped): {skipped_list}")

    else:
        # ── Manual mode: no holdings lookup, no weights, no diff view ─────────
        _rows: list[dict] = []
        no_price: list[str] = []
        _unit = st.session_state["rb_manual_unit_applied"]
        for tk, val in sorted(st.session_state["rb_manual_vals"].items()):
            price = _fetch_price(tk)
            if price <= 0:
                no_price.append(tk)
                continue
            qty = int(val) if _unit == "qty" else int(val / price)   # amount → truncate toward 0
            if qty == 0:
                continue
            _rows.append({
                "Ticker":    tk,
                "Action":    "BUY" if qty > 0 else "SELL",
                "Today Qty": qty,
                "Price ₹":   round(price, 2),
                "Today ₹":   round(qty * price, 2),
                "Ignore":    False,
            })
        if no_price:
            st.warning("No price available, order skipped: " + ", ".join(no_price))
        edit_diff = pd.DataFrame(
            _rows, columns=["Ticker", "Action", "Today Qty", "Price ₹", "Today ₹", "Ignore"]
        )
        edit_diff["Ignore"] = edit_diff["Ignore"].astype(bool)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — FINAL REBALANCE PLAN
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("🚀 Step 4 — Rebalance Plan" if is_scale_mode
                     else "🚀 Step 2 — Rebalance Plan", expanded=True):

        MIN_TRADE = 1_000
        active = edit_diff[
            (~edit_diff["Ignore"]) &
            (edit_diff["Action"].isin(["BUY", "SELL"])) &
            (edit_diff["Today ₹"].abs() >= (
                0 if (not is_scale_mode and
                      st.session_state["rb_manual_unit_applied"] == "qty") else MIN_TRADE))
            ].copy()

        if active.empty:
            st.info("Nothing to execute. Either all tickers are ignored, "
                    "held, or delta qty is 0.")
        else:
            # Style rows
            def _row_style(row):
                if row["Action"] == "BUY":
                    return ["background-color:#f0fdf4; color:#15803d"] * len(row)
                if row["Action"] == "SELL":
                    return ["background-color:#fff1f2; color:#b91c1c"] * len(row)
                return [""] * len(row)

            _plan_cols = ["Ticker", "Action", "Today Qty", "Price ₹", "Today ₹"]
            _fmt = {"Price ₹": "₹{:.2f}", "Today ₹": "₹{:,.0f}"}
            if is_scale_mode:                      # weights only make sense vs. holdings
                _plan_cols += ["Curr Wt%", "Target Wt%", "Δ Wt%"]
                _fmt.update({"Curr Wt%": "{:.2f}%", "Target Wt%": "{:.2f}%",
                             "Δ Wt%": "{:+.2f}%"})
            styled = active[_plan_cols].style.apply(_row_style, axis=1).format(_fmt)
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # Fee estimate
            buy_val  = active[active["Action"] == "BUY"]["Today ₹"].sum()
            sell_val = active[active["Action"] == "SELL"]["Today ₹"].abs().sum()
            turnover = buy_val + sell_val
            est_fees = round(turnover * 0.001, 2)   # rough STT only estimate
            n_sell   = int((active["Action"] == "SELL").sum())

            f1, f2, f3, f4, f5 = st.columns(5)
            f1.metric("Buy",         f"₹{buy_val:,.0f}")
            f2.metric("Sell",        f"₹{sell_val:,.0f}")
            f3.metric("Net Outflow", f"₹{buy_val - sell_val:,.0f}")
            f4.metric("Est. STT",    f"₹{est_fees:,.0f}")
            if is_scale_mode:
                f5.metric("Capital Change", f"{scale_pct:+.2f}%",
                          f"₹{capital_delta:+,.0f}", delta_color="off")
            else:
                f5.metric("Orders", len(active))

            if is_scale_mode:
                if capital_delta > 0:
                    st.caption(f"➕ Deploying **₹{capital_delta:,.0f}** (+{scale_pct:.2f}%) "
                               f"across holdings at current weights.")
                elif capital_delta < 0:
                    st.caption(f"➖ Withdrawing **₹{abs(capital_delta):,.0f}** ({scale_pct:.2f}%) "
                               f"by selling proportionally.")
                else:
                    st.caption("No capital change specified — enter a % in Step 2.")

            st.divider()
            approve_col, _ = st.columns([1, 3])
            if approve_col.button("✅ Approve & Execute",
                                  type="primary", use_container_width=True):
                plan = []
                for _, row in active.iterrows():
                    order = {
                        "ticker":           row["Ticker"],
                        "action":           row["Action"],
                        "today_qty":        int(row["Today Qty"]),
                        "price":            float(row["Price ₹"]),
                        "today_value":      float(row["Today ₹"]),
                        "side":             row["Action"],   # alias for order engine
                        "rebalance_mode":   "scale_capital" if is_scale_mode else "manual_orders",
                    }
                    if is_scale_mode:
                        order.update({
                            "curr_wt":   float(row["Curr Wt%"]),
                            "target_wt": float(row["Target Wt%"]),
                            "delta_wt":  float(row["Δ Wt%"]),
                        })
                    plan.append(order)
                st.session_state["rebalance_plan"] = plan
                _save_plan_to_disk(plan)
                st.success(
                    f"✅ Rebalance plan saved to session & disk — "
                    f"{len(plan)} orders "
                    f"({len([p for p in plan if p['action']=='BUY'])} BUY · "
                    f"{len([p for p in plan if p['action']=='SELL'])} SELL). "
                    f"Auto-expires in 24h."
                )
                with st.spinner("Redirecting to Live Execution in 4 seconds..."):
                    time.sleep(4)
                st.session_state["active_page"] = "trade_live_execution"
                st.rerun()

    # ── Show last approved plan ────────────────────────────────────────────────
    with st.expander("📌 Last Approved Plan", expanded=False):
        disk_meta = {}
        if os.path.exists(PLAN_FILE):
            try:
                with open(PLAN_FILE) as _f:
                    disk_meta = json.load(_f)
            except Exception:
                pass

        saved_at_str = disk_meta.get("saved_at", "")
        if saved_at_str:
            try:
                saved_dt  = datetime.fromisoformat(saved_at_str)
                expires_dt = saved_dt + timedelta(hours=PLAN_TTL_HOURS)
                now_utc    = datetime.now(timezone.utc)
                expired    = now_utc > expires_dt
                age_str    = saved_dt.strftime("%d %b %Y %H:%M UTC")
                exp_str    = expires_dt.strftime("%d %b %Y %H:%M UTC")
                if expired:
                    st.caption(f"⏰ Saved plan expired (saved {age_str})")
                else:
                    remaining = expires_dt - now_utc
                    hrs = int(remaining.total_seconds() // 3600)
                    mins = int((remaining.total_seconds() % 3600) // 60)
                    st.caption(f"💾 Saved {age_str} · expires {exp_str} (in {hrs}h {mins}m)")
            except Exception:
                pass

        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
        if btn_col1.button("📂 Load from disk", key="rb_load_plan", use_container_width=True):
            loaded = _load_plan_from_disk()
            if loaded:
                st.session_state["rebalance_plan"] = loaded
                st.success(f"Loaded {len(loaded)} orders from disk.")
                st.rerun()
            else:
                st.warning("No valid plan on disk (missing or expired).")

        if btn_col2.button("🗑 Clear Plan", key="rb_clear_plan", use_container_width=True):
            st.session_state["rebalance_plan"] = []
            if os.path.exists(PLAN_FILE):
                os.remove(PLAN_FILE)
            st.rerun()

        if st.session_state["rebalance_plan"]:
            plan_df = pd.DataFrame(st.session_state["rebalance_plan"])
            st.dataframe(plan_df, use_container_width=True, hide_index=True)

            st.divider()
            exec_col, _ = st.columns([1, 3])
            if exec_col.button("🚀 Execute Plan", key="rb_execute_plan",
                               type="primary", use_container_width=True,
                               help="Loads this plan into session for live order submission to the exchange."):
                plan = st.session_state["rebalance_plan"]
                st.session_state["execution_plan"] = plan
                st.session_state["execution_plan_ready"] = True
                n_buy  = len([o for o in plan if o.get("action") == "BUY"])
                n_sell = len([o for o in plan if o.get("action") == "SELL"])
                st.success(
                    f"✅ Plan loaded for execution — {len(plan)} orders "
                    f"({n_buy} BUY · {n_sell} SELL). "
                    f"Ready to send to the exchange execution program."
                )
        else:
            st.caption("No approved plan in session.")