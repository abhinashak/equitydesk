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

import io
import csv
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
# Parse target weights from text  (TICKER<TAB>weight  or  TICKER,weight)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_weights_text(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.strip().splitlines():
        parts = re.split(r"[\t,]+", line.strip())
        if len(parts) >= 2:
            ticker = parts[0].strip().upper()
            try:
                out[ticker] = float(parts[1].strip())
            except ValueError:
                pass
    return out


def _parse_weights_csv(uploaded) -> dict[str, float]:
    raw    = uploaded.read().decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    fields = reader.fieldnames or []
    t_col  = next((c for c in fields if "ticker" in c.lower()), None)
    w_col  = next((c for c in fields if "weight" in c.lower()), None)
    if not t_col:
        return {}
    out: dict[str, float] = {}
    for row in reader:
        ticker = row[t_col].strip().upper().split(".")[0]
        if not ticker:
            continue
        if w_col:
            try:
                out[ticker] = float(row[w_col])
                continue
            except (ValueError, KeyError):
                pass
        # No weight column — equal weight (filled in later)
        out[ticker] = 0.0
    return out


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
        ("rb_target_weights",  {}),
        ("rb_rebal_pct",       100.0),
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
        ["🔀 Rebalance with new weights", "💰 Scale capital (keep existing weights)"],
        horizontal=True,
        key="rb_mode",
        help=(
            "**Rebalance**: supply new target weights and move the portfolio to match them.\n\n"
            "**Scale capital**: keep the current weightings; add fresh capital to buy more "
            "of everything proportionally, or reduce capital to sell proportionally."
        ),
    )
    is_scale_mode = mode.startswith("💰")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — CURRENT PORTFOLIO (from session)
    # ══════════════════════════════════════════════════════════════════════════
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
    # STEP 2 — TARGET WEIGHTS  (Rebalance mode only)
    # ══════════════════════════════════════════════════════════════════════════
    if is_scale_mode:
        # In scale mode: use current weights as target, skip weight input UI
        if not curr_df.empty:
            tw_from_curr = {
                row["ticker"]: round(float(row["curr_wt%"]), 4)
                for _, row in curr_df.iterrows()
            }
            st.session_state["rb_target_weights"] = tw_from_curr
        st.info(
            "💰 **Scale Capital mode** — target weights are locked to your current portfolio. "
            "Set the capital change amount in Step 3 below."
        )

    if not is_scale_mode:
        with st.expander("🎯 Step 2 — Target Weights", expanded=True):
            st.markdown('<div class="rb-section">Choose how to supply target weights</div>',
                        unsafe_allow_html=True)

            tab_manual, tab_file = st.tabs(["✏️ Manual", "📂 File Upload"])

            # ── Manual ────────────────────────────────────────────────────────────
            with tab_manual:
                st.caption("One per line:  `TICKER<TAB>weight%`  or  `TICKER, weight%`")
                manual_txt = st.text_area(
                    "target weights manual", height=160,
                    label_visibility="collapsed",
                    key="rb_manual_txt",
                    placeholder="RELIANCE\t15\nINFY\t10\nHDFCBANK\t12",
                )
                if st.button("✅ Apply Manual Weights", key="rb_apply_manual"):
                    parsed = _parse_weights_text(manual_txt)
                    if parsed:
                        st.session_state["rb_target_weights"] = parsed
                        st.success(f"Applied {len(parsed)} target weights.")
                    else:
                        st.error("Could not parse any weights. Use TICKER<TAB>weight format.")

            # ── File Upload ───────────────────────────────────────────────────────
            with tab_file:
                st.caption("CSV with `ticker` column + optional `weight` column. "
                           "If no weight column, equal weights are assigned.")
                uw = st.file_uploader("Upload weights CSV", type=["csv", "txt"],
                                      key="rb_weights_upload",
                                      label_visibility="collapsed")
                if uw:
                    parsed_file = _parse_weights_csv(uw)
                    if parsed_file:
                        # Fill equal weights if all zero
                        if all(v == 0.0 for v in parsed_file.values()):
                            eq = round(100 / len(parsed_file), 4)
                            parsed_file = {k: eq for k in parsed_file}
                        st.session_state["rb_file_weights_parsed"] = parsed_file
                        st.caption(f"Parsed {len(parsed_file)} tickers from file.")
                        tickers_preview = ", ".join(list(parsed_file.keys())[:10])
                        if len(parsed_file) > 10:
                            tickers_preview += f" … +{len(parsed_file)-10} more"
                        st.info(f"**Tickers:** {tickers_preview}")
                    else:
                        st.error("Could not parse file. Ensure it has a `ticker` column.")
                if st.button("✅ Apply File Weights", key="rb_apply_file",
                             disabled="rb_file_weights_parsed" not in st.session_state):
                    st.session_state["rb_target_weights"] = \
                        st.session_state["rb_file_weights_parsed"]
                    st.success(f"Applied {len(st.session_state['rb_target_weights'])} weights.")

            # ── Show active target ────────────────────────────────────────────────
            tw: dict[str, float] = st.session_state["rb_target_weights"]
            if tw:
                total_tw = sum(tw.values())
                ok = abs(total_tw - 100) < 0.5
                st.divider()
                cash_pct = max(0.0, 100 - total_tw)
                if total_tw > 100:
                    st.markdown(
                        f"**Active target:** {len(tw)} tickers · "
                        f"🔴 Sum = **{total_tw:.2f}%** — exceeds 100%, weights rationalised proportionally."
                    )
                elif cash_pct > 0:
                    st.markdown(
                        f"**Active target:** {len(tw)} tickers · "
                        f"🟡 Sum = **{total_tw:.2f}%** · "
                        f"💵 **{cash_pct:.2f}% held as cash** (not rebalanced)"
                    )
                else:
                    st.markdown(
                        f"**Active target:** {len(tw)} tickers · 🟢 Sum = **100.00%**"
                    )

    if not is_scale_mode and not st.session_state["rb_target_weights"]:
        # Allow bypass if a saved plan was loaded from disk
        if not st.session_state.get("rebalance_plan"):
            st.stop()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — REBALANCE PARAMETERS
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("⚙️ Step 3 — Parameters", expanded=True):
        pc1, pc2 = st.columns(2)

        rebal_pct = pc1.number_input(
            "% of delta to execute today",
            min_value=1.0, max_value=100.0, step=5.0,
            value=float(st.session_state["rb_rebal_pct"]),
            format="%.0f",
            help="10% means only 10% of each delta is traded today. "
                 "Lets you phase in a rebalance gradually.",
            key="rb_rebal_pct_input",
        )
        st.session_state["rb_rebal_pct"] = rebal_pct
        pc1.caption(f"Only **{rebal_pct:.0f}%** of each BUY/SELL delta will be executed.")

        if is_scale_mode:
            capital_delta = pc2.number_input(
                "Capital change (₹)",
                min_value=-float(curr_df["curr_value"].sum()) if not curr_df.empty else -1e9,
                step=10_000.0, format="%.0f",
                value=0.0, key="rb_capital_delta",
                help=(
                    "Positive = fresh capital to deploy (buy more of each holding proportionally).\n\n"
                    "Negative = capital to withdraw (sell proportionally to maintain weights)."
                ),
            )
            fresh_capital = max(capital_delta, 0.0)   # used in total_val below
            total_curr_val = curr_df["curr_value"].sum() if not curr_df.empty else 0.0
            if capital_delta > 0:
                pc2.caption(f"➕ Deploying **₹{capital_delta:,.0f}** across all holdings proportionally.")
            elif capital_delta < 0:
                pct_exit = abs(capital_delta) / total_curr_val * 100 if total_curr_val else 0
                pc2.caption(f"➖ Withdrawing **₹{abs(capital_delta):,.0f}** ({pct_exit:.1f}% of portfolio).")
            else:
                pc2.caption("No capital change — enter a positive or negative amount.")
        else:
            capital_delta = 0.0
            fresh_capital = pc2.number_input(
                "Fresh Capital to deploy (₹)",
                min_value=0.0, step=10_000.0, format="%.0f",
                value=0.0, key="rb_fresh_capital",
                help="Added to total portfolio value when computing target quantities.",
            )

    # ══════════════════════════════════════════════════════════════════════════
    # BUILD DIFF TABLE — skip if no target weights (show saved plan only)
    # ══════════════════════════════════════════════════════════════════════════
    if not st.session_state["rb_target_weights"]:
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
        st.info("Set target weights in Step 2 above to build a new rebalance plan.")
        return

    tw        = st.session_state["rb_target_weights"]
    # In scale mode with a capital withdrawal, total_val shrinks
    total_val = curr_df["curr_value"].sum() + (capital_delta if is_scale_mode else fresh_capital)

    # > 100: rationalise (scale down proportionally)
    # < 100: keep as-is; remainder treated as cash (not allocated to any ticker)
    tw_sum = sum(tw.values())
    if tw_sum > 100:
        tw_norm = {k: v / tw_sum * 100 for k, v in tw.items()}
    else:
        tw_norm = dict(tw)   # use weights exactly as supplied; cash = 100 - tw_sum

    # Merge current + target
    all_tickers = sorted(set(curr_df["ticker"].tolist()) | set(tw_norm.keys()))
    curr_map    = curr_df.set_index("ticker").to_dict("index")

    diff_rows: list[dict] = []
    for tk in all_tickers:
        c       = curr_map.get(tk, {})
        curr_w  = float(curr_df.loc[curr_df["ticker"] == tk, "curr_wt%"].values[0]) \
            if tk in curr_map else 0.0
        tgt_w   = tw_norm.get(tk, 0.0)
        price   = float(c.get("last_price", 0))
        if price == 0:
            price = _fetch_price(tk)
        curr_qty   = int(c.get("qty", 0))
        tgt_qty    = int((total_val * tgt_w / 100) / price) if price > 0 else 0
        delta_qty  = tgt_qty - curr_qty
        today_qty  = int(delta_qty * rebal_pct / 100)   # scaled by rebalance %
        today_val  = round(today_qty * price, 2)

        if delta_qty > 0:
            action = "BUY"
        elif delta_qty < 0:
            action = "SELL"
        else:
            action = "HOLD"

        diff_rows.append({
            "Ticker":      tk,
            "Curr Wt%":    round(curr_w, 2),
            "Target Wt%":  round(tgt_w, 2),
            "Δ Wt%":       round(tgt_w - curr_w, 2),
            "Curr Qty":    curr_qty,
            "Target Qty":  tgt_qty,
            "Δ Qty":       delta_qty,
            "Today Qty":   today_qty,
            "Price ₹":     round(price, 2),
            "Today ₹":     today_val,
            "Action":      action,
            "_ignore":     tk in st.session_state["rb_ignored"],
        })

    diff_df = pd.DataFrame(diff_rows)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — DIFF VIEW + IGNORE
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("📊 Step 4 — Portfolio Diff", expanded=True):
        st.caption("Review the difference between your current and target portfolio. "
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

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — FINAL REBALANCE PLAN
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("🚀 Step 5 — Rebalance Plan", expanded=True):

        MIN_TRADE = 1_000
        active = edit_diff[
            (~edit_diff["Ignore"]) &
            (edit_diff["Action"].isin(["BUY", "SELL"])) &
            (edit_diff["Today ₹"].abs() >= MIN_TRADE)
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

            styled = active[["Ticker", "Action", "Today Qty", "Price ₹",
                             "Today ₹", "Curr Wt%", "Target Wt%", "Δ Wt%"]].style \
                .apply(_row_style, axis=1) \
                .format({
                "Price ₹":    "₹{:.2f}",
                "Today ₹":    "₹{:,.0f}",
                "Curr Wt%":   "{:.2f}%",
                "Target Wt%": "{:.2f}%",
                "Δ Wt%":      "{:+.2f}%",
            })
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
                cap_lbl = f"₹{capital_delta:+,.0f}" if capital_delta != 0 else "₹0"
                f5.metric("Capital Change", cap_lbl)
            else:
                f5.metric("Rebalance %", f"{rebal_pct:.0f}%")

            if is_scale_mode:
                if capital_delta > 0:
                    st.caption(
                        f"➕ Deploying **₹{capital_delta:,.0f}** across holdings at current weights. "
                        f"Today's execution is **{rebal_pct:.0f}%** of the full amount."
                    )
                elif capital_delta < 0:
                    st.caption(
                        f"➖ Withdrawing **₹{abs(capital_delta):,.0f}** by selling proportionally. "
                        f"Today's execution is **{rebal_pct:.0f}%** of the full amount."
                    )
                else:
                    st.caption("No capital change specified — enter an amount in Step 3.")
            else:
                st.caption(
                    f"⚠️ Today's plan is **{rebal_pct:.0f}%** of the full delta. "
                    f"Run again to continue phasing in the remaining "
                    f"**{100 - rebal_pct:.0f}%**."
                )

            st.divider()
            approve_col, _ = st.columns([1, 3])
            if approve_col.button("✅ Approve & Execute",
                                  type="primary", use_container_width=True):
                plan = []
                for _, row in active.iterrows():
                    plan.append({
                        "ticker":           row["Ticker"],
                        "action":           row["Action"],
                        "today_qty":        int(row["Today Qty"]),
                        "price":            float(row["Price ₹"]),
                        "today_value":      float(row["Today ₹"]),
                        "curr_wt":          float(row["Curr Wt%"]),
                        "target_wt":        float(row["Target Wt%"]),
                        "delta_wt":         float(row["Δ Wt%"]),
                        "side":             row["Action"],   # alias for order engine
                        "rebalance_mode":   "scale_capital" if is_scale_mode else "new_weights",
                    })
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