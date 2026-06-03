import streamlit as st
import pandas as pd
import requests
import json
import os
import random
import time
from datetime import datetime, timezone, timedelta

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

st.set_page_config(page_title="Kite Portfolio", layout="wide", page_icon="📈")

KITE_BASE_URL   = "https://api.kite.trade"
EXCLUSIONS_FILE = "excluded_symbols.json"
WEIGHTS_FILE    = "target_weights.txt"
LIVE_ORDERS_FILE = "live_orders.json"
MOCK_MODE       = True

# ─────────────────────────────────────────────────────────────────────────────
# Order state machine states
# ─────────────────────────────────────────────────────────────────────────────
# PENDING      → waiting for conditions to be met
# COOLING_OFF  → previous tier just filled; cooling off timer running
# PLACED       → sent to exchange (has order_id)
# FILLED       → confirmed filled via /trades reconciliation
# SKIPPED      → condition not met this cycle
# ERROR        → placement failed

# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_exclusions() -> set:
    if os.path.exists(EXCLUSIONS_FILE):
        try:
            with open(EXCLUSIONS_FILE) as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()

def save_exclusions(excl: set):
    with open(EXCLUSIONS_FILE, "w") as f:
        json.dump(sorted(excl), f, indent=2)

def load_weights() -> str:
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE) as f:
                return f.read()
        except Exception:
            pass
    return ""

def save_weights(text: str):
    with open(WEIGHTS_FILE, "w") as f:
        f.write(text)

def save_live_orders(orders: list):
    with open(LIVE_ORDERS_FILE, "w") as f:
        json.dump(orders, f, indent=2, default=str)

def load_live_orders() -> list:
    if os.path.exists(LIVE_ORDERS_FILE):
        try:
            with open(LIVE_ORDERS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []

# ─────────────────────────────────────────────────────────────────────────────
# Kite API helpers
# ─────────────────────────────────────────────────────────────────────────────

def kite_headers(token: str) -> dict:
    return {"X-Kite-Version": "3", "Authorization": f"token {token}"}

def fetch_holdings(token: str) -> list:
    r = requests.get(f"{KITE_BASE_URL}/portfolio/holdings",
                     headers=kite_headers(token), timeout=10)
    r.raise_for_status()
    resp = r.json()
    if resp.get("status") != "success":
        raise ValueError(resp.get("message", "API error"))
    return resp.get("data", [])

def place_order(token: str, symbol: str, exchange: str,
                side: str, qty: int, price: float,
                mock: bool = True) -> dict:
    if mock:
        fake_id = f"{int(time.time())}{random.randint(100000, 999999)}"
        return {"status": "success", "data": {"order_id": fake_id}}
    r = requests.post(
        f"{KITE_BASE_URL}/orders/regular",
        headers=kite_headers(token),
        data={"tradingsymbol": symbol, "exchange": exchange,
              "transaction_type": side, "order_type": "LIMIT",
              "quantity": qty, "price": price, "product": "CNC", "validity": "DAY"},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()

def cancel_order(token: str, order_id: str, mock: bool = False) -> dict:
    if mock:
        return {"status": "success", "data": {"order_id": order_id}}
    r = requests.delete(
        f"{KITE_BASE_URL}/orders/regular/{order_id}",
        headers=kite_headers(token),
        timeout=10,
    )
    r.raise_for_status()
    return r.json()

def fetch_trades(token: str, mock: bool = False) -> list:
    """Fetch all trades of the day for reconciliation."""
    if mock:
        # Return mock filled trades matching any PLACED orders
        filled = []
        for o in st.session_state.get("live_orders", []):
            if o.get("state") == "PLACED" and o.get("order_id"):
                filled.append({
                    "order_id":        o["order_id"],
                    "tradingsymbol":   o["Symbol"],
                    "transaction_type": o["Side"],
                    "quantity":        o["Qty"],
                    "average_price":   o["Price"],
                    "fill_timestamp":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                })
        return filled
    r = requests.get(f"{KITE_BASE_URL}/trades",
                     headers=kite_headers(token), timeout=10)
    r.raise_for_status()
    resp = r.json()
    if resp.get("status") != "success":
        raise ValueError(resp.get("message", "API error"))
    return resp.get("data", [])

# ─────────────────────────────────────────────────────────────────────────────
# Transaction fee calculation
# ─────────────────────────────────────────────────────────────────────────────

STT_RATE  = 0.001
ETC_RATE  = 0.0000325
SEBI_RATE = 0.000001
GST_RATE  = 0.18
DP_BASE   = 15.34

def calc_fees(buy_val: float, sell_val: float, sell_scrips: int) -> dict:
    turnover = buy_val + sell_val
    stt  = turnover * STT_RATE
    etc  = turnover * ETC_RATE
    sebi = turnover * SEBI_RATE
    gst  = (etc + sebi) * GST_RATE
    dp   = sell_scrips * DP_BASE * (1 + GST_RATE)
    return {
        "STT (0.1%)":        round(stt,  2),
        "Exch. (0.00325%)":  round(etc,  2),
        "SEBI (0.0001%)":    round(sebi, 2),
        "GST (18%)":         round(gst,  2),
        f"DP ×{sell_scrips}": round(dp,  2),
        "Total Fees":        round(stt + etc + sebi + gst + dp, 2),
    }

# ─────────────────────────────────────────────────────────────────────────────
# YFinance helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def get_fluctuation_pct(symbol: str, exchange: str = "NSE") -> float:
    if not HAS_YF:
        return 1.0
    suffix = ".NS" if exchange == "NSE" else ".BO"
    try:
        hist = yf.Ticker(f"{symbol}{suffix}").history(period="1d", interval="5m")
        if hist.empty:
            return 1.0
        return float((hist["High"].max() - hist["Low"].min()) / hist["Low"].min() * 100)
    except Exception:
        return 1.0

@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_yf(symbol: str, exchange: str = "NSE") -> float:
    if not HAS_YF:
        return 0.0
    suffix = ".NS" if exchange == "NSE" else ".BO"
    try:
        ticker = yf.Ticker(f"{symbol}{suffix}")
        info = ticker.fast_info
        price = float(info.get("last_price") or info.get("regularMarketPrice") or 0)
        if price > 0:
            return price
        hist = ticker.history(period="2d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Order generation helpers
# ─────────────────────────────────────────────────────────────────────────────

def geo_buy_orders(symbol: str, exchange: str, ltp: float,
                   delta_qty: int, fluc_pct: float) -> list[dict]:
    x = max(1, delta_qty // 15)
    tiers = [(x, 0), (2*x, fluc_pct/4), (4*x, fluc_pct/2), (8*x, fluc_pct*3/4)]
    orders = []
    for tier_idx, (qty, extra_pct) in enumerate(tiers):
        price = round(ltp * (1 - 0.0025 - extra_pct / 100), 2)
        orders.append({
            "Symbol": symbol, "Exchange": exchange, "Side": "BUY",
            "Qty": qty, "Price": price, "Value ₹": round(qty * price, 2),
            "Fluc%": round(fluc_pct, 3), "Tier": tier_idx + 1,
            "Min Hours": 1.5, "Last Filled": "",
            "Account": "", "Status": "",
        })
    return orders

def geo_sell_orders(symbol: str, exchange: str, ltp: float,
                    delta_qty: int, fluc_pct: float) -> list[dict]:
    x = max(1, abs(delta_qty) // 15)
    tiers = [(x, 0), (2*x, fluc_pct/4), (4*x, fluc_pct/2), (8*x, fluc_pct*3/4)]
    orders = []
    for tier_idx, (qty, extra_pct) in enumerate(tiers):
        price = round(ltp * (1 + 0.0025 + extra_pct / 100), 2)
        orders.append({
            "Symbol": symbol, "Exchange": exchange, "Side": "SELL",
            "Qty": qty, "Price": price, "Value ₹": round(qty * price, 2),
            "Fluc%": round(fluc_pct, 3), "Tier": tier_idx + 1,
            "Min Hours": 1.5, "Last Filled": "",
            "Account": "", "Status": "",
        })
    return orders

def fixed_bucket_sell_orders(symbol: str, exchange: str, ltp: float,
                             delta_qty: int, fluc_pct: float) -> list[dict]:
    buckets = [10_000, 20_000, 40_000, 80_000]
    extras  = [0, fluc_pct/4, fluc_pct/2, fluc_pct*3/4]
    orders, remaining = [], abs(delta_qty)
    for tier_idx, (inr_bucket, extra_pct) in enumerate(zip(buckets, extras)):
        if remaining <= 0:
            break
        price = round(ltp * (1 + 0.0025 + extra_pct / 100), 2)
        qty   = min(remaining, max(1, int(inr_bucket / price)))
        if qty > 0:
            orders.append({
                "Symbol": symbol, "Exchange": exchange, "Side": "SELL",
                "Qty": qty, "Price": price, "Value ₹": round(qty * price, 2),
                "Fluc%": round(fluc_pct, 3), "Tier": tier_idx + 1,
                "Min Hours": 1.5, "Last Filled": "",
                "Account": "", "Status": "",
            })
            remaining -= qty
    return orders

# ─────────────────────────────────────────────────────────────────────────────
# State machine helpers
# ─────────────────────────────────────────────────────────────────────────────

def now_ist_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def cooling_off_remaining(order: dict) -> float:
    """Minutes remaining in cooling-off, anchored to the previous tier's placement time."""
    anchor    = order.get("placed_at", "")   # set to prev-tier placed_at on cascade
    min_hours = float(order.get("Min Hours", 1.5) or 1.5)
    if not anchor:
        return min_hours * 60  # no anchor yet — show full window
    try:
        dt          = datetime.strptime(anchor, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        elapsed_min = (datetime.now(timezone.utc) - dt).total_seconds() / 60
        remaining   = min_hours * 60 - elapsed_min
        return max(0.0, remaining)
    except Exception:
        return 0.0

def state_label(order: dict) -> str:
    s = order.get("state", "PENDING")
    if s == "COOLING_OFF":
        mins = cooling_off_remaining(order)
        return f"⏱ Cooling off — {mins:.0f}m left"
    labels = {
        "PENDING":   "⏳ Pending",
        "PLACED":    f"📤 Placed ({order.get('order_id','')})",
        "FILLED":    f"✅ Filled ({order.get('order_id','')})",
        "SKIPPED":   f"⏭ {order.get('skip_reason','')}",
        "ERROR":     f"❌ {order.get('error','')}",
        "CANCELLED": f"🚫 Cancelled ({order.get('order_id','')})",
    }
    return labels.get(s, s)

def evaluate_and_place(orders: list, token_map: dict, ltp_map: dict,
                       mock: bool, max_sell_qty_map: dict | None = None) -> list:
    """One poll cycle: evaluate all PENDING/COOLING_OFF orders, place what qualifies.
    max_sell_qty_map: {symbol: max_qty} — caps SELL qty to account holdings."""
    now = datetime.now(timezone.utc)
    updated = []
    placed_ids = {o.get("order_id") for o in orders if o.get("state") == "PLACED"}

    # Reconcile fills (best-effort; errors don't abort the loop)
    filled_ids: set = set()
    if not mock:
        try:
            all_tokens = list(token_map.values())
            if all_tokens:
                trades = fetch_trades(all_tokens[0], mock=False)
                filled_ids = {t["order_id"] for t in trades}
        except Exception:
            pass
    else:
        # In mock, treat all PLACED orders as filled after one cycle
        filled_ids = placed_ids

    for o in orders:
        o = dict(o)  # shallow copy
        state    = o.get("state", "PENDING")
        sym      = o["Symbol"]
        tier     = o.get("Tier", "")
        is_tier1 = str(tier).strip() in ("1", "1.0")

        # ── SKIPPED → reset to PENDING each cycle so Poll Now re-evaluates ────
        if state == "SKIPPED":
            o["state"]       = "PENDING"
            o["skip_reason"] = ""
            state            = "PENDING"

        # ── Already placed → check if filled ─────────────────────────────────
        if state == "PLACED":
            if o.get("order_id") in filled_ids:
                o["state"]     = "FILLED"
                o["filled_at"] = now_ist_str()
            updated.append(o)
            continue

        # ── Terminal states → pass through ───────────────────────────────────
        if state in ("FILLED", "CANCELLED", "ERROR"):
            updated.append(o)
            continue

        # ── Cooling off → check if timer expired ─────────────────────────────
        if state == "COOLING_OFF":
            if cooling_off_remaining(o) <= 0:
                o["state"] = "PENDING"
            updated.append(o)
            continue

        # ── PENDING → evaluate conditions ─────────────────────────────────────
        acc_name = o.get("Account", "")
        token    = token_map.get(acc_name, "")
        price    = float(o.get("Price", 0))

        # T1 is unconditional — skip all guards and place immediately
        if not is_tier1:
            if not token or price <= 0:
                o["state"] = "PENDING"  # stay pending, can't act
                updated.append(o)
                continue
            # Price condition
            ltp = ltp_map.get(sym, 0)
            if ltp == 0 and HAS_YF:
                ltp = fetch_price_yf(sym, o.get("Exchange", "NSE"))
            if ltp > 0:
                if o["Side"] == "BUY"  and ltp > price:
                    o["state"]       = "SKIPPED"
                    o["skip_reason"] = f"LTP ₹{ltp:.2f} > grid ₹{price:.2f}"
                    updated.append(o)
                    continue
                if o["Side"] == "SELL" and ltp < price:
                    o["state"]       = "SKIPPED"
                    o["skip_reason"] = f"LTP ₹{ltp:.2f} < grid ₹{price:.2f}"
                    updated.append(o)
                    continue

            # Time / cooling-off condition (driven by Last Filled of *this* order's prev tier)
            min_hours   = float(o.get("Min Hours", 0) or 0)
            last_filled = str(o.get("Last Filled", "") or "").strip()
            if min_hours > 0 and last_filled:
                try:
                    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
                        try:
                            lf_dt = datetime.strptime(last_filled, fmt).replace(tzinfo=timezone.utc)
                            break
                        except ValueError:
                            continue
                    else:
                        raise ValueError("bad date")
                    hrs_elapsed = (now - lf_dt).total_seconds() / 3600
                    if hrs_elapsed < min_hours:
                        o["state"]     = "COOLING_OFF"
                        o["placed_at"] = last_filled
                        updated.append(o)
                        continue
                except Exception:
                    pass

        # ── Place it — T1 still needs account + price ────────────────────────
        if not token or price <= 0:
            o["state"] = "PENDING"
            updated.append(o)
            continue
        try:
            qty_to_place = int(o["Qty"])
            # Cap SELL qty to available holdings for selected account
            if o["Side"] == "SELL" and max_sell_qty_map:
                cap = max_sell_qty_map.get(o["Symbol"], qty_to_place)
                if cap <= 0:
                    o["state"]       = "SKIPPED"
                    o["skip_reason"] = "No holdings in selected account"
                    updated.append(o)
                    continue
                qty_to_place = min(qty_to_place, cap)
            resp = place_order(token, sym, o["Exchange"],
                               o["Side"], qty_to_place, price, mock=mock)
            oid = resp.get("data", {}).get("order_id", "?")
            o["state"]       = "PLACED"
            o["order_id"]    = oid
            o["Qty"]         = qty_to_place   # record actual placed qty
            o["placed_at"]   = now_ist_str()
            o["Last Filled"] = now_ist_str()
        except Exception as e:
            o["state"] = "ERROR"
            o["error"] = str(e)

        updated.append(o)

    # ── Cascade: when tier N is PLACED/FILLED, put tier N+1 into COOLING_OFF ──
    # Build map: (Symbol, Side, tier_num) → placed_at  for all placed/filled tiers
    placed_at_map: dict = {}   # (sym, side, tier) → placed_at timestamp
    for o in updated:
        if o.get("state") in ("PLACED", "FILLED"):
            key = (o["Symbol"], o["Side"], int(o.get("Tier", 0) or 0))
            placed_at_map[key] = o.get("placed_at", now_ist_str())

    for o in updated:
        sym  = o["Symbol"]
        side = o["Side"]
        tier = int(o.get("Tier", 0) or 0)
        prev_key       = (sym, side, tier - 1)
        prev_placed_at = placed_at_map.get(prev_key)

        if prev_placed_at and o.get("state") == "PENDING":
            # Previous tier was placed this cycle → start this tier's cool-off
            o["state"]     = "COOLING_OFF"
            o["placed_at"] = prev_placed_at   # anchor = when T(N-1) was placed

        elif o.get("state") == "COOLING_OFF":
            # Refresh placed_at from map in case it was updated this cycle
            if prev_placed_at:
                o["placed_at"] = prev_placed_at

    return updated

# ─────────────────────────────────────────────────────────────────────────────
# Misc helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_weights(text: str) -> dict:
    weights = {}
    for line in text.strip().splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 2:
            ticker = parts[0].strip().upper()
            try:
                weights[ticker] = float(parts[1].strip())
            except ValueError:
                pass
    return weights

ACTION_COLORS = {
    "BUY":      "background-color:#dcfce7;color:#15803d",
    "SELL":     "background-color:#fee2e2;color:#b91c1c",
    "ADD":      "background-color:#dbeafe;color:#1d4ed8",
    "ADD(BUY)": "background-color:#dbeafe;color:#1d4ed8",
    "IGNORE":   "background-color:#f3f4f6;color:#9ca3af",
}

def style_action(val):
    return ACTION_COLORS.get(val, "")

# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────

def ss_default(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

ss_default("accounts",          [{"name": "Account 1", "token": ""}])
ss_default("all_holdings",      {})
ss_default("excluded_symbols",  load_exclusions())
ss_default("action_overrides",  {})
ss_default("staged_orders",     [])
ss_default("live_orders",       [])   # state-machine order list
ss_default("runner_active",     False)
ss_default("last_poll_time",    None)
ss_default("weights_text",      load_weights())
ss_default("buy_strategy_used", "Geometric Progression")
ss_default("sell_strategy_used","Geometric Progression (GP)")

# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────

st.title("📈 Kite Portfolio")
left, main = st.columns([1, 3], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL
# ══════════════════════════════════════════════════════════════════════════════

with left:

    # ── Accounts ──────────────────────────────────────────────────────────────
    with st.expander("🔑 Accounts", expanded=True):
        accounts  = st.session_state.accounts
        to_remove = None
        for i, acc in enumerate(accounts):
            acc["name"]  = st.text_input("Name",  value=acc["name"],
                                         key=f"acc_name_{i}",  placeholder=f"Account {i+1}")
            acc["token"] = st.text_input("Token", value=acc["token"],
                                         key=f"acc_token_{i}", type="password",
                                         placeholder="api_key:access_token")
            if len(accounts) > 1 and st.button("🗑 Remove", key=f"acc_del_{i}",
                                               use_container_width=True):
                to_remove = i
            if i < len(accounts) - 1:
                st.divider()
        if to_remove is not None:
            accounts.pop(to_remove)
            st.rerun()

        c1, c2 = st.columns(2)
        if c1.button("➕ Add Account", use_container_width=True):
            accounts.append({"name": f"Account {len(accounts)+1}", "token": ""})
            st.rerun()
        if c2.button("🔄 Fetch Holdings", type="primary", use_container_width=True):
            fetched, errs = {}, []
            for acc in accounts:
                if not acc["token"]:
                    continue
                try:
                    with st.spinner(f"Fetching {acc['name']}…"):
                        fetched[acc["name"]] = fetch_holdings(acc["token"])
                except Exception as e:
                    errs.append(f"**{acc['name']}:** {e}")
            st.session_state.all_holdings = fetched
            for e in errs:
                st.error(e)
            if fetched and not errs:
                st.success(f"✓ {sum(len(v) for v in fetched.values())} holdings loaded")

    # ── Fresh Capital ──────────────────────────────────────────────────────────
    st.subheader("💰 Fresh Capital")
    fresh_capital = st.number_input(
        "Fresh Capital (₹)", min_value=0.0, value=0.0,
        step=10_000.0, format="%.0f", label_visibility="collapsed",
        help="Added to existing value for target quantity calculations",
    )

    # ── Strategy ──────────────────────────────────────────────────────────────
    st.subheader("⚙️ Strategy")
    buy_strategy = st.selectbox(
        "BUY strategy", ["Geometric Progression"],
        label_visibility="visible",
        help="Strategy used to split buy quantities across price tiers.",
    )
    sell_strategy = st.selectbox(
        "SELL strategy", ["Geometric Progression (GP)", "Fixed ₹ Buckets"],
        label_visibility="visible",
        help="GP: 1:2:4:8 qty split mirroring buy. Fixed ₹ Buckets: ₹10K/₹20K/₹40K/₹80K per tier (max ₹1.5L/run).",
    )

    # ── Ignore List ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🚫 Ignore List")
    excl: set = st.session_state.excluded_symbols
    all_known = sorted({h["tradingsymbol"]
                        for hlds in st.session_state.all_holdings.values()
                        for h in hlds})
    if excl:
        chip_cols = st.columns(2)
        for idx, sym in enumerate(sorted(excl)):
            if chip_cols[idx % 2].button(f"✕ {sym}", key=f"rm_excl_{sym}",
                                         use_container_width=True):
                excl.discard(sym)
                save_exclusions(excl)
                st.rerun()
    else:
        st.caption("_No symbols ignored._")

    with st.form("add_excl_form", clear_on_submit=True):
        opts = [""] + [s for s in all_known if s not in excl]
        pick  = st.selectbox("Pick from holdings", opts,
                             label_visibility="visible") if all_known else None
        typed = st.text_input("Or type symbol", placeholder="SGBMAR29")
        if st.form_submit_button("➕ Add to Ignore", use_container_width=True):
            to_add = typed.strip().upper() or (pick.strip().upper() if pick else "")
            if to_add:
                excl.add(to_add)
                save_exclusions(excl)
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

with main:

    all_holdings = st.session_state.all_holdings
    overrides    = st.session_state.action_overrides
    excl_set     = st.session_state.excluded_symbols

    # ── Target Weights — always visible at screen open; loads saved weights ──────────────────────
    with st.expander("🎯 Target Weights", expanded=True):
        tw_area, tw_btn = st.columns([5, 1])
        weights_text = tw_area.text_area(
            "weights", label_visibility="collapsed",
            placeholder="RELIANCE\t10\nINFY\t8.5\nHDFCBANK\t12",
            height=80, key="weights_text",
        )
        tw_btn.write("")  # vertical spacer to align button
        if tw_btn.button("💾 Save Weights", use_container_width=True):
            save_weights(weights_text)
            st.toast("✅ Weights saved!", icon="💾")
        target_weights = parse_weights(weights_text)
        if target_weights:
            total_w = sum(target_weights.values())
            ok = abs(total_w - 100) < 0.01
            st.caption(
                f"{'🟢' if ok else '🔴'} **Sum {total_w:.2f}%**"
                + (f" (off by {total_w-100:+.2f}%)" if not ok else "  ✅  `TICKER TAB weight%` — one per line")
            )
        else:
            target_weights = {}
    st.write("")

    if not all_holdings:
        st.info("Configure accounts on the left and click **Fetch Holdings**.")
        st.stop()

    account_names = list(all_holdings.keys())

    # ── Build ticker_data map ─────────────────────────────────────────────────
    ticker_data: dict = {}
    for acc_name, holdings in all_holdings.items():
        for h in holdings:
            sym = h["tradingsymbol"]
            if sym in excl_set:
                continue
            if sym not in ticker_data:
                ticker_data[sym] = {
                    "exchange":       h.get("exchange", "NSE"),
                    "last_price":     h.get("last_price", 0),
                    "close_price":    h.get("close_price", 0),
                    "day_change_pct": h.get("day_change_percentage", 0),
                    **{an: 0 for an in account_names},
                }
            ticker_data[sym][acc_name] = h.get("quantity", 0)

    # ── Build base DataFrame ──────────────────────────────────────────────────
    rows = []
    for sym, data in ticker_data.items():
        row: dict = {"Symbol": sym}
        total_qty = 0
        for an in account_names:
            qty = data.get(an, 0)
            row[an] = qty
            total_qty += qty
        row["Total Qty"]   = total_qty
        row["Last Price"]  = data["last_price"]
        row["Curr Value"]  = round(total_qty * data["last_price"], 2)
        row["Close Price"] = data["close_price"]
        row["Day Chg%"]    = data["day_change_pct"]
        row["_exchange"]   = data["exchange"]
        rows.append(row)

    df = (pd.DataFrame(rows)
          .sort_values("Curr Value", ascending=False)
          .reset_index(drop=True))

    existing_value = df["Curr Value"].sum()
    total_value    = existing_value + fresh_capital

    df["Portfolio Wt%"] = (
        (df["Curr Value"] / total_value * 100).round(2) if total_value else 0.0
    )

    has_weights = bool(target_weights)

    # ── Overlay target weights ────────────────────────────────────────────────
    if has_weights:
        df["Target Wt%"] = df["Symbol"].map(target_weights).fillna(0.0)
        df["Target Qty"] = (
                (total_value * df["Target Wt%"] / 100)
                / df["Last Price"].replace(0, pd.NA)
        ).fillna(0).astype(int)
        df["Delta Qty"] = df["Target Qty"] - df["Total Qty"]
        df["Delta ₹"]   = (df["Delta Qty"] * df["Last Price"]).round(2)
        df["Action"]    = df["Delta Qty"].apply(
            lambda d: "BUY" if d > 0 else ("SELL" if d < 0 else "HOLD")
        )

        missing = {t: w for t, w in target_weights.items()
                   if t not in df["Symbol"].values}
        if missing:
            add_rows = []
            for sym, wt in missing.items():
                yf_price  = fetch_price_yf(sym, "NSE")
                target_qty = int((total_value * wt / 100) / yf_price) if yf_price > 0 else 0
                add_rows.append({
                    "Symbol": sym, "_exchange": "NSE",
                    **{an: 0 for an in account_names},
                    "Total Qty": 0, "Last Price": yf_price, "Curr Value": 0.0,
                    "Close Price": yf_price, "Day Chg%": 0.0, "Portfolio Wt%": 0.0,
                    "Target Wt%": wt, "Target Qty": target_qty,
                    "Delta Qty": target_qty, "Delta ₹": round(target_qty * yf_price, 2),
                    "Action": "ADD(BUY)" if yf_price > 0 else "ADD",
                })
            add_df = pd.DataFrame(add_rows)
            yf_filled  = add_df[add_df["Last Price"] > 0]["Symbol"].tolist()
            yf_missing = add_df[add_df["Last Price"] == 0]["Symbol"].tolist()
            if yf_filled:
                st.info(f"🟡 Yahoo Finance prices fetched for: {', '.join(yf_filled)}")
            if yf_missing:
                st.warning(f"⚠️ Could not fetch price from Yahoo Finance for: {', '.join(yf_missing)} — edit manually.")
            df = pd.concat([df, add_df], ignore_index=True)

        df = df[~((df["Portfolio Wt%"] == 0) & (df["Target Wt%"] == 0))].reset_index(drop=True)

    for sym, act in overrides.items():
        df.loc[df["Symbol"] == sym, "Action"] = act

    # ── Transaction fees banner ───────────────────────────────────────────────
    if has_weights and "Action" in df.columns:
        act_df   = df[df["Action"].isin(["BUY", "ADD", "SELL"])]
        buy_val  = act_df[act_df["Action"].isin(["BUY","ADD"])]["Delta ₹"].abs().sum()
        sell_val = act_df[act_df["Action"] == "SELL"]["Delta ₹"].abs().sum()
        n_sell   = int((act_df["Action"] == "SELL").sum())
        fees     = calc_fees(buy_val, sell_val, n_sell)
        fee_cols = st.columns(len(fees))
        for col, (label, val) in zip(fee_cols, fees.items()):
            col.metric(label, f"₹{val:,.2f}")
        st.divider()

    # ── Summary metrics ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Existing Value",  f"₹{existing_value:,.0f}")
    m2.metric("Total (incl. fresh)", f"₹{total_value:,.0f}",
              delta=f"+₹{fresh_capital:,.0f}" if fresh_capital else None)
    excl_note = f" ({len(excl_set)} ignored)" if excl_set else ""
    m3.metric("Holdings", f"{len(ticker_data)}{excl_note}")
    if has_weights and "Action" in df.columns:
        buy_inr  = df[df["Action"].isin(["BUY","ADD"])]["Delta ₹"].abs().sum()
        sell_inr = df[df["Action"] == "SELL"]["Delta ₹"].abs().sum()
        m4.metric("Buys / Sells (₹)", f"₹{buy_inr:,.0f}  /  ₹{sell_inr:,.0f}")
    else:
        m4.metric("Accounts", len(account_names))

    # ── Portfolio table ───────────────────────────────────────────────────────
    show_cols = (
            ["Symbol"] + account_names
            + ["Total Qty", "Last Price", "Curr Value", "Close Price", "Day Chg%", "Portfolio Wt%"]
            + (["Target Wt%", "Target Qty", "Delta Qty", "Delta ₹", "Action"] if has_weights else [])
    )
    df_disp = df[show_cols].copy()

    col_cfg: dict = {
        "Last Price":    st.column_config.NumberColumn("Last Price",  format="₹%.2f"),
        "Curr Value":    st.column_config.NumberColumn("Curr Value",  format="₹%.0f"),
        "Close Price":   st.column_config.NumberColumn("Close Price", format="₹%.2f"),
        "Day Chg%":      st.column_config.NumberColumn("Day Chg%",   format="%.2f%%"),
        "Portfolio Wt%": st.column_config.NumberColumn("Port Wt%",   format="%.2f%%"),
    }
    if has_weights:
        col_cfg.update({
            "Target Wt%": st.column_config.NumberColumn("Target Wt%", format="%.2f%%"),
            "Delta ₹":    st.column_config.NumberColumn("Delta ₹",    format="₹%.0f"),
            "Action":     st.column_config.SelectboxColumn(
                "Action",
                options=["BUY", "SELL", "ADD", "ADD(BUY)", "HOLD", "IGNORE"],
                required=True, width="small",
            ),
        })

    disabled_cols = [c for c in show_cols if c != "Action"] if has_weights else show_cols

    def _style_row(row):
        if "Action" not in row.index:
            return [""] * len(row)
        return [ACTION_COLORS.get(row["Action"], "")] * len(row)

    edited_df = st.data_editor(
        df_disp,
        column_config=col_cfg,
        disabled=disabled_cols,
        use_container_width=True,
        hide_index=True,
        height=min(640, 36 + 35 * max(len(df_disp), 1)),
        key="portfolio_editor",
    )

    if has_weights and "Action" in edited_df.columns:
        auto = {}
        for _, row in df.iterrows():
            sym = row["Symbol"]
            d   = row.get("Delta Qty", 0)
            if row.get("Action") == "ADD" and d == 0:
                auto[sym] = "ADD"
            else:
                auto[sym] = "BUY" if d > 0 else ("SELL" if d < 0 else "HOLD")
        new_overrides: dict = {}
        for _, row in edited_df.iterrows():
            sym = row["Symbol"]
            if row["Action"] != auto.get(sym, "HOLD"):
                new_overrides[sym] = row["Action"]
        st.session_state.action_overrides = new_overrides

    if has_weights:
        st.markdown(
            "🟢 **BUY/ADD(BUY)** increase &nbsp;|&nbsp; "
            "🔴 **SELL** reduce &nbsp;|&nbsp; "
            "🔵 **ADD** new position &nbsp;|&nbsp; "
            "⚫ **IGNORE** skip"
        )

    # ── Generate staggered orders ─────────────────────────────────────────────
    if has_weights:
        st.write("")
        if st.button("🚀 Generate Staggered Orders", type="primary", use_container_width=True):
            if not HAS_YF:
                st.warning("Install yfinance for fluctuation data. Using 1% fallback.")
            actionable = edited_df[
                edited_df["Action"].isin(["BUY", "SELL", "ADD", "ADD(BUY)"])
            ].copy()
            actionable = actionable.merge(
                df[["Symbol", "_exchange"]].drop_duplicates(), on="Symbol", how="left"
            )
            all_orders = []
            prog = st.progress(0, text="Fetching intraday data…")
            total_syms = len(actionable)
            for i, (_, row) in enumerate(actionable.iterrows()):
                sym      = row["Symbol"]
                exchange = row.get("_exchange", "NSE")
                ltp      = float(row["Last Price"])
                action   = row["Action"]
                delta    = int(row.get("Delta Qty", 0))
                prog.progress((i + 1) / max(total_syms, 1),
                              text=f"Fetching fluctuation for {sym}…")
                fluc = get_fluctuation_pct(sym, exchange)
                if action in ("BUY", "ADD", "ADD(BUY)"):
                    qty = max(delta, 1) if delta > 0 else 1
                    all_orders.extend(geo_buy_orders(sym, exchange, ltp, qty, fluc))
                else:
                    if sell_strategy == "Fixed ₹ Buckets":
                        all_orders.extend(fixed_bucket_sell_orders(sym, exchange, ltp, delta, fluc))
                    else:
                        all_orders.extend(geo_sell_orders(sym, exchange, ltp, delta, fluc))
            prog.empty()
            # Seed state machine fields
            for o in all_orders:
                o["state"]      = "PENDING"
                o["order_id"]   = ""
                o["placed_at"]  = ""
                o["filled_at"]  = ""
                o["skip_reason"] = ""
                o["error"]      = ""
            st.session_state.staged_orders     = all_orders
            st.session_state.live_orders       = []   # reset live tracker
            st.session_state.runner_active     = False
            st.session_state.buy_strategy_used  = buy_strategy
            st.session_state.sell_strategy_used = sell_strategy
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ORDER REVIEW  (full-width, outside main column)
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.staged_orders:
    st.divider()
    st.subheader("📋 Order Review")

    account_options  = [a["name"] for a in st.session_state.accounts if a["token"]]
    buy_strat_label  = st.session_state.get("buy_strategy_used",  "GP")
    sell_strat_label = st.session_state.get("sell_strategy_used", "GP")
    st.caption(f"BUY: **{buy_strat_label}** &nbsp;|&nbsp; SELL: **{sell_strat_label}**")
    orders_df = pd.DataFrame(st.session_state.staged_orders)

    for col, default in [("Tier",""), ("Min Hours", 1.5), ("Last Filled",""),
                         ("state","PENDING"), ("order_id",""), ("placed_at",""),
                         ("filled_at",""), ("skip_reason",""), ("error","")]:
        if col not in orders_df.columns:
            orders_df[col] = default

    # ── Controls row ──────────────────────────────────────────────────────────
    ctl1, ctl2, ctl3, ctl4 = st.columns([1, 1, 1, 3])

    if ctl1.button("🔄 Refresh Prices", use_container_width=True):
        price_map = {
            h["tradingsymbol"]: h.get("last_price", 0)
            for hlds in st.session_state.all_holdings.values()
            for h in hlds
        }
        updated = []
        for order in st.session_state.staged_orders:
            sym  = order["Symbol"]
            exch = order.get("Exchange", "NSE")
            ltp  = price_map.get(sym, 0)
            if ltp == 0 and HAS_YF:
                ltp = fetch_price_yf(sym, exch)
            if ltp > 0:
                get_fluctuation_pct.clear()
                fluc      = get_fluctuation_pct(sym, exch)
                side      = order["Side"]
                qty       = order["Qty"]
                extra_pct = order.get("Fluc%", 1.0)
                if side == "BUY":
                    new_price = round(ltp * (1 - 0.0025 - extra_pct / 100), 2)
                else:
                    new_price = round(ltp * (1 + 0.0025 + extra_pct / 100), 2)
                order = {**order, "Price": new_price, "Value ₹": round(qty * new_price, 2)}
            updated.append(order)
        st.session_state.staged_orders = updated
        st.rerun()

    if ctl2.button("🗑 Clear Orders", use_container_width=True):
        st.session_state.staged_orders = []
        st.session_state.live_orders   = []
        st.rerun()

    mock_mode = ctl3.toggle(
        "🧪 Mock", value=MOCK_MODE,
        help="When ON, no real orders sent to Kite — fake order IDs returned.",
        key="mock_mode_toggle",
    )
    if mock_mode:
        ctl3.caption("Mock ON")

    # ── Bulk account selector ─────────────────────────────────────────────────
    st.write("")
    bulk_col1, bulk_col2 = st.columns([3, 1])
    bulk_account = bulk_col1.selectbox(
        "Fill account for all orders",
        options=[""] + account_options,
        format_func=lambda x: "— select account to fill all —" if x == "" else x,
        key="bulk_account_select",
        label_visibility="collapsed",
    )

    if bulk_col2.button("Apply to All", use_container_width=True, disabled=not bulk_account):
        # Build max sell qty map from selected account's holdings
        acc_holdings = st.session_state.all_holdings.get(bulk_account, [])
        holding_qty_map = {h["tradingsymbol"]: h.get("quantity", 0) for h in acc_holdings}

        updated = []
        for order in st.session_state.staged_orders:
            o = dict(order)
            o["Account"] = o.get("Account") or bulk_account
            # If this is a SELL order and account is now set, cap qty to holdings
            if o["Side"] == "SELL" and o["Account"] == bulk_account:
                available = holding_qty_map.get(o["Symbol"], 0)
                if available <= 0:
                    o["_sell_cap_warn"] = True
                else:
                    original_qty = o.get("_orig_qty", o["Qty"])
                    o["_orig_qty"] = original_qty  # preserve original
                    o["Qty"] = min(original_qty, available)
                    o["Value ₹"] = round(o["Qty"] * o["Price"], 2)
            updated.append(o)
        st.session_state.staged_orders = updated

        # Warn about capped sells
        capped = [o["Symbol"] for o in updated
                  if o.get("Side") == "SELL" and o.get("Account") == bulk_account
                  and o.get("Qty", 0) < o.get("_orig_qty", o.get("Qty", 0))]
        zero_hold = [o["Symbol"] for o in updated if o.get("_sell_cap_warn")]
        if capped:
            st.warning(f"⚠️ SELL qty capped to account holdings for: {', '.join(set(capped))}")
        if zero_hold:
            st.warning(f"⚠️ No holdings found in **{bulk_account}** for: {', '.join(set(zero_hold))} — SELL orders kept but may be rejected.")
        st.rerun()

    # ── Manual price warning ──────────────────────────────────────────────────
    add_syms = orders_df[orders_df["Price"] <= 0]["Symbol"].unique().tolist()
    if add_syms:
        st.warning(f"**Manual price needed** for: {', '.join(add_syms)} — edit the Price column below.")

    # ── Editable order table ──────────────────────────────────────────────────
    order_col_cfg = {
        "Tier":       st.column_config.NumberColumn("Tier",      format="%d",      width="small"),
        "Side":       st.column_config.SelectboxColumn("Side",   options=["BUY","SELL"], width="small"),
        "Price":      st.column_config.NumberColumn("Price ₹",   format="₹%.2f",   min_value=0.0),
        "Value ₹":    st.column_config.NumberColumn("Value ₹",   format="₹%.0f"),
        "Fluc%":      st.column_config.NumberColumn("Fluc%",     format="%.3f%%"),
        "Min Hours":  st.column_config.NumberColumn("Min Hours ⏱",
                                                    help="Min hours since previous tier filled before this order can be placed.",
                                                    min_value=0, step=1, format="%d hrs"),
        "Last Filled": st.column_config.TextColumn("Last Filled 🕐",
                                                   help="Timestamp of previous tier fill (YYYY-MM-DD HH:MM:SS). Blank = skip time check."),
        "Account":    st.column_config.SelectboxColumn("Account", options=account_options),
        "Status":     st.column_config.TextColumn("Status", disabled=True),
    }

    edited_orders = st.data_editor(
        orders_df[["Symbol","Exchange","Tier","Side","Qty","Price","Value ₹","Fluc%",
                   "Min Hours","Last Filled","Account","Status"]],
        column_config=order_col_cfg,
        disabled=["Symbol","Exchange","Tier","Qty","Fluc%","Status"],
        use_container_width=True,
        hide_index=True,
        key="orders_editor",
    )
    edited_orders["Value ₹"] = (edited_orders["Qty"] * edited_orders["Price"]).round(2)

    # Sync edits back to staged_orders
    for col in ["Price","Min Hours","Last Filled","Account","Side"]:
        if col in edited_orders.columns:
            orders_df[col] = edited_orders[col].values
    st.session_state.staged_orders = orders_df.to_dict("records")

    # ── Order summary ─────────────────────────────────────────────────────────
    buy_total  = edited_orders[edited_orders["Side"] == "BUY"]["Value ₹"].sum()
    sell_total = edited_orders[edited_orders["Side"] == "SELL"]["Value ₹"].sum()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Buy Orders",  len(edited_orders[edited_orders["Side"] == "BUY"]))
    s2.metric("Total Buy",   f"₹{buy_total:,.0f}")
    s3.metric("Total Sell",  f"₹{sell_total:,.0f}")
    s4.metric("Net Outflow", f"₹{buy_total - sell_total:,.0f}")

    # ── Condition legend ──────────────────────────────────────────────────────
    with st.expander("ℹ️ Order placement conditions", expanded=False):
        st.markdown(
            "**Tier 1** — placed unconditionally (always rolling in the exchange).\n\n"
            "**Tiers 2–4** — both conditions must be met:\n\n"
            "1. **Price** — LTP ≤ grid price (BUY) or LTP ≥ grid price (SELL).\n"
            "2. **Time** — `Min Hours` elapsed since `Last Filled`. Blank = skip time check.\n\n"
            "Set `Min Hours = 0` to disable the time gate for a row."
        )

    # ── Pre-execution summary ─────────────────────────────────────────────────
    st.write("")
    eligible = edited_orders[
        (edited_orders["Price"] > 0) &
        (edited_orders["Account"].astype(str).str.strip() != "")
        ]
    proj_buy  = eligible[eligible["Side"] == "BUY"]["Value ₹"].sum()
    proj_sell = eligible[eligible["Side"] == "SELL"]["Value ₹"].sum()
    proj_t1_buy  = eligible[(eligible["Side"]=="BUY")  & (eligible["Tier"].astype(str)=="1")]["Value ₹"].sum()
    proj_t1_sell = eligible[(eligible["Side"]=="SELL") & (eligible["Tier"].astype(str)=="1")]["Value ₹"].sum()
    n_buy  = len(eligible[eligible["Side"] == "BUY"])
    n_sell = len(eligible[eligible["Side"] == "SELL"])
    n_skip = len(edited_orders) - len(eligible)

    with st.container(border=True):
        st.markdown("#### 📊 Projected Execution")
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("🟢 Buy orders",  f"{n_buy} orders",  f"₹{proj_buy:,.0f} total")
        pc2.metric("🔴 Sell orders", f"{n_sell} orders", f"₹{proj_sell:,.0f} total")
        pc3.metric("⚡ Net outflow", f"₹{proj_buy - proj_sell:,.0f}", delta_color="inverse")
        st.caption(
            f"T1 unconditional — BUY ₹{proj_t1_buy:,.0f} &nbsp;|&nbsp; SELL ₹{proj_t1_sell:,.0f}"
            + (f"&nbsp; &nbsp; ⚠️ {n_skip} order(s) skipped (no account or price=0)" if n_skip else "")
        )

    # ── Place All Orders button ────────────────────────────────────────────────
    if st.button("⚡ Place All Orders", type="primary", use_container_width=True):
        _mock = st.session_state.get("mock_mode_toggle", MOCK_MODE)
        token_map = {a["name"]: a["token"] for a in st.session_state.accounts}
        ltp_map   = {
            h["tradingsymbol"]: h.get("last_price", 0)
            for hlds in st.session_state.all_holdings.values()
            for h in hlds
        }
        # Sell cap: max qty per symbol across all accounts
        max_sell_qty_map: dict = {}
        for hlds in st.session_state.all_holdings.values():
            for h in hlds:
                sym = h["tradingsymbol"]
                max_sell_qty_map[sym] = max_sell_qty_map.get(sym, 0) + h.get("quantity", 0)
        orders_with_state = st.session_state.staged_orders
        result = evaluate_and_place(orders_with_state, token_map, ltp_map,
                                    mock=_mock, max_sell_qty_map=max_sell_qty_map)
        st.session_state.live_orders   = result
        st.session_state.staged_orders = result
        st.session_state.last_poll_time = now_ist_str()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# LIVE ORDER MONITOR  (3-table view with auto-poll)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# LIVE ORDER MONITOR  (3-table view — manual poll)
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.live_orders:
    st.divider()
    _mock = st.session_state.get("mock_mode_toggle", MOCK_MODE)

    # ── Monitor controls ──────────────────────────────────────────────────────
    mc1, mc2, mc3, mc4, mc5 = st.columns([1, 1, 1, 1, 2])

    # Manual poll button
    if mc1.button("🔄 Poll Now", use_container_width=True,
                  help="Fetch live holdings, advance cool-offs, and re-evaluate all orders against live prices in 1-click"):
        token_map = {a["name"]: a["token"] for a in st.session_state.accounts}

        # 1. FETCH GENUINE LIVE PRICES (Simulating the "Fetch Holdings" button)
        fetched, errs = {}, []
        for acc in st.session_state.accounts:
            if not acc["token"]:
                continue
            try:
                # We fetch directly using the account token to guarantee true market snapshot prices
                fetched[acc["name"]] = fetch_holdings(acc["token"])
            except Exception as e:
                errs.append(f"**{acc['name']}:** {e}")

        # Update the session state holdings so the UI tables stay completely in sync
        if fetched:
            st.session_state.all_holdings = fetched
        for e in errs:
            st.error(e)

        # Generate the live Last Traded Price (LTP) map from fresh holdings data
        ltp_map = {
            h["tradingsymbol"]: h.get("last_price", 0)
            for hlds in st.session_state.all_holdings.values()
            for h in hlds
        }

        # Generate the max sell quantities from fresh holdings data
        max_sell_qty_map: dict = {}
        for hlds in st.session_state.all_holdings.values():
            for h in hlds:
                sym = h["tradingsymbol"]
                max_sell_qty_map[sym] = max_sell_qty_map.get(sym, 0) + h.get("quantity", 0)

        # 2. ADVANCE COOL-OFF TIMER IMMEDIATELY (Eliminating the "Double-Poll" requirement)
        # We pre-process the orders right here. If a timer is up, we promote it to PENDING
        # before sending it to evaluate_and_place. This removes the need for a second click!
        pre_processed_orders = []
        for o in st.session_state.live_orders:
            o_copy = dict(o)
            if o_copy.get("state") == "COOLING_OFF":
                if cooling_off_remaining(o_copy) <= 0:
                    o_copy["state"] = "PENDING"
            pre_processed_orders.append(o_copy)

        # 3. EVALUATE AND PLACE IN THE SAME CYCLE
        updated = evaluate_and_place(
            pre_processed_orders, token_map, ltp_map,
            mock=_mock, max_sell_qty_map=max_sell_qty_map
        )

        st.session_state.live_orders   = updated
        st.session_state.staged_orders = updated
        st.session_state.last_poll_time = now_ist_str()
        st.rerun()

    # Reconcile trades button (Tables 2 & 3)
    if mc2.button("🔁 Reconcile Trades", use_container_width=True,
                  help="Fetch today's /trades from Kite and update PLACED → FILLED"):
        token_map = {a["name"]: a["token"] for a in st.session_state.accounts}
        filled_ids: set = set()
        for acc in st.session_state.accounts:
            tok = acc.get("token","")
            if not tok:
                continue
            try:
                trades = fetch_trades(tok, mock=_mock)
                filled_ids.update(t["order_id"] for t in trades)
            except Exception:
                pass
        updated = []
        for o in st.session_state.live_orders:
            o = dict(o)
            if o.get("state") == "PLACED" and o.get("order_id") in filled_ids:
                o["state"]     = "FILLED"
                o["filled_at"] = now_ist_str()
            updated.append(o)
        st.session_state.live_orders   = updated
        st.session_state.staged_orders = updated
        st.rerun()

    # Save planned orders to disk
    if mc3.button("💾 Save Orders", use_container_width=True,
                  help="Persist live_orders.json — reload next session"):
        save_live_orders(st.session_state.live_orders)
        st.toast("✅ Orders saved to live_orders.json", icon="💾")

    # Load saved orders from disk
    if mc4.button("📂 Load Saved", use_container_width=True,
                  help="Restore from live_orders.json saved in a previous session"):
        saved = load_live_orders()
        if saved:
            st.session_state.live_orders   = saved
            st.session_state.staged_orders = saved
            st.toast(f"✅ Loaded {len(saved)} orders from disk", icon="📂")
            st.rerun()
        else:
            st.toast("No saved orders found.", icon="⚠️")

    last_poll = st.session_state.get("last_poll_time") or "—"
    mc5.caption(f"Last poll: **{last_poll}**  |  Poll manually using 🔄 Poll Now")

    live_df = pd.DataFrame(st.session_state.live_orders)
    live_df["State"] = live_df.apply(lambda r: state_label(r.to_dict()), axis=1)

    token_map_live = {a["name"]: a["token"] for a in st.session_state.accounts}

    def cancel_all_side(side: str):
        _m = st.session_state.get("mock_mode_toggle", MOCK_MODE)
        updated = []
        for o in st.session_state.live_orders:
            o = dict(o)
            if o.get("Side") == side and o.get("state") == "PLACED" and o.get("order_id"):
                token = token_map_live.get(o.get("Account",""), "")
                try:
                    cancel_order(token, o["order_id"], mock=_m)
                    o["state"] = "CANCELLED"
                except Exception as e:
                    o["error"] = str(e)
            updated.append(o)
        st.session_state.live_orders   = updated
        st.session_state.staged_orders = updated
        st.rerun()

    display_cols = ["Symbol","Exchange","Tier","Side","Qty","Price","Value ₹","Account","State"]

    base_col_cfg = {
        "Price":   st.column_config.NumberColumn("Price ₹",  format="₹%.2f"),
        "Value ₹": st.column_config.NumberColumn("Value ₹",  format="₹%.0f"),
        "Tier":    st.column_config.NumberColumn("Tier",      format="%d", width="small"),
        "State":   st.column_config.TextColumn("State",       width="large"),
    }
    placed_col_cfg = {**base_col_cfg,
                      "order_id":  st.column_config.TextColumn("Order ID"),
                      "placed_at": st.column_config.TextColumn("Placed At"),
                      "filled_at": st.column_config.TextColumn("Filled At"),
                      }

    # ════════════════════════════════════════════════════════════════════════
    # TABLE 1 — Planned Orders  (PENDING + COOLING_OFF + SKIPPED)
    # ════════════════════════════════════════════════════════════════════════
    pending_df = live_df[live_df["state"].isin(["PENDING","COOLING_OFF","SKIPPED"])].copy()
    # Add a numeric "Cool Off" column
    # — COOLING_OFF rows: live countdown from T1 placed_at
    # — PENDING T2–T4 rows: show full 90 min (not yet started)
    # — T1 PENDING rows: blank (unconditional)
    # Build a placed_at lookup from live_orders for cooling_off calc
    # Keys are normalised to plain int strings ("1","2","3","4") so that
    # "1.0" (pandas float) and "1" (python int) always hit the same bucket.
    def _tier_key(o):
        try: return str(int(float(str(o.get("Tier", 0) or 0))))
        except Exception: return str(o.get("Tier","")).strip()

    _placed_at_lookup = {
        (o.get("Symbol",""), o.get("Side",""), _tier_key(o)): o.get("placed_at","")
        for o in st.session_state.live_orders
    }
    _min_hours_lookup = {
        (o.get("Symbol",""), o.get("Side",""), _tier_key(o)): float(o.get("Min Hours", 1.5) or 1.5)
        for o in st.session_state.live_orders
    }

    def _cooloff_display(r):
        state     = r["state"]
        tier_raw  = r["Tier"]
        # Normalise to plain int string so it matches _placed_at_lookup keys
        try:
            tier_int  = int(float(str(tier_raw))) if pd.notna(tier_raw) else 1
        except Exception:
            tier_int  = 1
        tier_str  = str(tier_int)
        is_t1     = tier_int == 1
        key       = (r["Symbol"], r["Side"], tier_str)
        min_hours = _min_hours_lookup.get(key, 1.5)
        full_mins = round(min_hours * 60, 0)

        if state == "COOLING_OFF":
            placed_at = _placed_at_lookup.get(key, "")
            if not placed_at:
                return full_mins
            try:
                dt = datetime.strptime(placed_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                elapsed = (datetime.now(timezone.utc) - dt).total_seconds() / 60
                return max(0.0, round(full_mins - elapsed, 0))
            except Exception:
                return full_mins

        if state in ("PENDING", "SKIPPED") and not is_t1:
            # Cool Off = full_window - elapsed since prev tier was placed.
            # Formula: 90 min - (now - placed_at of same Symbol/Side Tier N-1)
            prev_key = (r["Symbol"], r["Side"], str(tier_int - 1))
            prev_placed_at = _placed_at_lookup.get(prev_key, "")
            if not prev_placed_at:
                return full_mins  # previous tier not yet placed — show full window
            try:
                dt = datetime.strptime(prev_placed_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                elapsed = (datetime.now(timezone.utc) - dt).total_seconds() / 60
                return max(0.0, round(full_mins - elapsed, 0))
            except Exception:
                return full_mins

        return None   # T1 or terminal states

    pending_df["Cool Off"] = pending_df.apply(_cooloff_display, axis=1)
    st.subheader(f"📋 Planned Orders  ({len(pending_df)} remaining)")
    if pending_df.empty:
        st.success("All orders have been placed or completed.")
    else:
        pending_display = ["Symbol","Exchange","Tier","Side","Qty","Price","Value ₹",
                           "Account","Cool Off","State"]
        pending_col_cfg = {**base_col_cfg,
            "Cool Off": st.column_config.NumberColumn(
                "⏱ Cool Off (min)",
                help="Minutes remaining before this tier is eligible to place.",
                format="%.0f min",
            ),
        }
        st.dataframe(
            pending_df[pending_display],
            use_container_width=True, hide_index=True,
            column_config=pending_col_cfg,
        )

    # ════════════════════════════════════════════════════════════════════════
    # TABLE 2 — BUY orders  (PLACED / FILLED / CANCELLED / ERROR)
    # ════════════════════════════════════════════════════════════════════════
    buy_live_df = live_df[
        (live_df["Side"] == "BUY") &
        (live_df["state"].isin(["PLACED","FILLED","CANCELLED","ERROR"]))
        ].copy()

    t2_hdr, t2_cancel = st.columns([4, 1])
    t2_hdr.subheader(f"🟢 BUY Orders  ({len(buy_live_df)})")
    if t2_cancel.button("🚫 Cancel ALL BUY", use_container_width=True,
                        disabled=not any(o.get("state")=="PLACED" and o.get("Side")=="BUY"
                                         for o in st.session_state.live_orders)):
        cancel_all_side("BUY")

    if buy_live_df.empty:
        st.caption("No BUY orders placed yet.")
    else:
        buy_cols = [c for c in display_cols + ["order_id","placed_at","filled_at"]
                    if c in buy_live_df.columns]
        st.dataframe(buy_live_df[buy_cols], use_container_width=True,
                     hide_index=True, column_config=placed_col_cfg)

    # ════════════════════════════════════════════════════════════════════════
    # TABLE 3 — SELL orders  (PLACED / FILLED / CANCELLED / ERROR)
    # ════════════════════════════════════════════════════════════════════════
    sell_live_df = live_df[
        (live_df["Side"] == "SELL") &
        (live_df["state"].isin(["PLACED","FILLED","CANCELLED","ERROR"]))
        ].copy()

    t3_hdr, t3_cancel = st.columns([4, 1])
    t3_hdr.subheader(f"🔴 SELL Orders  ({len(sell_live_df)})")
    if t3_cancel.button("🚫 Cancel ALL SELL", use_container_width=True,
                        disabled=not any(o.get("state")=="PLACED" and o.get("Side")=="SELL"
                                         for o in st.session_state.live_orders)):
        cancel_all_side("SELL")

    if sell_live_df.empty:
        st.caption("No SELL orders placed yet.")
    else:
        sell_cols = [c for c in display_cols + ["order_id","placed_at","filled_at"]
                     if c in sell_live_df.columns]
        st.dataframe(sell_live_df[sell_cols], use_container_width=True,
                     hide_index=True, column_config=placed_col_cfg)

# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption("⚠️ Live Zerodha Kite data. Verify all figures before acting. Not financial advice.")