import pandas as pd
import requests
import json
import os
import random
import time
from datetime import datetime, timezone

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

KITE_BASE_URL    = "https://api.kite.trade"
EXCLUSIONS_FILE  = "outputs/excluded_symbols.json"
WEIGHTS_FILE     = "outpus/target_weights.txt"
LIVE_ORDERS_FILE = "outputs/live_orders.json"
MOCK_MODE        = True

# ── Persistence Helpers ──
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

# ── Kite API Helpers ──
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

def fetch_trades(token: str, mock: bool = False, live_orders: list = None) -> list:
    if mock:
        filled = []
        for o in (live_orders or []):
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

# ── Business Logic & Fees ──
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

# ── State Machine Helpers ──
def now_ist_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def cooling_off_remaining(order: dict) -> float:
    anchor    = order.get("placed_at", "") 
    min_hours = float(order.get("Min Hours", 1.5) or 1.5)
    if not anchor:
        return min_hours * 60 
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
    now = datetime.now(timezone.utc)
    updated = []
    placed_ids = {o.get("order_id") for o in orders if o.get("state") == "PLACED"}
    filled_ids = placed_ids if mock else set()

    if not mock:
        try:
            all_tokens = list(token_map.values())
            if all_tokens:
                trades = fetch_trades(all_tokens[0], mock=False)
                filled_ids = {t["order_id"] for t in trades}
        except Exception:
            pass

    for o in orders:
        o = dict(o) 
        state    = o.get("state", "PENDING")
        sym      = o["Symbol"]
        tier     = o.get("Tier", "")
        is_tier1 = str(tier).strip() in ("1", "1.0")

        if state == "SKIPPED":
            o["state"]       = "PENDING"
            o["skip_reason"] = ""
            state            = "PENDING"

        if state == "PLACED":
            if o.get("order_id") in filled_ids:
                o["state"]     = "FILLED"
                o["filled_at"] = now_ist_str()
            updated.append(o)
            continue

        if state in ("FILLED", "CANCELLED", "ERROR"):
            updated.append(o)
            continue

        if state == "COOLING_OFF":
            if cooling_off_remaining(o) <= 0:
                o["state"] = "PENDING"
            updated.append(o)
            continue

        acc_name = o.get("Account", "")
        token    = token_map.get(acc_name, "")
        price    = float(o.get("Price", 0))

        if not is_tier1:
            if not token or price <= 0:
                o["state"] = "PENDING" 
                updated.append(o)
                continue
            
            ltp = ltp_map.get(sym, 0)
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

        if not token or price <= 0:
            o["state"] = "PENDING"
            updated.append(o)
            continue

        try:
            qty_to_place = int(o["Qty"])
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
            o["state"]       = "PLACED"
            o["order_id"]    = resp.get("data", {}).get("order_id", "?")
            o["Qty"]         = qty_to_place   
            o["placed_at"]   = now_ist_str()
            o["Last Filled"] = now_ist_str()
        except Exception as e:
            o["state"] = "ERROR"
            o["error"] = str(e)
        updated.append(o)

    placed_at_map: dict = {}
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
            o["state"]     = "COOLING_OFF"
            o["placed_at"] = prev_placed_at   
        elif o.get("state") == "COOLING_OFF":
            if prev_placed_at:
                o["placed_at"] = prev_placed_at
    return updated

# ── Order Builders ──
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
            "Min Hours": 1.5, "Last Filled": "", "Account": "", "Status": "",
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
            "Min Hours": 1.5, "Last Filled": "", "Account": "", "Status": "",
        })
    return orders

def fixed_bucket_sell_orders(symbol: str, exchange: str, ltp: float,
                             delta_qty: int, fluc_pct: float) -> list[dict]:
    buckets = [10_000, 20_000, 40_000, 80_000]
    extras  = [0, fluc_pct/4, fluc_pct/2, fluc_pct*3/4]
    orders, remaining = [], abs(delta_qty)
    for tier_idx, (inr_bucket, extra_pct) in enumerate(zip(buckets, extras)):
        if remaining <= 0: break
        price = round(ltp * (1 + 0.0025 + extra_pct / 100), 2)
        qty   = min(remaining, max(1, int(inr_bucket / price)))
        if qty > 0:
            orders.append({
                "Symbol": symbol, "Exchange": exchange, "Side": "SELL",
                "Qty": qty, "Price": price, "Value ₹": round(qty * price, 2),
                "Fluc%": round(fluc_pct, 3), "Tier": tier_idx + 1,
                "Min Hours": 1.5, "Last Filled": "", "Account": "", "Status": "",
            })
            remaining -= qty
    return orders
