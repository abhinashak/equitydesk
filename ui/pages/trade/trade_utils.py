import os
import json
import random
import time
import hashlib
import logging
import requests
import pandas as pd
import duckdb
import streamlit as st
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

BASE_DIR    = os.path.dirname(os.path.abspath(os.path.join(__file__, "../..")))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
CONFIG_DIR  = os.path.join(BASE_DIR, "config")
SQLS_DIR    = os.path.join(BASE_DIR, "sqls")
SECRET_DIR  = os.path.join(BASE_DIR, ".secret")

for directory in [OUTPUTS_DIR, CONFIG_DIR, SQLS_DIR, SECRET_DIR]:
    os.makedirs(directory, exist_ok=True)

EXCLUSIONS_FILE  = os.path.join(OUTPUTS_DIR, "excluded_symbols.json")
LIVE_ORDERS_FILE = os.path.join(OUTPUTS_DIR, "live_orders.json")
DB_PATH          = os.path.join(OUTPUTS_DIR, "market_data.db")
WEIGHTS_FILE     = os.path.join(CONFIG_DIR, "target_weights.txt")
TICKER_CONFIG    = os.path.join(CONFIG_DIR, "ticker.csv")
SECRET_FILE      = os.path.join(SECRET_DIR, "kite.secret")

KITE_BASE_URL    =  "https://api.kite.trade"
#KITE_BASE_URL    =  "http://localhost:8080"
KITE_PROXY_URL    = "https://api.kite.trade"
MOCK_MODE = False  # live by default; override per call

def init_db():
    conn = duckdb.connect(DB_PATH)
    try:
        init_sql_path = os.path.join(SQLS_DIR, "init.sql")
        if os.path.exists(init_sql_path):
            with open(init_sql_path, "r") as f:
                conn.execute(f.read())
        else:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ticker_prices (
                    Date DATE, Close DOUBLE, High DOUBLE, Low DOUBLE,
                    Open DOUBLE, Volume BIGINT, Ticker VARCHAR,
                    nse_symbol VARCHAR, year INT
                )
            """)
    finally:
        conn.close()

def load_exclusions() -> set:
    if os.path.exists(EXCLUSIONS_FILE):
        try:
            with open(EXCLUSIONS_FILE) as f:
                return set(json.load(f))
        except: pass
    return set()

def save_exclusions(excl: set):
    with open(EXCLUSIONS_FILE, "w") as f: json.dump(sorted(excl), f, indent=2)

def load_weights() -> str:
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE) as f: return f.read()
        except: pass
    return ""

def save_weights(text: str):
    with open(WEIGHTS_FILE, "w") as f: f.write(text)

def save_live_orders(orders: list):
    with open(LIVE_ORDERS_FILE, "w") as f: json.dump(orders, f, indent=2, default=str)

def load_live_orders() -> list:
    if os.path.exists(LIVE_ORDERS_FILE):
        try:
            with open(LIVE_ORDERS_FILE) as f: return json.load(f)
        except: pass
    return []

def load_sectors() -> dict:
    if os.path.exists(TICKER_CONFIG):
        try:
            df = pd.read_csv(TICKER_CONFIG)
            if "Symbol" in df.columns and "Sector" in df.columns:
                return dict(zip(df["Symbol"], df["Sector"]))
        except: pass
    return {}

def generate_access_token(login_url: str, api_key: str, api_secret: str) -> str:
    qs = parse_qs(urlparse(login_url).query)
    status = qs.get("status", [""])[0]
    if status != "success": raise ValueError(f"Login failed. status={status}")
    request_token = qs["request_token"][0]
    payload = api_key + request_token + api_secret
    checksum = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    logger.info("REQUEST POST %s/session/token | data=api_key=%s request_token=%s", KITE_BASE_URL, api_key, request_token)
    response = requests.post(f"{KITE_BASE_URL}/session/token", headers={"X-Kite-Version": "3"},
                             data={"api_key": api_key, "request_token": request_token, "checksum": checksum})
    response.raise_for_status()
    result = response.json()
    if result["status"] != "success": raise ValueError(json.dumps(result, indent=2))
    return f"{api_key}:{result['data']['access_token']}"

def kite_headers(token: str) -> dict:
    return {"X-Kite-Version": "3", "Authorization": f"token {token}"}

def fetch_holdings(token: str) -> list:
    logger.info("REQUEST GET %s/portfolio/holdings", KITE_BASE_URL)
    r = requests.get(f"{KITE_BASE_URL}/portfolio/holdings", headers=kite_headers(token), timeout=10)
    r.raise_for_status()
    resp = r.json()
    if resp.get("status") != "success": raise ValueError(resp.get("message", "API error"))
    return resp.get("data", [])

def place_order(token: str, symbol: str, exchange: str, side: str, qty: int,
                price: float, order_type: str = "LIMIT", mock: bool = False) -> dict:
    """Place a BUY/SELL order via the proxy (never direct to Kite)."""
    if mock:
        fake_id = f"{int(time.time())}{random.randint(100000, 999999)}"
        return {"status": "success", "data": {"order_id": fake_id}}

    data = {
        "tradingsymbol":    symbol,
        "exchange":         exchange,
        "transaction_type": side,
        "order_type":       order_type,
        "quantity":         qty,
        "product":          "CNC",
        "validity":         "DAY",
    }
    if order_type == "LIMIT":
        data["price"] = price

    logger.info("REQUEST POST %s/orders/regular | symbol=%s exchange=%s side=%s qty=%s price=%s order_type=%s",
                KITE_PROXY_URL, symbol, exchange, side, qty, price, order_type)
    r = requests.post(f"{KITE_PROXY_URL}/orders/regular",
                      headers=kite_headers(token), data=data, timeout=10)

    # Raise with Kite's actual error message instead of generic 400
    if not r.ok:
        try:
            msg = r.json().get("message", r.text)
        except Exception:
            msg = r.text
        raise requests.HTTPError(f"Kite API error: {msg}", response=r)

    return r.json()

def cancel_order(token: str, order_id: str, mock: bool = False) -> dict:
    """Cancel an open order via the proxy (never direct to Kite)."""
    if mock:
        return {"status": "success", "data": {"order_id": order_id}}
    logger.info("REQUEST DELETE %s/orders/regular/%s", KITE_PROXY_URL, order_id)
    r = requests.delete(f"{KITE_PROXY_URL}/orders/regular/{order_id}",
                        headers=kite_headers(token), timeout=10)
    # Raise with Kite's actual error message instead of generic 400
    if not r.ok:
        try:
            msg = r.json().get("message", r.text)
        except Exception:
            msg = r.text
        raise requests.HTTPError(f"Kite API error: {msg}", response=r)
    return r.json()

def fetch_trades(token: str, mock: bool = False, live_orders: list = None) -> list:
    """Fetch executed trades via the proxy.  In mock mode returns live_orders list."""
    if mock:
        return live_orders or []
    logger.info("REQUEST GET %s/trades", KITE_PROXY_URL)
    r = requests.get(f"{KITE_PROXY_URL}/trades",
                     headers=kite_headers(token), timeout=10)
    r.raise_for_status()
    resp = r.json()
    if resp.get("status") != "success":
        raise ValueError(resp.get("message", "API error"))
    return resp.get("data", [])


def fetch_orders(token: str, mock: bool = False, live_orders: list = None) -> list:
    """Fetch today's open + terminal orders via the proxy."""
    if mock:
        return live_orders or []
    logger.info("REQUEST GET %s/orders", KITE_PROXY_URL)
    r = requests.get(f"{KITE_PROXY_URL}/orders",
                     headers=kite_headers(token), timeout=10)
    r.raise_for_status()
    resp = r.json()
    if resp.get("status") != "success":
        raise ValueError(resp.get("message", "API error"))
    return resp.get("data", [])

def calc_fees(buy_val: float, sell_val: float, sell_scrips: int) -> dict:
    turnover = buy_val + sell_val
    stt = turnover * 0.001
    etc = turnover * 0.0000325
    sebi = turnover * 0.000001
    gst = (etc + sebi) * 0.18
    dp = sell_scrips * 15.34 * 1.18
    return {
        "STT (0.1%)": round(stt, 2), "Exch. (0.00325%)": round(etc, 2), "SEBI (0.0001%)": round(sebi, 2),
        "GST (18%)": round(gst, 2), f"DP ×{sell_scrips}": round(dp, 2), "Total Fees": round(stt + etc + sebi + gst + dp, 2),
    }

@st.cache_data(ttl=300, show_spinner=False)
def get_fluctuation_pct(symbol: str, exchange: str = "NSE") -> float:
    if not HAS_YF: return 1.0
    suffix = ".NS" if exchange == "NSE" else ".BO"
    try:
        hist = yf.Ticker(f"{symbol}{suffix}").history(period="1d", interval="5m")
        if hist.empty: return 1.0
        return float((hist["High"].max() - hist["Low"].min()) / hist["Low"].min() * 100)
    except: return 1.0

@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_yf(symbol: str, exchange: str = "NSE") -> float:
    if not HAS_YF: return 0.0
    suffix = ".NS" if exchange == "NSE" else ".BO"
    try:
        ticker = yf.Ticker(f"{symbol}{suffix}")
        info = ticker.fast_info
        price = float(info.get("last_price") or info.get("regularMarketPrice") or 0)
        if price > 0: return price
        hist = ticker.history(period="2d")
        if not hist.empty: return float(hist["Close"].iloc[-1])
    except: pass
    return 0.0

def get_historical_metrics(symbols_qty_map: dict):
    intervals = {"1d": relativedelta(days=1), "7d": relativedelta(days=7), "1m": relativedelta(months=1),
                 "2m": relativedelta(months=2), "3m": relativedelta(months=3), "6m": relativedelta(months=6),
                 "1y": relativedelta(years=1), "3y": relativedelta(years=3), "5y": relativedelta(years=5)}
    conn = duckdb.connect(DB_PATH)
    try:
        today = datetime.now()
        results = {}
        sym_list = "','".join(list(symbols_qty_map.keys()) + ["^NSEI"])
        for label, delta in intervals.items():
            target_date = (today - delta).strftime("%Y-%m-%d")
            query = f"""
                SELECT nse_symbol, Close FROM (
                    SELECT nse_symbol, Close, ROW_NUMBER() OVER (PARTITION BY nse_symbol ORDER BY Date DESC) as rn
                    FROM ticker_prices WHERE Date <= '{target_date}' AND nse_symbol IN ('{sym_list}')
                ) WHERE rn = 1
            """
            df_hist = conn.execute(query).df()
            if df_hist.empty:
                results[label] = {"Port Value": None, "Nifty Price": None}
                continue
            hist_map = dict(zip(df_hist["nse_symbol"], df_hist["Close"]))
            port_val = sum(hist_map[sym] * qty for sym, qty in symbols_qty_map.items() if sym in hist_map)
            results[label] = {"Port Value": port_val, "Nifty Price": hist_map.get("^NSEI")}
        return results
    except: return {}
    finally: conn.close()

def parse_weights(text: str) -> dict:
    weights = {}
    for line in text.strip().splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 2:
            try: weights[parts[0].strip().upper()] = float(parts[1].strip())
            except: pass
    return weights

def now_ist_str() -> str: return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def cooling_off_remaining(order: dict) -> float:
    anchor, min_hours = order.get("placed_at", ""), float(order.get("Min Hours", 1.5) or 1.5)
    if not anchor: return min_hours * 60
    try:
        dt = datetime.strptime(anchor, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return max(0.0, min_hours * 60 - (datetime.now(timezone.utc) - dt).total_seconds() / 60)
    except: return 0.0

def state_label(order: dict) -> str:
    s = order.get("state", "PENDING")
    if s == "COOLING_OFF": return f"⏱ Cooling off — {cooling_off_remaining(order):.0f}m left"
    return {"PENDING": "⏳ Pending", "PLACED": f"📤 Placed ({order.get('order_id','')})",
            "FILLED": f"✅ Filled ({order.get('order_id','')})", "SKIPPED": f"⏭ {order.get('skip_reason','')}",
            "ERROR": f"❌ {order.get('error','')}", "CANCELLED": f"🚫 Cancelled ({order.get('order_id','')})"}.get(s, s)

def init_trade_session():
    init_db()
    defaults = {
        "accounts": [{"name": "Account 1", "token": ""}], "all_holdings": {},
        "excluded_symbols": load_exclusions(), "action_overrides": {}, "staged_orders": [],
        "live_orders": [], "runner_active": False, "last_poll_time": None,
        "weights_text": load_weights(), "buy_strategy_used": "Geometric Progression",
        "sell_strategy_used": "Geometric Progression (GP)", "fresh_capital": 0.0
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

# ── Order Builders ──
def geo_buy_orders(symbol: str, exchange: str, ltp: float, delta_qty: int, fluc_pct: float) -> list[dict]:
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

def geo_sell_orders(symbol: str, exchange: str, ltp: float, delta_qty: int, fluc_pct: float) -> list[dict]:
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

def fixed_bucket_sell_orders(symbol: str, exchange: str, ltp: float, delta_qty: int, fluc_pct: float) -> list[dict]:
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