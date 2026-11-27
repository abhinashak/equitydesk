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
GTT_EXCLUSIONS_FILE = os.path.join(OUTPUTS_DIR, "gtt_excluded_symbols.json")   # persists across sessions
GTT_PARK_FILE    = os.path.join(OUTPUTS_DIR, "gtt_parked.json")   # GTTs deleted before open, to be re-added later
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

def load_gtt_exclusions() -> set:
    """Symbols that must NOT get a protective GTT (kept on disk, so it survives sessions)."""
    if os.path.exists(GTT_EXCLUSIONS_FILE):
        try:
            with open(GTT_EXCLUSIONS_FILE) as f:
                return {str(s).strip().upper() for s in json.load(f) if str(s).strip()}
        except Exception:
            pass
    return set()

def save_gtt_exclusions(excl) -> None:
    tmp = GTT_EXCLUSIONS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sorted({str(s).strip().upper() for s in excl if str(s).strip()}), f, indent=2)
    os.replace(tmp, GTT_EXCLUSIONS_FILE)

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
    elif order_type == "MARKET":
        data["market_protection"] = -1   # Kite requires market protection on API market orders (-1 = automatic)

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

def fetch_gtts(token: str) -> list:
    """Fetch active/recent GTT triggers from Kite."""
    logger.info("REQUEST GET %s/gtt/triggers", KITE_PROXY_URL)
    r = requests.get(f"{KITE_PROXY_URL}/gtt/triggers", headers=kite_headers(token), timeout=10)
    if not r.ok:
        try:
            msg = r.json().get("message", r.text)
        except Exception:
            msg = r.text
        raise requests.HTTPError(f"Kite GTT API error: {msg}", response=r)
    resp = r.json()
    if resp.get("status") != "success":
        raise ValueError(resp.get("message", "GTT API error"))
    return resp.get("data", [])

def place_gtt_market_sell(token: str, symbol: str, exchange: str, qty: int,
                          trigger_price: float, reference_price: float, product: str = "CNC") -> dict:
    """Place single-trigger GTT SELL using MARKET + automatic market protection.
    Public Kite Connect GTT docs currently describe LIMIT legs; MARKET support
    here is based on the Kite web UI and may be rejected for API clients.
    """
    condition = {"exchange": exchange, "tradingsymbol": symbol,
                 "trigger_values": [float(trigger_price)], "last_price": float(reference_price)}
    order_leg = {"exchange": exchange, "tradingsymbol": symbol,
                 "transaction_type": "SELL", "quantity": int(qty),
                 "order_type": "MARKET", "product": product, "market_protection": -1}
    data = {"type": "single", "condition": json.dumps(condition),
            "orders": json.dumps([order_leg])}
    logger.info("REQUEST POST %s/gtt/triggers | MARKET SELL GTT %s qty=%s trigger=%s",
                KITE_PROXY_URL, symbol, qty, trigger_price)
    r = requests.post(f"{KITE_PROXY_URL}/gtt/triggers",
                      headers=kite_headers(token), data=data, timeout=10)
    if not r.ok:
        try:
            msg = r.json().get("message", r.text)
        except Exception:
            msg = r.text
        raise requests.HTTPError(f"Kite GTT API error: {msg}", response=r)
    resp = r.json()
    if resp.get("status") != "success":
        raise ValueError(resp.get("message", "GTT placement failed"))
    return resp

# ─────────────────────────────────────────────────────────────────────────────
# Park GTTs over the open: SAVE → DELETE (before 9:15) → RE-ADD (≈10:00)
# ─────────────────────────────────────────────────────────────────────────────
def delete_gtt(token: str, trigger_id) -> dict:
    """DELETE /gtt/triggers/{id}"""
    logger.info("REQUEST DELETE %s/gtt/triggers/%s", KITE_PROXY_URL, trigger_id)
    r = requests.delete(f"{KITE_PROXY_URL}/gtt/triggers/{trigger_id}",
                        headers=kite_headers(token), timeout=10)
    if not r.ok:
        try:
            msg = r.json().get("message", r.text)
        except Exception:
            msg = r.text
        raise requests.HTTPError(f"Kite GTT delete error: {msg}", response=r)
    resp = r.json()
    if resp.get("status") != "success":
        raise ValueError(resp.get("message", "GTT delete failed"))
    return resp

def load_parked_gtts() -> list:
    """Parked-GTT snapshot. Missing file → []. A corrupt file RAISES (never silently
    treated as empty, because that would lose the only record of deleted GTTs)."""
    if not os.path.exists(GTT_PARK_FILE):
        return []
    with open(GTT_PARK_FILE) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{GTT_PARK_FILE} is not a list")
    return data

def save_parked_gtts(items: list) -> None:
    tmp = GTT_PARK_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(items, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, GTT_PARK_FILE)

def discard_parked_gtt(account: str, gtt_id) -> None:
    """Forget one parked entry (e.g. you decided to exit that stock manually)."""
    items = [p for p in load_parked_gtts()
             if not (p.get("account") == account and str(p.get("gtt_id")) == str(gtt_id))]
    save_parked_gtts(items)

def _ist_stamp() -> str:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")

def _parkable_gtt(g: dict):
    """Return the fields needed to re-create this GTT with place_gtt_market_sell(),
    or None if it can't be re-created exactly (those are NEVER deleted)."""
    if str(g.get("status", "")).lower() != "active" or str(g.get("type", "single")).lower() != "single":
        return None
    cond = g.get("condition") or {}
    legs = g.get("orders") or []
    vals = cond.get("trigger_values") or []
    if len(legs) != 1 or len(vals) != 1:
        return None
    leg = legs[0]
    if str(leg.get("transaction_type", "")).upper() != "SELL" or str(leg.get("order_type", "")).upper() != "MARKET":
        return None
    qty = int(leg.get("quantity") or 0)
    if qty <= 0:
        return None
    return {"exchange": str(cond.get("exchange") or leg.get("exchange") or "NSE").upper(),
            "symbol": str(cond.get("tradingsymbol") or leg.get("tradingsymbol") or "").upper(),
            "qty": qty, "trigger": float(vals[0]),
            "product": str(leg.get("product") or "CNC").upper(),
            "last_price": float(cond.get("last_price") or 0)}

def build_gtt_table(gtts: list, ltp_map: dict, qty_map=None) -> pd.DataFrame:
    """Kite-style GTT table: Created on · Instrument · Holdings · Type · Trigger · vs LTP % · LTP · Qty · Holding check · Status.
    Only ACTIVE GTTs are listed. `Parkable` says whether Park/Re-add can handle it exactly."""
    rows = []
    for g in gtts:
        if str(g.get("status", "")).lower() != "active":
            continue
        cond = g.get("condition") or {}
        legs = g.get("orders") or []
        vals = cond.get("trigger_values") or []
        sym = str(cond.get("tradingsymbol", "")).upper()
        trig = float(vals[0]) if vals else None
        ltp = float(ltp_map.get(sym) or 0) or None
        sides = "/".join(sorted({str(l.get("transaction_type", "")).upper() for l in legs})) or "—"
        rows.append({
            "Created on": str(g.get("created_at", ""))[:10],
            "Instrument": f"{sym} {str(cond.get('exchange', 'NSE')).upper()}",
            "Holdings": (int(qty_map.get(sym, 0)) if qty_map is not None else None),
            "Type": f"{str(g.get('type', 'single')).upper()} · {sides}",
            "Trigger": trig,
            "vs LTP %": round((trig - ltp) / ltp * 100, 2) if (trig and ltp) else None,
            "LTP": ltp,
            "Qty": (_q := sum(int(l.get("quantity") or 0) for l in legs)),
            "Holding check": ("" if qty_map is None else
                              f"⚠️ Short by {_q - int(qty_map.get(sym, 0))}" if _q > int(qty_map.get(sym, 0))
                              else "✅ OK"),
            "Status": "ACTIVE",
            "Parkable": "Yes" if _parkable_gtt(g) else "No",
            "ID": str(g.get("id")),
        })
    return pd.DataFrame(rows, columns=["Created on", "Instrument", "Holdings", "Type", "Trigger", "vs LTP %",
                                       "LTP", "Qty", "Holding check", "Status", "Parkable", "ID"])

def list_parkable_gtts(token: str) -> list:
    """Active GTTs that park_active_gtts() can delete and later re-create exactly."""
    out = []
    for g in fetch_gtts(token):
        info = _parkable_gtt(g)
        if info:
            out.append({"gtt_id": str(g.get("id")), **info})
    return out

def park_active_gtts(token: str, account: str, only_ids=None) -> dict:
    """1) fetch active SELL MARKET GTTs (only `only_ids` if given)  2) SAVE them to GTT_PARK_FILE  3) DELETE them.
    The snapshot is written to disk BEFORE the first delete, and after every delete."""
    gtts = fetch_gtts(token)                       # raises on failure → nothing is deleted
    parked = [p for p in load_parked_gtts() if not p.get("restored")]   # prune finished entries
    known = {(p.get("account"), str(p.get("gtt_id"))) for p in parked}
    skipped = []
    for g in gtts:
        if str(g.get("status", "")).lower() != "active":
            continue
        gid = str(g.get("id"))
        if only_ids is not None and gid not in {str(i) for i in only_ids}:
            continue
        if (account, gid) in known:
            continue
        info = _parkable_gtt(g)
        if info is None:
            skipped.append(g)
            continue
        parked.append({"account": account, "gtt_id": gid, **info, "created_at": g.get("created_at"),
                       "parked_at": _ist_stamp(), "deleted": False, "restored": False, "new_id": None})
    save_parked_gtts(parked)                       # ← snapshot first
    deleted, errors = 0, []
    for e in parked:
        if e.get("account") != account or e.get("restored") or e.get("deleted"):
            continue
        try:
            delete_gtt(token, e["gtt_id"])
            e["deleted"] = True
            deleted += 1
        except Exception as err:
            errors.append((e["symbol"], str(err)))
        save_parked_gtts(parked)
    return {"deleted": deleted, "skipped": skipped, "errors": errors}

def fetch_ltp_and_qty_maps(token: str):
    """One pass over Kite holdings + positions.
    Returns (ltp_map, qty_map): {SYMBOL: live last_price} and {SYMBOL: shares held}.
    Held = holdings quantity + t1_quantity (bought, not yet settled). Positions only fill in
    CNC symbols missing from holdings, so nothing is double-counted. qty_map is None when the
    holdings call failed (so the UI shows blanks instead of a misleading 0)."""
    ltp, qty = {}, {}
    holdings_ok = True
    try:
        for h in fetch_holdings(token):
            sym = str(h.get("tradingsymbol", "")).upper()
            lp = float(h.get("last_price") or 0)
            if sym and lp > 0:
                ltp[sym] = lp
            q = int(h.get("quantity") or 0) + int(h.get("t1_quantity") or 0)
            if sym:
                qty[sym] = qty.get(sym, 0) + q
    except Exception:
        holdings_ok = False
        logger.exception("fetch_ltp_and_qty_maps: holdings failed")
    try:
        pos = fetch_positions(token)
        for row in (pos.get("day") or []) + (pos.get("net") or []):
            sym = str(row.get("tradingsymbol", "")).upper()
            lp = float(row.get("last_price") or 0)
            if sym and lp > 0:
                ltp[sym] = lp
        if holdings_ok:
            for row in (pos.get("net") or []):
                sym = str(row.get("tradingsymbol", "")).upper()
                nq = int(row.get("quantity") or 0)
                if sym and sym not in qty and nq > 0 and str(row.get("product", "")).upper() == "CNC":
                    qty[sym] = nq
    except Exception:
        logger.exception("fetch_ltp_and_qty_maps: positions failed")
    return ltp, (qty if holdings_ok else None)

def fetch_ltp_map(token: str) -> dict:
    """{SYMBOL: live last_price} straight from Kite holdings + positions (no Quote API)."""
    return fetch_ltp_and_qty_maps(token)[0]

def restore_parked_gtts(token: str, account: str, ltp_map: dict) -> list:
    """Re-add every deleted-but-not-restored GTT for `account` with its saved qty/trigger.
    Skips (and leaves pending) when there is no live price, or LTP is already at/below the
    trigger — Kite would reject or instantly fire it; that stock needs a manual decision.
    Returns [(symbol, status, message)], status ∈ restored | exists | below_trigger | no_price | error."""
    parked = load_parked_gtts()
    live = fetch_gtts(token)                       # raises → nothing is placed (no duplicates)
    results = []
    for e in parked:
        if e.get("account") != account or not e.get("deleted") or e.get("restored"):
            continue
        sym, qty, trig = e["symbol"], int(e["qty"]), float(e["trigger"])
        dup = next((str(g.get("id")) for g in live
                    if str(g.get("status", "")).lower() == "active"
                    and (g.get("condition") or {}).get("tradingsymbol", "").upper() == sym
                    and abs(float(((g.get("condition") or {}).get("trigger_values") or [0])[0]) - trig) < 0.005
                    and any(str(l.get("transaction_type", "")).upper() == "SELL"
                            and int(l.get("quantity") or 0) == qty for l in (g.get("orders") or []))), None)
        if dup:
            e["restored"], e["new_id"] = True, dup
            save_parked_gtts(parked)
            results.append((sym, "exists", f"already active (ID {dup})"))
            continue
        ltp = float(ltp_map.get(sym) or 0)
        if ltp <= 0:
            results.append((sym, "no_price", "no live price available — try again"))
            continue
        if ltp <= trig:
            results.append((sym, "below_trigger",
                            f"LTP ₹{ltp:,.2f} is at/below trigger ₹{trig:,.2f} — NOT re-added, decide manually"))
            continue
        try:
            resp = place_gtt_market_sell(token, sym, e["exchange"], qty, trig, ltp, e.get("product", "CNC"))
            e["restored"] = True
            e["new_id"] = (resp.get("data") or {}).get("trigger_id")
            save_parked_gtts(parked)
            results.append((sym, "restored", f"SELL {qty} @ ₹{trig:,.2f} (new ID {e['new_id']})"))
        except Exception as err:
            results.append((sym, "error", str(err)))
    return results

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

def fetch_positions(token: str, mock: bool = False) -> dict:
    """Fetch current positions via the proxy.
    Returns {"net": [...], "day": [...]} (Kite GET /portfolio/positions)."""
    if mock:
        return {"net": [], "day": []}
    logger.info("REQUEST GET %s/portfolio/positions", KITE_PROXY_URL)
    r = requests.get(f"{KITE_PROXY_URL}/portfolio/positions",
                     headers=kite_headers(token), timeout=10)
    if not r.ok:
        try:
            msg = r.json().get("message", r.text)
        except Exception:
            msg = r.text
        raise requests.HTTPError(f"Kite positions API error: {msg}", response=r)
    resp = r.json()
    if resp.get("status") != "success":
        raise ValueError(resp.get("message", "Positions API error"))
    data = resp.get("data") or {}
    return {"net": data.get("net", []) or [], "day": data.get("day", []) or []}


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
def stagger_qtys(total: int) -> list[int]:
    """Split `total` shares into 4 stagger slots (1x, 2x, 4x, 8x, x = total // 15).
    The slots always add up to exactly `total`: earlier slots are capped by what is
    left, and the LAST slot absorbs the difference (remainder, or reduction when
    1x+2x+4x+8x would exceed the total). Slots that end up 0 can be skipped."""
    total = int(abs(total))
    x = max(1, total // 15)
    weights = [x, 2 * x, 4 * x, 8 * x]
    qtys, remaining = [], total
    for i, w in enumerate(weights):
        q = remaining if i == len(weights) - 1 else min(w, remaining)
        qtys.append(q)
        remaining -= q
    return qtys

def geo_buy_orders(symbol: str, exchange: str, ltp: float, delta_qty: int, fluc_pct: float) -> list[dict]:
    qtys = stagger_qtys(delta_qty)
    tiers = list(zip(qtys, [0, fluc_pct/4, fluc_pct/2, fluc_pct*3/4]))
    orders = []
    for tier_idx, (qty, extra_pct) in enumerate(tiers):
        if qty <= 0:
            continue
        price = round(ltp * (1 - 0.0025 - extra_pct / 100), 2)
        orders.append({
            "Symbol": symbol, "Exchange": exchange, "Side": "BUY",
            "Qty": qty, "Price": price, "Value ₹": round(qty * price, 2),
            "Fluc%": round(fluc_pct, 3), "Tier": tier_idx + 1,
            "Min Hours": 1.5, "Last Filled": "", "Account": "", "Status": "",
        })
    return orders

def geo_sell_orders(symbol: str, exchange: str, ltp: float, delta_qty: int, fluc_pct: float) -> list[dict]:
    qtys = stagger_qtys(delta_qty)
    tiers = list(zip(qtys, [0, fluc_pct/4, fluc_pct/2, fluc_pct*3/4]))
    orders = []
    for tier_idx, (qty, extra_pct) in enumerate(tiers):
        if qty <= 0:
            continue
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