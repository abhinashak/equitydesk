"""Kite tradebook FIFO analysis and P&L presentation."""
import datetime
from pathlib import Path
import pandas as pd
import streamlit as st
from utils.db_utils import load_csv_table

_STT_PATH = "config/stt.csv"
_FEE_DEFAULTS = {
    "_STT_INTRADAY_SELL": 0.00025, "_STT_DELIVERY": 0.001,
    "_BROKERAGE_INTRA": 0.0003, "_TXN_NSE": 0.0000322,
    "_SEBI_FEE": 0.000001, "_STAMP_INTRA_BUY": 0.00003,
    "_STAMP_DELIV_BUY": 0.00015, "_GST": 0.18, "_DP_PER_SCRIP": 15.34,
}

def _fee_config(platform: str = "kite") -> dict[str, float]:
    try:
        df = load_csv_table(str(_STT_PATH))
        required = {"platform", "FEE_TYPE", "PCT_OR_ABSOLUTE", "AMOUNT"}
        if not required.issubset(df.columns):
            raise ValueError(f"stt.csv must contain {sorted(required)}")
        df = df[df["platform"].astype(str).str.lower() == platform.lower()]
        out = dict(_FEE_DEFAULTS)
        for row in df.to_dict("records"):
            fee_type = str(row["FEE_TYPE"]).strip()
            unit = str(row["PCT_OR_ABSOLUTE"]).strip().upper()
            if unit not in {"PCT", "ABSOLUTE"}:
                raise ValueError(f"Unsupported fee unit {unit!r} for {fee_type}")
            out[fee_type] = float(row["AMOUNT"])
        return out
    except Exception as exc:
        # Config errors should not break the dashboard; preserve the prior fee defaults.
        print(f"[Portfolio] Could not load STT config: {exc}")
        return dict(_FEE_DEFAULTS)

_FEES = _fee_config("kite")
def _num(series, default=0.0):
    return pd.to_numeric(series, errors="coerce").fillna(default)

def _charges(buy_i=0.0, sell_i=0.0, buy_d=0.0, sell_d=0.0, dp_scrips=0):
    """(STT, other charges) for an intraday leg (buy_i/sell_i), a delivery buy (buy_d) and a delivery sell (sell_d)."""
    stt = _FEES["_STT_INTRADAY_SELL"] * sell_i + _FEES["_STT_DELIVERY"] * (buy_d + sell_d)
    brk = _FEES["_BROKERAGE_INTRA"] * (buy_i + sell_i)
    turnover = buy_i + sell_i + buy_d + sell_d
    txn, sebi = _FEES["_TXN_NSE"] * turnover, _FEES["_SEBI_FEE"] * turnover
    stamp = _FEES["_STAMP_INTRA_BUY"] * buy_i + _FEES["_STAMP_DELIV_BUY"] * buy_d
    gst = _FEES["_GST"] * (brk + txn + sebi)
    return stt, brk + txn + sebi + stamp + gst + dp_scrips * _FEES["_DP_PER_SCRIP"]

def _analyze_tradebook(udf: "pd.DataFrame", live: dict = None, today: "datetime.date" = None):
    """Zerodha tradebook CSV → (ticker_data, tracking).

    Columns used: symbol, trade_type (buy/sell), quantity, price, trade_date and/or
    order_execution_time (segment, if present, must be EQ).

    * Trades are replayed in execution-time order and matched FIFO (a sell closes the oldest buy).
        Realized P&L   = Σ (sell price − buy price) × matched qty
        Open lots      = buys not yet sold  → portfolio holdings (avg cost = weighted lot price)
        Unrealized P&L = Σ (live LTP − lot price) × open qty        (LTP from Kite holdings / positions)
    * Charges are computed per symbol per day from the executed trades. Qty bought and sold the same day
      is intraday (STT 0.025% on the sell, brokerage); the rest is delivery (STT 0.1% on buy and sell,
      DP charge on the sell). Exchange, SEBI, stamp duty and GST are included (see `_charges`).
        Net P&L = Realized + Unrealized − STT − other charges      (all charges paid so far)
    * Sells with no earlier buy in the file (older holdings, or a short) are NOT given a cost basis: they are
      excluded from P&L and listed as a warning — upload the full history to include them.
    """
    from collections import deque
    live = live or {}
    today = today or datetime.date.today()

    t = pd.DataFrame({
        "Symbol": udf["symbol"].astype(str).str.strip().str.upper(),
        "Type":   udf["trade_type"].astype(str).str.strip().str.lower(),
        "Qty":    _num(udf["quantity"]),
        "Price":  _num(udf["price"]),
    })
    _none = pd.Series(pd.NaT, index=udf.index)
    ts = pd.to_datetime(udf["order_execution_time"], errors="coerce") if "order_execution_time" in udf.columns else _none
    dd = pd.to_datetime(udf["trade_date"], errors="coerce") if "trade_date" in udf.columns else _none
    ts = ts.fillna(dd)
    t["_ts"] = ts
    t["Date"] = ts.dt.date
    ok = (t["Symbol"] != "") & t["Type"].isin(["buy", "sell"]) & (t["Qty"] > 0) & (t["Price"] > 0) & ts.notna()
    if "segment" in udf.columns:
        ok &= udf["segment"].astype(str).str.strip().str.upper().eq("EQ")
    skipped = int((~ok).sum())
    t = t[ok].sort_values("_ts", kind="stable").reset_index(drop=True)
    t["Val"] = t["Qty"] * t["Price"]
    if t.empty:
        return None, None

    # ── FIFO matching ────────────────────────────────────────────────────────
    lots, closed = {}, []                       # lots: {sym: deque([signed qty, price, date])}
    unmatched = {}                              # sold with no earlier buy in the file → no cost basis
    for r in t.itertuples(index=False):
        dq = lots.setdefault(r.Symbol, deque())
        if dq and dq[0][0] < 0 and any(l[2] != r.Date for l in dq):
            # a sell from an earlier day with no buy is older holdings, not a short a later buy can cover
            for l in dq:
                if l[2] != r.Date:
                    unmatched[r.Symbol] = unmatched.get(r.Symbol, 0.0) + abs(l[0])
            keep = [l for l in dq if l[2] == r.Date]
            dq.clear(); dq.extend(keep)
        rem = r.Qty if r.Type == "buy" else -r.Qty
        while abs(rem) > 1e-9 and dq and (dq[0][0] > 0) != (rem > 0):
            lot = dq[0]
            m = min(abs(rem), abs(lot[0]))
            if lot[0] > 0:                      # sell closes an earlier buy
                bpx, bdt, spx, sdt, side = lot[1], lot[2], r.Price, r.Date, "Long"
            else:                               # buy covers an earlier sell (same-day short)
                bpx, bdt, spx, sdt, side = r.Price, r.Date, lot[1], lot[2], "Short"
            closed.append({"Symbol": r.Symbol, "Side": side, "Qty": m, "Buy Date": bdt, "Buy ₹": bpx,
                           "Sell Date": sdt, "Sell ₹": spx, "P&L ₹": (spx - bpx) * m,
                           "Days": (max(bdt, sdt) - min(bdt, sdt)).days,
                           "Type": "Intraday" if bdt == sdt else "Delivery", "Close Date": r.Date})
            lot[0] += -m if lot[0] > 0 else m
            rem += m if rem < 0 else -m
            if abs(lot[0]) < 1e-9:
                dq.popleft()
        if abs(rem) > 1e-9:
            dq.append([rem, r.Price, r.Date])
    lots_cols = ["Symbol", "Side", "Qty", "Buy Date", "Buy ₹", "Sell Date", "Sell ₹", "P&L ₹", "Days", "Type", "Close Date"]
    closed_df = pd.DataFrame(closed, columns=lots_cols)

    # ── Charges per symbol per day ───────────────────────────────────────────
    ch = []
    for (sym, dt), g in t.groupby(["Symbol", "Date"], sort=False):
        b_, s_ = g[g["Type"] == "buy"], g[g["Type"] == "sell"]
        bq, bv, sq, sv = b_["Qty"].sum(), b_["Val"].sum(), s_["Qty"].sum(), s_["Val"].sum()
        iq = min(bq, sq)                                         # bought and sold the same day → intraday
        ab, asl = (bv / bq if bq else 0.0), (sv / sq if sq else 0.0)
        stt, oth = _charges(buy_i=iq * ab, sell_i=iq * asl, buy_d=(bq - iq) * ab, sell_d=(sq - iq) * asl,
                            dp_scrips=1 if (sq - iq) > 1e-9 else 0)
        ch.append({"Symbol": sym, "Date": dt, "STT ₹": stt, "Other ₹": oth})
    ch_df = pd.DataFrame(ch, columns=["Symbol", "Date", "STT ₹", "Other ₹"])

    # ── Open lots → holdings, unrealized P&L ─────────────────────────────────
    open_rows, ticker_data, short_open = [], {}, []
    for sym, dq in lots.items():
        longs = [l for l in dq if l[0] > 1e-9]
        for l in dq:
            if l[0] < -1e-9:
                unmatched[sym] = unmatched.get(sym, 0.0) + abs(l[0])
        q = sum(l[0] for l in longs)
        if q <= 1e-9:
            continue
        avg = sum(l[0] * l[1] for l in longs) / q
        tq = sum(l[0] for l in longs if l[2] == today)                 # part bought today
        tcost = sum(l[0] * l[1] for l in longs if l[2] == today)
        L = live.get(sym)
        if L:
            lp = L["last_price"]
            cp = L["close_price"] if L["close_price"] > 0 else avg
            pct = L["day_change_pct"] if L["close_price"] > 0 else ((lp - avg) / avg * 100 if avg else 0.0)
        else:
            lp = cp = avg
            pct = 0.0
        if tq > 0:                    # qty bought today moved from its buy price, not from yesterday's close
            cp = ((q - tq) * cp + tcost) / q
            pct = (lp - cp) / cp * 100 if cp else 0.0
        ticker_data[sym] = {"last_price": float(lp), "close_price": float(cp), "day_change_pct": float(pct),
                            "avg_cost": float(avg), "Uploaded": float(q)}
        s2, o2 = _charges(sell_d=q * lp, dp_scrips=1)                  # cost of selling the lot now
        open_rows.append({"Symbol": sym, "Qty": q, "Avg ₹": avg, "LTP ₹": lp, "Invested ₹": q * avg,
                          "Value ₹": q * lp, "Unrealized ₹": q * (lp - avg),
                          "Unreal %": (lp / avg - 1) * 100 if avg else 0.0, "Sell charges ₹": s2 + o2,
                          "Source": L["src"] if L else "No live price"})
    short_open = [f"{k} ({v:g})" for k, v in unmatched.items()]
    open_df = pd.DataFrame(open_rows, columns=["Symbol", "Qty", "Avg ₹", "LTP ₹", "Invested ₹", "Value ₹",
                                               "Unrealized ₹", "Unreal %", "Sell charges ₹", "Source"])

    # ── By-symbol and by-day summaries ───────────────────────────────────────
    real_s = closed_df.groupby("Symbol")["P&L ₹"].sum() if len(closed_df) else pd.Series(dtype=float)
    ch_s = ch_df.groupby("Symbol")[["STT ₹", "Other ₹"]].sum()
    od = open_df.set_index("Symbol") if len(open_df) else open_df.set_index(pd.Index([], name="Symbol"))
    sym_df = pd.DataFrame({"Symbol": sorted(t["Symbol"].unique())})
    sym_df["Open Qty"] = sym_df["Symbol"].map(od["Qty"]).fillna(0.0)
    sym_df["Avg ₹"] = sym_df["Symbol"].map(od["Avg ₹"])
    sym_df["LTP ₹"] = sym_df["Symbol"].map(od["LTP ₹"])
    sym_df["Unrealized ₹"] = sym_df["Symbol"].map(od["Unrealized ₹"]).fillna(0.0)
    sym_df["Realized ₹"] = sym_df["Symbol"].map(real_s).fillna(0.0)
    sym_df["STT ₹"] = sym_df["Symbol"].map(ch_s["STT ₹"]).fillna(0.0)
    sym_df["Other ₹"] = sym_df["Symbol"].map(ch_s["Other ₹"]).fillna(0.0)
    sym_df["Net P&L ₹"] = sym_df["Realized ₹"] + sym_df["Unrealized ₹"] - sym_df["STT ₹"] - sym_df["Other ₹"]
    sym_df = sym_df.sort_values("Net P&L ₹", ascending=False).reset_index(drop=True)

    dr = closed_df.groupby("Close Date")["P&L ₹"].sum().rename("Realized ₹") if len(closed_df) else pd.Series(dtype=float, name="Realized ₹")
    dc = ch_df.groupby("Date")[["STT ₹", "Other ₹"]].sum()
    daily = pd.concat([dr, dc], axis=1).fillna(0.0)
    daily.index.name = "Date"
    daily["Net Realized ₹"] = daily["Realized ₹"] - daily["STT ₹"] - daily["Other ₹"]
    daily = daily.sort_index(ascending=False).reset_index()

    stale = [r["Symbol"] for r in open_rows if r["Source"] == "No live price"]
    meta = {"connected": bool(live), "stale": stale, "short_open": short_open, "skipped": skipped,
            "n_trades": len(t), "first": min(t["Date"]), "last": max(t["Date"]),
            "sell_now": float(open_df["Sell charges ₹"].sum()) if len(open_df) else 0.0,
            "by_src": open_df["Source"].value_counts().to_dict() if len(open_df) else {}}
    return (ticker_data or None), {"symbols": sym_df, "open": open_df, "lots": closed_df,
                                   "daily": daily, "meta": meta}

def _render_trade_pnl(trk: dict, live_price_map) -> None:
    """Realized / unrealized / STT / net P&L from the Zerodha tradebook."""
    sym, o, lots, daily, meta = trk["symbols"], trk["open"], trk["lots"], trk["daily"], trk["meta"]
    st.subheader("🎯 Trade P&L")
    st.caption(f"📅 Tradebook **{meta['first']:%d %b %Y} → {meta['last']:%d %b %Y}** · "
               f"{meta['n_trades']} trades · FIFO matching")
    rc1, rc2 = st.columns([5, 1])
    if rc2.button("🔄 Refresh prices", key="pf_track_refresh", use_container_width=True):
        live_price_map(force=True)
        st.rerun()
    for _e in meta.get("errors", []) or []:
        rc1.caption(f"⚠️ Kite fetch: {_e}")
    if not meta.get("connected"):
        rc1.warning("No Kite account connected — open lots are valued at cost, so Unrealized P&L is 0. "
                    "Connect your account for live prices.")
    elif meta.get("stale"):
        rc1.warning(f"No live price for {', '.join(meta['stale'])} — valued at cost.")
    else:
        _src = ", ".join(f"{v} {k}" for k, v in (meta.get("by_src") or {}).items())
        rc1.success(f"🟢 Live prices for all {len(o)} open symbols ({_src}).")
    if meta.get("short_open"):
        st.warning("Sold with no earlier buy in this file (older holdings or a short) — excluded from P&L; "
                   "upload the full history to include them: " + ", ".join(meta["short_open"]))
    if meta.get("skipped"):
        st.caption(f"ℹ️ {meta['skipped']} row(s) skipped (blank, non-equity or invalid).")

    realized = float(sym["Realized ₹"].sum())
    unreal = float(sym["Unrealized ₹"].sum())
    stt = float(sym["STT ₹"].sum())
    other = float(sym["Other ₹"].sum())
    gross = realized + unreal
    net = gross - stt - other
    invested = float(o["Invested ₹"].sum()) if len(o) else 0.0
    value = float(o["Value ₹"].sum()) if len(o) else 0.0
    wins = int((lots["P&L ₹"] > 0).sum()) if len(lots) else 0

    m = st.columns(5)
    m[0].metric("Realized P&L", f"₹{realized:,.2f}")
    m[1].metric("Unrealized P&L", f"₹{unreal:,.2f}", f"{unreal / invested * 100:+.2f}%" if invested else "")
    m[2].metric("STT", f"₹{stt:,.2f}")
    m[3].metric("Other charges", f"₹{other:,.2f}",
                help="Brokerage on intraday, exchange, SEBI, stamp duty, GST and DP charges (estimated).")
    m[4].metric("Net P&L", f"₹{net:,.2f}", f"{net - gross:+,.2f} charges", delta_color="off",
                help="Realized + Unrealized − STT − other charges.")
    m2 = st.columns(5)
    m2[0].metric("Gross P&L", f"₹{gross:,.2f}")
    m2[1].metric("Invested (open)", f"₹{invested:,.0f}")
    m2[2].metric("Value (open, LTP)", f"₹{value:,.0f}")
    m2[3].metric("Closed lots", len(lots), f"{wins / len(lots) * 100:.0f}% winners" if len(lots) else "", delta_color="off")
    m2[4].metric("Net if all open sold now", f"₹{net - meta.get('sell_now', 0.0):,.2f}",
                 f"{-meta.get('sell_now', 0.0):+,.2f} sell charges", delta_color="off")

    def _clr(col):
        return ["color:#15803d" if v > 0 else "color:#b91c1c" if v < 0 else "" for v in col]
    def _show(df, money, pct=(), plain=()):
        df = df.reset_index(drop=True)
        fmt = {c: "{:+,.2f}" for c in money}
        fmt.update({c: "{:+.2f}%" for c in pct})
        fmt.update({c: "{:,.2f}" for c in plain})
        st.dataframe(df.style.apply(_clr, subset=list(money) + list(pct)).format(fmt, na_rep="–"),
                     use_container_width=True, hide_index=True)

    t_sym, t_open, t_lots, t_day = st.tabs(
        [f"📊 By symbol ({len(sym)})", f"🟢 Open ({len(o)})", f"✅ Closed lots ({len(lots)})", "🗓️ By day"])
    with t_sym:
        _show(sym, money=["Unrealized ₹", "Realized ₹", "Net P&L ₹"], plain=["Avg ₹", "LTP ₹", "STT ₹", "Other ₹"])
    with t_open:
        if o.empty:
            st.caption("No open positions.")
        else:
            _show(o.sort_values("Unrealized ₹", ascending=False),
                  money=["Unrealized ₹"], pct=["Unreal %"],
                  plain=["Qty", "Avg ₹", "LTP ₹", "Invested ₹", "Value ₹", "Sell charges ₹"])
    with t_lots:
        if lots.empty:
            st.caption("No closed lots yet.")
        else:
            _show(lots.drop(columns=["Close Date"]).sort_values("Sell Date", ascending=False),
                  money=["P&L ₹"], plain=["Qty", "Buy ₹", "Sell ₹"])
            st.caption("Each sell is matched to the oldest open buy (FIFO). Intraday = bought and sold the same day.")
    with t_day:
        _show(daily, money=["Realized ₹", "Net Realized ₹"], plain=["STT ₹", "Other ₹"])
        st.caption("Realized ₹ is booked on the day the closing trade happened; charges are those incurred that day.")
    st.caption("💸 Charges are Zerodha equity estimates (NSE, not the contract note): qty bought and sold the same day "
               "is intraday — STT 0.025% on the sell, brokerage 0.03%, no DP charge; the rest is delivery — "
               "STT 0.1% on buy and sell, zero brokerage, DP charge on the sell. Also counted: exchange "
               "0.00322%, SEBI ₹10/crore, stamp duty, 18% GST. STT is not rounded and the ₹20/order brokerage cap "
               "is not applied.")
