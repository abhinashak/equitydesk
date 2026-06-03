"""
dcf_analysis.py
───────────────
Intrinsic Value / DCF Analysis page.

Layout
──────
1. Portfolio DCF Scan   — loops every holding, shows Undervalued / Overvalued cards
                          clicking a card sets the active ticker and scrolls to detail
2. Stock Selector       — portfolio dropdown + all-stocks dropdown
3. DCF Detail           — sliders, verdict, four charts, sensitivity table
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.pages.one_ticker_common import COLORS, load_matrix


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL DATA HELPERS  (safe to call before render() enters)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def _get_net_profit(tkr: str, _conn) -> pd.DataFrame:
    try:
        df = _conn.execute(f"""
            SELECT dt, val FROM profit_loss
            WHERE ticker = '{tkr}' AND metric = 'Net Profit'
            ORDER BY dt
        """).df()
        df["dt"] = pd.to_datetime(df["dt"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def _get_eps(tkr: str, _conn) -> pd.DataFrame:
    try:
        df = _conn.execute(f"""
            SELECT dt, val FROM profit_loss
            WHERE ticker = '{tkr}' AND metric = 'EPS in Rs'
            ORDER BY dt
        """).df()
        df["dt"] = pd.to_datetime(df["dt"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def _get_price_history(tkr: str, _conn) -> pd.DataFrame:
    try:
        df = _conn.execute(f"""
            SELECT Date AS dt, Close AS price
            FROM ticker_prices
            WHERE REPLACE(Ticker, '.NS', '') = '{tkr}'
            ORDER BY Date
        """).df()
        df["dt"] = pd.to_datetime(df["dt"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def _get_growth_rates(tkr: str, _conn) -> dict:
    rates = {}
    try:
        df = _conn.execute(f"""
            SELECT horizon, val FROM compounded_profit_growth
            WHERE ticker = '{tkr}'
        """).df()
        for _, r in df.iterrows():
            rates[r["horizon"]] = float(r["val"])
    except Exception:
        pass
    return rates


@st.cache_data(ttl=300)
def _get_general_info(tkr: str, _conn) -> dict:
    try:
        row = _conn.execute(f"""
            SELECT current_price, stock_p_e, book_value, dividend_yield, roe, roce
            FROM general_info WHERE ticker = '{tkr}' LIMIT 1
        """).df()
        if not row.empty:
            return row.iloc[0].to_dict()
    except Exception:
        pass
    return {}


@st.cache_data(ttl=300)
def _get_shares_outstanding(tkr: str, _conn) -> float | None:
    try:
        eq = _conn.execute(f"""
            SELECT val FROM balance_sheet
            WHERE ticker = '{tkr}' AND metric = 'Equity Capital'
            ORDER BY dt DESC LIMIT 1
        """).df()
        fv = _conn.execute(f"""
            SELECT face_value FROM general_info WHERE ticker = '{tkr}' LIMIT 1
        """).df()
        if not eq.empty and not fv.empty:
            equity_cr = float(eq.iloc[0]["val"])
            face      = float(fv.iloc[0]["face_value"]) or 10.0
            shares    = equity_cr * 1e7 / face
            return shares / 1e7          # in crores
    except Exception:
        pass
    return None


def _run_dcf(base_profit: float, phase1: float, phase2: float,
             terminal: float, discount: float) -> float:
    """Two-stage DCF on Net Profit (₹ Cr). Returns total PV in ₹ Cr."""
    pv   = 0.0
    proj = base_profit
    for yr in range(1, 11):
        g    = phase1 if yr <= 5 else phase2
        proj = proj * (1 + g / 100)
        pv  += proj / (1 + discount / 100) ** yr
    if discount > terminal:
        tv  = proj * (1 + terminal / 100) / (discount / 100 - terminal / 100)
        pv += tv / (1 + discount / 100) ** 10
    return pv


def _quick_dcf(tkr: str, conn, dr: float = 12.0, tg: float = 5.0) -> dict | None:
    """
    Lightweight DCF for portfolio scan — uses historical 3Y growth as Phase 1,
    5Y as Phase 2. Returns dict with iv_base, current_price, premium_pct, or None.
    """
    np_df  = _get_net_profit(tkr, conn)
    if np_df.empty:
        return None
    np_ann = (
        np_df.set_index("dt")["val"]
        .resample("YE").sum()
        .reset_index()
        .rename(columns={"dt": "year", "val": "net_profit"})
    )
    np_ann = np_ann[np_ann["net_profit"] > 0].tail(10)
    if np_ann.empty:
        return None
    latest_profit = float(np_ann["net_profit"].iloc[-1])

    growth = _get_growth_rates(tkr, conn)
    g1 = growth.get("3 Years", growth.get("3Y", 12.0))
    g2 = growth.get("5 Years", growth.get("5Y", 10.0))
    g1 = max(g1, 0); g2 = max(min(g2, g1), 0)

    shares_cr = _get_shares_outstanding(tkr, conn)
    if shares_cr is None:
        eps_df = _get_eps(tkr, conn)
        if not eps_df.empty and latest_profit > 0:
            eps_ann = eps_df.set_index("dt")["val"].resample("YE").mean().reset_index()
            if not eps_ann.empty:
                latest_eps = float(eps_ann["val"].iloc[-1])
                if latest_eps > 0:
                    shares_cr = latest_profit / latest_eps
    if not shares_cr or shares_cr <= 0:
        return None

    dcf_val  = _run_dcf(latest_profit, g1, g2, tg, dr)
    iv_base  = dcf_val / shares_cr

    info     = _get_general_info(tkr, conn)
    price_df = _get_price_history(tkr, conn)
    cur_price = info.get("current_price") or (
        float(price_df["price"].iloc[-1]) if not price_df.empty else None
    )
    if not cur_price:
        return None

    premium = (cur_price - iv_base) / iv_base * 100
    return {
        "ticker":       tkr,
        "iv_base":      iv_base,
        "current_price": cur_price,
        "premium_pct":  premium,
        "g1": g1, "g2": g2,
        "company":      tkr,   # fallback; overwritten below if available
    }


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO SCAN SECTION
# ══════════════════════════════════════════════════════════════════════════════

def _render_portfolio_scan(portfolio_tickers: list[str], conn, df_matrix: pd.DataFrame):
    """Loops portfolio tickers, runs quick DCF, renders two card rows."""

    if not portfolio_tickers:
        st.info("No portfolio holdings found in session. Connect your broker or add holdings.")
        return

    # Company name lookup from matrix
    name_map = (
        df_matrix.set_index("Ticker")["Company"].to_dict()
        if "Company" in df_matrix.columns else {}
    )

    with st.spinner(f"Running DCF scan on {len(portfolio_tickers)} holdings…"):
        results = []
        for tkr in portfolio_tickers:
            r = _quick_dcf(tkr, conn)
            if r:
                r["company"] = name_map.get(tkr, tkr)
                results.append(r)

    if not results:
        st.warning("Could not compute DCF for any portfolio holding (missing profit data).")
        return

    undervalued = sorted([r for r in results if r["premium_pct"] < -20],
                         key=lambda x: x["premium_pct"])
    overvalued  = sorted([r for r in results if r["premium_pct"] >  20],
                         key=lambda x: x["premium_pct"], reverse=True)
    fair        = [r for r in results if -20 <= r["premium_pct"] <= 20]

    def _card_row(items, bg, border, icon):
        cols = st.columns(min(len(items), 4))
        for i, r in enumerate(items):
            with cols[i % 4]:
                discount_lbl = (
                    f"{abs(r['premium_pct']):.1f}% below DCF"
                    if r['premium_pct'] < 0
                    else f"{r['premium_pct']:.1f}% above DCF"
                )
                clicked = st.button(
                    f"{icon} **{r['ticker']}**\n\n"
                    f"₹{r['current_price']:,.0f}  →  DCF ₹{r['iv_base']:,.0f}\n\n"
                    f"_{discount_lbl}_",
                    key=f"port_card_{r['ticker']}",
                    use_container_width=True,
                    help=f"Click to open full DCF detail for {r['company']}",
                )
                if clicked:
                    st.session_state["dcf_active_ticker"] = r["ticker"]
                    st.session_state["_dcf_source"]       = "portfolio"
                    st.rerun()

    if undervalued:
        st.markdown(
            f"<div style='margin:6px 0 4px;font-size:1rem;font-weight:700;"
            f"color:#16a34a'>🟢 Undervalued  "
            f"<span style='font-weight:400;font-size:0.85rem;color:#64748b'>"
            f"trading below DCF intrinsic value by >20%</span></div>",
            unsafe_allow_html=True,
        )
        _card_row(undervalued, "#f0fdf4", "#16a34a", "🟢")

    if fair:
        st.markdown(
            f"<div style='margin:14px 0 4px;font-size:1rem;font-weight:700;"
            f"color:#ca8a04'>🟡 Fairly Valued  "
            f"<span style='font-weight:400;font-size:0.85rem;color:#64748b'>"
            f"within ±20% of DCF intrinsic value</span></div>",
            unsafe_allow_html=True,
        )
        _card_row(fair, "#fefce8", "#ca8a04", "🟡")

    if overvalued:
        st.markdown(
            f"<div style='margin:14px 0 4px;font-size:1rem;font-weight:700;"
            f"color:#dc2626'>🔴 Overvalued  "
            f"<span style='font-weight:400;font-size:0.85rem;color:#64748b'>"
            f"trading above DCF intrinsic value by >20%</span></div>",
            unsafe_allow_html=True,
        )
        _card_row(overvalued, "#fef2f2", "#dc2626", "🔴")

    skipped = len(portfolio_tickers) - len(results)
    if skipped:
        st.caption(f"⚠️ {skipped} holding(s) skipped — no profit history available.")


# ══════════════════════════════════════════════════════════════════════════════
# DCF DETAIL SECTION
# ══════════════════════════════════════════════════════════════════════════════

def _render_dcf_detail(ticker: str, conn, df_matrix: pd.DataFrame):
    np_df    = _get_net_profit(ticker, conn)
    eps_df   = _get_eps(ticker, conn)
    price_df = _get_price_history(ticker, conn)
    growth   = _get_growth_rates(ticker, conn)
    info     = _get_general_info(ticker, conn)

    current_price = info.get("current_price") or (
        float(price_df["price"].iloc[-1]) if not price_df.empty else None
    )

    if np_df.empty:
        st.warning(f"No Net Profit data found for {ticker}.")
        return

    np_annual = (
        np_df.set_index("dt")["val"]
        .resample("YE").sum()
        .reset_index()
        .rename(columns={"dt": "year", "val": "net_profit"})
    )
    np_annual = np_annual[np_annual["net_profit"] > 0].tail(10)
    if np_annual.empty:
        st.warning(f"Insufficient positive profit data for {ticker}.")
        return

    latest_profit = float(np_annual["net_profit"].iloc[-1])

    # ── Reset sliders when ticker changes ────────────────────────────────────
    if st.session_state.get("_dcf_detail_ticker") != ticker:
        for k in ["g1", "g2", "tg", "dr", "mos"]:
            st.session_state.pop(k, None)
        st.session_state["_dcf_detail_ticker"] = ticker

    # ── Assumptions ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⚙️ DCF Assumptions")

    _3y_raw  = growth.get("3 Years", growth.get("3Y",  None))
    _5y_raw  = growth.get("5 Years", growth.get("5Y",  None))
    hist_3y  = _3y_raw  if _3y_raw  is not None else 12.0
    hist_5y  = _5y_raw  if _5y_raw  is not None else 10.0
    hist_10y = growth.get("10 Years", growth.get("10Y", 8.0))

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown("**Phase 1 Growth** *(yr 1–5)*")
        st.caption(f"Historical 3Y: {hist_3y:.1f}%" if _3y_raw else f"Default: {hist_3y:.1f}% (no data)")
        g1 = st.slider("", 0.0, 40.0, float(round(hist_3y, 1)), step=0.5, key="g1", label_visibility="collapsed")
    with c2:
        st.markdown("**Phase 2 Growth** *(yr 6–10)*")
        st.caption(f"Historical 5Y: {hist_5y:.1f}%" if _5y_raw else f"Default: {hist_5y:.1f}% (no data)")
        g2 = st.slider("", 0.0, 30.0, float(round(min(hist_5y, g1), 1)), step=0.5, key="g2", label_visibility="collapsed")
    with c3:
        st.markdown("**Terminal Growth**")
        st.caption("Long-run perpetuity rate")
        tg = st.slider("", 2.0, 8.0, 5.0, step=0.5, key="tg", label_visibility="collapsed")
    with c4:
        st.markdown("**Discount Rate (WACC)**")
        st.caption("Risk-free ~7% + premium")
        dr = st.slider("", 8.0, 20.0, 12.0, step=0.5, key="dr", label_visibility="collapsed")
    with c5:
        st.markdown("**Margin of Safety**")
        st.caption("Buy below intrinsic by X%")
        mos = st.slider("", 0, 50, 25, step=5, key="mos", label_visibility="collapsed")

    # ── DCF calc ─────────────────────────────────────────────────────────────
    shares_cr = _get_shares_outstanding(ticker, conn)
    if shares_cr is None and not eps_df.empty and latest_profit > 0:
        eps_annual = eps_df.set_index("dt")["val"].resample("YE").mean().reset_index()
        if not eps_annual.empty:
            latest_eps = float(eps_annual["val"].iloc[-1])
            if latest_eps > 0:
                shares_cr = latest_profit / latest_eps

    dcf_base = _run_dcf(latest_profit, g1,       g2,       tg, dr)
    dcf_bull = _run_dcf(latest_profit, g1 * 1.3, g2 * 1.3, tg, max(dr - 1, tg + 0.5))
    dcf_bear = _run_dcf(latest_profit, g1 * 0.6, g2 * 0.6, tg, dr + 1)

    def to_ps(val_cr):
        return (val_cr / shares_cr) if (shares_cr and shares_cr > 0) else None

    iv_base  = to_ps(dcf_base)
    iv_bull  = to_ps(dcf_bull)
    iv_bear  = to_ps(dcf_bear)
    mos_price = iv_base * (1 - mos / 100) if iv_base else None

    # ── Verdict ──────────────────────────────────────────────────────────────
    st.markdown("---")
    if iv_base and current_price:
        premium = (current_price - iv_base) / iv_base * 100
        if premium > 20:
            vcls, vicon = "verdict-over",  "🔴"
            vtxt = f"OVERVALUED by {premium:.1f}%"
            vsub = (f"Current ₹{current_price:,.1f} trades at a {premium:.1f}% premium "
                    f"to base DCF ₹{iv_base:,.1f}. Market pricing in higher growth than assumed.")
        elif premium < -20:
            vcls, vicon = "verdict-under", "🟢"
            vtxt = f"UNDERVALUED by {abs(premium):.1f}%"
            vsub = (f"Current ₹{current_price:,.1f} trades at a {abs(premium):.1f}% discount "
                    f"to base DCF ₹{iv_base:,.1f}. MoS price: ₹{mos_price:,.1f}.")
        else:
            vcls, vicon = "verdict-fair",  "🟡"
            vtxt = f"FAIRLY VALUED ({premium:+.1f}%)"
            vsub = (f"Current ₹{current_price:,.1f} is within ±20% of base DCF ₹{iv_base:,.1f}.")

        v1, v2, v3, v4, v5 = st.columns(5)
        v1.metric("Base DCF / share", f"₹{iv_base:,.1f}"   if iv_base   else "—")
        v2.metric("Bull DCF / share", f"₹{iv_bull:,.1f}"   if iv_bull   else "—")
        v3.metric("Bear DCF / share", f"₹{iv_bear:,.1f}"   if iv_bear   else "—")
        v4.metric("MoS Price",        f"₹{mos_price:,.1f}" if mos_price else "—", f"{mos}% below base")
        v5.metric("Current Price",    f"₹{current_price:,.1f}", f"{premium:+.1f}% vs base", delta_color="inverse")

        st.markdown(f"""
        <div class="{vcls}">
            <div class="verdict-title">{vicon} {vtxt}</div>
            <div class="verdict-sub">{vsub}</div>
        </div>
        """, unsafe_allow_html=True)
    elif not shares_cr:
        st.info("⚠️ Could not derive shares outstanding — showing total DCF value in ₹ Cr.")
        v1, v2, v3 = st.columns(3)
        v1.metric("Base DCF (₹ Cr)", f"₹{dcf_base:,.0f}")
        v2.metric("Bull DCF (₹ Cr)", f"₹{dcf_bull:,.0f}")
        v3.metric("Bear DCF (₹ Cr)", f"₹{dcf_bear:,.0f}")

    # ── Chart 1: Price vs DCF band ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📈 Price vs DCF Intrinsic Value Band")
    if not price_df.empty and iv_base:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[price_df["dt"].iloc[0], price_df["dt"].iloc[-1]], y=[iv_bull, iv_bull],
            mode="lines", name=f"Bull DCF ₹{iv_bull:,.0f}",
            line=dict(color="#16a34a", width=1.5, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=[price_df["dt"].iloc[0], price_df["dt"].iloc[-1]], y=[iv_bear, iv_bear],
            mode="lines", name=f"Bear DCF ₹{iv_bear:,.0f}",
            line=dict(color="#dc2626", width=1.5, dash="dot"),
            fill="tonexty", fillcolor="rgba(22,163,74,0.07)",
        ))
        fig.add_trace(go.Scatter(
            x=[price_df["dt"].iloc[0], price_df["dt"].iloc[-1]], y=[iv_base, iv_base],
            mode="lines", name=f"Base DCF ₹{iv_base:,.0f}",
            line=dict(color="#2563eb", width=2, dash="dash"),
        ))
        if mos_price:
            fig.add_trace(go.Scatter(
                x=[price_df["dt"].iloc[0], price_df["dt"].iloc[-1]], y=[mos_price, mos_price],
                mode="lines", name=f"MoS ₹{mos_price:,.0f}",
                line=dict(color="#d97706", width=1.5, dash="longdash"),
            ))
        fig.add_trace(go.Scatter(
            x=price_df["dt"], y=price_df["price"],
            mode="lines", name=f"{ticker} Price",
            line=dict(color="#1e293b", width=2),
        ))
        fig.update_layout(
            height=400, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.08),
            xaxis=dict(showgrid=False),
            yaxis=dict(title="Price (₹)", showgrid=True, gridcolor="#f0f0f0"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Shaded band = Bear–Bull DCF range. Dashed blue = Base DCF.")

    # ── Chart 2: Value composition ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🧱 Value Composition")
    if iv_base:
        pv1, pv2 = 0.0, 0.0
        proj = latest_profit
        for yr in range(1, 6):
            proj = proj * (1 + g1 / 100)
            pv1 += proj / (1 + dr / 100) ** yr
        for yr in range(6, 11):
            proj = proj * (1 + g2 / 100)
            pv2 += proj / (1 + dr / 100) ** yr
        tv    = proj * (1 + tg / 100) / (dr / 100 - tg / 100) if dr > tg else 0
        pv_tv = tv / (1 + dr / 100) ** 10
        total = pv1 + pv2 + pv_tv or 1
        pcts  = [pv1 / total * 100, pv2 / total * 100, pv_tv / total * 100]
        fig_wf = go.Figure(go.Bar(
            x=["Phase 1 (Yr 1–5)", "Phase 2 (Yr 6–10)", "Terminal Value"],
            y=[round(p, 1) for p in pcts],
            marker_color=["#2563eb", "#0891b2", "#6b7280"],
            text=[f"{p:.1f}%" for p in pcts], textposition="outside",
        ))
        fig_wf.update_layout(
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(title="% of Total DCF Value", showgrid=False, range=[0, 105]),
            xaxis=dict(showgrid=False),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_wf, use_container_width=True)
        st.caption("Terminal Value > 70% → DCF highly sensitive to terminal growth rate.")

    # ── Chart 3: Historical + projected profit ────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 Historical Net Profit + DCF Projection")
    if not np_annual.empty:
        proj_years  = list(range(np_annual["year"].iloc[-1].year + 1,
                                 np_annual["year"].iloc[-1].year + 11))
        proj_profit, p = [], latest_profit
        for i in range(10):
            p = p * (1 + (g1 if i < 5 else g2) / 100)
            proj_profit.append(p)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=[y.year for y in np_annual["year"]], y=np_annual["net_profit"],
            name="Historical", marker_color=COLORS[0],
        ))
        fig2.add_trace(go.Bar(
            x=proj_years, y=proj_profit, name="Projected (Base)",
            marker_color="rgba(37,99,235,0.3)", marker_line=dict(color=COLORS[0], width=1.5),
        ))
        fig2.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0), bargap=0.2,
            yaxis=dict(title="Net Profit (₹ Cr)", showgrid=True, gridcolor="#f0f0f0"),
            xaxis=dict(showgrid=False), legend=dict(orientation="h", y=1.08),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Chart 4: Sensitivity table ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🎯 Sensitivity — Intrinsic Value per Share")
    if iv_base and shares_cr:
        rows_s = []
        for g in [g1 - 4, g1 - 2, g1, g1 + 2, g1 + 4]:
            row = {}
            for d in [dr - 2, dr - 1, dr, dr + 1, dr + 2]:
                if d <= tg:
                    row[f"DR {d:.1f}%"] = "N/A"
                else:
                    val = _run_dcf(latest_profit, max(g, 0), max(g * 0.7, 0), tg, d)
                    ps  = val / shares_cr
                    row[f"DR {d:.1f}%"] = f"₹{ps:,.0f}"
            rows_s.append({"Growth →": f"{g:.1f}%", **row})
        sens_df = pd.DataFrame(rows_s).set_index("Growth →")

        def _colour(val):
            if not isinstance(val, str) or val in ("N/A", "—"):
                return ""
            try:
                v = float(val.replace("₹", "").replace(",", ""))
                if current_price and v < current_price * 0.8:
                    return "background-color: rgba(220,38,38,0.12)"
                elif current_price and v > current_price * 1.2:
                    return "background-color: rgba(22,163,74,0.12)"
                return "background-color: rgba(202,138,4,0.10)"
            except Exception:
                return ""

        st.dataframe(sens_df.style.map(_colour), use_container_width=True)
        st.caption(
            f"🟢 Green = DCF > 120% of current price.  "
            f"🔴 Red = DCF < 80% of current price.  "
            f"Current price: ₹{current_price:,.1f}"
        )

    # ── Key metrics ───────────────────────────────────────────────────────────
    if info:
        st.markdown("---")
        st.markdown("#### 📋 Key Fundamentals")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Current Price", f"₹{info.get('current_price', 0):,.1f}")
        m2.metric("P/E",           f"{info.get('stock_p_e', 0):.1f}x")
        m3.metric("Book Value",    f"₹{info.get('book_value', 0):,.1f}")
        m4.metric("Div Yield",     f"{info.get('dividend_yield', 0):.2f}%")
        m5.metric("ROE",           f"{info.get('roe', 0):.1f}%")
        m6.metric("ROCE",          f"{info.get('roce', 0):.1f}%")

    with st.expander("📖 DCF assumptions guide", expanded=False):
        st.markdown(f"""
**What is DCF?**
Discounted Cash Flow values a stock by projecting future profits and discounting them
back to today's money.

**Your current assumptions for {ticker}:**
| Parameter | Value | What it means |
|---|---|---|
| Phase 1 Growth | {g1}% | Annual profit growth for years 1–5 |
| Phase 2 Growth | {g2}% | Slowing growth for years 6–10 |
| Terminal Growth | {tg}% | Forever growth rate after year 10 |
| Discount Rate | {dr}% | Your required return (risk-free + premium) |
| Margin of Safety | {mos}% | How much discount you demand before buying |

**Rules of thumb:**
- Terminal Value > 70% → highly speculative; small changes in `tg` swing value wildly
- Discount rate: Large-cap Indian 10–12%, Small/mid-cap 13–15%
- Terminal growth should not exceed India's long-run GDP growth (~6–7%)
        """)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ══════════════════════════════════════════════════════════════════════════════

def render():
    st.set_page_config(layout="wide", page_title="DCF Analysis", page_icon="🔬")

    st.markdown("""
    <style>
    .block-container { max-width:100%!important; padding:1rem 1.5rem 2rem; }
    .verdict-over  { background:#fef2f2; border-left:4px solid #dc2626; padding:12px 16px; border-radius:4px; }
    .verdict-fair  { background:#fefce8; border-left:4px solid #ca8a04; padding:12px 16px; border-radius:4px; }
    .verdict-under { background:#f0fdf4; border-left:4px solid #16a34a; padding:12px 16px; border-radius:4px; }
    .verdict-title { font-size:1.1rem; font-weight:700; margin-bottom:4px; }
    .verdict-sub   { font-size:0.85rem; color:#475569; }
    div[data-testid="stButton"] button {
        text-align:left; white-space:pre-wrap; height:auto;
        padding:10px 12px; line-height:1.4;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    if "df_matrix" not in st.session_state or "conn" not in st.session_state:
        df_matrix, conn, _ = load_matrix()
        st.session_state["df_matrix"] = df_matrix
        st.session_state["conn"]      = conn
    else:
        df_matrix = st.session_state["df_matrix"]
        conn      = st.session_state["conn"]

    if df_matrix.empty:
        st.error("❌ No data available.")
        st.stop()

    ticker_list = sorted(df_matrix["Ticker"].tolist())

    # ── Resolve portfolio tickers ─────────────────────────────────────────────
    _portfolio_syms = []
    try:
        _all_holdings = st.session_state.get("all_holdings") or {}
        _excl = {s.upper() for s in (st.session_state.get("excluded_symbols") or [])}
        for _holdings in _all_holdings.values():
            for _h in _holdings:
                _sym = _h.get("tradingsymbol", "")
                if _sym and _sym.upper() not in _excl and _sym not in _portfolio_syms:
                    _portfolio_syms.append(_sym)
    except Exception:
        pass
    portfolio_tickers = [s for s in _portfolio_syms if s in ticker_list]

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("### 🔬 DCF — Intrinsic Value Analysis")

    # ── Portfolio scan (collapsible, open by default if holdings exist) ───────
    with st.expander(
            f"📂 Portfolio DCF Scan  —  {len(portfolio_tickers)} holdings",
            expanded=bool(portfolio_tickers),
    ):
        _render_portfolio_scan(portfolio_tickers, conn, df_matrix)

    # ── Stock selector ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔍 Detailed DCF Analysis")

    saved  = st.session_state.get("dcf_active_ticker") or st.session_state.get("selected_asset")
    _source = st.session_state.get("_dcf_source", "all")

    sel_col1, sel_col2, _ = st.columns([2, 2, 3])

    with sel_col1:
        port_opts     = portfolio_tickers if portfolio_tickers else ["— no portfolio —"]
        port_disabled = not bool(portfolio_tickers)
        port_default  = (portfolio_tickers.index(saved)
                         if saved in portfolio_tickers else 0)
        port_pick = st.selectbox(
            "📂 Portfolio",
            port_opts,
            index=port_default,
            disabled=port_disabled,
            key="dcf_ticker_portfolio",
        )

    with sel_col2:
        all_default = ticker_list.index(saved) if saved in ticker_list else 0
        all_pick = st.selectbox(
            "🔍 All Stocks",
            ticker_list,
            index=all_default,
            key="dcf_ticker_all",
        )

    # Detect which picker changed
    _prev_port = st.session_state.get("_dcf_prev_port")
    _prev_all  = st.session_state.get("_dcf_prev_all")

    if port_pick != _prev_port and not port_disabled and port_pick != "— no portfolio —":
        ticker  = port_pick
        _source = "portfolio"
    elif all_pick != _prev_all:
        ticker  = all_pick
        _source = "all"
    else:
        ticker = port_pick if (_source == "portfolio" and portfolio_tickers) else all_pick

    st.session_state["_dcf_source"]       = _source
    st.session_state["_dcf_prev_port"]    = port_pick
    st.session_state["_dcf_prev_all"]     = all_pick
    st.session_state["dcf_active_ticker"] = ticker

    # ── Detail ────────────────────────────────────────────────────────────────
    _render_dcf_detail(ticker, conn, df_matrix)