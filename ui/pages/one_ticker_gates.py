"""
one_ticker_gates.py
────────────────────
Gate diagnostic panel for the Single-Ticker Deep Dive.
Standalone — runs its own data pipeline via one_ticker_common.py.

Run: streamlit run one_ticker_gates.py
"""

import pandas as pd
import streamlit as st
from ui.pages.one_ticker_doc import GATE_LOGIC_MD
from ui.pages.one_ticker_common import load_matrix

# ==============================================================================
# 1. PAGE CONFIG & STYLING
# ==============================================================================
def render():
    st.markdown("""
    <style>
        .block-container {
            max-width: 100% !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            padding-top: 1rem !important;
            padding-bottom: 1.5rem !important;
        }
        .status-box {
            background-color: #f4f6f7; border-radius: 4px;
            padding: 10px; font-family: monospace; font-size: 12px;
        }
        .gate-block {
            background: #fafbfc; border: 1px solid #e2e8f0;
            border-radius: 6px; padding: 12px; margin-bottom: 8px;
        }
        .section-divider { border-top: 2px solid #e2e8f0; margin: 20px 0 12px 0; }
        div[data-testid="metric-container"] { background: #f8fafc; border-radius: 6px; padding: 8px 12px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎯 Quality-on-Dip Intelligence Engine")
    st.markdown("Identifies fundamentally strong businesses currently trading at a valuation discount — separating business quality from price noise.")

    # ==============================================================================
    # 2. INLINE CONTROLS
    # ==============================================================================
    ctrl_col, info_col = st.columns([1, 5])
    with ctrl_col:
        min_roce = st.slider("Min ROCE %", 0.0, 30.0, 0.0, step=1.0)
    dip_mode = False

    # ==============================================================================
    # 3. LOAD DATA (self-contained)
    # ==============================================================================
    df_matrix, conn, env_status = load_matrix()

    with info_col:
        st.caption(f"Data source: **{'Real Parquet' if env_status == 'REAL' else 'Simulation Sandbox'}**")

    if df_matrix.empty:
        st.error("❌ No data returned. Check your parquet paths or ROCE filter.")
        st.stop()

    # Store in session_state so one_ticker_charts.py benefits if running in the same session
    st.session_state["df_matrix"] = df_matrix
    st.session_state["conn"]      = conn
    st.session_state["dip_mode"]  = dip_mode

    # ==============================================================================
    # 4. PRE-FLIGHT DIAGNOSTICS
    # ==============================================================================
    st.markdown("### 🔍 Pre-Flight Data Diagnostics")
    diag_cols = st.columns(4)

    with diag_cols[0]:
        try:
            cnt = conn.execute("SELECT COUNT(DISTINCT Ticker) FROM ticker_prices").fetchone()[0]
            st.markdown(f"<div class='status-box'><b>ticker_prices</b><br>Unique Tickers: {cnt}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"ticker_prices: {e}")

    with diag_cols[1]:
        try:
            cnt  = conn.execute("SELECT COUNT(DISTINCT ticker) FROM general_info").fetchone()[0]
            raw  = conn.execute("SELECT COUNT(*) FROM general_info").fetchone()[0]
            dupes = raw - cnt
            st.markdown(f"<div class='status-box'><b>general_info</b><br>Unique: {cnt} | Dupes removed: {dupes}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"general_info: {e}")

    with diag_cols[2]:
        try:
            cnt = conn.execute("SELECT COUNT(DISTINCT ticker) FROM quarterly_results").fetchone()[0]
            st.markdown(f"<div class='status-box'><b>quarterly_results</b><br>Tickers: {cnt}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"quarterly_results: {e}")

    with diag_cols[3]:
        try:
            cnt = conn.execute("""
                SELECT COUNT(*) FROM (SELECT DISTINCT Ticker FROM ticker_prices) t
                JOIN (SELECT DISTINCT ticker FROM general_info) g
                  ON t.Ticker = g.ticker OR REPLACE(t.Ticker,'.NS','') = g.ticker
            """).fetchone()[0]
            st.markdown(f"<div class='status-box'><b>Cross-Source Alignment</b><br>Matched Tickers: {cnt}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Alignment check: {e}")

    # ==============================================================================
    # 5. TICKER SELECTOR
    # ==============================================================================
    st.markdown("---")
    st.markdown("### 🔍 Single-Ticker Gate Breakdown")

    ticker_list = df_matrix['Ticker'].tolist()
    saved       = st.session_state.get("selected_asset")
    default_idx = ticker_list.index(saved) if saved in ticker_list else 0

    selected = st.selectbox(
        "Select stock for diagnostic breakdown:",
        ticker_list, index=default_idx, key="selected_asset",
    )

    r = df_matrix[df_matrix['Ticker'] == selected].iloc[0]

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        f"**{r['Company']}** &nbsp;|&nbsp; Sector: `{r['Sector']}` &nbsp;|&nbsp; "
        f"Stock P/E: `{r['Stock P/E']}` &nbsp;|&nbsp; Sector P/E: `{r['Sector P/E']}` &nbsp;|&nbsp; "
        f"PE Discount: `{r['PE Discount %']:+.1f}%` &nbsp;|&nbsp; "
        f"ROCE: `{r['ROCE %']}%` &nbsp;|&nbsp; D/E: `{r['D/E']}` &nbsp;|&nbsp; "
        f"Promoter: `{r['Promoter %']}%`",
        unsafe_allow_html=True,
    )

    # ==============================================================================
    # 6. GATE PANELS
    # ==============================================================================
    def gate_icon(pos, neg):
        if pos:  return "🟩 PASS"
        if neg:  return "🟥 FAIL"
        return "⬜ —"

    def prose(lbl, p, n, note):
        st.markdown(
            f"<div class='gate-block'><b>{lbl}</b><br>{gate_icon(p, n)}<br><small>{note}</small></div>",
            unsafe_allow_html=True,
        )

    col_q, col_v, col_t, col_tech = st.columns(4)

    with col_q:
        st.markdown("**🏛 Gate 1 — Business Quality**")
        prose("1a. ROCE",          r['G1a(+) ROCE'],    r['G1a(-) ROCE'],    f"ROCE = {r['ROCE %']}% (target ≥18%)")
        prose("1b. Free Cash Flow",r['G1b(+) FCF'],     r['G1b(-) FCF'],     f"FCF = {'positive' if r['G1b(+) FCF'] else 'negative/unknown'}")
        prose("1c. Profit CAGR",   r['G1c(+) ProfCAGR'],r['G1c(-) ProfCAGR'],f"3Y CAGR = {r['3Y Profit CAGR%']}% (target ≥15%)")
        prose("1d. CFO vs NP",     r['G1d(+) CFO>NP'],  r['G1d(-) CFO<NP'],  "CFO > Net Profit = earnings are real cash")
        _de_note = "N/A for financial sector" if any(fs.lower() in str(r['Sector']).lower() for fs in ['bank','finance','nbfc','insurance']) else f"D/E = {r['D/E']} (target <0.5)"
        prose("1e. Debt/Equity",   r['G1e(+) LowDebt'], r['G1e(-) HiDebt'],  _de_note)

    with col_v:
        st.markdown("**💰 Gate 2 — Valuation**")
        prose("2a. P/E vs Sector", r['G2a(+) PE<Sector'],r['G2a(-) PE>Sector'],f"Stock P/E {r['Stock P/E']} vs Sector {r['Sector P/E']} → {r['PE Discount %']:+.1f}%")
        prose("2b. PEG Ratio",     r['G2b(+) PEG<1'],   r['G2b(-) PEG>2.5'], f"PEG = {r['PEG']} (target <1.0 = cheap for growth)")
        prose("2c. Price/Book",    r['G2c(+) PB<3'],    r['G2c(-) PB>8'],    "P/B < 3 = not egregiously expensive")

    with col_t:
        st.markdown("**📈 Gate 3 — Timing**")
        prose("3a. Sales Momentum",  r['G3a(+) SalesAcc'],r['G3a(-) SalesDec'],f"Q0={r['_sales_q0']}Cr → Q1={r['_sales_q1']}Cr")
        prose("3b. Profit Accel",    r['G3b(+) ProfAcc'], r['G3b(-) ProfDec'], "Quarterly net profit trend")
        prose("3c. OPM Expansion",   r['G3c(+) OPMup'],   r['G3c(-) OPMsqz'], f"OPM: {r['_opm_q0']}% vs {r['_opm_q1']}%")
        prose("3d. Promoter Buying", r['G3d(+) Promoter'],r['G3d(-) PromoSell'],f"{r['_promoter']}% now vs {r['_promoter_prev']}% prev")
        prose("3e. Inst. Accum.",    r['G3e(+) InstAcc'], r['G3e(-) InstExit'], f"FII {r['_fii']}%→{r['_fii_prev']}% | DII {r['_dii']}%→{r['_dii_prev']}%")

    with col_tech:
        st.markdown("**📊 Gate 4 — Technical**")
        prose("4a. BB Squeeze",   r['G4a(+) BBsqz'],   r['G4a(-) BBchaos'],  f"BB width={round(r['_bb_width'],3) if pd.notna(r['_bb_width']) else '—'} (min63d={round(r['_min_bb_63d'],3) if pd.notna(r['_min_bb_63d']) else '—'})")
        prose("4b. MA50 Zone",    r['G4b(+) MA50zone'],r['G4b(-) MA50break'],f"Dist from 50DMA = {round(r['_dist_ma50'],1) if pd.notna(r['_dist_ma50']) else '—'}%")
        label_4c = "Dip Window (10–40% off high)" if dip_mode else "Near 52W High"
        prose(f"4c. {label_4c}",  r['G4c(+) Dip/High'],r['G4c(-) Crash'],    f"From 52W high = {round(r['_pct_52w'],1) if pd.notna(r['_pct_52w']) else '—'}%")
        prose("4d. Volume Conf.", r['G4d(+) VolConf'], r['G4d(-) VolDry'],    f"Vol ratio = {round(r['_vol_ratio'],2) if pd.notna(r['_vol_ratio']) else '—'}x")

    # ==============================================================================
    # 7. GATE LOGIC REFERENCE
    # ==============================================================================
    with st.expander("📖 Gate Logic Reference — How All Four Scores Are Computed"):
        st.markdown(GATE_LOGIC_MD)
