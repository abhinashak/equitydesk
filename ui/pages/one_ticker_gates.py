"""
one_ticker_gates.py
────────────────────
Gate diagnostic panel for the Single-Ticker Deep Dive.

Layout
──────
1. Portfolio Gate Matrix  — heatmap table of all 4×5 gates for every holding
2. Pre-flight diagnostics
3. Ticker selector        — portfolio dropdown + all-stocks dropdown
4. Single-ticker gate breakdown
5. Gate logic reference
"""

import pandas as pd
import streamlit as st
from ui.pages.one_ticker_doc import GATE_LOGIC_MD
from ui.pages.one_ticker_common import load_matrix

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _portfolio_tickers(ticker_list: list[str]) -> list[str]:
    """Return holdings that exist in the matrix, respecting excluded_symbols."""
    syms = []
    try:
        all_holdings = st.session_state.get("all_holdings") or {}
        excl = {s.upper() for s in (st.session_state.get("excluded_symbols") or [])}
        for holdings in all_holdings.values():
            for h in holdings:
                sym = h.get("tradingsymbol", "")
                if sym and sym.upper() not in excl and sym not in syms:
                    syms.append(sym)
    except Exception:
        pass
    return [s for s in syms if s in ticker_list]


def _gate_cell(pos: bool, neg: bool) -> str:
    if pos:  return "🟩"
    if neg:  return "🟥"
    return "⬜"


def _score_colour(score: int) -> str:
    if score >= 70: return "background-color:rgba(22,163,74,0.18)"
    if score >= 40: return "background-color:rgba(202,138,4,0.15)"
    return "background-color:rgba(220,38,38,0.13)"


# ──────────────────────────────────────────────────────────────────────────────
# PORTFOLIO GATE MATRIX
# ──────────────────────────────────────────────────────────────────────────────

# Column definitions: (display_label, pos_col, neg_col)
GATE_COLS = [
    # Gate 1 — Quality
    ("1a ROCE",     "G1a(+) ROCE",        "G1a(-) ROCE"),
    ("1b FCF",      "G1b(+) FCF",         "G1b(-) FCF"),
    ("1c ProfCAGR", "G1c(+) ProfCAGR",    "G1c(-) ProfCAGR"),
    ("1d CFO>NP",   "G1d(+) CFO>NP",      "G1d(-) CFO<NP"),
    ("1e Debt",     "G1e(+) LowDebt",     "G1e(-) HiDebt"),
    # Gate 2 — Valuation
    ("2a PE",       "G2a(+) PE<Sector",   "G2a(-) PE>Sector"),
    ("2b PEG",      "G2b(+) PEG<1",       "G2b(-) PEG>2.5"),
    ("2c PB",       "G2c(+) PB<3",        "G2c(-) PB>8"),
    # Gate 3 — Timing
    ("3a Sales",    "G3a(+) SalesAcc",    "G3a(-) SalesDec"),
    ("3b Profit",   "G3b(+) ProfAcc",     "G3b(-) ProfDec"),
    ("3c OPM",      "G3c(+) OPMup",       "G3c(-) OPMsqz"),
    ("3d Promoter", "G3d(+) Promoter",    "G3d(-) PromoSell"),
    ("3e Inst",     "G3e(+) InstAcc",     "G3e(-) InstExit"),
    # Gate 4 — Technical
    ("4a BBsqz",    "G4a(+) BBsqz",       "G4a(-) BBchaos"),
    ("4b MA50",     "G4b(+) MA50zone",    "G4b(-) MA50break"),
    ("4c 52W",      "G4c(+) Dip/High",    "G4c(-) Crash"),
    ("4d Volume",   "G4d(+) VolConf",     "G4d(-) VolDry"),
]

SCORE_COLS = [
    ("Q", "Quality Score"),
    ("V", "Valuation Score"),
    ("T", "Timing Score"),
    ("Tech", "Technical Score"),
]


def _render_portfolio_matrix(portfolio_tickers: list[str], df_matrix: pd.DataFrame):
    if not portfolio_tickers:
        st.info("No portfolio holdings matched the matrix. Connect your broker or add holdings.")
        return

    rows = df_matrix[df_matrix["Ticker"].isin(portfolio_tickers)].copy()
    if rows.empty:
        st.warning("Holdings found but no matching rows in the gate matrix.")
        return

    # Build display dataframe
    records = []
    for _, r in rows.iterrows():
        row = {
            "Ticker":  r["Ticker"],
            "Company": str(r.get("Company", r["Ticker"]))[:22],
        }
        # Score columns
        for short, col in SCORE_COLS:
            row[short] = int(r[col]) if pd.notna(r.get(col)) else 0
        # Gate columns
        for lbl, pos_c, neg_c in GATE_COLS:
            row[lbl] = _gate_cell(bool(r.get(pos_c, False)), bool(r.get(neg_c, False)))
        records.append(row)

    disp = pd.DataFrame(records).set_index("Ticker")

    # Styler — colour the score columns
    def _style(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for short, _ in SCORE_COLS:
            if short in df.columns:
                styles[short] = df[short].apply(
                    lambda v: _score_colour(int(v)) if str(v).lstrip("-").isdigit() else ""
                )
        return styles

    st.dataframe(
        disp.style.apply(_style, axis=None),
        use_container_width=True,
        height=min(80 + len(records) * 38, 520),
    )
    st.caption(
        "🟩 PASS &nbsp; 🟥 FAIL &nbsp; ⬜ N/A &nbsp;|&nbsp; "
        "Q = Quality · V = Valuation · T = Timing · Tech = Technical &nbsp;|&nbsp; "
        "Click a row to see full breakdown below."
    )

    # Row-click → set ticker
    # Streamlit doesn't support native row-click on st.dataframe yet,
    # so we offer a quick-select strip of buttons beneath the table.
    st.markdown(
        "<div style='font-size:0.78rem;color:#64748b;margin:6px 0 4px'>Quick select:</div>",
        unsafe_allow_html=True,
    )
    btn_cols = st.columns(min(len(records), 8))
    for i, rec in enumerate(records):
        tkr = rec["Ticker"]
        qs  = rec["Q"]
        col = "#16a34a" if qs >= 70 else ("#ca8a04" if qs >= 40 else "#dc2626")
        with btn_cols[i % 8]:
            if st.button(
                    tkr,
                    key=f"pm_btn_{tkr}",
                    help=f"{rec['Company']} · Q:{rec['Q']} V:{rec['V']} T:{rec['T']} Tech:{rec['Tech']}",
                    use_container_width=True,
            ):
                st.session_state["selected_asset"]   = tkr
                st.session_state["_gates_source"]    = "portfolio"
                st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ──────────────────────────────────────────────────────────────────────────────

def render():
    st.markdown("""
    <style>
        .block-container {
            max-width: 100% !important;
            padding: 1rem 1.5rem 1.5rem;
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
        div[data-testid="metric-container"] {
            background: #f8fafc; border-radius: 6px; padding: 8px 12px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎯 Quality-on-Dip Intelligence Engine")
    st.markdown(
        "Identifies fundamentally strong businesses currently trading at a valuation "
        "discount — separating business quality from price noise."
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl_col, info_col = st.columns([1, 5])
    with ctrl_col:
        min_roce = st.slider("Min ROCE %", 0.0, 30.0, 0.0, step=1.0)
    dip_mode = False

    # ── Load data ─────────────────────────────────────────────────────────────
    df_matrix, conn, env_status = load_matrix()

    with info_col:
        st.caption(
            f"Data source: **{'Real Parquet' if env_status == 'REAL' else 'Simulation Sandbox'}**"
        )

    if df_matrix.empty:
        st.error("❌ No data returned. Check your parquet paths or ROCE filter.")
        st.stop()

    st.session_state["df_matrix"] = df_matrix
    st.session_state["conn"]      = conn
    st.session_state["dip_mode"]  = dip_mode

    ticker_list      = df_matrix["Ticker"].tolist()
    portfolio_tkrs   = _portfolio_tickers(ticker_list)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION A — Portfolio gate matrix
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander(
            f"📂 Portfolio Gate Matrix — {len(portfolio_tkrs)} holdings",
            expanded=bool(portfolio_tkrs),
    ):
        _render_portfolio_matrix(portfolio_tkrs, df_matrix)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION B — Pre-flight diagnostics
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("🔍 Pre-Flight Data Diagnostics", expanded=False):
        diag_cols = st.columns(4)
        with diag_cols[0]:
            try:
                cnt = conn.execute("SELECT COUNT(DISTINCT Ticker) FROM ticker_prices").fetchone()[0]
                st.markdown(f"<div class='status-box'><b>ticker_prices</b><br>Unique Tickers: {cnt}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"ticker_prices: {e}")
        with diag_cols[1]:
            try:
                cnt   = conn.execute("SELECT COUNT(DISTINCT ticker) FROM general_info").fetchone()[0]
                raw   = conn.execute("SELECT COUNT(*) FROM general_info").fetchone()[0]
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

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION C — Ticker selector (portfolio + all-stocks)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🔍 Single-Ticker Gate Breakdown")

    saved   = st.session_state.get("selected_asset")
    _source = st.session_state.get("_gates_source", "all")

    sel1, sel2, _ = st.columns([2, 2, 3])

    with sel1:
        port_opts     = portfolio_tkrs if portfolio_tkrs else ["— no portfolio —"]
        port_disabled = not bool(portfolio_tkrs)
        port_default  = portfolio_tkrs.index(saved) if saved in portfolio_tkrs else 0
        port_pick = st.selectbox(
            "📂 Portfolio",
            port_opts,
            index=port_default,
            disabled=port_disabled,
            key="gates_ticker_portfolio",
        )

    with sel2:
        all_default = ticker_list.index(saved) if saved in ticker_list else 0
        all_pick = st.selectbox(
            "🔍 All Stocks",
            ticker_list,
            index=all_default,
            key="gates_ticker_all",
        )

    # Detect which picker changed this render
    _prev_port = st.session_state.get("_gates_prev_port")
    _prev_all  = st.session_state.get("_gates_prev_all")

    if port_pick != _prev_port and not port_disabled and port_pick != "— no portfolio —":
        selected = port_pick
        _source  = "portfolio"
    elif all_pick != _prev_all:
        selected = all_pick
        _source  = "all"
    else:
        selected = port_pick if (_source == "portfolio" and portfolio_tkrs) else all_pick

    st.session_state["_gates_source"]    = _source
    st.session_state["_gates_prev_port"] = port_pick
    st.session_state["_gates_prev_all"]  = all_pick
    st.session_state["selected_asset"]   = selected

    # ── Row for selected ticker ───────────────────────────────────────────────
    r = df_matrix[df_matrix["Ticker"] == selected].iloc[0]

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        f"**{r['Company']}** &nbsp;|&nbsp; Sector: `{r['Sector']}` &nbsp;|&nbsp; "
        f"Stock P/E: `{r['Stock P/E']}` &nbsp;|&nbsp; Sector P/E: `{r['Sector P/E']}` &nbsp;|&nbsp; "
        f"PE Discount: `{r['PE Discount %']:+.1f}%` &nbsp;|&nbsp; "
        f"ROCE: `{r['ROCE %']}%` &nbsp;|&nbsp; D/E: `{r['D/E']}` &nbsp;|&nbsp; "
        f"Promoter: `{r['Promoter %']}%`",
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION D — Four gate columns
    # ══════════════════════════════════════════════════════════════════════════

    def gate_icon(pos, neg):
        if pos: return "🟩 PASS"
        if neg: return "🟥 FAIL"
        return "⬜ —"

    def prose(lbl, p, n, note):
        st.markdown(
            f"<div class='gate-block'><b>{lbl}</b><br>{gate_icon(p, n)}"
            f"<br><small>{note}</small></div>",
            unsafe_allow_html=True,
        )

    col_q, col_v, col_t, col_tech = st.columns(4)

    with col_q:
        st.markdown("**🏛 Gate 1 — Business Quality**")
        prose("1a. ROCE",           r["G1a(+) ROCE"],        r["G1a(-) ROCE"],        f"ROCE = {r['ROCE %']}% (target ≥18%)")
        prose("1b. Free Cash Flow", r["G1b(+) FCF"],         r["G1b(-) FCF"],         f"FCF = {'positive' if r['G1b(+) FCF'] else 'negative/unknown'}")
        prose("1c. Profit CAGR",    r["G1c(+) ProfCAGR"],    r["G1c(-) ProfCAGR"],    f"3Y CAGR = {r['3Y Profit CAGR%']}% (target ≥15%)")
        prose("1d. CFO vs NP",      r["G1d(+) CFO>NP"],      r["G1d(-) CFO<NP"],      "CFO > Net Profit = earnings are real cash")
        _de_note = (
            "N/A for financial sector"
            if any(fs.lower() in str(r["Sector"]).lower() for fs in ["bank","finance","nbfc","insurance"])
            else f"D/E = {r['D/E']} (target <0.5)"
        )
        prose("1e. Debt/Equity",    r["G1e(+) LowDebt"],     r["G1e(-) HiDebt"],      _de_note)

    with col_v:
        st.markdown("**💰 Gate 2 — Valuation**")
        prose("2a. P/E vs Sector",  r["G2a(+) PE<Sector"],   r["G2a(-) PE>Sector"],   f"Stock P/E {r['Stock P/E']} vs Sector {r['Sector P/E']} → {r['PE Discount %']:+.1f}%")
        prose("2b. PEG Ratio",      r["G2b(+) PEG<1"],       r["G2b(-) PEG>2.5"],     f"PEG = {r['PEG']} (target <1.0)")
        prose("2c. Price/Book",     r["G2c(+) PB<3"],        r["G2c(-) PB>8"],        "P/B < 3 = not egregiously expensive")

    with col_t:
        st.markdown("**📈 Gate 3 — Timing**")
        prose("3a. Sales Momentum", r["G3a(+) SalesAcc"],    r["G3a(-) SalesDec"],    f"Q0={r['_sales_q0']}Cr → Q1={r['_sales_q1']}Cr")
        prose("3b. Profit Accel",   r["G3b(+) ProfAcc"],     r["G3b(-) ProfDec"],     "Quarterly net profit trend")
        prose("3c. OPM Expansion",  r["G3c(+) OPMup"],       r["G3c(-) OPMsqz"],      f"OPM: {r['_opm_q0']}% vs {r['_opm_q1']}%")
        prose("3d. Promoter Buying",r["G3d(+) Promoter"],    r["G3d(-) PromoSell"],   f"{r['_promoter']}% now vs {r['_promoter_prev']}% prev")
        prose("3e. Inst. Accum.",   r["G3e(+) InstAcc"],     r["G3e(-) InstExit"],    f"FII {r['_fii']}%→{r['_fii_prev']}% | DII {r['_dii']}%→{r['_dii_prev']}%")

    with col_tech:
        st.markdown("**📊 Gate 4 — Technical**")
        prose("4a. BB Squeeze",     r["G4a(+) BBsqz"],       r["G4a(-) BBchaos"],     f"BB width={round(r['_bb_width'],3) if pd.notna(r['_bb_width']) else '—'} (min63d={round(r['_min_bb_63d'],3) if pd.notna(r['_min_bb_63d']) else '—'})")
        prose("4b. MA50 Zone",      r["G4b(+) MA50zone"],    r["G4b(-) MA50break"],   f"Dist from 50DMA = {round(r['_dist_ma50'],1) if pd.notna(r['_dist_ma50']) else '—'}%")
        label_4c = "Dip Window (10–40% off high)" if dip_mode else "Near 52W High"
        prose(f"4c. {label_4c}",    r["G4c(+) Dip/High"],    r["G4c(-) Crash"],       f"From 52W high = {round(r['_pct_52w'],1) if pd.notna(r['_pct_52w']) else '—'}%")
        prose("4d. Volume Conf.",   r["G4d(+) VolConf"],     r["G4d(-) VolDry"],      f"Vol ratio = {round(r['_vol_ratio'],2) if pd.notna(r['_vol_ratio']) else '—'}x")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION E — Gate logic reference
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("📖 Gate Logic Reference — How All Four Scores Are Computed"):
        st.markdown(GATE_LOGIC_MD)