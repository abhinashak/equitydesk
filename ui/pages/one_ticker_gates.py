"""
one_ticker_gates.py
────────────────────
Gate diagnostic panel for the Single-Ticker Deep Dive.

Layout
──────
1. Portfolio Gate Matrix  — table of all 4×5 gates for every holding
2. Buy Zone               — full-universe scan
3. Pre-flight diagnostics
4. Ticker selector        — portfolio dropdown + all-stocks dropdown
5. Single-ticker gate breakdown
6. Gate logic reference
"""

import html
import io

import pandas as pd
import streamlit as st

from ui.components.table_widget import table_widget
from ui.pages.one_ticker_doc import GATE_LOGIC_MD
from utils.db_utils import get_connection, load_ticker_scores, run_readonly_query

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
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

# "1a" → (pos_col, neg_col); lets the breakdown section reuse GATE_COLS.
_GATE_PAIR = {label[:2]: (pos, neg) for label, pos, neg in GATE_COLS}

SCORE_COLS = [
    ("Q", "Quality Score"),
    ("V", "Valuation Score"),
    ("T", "Timing Score"),
    ("Tech", "Technical Score"),
]

# Buy Zone == the two highest-conviction outcomes of _recommendation().
_BUY_ZONE_RECS = ("⭐ Strong Buy", "✅ Buy")

_GATE_TABLE_CONFIG = {
    "ticker": st.column_config.TextColumn("Ticker"),
    **{
        short: st.column_config.ProgressColumn(short, min_value=0, max_value=100, format="%d")
        for short, _ in SCORE_COLS
    },
}

_CHIP_BASE = "border-radius:4px;padding:2px 7px;margin:2px;display:inline-block;"
_CHIP_OK   = _CHIP_BASE + "background:#dcfce7;color:#166534;"
_CHIP_BAD  = _CHIP_BASE + "background:#fee2e2;color:#991b1b;"

# Pre-flight diagnostics: (title, SQL, formatter(row) -> text). Run via db_utils.
_DIAGNOSTICS = [
    ("ticker_prices",
     "SELECT COUNT(DISTINCT Ticker) AS cnt FROM ticker_prices",
     lambda r: f"Unique Tickers: {r['cnt']}"),
    ("general_info",
     "SELECT COUNT(DISTINCT ticker) AS cnt, COUNT(*) AS raw FROM general_info",
     lambda r: f"Unique: {r['cnt']} | Dupes removed: {r['raw'] - r['cnt']}"),
    ("quarterly_results",
     "SELECT COUNT(DISTINCT ticker) AS cnt FROM quarterly_results",
     lambda r: f"Tickers: {r['cnt']}"),
    ("Cross-Source Alignment",
     """
     SELECT COUNT(*) AS cnt FROM (SELECT DISTINCT Ticker FROM ticker_prices) t
     JOIN (SELECT DISTINCT ticker FROM general_info) g
       ON t.Ticker = g.ticker OR REPLACE(t.Ticker,'.NS','') = g.ticker
     """,
     lambda r: f"Matched Tickers: {r['cnt']}"),
]


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner=False)
def _cached_ticker_scores() -> pd.DataFrame:
    """Scored matrix, cached for 2 minutes (st.cache_data.clear() drops it)."""
    df_matrix, _, _ = load_ticker_scores()
    return df_matrix


def _clean_symbol(raw) -> str:
    return str(raw).strip().upper().removesuffix(".NS")


def _fmt(value, ndigits: int) -> str:
    return round(value, ndigits) if pd.notna(value) else "—"


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


def _gate_cell(pos, neg, verbose: bool = False) -> str:
    if pos:
        return "🟩 PASS" if verbose else "🟩"
    if neg:
        return "🟥 FAIL" if verbose else "🟥"
    return "⬜ —" if verbose else "⬜"


def _recommendation(q: int, v: int, t: int, tech: int) -> str:
    """
    Derive a one-line textual recommendation from the four gate scores.

    Priority ladder (highest conviction first):
      Strong Buy   — Quality ≥70 AND Valuation ≥70 AND (Timing OR Technical ≥60)
      Buy          — Quality ≥70 AND Valuation ≥40 AND at least one of T/Tech ≥40
      Accumulate   — Quality ≥70 but Valuation borderline (40–69), wait for better price
      Watch        — Quality ≥40 but Valuation or Timing not confirmed yet
      Avoid        — Quality <40 (fundamentals weak regardless of price)
      Overvalued   — Quality ≥70 but Valuation <40 (good business, wrong price)
    """
    if q >= 70 and v >= 70 and (t >= 60 or tech >= 60):
        return "⭐ Strong Buy"
    if q >= 70 and v >= 40 and (t >= 40 or tech >= 40):
        return "✅ Buy"
    if q >= 70 and v >= 40:
        return "🟡 Accumulate"
    if q >= 70 and v < 40:
        return "🔴 Overvalued"
    if q >= 40:
        return "👀 Watch"
    return "❌ Avoid"


def _score(row: pd.Series, col: str) -> int:
    value = pd.to_numeric(row.get(col), errors="coerce")
    return int(value) if pd.notna(value) else 0


def _gate_table(df_matrix: pd.DataFrame, tickers: list[str] | None = None) -> pd.DataFrame:
    """Flat display table (ticker, company, scores, recommendation, gate cells)."""
    rows = df_matrix if tickers is None else df_matrix[df_matrix["Ticker"].isin(tickers)]
    records = []
    for _, r in rows.iterrows():
        rec = {
            "ticker":  r["Ticker"],
            "Company": str(r.get("Company", r["Ticker"]))[:22],
        }
        for short, col in SCORE_COLS:
            rec[short] = _score(r, col)
        rec["Rec"] = _recommendation(rec["Q"], rec["V"], rec["T"], rec["Tech"])
        for lbl, pos_c, neg_c in GATE_COLS:
            rec[lbl] = _gate_cell(bool(r.get(pos_c, False)), bool(r.get(neg_c, False)))
        records.append(rec)
    return pd.DataFrame(records)


def _table_height(n_rows: int, cap: int) -> int:
    return min(80 + n_rows * 38, cap)


def _quick_select(table: pd.DataFrame, key_prefix: str, source: str, show_rec: bool = False):
    """Button strip beneath a gate table; clicking a ticker selects it below."""
    st.markdown(
        "<div style='font-size:0.78rem;color:#64748b;margin:6px 0 4px'>Quick select:</div>",
        unsafe_allow_html=True,
    )
    records = table.to_dict("records")
    btn_cols = st.columns(min(len(records), 8))
    for i, rec in enumerate(records):
        tkr  = rec["ticker"]
        help_txt = f"{rec['Company']} · Q:{rec['Q']} V:{rec['V']} T:{rec['T']} Tech:{rec['Tech']}"
        if show_rec:
            help_txt += f" · {rec['Rec']}"
        with btn_cols[i % 8]:
            if st.button(tkr, key=f"{key_prefix}_btn_{tkr}", help=help_txt, use_container_width=True):
                st.session_state["selected_asset"] = tkr
                st.session_state["_gates_source"]  = source
                st.rerun()


def _render_portfolio_matrix(portfolio_tickers: list[str], df_matrix: pd.DataFrame, key_prefix: str = "pm"):
    if not portfolio_tickers:
        st.info("No portfolio holdings matched the matrix. Connect your broker or add holdings.")
        return

    table = _gate_table(df_matrix, portfolio_tickers)
    if table.empty:
        st.warning("Holdings found but no matching rows in the gate matrix.")
        return

    table_widget(
        table,
        key=f"{key_prefix}_table",
        height=_table_height(len(table), 520),
        highlight_portfolio=False,        # every row here is already a holding
        column_config=_GATE_TABLE_CONFIG,
    )
    st.caption(
        "🟩 PASS &nbsp; 🟥 FAIL &nbsp; ⬜ N/A &nbsp;|&nbsp; "
        "Q = Quality · V = Valuation · T = Timing · Tech = Technical · Rec = Recommendation &nbsp;|&nbsp; "
        "Use the quick-select buttons to see the full breakdown below."
    )
    _quick_select(table, key_prefix, source="portfolio")


def _parse_uploaded_tickers(uploaded) -> list[str]:
    """Read tickers from an uploaded CSV (ticker column or first column) or one-per-line TXT."""
    try:
        text = uploaded.read().decode("utf-8", errors="replace")
    except Exception as exc:
        st.error(f"Could not read file: {exc}")
        return []
    try:
        fdf = pd.read_csv(io.StringIO(text))
        fdf.columns = [c.strip().lower() for c in fdf.columns]
        col = "ticker" if "ticker" in fdf.columns else fdf.columns[0]
        return [_clean_symbol(t) for t in fdf[col].dropna()]
    except Exception:
        return [_clean_symbol(ln) for ln in text.splitlines() if ln.strip()]


def _render_scan(raw: list[str], ticker_set: set[str], df_matrix: pd.DataFrame, name: str):
    """Validate typed/uploaded tickers, show chips, and render the matrix after 'Proceed'."""
    valid   = [t for t in raw if t in ticker_set]
    invalid = [t for t in raw if t not in ticker_set]

    if raw:
        ok_chips  = "  ".join(f"<span style='{_CHIP_OK}'>{html.escape(t)}</span>" for t in valid)
        bad_chips = "  ".join(f"<span style='{_CHIP_BAD}'>{html.escape(t)} ✗</span>" for t in invalid)
        st.markdown(
            f"<div style='font-size:0.82rem;margin-bottom:4px'>{ok_chips}{bad_chips}</div>",
            unsafe_allow_html=True,
        )
        if invalid:
            st.caption(f"⚠️ {len(invalid)} ticker(s) not found in database and will be skipped.")

    state_key = f"_gates_scan_{name}_confirmed"
    if st.button(
            f"▶ Proceed  ({len(valid)} ticker{'s' if len(valid) != 1 else ''})",
            key=f"gates_scan_{name}_run",
            disabled=not valid,
            type="primary",
    ):
        st.session_state[state_key] = valid

    if st.session_state.get(state_key):
        _render_portfolio_matrix(st.session_state[state_key], df_matrix, key_prefix=f"pm_{name}")


def _gate_block(r: pd.Series, code: str, title: str, note: str):
    pos, neg = _GATE_PAIR[code]
    st.markdown(
        f"<div class='gate-block'><b>{code}. {title}</b><br>"
        f"{_gate_cell(r[pos], r[neg], verbose=True)}<br><small>{note}</small></div>",
        unsafe_allow_html=True,
    )


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
    df_matrix = _cached_ticker_scores()

    with info_col:
        st.caption("Data source: **sqls/all.sql (DuckDB)**")

    if df_matrix.empty:
        st.error("❌ No data returned. Check your parquet paths or ROCE filter.")
        st.stop()

    st.session_state["df_matrix"] = df_matrix
    st.session_state["conn"]      = get_connection()
    st.session_state["dip_mode"]  = dip_mode

    ticker_list    = df_matrix["Ticker"].tolist()
    ticker_set     = set(ticker_list)
    portfolio_tkrs = _portfolio_tickers(ticker_list)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION A — Portfolio gate matrix
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("📂 Portfolio Gate Matrix", expanded=bool(portfolio_tkrs)):
        tab_holdings, tab_manual, tab_file = st.tabs(
            ["📋 Current Holdings", "✏️ Manual", "📂 File Upload"]
        )

        with tab_holdings:
            _render_portfolio_matrix(portfolio_tkrs, df_matrix, key_prefix="pm_holdings")

        with tab_manual:
            st.markdown(
                "<div style='font-size:0.88rem;color:#475569;margin-bottom:6px'>"
                "Enter ticker symbols separated by commas or newlines.</div>",
                unsafe_allow_html=True,
            )
            man_text = st.text_area(
                "Tickers",
                placeholder="e.g.  RELIANCE, INFY, TCS",
                height=100,
                key="gates_scan_manual_text",
                label_visibility="collapsed",
            )
            man_raw = [_clean_symbol(t) for t in man_text.replace("\n", ",").split(",") if t.strip()]
            _render_scan(man_raw, ticker_set, df_matrix, name="manual")

        with tab_file:
            st.markdown(
                "<div style='font-size:0.88rem;color:#475569;margin-bottom:6px'>"
                "Upload a CSV or TXT file with a <code>ticker</code> column "
                "(or one ticker per line). <code>.NS</code> suffix is stripped automatically.</div>",
                unsafe_allow_html=True,
            )
            uploaded = st.file_uploader(
                "Upload file",
                type=["csv", "txt"],
                key="gates_scan_file_upload",
                label_visibility="collapsed",
            )
            file_raw = _parse_uploaded_tickers(uploaded) if uploaded is not None else []
            _render_scan(file_raw, ticker_set, df_matrix, name="file")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION A2 — Buy Zone (full universe scan)
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("🛒 Buy Zone — Full Universe Scan", expanded=True):
        st.markdown(
            "<div style='font-size:0.88rem;color:#475569;margin-bottom:10px'>"
            "Stocks from the <b>complete ticker universe</b> meeting: "
            "<code>Quality ≥ 70</code> &nbsp;·&nbsp; "
            "<code>Valuation ≥ 40</code> &nbsp;·&nbsp; "
            "<code>Timing ≥ 40</code> <b>or</b> <code>Technical ≥ 40</code>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Same criteria as the Strong Buy / Buy tiers of _recommendation().
        bz = _gate_table(df_matrix)
        bz = bz[bz["Rec"].isin(_BUY_ZONE_RECS)]

        if bz.empty:
            st.info("No stocks currently meet all Buy Zone criteria.")
        else:
            bz = bz.iloc[(-bz[[s for s, _ in SCORE_COLS]].sum(axis=1)).argsort(kind="stable")]

            st.markdown(
                f"<div style='font-size:0.9rem;font-weight:600;color:#166534;"
                f"margin-bottom:6px'>✅ {len(bz)} stock(s) in Buy Zone</div>",
                unsafe_allow_html=True,
            )
            table_widget(
                bz,
                key="bz_table",
                height=_table_height(len(bz), 600),
                portfolio_syms=portfolio_tkrs,
                column_config=_GATE_TABLE_CONFIG,
            )
            st.caption(
                "🟩 PASS &nbsp; 🟥 FAIL &nbsp; ⬜ N/A &nbsp;|&nbsp; "
                "Sorted by combined score (Q+V+T+Tech) descending. "
                "Q = Quality · V = Valuation · T = Timing · Tech = Technical"
            )
            _quick_select(bz, "bz", source="all", show_rec=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION B — Pre-flight diagnostics
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("🔍 Pre-Flight Data Diagnostics", expanded=False):
        for col, (title, sql, fmt) in zip(st.columns(len(_DIAGNOSTICS)), _DIAGNOSTICS):
            with col:
                try:
                    row = run_readonly_query(sql).iloc[0]
                    st.markdown(
                        f"<div class='status-box'><b>{title}</b><br>{fmt(row)}</div>",
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"{title}: {e}")

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
    col_q, col_v, col_t, col_tech = st.columns(4)

    with col_q:
        st.markdown("**🏛 Gate 1 — Business Quality**")
        _gate_block(r, "1a", "ROCE",           f"ROCE = {r['ROCE %']}% (target ≥18%)")
        _gate_block(r, "1b", "Free Cash Flow", f"FCF = {'positive' if r['G1b(+) FCF'] else 'negative/unknown'}")
        _gate_block(r, "1c", "Profit CAGR",    f"3Y CAGR = {r['3Y Profit CAGR%']}% (target ≥15%)")
        _gate_block(r, "1d", "CFO vs NP",      "CFO > Net Profit = earnings are real cash")
        is_financial = any(fs in str(r["Sector"]).lower() for fs in ("bank", "finance", "nbfc", "insurance"))
        _gate_block(r, "1e", "Debt/Equity",
                    "N/A for financial sector" if is_financial else f"D/E = {r['D/E']} (target <0.5)")

    with col_v:
        st.markdown("**💰 Gate 2 — Valuation**")
        _gate_block(r, "2a", "P/E vs Sector",
                    f"Stock P/E {r['Stock P/E']} vs Sector {r['Sector P/E']} → {r['PE Discount %']:+.1f}%")
        _gate_block(r, "2b", "PEG Ratio",  f"PEG = {r['PEG']} (target <1.0)")
        _gate_block(r, "2c", "Price/Book", "P/B < 3 = not egregiously expensive")

    with col_t:
        st.markdown("**📈 Gate 3 — Timing**")
        _gate_block(r, "3a", "Sales Momentum",  f"Q0={r['_sales_q0']}Cr → Q1={r['_sales_q1']}Cr")
        _gate_block(r, "3b", "Profit Accel",    "Quarterly net profit trend")
        _gate_block(r, "3c", "OPM Expansion",   f"OPM: {r['_opm_q0']}% vs {r['_opm_q1']}%")
        _gate_block(r, "3d", "Promoter Buying", f"{r['_promoter']}% now vs {r['_promoter_prev']}% prev")
        _gate_block(r, "3e", "Inst. Accum.",
                    f"FII {r['_fii']}%→{r['_fii_prev']}% | DII {r['_dii']}%→{r['_dii_prev']}%")

    with col_tech:
        st.markdown("**📊 Gate 4 — Technical**")
        _gate_block(r, "4a", "BB Squeeze",
                    f"BB width={_fmt(r['_bb_width'], 3)} (min63d={_fmt(r['_min_bb_63d'], 3)})")
        _gate_block(r, "4b", "MA50 Zone", f"Dist from 50DMA = {_fmt(r['_dist_ma50'], 1)}%")
        _gate_block(r, "4c", "Dip Window (10–40% off high)" if dip_mode else "Near 52W High",
                    f"From 52W high = {_fmt(r['_pct_52w'], 1)}%")
        _gate_block(r, "4d", "Volume Conf.", f"Vol ratio = {_fmt(r['_vol_ratio'], 2)}x")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION E — Gate logic reference
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("📖 Gate Logic Reference — How All Four Scores Are Computed"):
        st.markdown(GATE_LOGIC_MD)