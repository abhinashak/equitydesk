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
    total = q + v + t + tech
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


def _render_portfolio_matrix(portfolio_tickers: list[str], df_matrix: pd.DataFrame, key_prefix: str = "pm"):
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
        # Recommendation column
        row["Rec"] = _recommendation(row["Q"], row["V"], row["T"], row["Tech"])
        # Gate columns
        for lbl, pos_c, neg_c in GATE_COLS:
            row[lbl] = _gate_cell(bool(r.get(pos_c, False)), bool(r.get(neg_c, False)))
        records.append(row)

    disp = pd.DataFrame(records).set_index("Ticker")

    # Styler — colour the score columns
    _REC_COLOUR = {
        "⭐ Strong Buy":  "background-color:rgba(22,163,74,0.25);font-weight:600",
        "✅ Buy":         "background-color:rgba(22,163,74,0.13);font-weight:600",
        "🟡 Accumulate": "background-color:rgba(202,138,4,0.15)",
        "👀 Watch":       "background-color:rgba(148,163,184,0.18)",
        "🔴 Overvalued":  "background-color:rgba(220,38,38,0.13)",
        "❌ Avoid":       "background-color:rgba(220,38,38,0.20);font-weight:600",
    }

    def _style(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for short, _ in SCORE_COLS:
            if short in df.columns:
                styles[short] = df[short].apply(
                    lambda v: _score_colour(int(v)) if str(v).lstrip("-").isdigit() else ""
                )
        if "Rec" in df.columns:
            styles["Rec"] = df["Rec"].apply(lambda v: _REC_COLOUR.get(v, ""))
        return styles

    st.dataframe(
        disp.style.apply(_style, axis=None),
        use_container_width=True,
        height=min(80 + len(records) * 38, 520),
    )
    st.caption(
        "🟩 PASS &nbsp; 🟥 FAIL &nbsp; ⬜ N/A &nbsp;|&nbsp; "
        "Q = Quality · V = Valuation · T = Timing · Tech = Technical · Rec = Recommendation &nbsp;|&nbsp; "
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
                    key=f"{key_prefix}_btn_{tkr}",
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
    with st.expander("📂 Portfolio Gate Matrix", expanded=bool(portfolio_tkrs)):
        _tab_holdings, _tab_manual, _tab_file = st.tabs(
            ["📋 Current Holdings", "✏️ Manual", "📂 File Upload"]
        )

        with _tab_holdings:
            _render_portfolio_matrix(portfolio_tkrs, df_matrix, key_prefix="pm_holdings")

        with _tab_manual:
            st.markdown(
                "<div style='font-size:0.88rem;color:#475569;margin-bottom:6px'>"
                "Enter ticker symbols separated by commas or newlines.</div>",
                unsafe_allow_html=True,
            )
            _man_text = st.text_area(
                "Tickers",
                placeholder="e.g.  RELIANCE, INFY, TCS",
                height=100,
                key="gates_scan_manual_text",
                label_visibility="collapsed",
            )
            _man_raw = [
                t.strip().upper().replace(".NS", "")
                for t in _man_text.replace("\n", ",").split(",")
                if t.strip()
            ]
            _man_valid   = [t for t in _man_raw if t in ticker_list]
            _man_invalid = [t for t in _man_raw if t and t not in ticker_list]

            if _man_raw:
                _gs = "background:#dcfce7;color:#166534;border-radius:4px;padding:2px 7px;margin:2px;display:inline-block"
                _rs = "background:#fee2e2;color:#991b1b;border-radius:4px;padding:2px 7px;margin:2px;display:inline-block"
                _vp = "  ".join(f"<span style='{_gs}'>{t}</span>" for t in _man_valid)
                _ip = "  ".join(f"<span style='{_rs}'>{t} ✗</span>" for t in _man_invalid)
                st.markdown(
                    f"<div style='font-size:0.82rem;margin-bottom:4px'>{_vp}{_ip}</div>",
                    unsafe_allow_html=True,
                )
                if _man_invalid:
                    st.caption(f"⚠️ {len(_man_invalid)} ticker(s) not found in database and will be skipped.")

            if st.button(
                    f"▶ Proceed  ({len(_man_valid)} ticker{'s' if len(_man_valid) != 1 else ''})",
                    key="gates_scan_manual_run",
                    disabled=not _man_valid,
                    type="primary",
            ):
                st.session_state["_gates_scan_manual_confirmed"] = _man_valid

            if st.session_state.get("_gates_scan_manual_confirmed"):
                _render_portfolio_matrix(
                    st.session_state["_gates_scan_manual_confirmed"],
                    df_matrix,
                    key_prefix="pm_manual",
                )

        with _tab_file:
            st.markdown(
                "<div style='font-size:0.88rem;color:#475569;margin-bottom:6px'>"
                "Upload a CSV or TXT file with a <code>ticker</code> column "
                "(or one ticker per line). <code>.NS</code> suffix is stripped automatically.</div>",
                unsafe_allow_html=True,
            )
            _uploaded = st.file_uploader(
                "Upload file",
                type=["csv", "txt"],
                key="gates_scan_file_upload",
                label_visibility="collapsed",
            )
            _file_raw: list[str] = []
            if _uploaded is not None:
                try:
                    import io as _io
                    _text = _uploaded.read().decode("utf-8", errors="replace")
                    try:
                        _fdf = pd.read_csv(_io.StringIO(_text))
                        _fdf.columns = [c.strip().lower() for c in _fdf.columns]
                        _col = "ticker" if "ticker" in _fdf.columns else _fdf.columns[0]
                        _file_raw = (
                            _fdf[_col].dropna().astype(str)
                            .str.strip().str.upper()
                            .str.replace(r"\.NS$", "", regex=True).tolist()
                        )
                    except Exception:
                        _file_raw = [
                            ln.strip().upper().replace(".NS", "")
                            for ln in _text.splitlines() if ln.strip()
                        ]
                except Exception as _e:
                    st.error(f"Could not read file: {_e}")

            _file_valid   = [t for t in _file_raw if t in ticker_list]
            _file_invalid = [t for t in _file_raw if t and t not in ticker_list]

            if _file_raw:
                _gs = "background:#dcfce7;color:#166534;border-radius:4px;padding:2px 7px;margin:2px;display:inline-block"
                _rs = "background:#fee2e2;color:#991b1b;border-radius:4px;padding:2px 7px;margin:2px;display:inline-block"
                _vp = "  ".join(f"<span style='{_gs}'>{t}</span>" for t in _file_valid)
                _ip = "  ".join(f"<span style='{_rs}'>{t} ✗</span>" for t in _file_invalid)
                st.markdown(
                    f"<div style='font-size:0.82rem;margin-bottom:4px'>{_vp}{_ip}</div>",
                    unsafe_allow_html=True,
                )
                if _file_invalid:
                    st.caption(f"⚠️ {len(_file_invalid)} ticker(s) not found in database and will be skipped.")

            if st.button(
                    f"▶ Proceed  ({len(_file_valid)} ticker{'s' if len(_file_valid) != 1 else ''})",
                    key="gates_scan_file_run",
                    disabled=not _file_valid,
                    type="primary",
            ):
                st.session_state["_gates_scan_file_confirmed"] = _file_valid

            if st.session_state.get("_gates_scan_file_confirmed"):
                _render_portfolio_matrix(
                    st.session_state["_gates_scan_file_confirmed"],
                    df_matrix,
                    key_prefix="pm_file",
                )


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

        # Filter
        _bz = df_matrix.copy()
        _q_col   = "Quality Score"
        _v_col   = "Valuation Score"
        _t_col   = "Timing Score"
        _tec_col = "Technical Score"

        for _c in [_q_col, _v_col, _t_col, _tec_col]:
            _bz[_c] = pd.to_numeric(_bz[_c], errors="coerce").fillna(0)

        _bz_filtered = _bz[
            (_bz[_q_col]   >= 70) &
            (_bz[_v_col]   >= 40) &
            ((_bz[_t_col]  >= 40) | (_bz[_tec_col] >= 40))
            ].copy()

        if _bz_filtered.empty:
            st.info("No stocks currently meet all Buy Zone criteria.")
        else:
            # Sort: Strong Buy first, then by Q+V desc
            _bz_filtered["_total"] = (
                    _bz_filtered[_q_col] + _bz_filtered[_v_col] +
                    _bz_filtered[_t_col] + _bz_filtered[_tec_col]
            )
            _bz_filtered = _bz_filtered.sort_values("_total", ascending=False)

            # Build display records
            _bz_records = []
            for _, _r in _bz_filtered.iterrows():
                _q   = int(_r[_q_col])
                _v   = int(_r[_v_col])
                _t   = int(_r[_t_col])
                _tec = int(_r[_tec_col])
                _rec = _recommendation(_q, _v, _t, _tec)
                _row = {
                    "Ticker":  _r["Ticker"],
                    "Company": str(_r.get("Company", _r["Ticker"]))[:22],
                    "Q":   _q,
                    "V":   _v,
                    "T":   _t,
                    "Tech": _tec,
                    "Rec": _rec,
                }
                for _lbl, _pos_c, _neg_c in GATE_COLS:
                    _row[_lbl] = _gate_cell(bool(_r.get(_pos_c, False)), bool(_r.get(_neg_c, False)))
                _bz_records.append(_row)

            _bz_disp = pd.DataFrame(_bz_records).set_index("Ticker")

            _BZ_REC_COLOUR = {
                "⭐ Strong Buy":  "background-color:rgba(22,163,74,0.25);font-weight:600",
                "✅ Buy":         "background-color:rgba(22,163,74,0.13);font-weight:600",
                "🟡 Accumulate": "background-color:rgba(202,138,4,0.15)",
            }

            def _bz_style(df):
                styles = pd.DataFrame("", index=df.index, columns=df.columns)
                for _sc in ["Q", "V", "T", "Tech"]:
                    if _sc in df.columns:
                        styles[_sc] = df[_sc].apply(
                            lambda v: _score_colour(int(v)) if str(v).lstrip("-").isdigit() else ""
                        )
                if "Rec" in df.columns:
                    styles["Rec"] = df["Rec"].apply(lambda v: _BZ_REC_COLOUR.get(v, ""))
                return styles

            st.markdown(
                f"<div style='font-size:0.9rem;font-weight:600;color:#166534;"
                f"margin-bottom:6px'>✅ {len(_bz_records)} stock(s) in Buy Zone</div>",
                unsafe_allow_html=True,
            )

            st.dataframe(
                _bz_disp.style.apply(_bz_style, axis=None),
                use_container_width=True,
                height=min(80 + len(_bz_records) * 38, 600),
            )
            st.caption(
                "🟩 PASS &nbsp; 🟥 FAIL &nbsp; ⬜ N/A &nbsp;|&nbsp; "
                "Sorted by combined score (Q+V+T+Tech) descending. "
                "Q = Quality · V = Valuation · T = Timing · Tech = Technical"
            )

            # Quick-select strip
            st.markdown(
                "<div style='font-size:0.78rem;color:#64748b;margin:6px 0 4px'>Quick select:</div>",
                unsafe_allow_html=True,
            )
            _bz_btn_cols = st.columns(min(len(_bz_records), 8))
            for _i, _rec in enumerate(_bz_records):
                _tkr = _rec["Ticker"]
                with _bz_btn_cols[_i % 8]:
                    if st.button(
                            _tkr,
                            key=f"bz_btn_{_tkr}",
                            help=f"{_rec['Company']} · Q:{_rec['Q']} V:{_rec['V']} T:{_rec['T']} Tech:{_rec['Tech']} · {_rec['Rec']}",
                            use_container_width=True,
                    ):
                        st.session_state["selected_asset"] = _tkr
                        st.session_state["_gates_source"]  = "all"
                        st.rerun()

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