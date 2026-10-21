"""
ui/pages/page_buy_candidates.py  –  Screens › Buy Candidates

Renders the output of buy_candidates.sql (views: abinash_params,
abinash_screen, abinash_recommendations) as an interactive page:

    KPIs  ->  Shortlist table  ->  Stock drill-down  ->  Full screen  ->  SQL

Prerequisite views (built by the normal DB build): gate_results, fss_scored,
fss_book, ticker_momentum, v_dcf_valuation.
"""

import os
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
from bll import agent as ag

# Names the SQL script may be saved under (searched in this order)
_SQL_NAMES = ["buy_candidates.sql", "abinash_recommendations.sql"]
_SQL_DIRS  = ["sqls", "."]

# Scripts that build everything the shortlist depends on, in run order.
# Missing files are skipped; reorder here if your build order differs.
_PIPELINE = [
    "init_raw.sql",
    "init.sql",
    "quality_gates.sql",
    "dcf.sql",
    "algo_momentum.sql",
    "buy_candidates.sql",
]
# Objects buy_candidates.sql reads from
_PREREQS = ["gate_results", "fss_scored", "fss_book",
            "ticker_momentum", "v_dcf_valuation"]

_TIER_HELP = {
    "A": "3+ lenses pass and Quality ≥ strong threshold",
    "B": "3+ lenses pass",
    "C": "exactly 2 lenses pass (watchlist)",
}
_TIER_ICON = {"A": "🟢 A", "B": "🟡 B", "C": "⚪ C"}


# ══════════════════════════════════════════════════════════════════════════════
# Database / SQL plumbing
# ══════════════════════════════════════════════════════════════════════════════

def _find_sql() -> Path | None:
    base = Path(ag.BASE)
    for d in _SQL_DIRS:
        for n in _SQL_NAMES:
            p = base / d / n
            if p.exists():
                return p
    return None


@st.cache_resource(show_spinner=False)
def _get_con(db_path: str):
    """One shared DuckDB connection per server process (cursors are per-call)."""
    return duckdb.connect(db_path)


def _q(con, sql: str, params=None) -> pd.DataFrame:
    return con.cursor().execute(sql, params or []).df()


def _views_ready(con) -> bool:
    try:
        _q(con, "SELECT 1 FROM abinash_screen LIMIT 1")
        return True
    except Exception:
        return False


def _find_script(name: str) -> Path | None:
    base = Path(ag.BASE)
    for d in _SQL_DIRS:
        p = base / d / name
        if p.exists():
            return p
    return None


def _missing_prereqs(con) -> list[str]:
    missing = []
    for obj in _PREREQS:
        try:
            _q(con, f"SELECT 1 FROM {obj} LIMIT 1")
        except Exception:
            missing.append(obj)
    return missing


def _clean_script(text: str) -> str:
    """Drop DuckDB-CLI dot-commands (.mode, .output ...) that the Python API rejects."""
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("."))


def _run_scripts(con, names: list[str], status=None) -> None:
    """Run SQL scripts in order from ag.BASE; raise with the failing script's name."""
    cwd = os.getcwd()
    try:
        os.chdir(ag.BASE)  # relative parquet paths resolve from BASE
        for name in names:
            path = _find_script(name)
            if path is None:
                if status:
                    status.write(f"⏭ {name} – not found, skipped")
                continue
            if status:
                status.write(f"▶ {name}")
            try:
                con.execute(_clean_script(path.read_text(encoding="utf-8")))
            except Exception as e:
                raise RuntimeError(f"{name}: {type(e).__name__}: {str(e)[:400]}") from e
    finally:
        os.chdir(cwd)


def _build_views(con, full: bool = False) -> None:
    """Build the abinash_* views; run the whole pipeline if needed or asked."""
    if full or _missing_prereqs(con):
        names = _PIPELINE
    else:
        sql_path = _find_sql()
        if sql_path is None:
            raise FileNotFoundError(f"Could not find {' / '.join(_SQL_NAMES)} under {ag.BASE}")
        names = [sql_path.name]
    with st.status("Building views...", expanded=True) as status:
        try:
            _run_scripts(con, names, status)
        except Exception:
            status.update(label="Build failed", state="error")
            raise
        status.update(label="Views built", state="complete", expanded=False)


def _apply_params(con, p: dict) -> None:
    con.cursor().execute(
        """UPDATE abinash_params SET
               quality_min = ?, dcf_upside_min_pct = ?, momentum_tests_min = ?,
               rsi_lo = ?, rsi_hi = ?, lenses_min = ?""",
        [p["quality_min"], p["dcf_upside_min_pct"], p["momentum_tests_min"],
         p["rsi_lo"], p["rsi_hi"], p["lenses_min"]],
    )


def _load_screen(con) -> pd.DataFrame:
    return _q(con, "SELECT * FROM abinash_screen")


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(con) -> dict:
    sb = st.sidebar
    sb.header("🎚 Screen Thresholds")
    sb.caption("Defaults match the 'liberal' settings in the SQL. "
               "Changes re-run the screen instantly.")

    preset = sb.radio("Preset", ["Liberal (default)", "Stricter", "Custom"],
                      horizontal=True, key="bc_preset")
    presets = {
        "Liberal (default)": dict(quality_min=40, dcf_upside_min_pct=-10.0,
                                  momentum_tests_min=2, lenses_min=2),
        "Stricter":          dict(quality_min=60, dcf_upside_min_pct=0.0,
                                  momentum_tests_min=3, lenses_min=3),
    }
    base = presets.get(preset, presets["Liberal (default)"])
    custom = preset == "Custom"

    p = {}
    p["quality_min"] = sb.slider("Min quality score", 0, 100,
                                 base["quality_min"], 5, disabled=not custom and False,
                                 key=f"bc_q_{preset}")
    p["dcf_upside_min_pct"] = sb.slider(
        "DCF upside floor (%)", -50.0, 50.0, float(base["dcf_upside_min_pct"]), 5.0,
        help="-10 means price may be up to ~10% above DCF base value.",
        key=f"bc_d_{preset}")
    p["momentum_tests_min"] = sb.slider("Momentum tests needed (of 4)", 1, 4,
                                        base["momentum_tests_min"],
                                        key=f"bc_m_{preset}")
    p["rsi_lo"], p["rsi_hi"] = sb.slider("RSI band", 0, 100, (40, 75),
                                         key="bc_rsi")
    p["lenses_min"] = sb.slider("Lenses that must pass", 1, 4,
                                base["lenses_min"], key=f"bc_l_{preset}")

    sb.divider()
    sb.header("🔎 View Filters")
    return p


def render_view_filters(df: pd.DataFrame) -> dict:
    sb = st.sidebar
    f = {}
    f["tiers"] = sb.multiselect("Tier", ["A", "B", "C"], default=["A", "B", "C"],
                                format_func=lambda t: f"{t} – {_TIER_HELP[t]}")
    sectors = sorted(df["sector"].dropna().unique().tolist())
    f["sectors"] = sb.multiselect("Sector", sectors)
    f["min_conv"] = sb.slider("Min conviction", 0, 100, 0, 5)
    f["no_cautions"] = sb.checkbox("Hide stocks with cautions", value=False)
    f["search"] = sb.text_input("Search ticker / company").strip()
    return f


def apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    out = df
    if f["tiers"]:
        out = out[out["tier"].isin(f["tiers"])]
    if f["sectors"]:
        out = out[out["sector"].isin(f["sectors"])]
    out = out[out["conviction"].fillna(0) >= f["min_conv"]]
    if f["no_cautions"]:
        out = out[out["cautions"].isna()]
    if f["search"]:
        s = f["search"].lower()
        out = out[out["ticker"].str.lower().str.contains(s, na=False)
                  | out["company_name"].str.lower().str.contains(s, na=False)]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Sections
# ══════════════════════════════════════════════════════════════════════════════

def _kpis(screen: pd.DataFrame):
    sl = screen[screen["shortlisted"] == True]  # noqa: E712
    c = st.columns(6)
    c[0].metric("Universe", f"{len(screen):,}")
    c[1].metric("Shortlisted", f"{len(sl):,}",
                f"{len(sl) / max(len(screen), 1):.0%} of universe", delta_color="off")
    c[2].metric("Tier A", int((sl["tier"] == "A").sum()), help=_TIER_HELP["A"])
    c[3].metric("Tier B", int((sl["tier"] == "B").sum()), help=_TIER_HELP["B"])
    c[4].metric("Tier C", int((sl["tier"] == "C").sum()), help=_TIER_HELP["C"])
    c[5].metric("With DCF", int(screen["dcf_value"].notna().sum()))

    with st.expander("Lens pass-rates & data coverage"):
        rows = []
        for name, col in [("Quality", "l_quality"), ("Value", "l_value"),
                          ("Momentum", "l_momentum"), ("Algo", "l_algo")]:
            avail = int(screen[col].notna().sum())
            passed = int((screen[col] == True).sum())  # noqa: E712
            rows.append({"Lens": name, "Available": avail, "Passed": passed,
                         "Pass rate": passed / avail if avail else 0.0})
        lens_df = pd.DataFrame(rows)
        st.dataframe(
            lens_df, hide_index=True, use_container_width=True,
            column_config={"Pass rate": st.column_config.ProgressColumn(
                "Pass rate", min_value=0.0, max_value=1.0, format="%.0f%%")},
        )
        st.bar_chart(lens_df.set_index("Lens")[["Available", "Passed"]])


def _shortlist_tab(sl: pd.DataFrame):
    if sl.empty:
        st.info("No stocks match the current thresholds and filters.")
        return

    show = sl.copy()
    show["tier"] = show["tier"].map(_TIER_ICON)
    cols = ["tier", "ticker", "company_name", "sector", "conviction",
            "lenses_passed", "why", "cautions",
            "quality_score", "dcf_upside_pct", "mom_tests", "algo_composite",
            "rsi", "ret_6m", "current_price"]
    cols = [c for c in cols if c in show.columns]

    st.dataframe(
        show[cols], hide_index=True, use_container_width=True, height=520,
        column_config={
            "tier": st.column_config.TextColumn("Tier", width="small"),
            "ticker": "Ticker",
            "company_name": "Company",
            "sector": "Sector",
            "conviction": st.column_config.ProgressColumn(
                "Conviction", min_value=0, max_value=100, format="%.0f"),
            "lenses_passed": st.column_config.NumberColumn("Lenses", format="%d/4"),
            "why": st.column_config.TextColumn("Why it passed", width="large"),
            "cautions": st.column_config.TextColumn("⚠ Cautions", width="large"),
            "quality_score": st.column_config.NumberColumn("Quality", format="%d"),
            "dcf_upside_pct": st.column_config.NumberColumn("DCF upside", format="%+.1f%%"),
            "mom_tests": st.column_config.NumberColumn("Mom", format="%d/4"),
            "algo_composite": st.column_config.NumberColumn("Algo", format="%+.2f"),
            "rsi": st.column_config.NumberColumn("RSI", format="%.0f"),
            "ret_6m": st.column_config.NumberColumn("6M ret", format="%.1f%%"),
            "current_price": st.column_config.NumberColumn("CMP ₹", format="₹%.2f"),
        },
    )

    d1, d2 = st.columns([1, 5])
    d1.download_button(
        "⬇ Export CSV", sl.to_csv(index=False).encode(),
        file_name="abinash_recommendations.csv", mime="text/csv",
    )
    d2.caption("Conviction is a 0–100 sorting aid (average of the available "
               "lens scores), not a forecast.")


def _lens_badge(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "➖ n/a"
    return "✅ pass" if bool(v) else "❌ fail"


def _drilldown_tab(screen: pd.DataFrame, candidates: pd.DataFrame):
    pool = candidates if not candidates.empty else screen
    tickers = pool["ticker"].tolist()
    if not tickers:
        st.info("Nothing to inspect.")
        return

    pick_all = st.checkbox("Pick from whole universe (not just shortlist)", value=False)
    options = screen["ticker"].tolist() if pick_all else tickers
    ticker = st.selectbox("Ticker", options, key="bc_ticker")
    r = screen[screen["ticker"] == ticker].iloc[0]

    st.subheader(f"{r['company_name']}  ·  {ticker}")
    st.caption(f"{r['sector'] or '—'}  |  Gate: {r['gate_rec'] or '—'}"
               + (f"  |  Tier {r['tier']}" if pd.notna(r["tier"]) else "  |  Not shortlisted"))

    # lens verdicts
    lc = st.columns(4)
    lc[0].metric("1 · Quality", _lens_badge(r["l_quality"]), f"score {r['quality_score']:.0f}"
                 if pd.notna(r["quality_score"]) else None, delta_color="off")
    lc[1].metric("2 · Value", _lens_badge(r["l_value"]),
                 f"DCF {r['dcf_upside_pct']:+.0f}%" if pd.notna(r["dcf_upside_pct"])
                 else (f"val score {r['valuation_score']:.0f}"
                       if pd.notna(r["valuation_score"]) else None), delta_color="off")
    lc[2].metric("3 · Momentum", _lens_badge(r["l_momentum"]),
                 f"{r['mom_tests']:.0f}/4 tests" if pd.notna(r["mom_tests"]) else None,
                 delta_color="off")
    lc[3].metric("4 · Algo", _lens_badge(r["l_algo"]),
                 f"{r['algo_composite']:+.2f} ({r['algo_side'] or '–'})"
                 if pd.notna(r["algo_composite"]) else None, delta_color="off")

    if pd.notna(r["why"]):
        st.success(f"**Why:** {r['why']}")
    if pd.notna(r["cautions"]):
        st.warning(f"**Cautions:** {r['cautions']}")

    left, right = st.columns(2)

    with left:
        st.markdown("##### DCF vs price")
        if pd.notna(r["dcf_value"]):
            dcf = pd.DataFrame({
                "₹": [r["dcf_bear"], r["dcf_value"], r["dcf_bull"],
                      r["dcf_mos_price"], r["dcf_price"]],
            }, index=["Bear", "Base", "Bull", "MoS price", "Price"])
            st.bar_chart(dcf)
            st.caption(f"DCF verdict: **{r['dcf_verdict'] or '—'}**  ·  "
                       f"P/E verdict: **{r['dcf_pe_verdict'] or '—'}**"
                       + (f"  ·  Terminal value {r['dcf_tv_pct']:.0f}% of value"
                          if pd.notna(r["dcf_tv_pct"]) else ""))
        else:
            st.info("No DCF row for this ticker (valuation score used instead).")

    with right:
        st.markdown("##### Technicals")
        if pd.notna(r["mom_date"]):
            t1, t2, t3 = st.columns(3)
            t1.metric("RSI", f"{r['rsi']:.0f}" if pd.notna(r["rsi"]) else "—")
            t2.metric("vs 200DMA", f"{r['sma200_dist']:+.1f}%" if pd.notna(r["sma200_dist"]) else "—")
            t3.metric("vs 50DMA", f"{r['sma50_dist']:+.1f}%" if pd.notna(r["sma50_dist"]) else "—")
            t4, t5, t6 = st.columns(3)
            t4.metric("6M return", f"{r['ret_6m']:.1f}%" if pd.notna(r["ret_6m"]) else "—")
            t5.metric("1Y return", f"{r['ret_1y']:.1f}%" if pd.notna(r["ret_1y"]) else "—")
            t6.metric("MACD", "Bullish" if r["macd_bullish"] == 1 else "Bearish")
            st.caption(f"Latest technical row: {pd.to_datetime(r['mom_date']).date()}")
        else:
            st.info("No momentum data for this ticker.")

    st.markdown("##### Fundamentals")
    f = st.columns(6)
    f[0].metric("CMP", f"₹{r['current_price']:,.2f}" if pd.notna(r["current_price"]) else "—")
    f[1].metric("P/E", f"{r['stock_p_e']:.1f}" if pd.notna(r["stock_p_e"]) else "—")
    f[2].metric("ROCE", f"{r['roce']:.1f}" if pd.notna(r["roce"]) else "—")
    f[3].metric("D/E", f"{r['de']:.2f}" if pd.notna(r["de"]) else "—")
    f[4].metric("Profit CAGR", f"{r['profit_cagr']:.1f}" if pd.notna(r["profit_cagr"]) else "—")
    f[5].metric("P/E discount", f"{r['pe_discount_pct']:.0f}%" if pd.notna(r["pe_discount_pct"]) else "—")

    with st.expander("Raw row"):
        st.json({k: (None if pd.isna(v) else str(v)) for k, v in r.items()})


def _screen_tab(screen: pd.DataFrame):
    st.caption("Every ticker in the universe with all lens flags "
               "(abinash_screen).")
    only_short = st.toggle("Shortlisted only", value=False, key="bc_only_sl")
    df = screen[screen["shortlisted"] == True] if only_short else screen  # noqa: E712
    st.dataframe(df, hide_index=True, use_container_width=True, height=560)
    st.download_button("⬇ Export full screen CSV",
                       df.to_csv(index=False).encode(),
                       file_name="abinash_screen.csv", mime="text/csv")


def _sql_tab(con):
    p = _find_sql()
    st.caption(f"Script: `{p}`" if p else "Script not found.")
    st.markdown("**Run an ad-hoc query** against the screen:")
    default = "SELECT * FROM abinash_screen WHERE ticker = 'NATIONALUM'"
    q = st.text_area("SQL", value=default, height=100, key="bc_sql")
    if st.button("▶ Run", key="bc_run"):
        try:
            st.dataframe(_q(con, q), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"{type(e).__name__}: {str(e)[:400]}")
    if p:
        with st.expander("View buy_candidates.sql"):
            st.code(p.read_text(encoding="utf-8"), language="sql")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def render():
    st.header("🎯 Buy Candidates")
    st.caption("Liberal shortlist: Quality must pass, plus enough of Value · "
               "Momentum · Algo lenses. Use it to decide what deserves a closer look.")

    app_cfg = st.session_state.get("app_cfg", {})
    db_path = app_cfg.get("DB_PATH", str(ag.DB))

    try:
        con = _get_con(db_path)
    except Exception as e:
        st.error(f"Could not open the database: {e}")
        st.stop()

    top = st.columns([5, 1.6, 1.6, 1.2])
    top[0].caption(f"Database: `{db_path}`")
    if top[1].button("🔄 Rebuild shortlist", help="Re-run buy_candidates.sql only"):
        st.session_state["bc_build"] = "shortlist"
    if top[2].button("🏗 Rebuild all views", help="Run " + " → ".join(_PIPELINE)):
        st.session_state["bc_build"] = "full"
    if top[3].button("♻ Reload", help="Re-read the data"):
        st.rerun()

    # make sure the abinash_* views (and everything they read from) exist
    mode = st.session_state.pop("bc_build", None)
    if mode or not _views_ready(con):
        try:
            _build_views(con, full=(mode == "full"))
        except Exception as e:
            st.error(f"Could not build the shortlist.\n\n{e}")
            miss = _missing_prereqs(con)
            if miss:
                st.info("Still missing: " + ", ".join(f"`{m}`" for m in miss))
            st.stop()

    params = render_sidebar(con)
    try:
        _apply_params(con, params)
        screen = _load_screen(con)
    except Exception as e:
        st.error(f"Query failed: {type(e).__name__}: {str(e)[:500]}")
        st.stop()

    shortlist = screen[screen["shortlisted"] == True].copy()  # noqa: E712
    filters = render_view_filters(shortlist)
    view = apply_filters(shortlist, filters).sort_values(
        ["tier", "conviction", "ticker"], ascending=[True, False, True])

    _kpis(screen)
    st.divider()

    t1, t2, t3, t4 = st.tabs(
        [f"📋 Shortlist ({len(view)})", "🔬 Stock drill-down",
         "🗂 Full screen", "🧾 SQL"])
    with t1:
        _shortlist_tab(view)
    with t2:
        _drilldown_tab(screen, view)
    with t3:
        _screen_tab(screen)
    with t4:
        _sql_tab(con)
