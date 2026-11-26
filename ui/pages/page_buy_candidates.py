"""
ui/pages/page_buy_candidates.py  –  Screens › Buy Candidates

Renders the output of buy_candidates.sql (views: abinash_params,
abinash_screen, abinash_recommendations) as an interactive page:

    KPIs  ->  Shortlist table

UI only. The shared in-memory DuckDB, the build pipeline (sqls/all.sql), query
execution and file lookups live in utils/db_utils.py and utils/file_utils.py;
tables and the table/line-chart toggle come from ui/components.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ui.components.table_chart import table_or_chart
from ui.components.table_widget import table_widget
from utils.db_utils import (
    DB_SQL_PATH,
    execute_sql,
    reset_connection,
    run_readonly_query,
    run_sql_file,
)
from utils.file_utils import find_first_existing

# Names the SQL script may be saved under (searched in this order)
_SQL_NAMES = ["buy_candidates.sql", "abinash_recommendations.sql"]
_SQL_DIRS = ["sqls", "."]

_TIER_HELP = {
    "A": "3+ lenses pass and Quality ≥ strong threshold",
    "B": "3+ lenses pass",
    "C": "exactly 2 lenses pass (watchlist)",
}
_TIER_ICON = {"A": "🟢 A", "B": "🟡 B", "C": "⚪ C"}

_SHORTLIST_COLS = [
    "tier", "ticker", "company_name", "sector", "conviction",
    "lenses_passed", "why", "cautions",
    "quality_score", "dcf_upside_pct", "mom_tests", "algo_composite",
    "rsi", "ret_6m", "current_price",
]
_SHORTLIST_CONFIG = {
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
}


# ══════════════════════════════════════════════════════════════════════════════
# Data access
# ══════════════════════════════════════════════════════════════════════════════

def _find_sql() -> str | None:
    return find_first_existing(_SQL_NAMES, _SQL_DIRS)


def _apply_params(p: dict) -> None:
    execute_sql(
        """UPDATE abinash_params SET
               quality_min = ?, dcf_upside_min_pct = ?, momentum_tests_min = ?,
               rsi_lo = ?, rsi_hi = ?, lenses_min = ?""",
        [p["quality_min"], p["dcf_upside_min_pct"], p["momentum_tests_min"],
         p["rsi_lo"], p["rsi_hi"], p["lenses_min"]],
    )


def _load_screen() -> pd.DataFrame:
    return run_readonly_query("SELECT * FROM abinash_screen")


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> dict:
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

    # Widget keys include the preset so picking a preset resets the sliders to its values.
    p = {}
    p["quality_min"] = sb.slider("Min quality score", 0, 100,
                                 base["quality_min"], 5, key=f"bc_q_{preset}")
    p["dcf_upside_min_pct"] = sb.slider(
        "DCF upside floor (%)", -50.0, 50.0, float(base["dcf_upside_min_pct"]), 5.0,
        help="-10 means price may be up to ~10% above DCF base value.",
        key=f"bc_d_{preset}")
    p["momentum_tests_min"] = sb.slider("Momentum tests needed (of 4)", 1, 4,
                                        base["momentum_tests_min"], key=f"bc_m_{preset}")
    p["rsi_lo"], p["rsi_hi"] = sb.slider("RSI band", 0, 100, (40, 75), key="bc_rsi")
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
        table_widget(
            lens_df, key="bc_lens", height=None, show_filters=False,
            highlight_portfolio=False,
            column_config={"Pass rate": st.column_config.ProgressColumn(
                "Pass rate", min_value=0.0, max_value=1.0, format="%.0f%%")},
        )
        st.bar_chart(lens_df.set_index("Lens")[["Available", "Passed"]])


def _shortlist(sl: pd.DataFrame):
    if sl.empty:
        st.info("No stocks match the current thresholds and filters.")
        return

    show = sl.copy()
    show["tier"] = show["tier"].map(_TIER_ICON)
    cols = [c for c in _SHORTLIST_COLS if c in show.columns]

    table_or_chart(
        show[cols], "bc_sl", height=520, column_config=_SHORTLIST_CONFIG,
        portfolio_syms=st.session_state.get("portfolio_syms"),   # highlights held tickers if set
    )

    d1, d2 = st.columns([1, 5])
    d1.download_button(
        "⬇ Export CSV", sl.to_csv(index=False).encode(),
        file_name="abinash_recommendations.csv", mime="text/csv",
    )
    d2.caption("Conviction is a 0–100 sorting aid (average of the available "
               "lens scores), not a forecast.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def render():
    st.header("🎯 Buy Candidates")
    st.caption("Liberal shortlist: Quality must pass, plus enough of Value · "
               "Momentum · Algo lenses. Use it to decide what deserves a closer look.")

    sql_path = _find_sql()

    top = st.columns([5, 1.6, 1.6, 1.2])
    top[0].caption(f"Database: in-memory, built from `{DB_SQL_PATH}`")
    if top[1].button("🔄 Rebuild shortlist", help=f"Re-run {Path(sql_path).name if sql_path else _SQL_NAMES[0]} only"):
        st.session_state["bc_build"] = "shortlist"
    if top[2].button("🏗 Rebuild all views", help=f"Re-run {DB_SQL_PATH}"):
        st.session_state["bc_build"] = "full"
    if top[3].button("♻ Reload", help="Re-read the data"):
        st.rerun()

    mode = st.session_state.pop("bc_build", None)
    if mode:
        try:
            with st.spinner("Rebuilding views…"):
                if mode == "full":
                    reset_connection()               # next query rebuilds everything from all.sql
                elif sql_path is None:
                    raise FileNotFoundError(f"Could not find {' / '.join(_SQL_NAMES)}")
                else:
                    run_sql_file(sql_path)
        except Exception as e:
            st.error(f"Could not rebuild the shortlist.\n\n{type(e).__name__}: {str(e)[:400]}")
            st.stop()

    params = render_sidebar()
    try:
        with st.spinner("Loading screen…"):          # the first query builds the database
            _apply_params(params)
            screen = _load_screen()
    except Exception as e:
        st.error(f"Query failed: {type(e).__name__}: {str(e)[:500]}")
        st.info(f"Check that `{DB_SQL_PATH}` builds `abinash_params` and `abinash_screen`.")
        st.stop()

    shortlist = screen[screen["shortlisted"] == True].copy()  # noqa: E712
    filters = render_view_filters(shortlist)
    view = apply_filters(shortlist, filters).sort_values(
        ["tier", "conviction", "ticker"], ascending=[True, False, True])

    _kpis(screen)
    st.divider()

    st.subheader(f"📋 Shortlist ({len(view)})")
    _shortlist(view)