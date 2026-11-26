"""
ui/pages/page_sql_lab.py
─────────────────────────────────────────────────
SQL Lab — Write, preview, and publish SQL screeners as dashboard panels.

This module is UI only: the SQL editor, dashboard and schema browser, plus the
Streamlit session state behind them. Everything else lives elsewhere:

  • DuckDB connection, schema inspection, read-only execution → utils/db_utils.py
    (the whole build pipeline is described by sqls/all.sql)
  • JSON persistence                                        → utils/file_utils.py
  • filterable / sortable table                             → ui/components/table_widget.py

Flow:
  1. A shared in-memory DuckDB is built once per server process from sqls/all.sql
     (db_utils.get_connection). "Rebuild DB" calls reset_connection(); the next
     query rebuilds it.
  2. Schema browser shows CREATE statements + live sample rows.
  3. Editor: write SQL → Run → preview result.
  4. Assign a visualisation type (Table / Line / Bar / Area).
  5. Publish → panel saved to sqls/panels.json and rendered in the Dashboard tab.
"""

from __future__ import annotations

import csv
import html
import io
import re
from datetime import datetime

import pandas as pd
import streamlit as st

from ui.components.table_widget import table_widget
from utils.db_utils import (
    DB_SQL_PATH,
    ddl_for_table,
    list_tables,
    quote_ident,
    reset_connection,
    run_sql,
    sample_rows,
)
from utils.file_utils import load_json_list, save_json_atomic

# ── Constants ──────────────────────────────────────────────────────────────────

PANELS_JSON_PATH = "../../sqls/panels.json"      # relative to this file

VIZ_TYPES: list[str] = ["Table", "Line Chart", "Bar Chart", "Area Chart"]
VIZ_ICONS: dict[str, str] = {
    "Table":      "📋",
    "Line Chart": "📈",
    "Bar Chart":  "📊",
    "Area Chart": "🌊",
}


# ── Database status ────────────────────────────────────────────────────────────

def _database_status() -> tuple[list[str], str | None]:
    """Return (tables, error). The first call builds the shared database from all.sql."""
    try:
        return list_tables(), None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


# ── Panel persistence ──────────────────────────────────────────────────────────

def _load_panels() -> list[dict]:
    return [p for p in load_json_list(PANELS_JSON_PATH, base_file=__file__)
            if isinstance(p, dict)]


def _save_panels(panels: list[dict]) -> None:
    save_json_atomic(panels, PANELS_JSON_PATH, base_file=__file__)


def _init_panels() -> None:
    if "sqllab_panels_loaded" not in st.session_state:
        st.session_state["sqllab_panels"] = _load_panels()
        st.session_state["sqllab_panels_loaded"] = True


def _publish_panel(name: str, sql: str, viz: str, df: pd.DataFrame) -> None:
    panels = [p for p in st.session_state["sqllab_panels"] if p["name"] != name]
    panels.append({
        "name":         name,
        "sql":          sql,
        "viz":          viz,
        "published_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "row_count":    len(df),
        "columns":      list(df.columns),
    })
    st.session_state["sqllab_panels"] = panels
    _save_panels(panels)


def _remove_panel(name: str) -> None:
    panels = [p for p in st.session_state["sqllab_panels"] if p["name"] != name]
    st.session_state["sqllab_panels"] = panels
    _save_panels(panels)


# ── Portfolio helpers ──────────────────────────────────────────────────────────

def _load_portfolio_syms() -> list[str]:
    """Return deduplicated list of portfolio ticker symbols from session state."""
    syms: list[str] = []
    try:
        all_holdings = st.session_state.get("all_holdings") or {}
        excluded = {s.upper() for s in (st.session_state.get("excluded_symbols") or [])}
        for holdings in all_holdings.values():
            for h in holdings:
                sym = h.get("tradingsymbol", "")
                if sym and sym.upper() not in excluded and sym not in syms:
                    syms.append(sym)
    except Exception:
        pass
    return syms


def _split_symbols(text: str) -> list[str]:
    """Split on commas / semicolons / newlines; uppercase and dedupe, preserving order."""
    return list(dict.fromkeys(
        s.strip().upper() for s in re.split(r"[,\n;]+", text) if s.strip()
    ))


def _symbols_from_upload(uploaded) -> list[str]:
    """Parse tickers from an uploaded CSV (a `ticker` column, else the first) or TXT file."""
    raw = uploaded.read().decode("utf-8", errors="replace")
    if not uploaded.name.lower().endswith(".csv"):
        return _split_symbols(raw)

    reader = csv.DictReader(io.StringIO(raw))
    fields = reader.fieldnames or []
    col = next((c for c in fields if "ticker" in c.lower()), None) or (fields[0] if fields else None)
    if not col:
        return []
    return list(dict.fromkeys(
        (r.get(col) or "").strip().upper() for r in reader if (r.get(col) or "").strip()
    ))


def _render_portfolio_selector() -> list[str]:
    """Three-way portfolio picker; returns the active portfolio symbols."""
    if "portfolio_syms" not in st.session_state:
        st.session_state["portfolio_syms"] = _load_portfolio_syms()

    with st.expander("🗂️ Choose your portfolio", expanded=True):
        tab_hold, tab_manual, tab_file = st.tabs(
            ["📋 Current Holdings", "✏️ Manual", "📂 File Upload"]
        )

        with tab_hold:
            hold_syms = _load_portfolio_syms()
            if hold_syms:
                st.caption(f"{len(hold_syms)} ticker(s) loaded from broker session.")
                st.code("\n".join(hold_syms), language=None)
            else:
                st.info("No holdings found in session.")
            if st.button("▶ Use Current Holdings", disabled=not hold_syms, key="port_use_holdings"):
                st.session_state["portfolio_syms"] = hold_syms
                st.success(f"✅ {len(hold_syms)} ticker(s) loaded.")

        with tab_manual:
            st.caption("Enter ticker symbols separated by commas or newlines.")
            manual_txt = st.text_area(
                "Tickers", height=120, key="port_manual_txt",
                label_visibility="collapsed",
                placeholder="e.g.  RELIANCE, INFY, TCS",
            )
            manual_syms = _split_symbols(manual_txt)
            if st.button(f"▶ Proceed ({len(manual_syms)} tickers)",
                         disabled=not manual_syms, key="port_use_manual"):
                st.session_state["portfolio_syms"] = manual_syms
                st.success(f"✅ {len(manual_syms)} ticker(s) loaded.")

        with tab_file:
            st.caption("Upload a CSV or TXT file with a `ticker` column.")
            uploaded = st.file_uploader(
                "Drop file", type=["csv", "txt"], key="port_file_upload",
                label_visibility="collapsed",
            )
            file_syms: list[str] = []
            if uploaded:
                file_syms = _symbols_from_upload(uploaded)
                if file_syms:
                    st.caption(f"{len(file_syms)} ticker(s) parsed from {uploaded.name}.")
                    st.code("\n".join(file_syms), language=None)
                else:
                    st.warning("No tickers found in file.")
            if st.button(f"▶ Proceed ({len(file_syms)} tickers)",
                         disabled=not file_syms, key="port_use_file"):
                st.session_state["portfolio_syms"] = file_syms
                st.success(f"✅ {len(file_syms)} ticker(s) loaded.")

    return st.session_state["portfolio_syms"]


# ── Editor-load callbacks ──────────────────────────────────────────────────────
# Widgets with a `key` ignore `value=` after first render, so the editor's widget
# keys are written directly. Callbacks run BEFORE the script reruns, which is the
# only point where writing those keys is allowed.

def _load_into_editor(sql: str, name: str) -> None:
    st.session_state["sqllab_sql_input"]        = sql    # text_area widget key
    st.session_state["sqllab_panel_name_input"] = name   # text_input widget key
    st.session_state["sqllab_ran_sql"]          = ""
    st.session_state["sqllab_result_df"]        = None
    st.session_state["sqllab_run_error"]        = None
    st.session_state["sqllab_toast"]            = f"✏️ Loaded “{name}” — open the **Editor** tab."


def _clear_editor() -> None:
    _load_into_editor("", "")
    st.session_state.pop("sqllab_toast", None)


# ── Panel rendering ────────────────────────────────────────────────────────────

def _render_viz(
        df: pd.DataFrame,
        viz: str,
        height: int = 280,
        portfolio_syms: list[str] | None = None,
        key: str = "tbl",
        max_rows: int | None = None,
) -> None:
    if viz == "Table":
        table_widget(df, key=key, height=height, max_rows=max_rows,
                     portfolio_syms=portfolio_syms)
        return

    numeric_cols = df.select_dtypes("number").columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns for chart.")
    elif viz in ("Line Chart", "Area Chart"):
        chart = st.line_chart if viz == "Line Chart" else st.area_chart
        chart(df[numeric_cols].head(100), use_container_width=True)
    elif viz == "Bar Chart":
        st.bar_chart(df[numeric_cols].head(50), use_container_width=True)


def _render_single_panel(panel: dict, idx: int, portfolio_syms: list[str] | None = None) -> None:
    viz_icon = VIZ_ICONS.get(panel["viz"], "📋")
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:0.25rem;">'
        f'<span style="font-weight:600;font-size:0.9rem;">{html.escape(panel["name"])}</span>'
        f'<span class="badge-updated">{viz_icon} {html.escape(panel["viz"])}</span>'
        f'<span class="badge-pending">⏰ {html.escape(panel["published_at"])}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"{panel['row_count']} rows · {len(panel['columns'])} cols")

    df, err = run_sql(panel["sql"])
    if err:
        st.error(f"SQL error: {err}")
    elif df is not None:
        _render_viz(df, panel["viz"], portfolio_syms=portfolio_syms, key=f"panel_{idx}")

    act1, act2, act3 = st.columns(3)
    act1.button(
        "✏️ Edit", key=f"panel_edit_{idx}", use_container_width=True,
        on_click=_load_into_editor, args=(panel["sql"], panel["name"]),
    )
    if act2.button("🗑 Remove", key=f"panel_del_{idx}", use_container_width=True):
        _remove_panel(panel["name"])
        st.rerun()
    with act3.expander("SQL"):
        st.code(panel["sql"], language="sql")

    st.markdown("---")


# ── Tabs ───────────────────────────────────────────────────────────────────────

def _render_db_status(tables: list[str], error: str | None) -> None:
    stat_col, rebuild_col = st.columns([5, 1])
    if error:
        stat_col.error(f"⚠️ **DB build failed** — {error}")
    else:
        stat_col.success(f"DB ready · {len(tables)} tables/views", icon="🦆")

    if rebuild_col.button("🔄 Rebuild DB", use_container_width=True,
                          help=f"Re-run {DB_SQL_PATH}"):
        reset_connection()
        st.rerun()


def _render_editor_tab(portfolio_syms: list[str]) -> None:
    st.session_state.setdefault("sqllab_sql_input", "")
    st.session_state.setdefault("sqllab_panel_name_input", "")

    col_left, col_right = st.columns([3, 1], gap="medium")

    with col_left:
        st.markdown("#### SQL Editor")
        st.text_input(
            "Panel name",
            placeholder="e.g. High ROCE Compounders",
            key="sqllab_panel_name_input",
        )
        st.text_area(
            "SQL",
            height=220,
            key="sqllab_sql_input",
            label_visibility="collapsed",
            placeholder="Write your DuckDB SQL here…",
        )

        btn_col1, btn_col2, _ = st.columns([1, 1, 4])
        run_clicked = btn_col1.button("▶ Run", type="primary", use_container_width=True)
        btn_col2.button("✕ Clear", use_container_width=True, on_click=_clear_editor)

        if run_clicked:
            sql = st.session_state["sqllab_sql_input"]
            with st.spinner("Running query…"):
                df, err = run_sql(sql)
            # Remember the SQL that produced the preview: publishing uses this,
            # not whatever has been typed since.
            st.session_state["sqllab_ran_sql"]   = sql
            st.session_state["sqllab_result_df"] = df
            st.session_state["sqllab_run_error"] = err

    with col_right:
        st.markdown("#### Publish")
        st.radio(
            "Visualisation",
            VIZ_TYPES,
            format_func=lambda v: f"{VIZ_ICONS[v]} {v}",
            key="sqllab_viz_type",
        )

    # ── Preview ────────────────────────────────────────────────────────────────
    run_error: str | None          = st.session_state.get("sqllab_run_error")
    result_df: pd.DataFrame | None = st.session_state.get("sqllab_result_df")
    viz = st.session_state.get("sqllab_viz_type", "Table")

    if run_error:
        st.error(f"**Query error**\n\n```\n{run_error}\n```")

    if result_df is None:
        return

    st.markdown("---")
    hdr_col1, hdr_col2 = st.columns([3, 1])
    hdr_col1.markdown(
        f"#### Preview &nbsp;<span class='badge-updated'>{len(result_df)} rows</span>",
        unsafe_allow_html=True,
    )

    if hdr_col2.button("🚀 Publish Panel", type="primary", use_container_width=True):
        name = (st.session_state.get("sqllab_panel_name_input") or "").strip()
        if not name:
            st.warning("Enter a panel name before publishing.")
        else:
            _publish_panel(name, st.session_state.get("sqllab_ran_sql", ""), viz, result_df)
            st.success(f"✅ Published **{name}** → `sqls/panels.json`")

    _render_viz(result_df, viz, height=420, portfolio_syms=portfolio_syms,
                key="editor", max_rows=200)

    with st.expander("Column info"):
        col_info = pd.DataFrame({
            "Column":   result_df.columns,
            "Type":     [str(t) for t in result_df.dtypes],
            "Non-null": result_df.count().values,
            "Sample":   [
                str(result_df[c].iloc[0]) if len(result_df) > 0 else "—"
                for c in result_df.columns
            ],
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)


def _render_schema_tab(tables: list[str]) -> None:
    st.markdown("#### Schema Browser")
    st.caption("Expand any table to view its CREATE statement and sample rows.")

    if not tables:
        st.info(f"No tables found. Make sure `{DB_SQL_PATH}` exists and is valid.")
        return

    st.success(f"Found **{len(tables)}** table(s) / view(s).")
    half = max(1, len(tables) // 2 + len(tables) % 2)
    col_a, col_b = st.columns(2, gap="medium")

    for i, table in enumerate(tables):
        with (col_a if i < half else col_b):
            with st.expander(f"🗂 {table}"):
                try:
                    st.code(ddl_for_table(table), language="sql")
                    sample_df = sample_rows(table, n=5)
                except Exception as exc:
                    st.caption(f"Could not introspect {table}: {exc}")
                    sample_df = pd.DataFrame()

                if not sample_df.empty:
                    st.markdown("**Sample rows**")
                    table_widget(sample_df, key=f"schema_{table}", height=220,
                                 highlight_portfolio=False)
                else:
                    st.caption("(No rows available)")

                st.button(
                    "Open in Editor →", key=f"schema_open_{table}",
                    on_click=_load_into_editor,
                    args=(f"SELECT *\nFROM {quote_ident(table)}\nLIMIT 50", table),
                )


def _render_dashboard_tab(portfolio_syms: list[str]) -> None:
    panels: list[dict] = st.session_state.get("sqllab_panels", [])

    dash_hdr, dash_btn = st.columns([4, 1])
    dash_hdr.markdown(
        f"#### Dashboard &nbsp;<span class='badge-bench'>{len(panels)} panel(s)</span>"
        f"&nbsp;<span class='badge-pending' style='font-size:0.65rem;'>💾 sqls/panels.json</span>",
        unsafe_allow_html=True,
    )
    if dash_btn.button("🔄 Refresh all", use_container_width=True):
        st.rerun()

    if not panels:
        st.markdown(
            '<div style="text-align:center;padding:4rem 0;color:#484f58;">'
            '<div style="font-size:3rem;">📊</div>'
            '<p>No panels yet. Write a query in the Editor, preview it, '
            'then hit <strong>Publish Panel</strong>.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    for panel_idx, panel in enumerate(panels):
        with st.container():
            _render_single_panel(panel, panel_idx, portfolio_syms=portfolio_syms)


# ── Main render ────────────────────────────────────────────────────────────────

def render() -> None:
    _init_panels()
    portfolio_syms = _render_portfolio_selector()

    with st.spinner("Building SQL Lab database…"):
        tables, db_error = _database_status()
    _render_db_status(tables, db_error)
    if db_error:
        st.stop()                                  # nothing below works without the database

    st.markdown(
        '<div class="topbar">'
        '<h1>🧪 SQL Lab</h1>'
        '<span class="sync-info">Write SQL → Preview → Publish as panel</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    toast = st.session_state.pop("sqllab_toast", None)
    if toast:
        st.toast(toast)

    tab_dashboard, tab_editor, tab_schema = st.tabs(
        ["📊 Dashboard", "✏️ Editor", "🗄️ Schema Browser"]
    )
    with tab_dashboard:
        _render_dashboard_tab(portfolio_syms)
    with tab_editor:
        _render_editor_tab(portfolio_syms)
    with tab_schema:
        _render_schema_tab(tables)