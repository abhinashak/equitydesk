"""
ui/pages/page_sql_lab.py
─────────────────────────────────────────────────
SQL Lab — Write, preview, and publish SQL screeners as dashboard panels.

Flow:
  1. Each DB operation opens a fresh in-memory DuckDB, runs init.sql to
     set up views, executes the query, then closes immediately.
  2. Schema browser shows CREATE statements + live sample rows (collapsible).
  3. Editor: write SQL → Run → preview result.
  4. Assign a visualisation type (Table / Line / Bar / Area).
  5. Publish → panel saved to sqls/panels.json and rendered in Dashboard tab.
"""

from __future__ import annotations

import contextlib
import json
import os
from datetime import datetime

import duckdb
import pandas as pd
import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────

_SQLS_DIR        = os.path.join(os.path.dirname(__file__), "..", "..", "sqls")
INIT_SQL_PATH    = os.path.join(_SQLS_DIR, "init.sql")
PANELS_JSON_PATH = os.path.join(_SQLS_DIR, "panels.json")

VIZ_TYPES: list[str] = ["Table", "Line Chart", "Bar Chart", "Area Chart"]
VIZ_ICONS: dict[str, str] = {
    "Table":      "📋",
    "Line Chart": "📈",
    "Bar Chart":  "📊",
    "Area Chart": "🌊",
}

# ── DuckDB ─────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _db():
    """
    Open a fresh in-memory DuckDB connection, run init.sql, yield it,
    then close it — guaranteed even on exception.
    Usage:
        with _db() as con:
            df = con.execute(sql).df()
    """
    con = duckdb.connect(":memory:")
    try:
        if os.path.exists(INIT_SQL_PATH):
            with open(INIT_SQL_PATH) as f:
                con.execute(f.read())
        yield con
    finally:
        con.close()


def _check_init_sql() -> str | None:
    """Return an error string if init.sql is missing or fails, else None."""
    if not os.path.exists(INIT_SQL_PATH):
        return f"init.sql not found at: {os.path.abspath(INIT_SQL_PATH)}"
    with open(INIT_SQL_PATH) as f:
        sql = f.read()
    con = duckdb.connect(":memory:")
    try:
        con.execute(sql)
        return None
    except Exception as exc:
        return str(exc)
    finally:
        con.close()


# ── Schema helpers ─────────────────────────────────────────────────────────────

def _list_tables() -> list[str]:
    try:
        with _db() as con:
            rows = con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


def _ddl_for_table(table: str) -> str:
    try:
        with _db() as con:
            cols = con.execute(
                f"SELECT column_name, data_type FROM information_schema.columns "
                f"WHERE table_name = '{table}' ORDER BY ordinal_position"
            ).fetchall()
        col_defs = ",\n    ".join(f"{c} {t}" for c, t in cols)
        return f"CREATE TABLE {table} (\n    {col_defs}\n);"
    except Exception:
        return f"-- Could not introspect {table}"


def _sample_rows(table: str, n: int = 5) -> pd.DataFrame:
    try:
        with _db() as con:
            return con.execute(f"SELECT * FROM {table} LIMIT {n}").df()
    except Exception:
        return pd.DataFrame()


def _run_sql(sql: str) -> tuple[pd.DataFrame | None, str | None]:
    """Execute arbitrary SQL. Returns (dataframe, None) or (None, error_str)."""
    try:
        with _db() as con:
            df = con.execute(sql).df()
        return df, None
    except Exception as exc:
        return None, str(exc)


# ── Panel persistence ──────────────────────────────────────────────────────────

def _load_panels() -> list[dict]:
    if not os.path.exists(PANELS_JSON_PATH):
        return []
    try:
        with open(PANELS_JSON_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def _save_panels(panels: list[dict]) -> None:
    os.makedirs(_SQLS_DIR, exist_ok=True)
    with open(PANELS_JSON_PATH, "w") as f:
        json.dump(panels, f, indent=2)


def _init_panels() -> None:
    if "sqllab_panels_loaded" not in st.session_state:
        st.session_state["sqllab_panels"] = _load_panels()
        st.session_state["sqllab_panels_loaded"] = True


def _publish_panel(name: str, sql: str, viz: str, df: pd.DataFrame) -> None:
    panels: list[dict] = st.session_state["sqllab_panels"]
    panels = [p for p in panels if p["name"] != name]
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
    _portfolio_syms: list[str] = []
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
    return _portfolio_syms


def _highlight_portfolio(df: pd.DataFrame, portfolio_syms: list[str]) -> pd.io.formats.style.Styler:
    """
    Return a Styler that highlights rows whose 'ticker' column value
    is in *portfolio_syms* with a gold background.
    """
    portfolio_set = {s.upper() for s in portfolio_syms}

    def _row_style(row: pd.Series) -> list[str]:
        ticker_val = str(row.get("ticker", "")).upper()
        if ticker_val in portfolio_set:
            return ["background-color: #fff8c5; color: #1a1a1a;"] * len(row)
        return [""] * len(row)

    return df.style.apply(_row_style, axis=1)


# ── Panel rendering ────────────────────────────────────────────────────────────

def _render_viz(
        df: pd.DataFrame,
        viz: str,
        height: int = 280,
        portfolio_syms: list[str] | None = None,
) -> None:
    if viz == "Table":
        has_ticker = "ticker" in df.columns
        port_syms  = portfolio_syms or []
        if has_ticker and port_syms:
            # Find which portfolio tickers actually appear in this dataframe
            port_set   = {s.upper() for s in port_syms}
            hits       = df["ticker"].str.upper().isin(port_set)
            hit_count  = int(hits.sum())
            if hit_count:
                hit_tickers = df.loc[hits, "ticker"].str.upper().unique().tolist()
                ticker_badges = "".join(
                    f'<span style="display:inline-block;margin:0 2px;padding:1px 6px;'
                    f'background:#fff3cd;border:1px solid #ffc107;border-radius:4px;'
                    f'font-size:0.72rem;font-weight:600;color:#856404;">{t}</span>'
                    for t in hit_tickers
                )
                st.markdown(
                    f'<span style="font-size:0.78rem;color:#856404;">'
                    f'\U0001F31F <strong>{hit_count}</strong> of your portfolio stock(s) highlighted below:&nbsp;'
                    f'{ticker_badges}</span>',
                    unsafe_allow_html=True,
                )
                with st.popover("📋 Copy tickers"):
                    st.code("\n".join(hit_tickers), language=None)
            styled = _highlight_portfolio(df, port_syms)
            st.dataframe(styled, use_container_width=True, height=height)
        else:
            st.dataframe(df, use_container_width=True, height=height)
    elif viz in ("Line Chart", "Area Chart"):
        numeric_cols = df.select_dtypes("number").columns.tolist()
        if numeric_cols:
            chart_df = df[numeric_cols].head(100)
            (st.line_chart if viz == "Line Chart" else st.area_chart)(
                chart_df, use_container_width=True
            )
        else:
            st.info("No numeric columns for chart.")
    elif viz == "Bar Chart":
        numeric_cols = df.select_dtypes("number").columns.tolist()
        if numeric_cols:
            st.bar_chart(df[numeric_cols].head(50), use_container_width=True)
        else:
            st.info("No numeric columns for chart.")


def _render_single_panel(panel: dict, idx: int, portfolio_syms: list[str] | None = None) -> None:
    viz_icon = VIZ_ICONS.get(panel["viz"], "📋")
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:0.25rem;">'
        f'<span style="font-weight:600;font-size:0.9rem;">{panel["name"]}</span>'
        f'<span class="badge-updated">{viz_icon} {panel["viz"]}</span>'
        f'<span class="badge-pending">⏰ {panel["published_at"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"{panel['row_count']} rows · {len(panel['columns'])} cols")

    df, err = _run_sql(panel["sql"])
    if err:
        st.error(f"SQL error: {err}")
    elif df is not None:
        _render_viz(df, panel["viz"], portfolio_syms=portfolio_syms)

    act1, act2, act3 = st.columns(3)
    if act1.button("✏️ Edit", key=f"panel_edit_{idx}", use_container_width=True):
        st.session_state["sqllab_sql"]        = panel["sql"]
        st.session_state["sqllab_panel_name"] = panel["name"]
        st.rerun()
    if act2.button("🗑 Remove", key=f"panel_del_{idx}", use_container_width=True):
        _remove_panel(panel["name"])
        st.rerun()
    with act3.expander("SQL"):
        st.code(panel["sql"], language="sql")

    st.markdown("---")


# ── Main render ────────────────────────────────────────────────────────────────

def render() -> None:
    _init_panels()

    # ── Portfolio symbols ──────────────────────────────────────────────────────
    portfolio_syms: list[str] = _load_portfolio_syms()

    init_error = _check_init_sql()
    if init_error:
        st.error(f"⚠️ **init.sql failed** — {init_error}")
    else:
        st.success("✅ DB initialised from `init.sql`", icon="🦆")

    st.markdown(
        '<div class="topbar">'
        '<h1>🧪 SQL Lab</h1>'
        '<span class="sync-info">Write SQL → Preview → Publish as panel</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    tab_dashboard, tab_editor, tab_schema  = st.tabs(
        ["📊 Dashboard","✏️ Editor", "🗄️ Schema Browser" ]
    )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — EDITOR
    # ══════════════════════════════════════════════════════════════════════════
    with tab_editor:
        col_left, col_right = st.columns([3, 1], gap="medium")

        with col_left:
            st.markdown("#### SQL Editor")

            panel_name = st.text_input(
                "Panel name",
                value=st.session_state.get("sqllab_panel_name", ""),
                placeholder="e.g. High ROCE Compounders",
                key="sqllab_panel_name_input",
            )

            sql_input = st.text_area(
                "SQL",
                value=st.session_state.get("sqllab_sql", ""),
                height=220,
                key="sqllab_sql_input",
                label_visibility="collapsed",
                placeholder="Write your DuckDB SQL here…",
            )

            btn_col1, btn_col2, _ = st.columns([1, 1, 4])
            run_clicked   = btn_col1.button("▶ Run",   type="primary", use_container_width=True)
            clear_clicked = btn_col2.button("✕ Clear", use_container_width=True)

            if clear_clicked:
                st.session_state["sqllab_sql"]        = ""
                st.session_state["sqllab_panel_name"] = ""
                st.session_state["sqllab_result_df"]  = None
                st.session_state["sqllab_run_error"]  = None
                st.rerun()

            if run_clicked:
                st.session_state["sqllab_sql"]        = sql_input
                st.session_state["sqllab_panel_name"] = panel_name
                with st.spinner("Running query…"):
                    df, err = _run_sql(sql_input)
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

        # ── Preview ────────────────────────────────────────────────────────────
        run_error: str | None          = st.session_state.get("sqllab_run_error")
        result_df: pd.DataFrame | None = st.session_state.get("sqllab_result_df")

        if run_error:
            st.error(f"**Query error**\n\n```\n{run_error}\n```")

        if result_df is not None:
            st.markdown("---")
            hdr_col1, hdr_col2 = st.columns([3, 1])
            hdr_col1.markdown(
                f"#### Preview &nbsp;<span class='badge-updated'>{len(result_df)} rows</span>",
                unsafe_allow_html=True,
            )

            if hdr_col2.button("🚀 Publish Panel", type="primary", use_container_width=True):
                current_name = st.session_state.get("sqllab_panel_name_input") or panel_name
                current_sql  = st.session_state.get("sqllab_sql", sql_input)
                current_viz  = st.session_state.get("sqllab_viz_type", "Table")
                if not current_name.strip():
                    st.warning("Enter a panel name before publishing.")
                else:
                    _publish_panel(current_name, current_sql, current_viz, result_df)
                    st.success(f"✅ Published **{current_name}** → `sqls/panels.json`")

            _render_viz(result_df.head(200), st.session_state.get("sqllab_viz_type", "Table"), height=420, portfolio_syms=portfolio_syms)

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

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — SCHEMA BROWSER
    # ══════════════════════════════════════════════════════════════════════════
    with tab_schema:
        st.markdown("#### Schema Browser")
        st.caption("Expand any table to view its CREATE statement and sample rows.")

        tables = _list_tables()

        if not tables:
            st.info("No tables found. Make sure `sqls/init.sql` exists and is valid.")
            with st.expander("📂 Expected init.sql location"):
                st.code(os.path.abspath(INIT_SQL_PATH), language="text")
        else:
            st.success(f"Found **{len(tables)}** table(s) / view(s).")
            half = max(1, len(tables) // 2 + len(tables) % 2)
            col_a, col_b = st.columns(2, gap="medium")

            for i, table in enumerate(tables):
                with (col_a if i < half else col_b):
                    with st.expander(f"🗂 {table}"):
                        st.code(_ddl_for_table(table), language="sql")

                        sample_df = _sample_rows(table, n=5)
                        if not sample_df.empty:
                            st.markdown("**Sample rows**")
                            st.dataframe(sample_df, use_container_width=True, hide_index=True)
                        else:
                            st.caption("(No rows available)")

                        if st.button("Open in Editor →", key=f"schema_open_{table}"):
                            st.session_state["sqllab_sql"]        = f"SELECT *\nFROM {table}\nLIMIT 50"
                            st.session_state["sqllab_panel_name"] = table
                            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — DASHBOARD
    # ══════════════════════════════════════════════════════════════════════════
    with tab_dashboard:
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
        else:
            for panel_idx, panel in enumerate(panels):
                # Wraps each panel in a clean container that spans 100% width
                with st.container():
                    _render_single_panel(panel, panel_idx, portfolio_syms=portfolio_syms)