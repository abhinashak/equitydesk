"""
ui/pages/page_sql_lab.py
─────────────────────────────────────────────────
SQL Lab — Write, preview, and publish SQL screeners as dashboard panels.

Flow:
  1. A shared in-memory DuckDB is built once per server process by running the
     SQL pipeline (init_raw → init → quality_gates → dcf → algo_momentum →
     buy_candidates), so views like abinash_screen exist. Each operation uses
     its own cursor. Use "Rebuild DB" to re-run the pipeline.
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

# Same build order as page_buy_candidates._PIPELINE. Scripts that don't exist
# are skipped; reorder here if your build order differs.
_PIPELINE: list[str] = [
    "init_raw.sql",
    "init.sql",
    "quality_gates.sql",
    "dcf.sql",
    "algo_momentum.sql",
    "buy_candidates.sql|abinash_recommendations.sql",   # creates abinash_params / abinash_screen / abinash_recommendations
]                                                       # ("a|b" = first file that exists wins)

try:                                   # same app-wide base dir the Buy Candidates page uses
    from bll import agent as _ag
    _BASE = str(_ag.BASE)
except Exception:
    _BASE = os.path.abspath(os.path.join(_SQLS_DIR, ".."))

def _raise_fd_limit(want: int = 4096) -> None:
    """macOS defaults to 256 open files, which a multi-parquet pipeline can exceed."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard != resource.RLIM_INFINITY:
            want = min(want, hard)
        if soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    except Exception:
        pass                                   # Windows / restricted env


_raise_fd_limit()

_READ_ONLY_STARTS = ("select", "with", "from", "describe", "show", "summarize", "explain", "pragma", "table")


def _find_script(name: str) -> str | None:
    for d in (os.path.join(_BASE, "sqls"), _BASE, _SQLS_DIR):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def _clean_script(text: str) -> str:
    """Drop DuckDB-CLI dot-commands (.mode, .output ...) that the Python API rejects."""
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("."))


@st.cache_resource(show_spinner="Building SQL Lab database (init → quality → dcf → momentum → buy candidates)…")
def _lab_db() -> tuple[duckdb.DuckDBPyConnection, list[dict]]:
    """
    Build ONE shared in-memory DuckDB by running the whole pipeline, once per
    server process. Returns (connection, build_log). Stops at the first failing
    script (later ones depend on earlier ones) but keeps what was built so far.
    Clear with `_lab_db.clear()` to rebuild.
    """
    con = duckdb.connect(":memory:")
    con.execute("SET threads = 4")
    log: list[dict] = []
    cwd = os.getcwd()
    try:
        os.chdir(_BASE)                       # relative parquet paths resolve from BASE
        for name in _PIPELINE:
            path = next((p for p in map(_find_script, name.split("|")) if p), None)
            if path is not None:
                name = os.path.basename(path)
            if path is None:
                log.append({"script": name, "status": "skipped", "error": None})
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    con.execute(_clean_script(f.read()))
                log.append({"script": name, "status": "ok", "error": None})
            except Exception as exc:
                log.append({"script": name, "status": "failed",
                            "error": f"{type(exc).__name__}: {str(exc)[:400]}"})
                break
    finally:
        os.chdir(cwd)

    # Final sanity check: the build must actually have produced abinash_screen
    if not any(e["error"] for e in log):
        try:
            con.execute("SELECT 1 FROM abinash_screen LIMIT 1")
        except Exception as exc:
            ran     = [e["script"] for e in log if e["status"] == "ok"]
            skipped = [e["script"] for e in log if e["status"] == "skipped"]
            log.append({"script": "(check) abinash_screen", "status": "failed",
                        "error": f"{type(exc).__name__}: {str(exc)[:200]} — ran: {ran}; "
                                 f"not found: {skipped}; looked in: {os.path.join(_BASE, 'sqls')}"})
    return con, log


def _rebuild_lab_db() -> None:
    """Close the shared connection (releasing its file handles) and drop the cache."""
    try:
        con, _ = _lab_db()
        con.close()
    except Exception:
        pass
    _lab_db.clear()


@contextlib.contextmanager
def _db():
    """
    Yield a cursor on the shared pipeline-built DuckDB; closed on exit.
    Usage:
        with _db() as con:
            df = con.execute(sql).df()
    """
    con, _ = _lab_db()
    cur = con.cursor()
    try:
        yield cur
    finally:
        cur.close()


def _check_init_sql() -> str | None:
    """Return an error string if the pipeline failed (or init.sql is missing), else None."""
    _, log = _lab_db()
    for entry in log:
        if entry["error"]:
            return f"{entry['script']}: {entry['error']}"
    if not any(e["script"] == "init.sql" and e["status"] == "ok" for e in log):
        return f"init.sql not found under: {os.path.abspath(os.path.join(_BASE, 'sqls'))}"
    return None


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
    first = sql.lstrip().lstrip("(").split(None, 1)[0].lower() if sql.strip() else ""
    if first not in _READ_ONLY_STARTS:
        return None, ("Only read-only queries are allowed in SQL Lab "
                      "(SELECT / WITH / DESCRIBE / SHOW …) — the database is shared.")
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



# ── Editor-load callbacks ──────────────────────────────────────────────────────
# Widgets with a `key` ignore `value=` after first render, so the editor's
# widget keys must be written directly. Callbacks run BEFORE the script reruns,
# which is the only point where writing those keys is allowed.

def _load_into_editor(sql: str, name: str) -> None:
    st.session_state["sqllab_sql"]              = sql
    st.session_state["sqllab_panel_name"]       = name
    st.session_state["sqllab_sql_input"]        = sql    # text_area widget key
    st.session_state["sqllab_panel_name_input"] = name   # text_input widget key
    st.session_state["sqllab_result_df"]        = None
    st.session_state["sqllab_run_error"]        = None
    st.session_state["sqllab_toast"]            = f"✏️ Loaded “{name}” — open the **Editor** tab."


def _clear_editor() -> None:
    _load_into_editor("", "")
    st.session_state.pop("sqllab_toast", None)

# ── Column filters ─────────────────────────────────────────────────────────────

_MAX_CHOICES = 100      # above this many distinct values, use a "contains" text box instead


def _filter_df(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """
    Per-table column filters. Pick columns in the expander, then filter each by:
      numeric  → range slider          datetime → date range
      text/low-cardinality → multiselect of values
      text/high-cardinality → case-insensitive "contains"
    Widget keys include the column set, so editing a query never reuses stale state.
    """
    if df is None or df.empty:
        return df
    sig = abs(hash(tuple(df.columns))) % 10**8
    k   = f"flt_{key}_{sig}"

    with st.expander("🔎 Filters", expanded=False):
        cols = st.multiselect("Filter on columns", list(df.columns), key=f"{k}_cols",
                              placeholder="Choose one or more columns…")
        mask = pd.Series(True, index=df.index)

        for col in cols:
            ser, wkey = df[col], f"{k}_{col}"
            if pd.api.types.is_bool_dtype(ser):
                sel = st.multiselect(col, [True, False], key=wkey)
                if sel:
                    mask &= ser.isin(sel)
            elif pd.api.types.is_numeric_dtype(ser):
                clean = ser.dropna()
                if clean.empty:
                    st.caption(f"{col}: no values"); continue
                mn, mx = clean.min(), clean.max()
                if mn == mx:
                    st.caption(f"{col}: all rows = {mn}"); continue
                is_int = pd.api.types.is_integer_dtype(ser)
                mn, mx = (int(mn), int(mx)) if is_int else (float(mn), float(mx))
                lo, hi = st.slider(col, mn, mx, (mn, mx), key=wkey)
                if (lo, hi) != (mn, mx):
                    mask &= ser.between(lo, hi)
            elif pd.api.types.is_datetime64_any_dtype(ser):
                clean = ser.dropna()
                if clean.empty:
                    st.caption(f"{col}: no values"); continue
                mn, mx = clean.min().date(), clean.max().date()
                rng = st.date_input(col, (mn, mx), min_value=mn, max_value=mx, key=wkey)
                if isinstance(rng, (tuple, list)) and len(rng) == 2:
                    mask &= ser.dt.date.between(rng[0], rng[1])
            else:
                as_text = ser.astype("string")
                uniq = as_text.dropna().unique().tolist()
                if len(uniq) <= _MAX_CHOICES:
                    sel = st.multiselect(col, sorted(uniq), key=wkey)
                    if sel:
                        mask &= as_text.isin(sel)
                else:
                    txt = st.text_input(f"{col} contains", key=wkey)
                    if txt:
                        mask &= as_text.str.contains(txt, case=False, na=False, regex=False)

        out = df[mask]
        if cols:
            st.caption(f"Showing **{len(out):,}** of {len(df):,} rows")
    return out


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
        df = _filter_df(df, key)
        if max_rows:
            df = df.head(max_rows)
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


# ── Main render ────────────────────────────────────────────────────────────────

def render() -> None:
    _init_panels()

    # ── Portfolio symbols ──────────────────────────────────────────────────────
    # ── Portfolio symbols — 3-way selector ───────────────────────────────────
    if "portfolio_syms" not in st.session_state:
        st.session_state["portfolio_syms"] = _load_portfolio_syms()

    with st.expander("🗂️ Choose your portfolio", expanded=True):
        tab_hold, tab_manual, tab_file = st.tabs(
            ["📋 Current Holdings", "✏️ Manual", "📂 File Upload"]
        )

        # ── Tab 1: Current Holdings ──────────────────────────────────────────
        with tab_hold:
            hold_syms = _load_portfolio_syms()
            if hold_syms:
                st.caption(f"{len(hold_syms)} ticker(s) loaded from broker session.")
                st.code("\n".join(hold_syms), language=None)
            else:
                st.info("No holdings found in session.")
            if st.button("▶ Use Current Holdings",
                         disabled=not hold_syms, key="port_use_holdings"):
                st.session_state["portfolio_syms"] = hold_syms
                st.success(f"✅ {len(hold_syms)} ticker(s) loaded.")

        # ── Tab 2: Manual ────────────────────────────────────────────────────
        with tab_manual:
            st.caption("Enter ticker symbols separated by commas or newlines.")
            manual_txt = st.text_area(
                "Tickers", height=120, key="port_manual_txt",
                label_visibility="collapsed",
                placeholder="e.g.  RELIANCE, INFY, TCS",
            )
            import re as _re
            manual_syms = [s.strip().upper()
                           for s in _re.split(r"[,\n;]+", manual_txt)
                           if s.strip()]
            manual_syms = list(dict.fromkeys(manual_syms))  # dedupe, preserve order
            if st.button(f"▶ Proceed ({len(manual_syms)} tickers)",
                         disabled=not manual_syms, key="port_use_manual"):
                st.session_state["portfolio_syms"] = manual_syms
                st.success(f"✅ {len(manual_syms)} ticker(s) loaded.")

        # ── Tab 3: File Upload ───────────────────────────────────────────────
        with tab_file:
            st.caption("Upload a CSV or TXT file with a `ticker` column.")
            uploaded = st.file_uploader(
                "Drop file", type=["csv", "txt"], key="port_file_upload",
                label_visibility="collapsed",
            )
            file_syms: list[str] = []
            if uploaded:
                import io as _io, json as _json
                raw = uploaded.read().decode("utf-8", errors="replace")
                if uploaded.name.lower().endswith(".csv"):
                    import csv as _csv
                    reader = _csv.DictReader(_io.StringIO(raw))
                    col = next((c for c in (reader.fieldnames or [])
                                if "ticker" in c.lower()), None)
                    col = col or (reader.fieldnames[0] if reader.fieldnames else None)
                    if col:
                        file_syms = list(dict.fromkeys(
                            r[col].strip().upper() for r in reader if r[col].strip()
                        ))
                else:
                    import re as _re2
                    file_syms = list(dict.fromkeys(
                        s.strip().upper()
                        for s in _re2.split(r"[,\n;]+", raw) if s.strip()
                    ))
                if file_syms:
                    st.caption(f"{len(file_syms)} ticker(s) parsed from {uploaded.name}.")
                    st.code("\n".join(file_syms), language=None)
                else:
                    st.warning("No tickers found in file.")
            if st.button(f"▶ Proceed ({len(file_syms)} tickers)",
                         disabled=not file_syms, key="port_use_file"):
                st.session_state["portfolio_syms"] = file_syms
                st.success(f"✅ {len(file_syms)} ticker(s) loaded.")

    portfolio_syms: list[str] = st.session_state["portfolio_syms"]


    init_error = _check_init_sql()
    _, _build_log = _lab_db()
    stat_col, rebuild_col = st.columns([5, 1])
    if init_error:
        stat_col.error(f"⚠️ **DB build failed** — {init_error}")
    else:
        built = [e["script"] for e in _build_log if e["status"] == "ok"]
        has_screen = "abinash_screen" in _list_tables()
        stat_col.success(
            f"DB built: {' → '.join(built)}"
            + ("  ·  `abinash_screen` ✔" if has_screen else "  ·  ⚠️ `abinash_screen` not found"),
            icon="🦆",
        )
    if rebuild_col.button("🔄 Rebuild DB", use_container_width=True,
                          help="Re-run the whole SQL pipeline"):
        _rebuild_lab_db()
        st.rerun()
    with st.expander("Build log", expanded=bool(init_error)):
        st.dataframe(pd.DataFrame(_build_log), hide_index=True, use_container_width=True)

    st.markdown(
        '<div class="topbar">'
        '<h1>🧪 SQL Lab</h1>'
        '<span class="sync-info">Write SQL → Preview → Publish as panel</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    _toast = st.session_state.pop("sqllab_toast", None)
    if _toast:
        st.toast(_toast)

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
            btn_col2.button("✕ Clear", use_container_width=True, on_click=_clear_editor)

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

            _render_viz(result_df, st.session_state.get("sqllab_viz_type", "Table"), height=420, portfolio_syms=portfolio_syms, key="editor", max_rows=200)

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
                            st.dataframe(_filter_df(sample_df, f"schema_{table}"), use_container_width=True, hide_index=True)
                        else:
                            st.caption("(No rows available)")

                        st.button(
                            "Open in Editor →", key=f"schema_open_{table}",
                            on_click=_load_into_editor,
                            args=(f"SELECT *\nFROM {table}\nLIMIT 50", table),
                        )

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
