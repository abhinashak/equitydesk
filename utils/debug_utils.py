"""
debug_utils.py
──────────────
Generic SQL execution logging and Streamlit debug-panel rendering helpers.
"""

import pandas as pd
import streamlit as st

DEFAULT_SQL_LOG_KEY = "_sql_log"


def init_sql_log(log_key: str = DEFAULT_SQL_LOG_KEY) -> None:
    """Reset or initialize the SQL log list in Streamlit session state."""
    st.session_state[log_key] = []


def log_sql(
    label: str,
    sql: str,
    df: pd.DataFrame | None = None,
    elapsed_ms: float | None = None,
    error: str | None = None,
    log_key: str = DEFAULT_SQL_LOG_KEY,
    date_col: str = "date",
    group_col: str = "ticker",
) -> None:
    """Record an executed SQL statement (plus optional row, date range, and per-group stats)."""
    entry: dict = {
        "label": label,
        "sql": sql.strip(),
        "elapsed_ms": elapsed_ms,
        "error": error,
        "rows": None,
        "date_min": None,
        "date_max": None,
        "per_group_last": None,
        "group_col": group_col,
    }

    if df is not None:
        entry["rows"] = len(df)
        if date_col in df.columns and len(df) > 0:
            parsed_dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
            if not parsed_dates.empty:
                entry["date_min"] = str(parsed_dates.min().date())
                entry["date_max"] = str(parsed_dates.max().date())

            if group_col in df.columns:
                grouped = (
                    df.assign(**{date_col: pd.to_datetime(df[date_col], errors="coerce")})
                    .dropna(subset=[date_col])
                    .groupby(group_col, dropna=False)[date_col]
                    .max()
                    .dt.date.astype(str)
                    .reset_index()
                    .rename(columns={group_col: group_col, date_col: "last_date"})
                )
                entry["per_group_last"] = grouped

    st.session_state.setdefault(log_key, []).append(entry)


def render_sql_log(
    slot,
    log_key: str = DEFAULT_SQL_LOG_KEY,
    expanded: bool = False,
) -> None:
    """Render all recorded SQL statements for the current run into a Streamlit slot."""
    log = st.session_state.get(log_key, [])
    with slot.container():
        with st.expander(f"🧾 Executed SQL ({len(log)})", expanded=expanded):
            if not log:
                st.caption("No SQL executed in this run.")
                return

            for idx, entry in enumerate(log, 1):
                st.markdown(f"**{idx}. {entry['label']}**")
                st.code(entry["sql"], language="sql")

                meta_parts: list[str] = []
                if entry["rows"] is not None:
                    meta_parts.append(f"rows: **{entry['rows']}**")
                if entry["date_min"] and entry["date_max"]:
                    meta_parts.append(f"dates: **{entry['date_min']} → {entry['date_max']}**")
                elif entry["date_min"]:
                    meta_parts.append(f"date: **{entry['date_min']}**")
                if entry["elapsed_ms"] is not None:
                    meta_parts.append(f"{entry['elapsed_ms']:.0f} ms")
                if meta_parts:
                    st.caption("  ·  ".join(meta_parts))

                if entry["error"]:
                    st.error(entry["error"])

                per_group = entry.get("per_group_last")
                if per_group is not None and not per_group.empty:
                    group_label = entry.get("group_col") or "group"
                    st.caption(f"Last date per {group_label}")
                    st.dataframe(
                        per_group,
                        hide_index=True,
                        use_container_width=True,
                        height=180,
                    )
