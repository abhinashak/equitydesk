"""
ui/components/table_chart.py
────────────────────────────
Radio toggle that shows a DataFrame either as a filterable table (table_widget)
or as a configurable line chart.

Usage:
    table_or_chart(df, key="my_view", height=520, column_config={...})
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.components.table_widget import table_widget

_DATE_HINTS = ("date", "time", "_dt", "day", "month", "year", "period")


def _date_columns(df: pd.DataFrame) -> list[str]:
    """Columns that are datetimes, or text columns that look like (and parse as) dates."""
    out = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out.append(c)
        elif (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)) \
                and any(h in str(c).lower() for h in _DATE_HINTS):
            sample = s.dropna().head(200)
            if len(sample) and pd.to_datetime(sample, errors="coerce").notna().mean() >= 0.8:
                out.append(c)
    return out


def line_chart_widget(df: pd.DataFrame, key: str) -> None:
    """Interactive line chart: pick X, one or more numeric Y columns, optional split/aggregation."""
    if df.empty:
        st.info("Nothing to chart.")
        return

    date_cols = _date_columns(df)
    other_cols = [c for c in df.columns if c not in date_cols]
    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    if not num_cols:
        st.info("No numeric columns to plot.")
        return

    x_options = date_cols + other_cols
    c1, c2, c3, c4 = st.columns([1.3, 2, 1.3, 1])
    x = c1.selectbox("X axis / date column", x_options, key=f"{key}_x",
                     format_func=lambda c: f"📅 {c}" if c in date_cols else c,
                     help="Date columns are listed first.")
    y_choices = [c for c in num_cols if c != x]
    y = c2.multiselect("Y (numeric)", y_choices, default=y_choices[:1], key=f"{key}_y")

    split_choices = [c for c in df.columns
                     if c != x and c not in y_choices
                     and 1 < df[c].nunique() <= 25 and not pd.api.types.is_float_dtype(df[c])]
    split = c3.selectbox("One line per…", ["(none)"] + split_choices, key=f"{key}_split")
    agg = c4.selectbox("If rows share X", ["mean", "median", "sum", "last", "max", "min"],
                       key=f"{key}_agg")

    if not y:
        st.info("Pick at least one Y column.")
        return
    if split != "(none)" and len(y) > 1:
        st.info("Splitting into lines works with one Y column — pick just one.")
        return

    d = df[[x, *y] + ([split] if split != "(none)" else [])].copy()
    if x in date_cols:
        d[x] = pd.to_datetime(d[x], errors="coerce")
    d = d.dropna(subset=[x])
    if d[x].nunique() < 2:
        st.info(f"`{x}` has only one distinct value here, so there is no line to draw. "
                "Pick another X column — or use the SQL tab to query a time series "
                "(e.g. a price/momentum history table).")
        return

    try:
        if split == "(none)":
            plot = d.groupby(x)[y].agg(agg).sort_index()
        else:
            plot = d.groupby([x, split])[y[0]].agg(agg).unstack(split).sort_index()
    except Exception as e:
        st.error(f"Could not build chart: {type(e).__name__}: {str(e)[:300]}")
        return

    st.line_chart(plot, use_container_width=True)
    if x not in date_cols:
        st.caption(f"`{x}` is not a date column – the line connects values in sorted order.")


def table_or_chart(df: pd.DataFrame, key: str, **table_kwargs) -> pd.DataFrame:
    """
    Radio toggle: Table (via `table_widget`, extra kwargs are passed through) or Line chart.

    Returns the rows currently shown: the filtered DataFrame in table mode,
    or `df` unchanged in chart mode.
    """
    mode = st.radio("View as", ["📋 Table", "📈 Line chart"], horizontal=True,
                    key=f"{key}_mode", label_visibility="collapsed")
    if mode.startswith("📋"):
        return table_widget(df, key=key, **table_kwargs)
    line_chart_widget(df, key)
    return df
