"""
ui/components/table_widget.py
─────────────────────────────
Reusable filterable and sortable Streamlit table.

Usage:
    table_widget(sql, key="my_table")
    table_widget(df, key="my_table")

SQL strings are executed (read-only) through `utils.db_utils.run_readonly_query`.
Column filters:
    numeric               → range slider
    date / time           → date range
    boolean               → multiselect
    low-cardinality text  → multiselect
    high-cardinality text → "contains" search
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from utils.db_utils import run_readonly_query


_MAX_CHOICES = 100      # above this many distinct values, use a "contains" box instead


def _normalize_symbols(symbols: list[str] | None) -> set[str]:
    return {str(s).strip().upper() for s in (symbols or []) if s}


def _filter_dataframe(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Render column-specific filters and return the filtered DataFrame."""
    if df is None or df.empty:
        return df

    # Widget keys include the column set, so editing a query never reuses stale state.
    signature = abs(hash(tuple(
        (str(col), str(dtype)) for col, dtype in zip(df.columns, df.dtypes)
    ))) % 10**8
    prefix = f"table_filter_{key}_{signature}"

    with st.expander("🔎 Filters", expanded=False):
        selected_cols = st.multiselect(
            "Filter on columns",
            list(df.columns),
            key=f"{prefix}_columns",
            placeholder="Choose one or more columns…",
        )

        mask = pd.Series(True, index=df.index)

        for col in selected_cols:
            series = df[col]
            widget_key = f"{prefix}_{col}"

            # Boolean
            if pd.api.types.is_bool_dtype(series):
                selected = st.multiselect(col, [True, False], key=widget_key)
                if selected:
                    mask &= series.isin(selected)

            # Numeric
            elif pd.api.types.is_numeric_dtype(series):
                clean = series.dropna()
                if clean.empty:
                    st.caption(f"{col}: no values")
                    continue

                minimum, maximum = clean.min(), clean.max()
                if minimum == maximum:
                    st.caption(f"{col}: all rows = {minimum}")
                    continue

                if pd.api.types.is_integer_dtype(series):
                    minimum, maximum = int(minimum), int(maximum)
                else:
                    minimum, maximum = float(minimum), float(maximum)

                low, high = st.slider(
                    col,
                    min_value=minimum,
                    max_value=maximum,
                    value=(minimum, maximum),
                    key=widget_key,
                )
                if (low, high) != (minimum, maximum):
                    mask &= series.between(low, high)

            # Datetime
            elif pd.api.types.is_datetime64_any_dtype(series):
                clean = series.dropna()
                if clean.empty:
                    st.caption(f"{col}: no values")
                    continue

                minimum, maximum = clean.min().date(), clean.max().date()
                selected_range = st.date_input(
                    col,
                    value=(minimum, maximum),
                    min_value=minimum,
                    max_value=maximum,
                    key=widget_key,
                )
                # While the user is mid-selection only one date is returned; ignore it.
                if isinstance(selected_range, (tuple, list)) and len(selected_range) == 2:
                    mask &= series.dt.date.between(selected_range[0], selected_range[1])

            # Text / categorical
            else:
                text_series = series.astype("string")
                unique_values = text_series.dropna().unique().tolist()

                if len(unique_values) <= _MAX_CHOICES:
                    selected = st.multiselect(
                        col, sorted(unique_values, key=str), key=widget_key
                    )
                    if selected:
                        mask &= text_series.isin(selected)
                else:
                    search_text = st.text_input(f"{col} contains", key=widget_key)
                    if search_text:
                        mask &= text_series.str.contains(
                            search_text, case=False, na=False, regex=False
                        )

        filtered = df.loc[mask]

        if selected_cols:
            st.caption(f"Showing **{len(filtered):,}** of **{len(df):,}** rows")

    return filtered


def _highlight_portfolio(df: pd.DataFrame, portfolio_set: set[str]):
    """Return a Styler highlighting rows whose `ticker` is in `portfolio_set`."""

    def row_style(row: pd.Series) -> list[str]:
        if str(row.get("ticker", "")).strip().upper() in portfolio_set:
            return ["background-color: #fff8c5; color: #1a1a1a;"] * len(row)
        return [""] * len(row)

    return df.style.apply(row_style, axis=1)


def table_widget(
    data: str | pd.DataFrame,
    *,
    key: str = "table",
    height: int | None = 420,
    max_rows: int | None = None,
    portfolio_syms: list[str] | None = None,
    highlight_portfolio: bool = True,
    show_filters: bool = True,
    hide_index: bool = True,
    column_config: dict | None = None,
) -> pd.DataFrame:
    """
    Display a sortable, filterable table.

    Parameters
    ----------
    data:
        A read-only SQL query (run via `utils.db_utils`) or a pandas DataFrame.
    key:
        Unique key for Streamlit widgets. Required when several tables share a page.
    height:
        Table height in pixels; None lets Streamlit size it to the content.
    max_rows:
        Maximum number of filtered rows displayed. None means unlimited.
    portfolio_syms:
        Optional list of portfolio tickers to highlight.
    highlight_portfolio:
        Highlight matching rows when a `ticker` column exists.
    show_filters:
        Show or hide the expandable column filters.
    hide_index:
        Hide the DataFrame index.
    column_config:
        Optional `st.column_config` mapping (labels, number formats, progress bars …).

    Returns
    -------
    pandas.DataFrame
        The filtered DataFrame, before `max_rows` is applied.
    """

    # Accept either SQL or a DataFrame.
    if isinstance(data, str):
        try:
            df = run_readonly_query(data)
        except Exception as exc:
            st.error(f"SQL error: {exc}")
            return pd.DataFrame()
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("table_widget expects a SQL string or pandas DataFrame.")

    filtered = _filter_dataframe(df, key) if show_filters else df
    if filtered is None:
        return pd.DataFrame()

    display_df = filtered
    if max_rows is not None:
        display_df = display_df.head(max_rows)
        if len(filtered) > max_rows:
            st.caption(f"Displaying first {max_rows:,} of {len(filtered):,} filtered rows.")

    # Portfolio highlighting and ticker summary.
    portfolio_set = _normalize_symbols(portfolio_syms)
    if highlight_portfolio and portfolio_set and "ticker" in display_df.columns:
        ticker_values = display_df["ticker"].astype("string").str.strip().str.upper()
        hits = ticker_values.isin(portfolio_set)
        hit_count = int(hits.sum())

        if hit_count:
            hit_tickers = sorted(ticker_values[hits].dropna().unique().tolist())
            badges = "".join(
                '<span style="display:inline-block;margin:0 2px;'
                'padding:1px 6px;background:#fff3cd;'
                'border:1px solid #ffc107;border-radius:4px;'
                'font-size:0.72rem;font-weight:600;color:#856404;">'
                f'{html.escape(ticker)}</span>'
                for ticker in hit_tickers
            )
            st.markdown(
                '<span style="font-size:0.78rem;color:#856404;">'
                f'⭐ <strong>{hit_count}</strong> portfolio row(s) '
                f'highlighted:&nbsp;{badges}</span>',
                unsafe_allow_html=True,
            )
            with st.popover("📋 Copy tickers"):
                st.code("\n".join(hit_tickers), language=None)

            display_df = _highlight_portfolio(display_df, portfolio_set)

    # st.dataframe provides interactive column sorting.
    df_kwargs = {"height": height} if height is not None else {}   # omit -> Streamlit's default sizing
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=hide_index,
        column_config=column_config,
        **df_kwargs,
    )

    return filtered
