"""
ui/pages/page_signals.py  –  Screens › Signals
"""

import time
import streamlit as st
from bll.signal_service import SignalService

_RUNNING_KEY  = "signal_gen_running"
_JUST_RAN_KEY = "signal_gen_just_finished"

# ── Data-quality checks ───────────────────────────────────────────────────────
# Each check: (title, sql_template).  {threshold} is filled from the UI.
# Add more checks here (nulls, duplicates, outliers, ...) and they render
# automatically in the Data Quality tab.

_STALENESS_SQL = """
SELECT
    date_diff('day', CAST(date AS DATE), CURRENT_DATE) AS stale_days,
    ticker, date
FROM signals
QUALIFY date = MAX(date) OVER (PARTITION BY ticker)
    AND date_diff('day', CAST(date AS DATE), CURRENT_DATE) > {threshold}
ORDER BY stale_days DESC, ticker
"""


def _stale_df(svc, threshold: int):
    """Return DataFrame of tickers whose latest row is older than `threshold` days."""
    return svc.run_sql_screen(_STALENESS_SQL.format(threshold=int(threshold)))


def render():
    st.header("📡 Signals")

    app_cfg    = st.session_state.get("app_cfg", {})
    signal_dir = app_cfg.get("SIGNAL_DIR",    "data/signal_momentum")
    ticker_cfg = app_cfg.get("TICKER_CONFIG", "config/tickers.csv")
    svc        = SignalService(signal_dir=signal_dir)

    # ── Run button row (full width — log box MUST be outside any column) ──────
    running = st.session_state.get(_RUNNING_KEY, False)
    st.caption(f"Config: `{ticker_cfg}`  →  `{signal_dir}`")

    if st.button("▶ Generate Signals", type="primary",
                 key="run_signals", disabled=running):
        st.session_state[_RUNNING_KEY] = True
        # log box rendered here is full-width because we are NOT inside st.columns
        _run_blocking(svc.run_signal_generation(config_file=ticker_cfg))
        st.session_state[_RUNNING_KEY] = False
        st.session_state[_JUST_RAN_KEY] = True   # survives the rerun below
        st.rerun()

    # ── Post-run quality banner (shown once, right after generation ends) ─────
    if st.session_state.pop(_JUST_RAN_KEY, False):
        _render_post_run_banner(svc)

    st.divider()

    # ── View / Screen / Quality tabs ──────────────────────────────────────────
    tab_view, tab_screen, tab_quality = st.tabs(
        ["Latest Signals", "SQL Screen", "Data Quality"]
    )

    with tab_view:
        df = svc.get_latest_signals()
        if df is None or df.empty:
            st.info("No signal data found. Click **▶ Generate Signals** above to run the generator.")
        else:
            st.dataframe(df.head(50), use_container_width=True, hide_index=True)

    with tab_screen:
        query = st.text_area(
            "DuckDB SQL  (table: `signals`)",
            value="SELECT * FROM signals ORDER BY momentum_1m DESC LIMIT 20",
            height=120,
        )
        if st.button("▶ Run Screen", type="primary"):
            result = svc.run_sql_screen(query)
            st.dataframe(result, use_container_width=True, hide_index=True)

    with tab_quality:
        _render_quality_tab(svc)


# ── Data quality UI ───────────────────────────────────────────────────────────

def _render_post_run_banner(svc, threshold: int = 2):
    """One-line summary shown immediately after a generation run."""
    try:
        stale = _stale_df(svc, threshold)
    except Exception as e:
        st.warning(f"Signals generated, but the staleness check failed: {e}")
        return

    if stale is None or stale.empty:
        st.success(f"✅ Signals generated. No tickers stale by more than {threshold} days.")
    else:
        st.warning(
            f"⚠️ Signals generated, but **{len(stale)}** ticker(s) are stale by more "
            f"than {threshold} days. See the **Data Quality** tab."
        )


def _render_quality_tab(svc):
    st.subheader("Data staleness")
    st.caption(
        "Tickers whose most recent signal date is older than the threshold "
        "(calendar days vs. today)."
    )

    threshold = st.number_input(
        "Stale if older than N days", min_value=0, max_value=365,
        value=2, step=1, key="dq_stale_threshold",
    )

    try:
        stale = _stale_df(svc, threshold)
    except Exception as e:
        st.error(f"Staleness query failed: {e}")
        return

    if stale is None or stale.empty:
        st.success(f"No stale tickers (all updated within {int(threshold)} days).")
        return

    c1, c2 = st.columns(2)
    c1.metric("Stale tickers", len(stale))
    c2.metric("Worst staleness (days)", int(stale["stale_days"].max()))

    st.dataframe(stale, use_container_width=True, hide_index=True)

    with st.expander("SQL used"):
        st.code(_STALENESS_SQL.format(threshold=int(threshold)), language="sql")


# ── Blocking stream ───────────────────────────────────────────────────────────

def _run_blocking(gen):
    """
    Consume a line-yielding generator synchronously, rendering output into a
    full-width code block.  Must be called outside any st.columns() context
    so Streamlit gives the code block the full page width.
    """
    log_box = st.empty()
    lines: list[str] = []
    for line in gen:
        lines.append(line)
        log_box.code("\n".join(lines), language="text")
        time.sleep(0)