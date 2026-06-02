"""
ui/pages/page_signals.py  –  Screens › Signals
"""

import time
import streamlit as st
from bll.signal_service import SignalService

_RUNNING_KEY = "signal_gen_running"


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
        st.rerun()

    st.divider()

    # ── View / Screen tabs ────────────────────────────────────────────────────
    tab_view, tab_screen = st.tabs(["Latest Signals", "SQL Screen"])

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