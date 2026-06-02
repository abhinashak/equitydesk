"""
ui/pages/page_data_ticker.py  –  Data › Ticker + Benchmark
"""

import time
import streamlit as st
from bll.ticker_service import TickerService

_RUNNING_TICKER_HIST  = "ticker_hist_running"
_RUNNING_TICKER_LIVE  = "ticker_live_running"
_RUNNING_BENCH_HIST   = "bench_hist_running"
_RUNNING_BENCH_LIVE   = "bench_live_running"


def render():
    st.header("📈 Market Data")

    app_cfg        = st.session_state.get("app_cfg", {})
    ticker_root    = app_cfg.get("TICKER_DATA_DIR",    "data/ticker")
    benchmark_root = app_cfg.get("BENCHMARK_DATA_DIR", "data/benchmark")
    ticker_cfg     = app_cfg.get("TICKER_CONFIG",      "config/tickers.csv")
    benchmark_cfg  = app_cfg.get("BENCHMARK_CONFIG",   "config/benchmarks.csv")

    ticker_svc = TickerService(data_root=ticker_root)
    bench_svc  = TickerService(data_root=benchmark_root)   # same pipeline, different roots

    # ── Stats cards ───────────────────────────────────────────────────────────
    t_stats = ticker_svc.get_stats()
    b_stats = bench_svc.get_stats()

    st.subheader("Tickers")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbols",     t_stats["n_tickers"])
    c2.metric("Data points", f"{t_stats['n_points']:,}" if isinstance(t_stats["n_points"], int) else t_stats["n_points"])
    c3.metric("Last candle", str(t_stats["last_date"]))
    c4.metric("Data root",   ticker_root)

    st.subheader("Benchmarks")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Symbols",     b_stats["n_tickers"])
    b2.metric("Data points", f"{b_stats['n_points']:,}" if isinstance(b_stats["n_points"], int) else b_stats["n_points"])
    b3.metric("Last candle", str(b_stats["last_date"]))
    b4.metric("Data root",   benchmark_root)

    st.divider()

    # ── Tabs: Ticker Historical | Ticker Live | Benchmark Historical | Benchmark Live
    tab_t_hist, tab_t_live, tab_b_hist, tab_b_live = st.tabs([
        "📊 Ticker — Historical",
        "⚡ Ticker — Live",
        "📊 Benchmark — Historical",
        "⚡ Benchmark — Live",
    ])

    # ── Ticker Historical ─────────────────────────────────────────────────────
    with tab_t_hist:
        st.caption(f"Config: `{ticker_cfg}`  →  `{ticker_root}`")
        c1, c2 = st.columns([1, 1])
        rebase  = c1.checkbox("Enable rebase detection", value=False, key="t_rebase")
        running = st.session_state.get(_RUNNING_TICKER_HIST, False)
        if c2.button("▶ Run Ticker Loader", type="primary",
                     key="run_t_hist", disabled=running):
            st.session_state[_RUNNING_TICKER_HIST] = True
            _run_blocking(ticker_svc.run_historical(config_file=ticker_cfg, check_rebase=rebase))
            st.session_state[_RUNNING_TICKER_HIST] = False
            st.rerun()

    # ── Ticker Live ───────────────────────────────────────────────────────────
    with tab_t_live:
        st.caption(f"Config: `{ticker_cfg}`  — downloads today's candle only")
        running = st.session_state.get(_RUNNING_TICKER_LIVE, False)
        if st.button("▶ Run Live Loader", type="primary",
                     key="run_t_live", disabled=running):
            st.session_state[_RUNNING_TICKER_LIVE] = True
            _run_blocking(ticker_svc.run_live(config_file=ticker_cfg))
            st.session_state[_RUNNING_TICKER_LIVE] = False
            st.rerun()

    # ── Benchmark Historical ──────────────────────────────────────────────────
    with tab_b_hist:
        st.caption(f"Config: `{benchmark_cfg}`  →  `{benchmark_root}`")
        c1, c2 = st.columns([1, 1])
        rebase  = c1.checkbox("Enable rebase detection", value=False, key="b_rebase")
        running = st.session_state.get(_RUNNING_BENCH_HIST, False)
        if c2.button("▶ Run Benchmark Loader", type="primary",
                     key="run_b_hist", disabled=running):
            st.session_state[_RUNNING_BENCH_HIST] = True
            _run_blocking(bench_svc.run_historical(config_file=benchmark_cfg, check_rebase=rebase))
            st.session_state[_RUNNING_BENCH_HIST] = False
            st.rerun()

    # ── Benchmark Live ────────────────────────────────────────────────────────
    with tab_b_live:
        st.caption(f"Config: `{benchmark_cfg}`  — downloads today's candle only")
        running = st.session_state.get(_RUNNING_BENCH_LIVE, False)
        if st.button("▶ Run Benchmark Live Loader", type="primary",
                     key="run_b_live", disabled=running):
            st.session_state[_RUNNING_BENCH_LIVE] = True
            _run_blocking(bench_svc.run_live(config_file=benchmark_cfg))
            st.session_state[_RUNNING_BENCH_LIVE] = False
            st.rerun()


# ── Blocking stream ───────────────────────────────────────────────────────────

def _run_blocking(gen):
    """Stream generator output into a full-width code block."""
    log_box = st.empty()
    lines: list[str] = []
    for line in gen:
        lines.append(line)
        log_box.code("\n".join(lines), language="text")
        time.sleep(0)