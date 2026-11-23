"""
ui/pages/page_data_ticker.py  –  Data › Ticker + Benchmark (+ Quality report)
"""

import time
import streamlit as st
from bll.ticker_service import TickerService

_RUNNING_TICKER_HIST  = "ticker_hist_running"
_RUNNING_TICKER_LIVE  = "ticker_live_running"
_RUNNING_BENCH_HIST   = "bench_hist_running"
_RUNNING_BENCH_LIVE   = "bench_live_running"
_QUALITY_RESULT       = "quality_result"


def render():
    st.header("📈 Market Data")

    app_cfg        = st.session_state.get("app_cfg", {})
    ticker_root    = app_cfg.get("TICKER_DATA_DIR",    "data/ticker")
    benchmark_root = app_cfg.get("BENCHMARK_DATA_DIR", "data/benchmark")
    ticker_cfg     = app_cfg.get("TICKER_CONFIG",      "config/tickers.csv")
    benchmark_cfg  = app_cfg.get("BENCHMARK_CONFIG",   "config/benchmarks.csv")

    ticker_svc = TickerService(data_root=ticker_root)
    bench_svc  = TickerService(data_root=benchmark_root)   # same pipeline, different roots

    # ── Stats cards (one bad root must not crash the page) ───────────────────
    st.subheader("Tickers")
    _stats_row(_safe_stats(ticker_svc, "Ticker"), ticker_root)

    st.subheader("Benchmarks")
    _stats_row(_safe_stats(bench_svc, "Benchmark"), benchmark_root)

    st.divider()

    tab_t_hist, tab_t_live, tab_b_hist, tab_b_live, tab_q = st.tabs([
        "📊 Ticker — Historical",
        "⚡ Ticker — Live",
        "📊 Benchmark — Historical",
        "⚡ Benchmark — Live",
        "🩺 Quality Report",
    ])

    # ── Ticker Historical ─────────────────────────────────────────────────────
    with tab_t_hist:
        st.caption(f"Config: `{ticker_cfg}`  →  `{ticker_root}`")
        c1, c2 = st.columns([1, 1])
        rebase = c1.checkbox("Enable rebase detection", value=False, key="t_rebase")
        if c2.button("▶ Run Ticker Loader", type="primary", key="run_t_hist",
                     disabled=st.session_state.get(_RUNNING_TICKER_HIST, False)):
            _run_job(_RUNNING_TICKER_HIST,
                     lambda: ticker_svc.run_historical(config_file=ticker_cfg, check_rebase=rebase))

    # ── Ticker Live ───────────────────────────────────────────────────────────
    with tab_t_live:
        st.caption(f"Config: `{ticker_cfg}`  — downloads today's candle only")
        if st.button("▶ Run Live Loader", type="primary", key="run_t_live",
                     disabled=st.session_state.get(_RUNNING_TICKER_LIVE, False)):
            _run_job(_RUNNING_TICKER_LIVE,
                     lambda: ticker_svc.run_live(config_file=ticker_cfg))

    # ── Benchmark Historical ──────────────────────────────────────────────────
    with tab_b_hist:
        st.caption(f"Config: `{benchmark_cfg}`  →  `{benchmark_root}`")
        c1, c2 = st.columns([1, 1])
        rebase = c1.checkbox("Enable rebase detection", value=False, key="b_rebase")
        if c2.button("▶ Run Benchmark Loader", type="primary", key="run_b_hist",
                     disabled=st.session_state.get(_RUNNING_BENCH_HIST, False)):
            _run_job(_RUNNING_BENCH_HIST,
                     lambda: bench_svc.run_historical(config_file=benchmark_cfg, check_rebase=rebase))

    # ── Benchmark Live ────────────────────────────────────────────────────────
    with tab_b_live:
        st.caption(f"Config: `{benchmark_cfg}`  — downloads today's candle only")
        if st.button("▶ Run Benchmark Live Loader", type="primary", key="run_b_live",
                     disabled=st.session_state.get(_RUNNING_BENCH_LIVE, False)):
            _run_job(_RUNNING_BENCH_LIVE,
                     lambda: bench_svc.run_live(config_file=benchmark_cfg))

    # ── Quality report ────────────────────────────────────────────────────────
    with tab_q:
        _render_quality([
            ("Tickers",    ticker_svc, ticker_cfg),
            ("Benchmarks", bench_svc,  benchmark_cfg),
        ])


# ── Stats helpers ─────────────────────────────────────────────────────────────

def _safe_stats(svc, label: str) -> dict:
    try:
        return svc.get_stats()
    except Exception as e:
        st.error(f"{label} stats failed: {type(e).__name__}: {e}")
        return {"n_tickers": "—", "n_points": "—", "last_date": "—"}


def _stats_row(stats: dict, root: str):
    c1, c2, c3, c4 = st.columns(4)
    n = stats.get("n_points", "—")
    c1.metric("Symbols",     stats.get("n_tickers", "—"))
    c2.metric("Data points", f"{n:,}" if isinstance(n, int) else n)
    c3.metric("Last candle", str(stats.get("last_date", "—")))
    c4.metric("Data root",   root)


# ── Job runner ────────────────────────────────────────────────────────────────

def _run_job(flag_key: str, make_gen):
    """Run a streaming job. The running flag is ALWAYS cleared (even on error),
    otherwise one failure leaves the button disabled forever."""
    st.session_state[flag_key] = True
    try:
        _run_blocking(make_gen())
    except Exception as e:
        st.session_state[flag_key] = False
        st.error(f"Job failed: {type(e).__name__}: {e}")
        st.exception(e)
        return                      # no rerun, so the error stays visible
    st.session_state[flag_key] = False
    st.rerun()


def _run_blocking(gen):
    """Stream generator output into a full-width code block."""
    log_box = st.empty()
    lines: list[str] = []
    for line in gen:
        lines.append(str(line))
        log_box.code("\n".join(lines), language="text")
        time.sleep(0)


# ── Quality report (rendering only; logic lives in TickerService) ────────────

def _render_quality(targets: list):
    st.caption("Symbols whose last candle is older than the newest candle in the dataset.")
    c1, c2 = st.columns([1, 1])
    tol = c1.number_input("Tolerance (business days)", min_value=0, max_value=30,
                          value=0, step=1, key="q_tol")
    run = c2.button("🩺 Run quality scan", type="primary", key="q_run")

    if run:
        results = {}
        with st.spinner("Scanning symbols…"):
            for label, svc, cfg in targets:
                try:
                    results[label] = {
                        "issues": svc.get_config_issues(cfg),
                        "report": svc.get_stale_report(cfg, int(tol)),
                    }
                except Exception as e:
                    results[label] = {"error": f"{type(e).__name__}: {e}"}
        st.session_state[_QUALITY_RESULT] = results

    results = st.session_state.get(_QUALITY_RESULT)
    if not results:
        st.info("Click **Run quality scan** to check last-candle freshness.")
        return

    for label, _svc, _cfg in targets:
        st.subheader(label)
        res = results.get(label, {})
        if "error" in res:
            st.error(f"{label} scan failed: {res['error']}")
            continue

        for issue in res["issues"]:
            st.warning(f"Config: {issue}")

        rep = res["report"]
        ref, total, df = rep["reference"], rep["total"], rep["stale"]
        if ref is None:
            st.info("No readable data found for any configured symbol.")
        elif df.empty:
            st.success(f"All {total} symbols are up to date (last candle {ref.date()}).")
        else:
            st.warning(f"{len(df)} of {total} symbols are stale "
                       f"(reference last candle: {ref.date()}).")
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(f"⬇ Download {label} stale report (CSV)",
                               df.to_csv(index=False).encode(),
                               file_name=f"stale_{label.lower()}_{ref.date()}.csv",
                               mime="text/csv", key=f"q_dl_{label}")