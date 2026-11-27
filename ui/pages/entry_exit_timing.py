"""
Streamlit page: Entry / Exit timing analysis.

Place in your pages package (e.g. ui/pages/entry_exit_timing.py) next to entry_timing.py and
exit_timing.py (or adjust the imports below) and call `render()` from your router.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from ui.components.table_widget import table_widget

from algo.entry_timing import analyze_entry_timing
from algo.exit_timing import analyze_exit_timing

SECTOR_LEVELS = ["auto", "industry", "sector_group", "macro_sector", "all"]
DEFAULT_TRADES = "fasta_all_trades_complete_run.csv"
SS = "eet_"  # session_state prefix


def _parse_tickers(text: str) -> list[str]:
    raw = text.replace("\n", ",").replace(";", ",").replace(" ", ",")
    return list(dict.fromkeys(t.strip().upper() for t in raw.split(",") if t.strip()))


def _save_upload(upload) -> str:
    """analyze_* take a file path, so persist the uploaded CSV to a temp file (once per upload)."""
    tmp = Path(tempfile.gettempdir()) / f"eet_{upload.file_id if hasattr(upload, 'file_id') else upload.name}"
    tmp.write_bytes(upload.getvalue())
    return str(tmp)


def _run(kind: str, fn, tickers: list[str], params: dict, verbose: int, progress, base_pct: float, span: float):
    """Run one analysis ticker-by-ticker (for progress); return (result_df, captured_text)."""
    frames, log = [], io.StringIO()
    for i, t in enumerate(tickers):
        progress.progress(min(1.0, base_pct + span * i / max(len(tickers), 1)), text=f"{kind}: {t}")
        with contextlib.redirect_stdout(log):
            frames.append(fn(t, verbose=verbose, **params))
    return pd.concat(frames, ignore_index=True), log.getvalue()


def _confidence_cfg(prefix: str = "") -> dict:
    return {
        f"{prefix}CONFIDENCE": st.column_config.ProgressColumn(
            "Confidence", min_value=0, max_value=100, format="%.0f"),
        f"{prefix}confidence_label": st.column_config.TextColumn("Label", width="small"),
    }


def render():
    st.markdown("""
    <style>
    .eet-sub {color:#6b7280; font-size:0.9rem; margin:-0.4rem 0 0.8rem 0;}
    .eet-card {border:1px solid rgba(128,128,128,.25); border-radius:10px; padding:.6rem .9rem; margin-bottom:.6rem;}
    .eet-card b {font-size:1.25rem;}
    div[data-testid="stExpander"] {border-radius:10px;}
    </style>
    """, unsafe_allow_html=True)

    st.title("Entry / Exit Timing")
    st.markdown('<div class="eet-sub">Learns, per ticker, the best entry time and the best weekend-exit time '
                'from past back-test trades, optimising <b>net profit after transaction costs</b>.</div>',
                unsafe_allow_html=True)

    # ------------------------------------------------------------------ inputs
    with st.form("eet_form"):
        c1, c2 = st.columns([3, 2])
        with c1:
            tickers_txt = st.text_area("Tickers (comma-separated)", key=SS + "tickers", height=90,
                                       placeholder="RELIANCE, TCS, INFY.NS")
        with c2:
            analysis = st.radio("Analysis", ["Entry + Exit", "Entry only", "Exit only"], key=SS + "analysis")
            verbose = st.select_slider("Diagnostics detail (0 = none, 4 = full)", options=[0, 1, 2, 3, 4],
                                       value=0, key=SS + "verbose")

        c3, c4 = st.columns(2)
        with c3:
            upload = st.file_uploader("Trades CSV (optional)", type="csv", key=SS + "upload")
        with c4:
            trades_path = st.text_input("…or trades CSV path", value=DEFAULT_TRADES, key=SS + "trades")
            strategy = st.text_input("Strategy filter (optional)", key=SS + "strategy")

        with st.expander("Advanced parameters"):
            a1, a2, a3, a4 = st.columns(4)
            stop_loss = a1.number_input("Stop-loss %", value=-2.0, max_value=0.0, step=0.5, key=SS + "sl")
            sector_level = a2.selectbox("Sector level", SECTOR_LEVELS, key=SS + "sector")
            min_peers = a3.number_input("Min peers", 1, 50, 3, key=SS + "peers")
            min_edge = a4.number_input("Min edge % per trade", 0.0, 5.0, 0.10, 0.05, key=SS + "edge")
            b1, b2, b3, b4 = st.columns(4)
            min_past = b1.number_input("Min Past trades", 1, 500, 30, key=SS + "mpast")
            min_recent = b2.number_input("Min Recent trades", 1, 200, 10, key=SS + "mrecent")
            min_bucket = b3.number_input("Min bucket size", 1, 100, 8, key=SS + "mbucket")
            use_date = b4.checkbox("Split by date", key=SS + "usedate")
            if use_date:
                recent_since = st.date_input("Recent starts on", key=SS + "since")
                recent_frac = 0.3
            else:
                recent_frac = st.slider("Recent fraction", 0.05, 0.60, 0.30, 0.05, key=SS + "frac")
                recent_since = None

        go = st.form_submit_button("Run analysis", type="primary", use_container_width=True)

    # ------------------------------------------------------------------ run
    if go:
        tickers = _parse_tickers(tickers_txt)
        if not tickers:
            st.warning("Enter at least one ticker.")
        else:
            path = _save_upload(upload) if upload is not None else trades_path
            if upload is None and not Path(path).exists():
                st.error(f"Trades file not found: {path}")
            else:
                params = dict(
                    trades=path, strategy=strategy.strip() or None, stop_loss=float(stop_loss),
                    sector_level=sector_level, min_peers=int(min_peers), recent_frac=float(recent_frac),
                    recent_since=recent_since.strftime("%Y-%m-%d") if recent_since else None,
                    min_past=int(min_past), min_recent=int(min_recent), min_bucket=int(min_bucket),
                    min_edge=float(min_edge),
                )
                do_entry, do_exit = analysis != "Exit only", analysis != "Entry only"
                prog = st.progress(0.0, text="Starting…")
                out = {"tickers": tickers}
                try:
                    share = 0.5 if (do_entry and do_exit) else 1.0
                    if do_entry:
                        out["entry"], out["entry_log"] = _run(
                            "Entry", analyze_entry_timing, tickers, {**params, "out_dir": "entry_timing_out"},
                            verbose, prog, 0.0, share)
                    if do_exit:
                        out["exit"], out["exit_log"] = _run(
                            "Exit", analyze_exit_timing, tickers, {**params, "out_dir": "exit_timing_out"},
                            verbose, prog, 0.5 if do_entry else 0.0, share)
                    prog.progress(1.0, text="Done")
                    st.session_state[SS + "result"] = out
                except Exception as exc:  # surface DB / file problems instead of a stack trace
                    prog.empty()
                    st.error(f"Analysis failed: {exc}")
                    st.session_state.pop(SS + "result", None)

    res = st.session_state.get(SS + "result")
    if not res:
        st.info("Enter tickers and press **Run analysis**.")
        return

    # ------------------------------------------------------------------ results
    entry, exit_ = res.get("entry"), res.get("exit")
    frames = [df for df in (entry, exit_) if df is not None]

    m = st.columns(4)
    ok_conf = pd.concat([f["CONFIDENCE"] for f in frames]).dropna()
    m[0].metric("Tickers", len(res["tickers"]))
    m[1].metric("Analyses with a result", int(len(ok_conf)))
    m[2].metric("Avg confidence", f"{ok_conf.mean():.0f}" if len(ok_conf) else "–")
    m[3].metric("HIGH confidence", int(sum((f["confidence_label"] == "HIGH").sum() for f in frames)))

    tab_names = []
    if entry is not None and exit_ is not None:
        tab_names.append("Combined")
    if entry is not None:
        tab_names.append("Entry")
    if exit_ is not None:
        tab_names.append("Exit")
    tabs = dict(zip(tab_names, st.tabs(tab_names)))

    if "Combined" in tabs:
        with tabs["Combined"]:
            comb = entry.rename(columns={"CONFIDENCE": "entry_confidence", "confidence_label": "entry_label"}).merge(
                exit_.rename(columns={"CONFIDENCE": "exit_confidence", "confidence_label": "exit_label"}),
                on="ticker", how="outer")
            table_widget(comb, key="eet_combined", height=460, column_config={
                "entry_details": st.column_config.TextColumn("Entry policy", width="large"),
                "exit_details": st.column_config.TextColumn("Exit policy", width="large"),
                "entry_confidence": st.column_config.ProgressColumn("Entry conf.", min_value=0, max_value=100, format="%.0f"),
                "exit_confidence": st.column_config.ProgressColumn("Exit conf.", min_value=0, max_value=100, format="%.0f"),
            })
            st.download_button("Download combined CSV", comb.to_csv(index=False).encode(),
                               "entry_exit_timing.csv", "text/csv", key="eet_dl_comb")

    for name, df, col in (("Entry", entry, "entry_details"), ("Exit", exit_, "exit_details")):
        if df is None:
            continue
        with tabs[name]:
            table_widget(df, key=f"eet_{name.lower()}", height=420, column_config={
                col: st.column_config.TextColumn(f"{name} policy", width="large"), **_confidence_cfg()})
            st.download_button(f"Download {name.lower()} CSV", df.to_csv(index=False).encode(),
                               f"{name.lower()}_timing.csv", "text/csv", key=f"eet_dl_{name.lower()}")
            log = res.get(f"{name.lower()}_log", "").strip()
            if log:
                with st.expander(f"{name} diagnostics"):
                    st.code(log, language="text")

    st.caption("Confidence is a 0–100 heuristic that the Past-learned policy still holds in Recent trades; "
               "it is a guide, not a probability. Per-ticker CSVs are written to entry_timing_out/ and exit_timing_out/.")


if __name__ == "__main__":
    render()
