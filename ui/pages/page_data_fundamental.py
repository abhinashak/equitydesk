"""
ui/pages/page_data_fundamental.py  -  Data > Fundamental

Flow
----
1. Show stale/missing tickers.
2. User selects tickers -> copy the Selenium run command.
3. User runs screener_selenium.py in their terminal -> <TICKER>.html files saved.
4. User uploads those HTML files via the file uploader.
5. User clicks "Parse & Merge" -> pipeline runs, progress shown live.
"""

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from bll.fundamental_service import FundamentalService
from bll.config_service import ConfigService


# -- Cached data fetchers ------------------------------------------------------
# Decoration is deferred to first call so that importing this module outside a
# live Streamlit session does not trigger "No runtime found" warnings.

_get_coverage_fn = None
_get_stale_fn = None


def _get_coverage(out_dir: str) -> pd.DataFrame:
    global _get_coverage_fn
    if _get_coverage_fn is None:
        @st.cache_data(ttl=300, show_spinner=False)
        def _fn(out_dir: str) -> pd.DataFrame:
            return FundamentalService(out_dir=out_dir).get_coverage()
        _get_coverage_fn = _fn
    return _get_coverage_fn(out_dir)


def _get_stale_or_missing(out_dir: str, symbols: tuple[str, ...], stale_hours: float) -> pd.DataFrame:
    global _get_stale_fn
    if _get_stale_fn is None:
        @st.cache_data(ttl=300, show_spinner=False)
        def _fn(out_dir: str, symbols: tuple[str, ...], stale_hours: float) -> pd.DataFrame:
            return FundamentalService(out_dir=out_dir).get_stale_or_missing(list(symbols), stale_hours=stale_hours)
        _get_stale_fn = _fn
    return _get_stale_fn(out_dir, symbols, stale_hours)


def _clear_caches() -> None:
    if _get_coverage_fn is not None:
        _get_coverage_fn.clear()
    if _get_stale_fn is not None:
        _get_stale_fn.clear()


def render():
    st.header("Fundamental Data")

    app_cfg = st.session_state.get("app_cfg", {})
    out_dir = app_cfg.get("FUNDAMENTAL_DATA_DIR", "data/fundamental")
    cfg_svc = ConfigService()

    # -- Coverage --------------------------------------------------------------
    st.subheader("Coverage")
    coverage = _get_coverage(out_dir)
    has_data = coverage[coverage["tickers"] > 0]
    if has_data.empty:
        st.info("No fundamental data found yet. Run the loader below.")
    else:
        st.dataframe(has_data, use_container_width=True, hide_index=True)

    st.divider()

    # -- Step 1: Select Tickers ------------------------------------------------
    st.subheader("Step 1 — Select Tickers")
    st.caption("Tickers that are stale or missing required tables are shown below.")

    cfg_df = cfg_svc.get_tickers().dropna(subset=["Yahoo Symbol"])
    cfg_df["screener_ticker"] = (
        cfg_df["Yahoo Symbol"]
        .str.replace(r"\.(NS|BO)$", "", regex=True)
        .str.strip()
    )
    ticker_map: dict[str, str] = dict(
        zip(cfg_df["Name"].fillna(cfg_df["screener_ticker"]), cfg_df["screener_ticker"])
    )
    all_symbols = tuple(ticker_map.values())

    stale_hours = st.number_input(
        "Stale threshold (hours)", min_value=1, max_value=8760, value=168,
        key="fund_stale_hours",
    )

    stale_df = _get_stale_or_missing(out_dir, all_symbols, float(stale_hours))

    if stale_df.empty:
        st.success("All tickers are up-to-date and complete — nothing to run.")
        return

    display_df = (
        stale_df
        .sort_values(["reason", "hours_ago"], na_position="first")
        .reset_index(drop=True)
    )
    current_tickers = display_df["ticker"].tolist()

    # -- run-state -------------------------------------------------------------
    RUN_KEY = "fund_run_state"
    if RUN_KEY not in st.session_state:
        st.session_state[RUN_KEY] = {t: True for t in current_tickers}

    run_state: dict[str, bool] = st.session_state[RUN_KEY]
    for t in current_tickers:
        if t not in run_state:
            run_state[t] = True

    ctrl1, ctrl2, _ = st.columns([1, 1, 6])
    if ctrl1.button("Select All", key="sel_all_fundamental"):
        for t in current_tickers:
            run_state[t] = True
    if ctrl2.button("Clear All", key="clear_fundamental"):
        for t in current_tickers:
            run_state[t] = False

    display_df["run"] = display_df["ticker"].map(run_state).fillna(True)

    st.caption("Check / uncheck rows to include, then run the Selenium script.")
    edited = st.data_editor(
        display_df[["run", "ticker", "last_updated", "hours_ago", "missing", "reason"]],
        use_container_width=True,
        hide_index=True,
        column_config={"run": st.column_config.CheckboxColumn("Run", width="small")},
        disabled=["ticker", "last_updated", "hours_ago", "missing", "reason"],
        key="fund_stale_editor",
    )

    for _, row in edited.iterrows():
        run_state[row["ticker"]] = bool(row["run"])

    selected: list[str] = edited.loc[edited["run"], "ticker"].tolist()
    st.caption(f"{len(selected)} of {len(current_tickers)} ticker(s) selected.")

    st.divider()

    # -- Step 2: Node downloader -------------------------------------------
    st.subheader("Step 2 — Download HTML via node")

    if not selected:
        st.info("Select at least one ticker above to see the run command.")
    else:
        tickers_arg = " ".join(selected)
        st.caption("Basic command — paste this in your terminal:")
        js_code = """
        const fs = require("fs");
        const path = require("path");
        
        const tickers = `
        __TICKERS__
        `.trim().split(/\\s+/);
        
        async function downloadTicker(ticker) {
            const url = `https://www.screener.in/company/${ticker}/`;
        
            try {
                const response = await fetch(url, {
                    headers: {
                        "User-Agent": "Mozilla/5.0"
                    }
                });
        
                const html = await response.text();
        
                fs.writeFileSync(
                    path.join(__dirname, `${ticker}.html`),
                    html
                );
        
                console.log(`Saved ${ticker}.html`);
            } catch (err) {
                console.error(`Failed ${ticker}`, err.message);
            }
        }
        
        (async () => {
            for (const ticker of tickers) {
                await downloadTicker(ticker);
        
                // avoid hammering server
                await new Promise(r => setTimeout(r, 15000));
            }
        })();        
        """

        js_code = js_code.replace("__TICKERS__", tickers_arg)

        st.code(js_code, language="javascript")
        st.info(
            "HTML files are saved to `data/screener_html/` by default.  "
            "Upload them below once the script finishes."
        )

    st.divider()

    # -- Step 3: Upload HTML files ---------------------------------------------
    st.subheader("Step 3 — Upload Downloaded HTML Files")
    st.caption(
        "Upload the `<TICKER>.html` files produced by the node script. "
        "The file name must match the ticker symbol (e.g. `RELIANCE.html`)."
    )

    uploaded_files = st.file_uploader(
        "Select HTML files",
        type=["html", "htm"],
        accept_multiple_files=True,
        key="fund_html_uploader",
        label_visibility="collapsed",
    )

    # Map uploaded file -> ticker by filename stem (case-insensitive)
    selected_upper = {t.upper() for t in selected}
    ticker_paths: dict[str, Path] = {}
    unmatched: list[str] = []

    if uploaded_files:
        for f in uploaded_files:
            stem = Path(f.name).stem.upper()
            if stem in selected_upper:
                tmp_path = Path(tempfile.gettempdir()) / f.name
                tmp_path.write_bytes(f.read())
                match = next(t for t in selected if t.upper() == stem)
                ticker_paths[match] = tmp_path
            else:
                unmatched.append(f.name)

        if unmatched:
            st.warning(
                "These files did not match any selected ticker and will be ignored: "
                + ", ".join(unmatched)
            )

        matched = sorted(ticker_paths.keys())
        missing_uploads = [t for t in selected if t not in ticker_paths]

        if matched:
            st.success("Matched: " + ", ".join(matched))
        if missing_uploads:
            st.info("Still waiting for: " + ", ".join(missing_uploads))

    st.divider()

    # -- Step 4: Parse & Merge -------------------------------------------------
    st.subheader("Step 4 — Parse & Merge")

    run_disabled = not ticker_paths
    if run_disabled:
        st.caption("Upload at least one matched HTML file above to enable this button.")

    if st.button(
            "Parse & Merge",
            type="primary",
            disabled=run_disabled,
            key="run_fundamental",
    ):
        _clear_caches()

        loader = FundamentalService(out_dir=out_dir)
        log_area = st.empty()
        lines: list[str] = []

        for msg in loader.run_from_files(ticker_paths):
            lines.append(msg)
            log_area.code("\n".join(lines[-300:]), language="text")