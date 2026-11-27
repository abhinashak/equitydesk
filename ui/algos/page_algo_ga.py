"""
ui/pages/page_algo_ga.py  –  Algo › Genetic Optimizer
"""

import time
from pathlib import Path

import streamlit as st

from algo.ga.ga_service import GAService

# ── Session-state keys ────────────────────────────────────────────────────────
_SK_TRAIN     = "ga_train_running"
_SK_EVAL      = "ga_eval_running"
_SK_RUN       = "ga_current_run_name"
_SK_TICKERS   = "ga_ticker_override"       # list[str] | None
_SK_TRAIN_LOG = "ga_train_log"             # list[str] — persists across rerun
_SK_EVAL_LOG  = "ga_eval_log"             # list[str] — persists across rerun
_SK_PASTE     = "ga_ticker_paste"          # str — raw paste input
_SK_CFG_DF    = "ga_config_df"             # pd.DataFrame — editable ga_config


def render():
    st.header("🧬 Genetic Portfolio Optimizer")

    app_cfg    = st.session_state.get("app_cfg", {})
    base_out   = app_cfg.get("GA_OUT_DIR",    "outputs")
    ticker_cfg = app_cfg.get("TICKER_CONFIG", "config/tickers.csv")

    # ── 1. Experiment run name ────────────────────────────────────────────────
    st.subheader("Experiment")

    existing_runs = GAService.list_runs(base_out)

    col_new, col_existing = st.columns([2, 2])
    with col_new:
        run_name_input = st.text_input(
            "New run name",
            value=st.session_state.get(_SK_RUN, ""),
            placeholder="e.g. iran_war_v1",
            help="Outputs will be saved to outputs/<run_name>/",
        )
    with col_existing:
        if existing_runs:
            pick = st.selectbox(
                "…or load existing run",
                options=["— new run —"] + existing_runs,
                index=0,
            )
            if pick != "— new run —" and st.button("Load", key="ga_load_run"):
                st.session_state[_SK_RUN] = pick
                st.rerun()

    run_name = run_name_input.strip() or st.session_state.get(_SK_RUN, "")
    if run_name:
        st.session_state[_SK_RUN] = run_name

    if not run_name:
        st.info("Enter a run name above to begin.")
        return

    svc = GAService(run_name=run_name, base_out=base_out)
    st.caption(f"Output directory: `{svc.out_dir}`")

    st.divider()

    # ── 2. Ticker selector ────────────────────────────────────────────────────
    st.subheader("Ticker Universe")

    all_tickers = GAService.load_tickers_from_config(ticker_cfg)

    # Build a lookup: bare symbol (no exchange suffix) → full ticker
    # e.g. "INFY" → "INFY.NS",  "COALINDIA" → "COALINDIA.NS"
    _bare_to_full: dict[str, str] = {}
    for t in all_tickers:
        bare = t.split(".")[0].upper()
        _bare_to_full[bare] = t

    mode = st.radio(
        "Which tickers to include in the GA run?",
        options=["All tickers from config (default)", "Custom selection"],
        horizontal=True,
        key="ga_ticker_mode",
    )

    ticker_override: list[str] | None = None

    if mode == "Custom selection":
        if all_tickers:
            # ── Paste box ───────────────────────────────────────────────────
            paste_raw = st.text_area(
                "Paste tickers (comma or newline separated, .NS/.BO optional)",
                value=st.session_state.get(_SK_PASTE, ""),
                height=80,
                placeholder="e.g.  INFY, RELIANCE, TCS.NS, HDFCBANK",
                key="ga_paste_box",
                help="Paste a comma- or newline-separated list. "
                     "Exchange suffixes (.NS / .BO) are optional — they will be "
                     "matched automatically from the ticker config.",
            )
            st.session_state[_SK_PASTE] = paste_raw

            # Parse paste → resolve to full tickers
            pasted_resolved: list[str] = []
            pasted_unmatched: list[str] = []
            if paste_raw.strip():
                import re
                tokens = [t.strip().upper() for t in re.split(r"[,\n]+", paste_raw) if t.strip()]
                for tok in tokens:
                    bare = tok.split(".")[0]
                    if tok in [t.upper() for t in all_tickers]:
                        # Exact full-ticker match (case-insensitive)
                        matched = next(t for t in all_tickers if t.upper() == tok)
                        pasted_resolved.append(matched)
                    elif bare in _bare_to_full:
                        pasted_resolved.append(_bare_to_full[bare])
                    else:
                        pasted_unmatched.append(tok)

                if pasted_unmatched:
                    st.warning(
                        f"⚠️ Could not match {len(pasted_unmatched)} ticker(s) to config: "
                        f"`{'`, `'.join(pasted_unmatched)}`"
                    )

            # Merge paste results into multiselect default (deduped, ordered)
            existing_sel = st.session_state.get(_SK_TICKERS) or []
            merged_default = list(dict.fromkeys(pasted_resolved + existing_sel))

            selected = st.multiselect(
                f"Selected tickers  ({len(all_tickers)} available — edit here or paste above)",
                options=all_tickers,
                default=[t for t in merged_default if t in all_tickers],
                help="You can also type to search, or paste a list in the box above.",
            )
            st.session_state[_SK_TICKERS] = selected
            ticker_override = selected if selected else None
        else:
            st.warning(f"Could not load tickers from `{ticker_cfg}`. "
                       "Check TICKER_CONFIG in app settings.")

    if ticker_override:
        st.caption(f"Running with **{len(ticker_override)}** selected tickers "
                   f"(of {len(all_tickers)} available).")
    else:
        st.caption(f"Running with **all {len(all_tickers)} tickers** from `{ticker_cfg}`.")

    st.divider()

    # ── 3. Period config picker ───────────────────────────────────────────────
    st.subheader("Period Config")

    periods_file = app_cfg.get("PERIODS_CONFIG", "config/periods.json")
    try:
        import json
        with open(periods_file) as _f:
            _all_periods = json.load(_f)
        # Support both a bare list and {"periods": [...]} wrapper
        if isinstance(_all_periods, dict):
            _all_periods = _all_periods.get("periods", [])
        period_names  = [c["name"] for c in _all_periods if not c.get("disabled")]
        config_choice = st.selectbox(
            f"Run config  (`{periods_file}`)",
            options=["All enabled"] + period_names,
            help="'All enabled' runs every non-disabled period in periods.json",
        )
        config_name = None if config_choice == "All enabled" else config_choice
    except FileNotFoundError:
        st.warning(f"`{periods_file}` not found. Check PERIODS_CONFIG in app settings.")
        config_name = None
    except Exception as exc:
        st.warning(f"Could not load `{periods_file}`: {exc}")
        config_name = None

    st.divider()

    # ── 4. GA Config editor ───────────────────────────────────────────────────
    with st.expander("⚙️ GA Config  (edit before training)", expanded=False):
        import pandas as pd

        cfg_path = str(Path(ticker_cfg).parent / "ga_config.csv")

        # Load CSV into session state once (or on explicit reload)
        _reload_cfg = st.button("↺ Reload from disk", key="ga_cfg_reload")
        if _SK_CFG_DF not in st.session_state or _reload_cfg:
            try:
                _df_raw = pd.read_csv(cfg_path)
                st.session_state[_SK_CFG_DF] = _df_raw
            except FileNotFoundError:
                st.error(f"`{cfg_path}` not found. "
                         "Check that TICKER_CONFIG points to the right directory.")
                _df_raw = None
            except Exception as _exc:
                st.error(f"Could not load config: {_exc}")
                _df_raw = None
        else:
            _df_raw = st.session_state[_SK_CFG_DF]

        if _df_raw is not None:
            st.caption(f"Editing `{cfg_path}` — changes are saved to disk when you click **Save**.")

            # Only show section / key / value columns; keep others read-only for context
            _editable_cols = ["section", "key", "value"]
            _show_cols = [c for c in _df_raw.columns if c in _editable_cols + ["dtype", "description"]]

            edited = st.data_editor(
                _df_raw[_show_cols].copy(),
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                disabled=[c for c in _show_cols if c not in _editable_cols],
                column_config={
                    "value": st.column_config.TextColumn("value", width="small"),
                    "key":   st.column_config.TextColumn("key",   width="medium"),
                    "section": st.column_config.TextColumn("section", width="small"),
                },
                key="ga_cfg_editor",
            )

            col_save, col_status = st.columns([1, 4])
            with col_save:
                if st.button("💾 Save ga_config.csv", type="primary", key="ga_cfg_save"):
                    # Merge edited columns back onto original df (preserves any extra columns)
                    _saved = _df_raw.copy()
                    for col in _editable_cols:
                        if col in edited.columns and col in _saved.columns:
                            _saved[col] = edited[col].values
                    try:
                        _saved.to_csv(cfg_path, index=False)
                        st.session_state[_SK_CFG_DF] = _saved
                        with col_status:
                            st.success(f"Saved → `{cfg_path}`")
                    except Exception as _exc:
                        with col_status:
                            st.error(f"Save failed: {_exc}")

    st.divider()

    # ── 5. Train / Eval tabs ──────────────────────────────────────────────────
    tab_train, tab_eval, tab_results = st.tabs(["🏋️ Train", "📊 Evaluate", "📋 Results"])

    # ── Train ─────────────────────────────────────────────────────────────────
    with tab_train:
        st.caption("Runs the GA on the training window and writes weights CSVs.")
        running = st.session_state.get(_SK_TRAIN, False)

        if st.button("▶ Run Training", type="primary",
                     key="ga_run_train", disabled=running):
            st.session_state[_SK_TRAIN] = True
            st.session_state[_SK_TRAIN_LOG] = []
            log = _run_blocking(
                svc.run_train(
                    config_name      = config_name,
                    ticker_override  = ticker_override,
                ),
                log_key=_SK_TRAIN_LOG,
            )
            st.session_state[_SK_TRAIN] = False
            # No st.rerun() — let the log render in place below

        # Persistent log — survives reruns
        if st.session_state.get(_SK_TRAIN_LOG):
            st.code("\n".join(st.session_state[_SK_TRAIN_LOG][-300:]), language="text")
            if st.button("Clear log", key="ga_clear_train_log"):
                st.session_state[_SK_TRAIN_LOG] = []
                st.rerun()

        # Show train summary if available
        summary = svc.get_train_summary()
        if summary is not None and not summary.empty:
            st.markdown("**Last training summary**")
            st.dataframe(summary, use_container_width=True, hide_index=True)

    # ── Eval ──────────────────────────────────────────────────────────────────
    with tab_eval:
        st.caption("Evaluates trained weights on the held-out test window.")
        running = st.session_state.get(_SK_EVAL, False)

        skip_wf = st.checkbox("Skip walk-forward rebalance", value=False, key="ga_skip_wf")

        if st.button("▶ Run Evaluation", type="primary",
                     key="ga_run_eval", disabled=running):
            st.session_state[_SK_EVAL] = True
            st.session_state[_SK_EVAL_LOG] = []
            _run_blocking(
                svc.run_eval(
                    config_name      = config_name,
                    skip_walkforward = skip_wf,
                ),
                log_key=_SK_EVAL_LOG,
            )
            st.session_state[_SK_EVAL] = False

        # Persistent log
        if st.session_state.get(_SK_EVAL_LOG):
            st.code("\n".join(st.session_state[_SK_EVAL_LOG][-300:]), language="text")
            if st.button("Clear log", key="ga_clear_eval_log"):
                st.session_state[_SK_EVAL_LOG] = []
                st.rerun()

    # ── Results ───────────────────────────────────────────────────────────────
    with tab_results:
        r_tab_eval, r_tab_wf, r_tab_weights = st.tabs(
            ["Eval Scorecard", "Walk-Forward", "Weights"]
        )

        with r_tab_eval:
            df = svc.get_eval_results()
            if df is None or df.empty:
                st.info("No eval results yet. Run **Evaluate** first.")
            else:
                # Highlight alpha column
                _show_eval_scorecard(df)

        with r_tab_wf:
            wf = svc.get_walk_forward()
            if wf is None or wf.empty:
                st.info("No walk-forward results yet.")
            else:
                st.dataframe(wf, use_container_width=True, hide_index=True)

        with r_tab_weights:
            if config_name:
                wdf = svc.get_weights(config_name)
                if wdf is not None:
                    st.dataframe(
                        wdf.sort_values("weight", ascending=False),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.info(f"No weights file found for `{config_name}`.")
            else:
                # Show all weights files in out_dir
                out = Path(svc.out_dir)
                wfiles = sorted(out.glob("weights_train_*.csv")) if out.exists() else []
                if not wfiles:
                    st.info("No weights files found in output directory.")
                else:
                    chosen = st.selectbox("Select weights file", [f.name for f in wfiles])
                    if chosen:
                        import pandas as pd
                        wdf = pd.read_csv(out / chosen)
                        st.dataframe(
                            wdf.sort_values("weight", ascending=False),
                            use_container_width=True, hide_index=True,
                        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _show_eval_scorecard(df):
    """Render eval_results with colour-coded alpha column."""
    import pandas as pd

    def _colour_alpha(val):
        if pd.isna(val):
            return ""
        return "color: #7cfc7c" if val > 0 else "color: #fc7c7c"

    alpha_col = next((c for c in df.columns if "alpha" in c.lower()), None)
    if alpha_col:
        styled = df.style.applymap(_colour_alpha, subset=[alpha_col])
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


def _run_blocking(gen, log_key: str | None = None):
    """
    Stream generator lines into a live code block while running, then persist
    the full log to session_state[log_key] so it survives the next rerun.
    """
    log_box = st.empty()
    lines: list[str] = []
    batch:  list[str] = []

    for line in gen:
        batch.append(line)
        if len(batch) >= 5 or any(c in line for c in ("=", "✓", "❌", "🏁", "ℹ️", "⚠️")):
            lines.extend(batch)
            batch = []
            log_box.code("\n".join(lines[-300:]), language="text")

    # Final flush
    if batch:
        lines.extend(batch)
    if lines:
        log_box.code("\n".join(lines[-300:]), language="text")

    # Persist to session state so the log survives st.rerun()
    if log_key is not None:
        st.session_state[log_key] = lines

    # Clear the live box — the persistent block below will render it
    log_box.empty()