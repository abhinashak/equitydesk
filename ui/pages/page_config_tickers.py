"""
ui/pages/page_config_tickers.py  –  Config › Tickers + Periods

Schema lives in dal/ticker_config.py (TickerCol, TICKER_COLS, SEGMENTS, ...).
Sector choices come from config/sectors_hierarchy.csv via SectorHierarchy:
    • pick a Sector        → Sector Group + Macro Sector auto-populate
    • pick a Sector Group  → Macro Sector auto-populates
    • any field can still be overwritten manually afterwards
market_cap_segment is a dropdown: smallcap250 / midcap150 / nifty50 / NULL.

Required addition in bll/config_service.py (one line, delegates to the DAL):
    def get_sector_hierarchy(self) -> SectorHierarchy:
        return self.dal.load_sector_hierarchy()
"""

import streamlit as st
import pandas as pd

from bll.config_service import ConfigService
from dal.ticker_config import (
    TickerCol as C,
    TICKER_COLS, NUM_COLS, ENRICH_COLS, REQUIRED_COLS, SEGMENTS,
    SectorHierarchy, normalise_tickers,
)

_DEL = "🗑 Delete"


# ── Generic helpers ───────────────────────────────────────────────────────────
def _exchange(sym) -> str:
    s = str(sym or "").upper()
    if s.startswith("^"):
        return "Index"
    if s.endswith(".NS"):
        return "NSE"
    if s.endswith(".BO"):
        return "BSE"
    return "Other"


def _opts(series: pd.Series) -> list[str]:
    return sorted({str(v) for v in series.dropna() if str(v).strip()})


def _with_existing(base: list[str], series: pd.Series) -> list[str]:
    """Hierarchy values + any legacy values already in the data (so nothing is lost)."""
    return sorted(set(base) | set(_opts(series)))


def _none_label(v) -> str:
    return "NULL" if v is None else str(v)


def _same(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    return a == b


def _multiselect_keep(label: str, options: list[str], key: str) -> list[str]:
    """Multiselect whose options never drop a currently selected value (cascading safe)."""
    cur = [v for v in st.session_state.get(key, []) if v]
    options = sorted(set(options) | set(cur))
    return st.multiselect(label, options, key=key, placeholder="Any")


# ── Hierarchy auto-fill ───────────────────────────────────────────────────────
def _autofill_hierarchy(orig: pd.DataFrame, upd: pd.DataFrame,
                        h: SectorHierarchy) -> tuple[pd.DataFrame, int]:
    """For rows where the user changed Sector (or Sector Group), fill the parent
    levels from the hierarchy — unless the user also edited those parents."""
    if h.empty:
        return upd, 0
    upd = upd.copy()
    n = 0
    for idx in upd.index:
        o, u = orig.loc[idx], upd.loc[idx]
        sec_chg = not _same(o[C.SECTOR], u[C.SECTOR])
        grp_chg = not _same(o[C.SECTOR_GROUP], u[C.SECTOR_GROUP])
        mac_chg = not _same(o[C.MACRO_SECTOR], u[C.MACRO_SECTOR])
        grp, mac = u[C.SECTOR_GROUP], u[C.MACRO_SECTOR]
        if sec_chg and pd.notna(u[C.SECTOR]):
            g, m = h.group_macro_for_sector(u[C.SECTOR], prefer_group=grp)
            if g and not grp_chg:
                grp = g
            if m and not mac_chg:
                mac = m
        elif grp_chg and pd.notna(grp) and not mac_chg:
            m = h.macro_for_group(grp, prefer=mac)
            if m:
                mac = m
        if (grp, mac) != (u[C.SECTOR_GROUP], u[C.MACRO_SECTOR]):
            upd.at[idx, C.SECTOR_GROUP] = grp
            upd.at[idx, C.MACRO_SECTOR] = mac
            n += 1
    return upd, n


# ── Filtering ─────────────────────────────────────────────────────────────────
def _range_filter(df: pd.DataFrame, col: str, k: str) -> pd.Series:
    ser = df[col]
    clean = ser.dropna()
    if clean.empty:
        st.caption(f"{col}: no values")
        return pd.Series(True, index=df.index)
    mn, mx = float(clean.min()), float(clean.max())
    if mn == mx:
        st.caption(f"{col}: all = {mn}")
        return pd.Series(True, index=df.index)
    lo, hi = st.slider(col, mn, mx, (mn, mx), key=f"{k}_{col}_rng")
    blanks = st.checkbox("incl. blanks", value=True, key=f"{k}_{col}_blank")
    if (lo, hi) == (mn, mx) and blanks:
        return pd.Series(True, index=df.index)
    m = ser.between(lo, hi)
    return (m | ser.isna()) if blanks else m


def _advanced_filter(df: pd.DataFrame, k: str) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    cols = st.multiselect("Filter on any column", list(df.columns), key=f"{k}_advcols",
                          placeholder="Choose one or more columns…")
    for col in cols:
        ser, wkey = df[col], f"{k}_adv_{col}"
        if pd.api.types.is_numeric_dtype(ser):
            clean = ser.dropna()
            if clean.empty or clean.min() == clean.max():
                st.caption(f"{col}: nothing to filter")
                continue
            mn, mx = float(clean.min()), float(clean.max())
            lo, hi = st.slider(col, mn, mx, (mn, mx), key=wkey)
            if (lo, hi) != (mn, mx):
                mask &= ser.between(lo, hi)
        else:
            as_text = ser.astype("string")
            uniq = as_text.dropna().unique().tolist()
            if len(uniq) <= 50:
                sel = st.multiselect(col, sorted(uniq), key=wkey)
                if sel:
                    mask &= as_text.isin(sel)
            else:
                txt = st.text_input(f"{col} contains", key=wkey)
                if txt:
                    mask &= as_text.str.contains(txt, case=False, na=False, regex=False)
    return mask


def _filter_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """Render the filter panel; return the filtered subset (original index preserved)."""
    ver = st.session_state.setdefault("tk_flt_ver", 0)
    k = f"tkf{ver}"
    work = df.copy()
    work["_exchange"] = work[C.YAHOO].map(_exchange)
    mask = pd.Series(True, index=work.index)

    with st.expander("🔎 Filters", expanded=True):
        r1 = st.columns([3, 2, 2, 1])
        q = r1[0].text_input("Search", key=f"{k}_q",
                             placeholder="name, ticker, symbol or sector — comma-separate for OR")
        exch = r1[1].multiselect("Exchange", _opts(work["_exchange"]), key=f"{k}_exch",
                                 placeholder="Any")
        comp = r1[2].selectbox(
            "Data completeness",
            ["All", "Fully enriched", "Missing any enrichment", "Missing market cap",
             "Missing holdings (FII/public)", "Missing sector info"],
            key=f"{k}_comp",
        )
        r1[3].write("")
        if r1[3].button("↺ Reset", key=f"{k}_reset", use_container_width=True):
            st.session_state["tk_flt_ver"] = ver + 1
            st.rerun()

        if q.strip():
            terms = [t.strip() for t in q.split(",") if t.strip()]
            hay = (work[[C.NAME, C.YAHOO, C.SECTOR, C.TICKER]]
                   .fillna("").astype(str).agg(" | ".join, axis=1))
            tm = pd.Series(False, index=work.index)
            for t in terms:
                tm |= hay.str.contains(t, case=False, na=False, regex=False)
            mask &= tm
        if exch:
            mask &= work["_exchange"].isin(exch)

        enriched = work[ENRICH_COLS].notna()
        if comp == "Fully enriched":
            mask &= enriched.all(axis=1)
        elif comp == "Missing any enrichment":
            mask &= ~enriched.all(axis=1)
        elif comp == "Missing market cap":
            mask &= work[C.MCAP_CR].isna()
        elif comp == "Missing holdings (FII/public)":
            mask &= work[C.FII_PCT].isna() | work[C.PUBLIC_PCT].isna()
        elif comp == "Missing sector info":
            mask &= work[C.MACRO_SECTOR].isna() | work[C.SECTOR_GROUP].isna()

        r2 = st.columns(4)
        with r2[0]:
            macro = _multiselect_keep("Macro sector", _opts(work.loc[mask, C.MACRO_SECTOR]),
                                      f"{k}_macro")
        if macro:
            mask &= work[C.MACRO_SECTOR].isin(macro)
        with r2[1]:
            grp = _multiselect_keep("Sector group", _opts(work.loc[mask, C.SECTOR_GROUP]),
                                    f"{k}_grp")
        if grp:
            mask &= work[C.SECTOR_GROUP].isin(grp)
        with r2[2]:
            sec = _multiselect_keep("Sector", _opts(work.loc[mask, C.SECTOR]), f"{k}_sec")
        if sec:
            mask &= work[C.SECTOR].isin(sec)
        with r2[3]:
            seg = _multiselect_keep("Market-cap segment",
                                    _with_existing(SEGMENTS, work[C.SEGMENT]), f"{k}_seg")
        if seg:
            mask &= work[C.SEGMENT].isin(seg)

        r3 = st.columns(3)
        base = work[mask]
        for widget_col, col in zip(r3, NUM_COLS):
            with widget_col:
                sub_mask = _range_filter(base, col, k)
            mask &= sub_mask.reindex(work.index, fill_value=True)

        st.markdown("**Advanced**")
        adv = _advanced_filter(work[mask].drop(columns=["_exchange"]), k)
        mask &= adv.reindex(work.index, fill_value=True)

    return df[mask]


# ── Add-ticker form (callbacks give live cascading) ───────────────────────────
_ADD_TEXT   = ["add_name", "add_ticker", "add_yahoo"]
_ADD_SELECT = ["add_sector", "add_group", "add_macro", "add_seg"]
_ADD_NUM    = ["add_mcap", "add_fii", "add_pub"]


def _on_add_sector(h: SectorHierarchy) -> None:
    s = st.session_state.get("add_sector")
    g, m = h.group_macro_for_sector(s, prefer_group=st.session_state.get("add_group"))
    if g:
        st.session_state["add_group"] = g
    if m:
        st.session_state["add_macro"] = m


def _on_add_group(h: SectorHierarchy) -> None:
    g = st.session_state.get("add_group")
    m = h.macro_for_group(g, prefer=st.session_state.get("add_macro"))
    if m:
        st.session_state["add_macro"] = m


def _add_ticker_cb(svc: ConfigService) -> None:
    ss = st.session_state
    t = (ss.get("add_ticker") or "").strip().upper()
    if not t:
        ss["tk_add_msg"] = ("error", "ticker is required.")
        return
    row = {
        C.NAME:         (ss.get("add_name") or "").strip() or t,
        C.YAHOO:        (ss.get("add_yahoo") or "").strip() or f"{t}.NS",
        C.SECTOR:       ss.get("add_sector"),
        C.TICKER:       t,
        C.SECTOR_GROUP: ss.get("add_group"),
        C.MACRO_SECTOR: ss.get("add_macro"),
        C.SEGMENT:      ss.get("add_seg"),
        C.MCAP_CR:      ss.get("add_mcap"),
        C.FII_PCT:      ss.get("add_fii"),
        C.PUBLIC_PCT:   ss.get("add_pub"),
    }
    try:
        svc.add_ticker(row)
    except ValueError as e:
        ss["tk_add_msg"] = ("error", str(e))
        return
    for key in _ADD_TEXT:
        ss[key] = ""
    for key in _ADD_SELECT + _ADD_NUM:
        ss[key] = None
    ss["tk_add_msg"] = ("success", f"Added {t}.")


# ── Page ──────────────────────────────────────────────────────────────────────
def render():
    st.header("🗂 Ticker Configuration")
    svc = ConfigService()
    h: SectorHierarchy = svc.get_sector_hierarchy()
    if h.empty:
        st.warning("config/sectors_hierarchy.csv not found or empty — sector dropdowns "
                   "only offer values already present in tickers.csv.")

    tab_list, tab_add, tab_periods = st.tabs(["Ticker List", "Add Ticker", "Periods"])

    # ── Ticker list ───────────────────────────────────────────────────────────
    with tab_list:
        full = normalise_tickers(svc.get_tickers())
        view = _filter_tickers(full)
        is_filtered = len(view) != len(full)

        m = st.columns(5)
        m[0].metric("Tickers", f"{len(full):,}")
        m[1].metric("Showing", f"{len(view):,}")
        m[2].metric("Total mcap (₹ Cr)", f"{view[C.MCAP_CR].sum():,.0f}")
        m[3].metric("Avg FII %", f"{view[C.FII_PCT].mean():.2f}"
        if view[C.FII_PCT].notna().any() else "–")
        m[4].metric("Incomplete rows",
                    f"{int((~view[ENRICH_COLS].notna().all(axis=1)).sum()):,}")

        editor_df = view.copy()
        editor_df[_DEL] = False

        save_n = st.session_state.setdefault("tk_save_n", 0)
        sig = abs(hash(tuple(view.index))) % 10**8
        st.caption(
            "Changing **Sector** auto-fills Sector Group & Macro Sector on save (changing "
            "Sector Group fills Macro Sector) — unless you edit those cells yourself."
            + (" · Editing a filtered view: changes merge into the full list; changing "
               "filters discards unsaved edits." if is_filtered else "")
        )

        edited = st.data_editor(
            editor_df,
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            key=f"ticker_editor_{save_n}_{sig}",
            column_order=[_DEL] + TICKER_COLS,
            column_config={
                _DEL: st.column_config.CheckboxColumn(_DEL, help="Tick to delete on save",
                                                      width="small"),
                C.TICKER: st.column_config.TextColumn(C.TICKER, required=True),
                C.SECTOR: st.column_config.SelectboxColumn(
                    C.SECTOR, options=_with_existing(h.sectors(), full[C.SECTOR])),
                C.SECTOR_GROUP: st.column_config.SelectboxColumn(
                    C.SECTOR_GROUP, options=_with_existing(h.groups(), full[C.SECTOR_GROUP])),
                C.MACRO_SECTOR: st.column_config.SelectboxColumn(
                    C.MACRO_SECTOR, options=_with_existing(h.macros(), full[C.MACRO_SECTOR])),
                C.SEGMENT: st.column_config.SelectboxColumn(
                    C.SEGMENT, options=_with_existing(SEGMENTS, full[C.SEGMENT]),
                    help="Clear the cell for NULL"),
                C.MCAP_CR: st.column_config.NumberColumn(
                    C.MCAP_CR, min_value=0, step=0.01, format="%.2f"),
                C.FII_PCT: st.column_config.NumberColumn(
                    C.FII_PCT, min_value=0, max_value=100, step=0.01, format="%.2f"),
                C.PUBLIC_PCT: st.column_config.NumberColumn(
                    C.PUBLIC_PCT, min_value=0, max_value=100, step=0.01, format="%.2f"),
            },
        )

        n_del = int(edited[_DEL].sum())
        c1, c2, c3, c4 = st.columns([1.3, 1.3, 1.3, 3])
        if c1.button("💾 Save changes" + (f" (delete {n_del})" if n_del else ""),
                     type="primary", use_container_width=True):
            upd = edited.drop(columns=[_DEL])
            upd, n_fill = _autofill_hierarchy(view, upd, h)
            merged = full.copy()
            merged.loc[upd.index, TICKER_COLS] = upd[TICKER_COLS]
            merged = merged.drop(index=edited.index[edited[_DEL]]).reset_index(drop=True)
            svc.save_tickers(normalise_tickers(merged))
            st.session_state["tk_save_n"] = save_n + 1
            parts = ["Saved."]
            if n_del:
                parts.append(f"{n_del} deleted.")
            if n_fill:
                parts.append(f"Hierarchy auto-filled on {n_fill} row(s).")
            st.session_state["tk_flash"] = " ".join(parts)
            st.rerun()
        c2.download_button("⬇ Filtered CSV", data=view.to_csv(index=False),
                           file_name="tickers_filtered.csv", mime="text/csv",
                           use_container_width=True, disabled=view.empty)
        c3.download_button("⬇ Full CSV", data=full.to_csv(index=False),
                           file_name="tickers.csv", mime="text/csv",
                           use_container_width=True)
        if not view.empty:
            with c4.popover("📋 Copy filtered tickers"):
                st.code("\n".join(view[C.TICKER].dropna().astype(str)), language=None)

        if msg := st.session_state.pop("tk_flash", None):
            st.success(msg)

        st.divider()
        uploaded = st.file_uploader("Replace tickers.csv", type=["csv"])
        if uploaded:
            try:
                new_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                new_df = None
            if new_df is not None:
                missing_req = [c for c in REQUIRED_COLS if c not in new_df.columns]
                if missing_req:
                    st.error(f"Missing required column(s): {', '.join(missing_req)}")
                else:
                    new_df = normalise_tickers(new_df)
                    st.caption(f"Parsed {len(new_df):,} rows. Preview:")
                    st.dataframe(new_df.head(10), use_container_width=True)
                    if st.button("⚠ Replace ALL tickers with this file", key="tk_replace"):
                        svc.save_tickers(new_df)
                        st.session_state["tk_save_n"] = save_n + 1
                        st.session_state["tk_flash"] = "Uploaded and saved."
                        st.rerun()

    # ── Add ticker ────────────────────────────────────────────────────────────
    with tab_add:
        st.subheader("Add a new ticker")
        existing = normalise_tickers(svc.get_tickers())
        sec_opts = [None] + _with_existing(h.sectors(), existing[C.SECTOR])
        grp_opts = [None] + _with_existing(h.groups(), existing[C.SECTOR_GROUP])
        mac_opts = [None] + _with_existing(h.macros(), existing[C.MACRO_SECTOR])
        seg_opts = [None] + _with_existing(SEGMENTS, existing[C.SEGMENT])

        c1, c2 = st.columns(2)
        c1.text_input("Name (company)", key="add_name", placeholder="Zydus Wellness")
        c2.text_input("ticker (Screener symbol) *", key="add_ticker", placeholder="ZYDUSWELL")
        c1.text_input("Yahoo Symbol", key="add_yahoo",
                      placeholder="default: <ticker>.NS")
        c2.selectbox("market_cap_segment", seg_opts, key="add_seg", format_func=_none_label)

        st.caption("Pick a Sector to auto-fill its Sector Group and Macro Sector; "
                   "any of the three can be overridden afterwards.")
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Sector", sec_opts, key="add_sector", format_func=_none_label,
                     on_change=_on_add_sector, args=(h,))
        c2.selectbox("sector_group", grp_opts, key="add_group", format_func=_none_label,
                     on_change=_on_add_group, args=(h,))
        c3.selectbox("macro_sector", mac_opts, key="add_macro", format_func=_none_label)

        c1, c2, c3 = st.columns(3)
        c1.number_input("market_cap_cr", min_value=0.0, value=None, step=1.0, key="add_mcap")
        c2.number_input("fii_hold_pct", min_value=0.0, max_value=100.0, value=None,
                        step=0.01, key="add_fii")
        c3.number_input("public_hold_pct", min_value=0.0, max_value=100.0, value=None,
                        step=0.01, key="add_pub")

        st.button("➕ Add Ticker", type="primary", on_click=_add_ticker_cb, args=(svc,))
        if msg := st.session_state.pop("tk_add_msg", None):
            (st.success if msg[0] == "success" else st.error)(msg[1])

    # ── Periods ───────────────────────────────────────────────────────────────
    with tab_periods:
        st.subheader("Training / Testing Periods")
        periods = svc.get_periods()
        for i, p in enumerate(periods):
            c1, c2, c3 = st.columns([3, 1, 1])
            badge = "🟢" if p.get("enabled") else "🔴"
            c1.markdown(
                f"**{badge} {p['name']}**  "
                f"`Train {p['train']['start']} → {p['train']['end']}` "
                f"| `Test {p['test']['start']} → {p['test']['end']}`"
            )
            if c2.button("Toggle", key=f"tog_{i}"):
                svc.toggle_period(p["name"]); st.rerun()
            if c3.button("🗑 Delete", key=f"del_{i}"):
                svc.delete_period(p["name"]); st.rerun()

        st.divider()
        st.subheader("Add Period")
        with st.form("add_period_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            p_name    = c1.text_input("Period name", placeholder="EVT-Iran-War")
            p_enabled = c2.checkbox("Enabled", value=True)
            tr_start  = c1.date_input("Train start")
            tr_end    = c2.date_input("Train end")
            tr_type   = c1.text_input("Train type", value="fy")
            te_start  = c1.date_input("Test start")
            te_end    = c2.date_input("Test end")
            te_type   = c1.text_input("Test type", value="crash")
            if st.form_submit_button("➕ Add Period", type="primary") and p_name:
                svc.add_period({
                    "name": p_name, "enabled": int(p_enabled),
                    "train": {"start": str(tr_start), "end": str(tr_end), "type": tr_type},
                    "test":  {"start": str(te_start), "end": str(te_end), "type": te_type},
                })
                st.success(f"Added period '{p_name}'."); st.rerun()