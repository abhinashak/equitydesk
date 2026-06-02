"""
ui/pages/page_config_app.py  –  Config › App Settings
"""

import streamlit as st
from bll.config_service import ConfigService


def render():
    st.header("⚙️ App Configuration")
    st.caption("Edit `config/app_config.py` — changes take effect on next pipeline run.")

    svc = ConfigService()
    tab_visual, tab_raw = st.tabs(["Visual Editor", "Raw Editor"])

    # ── Visual editor ─────────────────────────────────────────────────────────
    with tab_visual:
        entries = svc.get_app_config_entries()
        pending: dict = {}

        for entry in entries:
            if entry["type"] == "section":
                st.subheader(entry["label"])
            elif entry["type"] == "entry":
                key = entry["key"]
                val = entry["value"]
                if val.lower() in ("true", "false", "1", "0", "yes", "no"):
                    new_val = st.selectbox(
                        key, options=["true", "false", "1", "0"],
                        index=["true","false","1","0"].index(val.lower())
                              if val.lower() in ["true","false","1","0"] else 0,
                        key=f"cfg_{key}",
                    )
                else:
                    try:
                        new_val = str(st.number_input(key, value=float(val),
                                                      format="%.8f", key=f"cfg_{key}"))
                    except ValueError:
                        new_val = st.text_input(key, value=val, key=f"cfg_{key}")
                if str(new_val) != str(val):
                    pending[key] = new_val

        if pending:
            st.info(f"Unsaved changes: {list(pending.keys())}")
            if st.button("💾 Save changes", type="primary"):
                for k, v in pending.items():
                    svc.set_app_value(k, v)
                st.session_state["app_cfg"] = svc.get_app_config()
                st.success("Saved.")
                st.rerun()
        else:
            st.caption("No unsaved changes.")

    # ── Raw editor ─────────────────────────────────────────────────────────────
    with tab_raw:
        raw     = svc.read_raw_config()
        new_raw = st.text_area("config/app_config.py", value=raw,
                               height=500, key="raw_config_editor")
        if st.button("💾 Save raw", type="primary", key="save_raw"):
            svc.write_raw_config(new_raw)
            st.session_state["app_cfg"] = svc.get_app_config()
            st.success("Raw config saved.")
            st.rerun()
