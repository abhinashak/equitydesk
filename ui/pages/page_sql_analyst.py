"""
ui/pages/page_sql_analyst.py  –  Screens › SQL Analyst
"""

import os

import streamlit as st
from bll import agent as ag

_RUNNING_KEY  = "analyst_running"
_MESSAGES_KEY = "analyst_messages"

def render_accumulation_sidebar():
    st.sidebar.header("🎯 Accumulation Calculator")
    st.sidebar.caption("Plan your tranches for deep-value stocks.")
    
    # Interactive inputs
    max_allocation = st.sidebar.number_input("Max Target Allocation (₹)", min_value=10000, value=100000, step=10000)
    cmp = st.sidebar.number_input("Current Stock Price (₹)", min_value=1.0, value=2941.0, step=10.0)
    
    st.sidebar.divider()
    
    with st.sidebar.expander("📍 Tranche 1: The Pilot (20%)", expanded=True):
        st.markdown(f"**Deploy:** ₹{max_allocation * 0.20:,.0f}")
        st.markdown(f"**Action:** Buy immediately at **₹{cmp:,.2f}**.")
        st.markdown("*Gets you in the game to track psychology.*")

    with st.sidebar.expander("📉 Tranche 2: Value Averaging (30%)"):
        st.markdown(f"**Deploy:** ₹{max_allocation * 0.30:,.0f}")
        st.markdown("**Action:** Manual SIP over 3-6 months.")
        st.markdown("*Increase SIP amount if price falls below initial entry.*")

    with st.sidebar.expander("📈 Tranche 3: Trend Reversal (30%)"):
        st.markdown(f"**Deploy:** ₹{max_allocation * 0.30:,.0f}")
        st.markdown("**Action:** Deploy when price holds > 200-day SMA.")
        st.markdown("*Wait for institutional money to step in.*")

    with st.sidebar.expander("📊 Tranche 4: Confirmation (20%)"):
        st.markdown(f"**Deploy:** ₹{max_allocation * 0.20:,.0f}")
        st.markdown("**Action:** Deploy after 2 strong earnings quarters.")
        st.markdown("*Confirms the DCF growth projections are real.*")



def render():
    st.header("🤖 Financial Analyst Agent")

    render_accumulation_sidebar()

    app_cfg = st.session_state.get("app_cfg", {})
    db_path = app_cfg.get("DB_PATH", str(ag.DB))
    try:
        with st.spinner("Loading database and agent (first run can take a few minutes)..."):
            agent = _get_agent()
    except Exception as e:
        st.error(f"Could not start the SQL Analyst: {e}")
        st.stop()

    running  = st.session_state.get(_RUNNING_KEY, False)
    messages = st.session_state.setdefault(_MESSAGES_KEY, [])

    # ── Control row (full width — chat output MUST be outside any column) ─────
    st.caption(f"Database: `{db_path}`")

    if st.button("🗑 Clear Chat", key="clear_analyst",
                 disabled=running or not messages):
        st.session_state[_MESSAGES_KEY] = []
        st.rerun()

    st.divider()

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Chat input ────────────────────────────────────────────────────────────
    prompt = st.chat_input("Ask a question about your DuckDB data...",
                           key="analyst_input", disabled=running)
    if prompt:
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state[_RUNNING_KEY] = True
        with st.chat_message("assistant"):
            try:
                output = _run_blocking(_stream_agent(agent, prompt))
            except Exception as e:
                output = f"⚠️ {type(e).__name__}: {str(e)[:300]}"
                st.error(output)
        messages.append({"role": "assistant", "content": output})
        st.session_state[_RUNNING_KEY] = False
        st.rerun()


# ── Agent init (cached once per session server) ───────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_agent():
    try:
        return _build_agent()
    except SystemExit as e:   # agent.py uses SystemExit; Streamlit hides it silently
        raise RuntimeError(str(e) or "agent setup exited") from None


def _build_agent():
    if not ag.DB.exists():
        # the SQL scripts use relative parquet paths that resolve from ag.BASE;
        # switch there only for the build, then restore the app's working dir
        cwd = os.getcwd()
        try:
            os.chdir(ag.BASE)
            ag.build_db()
        finally:
            os.chdir(cwd)
    profile = ag.load_profile()
    ag.STATE["profile"] = profile
    ag.STATE["schema"]  = profile["schema"]
    return ag.create_agent(ag.build_instructions(profile))


# ── Agent stream ──────────────────────────────────────────────────────────────

def _stream_agent(agent, prompt: str):
    """Yield text chunks from the Agno agent as they arrive."""
    for chunk in agent.run(prompt, stream=True):
        if getattr(chunk, "content", None):
            yield str(chunk.content)


# ── Blocking stream ───────────────────────────────────────────────────────────

def _run_blocking(gen) -> str:
    """
    Consume a chunk-yielding generator synchronously, rendering the growing
    answer into a single placeholder.  Returns the full text so the caller
    can store it in chat history.
    """
    box = st.empty()
    text = ""
    with st.spinner("Analyzing database..."):
        for chunk in gen:
            text += chunk
            box.markdown(text)
    return text
