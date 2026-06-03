import streamlit as st
import hashlib
import requests
import json
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from ui.pages.trade.shared_utils import (
    load_exclusions, load_weights, fetch_holdings, save_exclusions
)


st.set_page_config(page_title="Kite Portfolio Setup", layout="wide", page_icon="⚙️")
SECRET_PATH   = Path(".secret/kite.secret")
ACCOUNTS_PATH = Path(".secret/accounts.json")

def load_accounts():
    if ACCOUNTS_PATH.exists():
        try:
            return json.loads(ACCOUNTS_PATH.read_text())
        except Exception:
            pass
    return [{"name": "Account 1", "token": ""}]

def save_accounts(accounts):
    ACCOUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACCOUNTS_PATH.write_text(json.dumps(accounts, indent=2))

# Helper functions to read/write secrets dynamically
def read_secrets():
    secrets = {}
    if SECRET_PATH.exists():
        try:
            content = SECRET_PATH.read_text().strip()
            kvL = content.split(",") if "," in content else content.split()
            for item in kvL:
                if '=' in item:
                    key, value = item.split('=', 1)
                    secrets[key.strip()] = value.strip() if value.strip() else "[EMPTY]"
        except Exception:
            pass
    return secrets

def save_secrets(api_key, api_secret):
    SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
    SECRET_PATH.write_text(f"{api_key.strip()},{api_secret.strip()}")

# Initialize shared Session State globally
def init_session():
    defaults = {
        "accounts": load_accounts(),
        "all_holdings": {},
        "excluded_symbols": load_exclusions(),
        "action_overrides": {},
        "staged_orders": [],
        "live_orders": [],
        "runner_active": False,
        "last_poll_time": None,
        "weights_text": load_weights(),
        "buy_strategy_used": "Geometric Progression",
        "sell_strategy_used": "Geometric Progression (GP)",
        "fresh_capital": 0.0
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def render():
    init_session()
    st.title("⚙️ Setup Kite Accounts")

    st.markdown("Configure your Kite credentials and prepare data for the portfolio tools.")

    # ═ Token Helper Utilities Control Box ═══════════════════
    st.subheader("🛠 Token Generation Utility")

    # Wrapped tightly within an interactive container block
    with st.container(border=True):

        # Load any existing secrets dynamically
        secrets = read_secrets()

        with st.expander("🚀 Quick Start & Authorization Workflow", expanded=True):
            st.markdown("Follow these steps to authenticate your Zerodha Kite connect application:")
            st.markdown(
                "**1. Start the Process** \n"
                "Begin by navigating to the main dashboard:  \n"
                "[Start - Abhinashak Streamlit App](https://abhinashak.streamlit.app/)"
            )
            st.divider() # Visual separator line
            st.markdown(
                f"""**2. Authorize with Zerodha** \n"
                "Use your specific API key to log in and get your access request token:  \n"
                "[Kite Zerodha Connect Login]({secrets["connect_url"]}{secrets["api_key"]})"""
            )

        st.markdown("### 🔑 Quick Token Generator")
        st.caption("Paste your redirected login URL below to automatically sync credentials and extract your session tokens.")

        # Input: Redirection Response URL
        response_url = st.text_input(
            "Response URL",
            placeholder="https://abhinashak.streamlit.app/?request_token=rrrrr...",
            key="utility_url"
        )

        # Auto extraction of request_token
        request_token = ""
        if response_url:
            try:
                parsed_url = urlparse(response_url)
                queries = parse_qs(parsed_url.query)
                if "request_token" in queries:
                    request_token = queries["request_token"][0]
                    st.info(f"Detected Request Token: `{request_token}`")
                else:
                    st.error("No `request_token` found in the URL parameters.")
            except Exception as e:
                st.error(f"Error parsing components: {e}")

        # Inline configuration showing secret file credentials
        rc1, rc2 = st.columns(2)
        api_key = rc1.text_input("API Key", value=secrets["api_key"], placeholder="Key")
        api_secret = rc2.text_input("API Secret", value=secrets["api_secret"], type="password", placeholder="Secret")

        # Update .secret/kite.secret on changes dynamically
        secrets_changed = (api_key != secrets.get("api_key", "") or api_secret != secrets.get("api_secret", ""))

        if secrets_changed and (api_key or api_secret):
            if st.button("💾 Save Secrets", help="Detected changes — click to save"):
                save_secrets(api_key, api_secret)
                st.toast("Secrets file synchronized!", icon="💾")

        # Execution Action Button
        if st.button("🚀 Process Access Token", type="secondary", use_container_width=True):
            if not request_token or not api_key or not api_secret:
                st.warning("Ensure the URL contains a valid token and secrets are filled.")
            else:
                with st.spinner("Generating checksum & processing authorization..."):
                    try:
                        # Generate payload SHA-256 checksum
                        payload = api_key + request_token + api_secret
                        checksum = hashlib.sha256(payload.encode("utf-8")).hexdigest()

                        # POST connection payload to Zerodha endpoints
                        url = "https://api.kite.trade/session/token"
                        headers = {"X-Kite-Version": "3"}
                        data = {
                            "api_key": api_key,
                            "request_token": request_token,
                            "checksum": checksum
                        }

                        res = requests.post(url, headers=headers, data=data)
                        res_json = res.json()

                        if res.status_code == 200 and res_json.get("status") == "success":
                            access_token = res_json["data"]["access_token"]
                            final_token = f"{api_key}:{access_token}"

                            st.success("Access Token generated successfully!")
                            st.text_area(
                                "Token Result (api_key:access_token)",
                                value=final_token,
                                height=68,
                                help="Copy this token result straight into your account configuration fields on the left pane."
                            )
                        else:
                            st.error(f"Kite Engine Error ({res.status_code})")
                            st.json(res_json)

                    except Exception as ex:
                        st.error(f"Processing error: {ex}")

    # ── Accounts Management ──
    with st.expander("🔑 Accounts", expanded=True):
        accounts = st.session_state.accounts
        to_remove = None
        for i, acc in enumerate(accounts):
            c_name, c_token, c_clear, c_del = st.columns([2, 3, 1, 1])
            acc["name"]  = c_name.text_input("Name",  value=acc["name"],  key=f"acc_name_{i}",  placeholder=f"Account {i+1}")
            acc["token"] = c_token.text_input("Token", value=acc["token"], key=f"acc_token_{i}", type="password", placeholder="api_key:access_token")

            # Clear Token — wipes expired token but keeps the account row
            if c_clear.button("🧹", key=f"acc_clr_{i}", help="Clear expired token"):
                acc["token"] = ""
                save_accounts(accounts)
                st.toast(f"Token cleared for {acc['name']}", icon="🧹")
                st.rerun()

            if len(accounts) > 1 and c_del.button("🗑", key=f"acc_del_{i}", help="Remove account"):
                to_remove = i

            if i < len(accounts) - 1:
                st.divider()

        if to_remove is not None:
            accounts.pop(to_remove)
            save_accounts(accounts)
            st.rerun()

        st.divider()

        b1, b2 = st.columns(2)

        if b1.button("➕ Add Account", use_container_width=True):
            accounts.append({"name": f"Account {len(accounts)+1}", "token": ""})
            st.rerun()

        if b2.button("💾 Save Accounts", use_container_width=True):
            save_accounts(accounts)
            st.toast("Accounts saved to .secret/accounts.json!", icon="💾")


    # ── Exclusions ──
    st.divider()
    st.subheader("🚫 Ignore List")
    excl: set = st.session_state.excluded_symbols
    all_known = sorted({h["tradingsymbol"] for hlds in st.session_state.all_holdings.values() for h in hlds})

    if excl:
        chip_cols = st.columns(2)
        for idx, sym in enumerate(sorted(excl)):
            if chip_cols[idx % 2].button(f"✕ {sym}", key=f"rm_excl_{sym}"):
                excl.discard(sym)
                save_exclusions(excl)
                st.rerun()
    else:
        st.caption("_No symbols ignored._")

    with st.form("add_excl_form", clear_on_submit=True):
        opts = [""] + [s for s in all_known if s not in excl]
        pick = st.selectbox("Pick from holdings", opts) if all_known else None
        typed = st.text_input("Or type symbol", placeholder="SGBMAR29")
        if st.form_submit_button("➕ Add to Ignore"):
            to_add = typed.strip().upper() or (pick.strip().upper() if pick else "")
            if to_add:
                excl.add(to_add)
                save_exclusions(excl)
                st.rerun()

    # ── Fetch Holdings ──
    st.divider()
    if st.button("🔄 Fetch Holdings", type="primary", use_container_width=True):
        accounts = st.session_state.accounts
        fetched, errs = {}, []
        for acc in accounts:
            if not acc["token"]: continue
            try:
                with st.spinner(f"Fetching {acc['name']}…"):
                    fetched[acc["name"]] = fetch_holdings(acc["token"])
            except Exception as e:
                errs.append(f"**{acc['name']}:** {e}")
        st.session_state.all_holdings = fetched
        for e in errs: st.error(e)
        if fetched and not errs:
            st.success(f"✓ {sum(len(v) for v in fetched.values())} holdings loaded. Redirecting to Portfolio…")
            st.session_state["active_page"] = "trade_portfolio"
            st.rerun()