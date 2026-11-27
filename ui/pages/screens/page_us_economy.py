"""
ui/pages/page_us_economy.py  –  US Economy Pipeline screen
───────────────────────────────────────────────────────────
Extracted from us_economy_and_news.py (Dashboard 1).
Entry point: render()
"""

import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import pandas_datareader.data as web
import datetime
import calendar
import json
import plotly.graph_objects as go


# ── GIFT Nifty live banner (shared helper, imported by both screens) ──────────
@st.cache_data(ttl=60)
def fetch_gift_nifty():
    result = dict(price=None, open=None, high=None, low=None,
                  change=None, pct=None, source="", ts="")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept": "application/json, text/html, */*",
        "Referer": "https://zerodha.com/market/giftnifty/",
    }

    # Primary: NSE IX public API
    try:
        nse_url = "https://www.nseindia.com/api/quote-derivative?symbol=GIFTNIFTY"
        r = requests.get(nse_url, headers=headers, timeout=6)
        if r.status_code == 200:
            data = r.json()
            info = data.get("info", {})
            ltp  = info.get("lastPrice") or data.get("underlyingValue")
            if ltp:
                result.update(
                    price  = float(ltp),
                    open   = float(info.get("open",  ltp)),
                    high   = float(info.get("dayHigh", ltp)),
                    low    = float(info.get("dayLow",  ltp)),
                    change = float(info.get("change",  0)),
                    pct    = float(info.get("pChange", 0)),
                    source = "NSE IX",
                )
                result["ts"] = datetime.datetime.now().strftime("%H:%M:%S")
                return result
    except Exception:
        pass

    # Fallback 1: Zerodha HTML
    try:
        r = requests.get("https://zerodha.com/market/giftnifty/",
                         headers=headers, timeout=8)
        if r.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(r.text, "html.parser")
            nxt = soup.find("script", {"id": "__NEXT_DATA__"})
            if nxt:
                payload = json.loads(nxt.string)
                gn = (payload.get("props", {}).get("pageProps", {})
                      .get("giftNifty", {}))
                ltp = gn.get("lastPrice") or gn.get("last")
                if ltp:
                    result.update(
                        price  = float(ltp),
                        open   = float(gn.get("open",  ltp)),
                        high   = float(gn.get("high",  ltp)),
                        low    = float(gn.get("low",   ltp)),
                        change = float(gn.get("change", 0)),
                        pct    = float(gn.get("pct",    0)),
                        source = "Zerodha",
                    )
                    result["ts"] = datetime.datetime.now().strftime("%H:%M:%S")
                    return result
    except Exception:
        pass

    # Fallback 2: yfinance (Nifty 50 spot as proxy)
    try:
        ticker  = yf.Ticker("^NSEI")
        info    = ticker.fast_info
        ltp     = float(info.last_price)
        prev    = float(info.previous_close)
        chg     = round(ltp - prev, 2)
        pct     = round((chg / prev) * 100, 2) if prev else 0.0
        hist    = ticker.history(period="1d", interval="1m")
        open_   = float(hist["Open"].iloc[0])  if len(hist) else ltp
        high_   = float(hist["High"].max())     if len(hist) else ltp
        low_    = float(hist["Low"].min())      if len(hist) else ltp
        result.update(
            price  = ltp, open = open_, high = high_, low = low_,
            change = chg, pct  = pct,
            source = "yFinance (Nifty 50 proxy — GIFT Nifty unavailable)",
        )
        result["ts"] = datetime.datetime.now().strftime("%H:%M:%S")
    except Exception:
        result["source"] = "Unavailable"

    return result


def render_gift_nifty_banner():
    gn    = fetch_gift_nifty()
    price = gn["price"]
    chg   = gn["change"]
    pct   = gn["pct"]
    open_ = gn["open"]
    high_ = gn["high"]
    low_  = gn["low"]
    src   = gn["source"]
    ts    = gn["ts"]

    if price is None:
        st.warning("⚠️ GIFT Nifty data unavailable right now. Retrying in 60 s.")
        return

    is_up     = chg >= 0
    arrow     = "▲" if is_up else "▼"
    clr_main  = "#16a34a" if is_up else "#dc2626"
    clr_bg    = "#f0fdf4" if is_up else "#fef2f2"
    clr_bdr   = "#22c55e" if is_up else "#f87171"
    chg_sign  = "+" if is_up else ""

    open_str  = f"{open_:,.2f}"  if open_  else "—"
    high_str  = f"{high_:,.2f}"  if high_  else "—"
    low_str   = f"{low_:,.2f}"   if low_   else "—"

    st.markdown(f"""
    <div style="
        background:{clr_bg};border:1px solid {clr_bdr};border-left:4px solid {clr_main};
        border-radius:8px;padding:10px 18px;margin-bottom:16px;
        display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
      <div style="display:flex;align-items:baseline;gap:12px;">
        <span style="font-size:11px;font-weight:700;color:#374151;text-transform:uppercase;letter-spacing:0.07em;">
          🇮🇳 GIFT Nifty
        </span>
        <span style="font-size:26px;font-weight:800;color:#111827;font-family:'Inter',sans-serif;letter-spacing:-0.5px;">
          {price:,.2f}
        </span>
        <span style="font-size:15px;font-weight:700;color:{clr_main};">
          {arrow} {chg_sign}{chg:,.2f} &nbsp;({chg_sign}{pct:.2f}%)
        </span>
      </div>
      <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;">
        <span style="font-size:11px;color:#6b7280;">
          <b style="color:#374151;">O</b> {open_str} &nbsp;
          <b style="color:#16a34a;">H</b> {high_str} &nbsp;
          <b style="color:#dc2626;">L</b> {low_str}
        </span>
        <span style="font-size:10px;color:#9ca3af;">📡 {src} &nbsp;·&nbsp; ⏱ {ts}</span>
        <a href="https://zerodha.com/market/giftnifty/" target="_blank"
           style="font-size:10px;color:#6b7280;text-decoration:none;
                  border:1px solid #d1d5db;border-radius:4px;padding:2px 8px;">
          Zerodha ↗
        </a>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main render ───────────────────────────────────────────────────────────────
def render():
    render_gift_nifty_banner()

    st.title("THE US ECONOMY")
    st.subheader("The Monetary Pipeline Leading to Consumer Price Increases")
    st.markdown("---")

    # ── BLS API key loader ────────────────────────────────────────────────────
    from pathlib import Path
    def load_bls_api_key(path=".secret/bls.secret"):
        try:
            with open(path, "r") as f:
                abs_path = Path(path).resolve()
                print(f"File loaded from : {abs_path}")
                key = f.read().strip()
            print (key)
            return key if key else None
        except FileNotFoundError:
            abs_path = Path(path).resolve()
            print(f"File not found. Absolute path searched: {abs_path}")
            return None

    BLS_API_KEY = load_bls_api_key()

    # ── 401(k) account config loader ─────────────────────────────────────────
    def load_account_config(path="../../outputs/account.json"):
        defaults = {"baseline_balance": 562450.00, "total_contributions": 14500.00}
        try:
            with open(path, "r") as f:
                config = json.load(f)
            return (
                float(config.get("baseline_balance", defaults["baseline_balance"])),
                float(config.get("total_contributions", defaults["total_contributions"])),
                True,
            )
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
            return defaults["baseline_balance"], defaults["total_contributions"], False

    @st.cache_data(ttl=1800)
    def fetch_sp500_ytd_return():
        try:
            current_year = datetime.date.today().year
            ticker = yf.Ticker("^GSPC")
            hist = ticker.history(start=f"{current_year}-01-01")
            if len(hist) >= 2:
                jan_close = hist["Close"].iloc[0]
                current_close = hist["Close"].iloc[-1]
                return round(((current_close - jan_close) / jan_close) * 100, 2)
        except Exception:
            pass
        return 14.20

    def get_last_n_months(n=6, end_date=None):
        if end_date is None:
            end_date = datetime.date.today()
        year, month = end_date.year, end_date.month
        months = []
        for _ in range(n):
            months.append((year, month))
            month -= 1
            if month == 0:
                month = 12
                year -= 1
        months.reverse()
        return months

    def month_label(year, month):
        return f"{calendar.month_abbr[month]} {year}"

    @st.cache_data(ttl=3600)
    def fetch_bls_data(api_key, n_months=6, buffer_months=3):
        url = 'https://api.bls.gov/publicAPI/v1/timeseries/data/'
        series_mapping = {
            "WPSFD4131": "Producer Price Index",
            "EIUIR":     "Export Price Index",
            "EIUIR000":  "Import Price Index",
        }
        target_months = get_last_n_months(n_months + buffer_months)
        years_needed  = sorted({y for y, _ in target_months})
        month_labels  = [month_label(y, m) for y, m in target_months]
        payload = {
            "seriesid":  list(series_mapping.keys()),
            "startyear": str(years_needed[0]),
            "endyear":   str(years_needed[-1]),
        }
        if api_key:
            payload["registrationkey"] = api_key
        try:
            resp      = requests.post(url, json=payload,
                                      headers={'Content-type': 'application/json'})
            json_data = resp.json()
            if json_data.get('status') != 'REQUEST_SUCCEEDED':
                return None, month_labels
            target_set     = set(target_months)
            parsed_records = []
            for series in json_data['Results']['series']:
                index_name = series_mapping[series['seriesID']]
                for item in series['data']:
                    period = item['period']
                    year   = int(item['year'])
                    if not period.startswith('M'):
                        continue
                    month_num = int(period[1:])
                    if (year, month_num) not in target_set:
                        continue
                    parsed_records.append({
                        "Index Type": index_name,
                        "Month":      month_label(year, month_num),
                        "Value":      float(item['value']),
                    })
            if not parsed_records:
                return None, month_labels
            raw_df   = pd.DataFrame(parsed_records)
            pivot_df = raw_df.pivot(index="Index Type", columns="Month",
                                    values="Value").reset_index()
            for ml in month_labels:
                if ml not in pivot_df.columns:
                    pivot_df[ml] = pd.NA
            return pivot_df[["Index Type"] + month_labels], month_labels
        except Exception:
            return None, month_labels

    def get_pct_change(series, periods_back):
        if len(series) > periods_back:
            current = series.iloc[-1]
            past    = series.iloc[-(periods_back + 1)]
            if past != 0:
                return ((current - past) / past) * 100
        return 0.0

    @st.cache_data(ttl=3600)
    def fetch_pipeline_data():
        end_date   = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=180)
        metrics    = {}
        try:
            dxy_df = yf.Ticker("DX-Y.NYB").history(period="6mo")['Close']
            metrics['DXY'] = {'latest': dxy_df.iloc[-1],
                              '3D': get_pct_change(dxy_df, 3), '7D': get_pct_change(dxy_df, 7),
                              '1M': get_pct_change(dxy_df, 21), '3M': get_pct_change(dxy_df, 63)}
        except Exception:
            metrics['DXY'] = {'latest': 104.2, '3D': 0.1, '7D': -0.2, '1M': 0.5, '3M': -1.1}
        try:
            dbc_df = yf.Ticker("DBC").history(period="6mo")['Close']
            metrics['CRB'] = {'latest': dbc_df.iloc[-1],
                              '3D': get_pct_change(dbc_df, 3), '7D': get_pct_change(dbc_df, 7),
                              '1M': get_pct_change(dbc_df, 21), '3M': get_pct_change(dbc_df, 63)}
        except Exception:
            metrics['CRB'] = {'latest': 22.4, '3D': 0.4, '7D': 1.1, '1M': 3.2, '3M': 5.4}
        try:
            cpi_df = web.DataReader("CPIAUCSL", "fred", start_date, end_date)['CPIAUCSL']
            metrics['CPI'] = {'latest': cpi_df.iloc[-1],
                              'MoM': get_pct_change(cpi_df, 1), '3MoM': get_pct_change(cpi_df, 3)}
        except Exception:
            metrics['CPI'] = {'latest': 312.1, 'MoM': 0.2, '3MoM': 0.8}
        try:
            ppi_df = web.DataReader("PPIACO", "fred", start_date, end_date)['PPIACO']
            metrics['PPI'] = {'latest': ppi_df.iloc[-1],
                              'MoM': get_pct_change(ppi_df, 1), '3MoM': get_pct_change(ppi_df, 3)}
        except Exception:
            metrics['PPI'] = {'latest': 240.5, 'MoM': 0.3, '3MoM': 1.1}
        try:
            m2_df = web.DataReader("M2SL", "fred", start_date, end_date)['M2SL']
            metrics['M2'] = {'latest': m2_df.iloc[-1] / 1000,
                             'MoM': get_pct_change(m2_df, 1), '3MoM': get_pct_change(m2_df, 3)}
        except Exception:
            metrics['M2'] = {'latest': 20.9, 'MoM': 0.1, '3MoM': -0.4}
        try:
            debt_df = web.DataReader("GFDEBTN", "fred", start_date, end_date)['GFDEBTN']
            metrics['Debt'] = {'latest': debt_df.iloc[-1] / 1000000,
                               'QoQ': get_pct_change(debt_df, 1)}
        except Exception:
            metrics['Debt'] = {'latest': 34.6, 'QoQ': 1.8}
        return metrics

    data = fetch_pipeline_data()

    # ── Trend table helper ────────────────────────────────────────────────────
    def display_trend_table(trend_dict):
        st.markdown(f"""
        <table style="width:100%;font-size:12px;border:none;margin-top:5px;">
          <tr style="background-color:rgba(0,0,0,0.05);">
            <th>3D</th><th>7D</th><th>1M</th><th>3M</th>
          </tr>
          <tr>
            <td style="color:{'green' if trend_dict['3D']>=0 else 'red'}">{trend_dict['3D']:.2f}%</td>
            <td style="color:{'green' if trend_dict['7D']>=0 else 'red'}">{trend_dict['7D']:.2f}%</td>
            <td style="color:{'green' if trend_dict['1M']>=0 else 'red'}">{trend_dict['1M']:.2f}%</td>
            <td style="color:{'green' if trend_dict['3M']>=0 else 'red'}">{trend_dict['3M']:.2f}%</td>
          </tr>
        </table>""", unsafe_allow_html=True)

    # ── 5-step pipeline columns ───────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("### 🏛️ Step 1\n**Structural Baseline**")
        st.metric("National Debt",
                  f"${data['Debt']['latest']:.1f} T",
                  f"{data['Debt']['QoQ']:.2f}% QoQ")
        st.caption("Unfunded Liabilities: **$200+ T**")
        st.caption("Annual Deficit: **~$1.9 T**")
        st.caption("Governments eventually monetize debt indirectly through the banks and central bank liquidity.")

    with col2:
        st.markdown("### 🖨️ Step 2\n**Monetary Liquidity**")
        st.metric("M2 Money Supply",
                  f"${data['M2']['latest']:.1f} T",
                  f"{data['M2']['MoM']:.2f}% MoM")
        st.caption(f"3-Month Shift: {data['M2']['3MoM']:.2f}%")
        st.caption("Cash, Checking/Savings deposits, Money market funds")

    with col3:
        st.markdown("### 🌎 Step 3\n**Market Indicators**")
        st.write("**CRB Commodity Index **")
        display_trend_table(data['CRB'])
        st.caption("Measures: Oil, Metals, Agriculture, Raw materials")
        st.write("**US Dollar Index (DXY)**")
        display_trend_table(data['DXY'])
        st.caption("Measures USD against major currencies. A stronger dollar Makes imports cheaper hence Lowers commodity prices.")

    with col4:
        st.markdown("### 🏭 Step 4\n**Supply-Side Costs**")
        st.metric("Producer Price Index",
                  f"{data['PPI']['latest']:.1f}",
                  f"{data['PPI']['MoM']:.2f}% MoM")
        st.caption(f"3-Month Trajectory: +{data['PPI']['3MoM']:.2f}%")
        st.caption("PPI - paid by producers before products reach consumers.")
        st.caption("Increases because of Input, Labor, Transportation, Tariffs, Suppy Chain, Energy costs.")

    with col5:
        st.markdown("### 🛒 Step 5\n**End Impact**")
        st.markdown(f"""
        <div style="background-color:#2b2b2b;padding:15px;border-radius:8px;
                    text-align:center;color:white;border-left:6px solid #ff4b4b;">
          <p style="margin:0;font-size:12px;font-weight:bold;color:#ff4b4b;">CONSUMER PRICES (CPI)</p>
          <p style="margin:5px 0;font-size:28px;font-weight:bold;">+{data['CPI']['MoM']:.2f}%</p>
          <p style="margin:0;font-size:11px;opacity:0.8;">Latest MoM Print</p>
          <p style="margin:2px 0 0 0;font-size:11px;opacity:0.8;">
            3-Month Trend: {data['CPI']['3MoM']:.2f}%</p>
        </div>""", unsafe_allow_html=True)

    st.subheader("Losers")
    st.markdown("""
    1. Financial Repression → Savers lose purchasing power.<br>
    2. Nominal GDP Growth → AI, manufacturing, energy.<br>
    3. Controlled Inflation → Erode debt without panic.<br>
    4. Currency Devaluation → Foreign Treasury holders lose.<br>
    5. Yield Curve Control → Negative real yields.<br>
    6. Pension Adjustment → Raise retirement age.<br>
    7. Tax Inflation → Bracket creep boosts revenue.<br>
    8. Asset Inflation → Stocks & housing rise.<br>
    9. Energy Dominance → GDP and trade balance improve.<br>
    10. Selective Default → Inflation exceeds borrowing cost.<br>
    11. Sovereign Wealth Strategy → Build productive assets.<br>
    12. Gold Revaluation → Treasury balance sheet improves.<br>
    13. Capital Controls → Restrict capital movement.
    """, unsafe_allow_html=True)

    # ── 401(k) Status Box ─────────────────────────────────────────────────────
    st.markdown("""<style>
    .account-box { background:#F9FAFB;padding:20px;border-radius:8px;
                   border:1px solid #E5E7EB;margin-top:15px;margin-bottom:20px; }
    .account-title { font-size:14px;color:#111827;font-weight:600;
                     text-transform:uppercase;letter-spacing:0.05em;
                     margin-bottom:12px;display:flex;align-items:center;gap:6px; }
    .k-grid  { display:grid;grid-template-columns:repeat(4,1fr);gap:12px; }
    .k-card  { background:#FFFFFF;padding:14px;border-radius:6px;
                border:1px solid #E5E7EB;text-align:left; }
    .k-label { font-size:10.5px;color:#6B7280;font-weight:500;
                text-transform:uppercase;margin-bottom:2px; }
    .k-value { font-size:18px;color:#111827;font-weight:700; }
    .k-green { color:#10B981;font-weight:600;font-size:12px;margin-top:2px; }
    </style>""", unsafe_allow_html=True)

    sp500_ytd = fetch_sp500_ytd_return()
    baseline_balance, total_contributions, account_config_loaded = load_account_config()
    if not account_config_loaded:
        st.sidebar.warning("account.json not found or invalid – using default ledger values.")

    market_gains    = baseline_balance * (sp500_ytd / 100)
    current_balance = baseline_balance + total_contributions + market_gains

    st.markdown(f"""
    <div class='account-box'>
      <div class='account-title'>🇺🇸 401(k) Plan Account Status Summary &amp; Risk Vector Ledger</div>
      <div class='k-grid'>
        <div class='k-card'>
          <div class='k-label'>Total Vested Balance (USD)</div>
          <div class='k-value'>${current_balance:,.2f}</div>
          <div class='k-green'>⚡ Fully Vested Assets</div>
        </div>
        <div class='k-card'>
          <div class='k-label'>YTD Contribution Allocation</div>
          <div class='k-value'>${total_contributions:,.2f}</div>
          <div class='k-green'>Max Out Track Active</div>
        </div>
        <div class='k-card'>
          <div class='k-label'>Core Account Return Profile</div>
          <div class='k-value'>+{sp500_ytd}%</div>
          <div class='k-green'>📈 Tracking S&amp;P 500 Index</div>
        </div>
        <div class='k-card'>
          <div class='k-label'>Asset Growth Component</div>
          <div class='k-value'>+${market_gains:,.2f}</div>
          <div class='k-green'>Market Compounding Value</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── BLS price-index chart ─────────────────────────────────────────────────
    df_full, extended_month_labels = fetch_bls_data(BLS_API_KEY, buffer_months=4)

    df     = None
    months = None

    if df_full is not None:
        anchor_idx = None
        for i in range(len(extended_month_labels) - 1, 0, -1):
            cur_col  = extended_month_labels[i]
            prev_col = extended_month_labels[i - 1]
            if (cur_col in df_full.columns and prev_col in df_full.columns
                    and df_full[cur_col].notna().all()
                    and df_full[prev_col].notna().all()):
                anchor_idx = i
                break

        if anchor_idx is not None:
            start_idx    = max(0, anchor_idx - 6)
            window_labels = extended_month_labels[start_idx:anchor_idx + 1]
            if len(window_labels) >= 2:
                raw_window = df_full[["Index Type"] + window_labels].copy()
                pct_df     = raw_window[["Index Type"]].copy()
                for j in range(1, len(window_labels)):
                    cur_col  = window_labels[j]
                    prev_col = window_labels[j - 1]
                    pct_df[cur_col] = (
                        (raw_window[cur_col] - raw_window[prev_col])
                        / raw_window[prev_col]
                    ) * 100
                months = window_labels[1:]
                df     = pct_df

    if df is None:
        months = [month_label(y, m) for y, m in get_last_n_months(6)]

    latest_month       = months[-1]
    latest_month_short = latest_month.split(" ")[0]

    if df is None:
        st.sidebar.warning("BLS API unavailable – showing sample data.")
        fallback_values = {
            "Producer Price Index": [0.6, 0.5, 0.4, 0.7, 1.1, 1.4],
            "Export Price Index":   [1.9, 1.7, 1.6, 1.5, 2.4, 3.3],
            "Import Price Index":   [1.0, 0.9, 0.8, 0.9, 1.4, 1.9],
        }
        fallback_data = {"Index Type": list(fallback_values.keys())}
        for i, ml in enumerate(months):
            fallback_data[ml] = [fallback_values[k][i] for k in fallback_values]
        df = pd.DataFrame(fallback_data)

    st.subheader(f"🚀 Latest Focus: {latest_month} Performance")
    col1, col2, col3 = st.columns(3)

    def get_val(idx_name, month_name):
        return df.loc[df["Index Type"] == idx_name, month_name].values[0]

    with col1:
        st.metric(f"Producer Price Index ({latest_month_short})",
                  f"{get_val('Producer Price Index', latest_month):.2f}%",
                  "MoM Performance")
    with col2:
        st.metric(f"Export Price Index ({latest_month_short})",
                  f"{get_val('Export Price Index', latest_month):.2f}%",
                  "MoM Performance")
    with col3:
        st.metric(f"Import Price Index ({latest_month_short})",
                  f"{get_val('Import Price Index', latest_month):.2f}%",
                  "MoM Performance")

    st.markdown("---")

    left_chart_col, right_table_col = st.columns([2, 1])

    with left_chart_col:
        st.subheader("Live Trend Visualization (Last 6 Months)")
        fig = go.Figure()
        for _, row in df.iterrows():
            fig.add_trace(go.Scatter(
                x=months, y=[row[m] for m in months],
                mode='lines+markers', name=row["Index Type"],
                line=dict(width=3), marker=dict(size=8),
            ))
        fig.update_layout(
            xaxis_title="Reporting Period", yaxis_title="MoM Growth (%)",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right_table_col:
        st.subheader("Data Records (Live API)")
        formatted_df = df.copy()
        for col in months:
            formatted_df[col] = formatted_df[col].apply(
                lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
            )
        st.dataframe(formatted_df, hide_index=True, use_container_width=True)
        st.info("💡 API requests are cached via `st.cache_data` to stay within BLS daily limits.")
