"""
ui/pages/page_ai_ipo_macro.py  –  AI IPO Macro Dashboard screen
────────────────────────────────────────────────────────────────
Extracted from us_economy_and_news.py (Dashboard 2).
Entry point: render()
"""

import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import datetime
import json
import io
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Re-use the GIFT Nifty banner from page_us_economy (avoids duplicate cache keys)
from ui.pages.page_us_economy import render_gift_nifty_banner


# ── Static data tables ────────────────────────────────────────────────────────

IPO_DATA = [
    {
        'name': 'SpaceX / xAI', 'badge': 'LIVE TODAY', 'badge_class': 'badge-live',
        'rows': [
            ('Ticker',         'SPCX · Nasdaq'),
            ('IPO price',      '$135 / share'),
            ('Valuation',      '$1.75 trillion'),
            ('2025 Revenue',   '$18.67B'),
            ('Profitability',  'Unproven at scale'),
            ('Lockup expiry',  '~Dec 2026 (~180d)'),
            ('Key risk',       'Dual-class · Musk control'),
            ('xAI merged',     'Feb 2026 all-stock deal'),
        ]
    },
    {
        'name': 'Anthropic', 'badge': 'OCT 2026', 'badge_class': 'badge-oct',
        'rows': [
            ('S-1 filed',        'Jun 1, 2026 (confidential)'),
            ('Target listing',   'Oct 23, 2026'),
            ('Series H val.',    '$965B → ~$1T+ IPO'),
            ('ARR (May 2026)',   '$47B run-rate (5× in 5mo)'),
            ('First op. profit', 'Q2 2026 projected'),
            ('Lockup expiry',    '~Apr 2027 (THE SIGNAL)'),
            ('Lead banks',       'Goldman · JPMorgan · MS'),
            ('Key risk',         'Compute cost / SEC review'),
        ]
    },
    {
        'name': 'OpenAI', 'badge': '2027', 'badge_class': 'badge-2027',
        'rows': [
            ('S-1 filed',          'Jun 8, 2026 (confidential)'),
            ('Target listing',     'Sep–Dec 2026 or 2027'),
            ('Valuation',          '$730B–$1T+'),
            ('2025 ARR',           '$20B+'),
            ('2026 loss (proj.)',  '$14B GAAP ($25B full)'),
            ('Profitable by',      '~2030 (CFO guidance)'),
            ('Add. funding needed','$207B by 2030 (HSBC)'),
            ('Key risk',           'Losses, restructuring'),
        ]
    },
]

TIMELINE_ROWS = [
    {
        'period': 'Jun 2026\nTODAY', 'alert': True,
        'cells': [
            ('cell-ipo',   'SpaceX IPO live',       'SPCX lists $1.75T. Largest IPO in history. xAI merged in.'),
            ('cell-macro', 'Iran war inflation',    'CPI 4%+, front-end rates under pressure. Warsh paralysed.'),
            ('cell-401k',  '$141K avg, ↓4%',        'Q1 2026 Fidelity data. First cracks. Loan withdrawals rising.'),
            ('cell-gold',  'Gold $4,139 / INR 95.25','Stress already here. Accumulate SGBs now. Reduce US exposure.'),
            ('cell-india', 'Accumulate phase',      'SGBs aggressively. Exit IT stocks. Build cash dry powder.'),
        ]
    },
    {
        'period': 'Jul–Sep 2026', 'alert': False,
        'cells': [
            ('cell-ipo',   'AI melt-up',        'Anthropic IPO hype lifts Nvidia, MSFT, GOOGL. FOMO max.'),
            ('cell-macro', 'Fed paralysed',     "Won't hike into midterms. Warsh \"disinflationary\" narrative."),
            ('cell-401k',  'Brief recovery',    'AI rally lifts 401(k) briefly. False sense of safety.'),
            ('cell-gold',  'Gold grinds up',    'INR ~92–97. Use the rally to EXIT Indian IT stocks.'),
            ('cell-india', 'EXIT equities',     'Sell US-facing exposure into AI rally. These are gift prices.'),
        ]
    },
    {
        'period': 'Oct 2026\n🚨 RED ALERT', 'alert': True,
        'cells': [
            ('cell-ipo',   'Anthropic IPO',    '~$1T+ valuation. Peak AI narrative. Maximum global FOMO.'),
            ('cell-macro', 'Midterms Nov 2026','Every policy lever pulled. Political liquidity injections.'),
            ('cell-401k',  'Peak balances',    '40% S&P = 10 AI stocks. Concentration at extremes.'),
            ('cell-gold',  'Gold $4,500+?',    'Max defense. SGBs + physical. IPO = top signal, not opportunity.'),
            ('cell-india', 'MAX DEFENSE',      'Oct 1 = full defensive. Gold 45%, Cash 30%, Equity 20%.'),
        ]
    },
    {
        'period': 'Nov–Dec 2026', 'alert': False,
        'cells': [
            ('cell-ipo',   'OpenAI S-1 public', 'Files ~Aug, lists Sep–Dec. $14–25B loss disclosed. Market shock.'),
            ('cell-macro', 'Midterm results',   'Political cover fades post-election. Bond stress resumes.'),
            ('cell-401k',  'Deceptive calm',    "Lockups not expired. People complacent. False floor."),
            ('cell-gold',  'Hold gold firm',    "This period will feel like you're wrong. You are NOT."),
            ('cell-india', 'Do nothing',        'Hold position. Add gold on any weakness. Watch SPCX closely.'),
        ]
    },
    {
        'period': 'Jan–Mar 2027', 'alert': False,
        'cells': [
            ('cell-ipo',   'Earnings reality',  'SpaceX Q3/Q4 + Anthropic first 2 public quarters. Gap visible.'),
            ('cell-macro', 'Bond stress builds','Debt maturity wall. Japan/Korea EM pricing worsens.'),
            ('cell-401k',  'Volatility rising', '$10–50K swings per week. Anxiety builds among retirees.'),
            ('cell-gold',  'Gold $4,800+?',     'INR ~97–101. Gold INR flat to up. Keep all gold.'),
            ('cell-india', 'Watchlist ready',   'Finalise quality India names. Cash dry powder at maximum.'),
        ]
    },
    {
        'period': 'Apr 2027\n⚠ THE SIGNAL', 'alert': True,
        'cells': [
            ('cell-danger','Lockup expiry',    'Anthropic: Google, Amazon, employees ALL can sell. Structural.'),
            ('cell-danger','Cascade begins',   '"If Anthropic can\'t monetise, who can?" Nvidia guidance cut.'),
            ('cell-401k',  'Balances crater',  '40% S&P concentration = brutal 401(k) loss. Forced withdrawals.'),
            ('cell-gold',  'Gold dips briefly',"Margin call selling. INR 99–103. Gold INR flat. HOLD. DON'T SELL."),
            ('cell-india', 'Nifty -20%+',      'FII outflows. INR 99–103. Deploy 1st tranche of equity (10–15%).'),
        ]
    },
    {
        'period': 'May–Jun 2027', 'alert': False,
        'cells': [
            ('cell-danger','Bond contagion',  'Treasuries sold not bought. 10Y spikes. Everything liquidated.'),
            ('cell-danger','Fed choice point','Dollar or bonds? Warsh must decide. Extreme volatility.'),
            ('cell-401k',  'Retirement crisis','Boomers near retirement forced to sell at lows. Amplifies crash.'),
            ('cell-gold',  'Gold recovers',   'Fed signals printing. Gold starts explosive move. INR 101–106.'),
            ('cell-india', '2nd tranche',     'Nifty -30–35%. Add domestic: FMCG, pharma, utilities, PSU banks.'),
        ]
    },
    {
        'period': 'Jul–Sep 2027\n🖨 FED PRINTS', 'alert': True,
        'cells': [
            ('cell-ipo',   'AI reset',           'SpaceX may survive (rockets real). Anthropic/OpenAI 25x→8x rev.'),
            ('cell-macro', 'QE / YCC announced', 'Fed buys bonds. Dollar collapses. New monetary framework talk.'),
            ('cell-401k',  'Permanent damage',   'Early retirees locked in losses. Social Security stress narrative.'),
            ('cell-gold',  'Gold EXPLODES',       '$5,500–$7,000? INR recovering 95→88. Gold INR ₹480K–₹600K. HOLD.'),
            ('cell-india', '3rd tranche',         'Nifty bottoming. Quality mid-cap India. Build silver aggressively.'),
        ]
    },
    {
        'period': 'H2 2027–2028', 'alert': False,
        'cells': [
            ('cell-phase', 'Stabilisation',   'Surviving AI cos trade on actual earnings. Dot-com survivors.'),
            ('cell-macro', 'Dollar rebases 30%','INR strengthens 100→70–75. De-dollarisation accelerates.'),
            ('cell-401k',  'Policy response', '401(k) bailout political pressure. Inflationary. Validates thesis.'),
            ('cell-gold',  'EXIT GOLD NOW',   'INR strong = gold INR falls. Sell before INR hits 75. Rotate.'),
            ('cell-india', 'Full India equity','FII money floods back. INR+Nifty double tailwind. Generational.'),
        ]
    },
    {
        'period': '2028+', 'alert': False,
        'cells': [
            ('cell-silver','Silver peaks late', 'Gold/silver ratio 88→45. Silver $130–150. Exits AFTER gold.'),
            ('cell-macro', 'New cycle',          'India structural winner. RBI gold reserves = strong INR backing.'),
            ('cell-401k',  'US consumption shock','Boomers poorer. US recession deepens. India domestic shines.'),
            ('cell-gold',  '5–10% permanent',   'Keep small gold forever (SGBs for 2.5% yield). Rest → India equity.'),
            ('cell-india', 'Compound India',    'Domestic consumption. Pharma. Power infra. 5–10yr holding.'),
        ]
    },
]

PORTFOLIO_PHASES = {
    'Now – Sep 2026':      {'Gold/SGBs': 35, 'INR Cash': 25, 'India Equity': 30, 'Silver': 10},
    'Oct 2026 – Mar 2027': {'Gold/SGBs': 45, 'INR Cash': 30, 'India Equity': 20, 'Silver':  5},
    'Apr – Jun 2027':      {'Gold/SGBs': 50, 'INR Cash': 35, 'India Equity': 10, 'Silver':  5},
    'H2 2027':             {'Gold/SGBs': 35, 'INR Cash': 15, 'India Equity': 35, 'Silver': 15},
    '2028+':               {'Gold/SGBs': 25, 'INR Cash': 15, 'India Equity': 45, 'Silver': 15},
}

PHASE_EVENTS = [
    {'date': '2026-06-12', 'label': 'SpaceX IPO\n$1.75T',       'color': '#818cf8', 'symbol': 'star'},
    {'date': '2026-10-23', 'label': 'Anthropic IPO\n~$1T',      'color': '#c084fc', 'symbol': 'star'},
    {'date': '2026-11-03', 'label': 'US Midterms',               'color': '#60a5fa', 'symbol': 'triangle-up'},
    {'date': '2026-12-15', 'label': 'SpaceX lockup\nexpiry',    'color': '#f87171', 'symbol': 'x'},
    {'date': '2027-04-23', 'label': 'Anthropic\nlockup expiry', 'color': '#f87171', 'symbol': 'x'},
    {'date': '2027-06-15', 'label': 'Fed decision\npoint',      'color': '#fbbf24', 'symbol': 'diamond'},
    {'date': '2027-09-01', 'label': 'Fed prints\n(projected)',  'color': '#4ade80', 'symbol': 'circle'},
    {'date': '2028-03-01', 'label': 'USD rebase\n-30% INR',     'color': '#34d399', 'symbol': 'triangle-down'},
]


# ── Data fetching (all cached) ────────────────────────────────────────────────

@st.cache_data(ttl=300)
def fetch_gold_historical():
    try:
        url = "https://raw.githubusercontent.com/datasets/gold-prices/master/data/monthly.csv"
        r   = requests.get(url, timeout=8)
        df  = pd.read_csv(io.StringIO(r.text))
        df.columns = ['Date', 'Price']
        df['Date'] = pd.to_datetime(df['Date'])
        return df[df['Date'] >= '2000-01-01'].copy()
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_sp500_historical():
    try:
        url = "https://raw.githubusercontent.com/vega/vega-datasets/main/data/sp500.csv"
        r   = requests.get(url, timeout=8)
        df  = pd.read_csv(io.StringIO(r.text))
        df.columns = ['Date', 'Price']
        df['Date'] = pd.to_datetime(df['Date'], format='%b %d %Y', errors='coerce')
        return df.dropna().sort_values('Date')
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_crude_historical():
    oil_anchors = {
        '2000-01': 27,  '2001-01': 28,  '2002-01': 19,  '2003-01': 33,
        '2004-01': 34,  '2005-01': 47,  '2006-01': 65,  '2007-01': 58,
        '2008-01': 91,  '2008-07': 133, '2009-01': 41,  '2010-01': 79,
        '2011-01': 91,  '2012-01': 100, '2013-01': 94,  '2014-01': 95,
        '2015-01': 47,  '2016-01': 32,  '2017-01': 52,  '2018-01': 62,
        '2019-01': 53,  '2020-01': 61,  '2020-03': 20,  '2021-01': 52,
        '2022-01': 83,  '2022-06': 114, '2023-01': 76,  '2024-01': 73,
        '2025-01': 75,  '2025-06': 68,  '2026-01': 82,  '2026-06': 89,
    }
    anchor_df = pd.DataFrame([
        {'Date': pd.to_datetime(k), 'Price': v}
        for k, v in oil_anchors.items()
    ]).sort_values('Date')
    interp = (anchor_df.set_index('Date')
              .reindex(pd.date_range(anchor_df['Date'].min(),
                                     anchor_df['Date'].max(), freq='MS'))
              .interpolate('linear').reset_index())
    interp.columns = ['Date', 'Price']
    return interp


@st.cache_data(ttl=300)
def fetch_inr_historical():
    inr_anchors = {
        '2000-01': 44.9,  '2002-01': 48.2,  '2004-01': 45.3,  '2006-01': 44.1,
        '2008-01': 39.4,  '2008-07': 42.8,  '2009-01': 48.9,  '2010-01': 45.7,
        '2011-01': 44.7,  '2012-01': 52.6,  '2013-07': 59.8,  '2014-01': 62.5,
        '2015-01': 63.1,  '2016-01': 67.8,  '2017-01': 67.7,  '2018-10': 74.2,
        '2019-01': 71.3,  '2020-01': 71.3,  '2020-04': 76.4,  '2021-01': 73.1,
        '2022-01': 74.5,  '2022-10': 83.1,  '2023-01': 81.9,  '2024-01': 83.2,
        '2025-01': 86.5,  '2025-06': 84.1,  '2025-12': 87.3,
        '2026-01': 88.9,  '2026-03': 91.2,  '2026-06': 95.25,
    }
    anchor_df = pd.DataFrame([
        {'Date': pd.to_datetime(k), 'Rate': v}
        for k, v in inr_anchors.items()
    ]).sort_values('Date')
    interp = (anchor_df.set_index('Date')
              .reindex(pd.date_range(anchor_df['Date'].min(),
                                     anchor_df['Date'].max(), freq='MS'))
              .interpolate('linear').reset_index())
    interp.columns = ['Date', 'Rate']
    return interp


def _yf_fetch(ticker: str, label: str):
    """Fetch current price and day-change % from Yahoo Finance via yfinance."""
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = info.last_price
        prev  = info.previous_close
        if price is None or prev is None or prev == 0:
            raise ValueError("missing price")
        pct   = (price - prev) / prev * 100
        sign  = '+' if pct >= 0 else ''
        return {
            'value':  price,
            'change': f"{sign}{pct:.2f}%",
            'change_pct': pct,         # raw float for colour logic
            'source': f'Yahoo Finance · {label}',
            'status': 'live',
        }
    except Exception as e:
        return {
            'value':  None,
            'change': 'N/A',
            'change_pct': 0.0,
            'source': f'Yahoo Finance error ({label})',
            'status': 'error',
        }


@st.cache_data(ttl=300)
def get_live_prices():
    # Yahoo Finance tickers
    gold     = _yf_fetch('GC=F',    'GC=F')       # Gold futures (USD/oz)
    usd_inr  = _yf_fetch('USDINR=X','USDINR=X')   # USD/INR spot
    wti      = _yf_fetch('CL=F',    'CL=F')        # WTI Crude futures
    brent    = _yf_fetch('BZ=F',    'BZ=F')        # Brent Crude futures
    sp500    = _yf_fetch('^GSPC',   'GSPC')        # S&P 500
    nifty    = _yf_fetch('^NSEI',   'NSEI')        # Nifty 50
    btc      = _yf_fetch('BTC-USD', 'BTC-USD')     # Bitcoin
    us10y    = _yf_fetch('^TNX',    'TNX')         # US 10-Year yield

    prices = {
        'gold_usd':    gold,
        'usd_inr':     usd_inr,
        'crude_wti':   wti,
        'crude_brent': brent,
        'sp500':       sp500,
        'nifty':       nifty,
        'btc_usd':     btc,
        'us10y':       us10y,
        'gold_inr':    {'value': None, 'change': 'derived', 'change_pct': 0.0,
                        'source': 'Gold × INR / 31.1035', 'status': 'derived'},
    }

    # Derive Gold/INR (per gram)
    g_val   = gold.get('value')
    inr_val = usd_inr.get('value')
    if g_val and inr_val:
        gold_inr_val = g_val * inr_val / 31.1035
        # approximate combined day-change
        gold_pct = gold.get('change_pct', 0.0)
        inr_pct  = usd_inr.get('change_pct', 0.0)
        combined_pct = gold_pct + inr_pct           # additive approximation
        sign = '+' if combined_pct >= 0 else ''
        prices['gold_inr'] = {
            'value':      round(gold_inr_val, 0),
            'change':     f"{sign}{combined_pct:.2f}%",
            'change_pct': combined_pct,
            'source':     'Gold × INR / 31.1035',
            'status':     'derived',
        }

    return prices


@st.cache_data(ttl=600)
def build_scenario_projections():
    today  = pd.to_datetime('2026-06-12')
    end    = pd.to_datetime('2029-01-01')
    months = pd.date_range(today, end, freq='MS')

    def sigmoid_interp(x, x0, k=8):
        return 1 / (1 + np.exp(-k * (x - x0)))

    n  = len(months)
    t  = np.linspace(0, 1, n)

    ph1_end  = 0.14
    ph2_end  = 0.35
    ph3_peak = 0.50
    ph4_start = 0.60

    base_gold = 4139
    gold_base = base_gold * (1 + 0.12 * t)
    gold_rupture = base_gold * np.where(
        t < ph1_end,  1 + 0.08*t/ph1_end,
        np.where(t < ph2_end,
                 (1 + 0.08) * (1 - 0.05*sigmoid_interp(t, ph1_end+0.05)),
                 np.where(t < ph3_peak,
                          (1 + 0.03) + 0.75*sigmoid_interp(t, (ph2_end+ph3_peak)/2, k=12),
                          1.78 - 0.15*(t - ph3_peak)/(1-ph3_peak))))
    gold_bull = base_gold * np.where(
        t < ph3_peak,
        1 + 0.90*sigmoid_interp(t, ph2_end+0.05, k=10),
        1.90 - 0.05*(t - ph3_peak))

    base_inr = 95.25
    inr_rupture = base_inr * np.where(
        t < ph1_end, 1 + 0.05*t/ph1_end,
        np.where(t < ph2_end, 1.05 + 0.02*(t-ph1_end)/(ph2_end-ph1_end),
                 np.where(t < ph3_peak,
                          1.07 - 0.10*sigmoid_interp(t, ph2_end+0.08, k=10),
                          0.97 - 0.18*(t-ph3_peak)/(1-ph3_peak))))
    inr_rupture = np.clip(inr_rupture, 0.75, 1.12) * base_inr

    base_crude = 89.4
    crude_rupture = base_crude * np.where(
        t < ph1_end,  1 + 0.08*t/ph1_end,
        np.where(t < ph2_end, 1.08 + 0.12*(t-ph1_end)/(ph2_end-ph1_end),
                 np.where(t < ph3_peak,
                          1.20 - 0.10*sigmoid_interp(t, ph3_peak-0.05),
                          1.10 + 0.05*np.sin(20*(t-ph3_peak)))))

    base_sp = 5821
    sp_rupture = base_sp * np.where(
        t < ph1_end,  1 + 0.06*t/ph1_end,
        np.where(t < ph2_end, 1.06 - 0.02*(t-ph1_end)/(ph2_end-ph1_end),
                 np.where(t < ph3_peak,
                          1.04 - 0.40*sigmoid_interp(t, ph2_end+0.03, k=15),
                          0.65 + 0.20*sigmoid_interp(t, ph4_start, k=10))))

    base_nifty = 24350
    nifty_rupture = base_nifty * np.where(
        t < ph1_end,  1 + 0.04*t/ph1_end,
        np.where(t < ph2_end, 1.04 - 0.01*(t-ph1_end)/(ph2_end-ph1_end),
                 np.where(t < ph3_peak,
                          1.03 - 0.32*sigmoid_interp(t, ph2_end+0.05, k=12),
                          0.72 + 0.35*sigmoid_interp(t, ph4_start+0.05, k=8))))

    gold_inr_rupture = gold_rupture * inr_rupture / 31.1035

    return pd.DataFrame({
        'Date': months, 't': t,
        'gold_base': gold_base, 'gold_rupture': gold_rupture, 'gold_bull': gold_bull,
        'inr_rupture': inr_rupture, 'crude_rupture': crude_rupture,
        'sp_rupture': sp_rupture, 'nifty_rupture': nifty_rupture,
        'gold_inr_rupture': gold_inr_rupture,
    })


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    render_gift_nifty_banner()

    # ── News overlay loader ───────────────────────────────────────────────────
    def load_news_data():
        for path in [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../outputs/news.json'),
            os.path.join(os.getcwd(), 'news.json'),
            'news.json',
        ]:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
        return {}

    _raw_news    = load_news_data()
    NEWS_OVERLAY = {}
    for phase_data in _raw_news.values():
        if isinstance(phase_data, dict):
            for key, val in phase_data.items():
                if isinstance(val, dict) and 'status' in val:
                    NEWS_OVERLAY[key] = val

    STATUS_BADGE = {
        'Validated 🟢':   ('background:#EAF3DE;color:#27500A;border:0.5px solid #97C459;', '✅'),
        'Partial 🟡':     ('background:#FFF8E8;color:#854F0B;border:0.5px solid #FAC775;', '⚠️'),
        'Too Early ⏳':   ('background:#F1EFE8;color:#666560;border:0.5px solid #B4B2A9;', '⏳'),
        'Invalidated 🔴': ('background:#FCEBEB;color:#791F1F;border:0.5px solid #F09595;', '❌'),
    }
    DEFAULT_BADGE = ('background:#F1EFE8;color:#888480;border:0.5px solid #D6D3CA;', '·')

    def cell_key(row_idx, col_idx):
        return f"{chr(65 + col_idx)}{row_idx + 1}"

    def news_overlay_html(row_idx, col_idx):
        key  = cell_key(row_idx, col_idx)
        data = NEWS_OVERLAY.get(key)
        if not data:
            return ''
        status     = data.get('status', '')
        signal     = data.get('real_world_signal', '')
        divergence = data.get('thesis_divergence', '')
        confidence = data.get('confidence', '')
        style, icon = STATUS_BADGE.get(status, DEFAULT_BADGE)
        show_div = (
            '' if divergence in ('', 'None', 'None.')
            else f"<div style='margin-top:5px;padding-top:5px;"
                 f"border-top:0.5px solid rgba(0,0,0,0.1);"
                 f"font-size:9.5px;opacity:0.8;'>"
                 f"<b>Divergence:</b> {divergence}</div>"
        )
        return (
            f"<div style='margin-top:6px;padding:5px 7px;border-radius:5px;"
            f"font-size:10px;line-height:1.45;{style}'>"
            f"<div style='display:flex;align-items:center;justify-content:space-between;"
            f"margin-bottom:3px;'>"
            f"<span style='font-weight:600;font-size:9px;text-transform:uppercase;"
            f"letter-spacing:0.06em;'>{icon} {status}</span>"
            f"<span style='font-size:9px;opacity:0.7;'>conf: {confidence}</span></div>"
            f"<div style='font-size:10px;'>{signal}</div>{show_div}</div>"
        )

    # ── CSS ───────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
      .main { background-color: #F5F4F0; }
      .block-container { padding: 1.5rem 2rem; max-width: 1400px; background: #F5F4F0; }
      .metric-card { background:#FFFFFF;border:0.5px solid #D6D3CA;border-radius:8px;padding:10px 12px; }
      .metric-label   { font-size:10px;color:#666560;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:3px; }
      .metric-value   { font-size:20px;font-weight:500;color:#1A1A18;margin-bottom:2px; }
      .metric-sub     { font-size:10px;color:#888480; }
      .metric-up      { color:#2D6A0F; }
      .metric-down    { color:#7A1F1F; }
      .metric-warn    { color:#7A4A0A; }
      .metric-neutral { color:#1A1A18; }
      .section-header { font-size:11px;font-weight:500;color:#888480;text-transform:uppercase;
                        letter-spacing:0.06em;border-bottom:0.5px solid #D6D3CA;
                        padding-bottom:5px;margin:18px 0 10px; }
      .ipo-card  { background:#FFFFFF;border:0.5px solid #D6D3CA;border-radius:8px;padding:10px 12px;height:100%; }
      .ipo-title { font-size:12px;font-weight:500;color:#1A1A18;margin-bottom:8px;display:flex;align-items:center;gap:6px; }
      .ipo-row   { display:flex;justify-content:space-between;font-size:10px;color:#888480;
                   margin-bottom:3px;border-bottom:0.5px solid #ECEAE4;padding-bottom:3px; }
      .ipo-row span:last-child { color:#1A1A18;font-weight:500; }
      .badge      { font-size:10px;padding:2px 6px;border-radius:4px;font-weight:500; }
      .badge-live { background:#FCEBEB;color:#791F1F; }
      .badge-oct  { background:#EEEDFE;color:#3C3489; }
      .badge-2027 { background:#FAEEDA;color:#633806; }
      .badge-green{ background:#EAF3DE;color:#27500A; }
      .timeline-row { display:grid;grid-template-columns:90px 1fr 1fr 1fr 1fr 1fr;
                      gap:4px;margin-bottom:3px;align-items:stretch;min-height:44px; }
      .tl-date  { font-size:11px;font-weight:500;color:#888480;display:flex;
                  align-items:center;padding-right:6px;line-height:1.3; }
      .tl-cell  { border-radius:6px;padding:5px 7px;font-size:10.5px;line-height:1.35;
                  display:flex;flex-direction:column;justify-content:center; }
      .tl-cell b    { font-weight:500;font-size:11px;display:block;margin-bottom:1px;color:inherit; }
      .tl-cell span { font-size:10px;opacity:0.85; }
      .cell-ipo    { background:#EEEDFE;color:#3C3489;border:0.5px solid #AFA9EC; }
      .cell-macro  { background:#E6F1FB;color:#0C447C;border:0.5px solid #85B7EB; }
      .cell-401k   { background:#FAECE7;color:#712B13;border:0.5px solid #F0997B; }
      .cell-gold   { background:#FFF8E8;color:#854F0B;border:0.5px solid #FAC775; }
      .cell-india  { background:#EAF3DE;color:#27500A;border:0.5px solid #97C459; }
      .cell-danger { background:#FCEBEB;color:#791F1F;border:0.5px solid #F09595; }
      .cell-silver { background:#E1F5EE;color:#085041;border:0.5px solid #5DCAA5; }
      .cell-phase  { background:#F1EFE8;color:#2C2C2A;border:0.5px solid #B4B2A9; }
      .cell-warn   { background:#FAEEDA;color:#633806;border:0.5px solid #FAC775; }
      .phase-pill { display:inline-flex;align-items:center;gap:8px;border-radius:6px;
                    padding:4px 10px;font-size:11px;font-weight:500;margin-right:6px;margin-bottom:6px; }
      .phase-dot  { width:8px;height:8px;border-radius:50%;flex-shrink:0; }
      .ph-vacuum  { background:#EEEDFE;color:#3C3489;border:0.5px solid #AFA9EC; }
      .ph-stress  { background:#FAEEDA;color:#633806;border:0.5px solid #FAC775; }
      .ph-rupture { background:#FCEBEB;color:#791F1F;border:0.5px solid #F09595; }
      .ph-reset   { background:#EAF3DE;color:#27500A;border:0.5px solid #97C459; }
      .note-box { background:#FFFFFF;border:0.5px solid #D6D3CA;border-left:2px solid #B4B2A9;
                  border-radius:6px;padding:8px 12px;font-size:10px;color:#666560;margin-top:10px; }
      .warn-box { background:#FFFBF0;border:0.5px solid #FAC775;border-left:3px solid #E8940A;
                  border-radius:6px;padding:10px 14px;font-size:12px;color:#633806;margin-bottom:14px; }
      .stTabs [data-baseweb="tab-list"] { background:#ECEAE4;border-radius:8px;padding:3px;gap:3px; }
      .stTabs [data-baseweb="tab"]      { background:transparent;color:#888480;border-radius:6px;
                                           padding:5px 14px;font-size:13px; }
      .stTabs [aria-selected="true"]    { background:#FFFFFF !important;color:#1A1A18 !important;
                                           box-shadow:0 1px 3px rgba(0,0,0,0.08); }
      div[data-testid="stMarkdownContainer"] p { margin:0; }
    </style>
    """, unsafe_allow_html=True)

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown("<h1 style='font-size:22px;font-weight:500;color:#1A1A18;margin-bottom:4px;"
                "letter-spacing:-0.2px;'>AI IPO Macro Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:12px;color:#888480;margin-bottom:16px;letter-spacing:0.02em;'>"
                "AI bubble · sovereign debt · India positioning &nbsp;·&nbsp; Updated Jun 12, 2026"
                "</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class='warn-box'>
    ⚠ SpaceX (SPCX) began trading today — June 12, 2026 — at $135/share · $1.75T valuation.
    The AI vacuum phase is no longer theoretical. USD/INR is already at 95.25 — stress rupture
    scenario is weeks away, not months.
    </div>""", unsafe_allow_html=True)

    # ── Fetch data ────────────────────────────────────────────────────────────
    with st.spinner("Loading market data..."):
        prices     = get_live_prices()
        gold_hist  = fetch_gold_historical()
        sp500_hist = fetch_sp500_historical()
        inr_hist   = fetch_inr_historical()
        crude_hist = fetch_crude_historical()
        scenarios  = build_scenario_projections()

    # ── Live price metrics ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Live market snapshot</div>", unsafe_allow_html=True)
    cols = st.columns(8)
    metric_defs = [
        ('Gold USD',    'gold_usd',    '$',  ''),
        ('USD / INR',   'usd_inr',     '₹',  ''),
        ('Gold / INR*', 'gold_inr',    '₹',  '/g'),
        ('Crude WTI',   'crude_wti',   '$',  '/bbl'),
        ('Crude Brent', 'crude_brent', '$',  '/bbl'),
        ('S&P 500',     'sp500',       '',   ''),
        ('Nifty 50',    'nifty',       '',   ''),
        ('US 10Y Yield','us10y',       '',   '%'),
    ]
    for col, (label, key, prefix, suffix) in zip(cols, metric_defs):
        with col:
            p   = prices[key]
            v   = p['value']
            pct = p.get('change_pct', 0.0)
            status = p.get('status', '')
            # Colour logic: green = up, red = down, amber = derived/error
            if status in ('derived', 'error') or pct == 0.0:
                val_cls = 'metric-warn'
            elif pct > 0:
                val_cls = 'metric-up'
            else:
                val_cls = 'metric-down'
            # Change badge colour
            if pct > 0:
                chg_style = "color:#2D6A0F;background:#EAF3DE;padding:1px 5px;border-radius:3px;"
            elif pct < 0:
                chg_style = "color:#7A1F1F;background:#FCEBEB;padding:1px 5px;border-radius:3px;"
            else:
                chg_style = "color:#666560;background:#F1EFE8;padding:1px 5px;border-radius:3px;"
            if v is None:
                fmt = '—'
            elif key == 'gold_inr':
                fmt = f"₹{v:,.0f}/g"
            elif key == 'us10y':
                fmt = f"{v:.2f}%"
            elif key == 'usd_inr':
                fmt = f"₹{v:.2f}"
            else:
                fmt = f"{prefix}{v:,.2f}{suffix}"
            chg_text = p['change'] or '—'
            src_text = p['source']
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>{label}</div>
              <div class='metric-value {val_cls}'>{fmt}</div>
              <div class='metric-sub'>
                <span style='{chg_style};font-size:10px;font-weight:500;'>{chg_text}</span>
                &nbsp;·&nbsp;{src_text}
              </div>
            </div>""", unsafe_allow_html=True)

    # ── IPO pipeline cards ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>IPO pipeline — confirmed &amp; expected</div>",
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, ipo in zip([c1, c2, c3], IPO_DATA):
        with col:
            rows_html = ''.join(
                f"<div class='ipo-row'><span>{r[0]}</span><span>{r[1]}</span></div>"
                for r in ipo['rows']
            )
            st.markdown(f"""
            <div class='ipo-card'>
              <div class='ipo-title'>
                <span class='badge {ipo["badge_class"]}'>{ipo["badge"]}</span>{ipo["name"]}
              </div>
              {rows_html}
            </div>""", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Charts — historical + scenario projections</div>",
                unsafe_allow_html=True)

    CHART_THEME = dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#FFFFFF',
        font=dict(family='Inter', color='#666560', size=11),
        xaxis=dict(gridcolor='#ECEAE4', linecolor='#D6D3CA', showgrid=True),
        yaxis=dict(gridcolor='#ECEAE4', linecolor='#D6D3CA', showgrid=True),
        legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#D6D3CA'),
        margin=dict(l=50, r=20, t=30, b=40),
        hovermode='x unified',
    )

    def add_event_markers(fig, row=None, col=None):
        kwargs = dict(row=row, col=col) if row is not None else {}
        for ev in PHASE_EVENTS:
            fig.add_vline(x=pd.to_datetime(ev['date']), line_width=1,
                          line_dash='dot', line_color=ev['color'],
                          opacity=0.5, **kwargs)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Gold & Crude", "💱 USD/INR", "📊 Equities",
        "🇮🇳 Gold in INR", "📋 Portfolio allocation",
    ])

    with tab1:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.6, 0.4],
                            subplot_titles=('Gold price (USD/oz)', 'Crude oil WTI (USD/bbl)'),
                            vertical_spacing=0.08)
        if gold_hist is not None:
            fig.add_trace(go.Scatter(x=gold_hist['Date'], y=gold_hist['Price'],
                name='Gold historical', line=dict(color='#fbbf24', width=1.5),
                fill='tozeroy', fillcolor='rgba(251,191,36,0.06)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=scenarios['Date'], y=scenarios['gold_base'],
            name='Base (slow grind)', line=dict(color='#6366f1', width=1.5, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=scenarios['Date'], y=scenarios['gold_rupture'],
            name='Rupture scenario', line=dict(color='#f87171', width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=scenarios['Date'], y=scenarios['gold_bull'],
            name='Bull scenario', line=dict(color='#4ade80', width=1.5, dash='dash')), row=1, col=1)
        fig.add_hline(y=4139, line_width=1, line_color='#fbbf24', line_dash='dash',
                      row=1, col=1, annotation_text=' Current: $4,139',
                      annotation_font_color='#fbbf24')
        fig.add_trace(go.Scatter(x=crude_hist['Date'], y=crude_hist['Price'],
            name='Crude WTI historical', line=dict(color='#fb923c', width=1.5),
            fill='tozeroy', fillcolor='rgba(251,146,60,0.06)'), row=2, col=1)
        fig.add_trace(go.Scatter(x=scenarios['Date'], y=scenarios['crude_rupture'],
            name='Crude rupture', line=dict(color='#f87171', width=2, dash='dash')), row=2, col=1)
        fig.add_hline(y=89.4, line_width=1, line_color='#fb923c', line_dash='dash',
                      row=2, col=1, annotation_text=' Current: ~$89',
                      annotation_font_color='#fb923c')
        add_event_markers(fig, row=1, col=1)
        add_event_markers(fig, row=2, col=1)
        fig.add_vrect(x0='2027-04-01', x1='2027-09-30',
                      fillcolor='rgba(248,113,113,0.07)',
                      annotation_text='Rupture window', annotation_position='top left',
                      annotation_font_color='#f87171', annotation_font_size=10, line_width=0)
        fig.update_layout(**CHART_THEME, height=550)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='note-box'>Gold historical: datasets/gold-prices (GitHub). "
                    "Crude oil: estimated from known anchor points. "
                    "Scenarios are analytical estimates, not forecasts.</div>",
                    unsafe_allow_html=True)

    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=inr_hist['Date'], y=inr_hist['Rate'],
            name='USD/INR historical', line=dict(color='#60a5fa', width=1.5),
            fill='tozeroy', fillcolor='rgba(96,165,250,0.06)'))
        fig2.add_trace(go.Scatter(x=scenarios['Date'], y=scenarios['inr_rupture'],
            name='Rupture scenario', line=dict(color='#f87171', width=2.5)))
        fig2.add_hline(y=95.25, line_color='#fbbf24', line_dash='dash', line_width=1,
                       annotation_text=' Today: 95.25', annotation_font_color='#fbbf24')
        fig2.add_hrect(y0=95, y1=105, fillcolor='rgba(248,113,113,0.08)', line_width=0,
                       annotation_text='Phase 1 rupture zone (95–105)',
                       annotation_font_color='#f87171', annotation_font_size=10,
                       annotation_position='top right')
        fig2.add_hrect(y0=70, y1=80, fillcolor='rgba(74,222,128,0.06)', line_width=0,
                       annotation_text='USD rebase zone (70–80)',
                       annotation_font_color='#27500A', annotation_font_size=10,
                       annotation_position='bottom right')
        add_event_markers(fig2)
        fig2.add_vrect(x0='2027-04-01', x1='2027-09-30',
                       fillcolor='rgba(248,113,113,0.07)',
                       annotation_text='Rupture window',
                       annotation_font_color='#f87171', annotation_font_size=10, line_width=0)
        fig2.update_layout(**CHART_THEME, height=420,
                           title=dict(text='USD/INR — historical + rupture scenario',
                                      font=dict(color='#1A1A18', size=13)))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("""
        <div class='section-header' style='margin-top:8px;'>INR action guide</div>
        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:8px;'>
          <div class='tl-cell cell-gold' style='min-height:60px;'><b>Hold INR when</b>
            <span>System functioning · RBI defending · Current a/c near balance</span></div>
          <div class='tl-cell cell-danger' style='min-height:60px;'><b>Hold Gold when</b>
            <span>System under stress · FII outflows · Yields breaking out · 2026–2027</span></div>
          <div class='tl-cell cell-silver' style='min-height:60px;'><b>The key rule</b>
            <span>USD weak + INR strong = gold DOWN in INR. Exit gold BEFORE full INR rebase.</span></div>
        </div>""", unsafe_allow_html=True)

    with tab3:
        fig3 = make_subplots(rows=1, cols=2,
                             subplot_titles=('S&P 500 scenario', 'Nifty 50 scenario'))
        if sp500_hist is not None:
            fig3.add_trace(go.Scatter(x=sp500_hist['Date'], y=sp500_hist['Price'],
                name='S&P historical', line=dict(color='#60a5fa', width=1.5)), row=1, col=1)
        fig3.add_trace(go.Scatter(x=scenarios['Date'], y=scenarios['sp_rupture'],
            name='S&P rupture', line=dict(color='#f87171', width=2.5)), row=1, col=1)

        nifty_anchors = {
            '2015-01': 8808,  '2016-01': 7600,  '2017-01': 8186,  '2018-01': 10906,
            '2019-01': 10831, '2020-03': 7511,  '2021-01': 14018, '2022-01': 17354,
            '2023-01': 18105, '2024-01': 21731, '2025-01': 23169, '2026-01': 24800,
            '2026-06': 24350,
        }
        nifty_df = pd.DataFrame([
            {'Date': pd.to_datetime(k), 'Price': v}
            for k, v in nifty_anchors.items()
        ]).sort_values('Date')
        nifty_interp = (nifty_df.set_index('Date')
                        .reindex(pd.date_range(nifty_df['Date'].min(),
                                               nifty_df['Date'].max(), freq='MS'))
                        .interpolate('linear').reset_index())
        nifty_interp.columns = ['Date', 'Price']

        fig3.add_trace(go.Scatter(x=nifty_interp['Date'], y=nifty_interp['Price'],
            name='Nifty historical', line=dict(color='#4ade80', width=1.5)), row=1, col=2)
        fig3.add_trace(go.Scatter(x=scenarios['Date'], y=scenarios['nifty_rupture'],
            name='Nifty rupture', line=dict(color='#fb923c', width=2.5)), row=1, col=2)
        add_event_markers(fig3, row=1, col=1)
        add_event_markers(fig3, row=1, col=2)
        fig3.add_vrect(x0='2027-04-01', x1='2027-09-30',
                       fillcolor='rgba(248,113,113,0.07)', line_width=0, row=1, col=1)
        fig3.add_vrect(x0='2027-04-01', x1='2027-09-30',
                       fillcolor='rgba(248,113,113,0.07)', line_width=0, row=1, col=2)
        fig3.update_layout(**CHART_THEME, height=420)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("""<div class='note-box'>
        401(k) destruction: Top 10 S&P stocks = ~40% of index weight. All AI bets.
        A 50% AI complex drawdown = ~20–25% 401(k) loss before panic selling amplifies it.
        Political pressure to bail out will be the inflationary policy response that validates gold.
        </div>""", unsafe_allow_html=True)

    with tab4:
        fig4 = go.Figure()
        if gold_hist is not None and inr_hist is not None:
            gold_m   = gold_hist.set_index('Date').resample('MS').last()
            inr_m    = inr_hist.set_index('Date').resample('MS').last()
            combined = gold_m.join(inr_m, how='inner', lsuffix='_gold', rsuffix='_inr')
            combined.columns = ['gold_usd', 'usd_inr']
            combined['gold_inr_gram'] = combined['gold_usd'] * combined['usd_inr'] / 31.1035
            combined = combined.reset_index()
            fig4.add_trace(go.Scatter(x=combined['Date'], y=combined['gold_inr_gram'],
                name='Gold in INR (historical, ₹/gram)',
                line=dict(color='#fbbf24', width=2),
                fill='tozeroy', fillcolor='rgba(251,191,36,0.06)'))
        fig4.add_trace(go.Scatter(x=scenarios['Date'], y=scenarios['gold_inr_rupture'],
            name='Gold INR rupture scenario (₹/gram)',
            line=dict(color='#f87171', width=2.5)))
        current_gold_inr = 4139 * 95.25 / 31.1035
        fig4.add_hline(y=current_gold_inr, line_color='#fbbf24', line_dash='dash',
                       annotation_text=f' Today: ₹{current_gold_inr:,.0f}/g',
                       annotation_font_color='#fbbf24')
        fig4.add_vrect(x0='2027-07-01', x1='2027-12-31',
                       fillcolor='rgba(74,222,128,0.07)',
                       annotation_text='Gold INR EXIT WINDOW',
                       annotation_font_color='#27500A', annotation_font_size=11, line_width=0)
        fig4.add_vrect(x0='2027-04-01', x1='2027-07-01',
                       fillcolor='rgba(248,113,113,0.07)',
                       annotation_text="Hold — don't panic",
                       annotation_font_color='#f87171', annotation_font_size=10, line_width=0)
        add_event_markers(fig4)
        fig4.update_layout(**CHART_THEME, height=420,
                           title=dict(text='Gold price in INR (₹/gram) — the Indian investor view',
                                      font=dict(color='#1A1A18', size=13)),
                           yaxis_title='₹ per gram')
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("""
        <div class='section-header'>Gold INR phase math</div>
        <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px;'>
          <div class='tl-cell cell-gold'><b>Phase 1 (Apr–Jun 27)</b>
            <span>Gold USD ↓13% + INR ↓16% = Gold INR flat. Natural hedge. HOLD.</span></div>
          <div class='tl-cell cell-gold'><b>Phase 2 (Jul–Sep 27)</b>
            <span>Gold USD ↑92% + INR recovers to 88 = Gold INR ↑73%. HOLD ALL.</span></div>
          <div class='tl-cell cell-danger'><b>EXIT window (H2 27)</b>
            <span>Gold USD plateaus + INR appreciating = gold INR FALLS. Exit before INR hits 78.</span></div>
          <div class='tl-cell cell-silver'><b>Silver after gold</b>
            <span>Ratio 88→45. Silver peaks AFTER gold. Solar India demand adds fuel.</span></div>
        </div>""", unsafe_allow_html=True)

    with tab5:
        phases = list(PORTFOLIO_PHASES.keys())
        assets = ['Gold/SGBs', 'INR Cash', 'India Equity', 'Silver']
        colors = ['#fbbf24', '#60a5fa', '#4ade80', '#5eead4']
        fig5   = go.Figure()
        for asset, color in zip(assets, colors):
            vals = [PORTFOLIO_PHASES[p][asset] for p in phases]
            fig5.add_trace(go.Bar(
                name=asset, x=phases, y=vals,
                marker_color=color, marker_opacity=0.85,
                text=[f'{v}%' for v in vals], textposition='inside',
                textfont=dict(color='#0e1117', size=11, family='Inter'),
            ))
        fig5.update_layout(
            **CHART_THEME, height=380, barmode='stack',
            title=dict(text='Recommended portfolio allocation by phase (India investor)',
                       font=dict(color='#1A1A18', size=13)),
            yaxis_title='%', yaxis_range=[0, 105],
            legend_orientation='h', legend_y=1.12, legend_x=0,
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("""<div class='note-box'>
        <b style='color:#854F0B;'>SGB advantage:</b> Sovereign Gold Bonds offer 2.5% annual
        interest + gold price appreciation + tax-free maturity at 8 years.
        Best vehicle for the gold leg of this thesis. Avoid gold ETFs as primary vehicle —
        counterparty risk defeats the purpose in a rupture scenario.
        </div>""", unsafe_allow_html=True)

    # ── News cascade banner ───────────────────────────────────────────────────
    if NEWS_OVERLAY:
        _summary   = _raw_news.get('overall_summary', {})
        _score     = _summary.get('cascade_health_index', None)
        _flags     = _summary.get('immediate_red_flags', '')
        _validated = sum(1 for v in NEWS_OVERLAY.values() if 'Validated' in v.get('status', ''))
        _partial   = sum(1 for v in NEWS_OVERLAY.values() if 'Partial'   in v.get('status', ''))
        _total     = len(NEWS_OVERLAY)
        _bar_w     = f"{_score}%" if _score else "0%"
        _bar_color = ("#27500A" if _score and _score >= 75
                      else "#854F0B" if _score and _score >= 50 else "#791F1F")
        st.markdown(f"""
        <div style='background:#FFFFFF;border:0.5px solid #D6D3CA;
             border-left:3px solid {_bar_color};border-radius:8px;
             padding:10px 14px;margin-bottom:10px;'>
          <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;'>
            <span style='font-size:11px;font-weight:600;color:#1A1A18;'>
              📰 Live news overlay active — {_total} cells annotated</span>
            <span style='font-size:11px;font-weight:600;color:{_bar_color};'>
              Cascade health: {_score}/100</span>
          </div>
          <div style='background:#F1EFE8;border-radius:4px;height:6px;margin-bottom:8px;'>
            <div style='background:{_bar_color};width:{_bar_w};height:6px;border-radius:4px;'></div>
          </div>
          <div style='display:flex;gap:12px;margin-bottom:6px;'>
            <span style='font-size:10px;background:#EAF3DE;color:#27500A;
                  padding:2px 7px;border-radius:4px;'>✅ Validated: {_validated}</span>
            <span style='font-size:10px;background:#FFF8E8;color:#854F0B;
                  padding:2px 7px;border-radius:4px;'>⚠️ Partial: {_partial}</span>
            <span style='font-size:10px;background:#F1EFE8;color:#666560;
                  padding:2px 7px;border-radius:4px;'>⏳ Too early: {_total-_validated-_partial}</span>
          </div>
          <div style='font-size:10px;color:#666560;line-height:1.5;'>
            <b style='color:#791F1F;'>Red flags:</b> {_flags}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("📂 Place **news.json** in the same folder to activate live news overlays.")

    # ── Month-by-month timeline ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>Month-by-month timeline</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style='margin-bottom:12px;'>
      <span class='phase-pill ph-vacuum'><span class='phase-dot' style='background:#6366f1'></span>
        Phase 1: AI vacuum (now → Sep 2026)</span>
      <span class='phase-pill ph-stress'><span class='phase-dot' style='background:#f59e0b'></span>
        Phase 2: Stress fractures (Oct 2026 → Mar 2027)</span>
      <span class='phase-pill ph-rupture'><span class='phase-dot' style='background:#ef4444'></span>
        Phase 3: Rupture (Apr → Sep 2027)</span>
      <span class='phase-pill ph-reset'><span class='phase-dot' style='background:#22c55e'></span>
        Phase 4: Reset / deploy (H2 2027 → 2028)</span>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='display:grid;grid-template-columns:95px 1fr 1fr 1fr 1fr 1fr;
                gap:5px;margin-bottom:4px;'>
      <span style='font-size:10px;color:#4b5563;'></span>
      <span style='font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:0.06em;'>IPO events</span>
      <span style='font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:0.06em;'>Macro / bonds</span>
      <span style='font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:0.06em;'>401(k) impact</span>
      <span style='font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:0.06em;'>Gold / INR</span>
      <span style='font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:0.06em;'>India action</span>
    </div>""", unsafe_allow_html=True)

    for row_idx, row in enumerate(TIMELINE_ROWS):
        cells_html = ''
        for col_idx, (cls, title, desc) in enumerate(row['cells']):
            overlay     = news_overlay_html(row_idx, col_idx)
            cells_html += (f"<div class='tl-cell {cls}'>"
                           f"<b>{title}</b><span>{desc}</span>{overlay}</div>")
        period_display = row['period'].replace('\n', '<br>')
        st.markdown(f"""
        <div class='timeline-row'>
          <div class='tl-date'>{period_display}</div>
          {cells_html}
        </div>""", unsafe_allow_html=True)

    # ── 401k destruction ──────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>401(k) destruction mechanism — why it amplifies everything</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:12px;'>
      <div class='tl-cell cell-401k' style='min-height:72px;'>
        <b>Concentration risk</b>
        <span>Top 10 S&P stocks = ~40% of index. All AI bets. Fidelity avg $141K already ↓4% in Q1 2026.</span>
      </div>
      <div class='tl-cell cell-401k' style='min-height:72px;'>
        <b>Forced selling amplifier</b>
        <span>Boomers near retirement cannot wait for recovery. Each withdrawal accelerates the crash.</span>
      </div>
      <div class='tl-cell cell-401k' style='min-height:72px;'>
        <b>Policy response = inflationary</b>
        <span>Political pressure for 401(k) bailout = money printing. 65M boomers = electoral force.</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Crude section ─────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Crude oil — the physical world variable</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:12px;'>
      <div class='tl-cell cell-macro' style='min-height:72px;'>
        <b>Iran / Hormuz factor</b>
        <span>Strait still effectively closed. ~20% of global oil supply.</span>
      </div>
      <div class='tl-cell cell-macro' style='min-height:72px;'>
        <b>Front-end rate killer</b>
        <span>High oil = inflation = Fed can't cut. Oil up = front-end up = deficit explodes 6→10%.</span>
      </div>
      <div class='tl-cell cell-danger' style='min-height:72px;'>
        <b>Tank bottom signal</b>
        <span>Groman: "sometime between now and Labor Day, people start hitting tank bottoms."</span>
      </div>
      <div class='tl-cell cell-gold' style='min-height:72px;'>
        <b>India inflation channel</b>
        <span>India imports ~85% of oil. High crude = imported inflation = INR weakness = gold INR higher.</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class='note-box' style='margin-top:20px;'>
    <b>Data sources:</b> Gold historical: datasets/gold-prices (GitHub/LBMA). S&P 500: vega-datasets.
    Crude oil, INR, equities: estimated from known anchor points. IPO data: Reuters, Investing.com (Jun 2026).
    401(k) data: Fidelity Q1 2026. All projections are analytical scenario estimates, not investment advice.
    </div>""", unsafe_allow_html=True)

    st.markdown(
        f"<p style='font-size:10px;color:#374151;margin-top:8px;'>"
        f"Last refreshed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')} "
        f"· Data auto-refreshes every 5 minutes</p>",
        unsafe_allow_html=True,
    )
