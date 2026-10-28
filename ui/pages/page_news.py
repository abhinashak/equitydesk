"""
ui/pages/page_news.py  –  Market dashboard
──────────────────────────────────────────
  • GIFT Nifty banner
  • Controls  : period selector, % / absolute toggle, auto-refresh, alert threshold
  • Overview  : core 8 + DXY, VIX, India VIX, global indices, rates, 2s10s, real yield,
                BTC, FII/DII flows. Each card = sparkline + 1D/7D/1M/3M/1Y/YTD + 52W range
  • Metals    : precious, base, iron & steel, zinc/tin/tungsten proxies, India metals,
                miners/ETFs, custom tickers; ranking, rebased chart, ratios, correlation
  • Food grains: wheat, corn, soy, rice, oils, softs, agri ETFs, India food proxies, fertilizer,
                rupee-adjusted prices, spreads, drivers
  • Heatmap   : every instrument × every period
  • Analysis  : correlation matrix, Gold INR decomposition, Oil→INR→Nifty chain,
                real yield / breakeven, 2s10s
  • Drill-down: full chart + 50DMA + 52W band + latest news for any instrument
Entry point: render()
"""

import datetime
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from ui.pages.page_us_economy import render_gift_nifty_banner


# ═════════════════════════════════════════════════════════════════════════════
# Registry
# ═════════════════════════════════════════════════════════════════════════════

PERIODS = ['1D', '7D', '1M', '3M', '1Y', 'YTD']
PERIOD_DAYS = {'7D': 7, '1M': 30, '3M': 90, '1Y': 365}

REG = {}          # key -> meta
GROUP_ORDER = []  # display order of groups


def _m(key, label, ticker=None, group='Core', prefix='', suffix='', dec=2,
       kind='px', src='Yahoo', proxy=False, vol=False, fred=None, alts=()):
    """kind: 'px' price | 'yield' (level in %, moves in bps) | 'spread' (in %, moves in bps)"""
    REG[key] = dict(key=key, label=label, ticker=ticker, group=group, prefix=prefix,
                    suffix=suffix, dec=dec, kind=kind, src=src, proxy=proxy, vol=vol,
                    fred=fred, alts=alts)
    if group not in GROUP_ORDER:
        GROUP_ORDER.append(group)


# ── Macro / Overview ─────────────────────────────────────────────────────────
_m('gold_usd',    'Gold USD',     'GC=F',     'Core', '$', '', 2)
_m('usd_inr',     'USD / INR',    'USDINR=X', 'Core', '₹', '', 2)
_m('gold_inr',    'Gold / INR*',  None,       'Core', '₹', '/g', 0, src='Derived')
_m('crude_wti',   'Crude WTI',    'CL=F',     'Core', '$', '/bbl', 2)
_m('crude_brent', 'Crude Brent',  'BZ=F',     'Core', '$', '/bbl', 2)
_m('sp500',       'S&P 500',      '^GSPC',    'Core')
_m('nifty',       'Nifty 50',     '^NSEI',    'Core')
_m('us10y',       'US 10Y Yield', '^TNX',     'Core', kind='yield')

_m('dxy',         'Dollar Index (DXY)', 'DX-Y.NYB', 'Dollar & volatility')
_m('india_vix',   'India VIX',    '^INDIAVIX', 'Dollar & volatility', vol=True)
_m('vix',         'CBOE VIX',     '^VIX',      'Dollar & volatility', vol=True)

_m('banknifty',   'Bank Nifty',   '^NSEBANK', 'Global indices')
_m('sensex',      'Sensex',       '^BSESN',   'Global indices')
_m('nasdaq',      'Nasdaq Comp.', '^IXIC',    'Global indices')
_m('nikkei',      'Nikkei 225',   '^N225',    'Global indices')
_m('hangseng',    'Hang Seng',    '^HSI',     'Global indices')

_m('us2y',        'US 2Y Yield',  None, 'Rates & crypto', kind='yield', src='FRED · DGS2', fred='DGS2')
_m('us_2s10s',    'US 2s10s Spread', None, 'Rates & crypto', kind='spread', src='FRED · T10Y2Y', fred='T10Y2Y')
_m('real_yield',  'US 10Y Real Yield', None, 'Rates & crypto', kind='yield', src='FRED · DFII10', fred='DFII10')
_m('india10y',    'India 10Y G-Sec', '^IN10YT=RR', 'Rates & crypto', kind='yield')
_m('btc',         'Bitcoin',      'BTC-USD',  'Rates & crypto', '$', '', 0)

# ── Metals ───────────────────────────────────────────────────────────────────
_m('silver',      'Silver',       'SI=F',  'Precious (futures)', '$', '/oz', 2)
_m('m_platinum',  'Platinum',     'PL=F',  'Precious (futures)', '$', '/oz', 2)
_m('m_palladium', 'Palladium',    'PA=F',  'Precious (futures)', '$', '/oz', 2)
REG['gold_usd']['mgroup'] = 'Precious (futures)'

_m('m_copper',    'Copper (COMEX)',   'HG=F',  'Base metals (futures)', '$', '/lb', 3)
_m('m_aluminum',  'Aluminum (COMEX)', 'ALI=F', 'Base metals (futures)', '$', '/t', 0)
_m('m_dbb',       'Base Metals Fund (DBB)', 'DBB', 'Base metals (futures)', '$', '', 2, proxy=True)

_m('m_slx',       'Steel ETF (SLX)',  'SLX',   'Iron & steel (proxies)', '$', '', 2, proxy=True)
_m('m_vale',      'Vale (iron ore)',  'VALE',  'Iron & steel (proxies)', '$', '', 2, proxy=True)
_m('m_rio',       'Rio Tinto (iron ore)', 'RIO', 'Iron & steel (proxies)', '$', '', 2, proxy=True)
_m('m_nmdc',      'NMDC (iron ore)',  'NMDC.NS', 'Iron & steel (proxies)', '₹', '', 2, proxy=True)
_m('m_tatasteel', 'Tata Steel',       'TATASTEEL.NS', 'Iron & steel (proxies)', '₹', '', 2, proxy=True)
_m('m_jswsteel',  'JSW Steel',        'JSWSTEEL.NS', 'Iron & steel (proxies)', '₹', '', 2, proxy=True)
_m('m_sail',      'SAIL',             'SAIL.NS', 'Iron & steel (proxies)', '₹', '', 2, proxy=True)

_m('m_hindzinc',  'Zinc · Hindustan Zinc', 'HINDZINC.NS', 'Zinc / Tin / Tungsten (proxies)', '₹', '', 2, proxy=True)
_m('m_teck',      'Zinc · Teck Resources', 'TECK', 'Zinc / Tin / Tungsten (proxies)', '$', '', 2, proxy=True)
_m('m_tin',       'Tin · Alphamin',   'AFM.V', 'Zinc / Tin / Tungsten (proxies)', 'C$', '', 2, proxy=True)
_m('m_tungsten',  'Tungsten · Almonty', 'ALM', 'Zinc / Tin / Tungsten (proxies)', '$', '', 2, proxy=True,
   alts=('AII.TO',))

_m('m_cnxmetal',  'Nifty Metal Index', '^CNXMETAL', 'India metals & mining')
_m('m_hindalco',  'Hindalco',         'HINDALCO.NS', 'India metals & mining', '₹', '', 2, proxy=True)
_m('m_vedl',      'Vedanta',          'VEDL.NS', 'India metals & mining', '₹', '', 2, proxy=True)
_m('m_hindcopper','Hindustan Copper', 'HINDCOPPER.NS', 'India metals & mining', '₹', '', 2, proxy=True)
_m('m_nalco',     'NALCO',            'NATIONALUM.NS', 'India metals & mining', '₹', '', 2, proxy=True)

_m('m_copx',      'Copper Miners (COPX)', 'COPX', 'Global miners & ETFs', '$', '', 2, proxy=True)
_m('m_fcx',       'Freeport-McMoRan', 'FCX',  'Global miners & ETFs', '$', '', 2, proxy=True)
_m('m_gdx',       'Gold Miners (GDX)', 'GDX', 'Global miners & ETFs', '$', '', 2, proxy=True)
_m('m_sil',       'Silver Miners (SIL)', 'SIL', 'Global miners & ETFs', '$', '', 2, proxy=True)
_m('m_xme',       'Metals & Mining (XME)', 'XME', 'Global miners & ETFs', '$', '', 2, proxy=True)
_m('m_remx',      'Rare Earth / Strategic (REMX)', 'REMX', 'Global miners & ETFs', '$', '', 2, proxy=True)
_m('m_lit',       'Lithium (LIT)',    'LIT',  'Global miners & ETFs', '$', '', 2, proxy=True)
_m('m_ura',       'Uranium (URA)',    'URA',  'Global miners & ETFs', '$', '', 2, proxy=True)

# ── Food grains & agri ───────────────────────────────────────────────────────
_m('f_wheat',   'Wheat (CBOT)',        'ZW=F', 'Grains & oilseeds (futures)', '', '¢/bu', 2)
_m('f_kcwheat', 'Wheat HRW (KC)',      'KE=F', 'Grains & oilseeds (futures)', '', '¢/bu', 2)
_m('f_corn',    'Corn',                'ZC=F', 'Grains & oilseeds (futures)', '', '¢/bu', 2)
_m('f_soy',     'Soybeans',            'ZS=F', 'Grains & oilseeds (futures)', '', '¢/bu', 2)
_m('f_rice',    'Rough Rice',          'ZR=F', 'Grains & oilseeds (futures)', '$', '/cwt', 2)
_m('f_oats',    'Oats',                'ZO=F', 'Grains & oilseeds (futures)', '', '¢/bu', 2)

_m('f_soyoil',  'Soybean Oil',         'ZL=F', 'Veg oils & meal', '', '¢/lb', 2)
_m('f_soymeal', 'Soybean Meal',        'ZM=F', 'Veg oils & meal', '$', '/ton', 2)

_m('f_sugar',   'Sugar #11',           'SB=F', 'Softs', '', '¢/lb', 2)
_m('f_coffee',  'Coffee',              'KC=F', 'Softs', '', '¢/lb', 2)
_m('f_cotton',  'Cotton',              'CT=F', 'Softs', '', '¢/lb', 2)
_m('f_cocoa',   'Cocoa',               'CC=F', 'Softs', '$', '/t', 0)
_m('f_oj',      'Orange Juice',        'OJ=F', 'Softs', '', '¢/lb', 2)

_m('f_dba',     'Agriculture Fund (DBA)', 'DBA', 'Agri ETFs & global', '$', '', 2, proxy=True)
_m('f_weat',    'Wheat ETF (WEAT)',    'WEAT', 'Agri ETFs & global', '$', '', 2, proxy=True)
_m('f_cornetf', 'Corn ETF (CORN)',     'CORN', 'Agri ETFs & global', '$', '', 2, proxy=True)
_m('f_soyb',    'Soybean ETF (SOYB)',  'SOYB', 'Agri ETFs & global', '$', '', 2, proxy=True)
_m('f_moo',     'Agribusiness (MOO)',  'MOO',  'Agri ETFs & global', '$', '', 2, proxy=True)
_m('f_adm',     'Archer-Daniels-Midland', 'ADM', 'Agri ETFs & global', '$', '', 2, proxy=True)
_m('f_bg',      'Bunge',               'BG',   'Agri ETFs & global', '$', '', 2, proxy=True)
_m('f_de',      'Deere & Co',          'DE',   'Agri ETFs & global', '$', '', 2, proxy=True)

_m('f_krbl',    'KRBL (basmati rice)', 'KRBL.NS', 'India agri & food (proxies)', '₹', '', 2, proxy=True)
_m('f_awl',     'AWL Agri (edible oil)', 'AWL.NS', 'India agri & food (proxies)', '₹', '', 2, proxy=True)
_m('f_itc',     'ITC (atta, foods)',   'ITC.NS', 'India agri & food (proxies)', '₹', '', 2, proxy=True)
_m('f_tataco',  'Tata Consumer',       'TATACONSUM.NS', 'India agri & food (proxies)', '₹', '', 2, proxy=True)
_m('f_balram',  'Balrampur Chini (sugar)', 'BALRAMCHIN.NS', 'India agri & food (proxies)', '₹', '', 2, proxy=True)
_m('f_fmcg',    'Nifty FMCG',          '^CNXFMCG', 'India agri & food (proxies)', proxy=True)

_m('f_chambal', 'Chambal Fertilisers', 'CHAMBLFERT.NS', 'Fertilizer & inputs', '₹', '', 2, proxy=True)
_m('f_coro',    'Coromandel Intl',     'COROMANDEL.NS', 'Fertilizer & inputs', '₹', '', 2, proxy=True)
_m('f_upl',     'UPL (agrochem)',      'UPL.NS', 'Fertilizer & inputs', '₹', '', 2, proxy=True)
_m('f_ntr',     'Nutrien',             'NTR',  'Fertilizer & inputs', '$', '', 2, proxy=True)
_m('f_mos',     'Mosaic',              'MOS',  'Fertilizer & inputs', '$', '', 2, proxy=True)
_m('f_cf',      'CF Industries (urea/N)', 'CF', 'Fertilizer & inputs', '$', '', 2, proxy=True)

# Derived ratios (computed in load_data)
_m('gold_silver', 'Gold / Silver Ratio', None, 'Ratios', '', '', 1, src='Derived')
_m('copper_gold', 'Copper / Gold (×1000)', None, 'Ratios', '', '', 3, src='Derived')
_m('wheat_corn',  'Wheat / Corn Ratio', None, 'Ratios', '', '', 2, src='Derived')
_m('soy_crush',   'Soy Crush Margin', None, 'Ratios', '$', '/bu', 2, src='Derived')

OVERVIEW_SECTIONS = [
    ('Core markets',          ['gold_usd', 'usd_inr', 'gold_inr', 'crude_wti',
                               'crude_brent', 'sp500', 'nifty', 'us10y']),
    ('Dollar & volatility',   ['dxy', 'india_vix', 'vix', 'silver']),
    ('Global indices',        ['banknifty', 'sensex', 'nasdaq', 'nikkei', 'hangseng']),
    ('Rates & crypto',        ['us2y', 'us_2s10s', 'real_yield', 'india10y', 'btc']),
]

METAL_GROUPS = {
    'Precious (futures)':      ['gold_usd', 'silver', 'm_platinum', 'm_palladium'],
    'Base metals (futures)':   ['m_copper', 'm_aluminum', 'm_dbb'],
    'Iron & steel (proxies)':  ['m_slx', 'm_vale', 'm_rio', 'm_nmdc', 'm_tatasteel', 'm_jswsteel', 'm_sail'],
    'Zinc / Tin / Tungsten (proxies)': ['m_hindzinc', 'm_teck', 'm_tin', 'm_tungsten'],
    'India metals & mining':   ['m_cnxmetal', 'm_hindalco', 'm_vedl', 'm_hindcopper', 'm_nalco'],
    'Global miners & ETFs':    ['m_copx', 'm_fcx', 'm_gdx', 'm_sil', 'm_xme', 'm_remx', 'm_lit', 'm_ura'],
}
ALL_METAL_KEYS = [k for ks in METAL_GROUPS.values() for k in ks]

FOOD_GROUPS = {
    'Grains & oilseeds (futures)': ['f_wheat', 'f_kcwheat', 'f_corn', 'f_soy', 'f_rice', 'f_oats'],
    'Veg oils & meal':             ['f_soyoil', 'f_soymeal'],
    'Softs':                       ['f_sugar', 'f_coffee', 'f_cotton', 'f_cocoa', 'f_oj'],
    'Agri ETFs & global':          ['f_dba', 'f_weat', 'f_cornetf', 'f_soyb', 'f_moo', 'f_adm', 'f_bg', 'f_de'],
    'India agri & food (proxies)': ['f_krbl', 'f_awl', 'f_itc', 'f_tataco', 'f_balram', 'f_fmcg'],
    'Fertilizer & inputs':         ['f_chambal', 'f_coro', 'f_upl', 'f_ntr', 'f_mos', 'f_cf'],
}
ALL_FOOD_KEYS = [k for ks in FOOD_GROUPS.values() for k in ks]
FOOD_FUTURES = [k for g in list(FOOD_GROUPS)[:3] for k in FOOD_GROUPS[g]]

FOOD_MAP = pd.DataFrame([
    ('Wheat',    'Direct futures (CBOT, KC)', 'ZW=F, KE=F · quoted in US cents per bushel'),
    ('Corn, Soybeans, Oats', 'Direct futures (CBOT)', 'ZC=F, ZS=F, ZO=F · cents per bushel'),
    ('Rice',     'Direct futures (US rough rice) + India proxy', 'ZR=F ($/cwt), KRBL.NS. Not Asian export rice (Thai/Viet/India)'),
    ('Edible oils', 'Soybean oil futures + proxy', 'ZL=F, AWL.NS. No free palm-oil feed'),
    ('Sugar, Coffee, Cotton, Cocoa', 'Direct futures (ICE)', 'SB=F, KC=F, CT=F, CC=F'),
    ('Pulses, Indian wheat/rice mandi prices, MSP', 'No free feed', 'Needs Agmarknet / e-NAM via a data.gov.in key'),
    ('Fertilizer', 'Proxy: producers', 'NTR, MOS, CF, CHAMBLFERT.NS, COROMANDEL.NS'),
], columns=['Commodity', 'How it is tracked here', 'Tickers / note'])

METAL_MAP = pd.DataFrame([
    ('Gold, Silver, Platinum, Palladium', 'Direct futures (COMEX / NYMEX)', 'GC=F, SI=F, PL=F, PA=F'),
    ('Copper',     'Direct futures (COMEX)',  'HG=F'),
    ('Aluminum',   'Direct futures (COMEX)',  'ALI=F'),
    ('Iron ore',   'Proxy: producers + steel ETF', 'VALE, RIO, NMDC.NS, SLX'),
    ('Steel',      'Proxy: steel makers',     'TATASTEEL.NS, JSWSTEEL.NS, SAIL.NS, SLX'),
    ('Zinc',       'Proxy: zinc producers',   'HINDZINC.NS, TECK'),
    ('Tin',        'Proxy: tin producer',     'AFM.V (Alphamin)'),
    ('Tungsten',   'Proxy: tungsten producer', 'ALM (Almonty)'),
    ('Nickel, Lead', 'No free feed', 'Use DBB as a rough read, or add a paid LME key'),
    ('Rare earths, Lithium, Uranium', 'Proxy: thematic ETFs', 'REMX, LIT, URA'),
], columns=['Metal', 'How it is tracked here', 'Tickers'])


# ═════════════════════════════════════════════════════════════════════════════
# Data
# ═════════════════════════════════════════════════════════════════════════════

def _extract_close(df, t):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        if t in df.columns.get_level_values(0):
            return df[t]['Close']
        try:
            return df['Close'][t]
        except KeyError:
            return None
    return df['Close'] if 'Close' in df.columns else None


def _clean(s):
    s = s.dropna().copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s[~s.index.duplicated(keep='last')].sort_index()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(tickers: tuple) -> dict:
    """One batched Yahoo call for ~2y of daily closes, per-ticker fallback if empty."""
    out = {}
    try:
        df = yf.download(list(tickers), period='2y', interval='1d', auto_adjust=False,
                         group_by='ticker', progress=False, threads=True)
    except Exception:
        df = None
    for t in tickers:
        s = None
        try:
            s = _extract_close(df, t)
            if s is not None:
                s = s.dropna()
        except Exception:
            s = None
        if s is None or len(s) < 2:
            try:
                s = yf.Ticker(t).history(period='2y', interval='1d', auto_adjust=False)['Close'].dropna()
            except Exception:
                s = None
        if s is not None and len(s) >= 2:
            out[t] = _clean(s)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred(series_id: str):
    try:
        r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}", timeout=8)
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ['Date', 'V']
        df['Date'] = pd.to_datetime(df['Date'])
        df['V'] = pd.to_numeric(df['V'], errors='coerce')
        s = df.dropna().set_index('Date')['V']
        s = s[s.index >= pd.Timestamp.today() - pd.Timedelta(days=800)]
        return s if len(s) > 2 else None
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fii_dii():
    """Best-effort NSE FII/DII feed (NSE blocks bots often → None)."""
    try:
        h = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
                           '(KHTML, like Gecko) Chrome/124 Safari/537.36',
             'Accept': 'application/json', 'Referer': 'https://www.nseindia.com/reports/fii-dii'}
        s = requests.Session()
        s.get('https://www.nseindia.com', headers=h, timeout=8)
        rows = s.get('https://www.nseindia.com/api/fiidiiTradeReact', headers=h, timeout=8).json()
        out = {}
        for r in rows:
            cat = 'FII' if 'FII' in r.get('category', '') else 'DII'
            out[cat] = {'net': float(str(r.get('netValue', '0')).replace(',', '')),
                        'buy': float(str(r.get('buyValue', '0')).replace(',', '')),
                        'sell': float(str(r.get('sellValue', '0')).replace(',', '')),
                        'date': r.get('date', '')}
        return out if out else None
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def fetch_news(ticker: str):
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        return []
    out = []
    for it in items[:8]:
        c = it.get('content', it)
        title = c.get('title')
        url = ((c.get('canonicalUrl') or {}).get('url')
               or (c.get('clickThroughUrl') or {}).get('url') or it.get('link'))
        pub = (c.get('provider') or {}).get('displayName') or it.get('publisher') or ''
        if title and url:
            out.append((title, url, pub))
    return out


def load_data(extra_tickers=()):
    """Returns S: key -> daily close Series (tz-naive)."""
    tick = {k: m['ticker'] for k, m in REG.items() if m['ticker']}
    px = fetch_prices(tuple(sorted(set(tick.values()))))
    S = {}
    for k, t in tick.items():
        s = px.get(t)
        if s is None:
            for alt in REG[k]['alts']:
                s = fetch_prices((alt,)).get(alt)
                if s is not None:
                    break
        if s is not None:
            S[k] = s
    for k, m in REG.items():
        if m['fred']:
            s = fetch_fred(m['fred'])
            if s is not None:
                S[k] = s
    # derived
    def combine(a, b, f):
        df = pd.concat([a, b], axis=1, keys=['a', 'b']).ffill().dropna()
        return f(df['a'], df['b'])
    if 'gold_usd' in S and 'usd_inr' in S:
        S['gold_inr'] = combine(S['gold_usd'], S['usd_inr'], lambda g, i: g * i / 31.1035)
    if 'gold_usd' in S and 'silver' in S:
        S['gold_silver'] = combine(S['gold_usd'], S['silver'], lambda g, s: g / s)
    if 'gold_usd' in S and 'm_copper' in S:
        S['copper_gold'] = combine(S['m_copper'], S['gold_usd'], lambda c, g: c / g * 1000)
    if 'f_wheat' in S and 'f_corn' in S:
        S['wheat_corn'] = combine(S['f_wheat'], S['f_corn'], lambda w, c: w / c)
    if all(k in S for k in ('f_soy', 'f_soymeal', 'f_soyoil')):
        df = pd.concat([S['f_soy'], S['f_soymeal'], S['f_soyoil']], axis=1,
                       keys=['b', 'm', 'o']).ffill().dropna()
        S['soy_crush'] = df['m'] * 0.022 + df['o'] * 0.11 - df['b'] / 100
    # custom tickers
    for t in extra_tickers:
        got = fetch_prices((t,)).get(t)
        key = f'c_{t}'
        REG[key] = dict(key=key, label=t, ticker=t, group='Custom', prefix='', suffix='', dec=2,
                        kind='px', src='Yahoo', proxy=True, vol=False, fred=None, alts=())
        if got is not None:
            S[key] = got
    return S


# ═════════════════════════════════════════════════════════════════════════════
# Stats / formatting
# ═════════════════════════════════════════════════════════════════════════════

def stats(s: pd.Series) -> dict:
    s = s.dropna()
    cur, prev, last = float(s.iloc[-1]), float(s.iloc[-2]), s.index[-1]
    bases = {'1D': prev}
    for lab, d in PERIOD_DAYS.items():
        o = s[s.index <= last - pd.Timedelta(days=d)]
        bases[lab] = float(o.iloc[-1]) if len(o) else None
    y = s[s.index < pd.Timestamp(last.year, 1, 1)]
    bases['YTD'] = float(y.iloc[-1]) if len(y) else None
    pct, ab, bps = {}, {}, {}
    for p, b in bases.items():
        if b is None or b == 0 or np.isnan(b):
            pct[p] = ab[p] = bps[p] = None
        else:
            pct[p], ab[p], bps[p] = (cur - b) / abs(b) * 100, cur - b, (cur - b) * 100
    w = s[s.index >= last - pd.Timedelta(days=365)]
    hi, lo = float(w.max()), float(w.min())
    return dict(cur=cur, last=last, pct=pct, abs=ab, bps=bps, hi=hi, lo=lo,
                pos=(cur - lo) / (hi - lo) if hi > lo else 0.5)


def get_chg(stt, meta, p, mode):
    if meta['kind'] in ('yield', 'spread'):
        return stt['bps'][p]
    return stt['pct'][p] if mode == '%' else stt['abs'][p]


def fmt_value(meta, v):
    if v is None:
        return '—'
    if meta['kind'] == 'yield':
        return f"{v:.2f}%"
    if meta['kind'] == 'spread':
        return f"{v * 100:+.0f} bps"
    return f"{meta['prefix']}{v:,.{meta['dec']}f}{meta['suffix']}"


def fmt_chg(meta, v, mode):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    if meta['kind'] in ('yield', 'spread'):
        return f"{v:+.0f} bps"
    if mode == '%':
        return f"{v:+.2f}%"
    return f"{v:+,.{min(meta['dec'], 2)}f}"


def spark_slice(s, period):
    last = s.index[-1]
    if period == '1D':
        return s.iloc[-6:]
    if period == 'YTD':
        return s[s.index >= pd.Timestamp(last.year, 1, 1)]
    return s[s.index >= last - pd.Timedelta(days=PERIOD_DAYS[period])]


def sparkline(vals, up):
    v = np.asarray(vals, float)
    if len(v) < 2:
        return ''
    if len(v) > 90:
        v = v[np.linspace(0, len(v) - 1, 90).astype(int)]
    lo, hi = v.min(), v.max()
    rng = (hi - lo) or 1.0
    xs = np.linspace(0, 100, len(v))
    ys = 29 - (v - lo) / rng * 27
    pts = ' '.join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    col = '#2D6A0F' if up else '#7A1F1F'
    return (f"<svg viewBox='0 0 100 30' preserveAspectRatio='none' "
            f"style='width:100%;height:48px;display:block;margin:2px 0 14px'>"
            f"<polyline fill='none' stroke='{col}' stroke-width='1.8' "
            f"vector-effect='non-scaling-stroke' points='{pts}'/></svg>")


def is_alert(meta, stt, thr):
    d = stt['pct']['1D']
    if meta['kind'] in ('yield', 'spread'):
        b = stt['bps']['1D']
        return b is not None and abs(b) >= 10
    if d is None:
        return False
    return abs(d) >= thr * (4 if meta['vol'] else 1)


def card(key, S, period, mode, thr):
    meta, s = REG[key], S[key]
    stt = stats(s)
    sel = get_chg(stt, meta, period, mode)
    vc = 'v-warn' if not sel else 'v-up' if sel > 0 else 'v-down'
    sp = spark_slice(s, period)
    spark = sparkline(sp.values, len(sp) < 2 or sp.iloc[-1] >= sp.iloc[0])
    chips = ''
    for p in PERIODS:
        c = get_chg(stt, meta, p, mode)
        cls = 'chg-na' if not c else 'chg-up' if c > 0 else 'chg-down'
        chips += (f"<div class='chip {cls}{' sel' if p == period else ''}'>"
                  f"<div class='chip-l'>{p}</div><div class='chip-v'>{fmt_chg(meta, c, mode)}</div></div>")
    d = meta['decimals'] if 'decimals' in meta else meta['dec']
    num = lambda x: f"{x:,.{d}f}" if meta['kind'] == 'px' else f"{x:.2f}"
    rng = (f"<div class='rng'><div class='rng-dot' style='left:{stt['pos'] * 100:.0f}%'></div></div>"
           f"<div class='rng-l'><span>{num(stt['lo'])}</span><span>52W range</span>"
           f"<span>{num(stt['hi'])}</span></div>")
    age = (pd.Timestamp(datetime.date.today()) - stt['last']).days
    stale = "<span class='tag tag-stale'>STALE</span>" if age > 5 else ''
    src = f"{meta['src']} · {meta['ticker'] or ''}".strip(' ·') if meta['ticker'] else meta['src']
    asof = f"as of {stt['last']:%d %b}"
    alerted = is_alert(meta, stt, thr)
    tags = ('<span class="tag tag-alert">ALERT</span>' if alerted else '') + \
           ('<span class="tag tag-proxy">PROXY</span>' if meta['proxy'] else '') + stale
    html = (f"<div class='mcard{' alert' if alerted else ''}'>"
            f"<div class='mtop'><div class='mlabel'>{meta['label']}</div><div>{tags}</div></div>"
            f"<div class='mvalue {vc}'>{fmt_value(meta, stt['cur'])}</div>"
            f"{spark}<div class='chips'>{chips}</div>{rng}"
            f"<div class='msrc'>{src} · {asof}</div></div>")
    return html.replace('$', '&#36;')


def grid(keys, S, period, mode, thr, ncols=4):
    present = [k for k in keys if k in S]
    for i in range(0, len(present), ncols):
        cols = st.columns(ncols, gap='medium')
        for col, k in zip(cols, present[i:i + ncols]):
            col.markdown(card(k, S, period, mode, thr), unsafe_allow_html=True)
    missing = [REG[k]['label'] for k in keys if k not in S]
    if missing:
        st.caption("No data right now: " + ', '.join(missing))


def header(text):
    st.markdown(f"<div class='section-header'>{text}</div>", unsafe_allow_html=True)


CHART = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#FFFFFF',
             font=dict(family='Inter', color='#666560', size=12),
             margin=dict(l=50, r=20, t=40, b=40), hovermode='x unified',
             legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#D6D3CA'))


def style(fig, h=380, **kw):
    fig.update_layout(**{**CHART, 'height': h, **kw})
    fig.update_xaxes(gridcolor='#ECEAE4', linecolor='#D6D3CA')
    fig.update_yaxes(gridcolor='#ECEAE4', linecolor='#D6D3CA')
    return fig


def rebased(S, keys, days):
    out = {}
    for k in keys:
        if k in S:
            s = S[k]
            w = s[s.index >= s.index[-1] - pd.Timedelta(days=days)]
            if len(w) > 1:
                out[k] = w / w.iloc[0] * 100
    return out


def window(s, days):
    return s[s.index >= s.index[-1] - pd.Timedelta(days=days)]


WIN = {'1M': 30, '3M': 90, '6M': 180, '1Y': 365}


def labels_of(keys):
    return {k: REG[k]['label'] for k in keys}


# ═════════════════════════════════════════════════════════════════════════════
# Tabs
# ═════════════════════════════════════════════════════════════════════════════

def tab_overview(S, period, mode, thr):
    # alert banner
    hits = []
    for k in S:
        if REG[k]['group'] in ('Ratios', 'Custom'):
            continue
        stt = stats(S[k])
        if is_alert(REG[k], stt, thr):
            m = REG[k]
            c = stt['bps']['1D'] if m['kind'] in ('yield', 'spread') else stt['pct']['1D']
            hits.append(f"<b>{m['label']}</b> {fmt_chg(m, c, '%')}")
    if hits:
        st.markdown("<div class='alert-box'>⚠ 1D alerts: " + ' &nbsp;·&nbsp; '.join(hits) + "</div>",
                    unsafe_allow_html=True)
    for title, keys in OVERVIEW_SECTIONS:
        header(title)
        grid(keys, S, period, mode, thr)
    header("India institutional flows (NSE, ₹ crore)")
    fd = fetch_fii_dii()
    if not fd:
        st.markdown("<div class='note-box'>NO FEED — NSE blocks automated requests fairly often. "
                    "Check <a href='https://www.nseindia.com/reports/fii-dii' target='_blank'>nseindia.com/reports/fii-dii</a>. "
                    "A stable alternative is a paid data API or a daily CSV you drop in.</div>",
                    unsafe_allow_html=True)
    else:
        cols = st.columns(4, gap='medium')
        for col, cat, name in zip(cols[:2], ['FII', 'DII'], ['FII / FPI net', 'DII net']):
            d = fd.get(cat)
            if not d:
                continue
            vc = 'v-up' if d['net'] > 0 else 'v-down'
            col.markdown(
                f"<div class='mcard'><div class='mlabel'>{name}</div>"
                f"<div class='mvalue {vc}'>{d['net']:+,.0f} Cr</div>"
                f"<div class='msrc'>Buy {d['buy']:,.0f} · Sell {d['sell']:,.0f} · {d['date']}</div></div>",
                unsafe_allow_html=True)


def tab_metals(S, period, mode, thr):
    header("Metals tracker")
    c1, c2 = st.columns([2, 3])
    group = c1.selectbox("Group", list(METAL_GROUPS) + ['Custom tickers'], key='mgroup')
    custom_raw = c2.text_input("Add Yahoo tickers (comma-separated, e.g. SQM, MP, 1211.HK)",
                               key='mcustom')
    custom = [t.strip() for t in custom_raw.split(',') if t.strip()]
    if custom:
        S = {**S, **load_data(tuple(custom))}   # cached; fetches only the new ones
    if group == 'Custom tickers':
        keys = [f'c_{t}' for t in custom]
        if not keys:
            st.info("Type one or more Yahoo tickers above to track them here.")
    else:
        keys = METAL_GROUPS[group]
    grid(keys, S, period, mode, thr)

    st.markdown(
        "<div class='note-box'><b>PROXY</b> tiles are producer stocks / ETFs, not spot prices. Yahoo has no free "
        "spot or futures feed for tin, tungsten, zinc, nickel, lead or iron ore, so these are the closest "
        "tradable reads. Real LME prices need a paid feed.</div>", unsafe_allow_html=True)
    with st.expander("How each metal is tracked"):
        st.dataframe(METAL_MAP, hide_index=True, use_container_width=True)

    # ranking
    header(f"Metals ranking · {period}")
    rows = []
    for k in ALL_METAL_KEYS:
        if k in S:
            v = stats(S[k])['pct'][period]
            if v is not None:
                rows.append((REG[k]['label'], v))
    if rows:
        rows.sort(key=lambda x: x[1])
        fig = go.Figure(go.Bar(
            x=[r[1] for r in rows], y=[r[0] for r in rows], orientation='h',
            marker_color=['#2D6A0F' if r[1] >= 0 else '#A32D2D' for r in rows],
            text=[f"{r[1]:+.1f}%" for r in rows], textposition='outside'))
        style(fig, h=max(360, 26 * len(rows)), hovermode='closest', xaxis_title=f'% change ({period})')
        st.plotly_chart(fig, use_container_width=True, key='metal_rank')

    # rebased performance
    header("Relative performance (rebased to 100)")
    c1, c2 = st.columns([4, 1])
    default = [k for k in ['gold_usd', 'silver', 'm_copper', 'm_aluminum', 'm_cnxmetal'] if k in S]
    pick = c1.multiselect("Metals", [k for k in ALL_METAL_KEYS if k in S], default=default,
                          format_func=lambda k: REG[k]['label'], key='mreb')
    win = c2.selectbox("Window", list(WIN), index=1, key='mrebw')
    reb = rebased(S, pick, WIN[win])
    if reb:
        fig = go.Figure()
        for k, s in reb.items():
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=REG[k]['label'], mode='lines'))
        fig.add_hline(y=100, line_dash='dot', line_color='#B4B2A9')
        st.plotly_chart(style(fig), use_container_width=True, key='metal_reb')

    # ratios
    header("Metal ratios")
    r1, r2 = st.columns(2, gap='medium')
    for col, k, note in [(r1, 'gold_silver', "Higher = silver cheap vs gold (risk-off / defensive)."),
                         (r2, 'copper_gold', "Higher = growth optimism; falling = defensive.")]:
        with col:
            if k in S:
                w = window(S[k], 365)
                fig = go.Figure(go.Scatter(x=w.index, y=w.values, mode='lines', line=dict(color='#854F0B')))
                style(fig, h=280, title=f"{REG[k]['label']} · now {stats(S[k])['cur']:.{REG[k]['dec']}f}",
                      showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key=f'ratio_{k}')
                st.caption(note)

    # correlation
    header("Metals correlation (daily returns)")
    cw = st.selectbox("Window", ['1M', '3M', '6M', '1Y'], index=1, key='mcorrw')
    ck = [k for k in ['gold_usd', 'silver', 'm_platinum', 'm_copper', 'm_aluminum', 'm_cnxmetal',
                      'm_vale', 'm_hindzinc', 'm_tin', 'm_tungsten', 'dxy', 'usd_inr'] if k in S]
    corr_heat(S, ck, WIN[cw], 'metal_corr')


def tab_food(S, period, mode, thr):
    header("Food grains & agri tracker")
    group = st.selectbox("Group", list(FOOD_GROUPS), key='fgroup')
    grid(FOOD_GROUPS[group], S, period, mode, thr)
    st.markdown(
        "<div class='note-box'>Grain futures are CBOT / ICE prices in US units (mostly cents per bushel), "
        "so they reflect global benchmarks rather than Indian mandi prices. <b>PROXY</b> tiles are stocks / ETFs. "
        "Indian wheat, rice and pulse prices have no free live feed; they need an Agmarknet / e-NAM key.</div>",
        unsafe_allow_html=True)
    with st.expander("How each commodity is tracked"):
        st.dataframe(FOOD_MAP, hide_index=True, use_container_width=True)

    header(f"Food ranking · {period}")
    rows = []
    for k in FOOD_FUTURES:
        if k in S:
            v = stats(S[k])['pct'][period]
            if v is not None:
                rows.append((REG[k]['label'], v))
    if rows:
        rows.sort(key=lambda x: x[1])
        fig = go.Figure(go.Bar(
            x=[r[1] for r in rows], y=[r[0] for r in rows], orientation='h',
            marker_color=['#2D6A0F' if r[1] >= 0 else '#A32D2D' for r in rows],
            text=[f"{r[1]:+.1f}%" for r in rows], textposition='outside'))
        style(fig, h=max(340, 30 * len(rows)), hovermode='closest', xaxis_title=f'% change ({period})')
        st.plotly_chart(fig, use_container_width=True, key='food_rank')

    header("Relative performance (rebased to 100)")
    c1, c2 = st.columns([4, 1])
    default = [k for k in ['f_wheat', 'f_corn', 'f_soy', 'f_rice', 'f_sugar'] if k in S]
    pick = c1.multiselect("Commodities", [k for k in ALL_FOOD_KEYS if k in S], default=default,
                          format_func=lambda k: REG[k]['label'], key='freb')
    win = c2.selectbox("Window", list(WIN), index=1, key='frebw')
    reb = rebased(S, pick, WIN[win])
    if reb:
        fig = go.Figure()
        for k, sr in reb.items():
            fig.add_trace(go.Scatter(x=sr.index, y=sr.values, name=REG[k]['label'], mode='lines'))
        fig.add_hline(y=100, line_dash='dot', line_color='#B4B2A9')
        st.plotly_chart(style(fig), use_container_width=True, key='food_reb')

    header("In rupee terms (what India actually pays)")
    rows = []
    if 'usd_inr' in S:
        fx = stats(S['usd_inr'])['pct'][period]
        for k in FOOD_FUTURES:
            if k in S and fx is not None:
                a = stats(S[k])['pct'][period]
                if a is not None:
                    rows.append({'Commodity': REG[k]['label'], f'USD price {period}': f"{a:+.2f}%",
                                 f'USD/INR {period}': f"{fx:+.2f}%",
                                 f'INR price {period}': f"{((1 + a / 100) * (1 + fx / 100) - 1) * 100:+.2f}%"})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("A weaker rupee adds to imported food inflation (edible oil, sugar, pulses) even when global prices are flat.")

    header("Grain spreads")
    r1, r2 = st.columns(2, gap='medium')
    for col, k, note in [(r1, 'wheat_corn', "Rising = wheat getting dearer vs corn (feed substitution shifts to corn)."),
                         (r2, 'soy_crush', "Processor margin on crushing soybeans into meal and oil; low = weak processing demand.")]:
        with col:
            if k in S:
                w = window(S[k], 365)
                fig = go.Figure(go.Scatter(x=w.index, y=w.values, mode='lines', line=dict(color='#27500A')))
                style(fig, h=280, title=f"{REG[k]['label']} · now {stats(S[k])['cur']:.2f}", showlegend=False)
                st.plotly_chart(fig, use_container_width=True, key=f'fspread_{k}')
                st.caption(note)

    header("What drives food prices (daily-return correlation)")
    cw = st.selectbox("Window", ['1M', '3M', '6M', '1Y'], index=1, key='fcorrw')
    ck = [k for k in ['f_wheat', 'f_corn', 'f_soy', 'f_rice', 'f_soyoil', 'f_sugar', 'crude_brent',
                      'dxy', 'usd_inr', 'f_ntr', 'nifty'] if k in S]
    corr_heat(S, ck, WIN[cw], 'food_corr')


def corr_heat(S, keys, days, key):
    if len(keys) < 2:
        st.info("Not enough data for a correlation matrix.")
        return
    df = pd.concat({k: window(S[k], days + 10) for k in keys}, axis=1).ffill().dropna()
    df = df[df.index >= df.index[-1] - pd.Timedelta(days=days)]
    r = df.pct_change().dropna()
    if len(r) < 5:
        st.info("Not enough overlapping history.")
        return
    c = r.corr()
    names = [REG[k]['label'] for k in c.columns]
    fig = go.Figure(go.Heatmap(z=c.values, x=names, y=names, zmin=-1, zmax=1,
                               colorscale='RdYlGn', text=np.round(c.values, 2),
                               texttemplate='%{text}', hovertemplate='%{x} / %{y}: %{z:.2f}<extra></extra>'))
    style(fig, h=max(420, 44 * len(keys)), hovermode='closest')
    fig.update_yaxes(autorange='reversed')
    st.plotly_chart(fig, use_container_width=True, key=key)
    st.caption(f"{len(r)} daily observations · series forward-filled across market holidays.")


def tab_heatmap(S, period):
    header("Everything vs everything")
    groups = [g for g in GROUP_ORDER if g not in ('Ratios', 'Custom')]
    sel = st.multiselect("Groups", groups, default=groups[:4] + ['Precious (futures)', 'Base metals (futures)'],
                         key='hmgroups')
    keys = [k for k, m in REG.items() if m['group'] in sel and k in S and m['kind'] != 'spread']
    if not keys:
        st.info("Pick at least one group.")
        return
    z, labels = [], []
    for k in keys:
        pc = stats(S[k])['pct']
        z.append([pc[p] if pc[p] is not None else np.nan for p in PERIODS])
        labels.append(REG[k]['label'] + (' ·P' if REG[k]['proxy'] else ''))
    z = np.array(z, float)
    scale = np.nanpercentile(np.abs(z), 90, axis=0)
    scale[scale == 0] = 1
    color = np.clip(z / scale, -1.5, 1.5)
    text = np.where(np.isnan(z), '—', np.char.add(np.char.mod('%+.1f', np.nan_to_num(z)), '%'))
    fig = go.Figure(go.Heatmap(z=color, x=PERIODS, y=labels, text=text, texttemplate='%{text}',
                               colorscale='RdYlGn', zmid=0, zmin=-1.5, zmax=1.5, showscale=False,
                               customdata=z, hovertemplate='%{y} · %{x}: %{customdata:+.2f}%<extra></extra>'))
    style(fig, h=max(400, 30 * len(keys) + 80), hovermode='closest')
    fig.update_yaxes(autorange='reversed')
    fig.update_xaxes(side='top')
    st.plotly_chart(fig, use_container_width=True, key='heatmap')
    st.caption("Colour is scaled per column (so 1D and 1Y are comparable). ·P = proxy (stock / ETF). "
               "Yield rows show % change of the yield level.")


def tab_analysis(S, period):
    # 1 ─ correlation
    header("Cross-asset correlation")
    cw = st.selectbox("Window", ['1M', '3M', '6M', '1Y'], index=1, key='acorrw')
    ck = [k for k in ['gold_usd', 'silver', 'm_copper', 'crude_brent', 'dxy', 'usd_inr', 'nifty',
                      'sp500', 'us10y', 'btc', 'vix'] if k in S]
    corr_heat(S, ck, WIN[cw], 'asset_corr')

    # 2 ─ gold INR decomposition
    header(f"Gold in INR: metal vs currency · {period}")
    if all(k in S for k in ('gold_usd', 'usd_inr', 'gold_inr')):
        a = stats(S['gold_usd'])['pct'][period]
        b = stats(S['usd_inr'])['pct'][period]
        t = stats(S['gold_inr'])['pct'][period]
        if None not in (a, b, t):
            c1, c2, c3 = st.columns(3, gap='medium')
            for col, lab, v in [(c1, 'Gold USD (the metal)', a), (c2, 'USD/INR (the rupee)', b),
                                (c3, 'Gold INR (what you earn)', t)]:
                vc = 'v-up' if v > 0 else 'v-down'
                col.markdown(f"<div class='mcard'><div class='mlabel'>{lab}</div>"
                             f"<div class='mvalue {vc}'>{v:+.2f}%</div></div>", unsafe_allow_html=True)
            share = (b / t * 100) if t else 0
            st.caption(f"Of the {t:+.2f}% move in rupee gold, about {a:+.2f}% came from the metal and "
                       f"{b:+.2f}% from the rupee ({share:.0f}% of the total from currency).")
        reb = rebased(S, ['gold_usd', 'usd_inr', 'gold_inr'], 180)
        fig = go.Figure()
        for k, s in reb.items():
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=REG[k]['label'], mode='lines'))
        st.plotly_chart(style(fig, h=340), use_container_width=True, key='goldinr')

    # 3 ─ Oil → INR → Nifty
    header(f"Oil → Rupee → Nifty chain · {period}")
    if all(k in S for k in ('crude_brent', 'usd_inr', 'nifty')):
        chain = [('crude_brent', 'Brent crude'), ('usd_inr', 'USD/INR (↑ = weaker rupee)'), ('nifty', 'Nifty 50')]
        cols = st.columns(5, gap='small')
        slots = [cols[0], cols[2], cols[4]]
        for col, (k, lab) in zip(slots, chain):
            v = stats(S[k])['pct'][period]
            vc = 'v-warn' if not v else 'v-up' if v > 0 else 'v-down'
            col.markdown(f"<div class='mcard'><div class='mlabel'>{lab}</div>"
                         f"<div class='mvalue {vc}' style='font-size:34px'>{fmt_chg(REG[k], v, '%')}</div></div>",
                         unsafe_allow_html=True)
        for col in (cols[1], cols[3]):
            col.markdown("<div style='text-align:center;font-size:40px;padding-top:44px;color:#B4B2A9'>→</div>",
                         unsafe_allow_html=True)
        df = pd.concat({k: window(S[k], 130) for k in ('crude_brent', 'usd_inr', 'nifty')},
                       axis=1).ffill().dropna()
        r = df.pct_change().dropna()
        if len(r) > 10:
            r3 = r[r.index >= r.index[-1] - pd.Timedelta(days=90)]
            beta = r3['nifty'].cov(r3['crude_brent']) / r3['crude_brent'].var()
            m1, m2, m3, m4 = st.columns(4, gap='medium')
            m1.metric("Brent ↔ USDINR (3M)", f"{r3['crude_brent'].corr(r3['usd_inr']):+.2f}")
            m2.metric("Brent ↔ Nifty (3M)", f"{r3['crude_brent'].corr(r3['nifty']):+.2f}")
            m3.metric("USDINR ↔ Nifty (3M)", f"{r3['usd_inr'].corr(r3['nifty']):+.2f}")
            m4.metric("Nifty beta to Brent", f"{beta:+.2f}")
        reb = rebased(S, ['crude_brent', 'usd_inr', 'nifty'], 180)
        fig = go.Figure()
        for k, s in reb.items():
            fig.add_trace(go.Scatter(x=s.index, y=s.values, name=REG[k]['label'], mode='lines'))
        st.plotly_chart(style(fig, h=340), use_container_width=True, key='chain')

    # 4 ─ real yield
    header("US real yield, breakeven & gold")
    if 'real_yield' in S and 'us10y' in S:
        df = pd.concat({'nominal': S['us10y'], 'real': S['real_yield']}, axis=1).ffill().dropna()
        df['breakeven'] = df['nominal'] - df['real']
        w = df[df.index >= df.index[-1] - pd.Timedelta(days=365)]
        c1, c2, c3 = st.columns(3, gap='medium')
        rs = stats(S['real_yield'])
        c1.metric("10Y real yield", f"{rs['cur']:.2f}%", f"{rs['bps']['1M']:+.0f} bps (1M)" if rs['bps']['1M'] is not None else None)
        c2.metric("10Y nominal", f"{df['nominal'].iloc[-1]:.2f}%")
        c3.metric("10Y breakeven inflation", f"{df['breakeven'].iloc[-1]:.2f}%")
        fig = go.Figure()
        for col, name, colr in [('nominal', 'Nominal', '#0C447C'), ('real', 'Real', '#854F0B'),
                                ('breakeven', 'Breakeven', '#A32D2D')]:
            fig.add_trace(go.Scatter(x=w.index, y=w[col], name=name, line=dict(color=colr)))
        st.plotly_chart(style(fig, h=320), use_container_width=True, key='realy')
        if 'gold_usd' in S:
            g = window(S['gold_usd'], 365)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=w.index, y=w['real'], name='Real yield (lhs)', line=dict(color='#854F0B')))
            fig.add_trace(go.Scatter(x=g.index, y=g.values, name='Gold USD (rhs)', yaxis='y2',
                                     line=dict(color='#C9A227')))
            style(fig, h=320, yaxis2=dict(overlaying='y', side='right', showgrid=False))
            st.plotly_chart(fig, use_container_width=True, key='goldreal')
            st.caption("Gold usually moves inversely to real yields: falling real yield is supportive for gold.")
    else:
        st.info("FRED real-yield data unavailable right now.")

    # 5 ─ 2s10s
    header("US 2s10s curve")
    if 'us_2s10s' in S:
        w = window(S['us_2s10s'], 730) * 100
        fig = go.Figure(go.Scatter(x=w.index, y=w.values, fill='tozeroy', line=dict(color='#0C447C')))
        fig.add_hline(y=0, line_color='#A32D2D', line_dash='dot')
        style(fig, h=300, yaxis_title='bps', showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key='curve')
        st.caption("Below zero = inverted curve (classic recession signal); the steepening after inversion is often "
                   "when the downturn arrives.")


def tab_drill(S):
    header("Drill-down")
    keys = [k for k in S if REG[k]['group'] != 'Custom'] + [k for k in S if REG[k]['group'] == 'Custom']
    k = st.selectbox("Instrument", keys, format_func=lambda x: REG[x]['label'], key='drill')
    meta, s = REG[k], S[k]
    stt = stats(s)
    c = st.columns(6)
    for col, p in zip(c, PERIODS):
        v = get_chg(stt, meta, p, '%')
        col.metric(p, fmt_chg(meta, v, '%'))
    w = s[s.index >= s.index[-1] - pd.Timedelta(days=730)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w.index, y=w.values, name='Close', line=dict(color='#1A1A18', width=2)))
    fig.add_trace(go.Scatter(x=w.index, y=w.rolling(50).mean(), name='50-day avg',
                             line=dict(color='#854F0B', dash='dot')))
    fig.add_hline(y=stt['hi'], line_dash='dash', line_color='#2D6A0F', annotation_text='52W high')
    fig.add_hline(y=stt['lo'], line_dash='dash', line_color='#A32D2D', annotation_text='52W low')
    style(fig, h=430)
    fig.update_xaxes(rangeselector=dict(buttons=[
        dict(count=1, label='1M', step='month', stepmode='backward'),
        dict(count=3, label='3M', step='month', stepmode='backward'),
        dict(count=6, label='6M', step='month', stepmode='backward'),
        dict(count=1, label='1Y', step='year', stepmode='backward'),
        dict(step='all', label='2Y')]))
    st.plotly_chart(fig, use_container_width=True, key='drillchart')
    if meta['ticker']:
        st.markdown("**Latest news**")
        news = fetch_news(meta['ticker'])
        if news:
            for title, url, pub in news:
                st.markdown(f"- [{title.replace('$', chr(92) + '$')}]({url})  \n  <span style='color:#888480;font-size:12px'>{pub}</span>",
                            unsafe_allow_html=True)
        else:
            st.caption("No headlines returned by Yahoo for this ticker.")


# ═════════════════════════════════════════════════════════════════════════════
# Body (refreshable fragment) + render
# ═════════════════════════════════════════════════════════════════════════════

def _body(period, mode, thr):
    with st.spinner("Loading market data..."):
        S = load_data()
    st.caption(f"Last updated {datetime.datetime.now():%d %b %Y, %H:%M:%S} · "
               f"{len(S)} series loaded · Yahoo Finance & FRED (daily bars, ~15 min delayed on some feeds)")
    t1, t2, t2b, t3, t4, t5 = st.tabs(["Overview", "Metals", "Food grains", "Heatmap", "Analysis", "Drill-down"])
    with t1:
        tab_overview(S, period, mode, thr)
    with t2:
        tab_metals(S, period, mode, thr)
    with t2b:
        tab_food(S, period, mode, thr)
    with t3:
        tab_heatmap(S, period)
    with t4:
        tab_analysis(S, period)
    with t5:
        tab_drill(S)


_frag = getattr(st, 'fragment', None)
_body_live = _frag(run_every=300)(_body) if _frag else _body
_body_static = _frag(_body) if _frag else _body

CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background-color: #F5F4F0; }
  .block-container { padding: 1.5rem 2rem; max-width: 1500px; background: #F5F4F0; }
  .section-header { font-size:14px;font-weight:600;color:#888480;text-transform:uppercase;
                    letter-spacing:0.08em;border-bottom:0.5px solid #D6D3CA;
                    padding-bottom:8px;margin:28px 0 18px; }
  .mcard  { background:#FFFFFF;border:0.5px solid #D6D3CA;border-radius:14px;
            padding:22px 24px;margin-bottom:18px; }
  .mcard.alert { border:1.5px solid #E8940A;background:#FFFDF7; }
  .mtop   { display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:6px; }
  .mlabel { font-size:13px;color:#666560;text-transform:uppercase;letter-spacing:0.08em;font-weight:500; }
  .mvalue { font-size:40px;font-weight:600;line-height:1.1;margin-bottom:6px; }
  .v-up { color:#2D6A0F; } .v-down { color:#7A1F1F; } .v-warn { color:#7A4A0A; }
  .chips  { display:grid;grid-template-columns:repeat(3,1fr);gap:8px; }
  .chip   { border-radius:9px;padding:7px 4px;text-align:center; }
  .chip.sel { outline:2px solid #1A1A18; }
  .chip-l { font-size:11px;font-weight:600;letter-spacing:0.06em;opacity:0.7;margin-bottom:2px; }
  .chip-v { font-size:15px;font-weight:600; }
  .chg-up { background:#EAF3DE;color:#27500A; } .chg-down { background:#FCEBEB;color:#791F1F; }
  .chg-na { background:#F1EFE8;color:#666560; }
  .rng { position:relative;height:6px;background:#ECEAE4;border-radius:3px;margin-top:16px; }
  .rng-dot { position:absolute;top:-3px;width:12px;height:12px;border-radius:50%;
             background:#1A1A18;transform:translateX(-50%); }
  .rng-l { display:flex;justify-content:space-between;font-size:11px;color:#888480;margin-top:6px; }
  .msrc { font-size:12px;color:#888480;margin-top:12px; }
  .tag { font-size:10px;font-weight:600;padding:2px 6px;border-radius:4px;margin-left:4px; }
  .tag-alert { background:#FAEEDA;color:#633806; } .tag-proxy { background:#EEEDFE;color:#3C3489; }
  .tag-stale { background:#F1EFE8;color:#666560; }
  .alert-box { background:#FFFBF0;border:0.5px solid #FAC775;border-left:3px solid #E8940A;
               border-radius:6px;padding:12px 16px;font-size:14px;color:#633806;margin-bottom:6px; }
  .note-box { background:#FFFFFF;border:0.5px solid #D6D3CA;border-left:2px solid #B4B2A9;
              border-radius:6px;padding:10px 14px;font-size:12px;color:#666560;margin-top:6px; }
  .stTabs [data-baseweb="tab-list"] { background:#ECEAE4;border-radius:8px;padding:3px;gap:3px; }
  .stTabs [data-baseweb="tab"] { background:transparent;color:#888480;border-radius:6px;padding:6px 18px;font-size:14px; }
  .stTabs [aria-selected="true"] { background:#FFFFFF !important;color:#1A1A18 !important;
                                   box-shadow:0 1px 3px rgba(0,0,0,0.08); }
  div[data-testid="stMarkdownContainer"] p { margin:0; }
</style>
"""


def render():
    render_gift_nifty_banner()
    st.markdown(CSS, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([4, 2, 2, 2])
    period = c1.radio("Period (colours, sparklines & ranking)", PERIODS, horizontal=True, key='period')
    mode = c2.radio("Change shown as", ['%', 'Absolute'], horizontal=True, key='mode')
    auto = c3.toggle("Auto-refresh (5 min)", value=True, key='auto')
    thr = c4.number_input("Alert at 1D move ≥ %", 0.5, 20.0, 2.0, 0.5, key='thr',
                          help="Yields/spreads alert at 10 bps; VIX at 4× this value.")
    (_body_live if auto else _body_static)(period, mode, thr)
