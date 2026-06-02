import os
import glob
import duckdb
import pandas as pd
import numpy as np
import streamlit as st
from mixer_doc import GATE_LOGIC_MD

# ==============================================================================
# 1. PAGE CONFIG & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Quality-on-Dip Intelligence Engine",
    page_icon="🎯",
    layout="wide"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; }
    .status-box {
        background-color: #f4f6f7; border-radius: 4px;
        padding: 10px; font-family: monospace; font-size: 12px;
    }
    .score-pill {
        display: inline-block; padding: 3px 10px;
        border-radius: 12px; font-weight: bold; font-size: 13px;
    }
    .gate-block {
        background: #fafbfc; border: 1px solid #e2e8f0;
        border-radius: 6px; padding: 12px; margin-bottom: 8px;
    }
    .section-divider { border-top: 2px solid #e2e8f0; margin: 20px 0 12px 0; }
    div[data-testid="metric-container"] { background: #f8fafc; border-radius: 6px; padding: 8px 12px; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Quality-on-Dip Intelligence Engine")
st.markdown("Identifies fundamentally strong businesses currently trading at a valuation discount — separating business quality from price noise.")

# ==============================================================================
# 2. DUCKDB CONNECTION
# ==============================================================================
@st.cache_resource
def get_duckdb_connection():
    return duckdb.connect(database=':memory:')

conn = get_duckdb_connection()

# ==============================================================================
# 3. DATA INITIALISATION — REAL PARQUET OR MOCK SANDBOX
# ==============================================================================
def initialize_views_or_mock(conn):
    sql_path = "sqls/init.sql"

    if not os.path.exists(sql_path):
        st.sidebar.warning("⚠️ init.sql not found — running Simulation Sandbox.")
        _build_mock_data(conn)
        return "MOCK"

    st.sidebar.success("📊 Connected to Local Parquet Data Lake")
    try:
        with open(sql_path, "r") as f:
            sql = f.read()
        # Split on semicolons and execute each non-empty statement
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
        return "REAL"
    except Exception as e:
        st.sidebar.error(f"Error executing init.sql: {str(e)}")
        return "ERROR"


def _build_mock_data(conn):
    assert True, "general_info table should exist"


env_status = initialize_views_or_mock(conn)

# ==============================================================================
# 4. SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.header("⚙️ Filters")
min_roce    = st.sidebar.slider("Min ROCE %", 0.0, 30.0, 0.0, step=1.0)
dip_mode    = False   # kept for gate 4c logic compatibility

st.sidebar.markdown("---")
st.sidebar.caption(f"Data source: **{'Real Parquet' if env_status == 'REAL' else 'Simulation Sandbox'}**")

# ==============================================================================
# 5. PRE-FLIGHT DIAGNOSTICS
# ==============================================================================
st.markdown("### 🔍 Pre-Flight Data Diagnostics")
diag_cols = st.columns(4)

with diag_cols[0]:
    try:
        cnt = conn.execute("SELECT COUNT(DISTINCT Ticker) FROM ticker_prices").fetchone()[0]
        st.markdown(f"<div class='status-box'><b>ticker_prices</b><br>Unique Tickers: {cnt}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"ticker_prices: {e}")

with diag_cols[1]:
    try:
        cnt = conn.execute("SELECT COUNT(DISTINCT ticker) FROM general_info").fetchone()[0]
        raw = conn.execute("SELECT COUNT(*) FROM general_info").fetchone()[0]
        dupes = raw - cnt
        st.markdown(f"<div class='status-box'><b>general_info</b><br>Unique: {cnt} | Dupes removed: {dupes}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"general_info: {e}")

with diag_cols[2]:
    try:
        cnt = conn.execute("SELECT COUNT(DISTINCT ticker) FROM quarterly_results").fetchone()[0]
        st.markdown(f"<div class='status-box'><b>quarterly_results</b><br>Tickers: {cnt}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"quarterly_results: {e}")

with diag_cols[3]:
    try:
        cnt = conn.execute("""
            SELECT COUNT(*) FROM (SELECT DISTINCT Ticker FROM ticker_prices) t
            JOIN (SELECT DISTINCT ticker FROM general_info) g
              ON t.Ticker = g.ticker OR REPLACE(t.Ticker,'.NS','') = g.ticker
        """).fetchone()[0]
        st.markdown(f"<div class='status-box'><b>Cross-Source Alignment</b><br>Matched Tickers: {cnt}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Alignment check: {e}")

# ==============================================================================
# 6. CORE ANALYTICS PIPELINE
# ==============================================================================

def run_pipeline(min_roce_filter):
    """
    Single query that computes all raw metrics needed for Quality, Valuation,
    Timing, and Technical scores. general_info is deduplicated with QUALIFY.
    """
    query = f"""
    -- ── Deduplicated company context ──
    -- general_info already has clean numeric columns; nse_symbol is the join key
    WITH company AS (
        SELECT
         nse_symbol AS ticker, company_name, sector, market_cap, stock_p_e, roce, roe, book_value, current_price from general_info
        QUALIFY ROW_NUMBER() OVER (PARTITION BY nse_symbol ORDER BY nse_symbol) = 1
    ),

    -- ── Sector median P/E ──
    sector_pe AS (
        SELECT sector, MEDIAN(stock_p_e) AS sector_median_pe
        FROM company
        WHERE stock_p_e > 0
        GROUP BY sector
    ),

    -- ── Quarterly results: dt/val already clean in view ──
    qr_ranked AS (
        SELECT *,
            DENSE_RANK() OVER (PARTITION BY ticker ORDER BY dt DESC) AS rk
        FROM (SELECT DISTINCT ticker, dt, metric, val FROM quarterly_results)
    ),
    qr_pivot AS (
        SELECT
            ticker,
            MAX(CASE WHEN rk=1 AND metric='Sales'      THEN val END) AS sales_q0,
            MAX(CASE WHEN rk=2 AND metric='Sales'      THEN val END) AS sales_q1,
            MAX(CASE WHEN rk=3 AND metric='Sales'      THEN val END) AS sales_q2,
            MAX(CASE WHEN rk=4 AND metric='Sales'      THEN val END) AS sales_q3,
            MAX(CASE WHEN rk=1 AND metric='OPM %'      THEN val END) AS opm_q0,
            MAX(CASE WHEN rk=2 AND metric='OPM %'      THEN val END) AS opm_q1,
            MAX(CASE WHEN rk=1 AND metric='Net Profit' THEN val END) AS profit_q0,
            MAX(CASE WHEN rk=2 AND metric='Net Profit' THEN val END) AS profit_q1,
            MAX(CASE WHEN rk=3 AND metric='Net Profit' THEN val END) AS profit_q2,
            MAX(CASE WHEN rk=4 AND metric='Net Profit' THEN val END) AS profit_q3,
            MAX(CASE WHEN rk=1 AND metric='EPS in Rs'  THEN val END) AS eps_q0,
            MAX(CASE WHEN rk=2 AND metric='EPS in Rs'  THEN val END) AS eps_q1
        FROM qr_ranked
        WHERE rk <= 4
        GROUP BY ticker
    ),

    -- ── Annual P&L (dt is a DATE; exclude TTM by filtering dt < today) ──
    pl_years AS (
        SELECT ticker, dt,
            DENSE_RANK() OVER (PARTITION BY ticker ORDER BY dt DESC) AS yr_rk
        FROM (SELECT DISTINCT ticker, dt FROM profit_loss WHERE dt < CURRENT_DATE)
    ),
    pl_latest AS (
        SELECT a.ticker,
            MAX(CASE WHEN metric='Net Profit' THEN val END) AS profit_now,
            MAX(CASE WHEN metric='Sales'      THEN val END) AS sales_now,
            MAX(CASE WHEN metric='Interest'   THEN val END) AS interest
        FROM profit_loss a
        JOIN pl_years y ON a.ticker = y.ticker AND a.dt = y.dt
        WHERE y.yr_rk = 1
        GROUP BY a.ticker
    ),
    pl_3y AS (
        SELECT a.ticker,
            MAX(CASE WHEN metric='Net Profit' THEN val END) AS profit_3y,
            MAX(CASE WHEN metric='Sales'      THEN val END) AS sales_3y
        FROM profit_loss a
        JOIN pl_years y ON a.ticker = y.ticker AND a.dt = y.dt
        WHERE y.yr_rk = 4
        GROUP BY a.ticker
    ),

    -- ── Cash flows ──
    cf_latest AS (
        SELECT a.ticker,
            MAX(CASE WHEN metric='Cash from Operating Activity' THEN val END) AS cfo,
            MAX(CASE WHEN metric='Free Cash Flow'               THEN val END) AS fcf
        FROM cash_flows a
        WHERE a.dt = (SELECT MAX(dt) FROM cash_flows WHERE ticker = a.ticker)
        GROUP BY a.ticker
    ),

    -- ── Balance sheet ──
    bs_latest AS (
        SELECT a.ticker,
            MAX(CASE WHEN metric IN ('Borrowing','Borrowings') THEN val END) AS debt,
            MAX(CASE WHEN metric='Reserves'                    THEN val END) AS reserves,
            MAX(CASE WHEN metric='Equity Capital'              THEN val END) AS equity_capital
        FROM balance_sheet a
        WHERE a.dt = (SELECT MAX(dt) FROM balance_sheet WHERE ticker = a.ticker)
        GROUP BY a.ticker
    ),

    -- ── Shareholding (dt/val/category from init.sql view) ──
    sh_ranked AS (
        SELECT ticker, dt, category, val,
            DENSE_RANK() OVER (PARTITION BY ticker ORDER BY dt DESC) AS rk
        FROM shareholding_quarterly
    ),
    sh_latest AS (
        SELECT ticker,
            MAX(CASE WHEN category='Promoters' THEN val END) AS promoter,
            MAX(CASE WHEN category='FIIs'      THEN val END) AS fii,
            MAX(CASE WHEN category='DIIs'      THEN val END) AS dii
        FROM sh_ranked WHERE rk = 1
        GROUP BY ticker
    ),
    sh_prev AS (
        SELECT ticker,
            MAX(CASE WHEN category='Promoters' THEN val END) AS promoter_prev,
            MAX(CASE WHEN category='FIIs'      THEN val END) AS fii_prev,
            MAX(CASE WHEN category='DIIs'      THEN val END) AS dii_prev
        FROM sh_ranked WHERE rk = 2
        GROUP BY ticker
    ),

    -- ── Technical indicators (ticker_prices view from init.sql) ──
    daily_ind AS (
        SELECT
            nse_symbol AS Ticker, Date, Close, Volume,
            LAG(Close, 21) OVER (PARTITION BY nse_symbol ORDER BY Date) AS close_1m,
            LAG(Close, 63) OVER (PARTITION BY nse_symbol ORDER BY Date) AS close_3m,
            AVG(Close)    OVER (PARTITION BY nse_symbol ORDER BY Date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma_20,
            STDDEV(Close) OVER (PARTITION BY nse_symbol ORDER BY Date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS std_20,
            AVG(Close)    OVER (PARTITION BY nse_symbol ORDER BY Date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS ma_50,
            MAX(Close)    OVER (PARTITION BY nse_symbol ORDER BY Date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) AS high_52w,
            AVG(Volume)   OVER (PARTITION BY nse_symbol ORDER BY Date ROWS BETWEEN 4  PRECEDING AND CURRENT ROW) AS vol_ma_5,
            AVG(Volume)   OVER (PARTITION BY nse_symbol ORDER BY Date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS vol_ma_50
        FROM ticker_prices
    ),
    tech_states AS (
        SELECT *,
            ((Close - close_1m) / NULLIF(close_1m,0)) * 100  AS return_1m,
            ((Close - close_3m) / NULLIF(close_3m,0)) * 100  AS return_3m,
            (4.0 * std_20) / NULLIF(ma_20,0)                  AS bb_width,
            ((Close - high_52w) / NULLIF(high_52w,0)) * 100   AS pct_from_52w,
            vol_ma_5 / NULLIF(vol_ma_50,0)                    AS volume_ratio,
            ((Close - ma_50) / NULLIF(ma_50,0)) * 100         AS dist_from_ma50
        FROM daily_ind
    ),
    tech_latest AS (
        SELECT *,
            MIN(bb_width) OVER (PARTITION BY Ticker ORDER BY Date ROWS BETWEEN 62 PRECEDING AND CURRENT ROW) AS min_bb_63d
        FROM tech_states
        QUALIFY ROW_NUMBER() OVER (PARTITION BY Ticker ORDER BY Date DESC) = 1
    ),

    -- ── FINAL JOIN ──
    combined AS (
        SELECT
            c.ticker, c.company_name, c.sector,
            c.market_cap, c.stock_p_e, c.roce, c.roe,
            c.book_value, c.current_price,
            sp.sector_median_pe,

            q.sales_q0, q.sales_q1, q.sales_q2, q.sales_q3,
            q.opm_q0,   q.opm_q1,
            q.profit_q0,q.profit_q1,q.profit_q2,q.profit_q3,
            q.eps_q0,   q.eps_q1,

            pl.profit_now, pl.sales_now, pl.interest,
            py.profit_3y,  py.sales_3y,
            cf.cfo,        cf.fcf,
            bs.debt,       bs.reserves, bs.equity_capital,

            sh.promoter,       sh.fii,       sh.dii,
            sp2.promoter_prev, sp2.fii_prev, sp2.dii_prev,

            t.Close AS price, t.return_1m, t.return_3m,
            t.bb_width, t.min_bb_63d, t.pct_from_52w,
            t.volume_ratio, t.dist_from_ma50

        FROM company c
        JOIN sector_pe   sp  ON c.sector = sp.sector
        JOIN tech_latest t   ON c.ticker = t.Ticker
        LEFT JOIN qr_pivot  q   ON c.ticker = q.ticker
        LEFT JOIN pl_latest pl  ON c.ticker = pl.ticker
        LEFT JOIN pl_3y     py  ON c.ticker = py.ticker
        LEFT JOIN cf_latest cf  ON c.ticker = cf.ticker
        LEFT JOIN bs_latest bs  ON c.ticker = bs.ticker
        LEFT JOIN sh_latest sh  ON c.ticker = sh.ticker
        LEFT JOIN sh_prev   sp2 ON c.ticker = sp2.ticker
        WHERE COALESCE(c.roce, 0) >= {min_roce_filter}
    )

    SELECT * FROM combined
    ORDER BY ticker
    """
    return conn.execute(query).df()


@st.cache_data(ttl=120)
def get_pipeline_data(min_roce_filter):
    return run_pipeline(min_roce_filter)


df_raw = get_pipeline_data(min_roce)

# ==============================================================================
# 7. SCORE ENGINE — Quality / Valuation / Timing / Technical
# ==============================================================================

def compute_scores(df_raw):
    rows = []
    dip_mode = False

    for _, r in df_raw.iterrows():

        # ── GATE 1: QUALITY ─────────────────────────────────────────────────
        # Is this a genuinely good business?

        # 1a. ROCE health (>18% = strong, >12% = ok, <8% = weak)
        roce = r['roce'] or 0
        g1a_pos = bool(roce >= 18)
        g1a_neg = bool(roce < 8)

        # 1b. Free Cash Flow positive
        g1b_pos = bool(pd.notna(r['fcf']) and r['fcf'] > 0)
        g1b_neg = bool(pd.notna(r['fcf']) and r['fcf'] < 0)

        # 1c. Profit CAGR > 15% over 3 years
        profit_cagr = None
        if (
                pd.notna(r['profit_now'])
                and pd.notna(r['profit_3y'])
                and r['profit_now'] > 0
                and r['profit_3y'] > 0
        ):
            profit_cagr = ((r['profit_now'] / r['profit_3y']) ** (1/3) - 1) * 100
        g1c_pos = bool(profit_cagr is not None and profit_cagr >= 15)
        g1c_neg = bool(profit_cagr is not None and profit_cagr < 0)

        # 1d. CFO > Net Profit (real earnings, not accounting fiction)
        g1d_pos = bool(pd.notna(r['cfo']) and pd.notna(r['profit_now']) and r['profit_now'] > 0 and r['cfo'] > r['profit_now'])
        g1d_neg = bool(pd.notna(r['cfo']) and pd.notna(r['profit_now']) and r['profit_now'] > 0 and r['cfo'] < r['profit_now'] * 0.5)

        # 1e. Debt/Equity < 0.5 (low leverage)
        # Skip for Banks/NBFCs — high leverage is their business model, not a risk signal
        FINANCIAL_SECTORS = {'Banks', 'Finance', 'NBFC', 'Insurance'}
        is_financial = any(fs.lower() in str(r['sector']).lower() for fs in FINANCIAL_SECTORS)
        de = None
        if pd.notna(r['debt']) and pd.notna(r['reserves']) and pd.notna(r['equity_capital']):
            equity = r['reserves'] + r['equity_capital']
            de = r['debt'] / equity if equity > 0 else None
        if is_financial:
            g1e_pos, g1e_neg = False, False   # neutral — not applicable
        else:
            g1e_pos = bool(de is not None and de < 0.5)
            g1e_neg = bool(de is not None and de > 1.5)

        q_positives = sum([g1a_pos, g1b_pos, g1c_pos, g1d_pos, g1e_pos])
        q_negatives = sum([g1a_neg, g1b_neg, g1c_neg, g1d_neg, g1e_neg])
        quality_raw = (q_positives * 20) - (q_negatives * 15)            # 0–100 range
        quality_score = max(0, min(100, quality_raw))

        # ── GATE 2: VALUATION ────────────────────────────────────────────────
        # Is this stock cheap vs peers and its own history?

        stock_pe = r['stock_p_e']
        sector_pe = r['sector_median_pe']

        # 2a. P/E discount vs sector median
        pe_discount_pct = None
        g2a_pos, g2a_neg = False, False
        if pd.notna(stock_pe) and pd.notna(sector_pe) and sector_pe > 0 and stock_pe > 0:
            pe_discount_pct = ((sector_pe - stock_pe) / sector_pe) * 100
            g2a_pos = bool(pe_discount_pct >= 15)    # ≥15% cheaper than sector
            g2a_neg = bool(pe_discount_pct < -30)    # >30% premium to sector

        # 2b. PEG proxy (P/E ÷ profit CAGR; <1 = cheap for growth)
        peg = None
        g2b_pos, g2b_neg = False, False
        if pd.notna(stock_pe) and profit_cagr is not None and profit_cagr > 0:
            peg = stock_pe / profit_cagr
            g2b_pos = bool(peg < 1.0)
            g2b_neg = bool(peg > 2.5)

        # 2c. P/B < 3 (not egregiously expensive on assets)
        pb = None
        g2c_pos, g2c_neg = False, False
        if pd.notna(r['current_price']) and pd.notna(r['book_value']) and r['book_value'] > 0:
            pb = r['current_price'] / r['book_value']
            g2c_pos = bool(pb < 3.0)
            g2c_neg = bool(pb > 8.0)

        v_positives = sum([g2a_pos, g2b_pos, g2c_pos])
        v_negatives = sum([g2a_neg, g2b_neg, g2c_neg])
        valuation_raw   = (v_positives * 33) - (v_negatives * 20)
        valuation_score = max(0, min(100, valuation_raw))

        # ── GATE 3: TIMING ───────────────────────────────────────────────────
        # Is the business still accelerating? Are institutions accumulating?

        # 3a. Quarterly sales growth (q0 > q1 > q2)
        g3a_pos, g3a_neg = False, False
        if pd.notna(r['sales_q0']) and pd.notna(r['sales_q1']) and pd.notna(r['sales_q2']):
            g3a_pos = bool(r['sales_q0'] > r['sales_q1'] > r['sales_q2'])
            g3a_neg = bool(r['sales_q0'] < r['sales_q1'])

        # 3b. Quarterly profit growth (q0 > q1 > q2 > q3)
        g3b_pos, g3b_neg = False, False
        if pd.notna(r['profit_q0']) and pd.notna(r['profit_q1']):
            g3b_pos = bool(r['profit_q0'] > r['profit_q1'])
            g3b_neg = bool(r['profit_q0'] < r['profit_q1'] and r['profit_q0'] > 0)

        # 3c. OPM expanding (margin improvement)
        g3c_pos, g3c_neg = False, False
        if pd.notna(r['opm_q0']) and pd.notna(r['opm_q1']):
            g3c_pos = bool(r['opm_q0'] > r['opm_q1'])
            g3c_neg = bool(r['opm_q0'] < r['opm_q1'] - 3)   # >3pp margin squeeze

        # 3d. Promoter holding stable/rising
        g3d_pos, g3d_neg = False, False
        if pd.notna(r['promoter']) and pd.notna(r['promoter_prev']):
            g3d_pos = bool(r['promoter'] >= r['promoter_prev'])
            g3d_neg = bool(r['promoter'] < r['promoter_prev'] - 2)  # >2pp drop = concern

        # 3e. FII + DII accumulating
        g3e_pos, g3e_neg = False, False
        if pd.notna(r['fii']) and pd.notna(r['fii_prev']) and pd.notna(r['dii']) and pd.notna(r['dii_prev']):
            fii_up = r['fii'] >= r['fii_prev']
            dii_up = r['dii'] >= r['dii_prev']
            g3e_pos = bool(fii_up or dii_up)
            g3e_neg = bool(not fii_up and not dii_up)

        t_positives = sum([g3a_pos, g3b_pos, g3c_pos, g3d_pos, g3e_pos])
        t_negatives = sum([g3a_neg, g3b_neg, g3c_neg, g3d_neg, g3e_neg])
        timing_raw   = (t_positives * 20) - (t_negatives * 15)
        timing_score = max(0, min(100, timing_raw))

        # ── GATE 4: TECHNICAL ────────────────────────────────────────────────
        # Price structure — where is it in its range?

        # 4a. Price in Bollinger squeeze (low volatility coiling)
        g4a_pos = bool(pd.notna(r['bb_width']) and pd.notna(r['min_bb_63d'])
                       and r['bb_width'] <= r['min_bb_63d'] * 1.15)
        g4a_neg = bool(pd.notna(r['bb_width']) and r['bb_width'] > 0.35)

        # 4b. Within constructive zone of 50 DMA (0% to +12%)
        g4b_pos = bool(pd.notna(r['dist_from_ma50'])
                       and 0 < r['dist_from_ma50'] < 12)
        g4b_neg = bool(pd.notna(r['dist_from_ma50'])
                       and (r['dist_from_ma50'] > 25 or r['dist_from_ma50'] < -8))

        # 4c. Drawdown from 52W high (dip opportunity detector)
        # In Dip Mode: being 15-40% below 52W high is a positive (buying window)
        g4c_dip_pos = bool(pd.notna(r['pct_from_52w']) and -40 <= r['pct_from_52w'] <= -10)
        g4c_pos     = bool(pd.notna(r['pct_from_52w']) and r['pct_from_52w'] >= -5)
        g4c_neg     = bool(pd.notna(r['pct_from_52w']) and r['pct_from_52w'] < -40)

        # 4d. Volume confirmation
        g4d_pos = bool(pd.notna(r['volume_ratio']) and r['volume_ratio'] >= 1.1)
        g4d_neg = bool(pd.notna(r['volume_ratio']) and r['volume_ratio'] < 0.6)

        # Pick gate 4c signal based on mode
        active_g4c_pos = g4c_dip_pos if dip_mode else g4c_pos

        tech_positives = sum([g4a_pos, g4b_pos, active_g4c_pos, g4d_pos])
        tech_negatives = sum([g4a_neg, g4b_neg, g4c_neg, g4d_neg])
        technical_raw   = (tech_positives * 25) - (tech_negatives * 15)
        technical_score = max(0, min(100, technical_raw))

        # ── COMPOSITE SCORE ──────────────────────────────────────────────────
        # (removed — composite scoring not used in this view)

        rows.append({
            # Identity
            'Ticker':        r['ticker'],
            'Company':       r['company_name'],
            'Sector':        r['sector'],
            'Mkt Cap (Cr)':  r['market_cap'],
            'Stock P/E':     round(stock_pe,  1) if pd.notna(stock_pe)  else None,
            'Sector P/E':    round(sector_pe, 1) if pd.notna(sector_pe) else None,
            'PE Discount %': round(pe_discount_pct, 1) if pe_discount_pct is not None else None,
            'PEG':           round(peg, 2) if peg is not None else None,
            'ROCE %':        round(roce, 1),
            '3Y Profit CAGR%': round(profit_cagr, 1) if profit_cagr is not None else None,
            'D/E':           round(de, 2) if de is not None else None,
            'Promoter %':    round(r['promoter'], 1) if pd.notna(r['promoter']) else None,

            # Scores
            'Quality Score':    round(quality_score),
            'Valuation Score':  round(valuation_score),
            'Timing Score':     round(timing_score),
            'Technical Score':  round(technical_score),

            # Gate detail booleans
            'G1a(+) ROCE':       g1a_pos, 'G1a(-) ROCE':    g1a_neg,
            'G1b(+) FCF':        g1b_pos, 'G1b(-) FCF':     g1b_neg,
            'G1c(+) ProfCAGR':   g1c_pos, 'G1c(-) ProfCAGR':g1c_neg,
            'G1d(+) CFO>NP':     g1d_pos, 'G1d(-) CFO<NP':  g1d_neg,
            'G1e(+) LowDebt':    g1e_pos, 'G1e(-) HiDebt':  g1e_neg,

            'G2a(+) PE<Sector':  g2a_pos, 'G2a(-) PE>Sector':g2a_neg,
            'G2b(+) PEG<1':      g2b_pos, 'G2b(-) PEG>2.5': g2b_neg,
            'G2c(+) PB<3':       g2c_pos, 'G2c(-) PB>8':    g2c_neg,

            'G3a(+) SalesAcc':   g3a_pos, 'G3a(-) SalesDec': g3a_neg,
            'G3b(+) ProfAcc':    g3b_pos, 'G3b(-) ProfDec':  g3b_neg,
            'G3c(+) OPMup':      g3c_pos, 'G3c(-) OPMsqz':   g3c_neg,
            'G3d(+) Promoter':   g3d_pos, 'G3d(-) PromoSell': g3d_neg,
            'G3e(+) InstAcc':    g3e_pos, 'G3e(-) InstExit':  g3e_neg,

            'G4a(+) BBsqz':      g4a_pos, 'G4a(-) BBchaos':  g4a_neg,
            'G4b(+) MA50zone':   g4b_pos, 'G4b(-) MA50break': g4b_neg,
            'G4c(+) Dip/High':   active_g4c_pos, 'G4c(-) Crash': g4c_neg,
            'G4d(+) VolConf':    g4d_pos, 'G4d(-) VolDry':   g4d_neg,

            # Raw technicals for inspector
            '_return_1m':    r['return_1m'],
            '_return_3m':    r['return_3m'],
            '_bb_width':     r['bb_width'],
            '_min_bb_63d':   r['min_bb_63d'],
            '_pct_52w':      r['pct_from_52w'],
            '_vol_ratio':    r['volume_ratio'],
            '_dist_ma50':    r['dist_from_ma50'],
            '_sales_q0':     r['sales_q0'],
            '_sales_q1':     r['sales_q1'],
            '_opm_q0':       r['opm_q0'],
            '_opm_q1':       r['opm_q1'],
            '_profit_cagr':  profit_cagr,
            '_de':           de,
            '_pb':           pb,
            '_peg':          peg,
            '_pe_disc':      pe_discount_pct,
            '_promoter':     r['promoter'],
            '_promoter_prev':r['promoter_prev'],
            '_fii':          r['fii'],
            '_fii_prev':     r['fii_prev'],
            '_dii':          r['dii'],
            '_dii_prev':     r['dii_prev'],
        })

    sort_col = 'Quality Score'
    df = pd.DataFrame(rows).sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    return df


# ==============================================================================
# 8. RENDER MATRIX TABLE
# ==============================================================================
if df_raw.empty:
    st.error("❌ No data returned. Check your parquet paths or ROCE filter.")
    st.stop()

df_matrix = compute_scores(df_raw)

if df_matrix.empty:
    st.warning("No stocks pass the current filters. Relax the ROCE minimum.")
    st.stop()

# ==============================================================================
# 9. SINGLE-TICKER DEEP DIVE
# ==============================================================================
st.markdown("---")
st.markdown("### 🔍 Single-Ticker Deep Dive")

ticker_list  = df_matrix['Ticker'].tolist()
saved        = st.session_state.get("selected_asset")
default_idx  = ticker_list.index(saved) if saved in ticker_list else 0

# ==============================================================================
# METRIC CATALOGUE  (built once, used by selectors + chart renderer)
# ==============================================================================
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS      = ["#2563eb", "#dc2626", "#16a34a", "#d97706"]
COLORS_PALE = ["#93c5fd", "#fca5a5", "#86efac", "#fde68a"]   # lighter twin for 2nd y-axis

TIMESERIES_METRICS = ["Price (Close)", "Return %"]

# df_matrix snapshot columns (numeric, skip booleans + internals)
_SNAP_EXCLUDE = {
    'Ticker', 'Company', 'Sector',
    *[c for c in df_matrix.columns if c.startswith('G') or c.startswith('_')]
}
SNAPSHOT_METRICS = [
    c for c in df_matrix.columns
    if c not in _SNAP_EXCLUDE and pd.api.types.is_numeric_dtype(df_matrix[c])
]

# ticker_momentum columns (prefix [M] so user knows the source)
@st.cache_data(ttl=300)
def get_momentum_columns():
    try:
        cols = conn.execute("SELECT * FROM ticker_momentum LIMIT 0").df().columns.tolist()
        # exclude join keys / dates
        skip = {'ticker', 'Ticker', 'date', 'Date', 'nse_symbol'}
        return [f"[M] {c}" for c in cols if c not in skip]
    except Exception:
        return []

MOMENTUM_COLS = get_momentum_columns()

# Fundamental views: (view_name, display_label)
FUNDAMENTAL_VIEWS = [
    ("quarterly_results", "Quarterly"),
    ("profit_loss",       "P&L"),
    ("cash_flows",        "Cash Flow"),
    ("ratios",            "Ratios"),
    ("shareholding_quarterly", "Shareholding"),
    ("balance_sheet",     "Balance Sheet"),
]

@st.cache_data(ttl=300)
def get_fundamental_metrics():
    """Return list of '[F] View::metric' strings for the metric selector."""
    opts = []
    for view, label in FUNDAMENTAL_VIEWS:
        try:
            rows = conn.execute(
                f"SELECT DISTINCT metric FROM {view} ORDER BY metric"
            ).df()
            for m in rows['metric'].tolist():
                opts.append(f"[F] {label}::{m}")
        except Exception:
            pass
    return opts

FUNDAMENTAL_COLS = get_fundamental_metrics()

ALL_METRIC_OPTIONS = TIMESERIES_METRICS + SNAPSHOT_METRICS + MOMENTUM_COLS + FUNDAMENTAL_COLS

# ==============================================================================
# SELECTORS  — stock | compare tickers | up to 2 metrics
# ==============================================================================
sel_col, cmp_col, met_col = st.columns([1, 2, 2])

with sel_col:
    selected = st.selectbox(
        "Select stock for diagnostic breakdown:",
        ticker_list,
        index=default_idx,
        key="selected_asset"
    )

with cmp_col:
    compare_options = [t for t in ticker_list if t != selected]
    compare_tickers = st.multiselect(
        "Compare with up to 3 tickers:",
        compare_options,
        default=[],
        max_selections=3,
        placeholder="Choose tickers to overlay…",
        key="compare_tickers"
    )

with met_col:
    compare_metrics = st.multiselect(
        "Compare metrics (pick 1 or 2):",
        ALL_METRIC_OPTIONS,
        default=["Price (Close)"],
        max_selections=2,
        placeholder="Choose metrics…",
        key="compare_metrics",
        help=(
            "Price (Close) / Return % → time-series line chart.\n"
            "[M] columns → momentum time-series overlay.\n"
            "[F] columns → quarterly fundamentals as bars on secondary Y.\n"
            "All others → snapshot bar chart.  Pick 2 to compare."
        )
    )

if not compare_metrics:
    compare_metrics = ["Price (Close)"]

r = df_matrix[df_matrix['Ticker'] == selected].iloc[0]

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
@st.cache_data(ttl=120)
def get_price_history(tickers: tuple):
    ticker_sql = ", ".join(f"'{t}'" for t in tickers)
    query = f"""
        SELECT REPLACE(Ticker, '.NS', '') AS Ticker, Date, Close
        FROM ticker_prices
        WHERE REPLACE(Ticker, '.NS', '') IN ({ticker_sql})
        ORDER BY Date
    """
    try:
        return conn.execute(query).df()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=120)
def get_momentum_snapshot(tickers: tuple, col_raw: str):
    """Fetch the latest value of a momentum column for each ticker."""
    ticker_sql = ", ".join(f"'{t}'" for t in tickers)
    try:
        sql_stmt = f"""
            SELECT date,nse_symbol as ticker, {col_raw}
            FROM ticker_momentum
            WHERE nse_symbol IN ({ticker_sql})
            order by date asc
        """
        print ( sql_stmt)
        df = conn.execute(sql_stmt).df()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_fundamental_ts(tickers: tuple, view_label: str, metric: str):
    """Fetch (ticker, dt, val) for a fundamental metric across tickers."""
    view_name = next(
        (v for v, lbl in FUNDAMENTAL_VIEWS if lbl == view_label), None
    )
    if view_name is None:
        return pd.DataFrame()
    ticker_sql = ", ".join(f"'{t}'" for t in tickers)
    try:
        df = conn.execute(f"""
            SELECT ticker, dt, val
            FROM {view_name}
            WHERE metric = '{metric}'
              AND ticker IN ({ticker_sql})
            ORDER BY dt
        """).df()
        df['dt'] = pd.to_datetime(df['dt'])
        return df
    except Exception:
        return pd.DataFrame()


def _looks_pct(col_name: str) -> bool:
    return any(kw in col_name.lower() for kw in ['%', 'pct', 'ratio', 'score', 'dist', 'return'])


def _snap_bar_fig(metric: str, snapshot_tickers: list, key_suffix: str) -> go.Figure:
    """Build a grouped bar chart for a single snapshot metric."""
    is_momentum = metric.startswith("[M] ")
    col_raw     = metric[4:] if is_momentum else metric

    if is_momentum:
        raw = get_momentum_snapshot(tuple(snapshot_tickers), col_raw)
        if raw.empty:
            return None
        raw = raw.rename(columns={'ticker': 'Ticker', col_raw: metric})
    else:
        raw = df_matrix[df_matrix['Ticker'].isin(snapshot_tickers)][['Ticker', metric]].copy()

    raw = raw.dropna(subset=[metric])
    # preserve ticker order (primary first)
    ordered = [t for t in snapshot_tickers if t in raw['Ticker'].values]
    raw = raw.set_index('Ticker').reindex(ordered).reset_index()

    if raw.empty:
        return None

    fig = go.Figure()
    primary_val = raw[raw['Ticker'] == snapshot_tickers[0]][metric].values
    primary_val = primary_val[0] if len(primary_val) else None

    for i, row in raw.iterrows():
        tk, val = row['Ticker'], row[metric]
        is_primary = (tk == snapshot_tickers[0])
        fig.add_trace(go.Bar(
            x=[tk], y=[val],
            name=tk,
            marker_color=COLORS[i % len(COLORS)],
            # BUG FIX: use rgba(0,0,0,0) instead of "transparent"
            marker_line_color="#1e3a8a" if is_primary else "rgba(0,0,0,0)",
            marker_line_width=2.5 if is_primary else 0,
            text=[f"{val:,.2f}"],
            textposition="outside",
        ))

    tick_suffix = "%" if _looks_pct(metric) else ""
    fig.update_layout(
        height=300, margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=metric, font=dict(size=13)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=True,
                   zerolinecolor="#94a3b8", ticksuffix=tick_suffix, title=metric),
        bargap=0.35,
    )
    return fig, raw, primary_val, metric


def _snap_pills(raw_df, metric, primary_ticker, container_cols):
    primary_val = raw_df[raw_df['Ticker'] == primary_ticker][metric].values
    primary_val = primary_val[0] if len(primary_val) else None
    for i, row in raw_df.iterrows():
        tk, val = row['Ticker'], row[metric]
        if primary_val is not None and tk != primary_ticker:
            delta = val - primary_val
            c = "#16a34a" if delta >= 0 else "#dc2626"
            delta_str = f"<br><span style='color:{c};font-size:11px'>{delta:+.2f} vs {primary_ticker}</span>"
        else:
            delta_str = ""
        with container_cols[i]:
            st.markdown(
                f"<div style='text-align:center;padding:4px 0;'>"
                f"<b style='font-size:12px'>{tk}</b><br>"
                f"<span style='font-size:16px;font-weight:700'>{val:,.2f}</span>{delta_str}</div>",
                unsafe_allow_html=True
            )


# ==============================================================================
# CHART RENDERER
# ==============================================================================
all_tickers_to_plot = tuple([selected] + compare_tickers)
snapshot_tickers    = [selected] + compare_tickers

m1 = compare_metrics[0]
m2 = compare_metrics[1] if len(compare_metrics) > 1 else None

def _is_ts(metric):
    """True for metrics that are plotted as time-series lines/bars (not snapshot)."""
    return (metric in TIMESERIES_METRICS
            or (metric is not None and metric.startswith("[M] "))
            or (metric is not None and metric.startswith("[F] ")))

m1_is_ts = _is_ts(m1)
m2_is_ts = _is_ts(m2) if m2 else False

# ── CASE A: both time-series → dual y-axis line chart in period tabs ──────────
if m1_is_ts and (m2 is None or m2_is_ts):
    price_df = get_price_history(all_tickers_to_plot)
    if not price_df.empty:
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        period_tabs = st.tabs(["📅 5Y","📅 3Y","📅 2Y","📅 1Y","📅 6M","📅 3M","📅 1M","📅 7D"])
        period_map  = {0:(1825,), 1:(1095,), 2:(730,), 3:(365,), 4:(182,), 5:(91,), 6:(30,), 7:(7,)}
        tab_labels  = ["5Y","3Y","2Y","1Y","6M","3M","1M","7D"]

        for tab_idx, tab in enumerate(period_tabs):
            days  = period_map[tab_idx][0]
            label = tab_labels[tab_idx]
            with tab:
                cutoff = price_df['Date'].max() - pd.Timedelta(days=days)
                sliced = price_df[price_df['Date'] >= cutoff].copy()
                if sliced.empty:
                    st.info(f"No data for {label}.")
                    continue

                pivot = sliced.pivot_table(index='Date', columns='Ticker', values='Close').ffill()
                ticker_order = [t for t in ([selected] + compare_tickers) if t in pivot.columns]

                # Dual axis only when both metrics selected and both are different
                # [F] fundamentals always need secondary Y → force dual
                dual = (m2 is not None and m1 != m2) or (m1.startswith('[F] ') or (m2 is not None and m2.startswith('[F] ')))
                if dual:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                else:
                    fig = go.Figure()

                def _add_ts(metric, secondary=False):
                    palette = COLORS_PALE if secondary else COLORS
                    if metric == "Price (Close)":
                        for i, tk in enumerate(ticker_order):
                            s = pivot[tk].dropna()
                            trace = go.Scatter(
                                x=s.index, y=s.values, mode="lines",
                                name=f"{tk} Price",
                                line=dict(color=COLORS[i % len(COLORS)],
                                          width=2.5 if i==0 else 1.8,
                                          dash="solid" if i==0 else "dot"),
                                hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.2f}<extra>"+tk+"</extra>",
                                yaxis="y2" if (dual and secondary) else "y",
                            )
                            if dual:
                                fig.add_trace(trace, secondary_y=secondary)
                            else:
                                fig.add_trace(trace)
                        return "Price (₹)", ""
                    elif metric.startswith("[M] "):
                        # ── Momentum time-series overlay ──────────────────────
                        col_raw = metric[4:]
                        mom_df  = get_momentum_snapshot(all_tickers_to_plot, col_raw)
                        if mom_df.empty:
                            return col_raw, ""
                        mom_df['date'] = pd.to_datetime(mom_df['date'])
                        mom_sliced = mom_df[mom_df['date'] >= cutoff]
                        mom_pivot  = (
                            mom_sliced
                            .pivot_table(index='date', columns='ticker', values=col_raw)
                            .ffill()
                        )
                        tick_suffix = "%" if _looks_pct(col_raw) else ""
                        for i, tk in enumerate(ticker_order):
                            if tk not in mom_pivot.columns:
                                continue
                            s = mom_pivot[tk].dropna()
                            trace = go.Scatter(
                                x=s.index, y=s.values, mode="lines",
                                name=f"{tk} {col_raw}",
                                line=dict(color=palette[i % len(palette)],
                                          width=2.5 if i==0 else 1.8,
                                          dash="dot" if secondary else ("solid" if i==0 else "dot")),
                                hovertemplate=f"%{{x|%d %b %Y}}<br>%{{y:,.2f}}{tick_suffix}<extra>{tk}</extra>",
                                yaxis="y2" if (dual and secondary) else "y",
                            )
                            if dual:
                                fig.add_trace(trace, secondary_y=secondary)
                            else:
                                fig.add_trace(trace)
                        return col_raw, tick_suffix
                    elif metric.startswith("[F] "):
                        # ── Fundamental overlay — quarterly bars always on secondary Y ─
                        _, rest = metric.split(" ", 1)
                        view_label, fund_metric = rest.split("::", 1)
                        fund_df = get_fundamental_ts(
                            all_tickers_to_plot, view_label, fund_metric
                        )
                        if fund_df.empty:
                            return fund_metric, ""
                        fund_df = fund_df[fund_df['dt'] >= cutoff]
                        tick_suffix = "%" if _looks_pct(fund_metric) else ""
                        # 45-day bar width in milliseconds
                        bar_width_ms = 45 * 24 * 3600 * 1000
                        for i, tk in enumerate([selected] + compare_tickers):
                            tk_data = fund_df[fund_df['ticker'] == tk].dropna(subset=['val'])
                            if tk_data.empty:
                                continue
                            trace = go.Bar(
                                x=tk_data['dt'],
                                y=tk_data['val'],
                                name=f"{tk} {fund_metric}",
                                marker_color=COLORS_PALE[i % len(COLORS_PALE)],
                                marker_line=dict(color=COLORS[i % len(COLORS)], width=1),
                                opacity=0.75,
                                width=bar_width_ms,
                                hovertemplate=f"%{{x|%b %Y}}<br>%{{y:,.2f}}{tick_suffix}<extra>{tk} {fund_metric}</extra>",
                            )
                            # [F] always goes on secondary Y — force dual if not already
                            if dual:
                                fig.add_trace(trace, secondary_y=True)
                            else:
                                fig.add_trace(trace, secondary_y=True)
                        return fund_metric, tick_suffix
                    else:  # Return %
                        first_valid = pivot.bfill().iloc[0]
                        perf = ((pivot / first_valid) - 1) * 100
                        for i, tk in enumerate(ticker_order):
                            s = perf[tk].dropna()
                            last_ret = s.iloc[-1] if not s.empty else 0
                            trace = go.Scatter(
                                x=s.index, y=s.values, mode="lines",
                                name=f"{tk} Ret%",
                                line=dict(color=palette[i % 4],
                                          width=2.5 if i==0 else 1.8,
                                          dash="dot" if secondary else ("solid" if i==0 else "dot")),
                                hovertemplate="%{x|%d %b %Y}<br>%{y:+.2f}%<extra>"+tk+"</extra>",
                                yaxis="y2" if (dual and secondary) else "y",
                            )
                            if dual:
                                fig.add_trace(trace, secondary_y=secondary)
                            else:
                                fig.add_trace(trace)
                        return "Return (%)", "%"

                # Render [F] bars first so price line draws on top
                if dual and m2 is not None and m2.startswith("[F] "):
                    y2_title, y2_sfx = _add_ts(m2, secondary=True)
                    y1_title, y1_sfx = _add_ts(m1, secondary=False)
                else:
                    y1_title, y1_sfx = _add_ts(m1, secondary=False)
                    if dual:
                        y2_title, y2_sfx = _add_ts(m2, secondary=True)

                layout_kwargs = dict(
                    height=340, margin=dict(l=0, r=0, t=28, b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(showgrid=False, zeroline=False),
                    hovermode="x unified",
                )
                if dual:
                    fig.update_yaxes(title_text=y1_title, ticksuffix=y1_sfx,
                                     showgrid=True, gridcolor="#e2e8f0", secondary_y=False)
                    fig.update_yaxes(title_text=y2_title, ticksuffix=y2_sfx,
                                     showgrid=False, secondary_y=True)
                    fig.update_layout(**layout_kwargs)
                else:
                    layout_kwargs["yaxis"] = dict(showgrid=True, gridcolor="#e2e8f0",
                                                  zeroline=False, ticksuffix=y1_sfx, title=y1_title)
                    fig.update_layout(**layout_kwargs)

                if "Return" in m1:
                    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", line_width=1)

                st.plotly_chart(fig, use_container_width=True, key=f"ts_chart_{label}_{m1}_{m2}")
    else:
        st.warning("⚠️ Could not load price history.")

# ── CASE B: one or both snapshot metrics → side-by-side columns ──────────────
else:
    snapshot_metrics = compare_metrics  # could be [snap] or [ts, snap] or [snap, snap]

    # if one is time-series (incl. [M] and [F] metrics), render it first
    ts_metrics   = [m for m in compare_metrics if _is_ts(m)]
    snap_metrics = [m for m in compare_metrics if not _is_ts(m)]

    if ts_metrics:
        price_df = get_price_history(all_tickers_to_plot)
        if not price_df.empty:
            price_df['Date'] = pd.to_datetime(price_df['Date'])
        period_tabs = st.tabs(["📅 5Y","📅 3Y","📅 2Y","📅 1Y","📅 6M","📅 3M","📅 1M","📅 7D"])
        period_map  = {0:1825,1:1095,2:730,3:365,4:182,5:91,6:30,7:7}
        tab_labels  = ["5Y","3Y","2Y","1Y","6M","3M","1M","7D"]
        for tab_idx, tab in enumerate(period_tabs):
            days_b = period_map[tab_idx]
            with tab:
                fig = go.Figure()
                has_traces = False

                # ── Price / Return % ─────────────────────────────────────────
                price_ms = [m for m in ts_metrics if not m.startswith("[M] ") and not m.startswith("[F] ")]
                if price_ms and not price_df.empty:
                    cutoff = price_df['Date'].max() - pd.Timedelta(days=days_b)
                    sliced = price_df[price_df['Date'] >= cutoff].copy()
                    if not sliced.empty:
                        pivot = sliced.pivot_table(index='Date', columns='Ticker', values='Close').ffill()
                        ticker_order = [t for t in ([selected]+compare_tickers) if t in pivot.columns]
                        for metric in price_ms:
                            for i, tk in enumerate(ticker_order):
                                if metric == "Price (Close)":
                                    s = pivot[tk].dropna()
                                    fig.add_trace(go.Scatter(
                                        x=s.index, y=s.values, mode="lines",
                                        name=f"{tk} ₹",
                                        line=dict(color=COLORS[i%4], width=2.5 if i==0 else 1.8,
                                                  dash="solid" if i==0 else "dot"),
                                    ))
                                else:  # Return %
                                    first_valid = pivot.bfill().iloc[0]
                                    perf = ((pivot / first_valid) - 1) * 100
                                    s = perf[tk].dropna()
                                    last_ret = s.iloc[-1] if not s.empty else 0
                                    fig.add_trace(go.Scatter(
                                        x=s.index, y=s.values, mode="lines",
                                        name=f"{tk} ({last_ret:+.1f}%)",
                                        line=dict(color=COLORS[i%4], width=2.5 if i==0 else 1.8,
                                                  dash="solid" if i==0 else "dot"),
                                    ))
                                has_traces = True

                # ── [M] Momentum overlays ────────────────────────────────────
                for metric in [m for m in ts_metrics if m.startswith("[M] ")]:
                    col_raw = metric[4:]
                    mom_df  = get_momentum_snapshot(all_tickers_to_plot, col_raw)
                    if mom_df.empty:
                        continue
                    mom_df['date'] = pd.to_datetime(mom_df['date'])
                    m_cutoff = mom_df['date'].max() - pd.Timedelta(days=days_b)
                    mom_sliced = mom_df[mom_df['date'] >= m_cutoff]
                    mom_pivot  = mom_sliced.pivot_table(index='date', columns='ticker', values=col_raw).ffill()
                    tick_sfx = "%" if _looks_pct(col_raw) else ""
                    for i, tk in enumerate([selected]+compare_tickers):
                        if tk not in mom_pivot.columns:
                            continue
                        s = mom_pivot[tk].dropna()
                        fig.add_trace(go.Scatter(
                            x=s.index, y=s.values, mode="lines",
                            name=f"{tk} {col_raw}",
                            line=dict(color=COLORS[i%4], width=2.5 if i==0 else 1.8,
                                      dash="dot"),
                            hovertemplate=f"%{{x|%d %b %Y}}<br>%{{y:,.2f}}{tick_sfx}<extra>{tk}</extra>",
                        ))
                        has_traces = True

                # ── [F] Fundamental bar overlays ─────────────────────────────
                for metric in [m for m in ts_metrics if m.startswith("[F] ")]:
                    _, rest = metric.split(" ", 1)
                    view_label, fund_metric = rest.split("::", 1)
                    if not price_df.empty:
                        f_cutoff = price_df['Date'].max() - pd.Timedelta(days=days_b)
                    else:
                        f_cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_b)
                    fund_df = get_fundamental_ts(all_tickers_to_plot, view_label, fund_metric)
                    if fund_df.empty:
                        continue
                    fund_df = fund_df[fund_df['dt'] >= f_cutoff]
                    tick_sfx = "%" if _looks_pct(fund_metric) else ""
                    bar_width_ms = 45 * 24 * 3600 * 1000  # 45-day bar width in ms
                    for i, tk in enumerate([selected]+compare_tickers):
                        tk_data = fund_df[fund_df['ticker'] == tk].dropna(subset=['val'])
                        if tk_data.empty:
                            continue
                        fig.add_trace(go.Bar(
                            x=tk_data['dt'], y=tk_data['val'],
                            name=f"{tk} {fund_metric}",
                            marker_color=COLORS_PALE[i % len(COLORS_PALE)],
                            opacity=0.65,
                            width=bar_width_ms,
                            hovertemplate=f"%{{x|%b %Y}}<br>%{{y:,.2f}}{tick_sfx}<extra>{tk} {fund_metric}</extra>",
                            yaxis="y2",
                        ))
                        has_traces = True

                if not has_traces:
                    st.info("No data."); continue

                # if fundamentals present, use dual y-axis layout
                has_fund = any(m.startswith("[F] ") for m in ts_metrics)
                if has_fund:
                    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                    for tr in fig.data:
                        is_sec = getattr(tr, 'yaxis', 'y') == 'y2'
                        fig2.add_trace(tr, secondary_y=is_sec)
                    fig2.update_yaxes(title_text="Price / Value", showgrid=True,
                                      gridcolor="#e2e8f0", secondary_y=False)
                    fund_labels = [m.split("::", 1)[1] for m in ts_metrics if m.startswith("[F] ")]
                    fig2.update_yaxes(title_text=" / ".join(fund_labels),
                                      showgrid=False, secondary_y=True)
                    fig2.update_layout(
                        height=340, margin=dict(l=0,r=0,t=28,b=0),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
                        xaxis=dict(showgrid=False), hovermode="x unified",
                        barmode="group",
                    )
                    st.plotly_chart(fig2, use_container_width=True,
                                    key=f"ts_mix_{tab_labels[tab_idx]}")
                else:
                    fig.update_layout(
                        height=300, margin=dict(l=0,r=0,t=28,b=0),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
                        xaxis=dict(showgrid=False), hovermode="x unified",
                        yaxis=dict(showgrid=True,gridcolor="#e2e8f0"),
                    )
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"ts_mix_{tab_labels[tab_idx]}")

    # ── Snapshot bar charts side-by-side ─────────────────────────────────────
    if snap_metrics:
        if len(snap_metrics) == 2:
            bar_cols = st.columns(2)
        else:
            bar_cols = [st.container()]   # full width

        for col_idx, metric in enumerate(snap_metrics):
            container = bar_cols[col_idx] if len(snap_metrics) == 2 else bar_cols[0]
            with container:
                is_momentum = metric.startswith("[M] ")
                col_raw     = metric[4:] if is_momentum else metric

                if is_momentum:
                    raw = get_momentum_snapshot(tuple(snapshot_tickers), col_raw)
                    if not raw.empty:
                        raw = raw.rename(columns={'ticker': 'Ticker', col_raw: metric})
                else:
                    raw = df_matrix[df_matrix['Ticker'].isin(snapshot_tickers)][['Ticker', metric]].copy()

                raw = raw.dropna(subset=[metric]) if not raw.empty else raw
                if raw.empty:
                    st.warning(f"No data for **{metric}**.")
                    continue

                ordered = [t for t in snapshot_tickers if t in raw['Ticker'].values]
                raw = raw.set_index('Ticker').reindex(ordered).reset_index()

                fig = go.Figure()
                for i, row in raw.iterrows():
                    tk, val = row['Ticker'], row[metric]
                    is_primary = (tk == selected)
                    fig.add_trace(go.Bar(
                        x=[tk], y=[val], name=tk,
                        marker_color=COLORS[i % len(COLORS)],
                        marker_line_color="#1e3a8a" if is_primary else "rgba(0,0,0,0)",
                        marker_line_width=2.5 if is_primary else 0,
                        text=[f"{val:,.2f}"], textposition="outside",
                    ))

                tick_suffix = "%" if _looks_pct(metric) else ""
                fig.update_layout(
                    height=300, margin=dict(l=0, r=0, t=30, b=0),
                    title=dict(text=metric, font=dict(size=13)),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="#e2e8f0", zeroline=True,
                               zerolinecolor="#94a3b8", ticksuffix=tick_suffix, title=metric),
                    bargap=0.35,
                )
                st.plotly_chart(fig, use_container_width=True,
                                key=f"snap_bar_{metric}_{col_idx}")

                # pills
                primary_val = raw[raw['Ticker'] == selected][metric].values
                primary_val = primary_val[0] if len(primary_val) else None
                pill_cols = st.columns(len(raw))
                for i, row in raw.iterrows():
                    tk, val = row['Ticker'], row[metric]
                    if primary_val is not None and tk != selected:
                        delta = val - primary_val
                        c = "#16a34a" if delta >= 0 else "#dc2626"
                        delta_str = f"<br><span style='color:{c};font-size:11px'>{delta:+.2f} vs {selected}</span>"
                    else:
                        delta_str = ""
                    with pill_cols[i]:
                        st.markdown(
                            f"<div style='text-align:center;padding:4px 0;'>"
                            f"<b style='font-size:12px'>{tk}</b><br>"
                            f"<span style='font-size:16px;font-weight:700'>{val:,.2f}</span>{delta_str}</div>",
                            unsafe_allow_html=True
                        )

# ── Company context strip ──
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown(
    f"**{r['Company']}** &nbsp;|&nbsp; Sector: `{r['Sector']}` &nbsp;|&nbsp; "
    f"Stock P/E: `{r['Stock P/E']}` &nbsp;|&nbsp; Sector P/E: `{r['Sector P/E']}` &nbsp;|&nbsp; "
    f"PE Discount: `{r['PE Discount %']:+.1f}%` &nbsp;|&nbsp; "
    f"ROCE: `{r['ROCE %']}%` &nbsp;|&nbsp; D/E: `{r['D/E']}` &nbsp;|&nbsp; "
    f"Promoter: `{r['Promoter %']}%`",
    unsafe_allow_html=True
)

def gate_icon(pos, neg):
    if pos:  return "🟩 PASS"
    if neg:  return "🟥 FAIL"
    return "⬜ —"

col_q, col_v, col_t, col_tech = st.columns(4)

with col_q:
    st.markdown("**🏛 Gate 1 — Business Quality**")
    prose = lambda lbl, p, n, note: st.markdown(
        f"<div class='gate-block'><b>{lbl}</b><br>{gate_icon(p,n)}<br><small>{note}</small></div>",
        unsafe_allow_html=True
    )
    prose("1a. ROCE",         r['G1a(+) ROCE'],     r['G1a(-) ROCE'],
          f"ROCE = {r['ROCE %']}% (target ≥18%)")
    prose("1b. Free Cash Flow",r['G1b(+) FCF'],     r['G1b(-) FCF'],
          f"FCF = {'positive' if r['G1b(+) FCF'] else 'negative/unknown'}")
    prose("1c. Profit CAGR",  r['G1c(+) ProfCAGR'],r['G1c(-) ProfCAGR'],
          f"3Y CAGR = {r['3Y Profit CAGR%']}% (target ≥15%)")
    prose("1d. CFO vs NP",    r['G1d(+) CFO>NP'],  r['G1d(-) CFO<NP'],
          "CFO > Net Profit = earnings are real cash")
    _de_note = "N/A for financial sector" if any(fs.lower() in str(r['Sector']).lower() for fs in ['bank','finance','nbfc','insurance']) else f"D/E = {r['D/E']} (target <0.5)"
    prose("1e. Debt/Equity",  r['G1e(+) LowDebt'], r['G1e(-) HiDebt'], _de_note)

with col_v:
    st.markdown("**💰 Gate 2 — Valuation**")
    prose("2a. P/E vs Sector",r['G2a(+) PE<Sector'],r['G2a(-) PE>Sector'],
          f"Stock P/E {r['Stock P/E']} vs Sector {r['Sector P/E']} → {r['PE Discount %']:+.1f}%")
    prose("2b. PEG Ratio",    r['G2b(+) PEG<1'],   r['G2b(-) PEG>2.5'],
          f"PEG = {r['PEG']} (target <1.0 = cheap for growth)")
    prose("2c. Price/Book",   r['G2c(+) PB<3'],    r['G2c(-) PB>8'],
          "P/B < 3 = not egregiously expensive")

with col_t:
    st.markdown("**📈 Gate 3 — Timing**")
    prose("3a. Sales Momentum", r['G3a(+) SalesAcc'], r['G3a(-) SalesDec'],
          f"Q0={r['_sales_q0']}Cr → Q1={r['_sales_q1']}Cr")
    prose("3b. Profit Accel",   r['G3b(+) ProfAcc'],  r['G3b(-) ProfDec'],  "Quarterly net profit trend")
    prose("3c. OPM Expansion",  r['G3c(+) OPMup'],    r['G3c(-) OPMsqz'],
          f"OPM: {r['_opm_q0']}% vs {r['_opm_q1']}%")
    prose("3d. Promoter Buying",r['G3d(+) Promoter'], r['G3d(-) PromoSell'],
          f"{r['_promoter']}% now vs {r['_promoter_prev']}% prev")
    prose("3e. Inst. Accum.",   r['G3e(+) InstAcc'],  r['G3e(-) InstExit'],
          f"FII {r['_fii']}%→{r['_fii_prev']}% | DII {r['_dii']}%→{r['_dii_prev']}%")

with col_tech:
    st.markdown("**📊 Gate 4 — Technical**")
    prose("4a. BB Squeeze",    r['G4a(+) BBsqz'],   r['G4a(-) BBchaos'],
          f"BB width={round(r['_bb_width'],3) if pd.notna(r['_bb_width']) else '—'} (min63d={round(r['_min_bb_63d'],3) if pd.notna(r['_min_bb_63d']) else '—'})")
    prose("4b. MA50 Zone",     r['G4b(+) MA50zone'],r['G4b(-) MA50break'],
          f"Dist from 50DMA = {round(r['_dist_ma50'],1) if pd.notna(r['_dist_ma50']) else '—'}%")
    label_4c = "Dip Window (10–40% off high)" if dip_mode else "Near 52W High"
    prose(f"4c. {label_4c}",   r['G4c(+) Dip/High'],r['G4c(-) Crash'],
          f"From 52W high = {round(r['_pct_52w'],1) if pd.notna(r['_pct_52w']) else '—'}%")
    prose("4d. Volume Conf.",  r['G4d(+) VolConf'], r['G4d(-) VolDry'],
          f"Vol ratio = {round(r['_vol_ratio'],2) if pd.notna(r['_vol_ratio']) else '—'}x")

# ==============================================================================
# 10. GATE LOGIC REFERENCE
# ==============================================================================
with st.expander("📖 Gate Logic Reference — How All Four Scores Are Computed"):
    st.markdown(GATE_LOGIC_MD)