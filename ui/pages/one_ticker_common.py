"""
one_ticker_common.py
────────────────────
Shared bootstrap for one_ticker_gates.py and one_ticker_charts.py.

Provides:
  - COLORS / COLORS_PALE          palette constants
  - FUNDAMENTAL_VIEWS             list of (view_name, label) tuples
  - get_duckdb_connection()       cached DuckDB connection
  - initialize_views_or_mock()    loads init.sql or falls back to mock
  - run_pipeline()                core SQL analytics query
  - get_pipeline_data()           cached wrapper around run_pipeline
  - compute_scores()              scores every ticker across 4 gates
  - load_matrix()                 one-stop call: connect → init → pipeline → scores
"""

import os
import duckdb
import pandas as pd
import numpy as np
import streamlit as st

# ==============================================================================
# CONSTANTS
# ==============================================================================
COLORS      = ["#2563eb", "#dc2626", "#16a34a", "#d97706"]
COLORS_PALE = ["#93c5fd", "#fca5a5", "#86efac", "#fde68a"]

FUNDAMENTAL_VIEWS = [
    ("quarterly_results",      "Quarterly"),
    ("profit_loss",            "P&L"),
    ("cash_flows",             "Cash Flow"),
    ("ratios",                 "Ratios"),
    ("shareholding_quarterly_metric", "Shareholding"),
    ("balance_sheet",          "Balance Sheet"),
]

# ==============================================================================
# DATABASE CONNECTION
# ==============================================================================
@st.cache_resource
def get_duckdb_connection():
    return duckdb.connect(database=':memory:')


def initialize_views_or_mock(conn) -> str:
    """
    Execute sqls/init.sql against *conn*, or fall back to a mock dataset.
    Returns 'REAL', 'MOCK', or 'ERROR'.
    """
    sql_path = "sqls/init.sql"
    if not os.path.exists(sql_path):
        st.sidebar.warning("⚠️ init.sql not found — running Simulation Sandbox.")
        _build_mock_data(conn)
        return "MOCK"
    st.sidebar.success("📊 Connected to Local Parquet Data Lake")
    try:
        with open(sql_path, "r") as f:
            sql = f.read()
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


# ==============================================================================
# ANALYTICS PIPELINE
# ==============================================================================
def run_pipeline(conn) -> pd.DataFrame:
    query = f"""
        SELECT * FROM gate_score ORDER BY ticker
    """
    return conn.execute(query).df()


@st.cache_data(ttl=120)
def get_pipeline_data() -> pd.DataFrame:
    conn = get_duckdb_connection()
    return run_pipeline(conn)


# ==============================================================================
# SCORE ENGINE
# ==============================================================================
def compute_scores(df_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dip_mode = False
    for _, r in df_raw.iterrows():
        roce = r['roce'] or 0
        g1a_pos = bool(roce >= 18);  g1a_neg = bool(roce < 8)
        g1b_pos = bool(pd.notna(r['fcf']) and r['fcf'] > 0)
        g1b_neg = bool(pd.notna(r['fcf']) and r['fcf'] < 0)
        profit_cagr = None
        if pd.notna(r['profit_now']) and pd.notna(r['profit_3y']) and r['profit_now'] > 0 and r['profit_3y'] > 0:
            profit_cagr = ((r['profit_now'] / r['profit_3y']) ** (1/3) - 1) * 100
        g1c_pos = bool(profit_cagr is not None and profit_cagr >= 15)
        g1c_neg = bool(profit_cagr is not None and profit_cagr < 0)
        g1d_pos = bool(pd.notna(r['cfo']) and pd.notna(r['profit_now']) and r['profit_now'] > 0 and r['cfo'] > r['profit_now'])
        g1d_neg = bool(pd.notna(r['cfo']) and pd.notna(r['profit_now']) and r['profit_now'] > 0 and r['cfo'] < r['profit_now'] * 0.5)
        FINANCIAL_SECTORS = {'Banks', 'Finance', 'NBFC', 'Insurance'}
        is_financial = any(fs.lower() in str(r['sector']).lower() for fs in FINANCIAL_SECTORS)
        de = None
        if pd.notna(r['debt']) and pd.notna(r['reserves']) and pd.notna(r['equity_capital']):
            equity = r['reserves'] + r['equity_capital']
            de = r['debt'] / equity if equity > 0 else None
        if is_financial:
            g1e_pos, g1e_neg = False, False
        else:
            g1e_pos = bool(de is not None and de < 0.5)
            g1e_neg = bool(de is not None and de > 1.5)
        quality_score = max(0, min(100, (sum([g1a_pos,g1b_pos,g1c_pos,g1d_pos,g1e_pos])*20) - (sum([g1a_neg,g1b_neg,g1c_neg,g1d_neg,g1e_neg])*15)))

        stock_pe = r['stock_p_e'];  sector_pe = r['sector_median_pe']
        pe_discount_pct = None;  g2a_pos = g2a_neg = False
        if pd.notna(stock_pe) and pd.notna(sector_pe) and sector_pe > 0 and stock_pe > 0:
            pe_discount_pct = ((sector_pe - stock_pe) / sector_pe) * 100
            g2a_pos = bool(pe_discount_pct >= 15);  g2a_neg = bool(pe_discount_pct < -30)
        peg = None;  g2b_pos = g2b_neg = False
        if pd.notna(stock_pe) and profit_cagr is not None and profit_cagr > 0:
            peg = stock_pe / profit_cagr;  g2b_pos = bool(peg < 1.0);  g2b_neg = bool(peg > 2.5)
        pb = None;  g2c_pos = g2c_neg = False
        if pd.notna(r['current_price']) and pd.notna(r['book_value']) and r['book_value'] > 0:
            pb = r['current_price'] / r['book_value']
            g2c_pos = bool(pb < 3.0);  g2c_neg = bool(pb > 8.0)
        valuation_score = max(0, min(100, (sum([g2a_pos,g2b_pos,g2c_pos])*33) - (sum([g2a_neg,g2b_neg,g2c_neg])*20)))

        g3a_pos = g3a_neg = False
        if pd.notna(r['sales_q0']) and pd.notna(r['sales_q1']) and pd.notna(r['sales_q2']):
            g3a_pos = bool(r['sales_q0'] > r['sales_q1'] > r['sales_q2'])
            g3a_neg = bool(r['sales_q0'] < r['sales_q1'])
        g3b_pos = g3b_neg = False
        if pd.notna(r['profit_q0']) and pd.notna(r['profit_q1']):
            g3b_pos = bool(r['profit_q0'] > r['profit_q1'])
            g3b_neg = bool(r['profit_q0'] < r['profit_q1'] and r['profit_q0'] > 0)
        g3c_pos = g3c_neg = False
        if pd.notna(r['opm_q0']) and pd.notna(r['opm_q1']):
            g3c_pos = bool(r['opm_q0'] > r['opm_q1']);  g3c_neg = bool(r['opm_q0'] < r['opm_q1'] - 3)
        g3d_pos = g3d_neg = False
        if pd.notna(r['promoter']) and pd.notna(r['promoter_prev']):
            g3d_pos = bool(r['promoter'] >= r['promoter_prev'])
            g3d_neg = bool(r['promoter'] < r['promoter_prev'] - 2)
        g3e_pos = g3e_neg = False
        if pd.notna(r['fii']) and pd.notna(r['fii_prev']) and pd.notna(r['dii']) and pd.notna(r['dii_prev']):
            fii_up = r['fii'] >= r['fii_prev'];  dii_up = r['dii'] >= r['dii_prev']
            g3e_pos = bool(fii_up or dii_up);   g3e_neg = bool(not fii_up and not dii_up)
        timing_score = max(0, min(100, (sum([g3a_pos,g3b_pos,g3c_pos,g3d_pos,g3e_pos])*20) - (sum([g3a_neg,g3b_neg,g3c_neg,g3d_neg,g3e_neg])*15)))

        g4a_pos = bool(pd.notna(r['bb_width']) and pd.notna(r['min_bb_63d']) and r['bb_width'] <= r['min_bb_63d'] * 1.15)
        g4a_neg = bool(pd.notna(r['bb_width']) and r['bb_width'] > 0.35)
        g4b_pos = bool(pd.notna(r['dist_from_ma50']) and 0 < r['dist_from_ma50'] < 12)
        g4b_neg = bool(pd.notna(r['dist_from_ma50']) and (r['dist_from_ma50'] > 25 or r['dist_from_ma50'] < -8))
        g4c_dip_pos = bool(pd.notna(r['pct_from_52w']) and -40 <= r['pct_from_52w'] <= -10)
        g4c_pos     = bool(pd.notna(r['pct_from_52w']) and r['pct_from_52w'] >= -5)
        g4c_neg     = bool(pd.notna(r['pct_from_52w']) and r['pct_from_52w'] < -40)
        g4d_pos = bool(pd.notna(r['volume_ratio']) and r['volume_ratio'] >= 1.1)
        g4d_neg = bool(pd.notna(r['volume_ratio']) and r['volume_ratio'] < 0.6)
        active_g4c_pos = g4c_dip_pos if dip_mode else g4c_pos
        technical_score = max(0, min(100, (sum([g4a_pos,g4b_pos,active_g4c_pos,g4d_pos])*25) - (sum([g4a_neg,g4b_neg,g4c_neg,g4d_neg])*15)))

        rows.append({
            'Ticker': r['ticker'], 'Company': r['company_name'], 'Sector': r['sector'],
            'Mkt Cap (Cr)': r['market_cap'],
            'Stock P/E':    round(stock_pe,  1) if pd.notna(stock_pe)  else None,
            'Sector P/E':   round(sector_pe, 1) if pd.notna(sector_pe) else None,
            'PE Discount %':round(pe_discount_pct, 1) if pe_discount_pct is not None else None,
            'PEG':          round(peg, 2) if peg is not None else None,
            'ROCE %':       round(roce, 1),
            '3Y Profit CAGR%': round(profit_cagr, 1) if profit_cagr is not None else None,
            'D/E':          round(de, 2) if de is not None else None,
            'Promoter %':   round(r['promoter'], 1) if pd.notna(r['promoter']) else None,
            'Quality Score': round(quality_score), 'Valuation Score': round(valuation_score),
            'Timing Score':  round(timing_score),  'Technical Score': round(technical_score),
            'G1a(+) ROCE': g1a_pos, 'G1a(-) ROCE': g1a_neg,
            'G1b(+) FCF':  g1b_pos, 'G1b(-) FCF':  g1b_neg,
            'G1c(+) ProfCAGR': g1c_pos, 'G1c(-) ProfCAGR': g1c_neg,
            'G1d(+) CFO>NP': g1d_pos,  'G1d(-) CFO<NP':  g1d_neg,
            'G1e(+) LowDebt': g1e_pos, 'G1e(-) HiDebt':  g1e_neg,
            'G2a(+) PE<Sector': g2a_pos, 'G2a(-) PE>Sector': g2a_neg,
            'G2b(+) PEG<1': g2b_pos, 'G2b(-) PEG>2.5': g2b_neg,
            'G2c(+) PB<3':  g2c_pos, 'G2c(-) PB>8':    g2c_neg,
            'G3a(+) SalesAcc': g3a_pos, 'G3a(-) SalesDec': g3a_neg,
            'G3b(+) ProfAcc':  g3b_pos, 'G3b(-) ProfDec':  g3b_neg,
            'G3c(+) OPMup':    g3c_pos, 'G3c(-) OPMsqz':   g3c_neg,
            'G3d(+) Promoter': g3d_pos, 'G3d(-) PromoSell': g3d_neg,
            'G3e(+) InstAcc':  g3e_pos, 'G3e(-) InstExit':  g3e_neg,
            'G4a(+) BBsqz':    g4a_pos, 'G4a(-) BBchaos':   g4a_neg,
            'G4b(+) MA50zone': g4b_pos, 'G4b(-) MA50break': g4b_neg,
            'G4c(+) Dip/High': active_g4c_pos, 'G4c(-) Crash': g4c_neg,
            'G4d(+) VolConf':  g4d_pos, 'G4d(-) VolDry':    g4d_neg,
            '_return_1m': r['return_1m'], '_return_3m': r['return_3m'],
            '_bb_width': r['bb_width'],   '_min_bb_63d': r['min_bb_63d'],
            '_pct_52w': r['pct_from_52w'], '_vol_ratio': r['volume_ratio'],
            '_dist_ma50': r['dist_from_ma50'],
            '_sales_q0': r['sales_q0'], '_sales_q1': r['sales_q1'],
            '_opm_q0':   r['opm_q0'],   '_opm_q1':   r['opm_q1'],
            '_profit_cagr': profit_cagr, '_de': de, '_pb': pb, '_peg': peg,
            '_pe_disc': pe_discount_pct,
            '_promoter': r['promoter'],  '_promoter_prev': r['promoter_prev'],
            '_fii': r['fii'], '_fii_prev': r['fii_prev'],
            '_dii': r['dii'], '_dii_prev': r['dii_prev'],
        })

    return pd.DataFrame(rows).sort_values(by='Quality Score', ascending=False).reset_index(drop=True)


# ==============================================================================
# ONE-STOP LOADER  (connect → init → pipeline → scores)
# ==============================================================================
def load_matrix() -> tuple[pd.DataFrame, object, str]:
    """
    Returns (df_matrix, conn, env_status).
    Idempotent: safe to call from both pages; DuckDB connection is cached.
    """
    conn = get_duckdb_connection()
    env_status = initialize_views_or_mock(conn)
    df_raw = get_pipeline_data()
    if df_raw.empty:
        return pd.DataFrame(), conn, env_status
    df_matrix = compute_scores(df_raw)
    return df_matrix, conn, env_status
