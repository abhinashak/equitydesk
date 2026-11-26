"""Delivery transaction-cost calculations for FASTA.

Pure dataframe transformation; no Streamlit or database dependencies.
"""
from __future__ import annotations

import pandas as pd


def apply_transaction_costs(trades_df, p):
    """Apply delivery transaction costs. Uses each trade's `position_capital` (set by allocate_capital);
    without it, each week's pool is split equally among that week's trades."""
    if trades_df.empty:
        return trades_df

    df = trades_df.copy()
    df['buy_date'] = pd.to_datetime(df['buy_date'])
    df['week_period'] = df['buy_date'].dt.to_period('W')
    if 'position_capital' not in df.columns:
        weekly_trade_count = df.groupby('week_period')['ticker'].transform('count')
        df['position_capital'] = p.weekly_capital / weekly_trade_count
    buy_value = df['position_capital'].astype(float)
    sell_value = buy_value * (1.0 + df['pnl_pct'].astype(float) / 100.0)
    sell_value = sell_value.clip(lower=0.0)

    brokerage_buy = buy_value * p.brokerage_rate
    brokerage_sell = sell_value * p.brokerage_rate
    if p.brokerage_cap > 0:
        brokerage_buy = brokerage_buy.clip(upper=p.brokerage_cap)
        brokerage_sell = brokerage_sell.clip(upper=p.brokerage_cap)

    stt_buy = buy_value * p.stt_buy_rate
    stt_sell = sell_value * p.stt_sell_rate
    exchange_buy = buy_value * p.exchange_txn_rate
    exchange_sell = sell_value * p.exchange_txn_rate
    sebi_buy = buy_value * p.sebi_turnover_rate
    sebi_sell = sell_value * p.sebi_turnover_rate
    stamp_buy = buy_value * p.stamp_duty_buy_rate
    stamp_sell = 0.0
    gst_buy = (brokerage_buy + exchange_buy + sebi_buy) * p.gst_rate
    gst_sell = (brokerage_sell + exchange_sell + sebi_sell) * p.gst_rate
    dp_sell = p.dp_charge_per_sell

    total_buy_cost = stt_buy + exchange_buy + sebi_buy + stamp_buy + gst_buy + brokerage_buy
    total_sell_cost = stt_sell + exchange_sell + sebi_sell + stamp_sell + gst_sell + dp_sell + brokerage_sell
    total_cost = total_buy_cost + total_sell_cost
    gross_profit_rupees = sell_value - buy_value
    net_profit_rupees = gross_profit_rupees - total_cost

    df['gross_profit_rupees'] = gross_profit_rupees
    df['buy_stt'] = stt_buy
    df['sell_stt'] = stt_sell
    df['buy_exchange_fee'] = exchange_buy
    df['sell_exchange_fee'] = exchange_sell
    df['buy_sebi_fee'] = sebi_buy
    df['sell_sebi_fee'] = sebi_sell
    df['buy_stamp_duty'] = stamp_buy
    df['sell_stamp_duty'] = stamp_sell
    df['buy_gst'] = gst_buy
    df['sell_gst'] = gst_sell
    df['sell_dp_charge'] = dp_sell
    df['buy_brokerage'] = brokerage_buy
    df['sell_brokerage'] = brokerage_sell
    df['total_buy_cost'] = total_buy_cost
    df['total_sell_cost'] = total_sell_cost
    df['total_transaction_cost'] = total_cost
    df['net_profit_rupees'] = net_profit_rupees
    df['net_pnl_pct'] = (net_profit_rupees / buy_value) * 100.0
    df['buy_date'] = df['buy_date'].dt.strftime('%Y-%m-%d')
    df = df.drop(columns=['week_period'])
    return df

