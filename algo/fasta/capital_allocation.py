"""Point-in-time strategy and pick capital allocation for FASTA.

Allocation uses only trades from weeks strictly before the allocation week.
No Streamlit or database dependencies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


_WEEK0 = pd.Timestamp("1970-01-05")
_NO_TRADES = (np.array([], dtype=np.int64), np.array([], dtype=float))

def _normalize_market_cap(value):
    if pd.isna(value):
        return ""
    value = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    return value


def _apply_market_cap_ticker_caps(df, amount_col, capital, market_cap_caps_pct=None):
    """Apply per-segment caps per ticker and redistribute excess without breaching caps.

    `market_cap_caps_pct` contains percentage points (10.0 means 10%). Missing or
    zero segment limits are uncapped. Unknown market-cap classifications are also
    left uncapped rather than being silently assigned to a segment.
    """
    if df.empty or amount_col not in df or 'ticker' not in df:
        return df
    # The cap keys come from the database's current distinct market_cap values.
    # There are deliberately no built-in segment names or default percentages.
    caps = {
        _normalize_market_cap(k): float(v)
        for k, v in (market_cap_caps_pct or {}).items()
    }

    original = pd.to_numeric(df[amount_col], errors='coerce').fillna(0.0).clip(lower=0.0)
    ticker_base = original.groupby(df['ticker']).sum()
    if ticker_base.empty:
        df[amount_col] = original
        return df

    if 'market_cap' in df:
        ticker_segment = df.assign(_segment=df['market_cap'].map(_normalize_market_cap)).groupby('ticker')['_segment'].first()
    else:
        ticker_segment = pd.Series('', index=ticker_base.index)
    cap_by_ticker = ticker_segment.map(lambda segment: max(caps.get(segment, 0.0), 0.0) / 100.0 * max(float(capital), 0.0))
    # A zero/missing segment percentage means no cap (infinite allocation room).
    cap_by_ticker = cap_by_ticker.where(cap_by_ticker > 0, np.inf)

    ticker_alloc = pd.concat([ticker_base.rename('base'), cap_by_ticker.rename('cap')], axis=1)
    ticker_alloc['allocated'] = ticker_alloc[['base', 'cap']].min(axis=1)
    remaining = max(float(original.sum()) - float(ticker_alloc['allocated'].sum()), 0.0)
    active = ticker_alloc.index[ticker_alloc['allocated'] < ticker_alloc['cap'] - 1e-9]

    while remaining > 1e-8 and len(active):
        weights = ticker_base.loc[active].clip(lower=0.0)
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            weights = pd.Series(1.0, index=active)
            weight_sum = float(len(active))
        proposed = remaining * weights / weight_sum
        room = (ticker_alloc.loc[active, 'cap'] - ticker_alloc.loc[active, 'allocated']).clip(lower=0.0)
        additions = pd.concat([proposed, room], axis=1).min(axis=1)
        added = float(additions.sum())
        ticker_alloc.loc[active, 'allocated'] += additions
        remaining -= added
        active = ticker_alloc.index[ticker_alloc['allocated'] < ticker_alloc['cap'] - 1e-9]
        if added <= 1e-10:
            break

    ticker_totals = original.groupby(df['ticker']).transform('sum').replace(0.0, np.nan)
    row_share = (original / ticker_totals).fillna(0.0)
    capped_totals = df['ticker'].map(ticker_alloc['allocated'])
    df[amount_col] = capped_totals.to_numpy(dtype=float) * row_share.to_numpy(dtype=float)
    return df


def _market_cap_caps(p):
    """Return user-configured limits keyed by the database market_cap values."""
    return dict(getattr(p, "market_cap_allocation_pct", {}) or {})


def _week_no(ts):
    return (pd.Timestamp(ts).normalize() - _WEEK0).days // 7


def _week_nos(dates):
    return ((pd.to_datetime(dates).dt.normalize() - _WEEK0).dt.days // 7).astype(np.int64)


def _track_record(trades_df):
    if trades_df.empty:
        return {}
    weeks = _week_nos(trades_df['buy_date']).to_numpy()
    wins = (trades_df['pnl_pct'].to_numpy(dtype=float) > 0.0).astype(float)
    record = {}
    for key in trades_df['strategy'].unique():
        mask = (trades_df['strategy'] == key).to_numpy()
        record[key] = (weeks[mask], wins[mask])
    return record


def recency_win_ratio(weeks, wins, as_of, p):
    past = weeks < as_of
    w = p.allocation_decay ** (as_of - weeks[past])
    denom = float(w.sum()) + p.win_ratio_prior
    if denom <= 0:
        return 0.5
    return (float((w * wins[past]).sum()) + 0.5 * p.win_ratio_prior) / denom


def split_capital(win_ratios, active):
    total = sum(win_ratios[k] for k in active)
    return {k: (win_ratios[k] / total if total > 0 else 1.0 / len(active)) for k in active}


def allocate_capital(trades_df, p):
    if trades_df.empty:
        return trades_df, pd.DataFrame()
    df = trades_df.copy()
    df['_wk'] = _week_nos(df['buy_date'])
    record = _track_record(trades_df)
    keys = sorted(record)

    rows = []
    for wk, g in df.groupby('_wk'):
        ratios = {k: recency_win_ratio(*record[k], wk, p) for k in keys}
        weights = split_capital(ratios, sorted(g['strategy'].unique()))
        rows += [{'_wk': wk, 'strategy': k, 'win_ratio': ratios[k], 'weight': weights.get(k, 0.0)} for k in keys]
    alloc = pd.DataFrame(rows)

    df = df.merge(alloc.rename(columns={'win_ratio': 'strategy_win_ratio', 'weight': 'strategy_weight'}),
                  on=['_wk', 'strategy'], how='left')
    per_strategy_count = df.groupby(['_wk', 'strategy'])['ticker'].transform('count')
    df['position_capital'] = p.weekly_capital * df['strategy_weight'] / per_strategy_count
    # Cap the combined ticker exposure across strategies for each trading week.
    for _, week_idx in df.groupby('_wk').groups.items():
        week_df = df.loc[week_idx].copy()
        df.loc[week_idx, 'position_capital'] = _apply_market_cap_ticker_caps(
            week_df, 'position_capital', p.weekly_capital, _market_cap_caps(p)
        )['position_capital'].to_numpy()
    alloc['week'] = _WEEK0 + pd.to_timedelta(alloc['_wk'] * 7, unit='D')
    return df.drop(columns='_wk'), alloc.drop(columns='_wk')


def allocate_picks(picks_df, win_ratios, p):
    df = picks_df.copy()
    weights = split_capital(win_ratios, sorted(df['strategy'].unique()))
    df['strategy_win_ratio'] = df['strategy'].map(win_ratios)
    df['strategy_weight'] = df['strategy'].map(weights)
    df['allocation'] = p.weekly_capital * df['strategy_weight'] / df.groupby('strategy')['ticker'].transform('count')
    df = _apply_market_cap_ticker_caps(df, 'allocation', p.weekly_capital, _market_cap_caps(p))
    return df

