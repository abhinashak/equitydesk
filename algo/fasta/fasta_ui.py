"""FASTA ALGO - weekly rotation back-test, rendered as a Streamlit page.
# Advance days by x days.

"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .capital_allocation import (
    _NO_TRADES, _track_record, _week_no, _normalize_market_cap, _apply_market_cap_ticker_caps, allocate_capital, allocate_picks,
    recency_win_ratio, split_capital,
)
from .transaction_costs import apply_transaction_costs

# Shared infrastructure: one process-wide DuckDB database and common JSON persistence.
from utils.db_utils import get_connection, reset_connection
from utils.file_utils import load_json_dict, save_json_atomic
from utils.yahoo_rss_news import show_news
from algo.fasta import fasta_sql as _FASTA_SQL
from algo.fasta import fasta_css

# ==========================================
# STATIC CONFIGURATION (paths)
# ==========================================
OUTPUT_FILE = "outputs/fasta/fasta.txt"
CACHE_DIR = Path("outputs/fasta")
SETTINGS_FILE = CACHE_DIR / "fasta_settings.json"  # remembered UI settings (not touched by "Clear cached data")
RESULT_KEY = "fasta_result_v2"  # v2: result layout changed with multi-strategy (ignores stale sessions)

# `use_container_width` is being replaced by `width="stretch"` in newer Streamlit.
try:
    _ver = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _ver = (0, 0)
_W = {"width": "stretch"} if _ver >= (1, 49) else {"use_container_width": True}


# ==========================================
# PARAMETERS
# ==========================================
@dataclass(frozen=True)
class Params:
    # Data window
    start_date: str = "2017-04-01"
    cutoff_date: str = ""
    restrict_tickers: tuple = ()      # bare upper-case symbols; empty = analyse the whole market universe
    exclude_tickers: tuple = ()       # bare upper-case symbols never traded or picked (applied after restrict_tickers)

    # Universe inclusion filters (AND-ed; read from the `tickers` table). Unset = no filtering on that field.
    market_caps: tuple = ()           # selected `market_cap` buckets; empty = ALL (no filter, NULL rows included)
    min_market_cap_amt: float | None = None   # keep only market_cap_amt > this
    min_fii_hold_pct: float | None = None     # keep only fii_hold_pct > this
    min_public_hold_pct: float | None = None  # keep only public_hold_pct > this

    # Strategy rules & filters
    confidence_cutoff: float = 72.0
    entry_stoploss_gtt_multiplier: float = 0.98  # Entry stoploss GTT multiplier on buy_price (e.g. 0.98 = -2.00%)
    exit_day_mult: float = 0.99                  # Exit-day stoploss multiplier on the exit day's open; 0.0 = disabled
    stop_loss_pct: float = -2.00                 # Derived/synced base stop-loss % from entry_stoploss_gtt_multiplier
    gap_aware_stop: bool = True       # a stock that OPENS below the stop exits at that open, not at the stop
    portfolio_halt_pct: float = -4.0
    min_history: int = 10
    decay_factor: float = 0.98
    equal_weight_basket: bool = False # False = follow strategy/pick allocation; True = split equally across unique tickers
    compound_weekly_capital: bool = True  # True = rebase next week's capital on both weekly net profit and loss
    advance_days: int = 0              # 0, 1, 2, or 3 days to advance anchor/cutoff date

    # Strategies & capital split (keys from the STRATEGIES registry)
    strategies: tuple = ("trend_shift_off", "trend_shift_on")
    allocation_decay: float = 0.97     # per-week decay applied to a strategy's past trades when scoring its win ratio
    win_ratio_prior: float = 5.0       # pseudo-trades at a 50% win ratio; keeps a strategy's first trades from swinging the split
    market_cap_allocation_pct: dict = field(default_factory=dict)

    # Transaction cost model (Indian cash equity delivery) - verify against your broker/exchange
    weekly_capital: float = 1000000.0  # total capital pool per trading week, split across selected stocks
    brokerage_rate: float = 0.0
    brokerage_cap: float = 0.0         # per order; 0 = no cap
    stt_buy_rate: float = 0.001        # 0.10%
    stt_sell_rate: float = 0.001       # 0.10%
    exchange_txn_rate: float = 0.0000325
    sebi_turnover_rate: float = 0.000001
    stamp_duty_buy_rate: float = 0.00015
    gst_rate: float = 0.18
    dp_charge_per_sell: float = 18.80
    idle_interest_rate: float = 0.065  # annual rate earned daily (simple, /365) on capital not invested

def _fasta_sql():
    return _FASTA_SQL

# ==========================================
# STOP-LOSS LOGIC
# ==========================================
def entry_stop_pct_from_mult(entry_mult, fallback_pct=-2.0):
    """Convert an entry GTT multiplier (e.g. 0.98) to percentage offset (e.g. -2.0)."""
    try:
        m = float(entry_mult)
        if np.isfinite(m) and m > 0.0:
            return round((m - 1.0) * 100.0, 4)
    except (TypeError, ValueError):
        pass
    return float(fallback_pct)


def gtt_register(buy_price, stop_loss_pct=None, entry_stoploss_gtt_multiplier=None):
    """Register a GTT stop price from the executed buy price."""
    buy_price = float(buy_price)
    if not np.isfinite(buy_price) or buy_price <= 0:
        raise ValueError("GTT registration requires a positive finite buy price.")
    if entry_stoploss_gtt_multiplier is not None:
        m = float(entry_stoploss_gtt_multiplier)
        if np.isfinite(m) and m > 0.0:
            return buy_price * m
    pct = -2.0 if stop_loss_pct is None else float(stop_loss_pct)
    return buy_price * (1.0 + pct / 100.0)


def gtt_trigger(stop_price, day_low):
    """Daily GTT trigger: signal when the day's low reaches or crosses the stop."""
    return bool(
        np.isfinite(day_low)
        and np.isfinite(stop_price)
        and float(day_low) <= float(stop_price)
    )


def _lookup_daily_stop(daily_stops, ticker):
    if not daily_stops:
        return None
    if ticker in daily_stops:
        return daily_stops[ticker]
    bare = _bare(str(ticker)).upper()
    return daily_stops.get(bare)


def gtt_daily_exit(
        daily_stops, ticker, entry_date, end_date, stop_price, gap_aware=True,
        debug_log=None, buy_price=None, entry_stoploss_gtt_multiplier=0.98,
        exit_day_mult=0.99,
):
    """Return (fill_price, reason, fill_date) for the first daily GTT trigger.

    Order of stoploss GTT multipliers (0.0 disables a day-of-week ratchet):
    1. Entry stoploss GTT multiplier (`entry_stoploss_gtt_multiplier`, default 0.98):
       sets base stop `buy_price * entry_stoploss_gtt_multiplier` (falling back to
       `stop_price` if `buy_price` is not provided).
    2. Exit-day stoploss multiplier (`exit_day_mult`, default 0.99, 0.0 = off):
       on the exit day (hardcoded to Friday, weekday 4), candidate is
       `max(base_stop, day_open * exit_day_mult)`.
    """
    rec = _lookup_daily_stop(daily_stops, ticker)
    if rec is None:
        return None

    dates, opens, lows = rec
    if len(dates) == 0:
        return None

    entry_day = pd.Timestamp(entry_date).normalize()
    end_day = pd.Timestamp(end_date).normalize()
    entry_mult = float(entry_stoploss_gtt_multiplier) if entry_stoploss_gtt_multiplier is not None else 0.0
    exit_day_mult = float(exit_day_mult) if exit_day_mult is not None else 0.0

    order = np.argsort(dates, kind="stable")
    dates, opens, lows = dates[order], opens[order], lows[order]

    start_idx = np.searchsorted(dates, entry_day.to_datetime64(), side="left")
    end_idx = np.searchsorted(dates, end_day.to_datetime64(), side="right")
    original_stop = float(stop_price)
    if buy_price is not None and np.isfinite(buy_price) and float(buy_price) > 0 and np.isfinite(entry_mult) and entry_mult > 0.0:
        base_stop = float(buy_price) * entry_mult
    else:
        base_stop = original_stop
    active_stop = max(original_stop, base_stop) if (np.isfinite(original_stop) and np.isfinite(base_stop)) else base_stop
    ratcheted_on = None
    for idx in range(start_idx, end_idx):
        trigger_date = pd.Timestamp(dates[idx]).normalize()
        day_open = opens[idx]
        # 3. Exit-day stoploss multiplier: exit day is hardcoded to Friday (weekday 4)
        if np.isfinite(exit_day_mult) and exit_day_mult > 0.0 and trigger_date.weekday() == 4:
            if np.isfinite(day_open) and day_open > 0:
                exit_day_candidate = max(base_stop, float(day_open) * exit_day_mult)
                prev_stop = active_stop
                active_stop = max(active_stop, exit_day_candidate)
                if active_stop > prev_stop:
                    ratcheted_on = trigger_date
                if debug_log is not None:
                    debug_log(
                        f"[GTT-EXITDAY] {ticker} entry={entry_day.date()} exit_day={trigger_date.date()} "
                        f"base_stop={base_stop:.2f} prev_stop={prev_stop:.2f} exit_day_open={float(day_open):.2f} "
                        f"exit_day_mult={exit_day_mult:.5f} cand={exit_day_candidate:.2f} -> active_stop={active_stop:.2f} low={float(lows[idx]):.2f}"
                    )

        if not gtt_trigger(active_stop, lows[idx]):
            continue

        is_gap = bool(
            gap_aware
            and trigger_date > entry_day
            and np.isfinite(opens[idx])
            and opens[idx] <= active_stop
        )
        if is_gap:
            reason = "STOP_OPENGAP"
            fill_price = float(opens[idx])
        elif trigger_date == entry_day:
            reason = "STOPLOSS_ENTRYDAY"
            fill_price = float(active_stop)
        elif trigger_date == end_day:
            reason = "STOPLOSS_EXITDAY"
            fill_price = float(active_stop)
        else:
            reason = "STOPLOSS_MIDWEK"
            fill_price = float(active_stop)
        if debug_log is not None:
            debug_log(
                f"[GTT-FIRE] {ticker} entry={entry_day.date()} fire={trigger_date.date()} "
                f"exit_day_reset={ratcheted_on.date() if ratcheted_on else None} "
                f"orig_stop={original_stop:.2f} final_stop={active_stop:.2f} "
                f"low={float(lows[idx]):.2f} fill={fill_price:.2f} reason={reason}"
            )
        return fill_price, reason, trigger_date
    return None


def stop_check(con, ticker, week_start, buy_price, stop_loss_pct=-2.0,
               entry_stoploss_gtt_multiplier=0.98, exit_day_mult=0.99):
    """Daily open/low of one ticker-week and what the stop logic does with it (was `explain_stop`)."""
    week_start = pd.Timestamp(week_start).normalize()
    daily = _FASTA_SQL.load_stop_check_data(con, ticker, week_start)
    if daily.empty:
        return None
    daily.columns = [c.lower() for c in daily.columns]
    daily['date'] = pd.to_datetime(daily['date']).dt.normalize()
    daily = daily.sort_values('date', kind='stable').reset_index(drop=True)
    if buy_price:
        entry = float(buy_price)
    else:
        first = daily.iloc[0]
        ohlc = [first[c] for c in ('open', 'high', 'low', 'close') if c in daily.columns]
        entry = float(np.nanmean(ohlc))  # day average of the entry day
    eff_pct = entry_stop_pct_from_mult(entry_stoploss_gtt_multiplier, stop_loss_pct)
    stop_px = entry * (1.0 + eff_pct / 100.0)
    daily['low <= stop'] = daily['low'] <= stop_px
    daily['open <= stop'] = daily['open'] <= stop_px
    rec = {ticker: (pd.to_datetime(daily['date']).values.astype('datetime64[ns]'),
                    daily['open'].to_numpy(float), daily['low'].to_numpy(float))}
    fill = gtt_daily_exit(
        rec, ticker, pd.Timestamp(daily['date'].iloc[0]),
        pd.Timestamp(daily['date'].iloc[-1]), stop_px,
        gap_aware=True,
        buy_price=entry,
        entry_stoploss_gtt_multiplier=entry_stoploss_gtt_multiplier,
        exit_day_mult=exit_day_mult,
    )
    return {"daily": daily, "entry": entry, "stop_px": stop_px, "fill": fill, "week_start": week_start}


# ==========================================
# N-GRAM PREDICTION + SIMULATION
# ==========================================
def get_recency_ngram_prediction(train_str, n, decay_factor):
    if len(train_str) >= n:
        last_n = train_str[-n:]
        transitions = {}
        total_weight = 0.0
        max_idx = len(train_str) - n
        for i in range(max_idx):
            if train_str[i:i + n] == last_n:
                nxt = train_str[i + n]
                age = max_idx - 1 - i
                weight = decay_factor ** age
                transitions[nxt] = transitions.get(nxt, 0.0) + weight
                total_weight += weight
        if transitions:
            best_char, best_weight = max(transitions.items(), key=lambda x: x[1])
            prob = best_weight / total_weight if total_weight > 0 else 0
            return best_char, prob

    if n > 1:
        return get_recency_ngram_prediction(train_str, n - 1, decay_factor)

    counts = {}
    total_weight = 0.0
    for i, char in enumerate(train_str):
        age = len(train_str) - 1 - i
        weight = decay_factor ** age
        counts[char] = counts.get(char, 0.0) + weight
        total_weight += weight
    if counts:
        best_char, best_weight = max(counts.items(), key=lambda x: x[1])
        return best_char, (best_weight / total_weight if total_weight > 0 else 0)
    return 'P', 0.0


# ==========================================
# STRATEGIES (pluggable "visitors")
# ==========================================
@dataclass(frozen=True)
class TickerWeek:
    """One ticker in one signal week, as a strategy sees it.
    `history` contains only completed prior weeks. `volume` is the prior week's average volume."""
    ticker: str
    week: pd.Timestamp
    history: np.ndarray      # bucket chars ('L' / 'P' / ...) of past completed weeks, oldest first
    volume: float            # previous completed week's average volume


@dataclass(frozen=True)
class Signal:
    buy: bool
    confidence: float = 0.0  # in %, used to rank the picks


class Strategy(ABC):
    key: str = ""            # stable id: stored in every trade row and in Params.strategies
    label: str = ""          # shown in the UI
    description: str = ""

    @abstractmethod
    def visit(self, ctx: TickerWeek, p: Params) -> Signal:
        """Decide whether to buy `ctx.ticker` this week."""


STRATEGIES: dict = {}


def register(cls):
    """Class decorator: make a strategy selectable by its key."""
    inst = cls()
    if not inst.key:
        raise ValueError(f"{cls.__name__} needs a non-empty `key`.")
    STRATEGIES[inst.key] = inst
    return cls


def resolve_strategies(keys):
    keys = list(dict.fromkeys(keys))  # de-duplicate, keep order
    if not keys:
        raise ValueError("Select at least one strategy.")
    unknown = [k for k in keys if k not in STRATEGIES]
    if unknown:
        raise ValueError(f"Unknown strategy {unknown}. Registered: {list(STRATEGIES)}")
    return [STRATEGIES[k] for k in keys]


class _NGramStrategy(Strategy):
    """Recency-weighted 4-gram predictor over the L/P history; buys when it predicts 'P' with enough confidence."""
    invert = False

    def visit(self, ctx, p):
        train_str = "".join(ctx.history)
        if self.invert:  # trend shift: swap the L/P labels before predicting
            train_str = train_str.translate(str.maketrans("LP", "PL"))
        pred_char, prob = get_recency_ngram_prediction(train_str, 4, p.decay_factor)
        confidence = round(prob * 100, 2)
        return Signal(pred_char == 'P' and confidence >= p.confidence_cutoff, confidence)


@register
class TrendShiftOff(_NGramStrategy):
    key = "trend_shift_off"
    label = "Trend shift OFF"
    description = "N-gram on the L/P history as is: buys where the pattern predicts 'P'."


@register
class TrendShiftOn(_NGramStrategy):
    key = "trend_shift_on"
    label = "Trend shift ON"
    description = "Same n-gram with L/P swapped first: buys where the plain pattern would predict 'L'."
    invert = True


# ==========================================
# WEEKLY CAPITAL COMPOUNDING / REBASING
# ==========================================
def apply_weekly_capital_compounding(weekly_df, p: Params):
    """Rebase each week's base capital for investment on both profit and loss when enabled.

    When `p.compound_weekly_capital` is True:
    - Week 1 starts at `p.weekly_capital`.
    - Each subsequent week's base capital is rebased to `max(0.0, prior_base_capital + prior_net_rupees)`
      so both profits and losses compound into the next investment base.
    Idle capital earns `p.idle_interest_rate` p.a. accrued daily (rate / 365): the part of the week's base capital
    that is not invested earns for the full 7 days, and cash freed by a sale earns from the sell date until the
    next Monday. The interest is added to `net_rupees` (and reported in `interest_rupees`).
    When `p.compound_weekly_capital` is False:
    - Every week uses a fixed `p.weekly_capital` base.
    """
    out = weekly_df.copy()
    if out.empty:
        out['base_capital'] = pd.Series(dtype=float)
        out['rebased_next_capital'] = pd.Series(dtype=float)
        return out

    base_caps = []
    next_caps = []
    current_base = float(p.weekly_capital)
    compounding = bool(getattr(p, "compound_weekly_capital", False))

    daily_rate = float(getattr(p, "idle_interest_rate", 0.0) or 0.0) / 365.0
    invested = out['invested'].to_numpy(dtype=float) if 'invested' in out else np.zeros(len(out))
    freed = out['freed_cash_days'].to_numpy(dtype=float) if 'freed_cash_days' in out else np.zeros(len(out))
    interest = []
    totals = []

    for k, net_pnl in enumerate(out['net_rupees']):
        base_cap = current_base
        idle_interest = daily_rate * (max(base_cap - invested[k], 0.0) * 7.0 + freed[k])
        total_net = float(net_pnl) + idle_interest
        rebased_end = max(0.0, base_cap + total_net)
        base_caps.append(base_cap)
        next_caps.append(rebased_end)
        interest.append(idle_interest)
        totals.append(total_net)
        current_base = rebased_end if compounding else float(p.weekly_capital)

    out['interest_rupees'] = interest
    out['net_rupees'] = totals  # trade net PnL + idle-capital interest
    out['base_capital'] = base_caps
    out['rebased_next_capital'] = next_caps
    return out


# ==========================================
# SIMULATION ENGINE
# ==========================================
@dataclass
class _StrategyRun:
    """Per-strategy state. Every strategy keeps its own portfolio halt and its own trade list
    (its stand-alone track record, which is what the capital split is scored on)."""
    halted: bool = False
    week_pnls: list = field(default_factory=list)
    shadow_pnls: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    open_pos: dict = field(default_factory=dict)  # ticker -> position still held at last Friday's exit (carry candidate)


def resolve_picks_window(df, p: Params):
    """Resolve the entry week and completed training cutoff for PICKS without intra-week look-ahead.

    - If `p.picks_training_cutoff` is provided (e.g. Friday, Saturday/Sunday weekend, or any day in the
      running week), anchors to the last completed Friday on or before that date (`weekday`: Mon=0..Sun=6).
    - Passing Saturday or Sunday includes the completed Friday week (and weekend shadow updates),
      targeting the following Monday's entry window.
    - Passing any day of the running Mon–Thu week resolves back to the prior Friday so live intra-week
      bars never mutate `stock_in_shadow` or pick prices.
    """
    latest_data_date = pd.Timestamp(pd.to_datetime(df['last_date']).max()).normalize()
    if getattr(p, "cutoff_date", ""):
        anchor_day = pd.Timestamp(p.cutoff_date).normalize()
    else:
        anchor_day = latest_data_date
    adv = int(getattr(p, "advance_days", 0) or 0)
    if adv > 0:
        anchor_day = pd.Timestamp(_effective_today(anchor_day.date(), advance_days=adv)).normalize()

    days_since_friday = (anchor_day.weekday() - 4) % 7
    last_completed_friday = anchor_day - pd.Timedelta(days=days_since_friday)
    target_entry_week = last_completed_friday + pd.Timedelta(days=3)  # Entry Monday
    signal_week_end = target_entry_week + pd.Timedelta(days=3)      # Signal Thursday
    is_next_week = last_completed_friday >= latest_data_date
    return last_completed_friday, target_entry_week, signal_week_end, is_next_week


def run_simulation(df, daily_stops, strategies, p: Params, progress=None, log=None):
    """Back-test with point-in-time inputs and completed-week outcomes only."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    work = df.copy()
    work['w_date'] = pd.to_datetime(work['w_date']).dt.normalize()
    if 'last_date' in work.columns:
        work['last_date'] = pd.to_datetime(work['last_date']).dt.normalize()
    else:
        work['last_date'] = work['w_date'] + pd.Timedelta(days=4)
    if 'first_date' in work.columns:
        work['first_date'] = pd.to_datetime(work['first_date']).dt.normalize()
    else:
        work['first_date'] = work['w_date']

    all_weeks = sorted(work['w_date'].dropna().unique())
    if not all_weeks:
        return pd.DataFrame(), pd.DataFrame()

    latest_week = pd.Timestamp(all_weeks[-1]).normalize()
    latest_rows = work.loc[work['w_date'] == latest_week]
    latest_data_date = latest_rows['last_date'].max()
    if getattr(p, "cutoff_date", ""):
        cutoff_day = pd.Timestamp(p.cutoff_date).normalize()
    else:  # no cutoff: run through the latest available data
        cutoff_day = latest_data_date + pd.Timedelta(days=1)
    adv = int(getattr(p, "advance_days", 0) or 0)
    if adv > 0:
        eff_anchor = pd.Timestamp(_effective_today(cutoff_day.date(), advance_days=adv)).normalize()
        if eff_anchor > cutoff_day:
            cutoff_day = eff_anchor
    cutoff_week_start = cutoff_day.to_period('W-SUN').start_time.normalize()
    latest_friday = latest_week + pd.Timedelta(days=4)
    if latest_week < cutoff_week_start:
        latest_week_complete = True
    else:
        latest_week_complete = bool(
            (pd.notna(latest_data_date) and latest_data_date >= latest_friday)
            or cutoff_day > latest_friday
        )

    completed_weeks = set(pd.Timestamp(w).normalize() for w in all_weeks[:-1])
    if latest_week_complete:
        completed_weeks.add(latest_week)

    last_completed_friday, target_entry_week, _, _ = resolve_picks_window(work, p)

    has_volume = 'avg_volume' in work.columns
    median_volume_by_week = (
        work.groupby('w_date')['avg_volume'].median().to_dict() if has_volume else {}
    )

    ticker_data = {}
    for ticker, grp in work.groupby('ticker'):
        grp = grp.sort_values('w_date').reset_index(drop=True)
        if len(grp) < p.min_history:
            continue
        dates = pd.to_datetime(grp['w_date']).values.astype('datetime64[ns]')
        opens_arr = grp['open_price'].to_numpy(dtype=float) if 'open_price' in grp.columns else grp['close_price'].to_numpy(dtype=float)
        # Buy price = day average of the entry day (falls back to the open if unavailable)
        entries_arr = grp['entry_price'].to_numpy(dtype=float) if 'entry_price' in grp.columns else opens_arr
        entries_arr = np.where(np.isfinite(entries_arr) & (entries_arr > 0), entries_arr, opens_arr)
        lows_arr = grp['low_price'].to_numpy(dtype=float) if 'low_price' in grp.columns else grp['close_price'].to_numpy(dtype=float)
        closes_arr = grp['close_price'].to_numpy(dtype=float)
        raw_volumes = grp['avg_volume'].to_numpy(dtype=float) if has_volume else np.full(len(grp), np.nan)
        volume_by_week = {
            pd.Timestamp(d).normalize(): float(v)
            for d, v in zip(dates, raw_volumes)
            if np.isfinite(v)
        }
        prior_volumes = []
        prior_volume_medians = []
        for d in dates:
            prior_week = pd.Timestamp(d).normalize() - pd.Timedelta(days=7)
            prior_volumes.append(volume_by_week.get(prior_week, np.nan))
            prior_volume_medians.append(median_volume_by_week.get(prior_week, np.nan))

        ticker_data[ticker] = {
            'dates': dates,
            'first_dates': pd.to_datetime(grp['first_date']).values.astype('datetime64[ns]'),
            'last_dates': pd.to_datetime(grp['last_date']).values.astype('datetime64[ns]'),
            'opens': opens_arr,
            'entries': entries_arr,
            'closes': closes_arr,
            'lows': lows_arr,
            'volumes': raw_volumes,
            'prior_volumes': np.asarray(prior_volumes, dtype=float),
            'prior_volume_medians': np.asarray(prior_volume_medians, dtype=float),
            'chars': grp['bucket_char'].to_numpy(),
        }

    def passes_liquidity(tdict, idx, is_carry_trade):
        if not has_volume:
            return True
        volume = tdict['prior_volumes'][idx]
        threshold = tdict['prior_volume_medians'][idx]
        return bool(np.isfinite(volume) and np.isfinite(threshold) and volume >= threshold)

    stock_in_shadow = set()
    runs = {s.key: _StrategyRun() for s in strategies}

    for w_idx, w_date in enumerate(all_weeks):
        if progress and w_idx % 3 == 0:
            progress(w_idx / max(len(all_weeks), 1), f"Simulating week {w_idx + 1} of {len(all_weeks)}…")
        week_dt = pd.Timestamp(w_date).normalize()

        # Stop strictly before `target_entry_week` and skip incomplete weeks so intra-week
        # price movements cannot mutate `stock_in_shadow` before picks are generated.
        if week_dt >= target_entry_week or week_dt not in completed_weeks:
            continue

        for run in runs.values():
            run.week_pnls.clear()
            run.shadow_pnls.clear()

        for ticker, tdict in ticker_data.items():
            dates = tdict['dates']
            idx_match = np.where(dates == week_dt.to_datetime64())[0]
            if len(idx_match) == 0:
                continue
            i = int(idx_match[0])
            if i < p.min_history or len(tdict['chars'][:i]) < 5:
                continue

            prior_volume = tdict['prior_volumes'][i]
            was_in_shadow = ticker in stock_in_shadow

            buy_price = tdict['entries'][i]
            if not np.isfinite(buy_price) or buy_price <= 0:
                continue
            buy_date = pd.Timestamp(tdict['first_dates'][i])
            is_carry_trade = False
            effective_entry_mult = 0.98 if is_carry_trade else p.entry_stoploss_gtt_multiplier
            effective_stop_pct = entry_stop_pct_from_mult(effective_entry_mult, p.stop_loss_pct)
            stop_price = gtt_register(buy_price, effective_stop_pct, effective_entry_mult)
            week_last_day = pd.Timestamp(tdict['last_dates'][i]).normalize()
            holding_end = week_last_day
            has_stop_rec = _lookup_daily_stop(daily_stops, ticker) is not None
            gtt_fill = gtt_daily_exit(
                daily_stops, ticker, buy_date, holding_end, stop_price,
                gap_aware=p.gap_aware_stop,
                debug_log=log,
                buy_price=buy_price,
                entry_stoploss_gtt_multiplier=effective_entry_mult,
                exit_day_mult=p.exit_day_mult,
            )

            if gtt_fill is not None:
                sell_price, stop_reason, sell_date = gtt_fill
                pnl_pct = round(((sell_price - buy_price) / buy_price) * 100.0, 2)
                exit_reason = stop_reason
                stock_in_shadow.add(ticker)
            elif daily_stops is None or not has_stop_rec:
                # Compatibility fallback when daily OHLC is unavailable for this ticker.
                week_low = float(tdict['lows'][i]) if np.isfinite(tdict['lows'][i]) else np.nan
                if gtt_trigger(stop_price, week_low):
                    sell_price = stop_price
                    sell_date = week_last_day
                    pnl_pct = round(((sell_price - buy_price) / buy_price) * 100.0, 2)
                    exit_reason = (
                        "STOCK_STOP_LOSS_NODATA" if daily_stops is not None
                        else "STOCK_STOP_LOSS"
                    )
                    stock_in_shadow.add(ticker)
                else:
                    sell_price = tdict['closes'][i]
                    sell_date = week_last_day
                    pnl_pct = round(((sell_price - buy_price) / buy_price) * 100, 2)
                    exit_reason = "NORMAL_EXIT"
                    if pnl_pct <= p.stop_loss_pct:
                        exit_reason = "STOCK_STOP_LOSS"
                        stock_in_shadow.add(ticker)
            else:
                sell_price = tdict['closes'][i]
                sell_date = week_last_day
                pnl_pct = round(((sell_price - buy_price) / buy_price) * 100, 2)
                exit_reason = "NORMAL_EXIT"
                if pnl_pct <= p.stop_loss_pct:
                    exit_reason = "STOCK_STOP_LOSS"
                    stock_in_shadow.add(ticker)

            if was_in_shadow:
                if pnl_pct > 0.0:
                    stock_in_shadow.discard(ticker)
                else:
                    continue

            ctx = TickerWeek(ticker, week_dt, tdict['chars'][:i], prior_volume)
            liquidity_ok = passes_liquidity(tdict, i, is_carry_trade)
            has_complete_exit = True
            for strat in strategies:
                run = runs[strat.key]
                if run.halted:
                    run.open_pos.pop(ticker, None)
                    if has_complete_exit:
                        run.shadow_pnls.append(pnl_pct)
                    continue
                sig = strat.visit(ctx, p)
                if sig.buy and liquidity_ok and has_complete_exit:
                    # CARRY: held through last Friday's exit (not stopped) and signalled again. The weekly
                    # trade is unchanged (Monday-open buy, own stop, profit shown this week); only the
                    # charges differ: no sell charges for last week's row, no buy charges for this one.
                    pos = run.open_pos.get(ticker)
                    carried = bool(pos is not None and pos['last_week'] == week_dt - pd.Timedelta(days=7))
                    if carried:
                        pos['trade']['carry_out'] = True
                    trade = {
                        'ticker': ticker,
                        'strategy': strat.key,
                        'buy_date': buy_date.strftime('%Y-%m-%d'),
                        'sell_date': sell_date.strftime('%Y-%m-%d'),
                        'buy_price': round(buy_price, 2),
                        'sell_price': round(sell_price, 2),
                        'confidence': sig.confidence,
                        'exit_reason': exit_reason,
                        'pnl_pct': pnl_pct,
                        'carry_in': carried,
                        'carry_out': False,
                    }
                    run.trades.append(trade)
                    run.week_pnls.append(pnl_pct)
                    if exit_reason == "NORMAL_EXIT":
                        run.open_pos[ticker] = {'trade': trade, 'last_week': week_dt}
                    else:
                        run.open_pos.pop(ticker, None)
                else:
                    run.open_pos.pop(ticker, None)

        for run in runs.values():
            if run.halted:
                if run.shadow_pnls and sum(run.shadow_pnls) / len(run.shadow_pnls) > 0.0:
                    run.halted = False
            elif run.week_pnls and sum(run.week_pnls) / len(run.week_pnls) <= p.portfolio_halt_pct:
                run.halted = True

    # Generate PICKS strictly from completed history prior to `target_entry_week`
    picks = []
    expected_prior_week = target_entry_week - pd.Timedelta(days=7)
    for strat in strategies:
        if runs[strat.key].halted:
            continue
        for ticker, tdict in ticker_data.items():
            if ticker in stock_in_shadow:
                continue
            valid_idx = np.where(tdict['dates'] < target_entry_week.to_datetime64())[0]
            if len(valid_idx) == 0:
                continue
            last_i = int(valid_idx[-1])
            last_week_dt = pd.Timestamp(tdict['dates'][last_i]).normalize()
            if last_week_dt != expected_prior_week:
                continue

            history_slice = tdict['chars'][:last_i + 1]
            if len(history_slice) < p.min_history:
                continue

            if has_volume:
                vol = tdict['volumes'][last_i]
                med = median_volume_by_week.get(last_week_dt, np.nan)
                if not (np.isfinite(vol) and np.isfinite(med) and vol >= med):
                    continue
            else:
                vol = np.nan

            sig = strat.visit(TickerWeek(ticker, target_entry_week, history_slice, vol), p)
            if sig.buy:
                picks.append({
                    'ticker': ticker,
                    'strategy': strat.key,
                    'entry_date': target_entry_week.strftime('%Y-%m-%d'),
                    'price': float(tdict['closes'][last_i]),
                    'confidence': sig.confidence
                })

    trades = [t for run in runs.values() for t in run.trades]
    return pd.DataFrame(trades), pd.DataFrame(picks)


# ==========================================
# CAPITAL ALLOCATION ACROSS STRATEGIES
# ==========================================
def pick_window(df):
    if df.empty:
        return pd.NaT, pd.NaT, False

    required = {'w_date', 'last_date'}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"pick_window() requires columns: {sorted(missing)}")

    latest_week = pd.Timestamp(pd.to_datetime(df['w_date']).max()).normalize()
    latest_data_date = pd.Timestamp(pd.to_datetime(df['last_date']).max()).normalize()
    signal_week_end = latest_week + pd.Timedelta(days=3)
    for_next_week = latest_data_date >= signal_week_end
    return latest_week, signal_week_end, for_next_week


# ==========================================
# ANALYSIS
# ==========================================
_DROP_FOR_DISPLAY = ['sort_date', 'week_period', 'quarter_period', 'year_period']


def _parse_symbols(text):
    seen = {}
    for tok in re.split(r"[\s,;]+", text or ""):
        sym = _bare(tok.strip()).upper()
        if sym:
            seen[sym] = None
    return tuple(seen)


def _load_settings():
    return load_json_dict(SETTINGS_FILE)


def _save_settings(**updates):
    try:
        save_json_atomic({**_load_settings(), **updates}, SETTINGS_FILE)
        return True
    except OSError:
        return False

def has_universe_filter(p: Params) -> bool:
    return bool(p.market_caps) or any(
        v is not None for v in (p.min_market_cap_amt, p.min_fii_hold_pct, p.min_public_hold_pct))


@st.cache_data(show_spinner=False, ttl=600)
def load_market_cap_options():
    try:
        return _FASTA_SQL.load_market_cap_options(get_connection())
    except Exception:
        return []


def attach_market_cap(df, market_cap_map):
    if df.empty or "ticker" not in df:
        return df
    out = df.copy()
    bare = out["ticker"].map(lambda ticker: _bare(str(ticker)).upper())
    out["market_cap"] = bare.map(market_cap_map)
    return out


def universe_counts(p: Params):
    try:
        con = get_connection()
        everyone = {x for x in _FASTA_SQL.universe_symbols(con, Params()) if not x.startswith("^")}
        passing = {x for x in _FASTA_SQL.universe_symbols(con, p) if not x.startswith("^")}
        return len(passing), len(everyone)
    except Exception:
        return 0, None


def analyze(p: Params, progress=None, log=lambda m: None):
    con = get_connection()
    df = _FASTA_SQL.load_market_data(con, p, log, CACHE_DIR)
    if df.empty:
        raise ValueError("The market-data query returned no rows for this date window.")
    if has_universe_filter(p):
        allowed = _FASTA_SQL.universe_symbols(con, p)
        bare = df['ticker'].map(lambda t: _bare(t).upper())
        before = bare.nunique()
        df = df[bare.isin(allowed)].copy()
        log(f"[Universe filter] {df['ticker'].nunique()} of {before} tickers pass "
            f"(market_cap in {list(p.market_caps) or 'ALL'}, market_cap_amt > {p.min_market_cap_amt}, "
            f"fii_hold_pct > {p.min_fii_hold_pct}, public_hold_pct > {p.min_public_hold_pct}).")
        if df.empty:
            raise ValueError("No tickers pass the universe filters. Relax market cap / holding thresholds.")
    if p.restrict_tickers:
        wanted = set(p.restrict_tickers)
        bare = df['ticker'].map(lambda t: _bare(t).upper())
        df = df[bare.isin(wanted)].copy()
        found = set(bare[bare.isin(wanted)])
        log(f"[Portfolio filter] {len(found)} of {len(wanted)} portfolio tickers found in market data.")
        missing = sorted(wanted - found)
        if missing:
            log(f"[Portfolio filter] No market data for: {', '.join(missing)}")
        if df.empty:
            raise ValueError("None of the portfolio tickers were found in the market data.")
    if p.exclude_tickers:
        unwanted = set(p.exclude_tickers)
        bare = df['ticker'].map(lambda t: _bare(t).upper())
        removed = sorted(set(bare[bare.isin(unwanted)]))
        df = df[~bare.isin(unwanted)].copy()
        log(f"[Exclude list] Removed {len(removed)} of {len(unwanted)} listed tickers from the universe.")
        unknown = sorted(unwanted - set(removed))
        if unknown:
            log(f"[Exclude list] Not found in market data: {', '.join(unknown)}")
        if df.empty:
            raise ValueError("Every ticker in the universe is on the exclude list.")
    daily_stops = _FASTA_SQL.load_daily_stop_data(con, p, log, CACHE_DIR)
    nifty_weekly, nifty_q, nifty_y = _FASTA_SQL.load_nifty_benchmark(con, p, log, CACHE_DIR)

    # Carry point-in-time-unaware static segment metadata from the tickers dimension
    # into trades and picks so the segment caps can be applied consistently.
    market_cap_map = _FASTA_SQL.ticker_market_cap_map(con)
    strategies = resolve_strategies(p.strategies)
    raw_trades, picks_df = run_simulation(df, daily_stops, strategies, p, progress, log=log)
    raw_trades = attach_market_cap(raw_trades, market_cap_map)
    picks_df = attach_market_cap(picks_df, market_cap_map)
    training_through, wk_start, wk_end, is_next = resolve_picks_window(df, p)
    training_label = (
        f"training data available through {training_through:%a %Y-%m-%d}"
        if pd.notna(training_through) else "no prior-week training data available"
    )
    picks_title = (
        f"{'NEXT-WEEK' if is_next else 'CURRENT-WEEK'} PICKS — "
        f"entry/signal window Mon {wk_start:%Y-%m-%d} to Thu {wk_end:%Y-%m-%d} · "
        f"{training_label} (data strictly before entry week; signals at Thursday close | "
        f"CARRY = already held)"
    )

    record = _track_record(raw_trades)
    win_ratios_now = {s.key: recency_win_ratio(*record.get(s.key, _NO_TRADES), _week_no(wk_start), p)
                      for s in strategies}
    trades_df, alloc_df = allocate_capital(raw_trades, p)
    trades_df = apply_transaction_costs(trades_df, p)
    if not picks_df.empty:
        picks_df = (allocate_picks(picks_df, win_ratios_now, p)
                    .sort_values('confidence', ascending=False).reset_index(drop=True))

    res = {
        "params": asdict(p), "picks": picks_df, "picks_title": picks_title,
        "data_through": pd.Timestamp(pd.to_datetime(df['last_date']).max()).normalize(),
        "trades": trades_df, "weekly": None, "quarterly": None, "yearly": None,
        "kpis": None, "exit_stats": None, "stop_stats": None, "chart": None,
        "strategies": [s.key for s in strategies], "strategy_stats": None, "win_ratio_chart": None,
        "gap_check_on": bool(p.gap_aware_stop and daily_stops is not None), "log": [],
    }

    if not trades_df.empty:
        stop_reasons = (
            "STOCK_STOP_LOSS",
            "STOPLOSS_ENTRYDAY",
            "STOPLOSS_EXITDAY",
            "STOPLOSS_MIDWEK",
            "STOP_OPENGAP",
            "STOCK_STOP_LOSS_GAP",
            "STOCK_STOP_LOSS_NODATA",
        )
        n_stops = int(trades_df['exit_reason'].isin(stop_reasons).sum())
        gaps = trades_df[trades_df['exit_reason'].isin(("STOP_OPENGAP", "STOCK_STOP_LOSS_GAP"))]
        nodata = int((trades_df['exit_reason'] == 'STOCK_STOP_LOSS_NODATA').sum())
        at_stop = int((
                              trades_df['exit_reason'].isin(("STOCK_STOP_LOSS", "STOPLOSS_ENTRYDAY", "STOPLOSS_EXITDAY", "STOPLOSS_MIDWEK"))
                              & (trades_df['pnl_pct'] == p.stop_loss_pct)
                      ).sum())
        res["stop_stats"] = {
            "n_stops": n_stops, "at_stop": at_stop, "gaps": len(gaps),
            "gap_avg": float(gaps['pnl_pct'].mean()) if len(gaps) else None,
            "nodata": nodata, "below_stop_weekly_exit": max(0, n_stops - at_stop - len(gaps) - nodata),
        }

        trades_df['sort_date'] = pd.to_datetime(trades_df['sell_date'])
        trades_df = trades_df.sort_values('sort_date', kind='stable').reset_index(drop=True)
        trades_df['week_period'] = pd.to_datetime(trades_df['buy_date']).dt.to_period('W')
        trades_df['quarter_period'] = trades_df['sort_date'].dt.to_period('Q')
        trades_df['year_period'] = trades_df['sort_date'].dt.to_period('Y')
        res["trades"] = trades_df

        weekly_net = trades_df.groupby('week_period')['net_profit_rupees'].sum()
        weekly_cnt = trades_df.groupby('week_period').size()
        market_week_starts = pd.to_datetime(df['w_date']).dt.normalize()
        first_market_week = market_week_starts.min().to_period('W')
        last_closed_entry_week = (pd.Timestamp(wk_start).normalize() - pd.Timedelta(days=1)).to_period('W')
        if first_market_week <= last_closed_entry_week:
            calendar_weeks = pd.period_range(
                start=first_market_week, end=last_closed_entry_week, freq='W'
            )
        else:
            calendar_weeks = pd.PeriodIndex([], freq='W')
        weekly_net = weekly_net.reindex(calendar_weeks, fill_value=0.0)
        weekly_cnt = weekly_cnt.reindex(calendar_weeks, fill_value=0).astype(int)
        # Idle-capital interest inputs: capital invested each week, and cash freed by sales
        # (proceeds x days until next Monday). A carried-out row is not sold, so frees nothing.
        week_end = trades_df['week_period'].dt.start_time + pd.Timedelta(days=7)
        idle_days = (week_end - pd.to_datetime(trades_df['sell_date'])).dt.days.clip(lower=0)
        sold = ~trades_df['carry_out'].fillna(False).astype(bool) if 'carry_out' in trades_df else True
        proceeds = trades_df['position_capital'] + trades_df['net_profit_rupees']
        weekly_invested = trades_df.groupby('week_period')['position_capital'].sum().reindex(calendar_weeks, fill_value=0.0)
        weekly_freed = (proceeds * idle_days * sold).groupby(trades_df['week_period']).sum().reindex(calendar_weeks, fill_value=0.0)
        weekly_df = pd.DataFrame({
            'week_period': calendar_weeks, 'trades': weekly_cnt.values,
            'net_rupees': weekly_net.values,
            'invested': weekly_invested.values, 'freed_cash_days': weekly_freed.values})
        weekly_df = apply_weekly_capital_compounding(weekly_df, p)
        weekly_net = pd.Series(weekly_df['net_rupees'].values, index=calendar_weeks)  # now includes interest
        weekly_pct = (weekly_df['net_rupees'] / weekly_df['base_capital'].replace(0.0, np.nan)).fillna(0.0) * 100.0
        weekly_df['portfolio_ret'] = weekly_pct.values
        weekly_df['nifty_ret'] = weekly_df['week_period'].map(lambda w: nifty_weekly.get(w, 0.0))
        weekly_df['quarter_period'] = weekly_df['week_period'].dt.asfreq('Q')
        weekly_df['year_period'] = weekly_df['week_period'].dt.asfreq('Y')
        res["weekly"] = weekly_df

        q = weekly_df.groupby('quarter_period')['portfolio_ret'].sum()
        res["quarterly"] = pd.DataFrame({
            'Quarter': q.index.astype(str), 'Portfolio PnL %': q.values,
            'Nifty 50 %': [nifty_q.get(k, 0.0) for k in q.index]})
        weekly_pct.index = weekly_df['week_period']
        quarter_net = weekly_net.groupby(weekly_net.index.asfreq('Q')).sum()
        quarter_pct = weekly_pct.groupby(weekly_pct.index.asfreq('Q')).sum()
        best_q, worst_q = quarter_pct.idxmax(), quarter_pct.idxmin()
        y = weekly_df.groupby('year_period')['portfolio_ret'].sum()
        res["yearly"] = pd.DataFrame({
            'Year': y.index.astype(str), 'Portfolio PnL %': y.values,
            'Nifty 50 %': [nifty_y.get(k, 0.0) for k in y.index]})
        for key in ("quarterly", "yearly"):
            res[key]['Alpha %'] = res[key]['Portfolio PnL %'] - res[key]['Nifty 50 %']

        ch = weekly_df.sort_values('week_period')
        res["chart"] = pd.DataFrame({
            'Portfolio (cumulative %)': ch['portfolio_ret'].cumsum().values,
            'Nifty 50 (cumulative %, traded weeks)': ch['nifty_ret'].cumsum().values,
        }, index=ch['week_period'].dt.start_time)

        res["exit_stats"] = (trades_df.groupby('exit_reason')
                             .agg(trades=('ticker', 'count'), avg_pnl_pct=('pnl_pct', 'mean'),
                                  avg_net_pnl_pct=('net_pnl_pct', 'mean'), net_rupees=('net_profit_rupees', 'sum'))
                             .reset_index().sort_values('trades', ascending=False))

        keys = [s.key for s in strategies]
        by = trades_df.groupby('strategy')
        stats = pd.DataFrame({
            'trades': by.size(),
            'win rate %': by['pnl_pct'].apply(lambda s: float((s > 0).mean() * 100)),
            'avg net PnL %': by['net_pnl_pct'].mean(),
            'net ₹': by['net_profit_rupees'].sum(),
            'capital ₹': by['position_capital'].sum(),
        }).reindex(keys)
        stats['trades'] = stats['trades'].fillna(0).astype(int)
        stats['avg capital share %'] = alloc_df[alloc_df['weight'] > 0].groupby('strategy')['weight'].mean() * 100
        stats['win ratio now %'] = pd.Series(win_ratios_now) * 100
        res["strategy_stats"] = stats.rename_axis('strategy').reset_index()
        res["win_ratio_chart"] = alloc_df.pivot(index='week', columns='strategy', values='win_ratio') * 100

        total_trades = len(trades_df)
        overall = float(weekly_pct.sum())
        capital = float(trades_df.groupby('week_period')['position_capital'].sum().sum())
        rebased_capital_end = float(weekly_df['rebased_next_capital'].iloc[-1]) if not weekly_df.empty else float(p.weekly_capital)
        net_abs = float(trades_df['net_profit_rupees'].sum())
        gw, lw = weekly_pct.idxmax(), weekly_pct.idxmin()
        res["kpis"] = {
            "overall_pnl_pct": overall,
            "avg_pnl_per_trade_pct": overall / total_trades if total_trades else 0.0,
            "total_trades": total_trades,
            "total_costs": float(trades_df['total_transaction_cost'].sum()),
            "gross_abs": float(trades_df['gross_profit_rupees'].sum()),
            "rebased_capital_end": rebased_capital_end,
            "net_abs": net_abs,
            "interest_abs": float(weekly_df['interest_rupees'].sum()),
            "avg_net_abs_per_trade": float(trades_df['net_profit_rupees'].mean()),
            "capital_deployed": capital,
            "net_return_on_capital_pct": (net_abs / capital * 100.0) if capital else 0.0,
            "avg_net_pnl_pct": float(trades_df['net_pnl_pct'].mean()),
            "max_gain_week": (str(gw), float(weekly_pct.loc[gw]), float(weekly_net.loc[gw])),
            "max_loss_week": (str(lw), float(weekly_pct.loc[lw]), float(weekly_net.loc[lw])),
            "best_quarter": (str(best_q), float(quarter_pct.loc[best_q]), float(quarter_net.loc[best_q])),
            "worst_quarter": (str(worst_q), float(quarter_pct.loc[worst_q]), float(quarter_net.loc[worst_q])),
        }

    res["text_report"] = build_text_report(res, p)
    try:
        out = Path(OUTPUT_FILE)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(res["text_report"])
        log(f"[Details saved to {OUTPUT_FILE}]")
    except Exception as exc:
        log(f"[Could not write {OUTPUT_FILE}: {exc}]")
    return res


def build_text_report(res, p: Params):
    """Plain-text report, same layout the command line tool wrote to fasta.txt."""
    L = ["=" * 115,
         f"  WEEKLY ROTATION TRADING LOG (Weekly ₹{p.weekly_capital:,.0f} Capital Pool, Point-in-Time N-Gram + Prior-Week Liquidity Features)",
         "=" * 115]
    trades, weekly, k = res["trades"], res["weekly"], res["kpis"]
    if k:
        w_pct = dict(zip(weekly['week_period'], weekly['portfolio_ret']))
        w_nif = dict(zip(weekly['week_period'], weekly['nifty_ret']))
        for week, g in trades.groupby('week_period'):
            L.append(f"\n--- Week: {week} ---")
            L.append(g.drop(columns=_DROP_FOR_DISPLAY).to_string(index=False))
            L.append(f"  -> **End of Week Summary**: Portfolio Average PnL = {w_pct.get(week, 0.0):+.2f}% "
                     f"| Nifty 50 Return = {w_nif.get(week, 0.0):+.2f}%")
            L.append("_" * 105)

        L += ["\n" + "=" * 115, "  QUARTERLY PERFORMANCE SUMMARY (Portfolio vs Nifty 50)", "=" * 115]
        for r in res["quarterly"].itertuples():
            L.append(f"Quarter {r.Quarter} | Portfolio PnL: {r._2:+.2f}% | Nifty 50 Return: {r._3:+.2f}%")
        L += ["\n" + "=" * 115, "  YEARLY PERFORMANCE SUMMARY (Portfolio vs Nifty 50)", "=" * 115]
        for r in res["yearly"].itertuples():
            L.append(f"Year {r.Year}    | Portfolio PnL: {r._2:+.2f}% | Nifty 50 Return: {r._3:+.2f}%")

        gw, lw = k["max_gain_week"], k["max_loss_week"]
        bq, wq = k["best_quarter"], k["worst_quarter"]
        L.append(f"""
=========================================================================
  OVERALL CUMULATIVE REALISED PnL ACROSS ALL WEEKS : {k['overall_pnl_pct']:+.2f}%
  AVERAGE PnL PER COMPLETED TRADE                : {k['avg_pnl_per_trade_pct']:+.2f}%
  TOTAL COMPLETED TRADES EXECUTED                : {k['total_trades']}
  TOTAL TRANSACTION COSTS (₹)                    : {k['total_costs']:,.2f}
  TOTAL GROSS ABSOLUTE PnL (₹)                   : {k['gross_abs']:+,.2f}
  TOTAL NET ABSOLUTE PnL (₹)                     : {k['net_abs']:+,.2f}
  AVERAGE NET ABSOLUTE PnL / TRADE (₹)           : {k['avg_net_abs_per_trade']:+,.2f}
  TOTAL CAPITAL DEPLOYED ACROSS TRADES (₹)       : {k['capital_deployed']:,.2f}
  NET RETURN ON TOTAL CAPITAL TRADED             : {k['net_return_on_capital_pct']:+.2f}%
  AVERAGE NET PnL PER TRADE                      : {k['avg_net_pnl_pct']:+.2f}%
  MAXIMUM GAIN WEEK                              : {gw[0]} | {gw[1]:+.2f}% | ₹{gw[2]:+,.2f}
  MAXIMUM LOSS WEEK                              : {lw[0]} | {lw[1]:+.2f}% | ₹{lw[2]:+,.2f}
  BEST QUARTER                                   : {bq[0]} | {bq[1]:+.2f}% | ₹{bq[2]:+,.2f}
  WORST QUARTER                                  : {wq[0]} | {wq[1]:+.2f}% | ₹{wq[2]:+,.2f}

=========================================================================
""")
        L += ["=" * 115, "  STRATEGY BREAKDOWN (capital split by recency-weighted win ratio)", "=" * 115,
              res["strategy_stats"].round(2).to_string(index=False), ""]
    else:
        L.append("No completed transactions generated with current rules.")

    L += ["\n" + "=" * 75 + "\n  " + res["picks_title"] + "\n" + "=" * 75]
    picks = res["picks"]
    L.append(picks.to_string(index=False) if not picks.empty
             else "No active buy picks for the current week matching criteria.")
    L.append("=" * 75)
    if not picks.empty:
        L.extend(picks['ticker'].drop_duplicates().tolist())
    return "\n".join(L)


# ==========================================
# UI HELPERS
# ==========================================
def _display(df):
    out = df.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.PeriodDtype):
            out[c] = out[c].astype(str)
    return out


def _table(df, **kw):
    st.dataframe(_display(df), hide_index=True, **_W, **kw)


def _signed(x, money=False, pct=False):
    s = f"{x:+,.2f}"
    return f"₹{s}" if money else (f"{s}%" if pct else s)


def _portfolio_symbols():
    holdings = st.session_state.get("all_holdings") or {}
    excl = {s.upper() for s in (st.session_state.get("excluded_symbols") or set())}
    return sorted({h["tradingsymbol"].upper()
                   for hs in holdings.values() for h in hs
                   if h.get("tradingsymbol") and h["tradingsymbol"].upper() not in excl})


def _effective_today(raw_today: date | None = None, advance_days: int = 0) -> date:
    """Return `raw_today` (defaulting to `date.today()`), optionally advanced by 0..3 days."""
    d = raw_today or date.today()
    days = int(advance_days or 0)
    if days <= 0:
        return d
    return (pd.Timestamp(d) + pd.Timedelta(days=days)).date()


NO_CUTOFF = date(1970, 1, 1)  # cutoff date value meaning "cleared / not used"


def _parameter_form():
    saved = _load_settings()
    adv_saved = int(saved.get("advance_days", 0) or 0)
    eff_today = _effective_today(date.today(), advance_days=adv_saved)
    D = Params(cutoff_date=eff_today.isoformat(), advance_days=adv_saved)
    cap_opts = load_market_cap_options()

    # A Revert click is handled at the start of the next rerun, before any
    # widgets are instantiated. The dynamic form key recreates form widgets
    # from the persisted values instead of retaining their unsaved state.
    if st.session_state.pop("fasta_revert_pending", False):
        saved = _load_settings()
        st.session_state["fasta_market_caps"] = [x for x in saved.get("market_caps", []) if x in cap_opts]
        st.session_state["fasta_amt_on"] = saved.get("min_market_cap_amt") is not None
        st.session_state["fasta_amt_val"] = float(saved.get("min_market_cap_amt", 10000.0) or 10000.0)
        st.session_state["fasta_fii_on"] = saved.get("min_fii_hold_pct") is not None
        st.session_state["fasta_fii_val"] = float(saved.get("min_fii_hold_pct", 5.0) if saved.get("min_fii_hold_pct") is not None else 5.0)
        st.session_state["fasta_pub_on"] = saved.get("min_public_hold_pct") is not None
        st.session_state["fasta_pub_val"] = float(saved.get("min_public_hold_pct", 10.0) if saved.get("min_public_hold_pct") is not None else 10.0)
        st.session_state["fasta_form_nonce"] = int(st.session_state.get("fasta_form_nonce", 0)) + 1
        st.session_state["fasta_revert_notice"] = True

    form_nonce = int(st.session_state.get("fasta_form_nonce", 0))

    def setting(name, default):
        return saved.get(name, default)

    def saved_date(name, default, blank_as=None):
        value = setting(name, default.isoformat())
        if blank_as is not None and (value is None or str(value).strip() == ""):
            return blank_as
        try:
            return date.fromisoformat(str(value))
        except (TypeError, ValueError):
            return default

    open_it = RESULT_KEY not in st.session_state
    with st.expander("⚙️ Parameters", expanded=open_it):
        st.markdown("**Universe filters** (all conditions are AND-ed)")
        caps_sel = st.multiselect(
            "Market cap", cap_opts,
            default=[x for x in setting("market_caps", []) if x in cap_opts],
            disabled=not cap_opts, key="fasta_market_caps",
            placeholder="ALL (no filter)",
            help="Distinct market_cap values from the tickers table. Leave empty for ALL "
                 "(includes stocks with no market cap data). Selecting any excludes unclassified stocks.")
        f1, f2, f3 = st.columns(3)
        amt_saved = setting("min_market_cap_amt", None)
        fii_saved = setting("min_fii_hold_pct", None)
        pub_saved = setting("min_public_hold_pct", None)
        amt_on = f1.checkbox("Market cap amount >", amt_saved is not None, key="fasta_amt_on")
        amt_val = f1.number_input("market_cap_amt", 0.0, 1e9,
                                  float(amt_saved if amt_saved is not None else 10000.0), 500.0,
                                  disabled=not amt_on, label_visibility="collapsed", key="fasta_amt_val")
        fii_on = f2.checkbox("FII holding % >", fii_saved is not None, key="fasta_fii_on")
        fii_val = f2.number_input("fii_hold_pct", 0.0, 100.0,
                                  float(fii_saved if fii_saved is not None else 5.0), 0.5,
                                  disabled=not fii_on, label_visibility="collapsed", key="fasta_fii_val")
        pub_on = f3.checkbox("Public holding % >", pub_saved is not None, key="fasta_pub_on")
        pub_val = f3.number_input("public_hold_pct", 0.0, 100.0,
                                  float(pub_saved if pub_saved is not None else 10.0), 0.5,
                                  disabled=not pub_on, label_visibility="collapsed", key="fasta_pub_val")
        uf = Params(market_caps=tuple(caps_sel),
                    min_market_cap_amt=float(amt_val) if amt_on else None,
                    min_fii_hold_pct=float(fii_val) if fii_on else None,
                    min_public_hold_pct=float(pub_val) if pub_on else None)
        n_in, n_all = universe_counts(uf)
        if n_all is None:
            st.caption("Could not read the tickers table, so the stock count is unavailable.")
        elif n_in == 0:
            st.warning(f"0 of {n_all:,} stocks match these filters. Relax them before running.")
        elif n_in == n_all:
            st.success(f"All {n_all:,} stocks are included (no filter narrows the universe).")
        else:
            st.success(f"{n_in:,} of {n_all:,} stocks are included.")
        st.caption("Tick a box to apply that filter. A stock with no data in a ticked field is excluded. "
                   "Counts come from the tickers table; stocks without enough price history are dropped at run time.")

        with st.form(f"fasta_params_{form_nonce}"):
            t_strat, t_data, t_cost = st.tabs(["Strategy", "Data window", "Transaction costs"])
            with t_strat:
                c1, c2, c3 = st.columns(3)
                conf = c1.number_input("Confidence cutoff (%)", 0.0, 100.0,
                                       float(setting("confidence_cutoff", D.confidence_cutoff)), 1.0)
                halt = c2.number_input("Portfolio halt (%)", -50.0, 0.0,
                                       float(setting("portfolio_halt_pct", D.portfolio_halt_pct)), 0.5)
                min_hist = c3.number_input("Min history (weeks)", 5, 200,
                                           int(setting("min_history", D.min_history)), 1)
                decay = c1.number_input("Recency decay factor", 0.5, 1.0,
                                        float(setting("decay_factor", D.decay_factor)), 0.01)
                entry_mult_default = float(setting("entry_stoploss_gtt_multiplier", D.entry_stoploss_gtt_multiplier))
                entry_mult = c2.number_input(
                    "Entry stoploss multiplier", 0.0, 1.10,
                    entry_mult_default, 0.005,
                    format="%.5f",
                    help="Base stop multiplier applied to buy_price (e.g. 0.98 sets a 2% stop below entry).")
                exit_day_mult = c3.number_input(
                    "Exitday stoploss multiplier (0 = off)", 0.0, 1.10,
                    float(setting("exit_day_mult", setting("friday_stop_loss", D.exit_day_mult))), 0.005,
                    format="%.5f",
                    help="Exit-day stop candidate: max(entry_stop, day_open * exit_day_mult). Set to 0.0 to disable.")
                stop = entry_stop_pct_from_mult(entry_mult, float(setting("stop_loss_pct", D.stop_loss_pct)))
                gap = c1.checkbox("Gap-aware stop fills", bool(setting("gap_aware_stop", D.gap_aware_stop)),
                                  help="A stock that opens below the stop price exits at that open, not at the stop.")
                eq_basket = st.checkbox(
                    "Equal-weight order basket across unique tickers",
                    value=bool(setting("equal_weight_basket", D.equal_weight_basket)),
                    help="When unchecked, the order basket sums each ticker's strategy-weighted allocation from the "
                         "picks table. When checked, capital is split equally across unique tickers.")
                strategies = st.multiselect(
                    "Strategies", list(STRATEGIES),
                    default=[k for k in setting("strategies", list(D.strategies)) if k in STRATEGIES],
                    format_func=lambda k: STRATEGIES[k].label,
                    help="\\n".join(f"- **{s.label}**: {s.description}" for s in STRATEGIES.values()))
                c1, c2 = st.columns(2)
                alloc_decay = c1.number_input(
                    "Win-ratio recency decay (per week)", 0.5, 1.0,
                    float(setting("allocation_decay", D.allocation_decay)), 0.01,
                    help="A strategy's trade n weeks old counts decay^n when scoring its win ratio. Lower = reacts faster.")
                win_prior = c2.number_input(
                    "Win-ratio prior (pseudo-trades)", 0.0, 100.0,
                    float(setting("win_ratio_prior", D.win_ratio_prior)), 1.0,
                    help="Pseudo-trades at a 50% win ratio. Keeps the split near even until a strategy has a record.")
                st.caption("Maximum allocation per ticker for each market_cap value currently present in the tickers table. "
                           "These controls are generated from the database values, so new or renamed values appear "
                           "automatically. New/unconfigured categories default to 5%. Excess is redistributed to "
                           "other eligible tickers; any amount that cannot be placed remains as cash. "
                           "Set a value to 0% for no cap on that category.")
                st.warning("Disclaimer: Point-in-time market segmentation will have look-ahead bias in historical results.")
                saved_caps = setting("market_cap_allocation_pct", {})
                market_cap_allocation_pct = {}
                if cap_opts:
                    cap_cols = st.columns(min(3, len(cap_opts)))
                    for i, market_cap_value in enumerate(cap_opts):
                        normalized_key = _normalize_market_cap(market_cap_value)
                        default_pct = float(saved_caps.get(str(market_cap_value), 5.0))
                        with cap_cols[i % len(cap_cols)]:
                            market_cap_allocation_pct[str(market_cap_value)] = st.number_input(
                                f"{market_cap_value} max allocation (%)", 0.0, 100.0,
                                default_pct, 0.5, key=f"fasta_market_cap_alloc_{normalized_key}_{form_nonce}",
                            )
                else:
                    st.info("No market_cap values found in the tickers table. No segment allocation caps are available.")
                exclude_text = st.text_area(
                    "Exclude stocks", value="\\n".join(setting("exclude_tickers", [])), height=110,
                    key=f"fasta_exclude_text_{form_nonce}", placeholder="One symbol per line or comma-separated, e.g. WELCORP",
                    help="Never traded or picked, in the back-test and in the picks. .NS / .BO is ignored.")

            with t_data:
                c1, c2, c3 = st.columns(3)
                start = c1.date_input("Start date", saved_date("start_date", date(2017, 4, 1)))
                cutoff = c2.date_input(
                    "Cutoff date", saved_date("cutoff_date", eff_today, blank_as=NO_CUTOFF),
                    min_value=NO_CUTOFF,
                    help="Set to 01/01/1970 to clear the cutoff: the run then uses all data up to the latest available date.")
                advance_days_ui = c3.selectbox(
                    "Advance date by (days)",
                    options=[0, 1, 2, 3],
                    index=[0, 1, 2, 3].index(adv_saved if adv_saved in (0, 1, 2, 3) else 0),
                    help="Advance the anchor/cutoff date by 1, 2, or 3 days (0 = no advance) so limited data rolls forward.",
                )
                held = _portfolio_symbols()
                saved_restrict = bool(setting("restrict_to_portfolio", False))
                restrict = st.checkbox(
                    "Restrict to portfolio tickers", value=saved_restrict, disabled=not held,
                    help="Run the back-test and picks only on stocks currently held (live holdings, excluding "
                         "excluded symbols). Note: the weekly capital pool is then split across fewer stocks.")
                if held:
                    st.caption(f"{len(held)} tickers in your holdings.")
                else:
                    st.caption("Holdings not loaded - go to Setup and click 'Fetch Holdings' to enable this option.")

            with t_cost:
                st.caption("Indian cash-equity delivery. Rates are assumptions - verify against your broker/exchange.")
                c1, c2, c3 = st.columns(3)
                cap = c1.number_input("Weekly capital pool (₹)", 1000.0, 1e9,
                                      float(setting("weekly_capital", D.weekly_capital)), 10_000.0)
                brk = c2.number_input("Brokerage rate", 0.0, 0.05,
                                      float(setting("brokerage_rate", D.brokerage_rate)), 1e-4, format="%.6f")
                brk_cap = c3.number_input("Brokerage cap per order (₹, 0 = none)", 0.0, 1e5,
                                          float(setting("brokerage_cap", D.brokerage_cap)), 10.0)
                stt_b = c1.number_input("STT buy rate", 0.0, 0.01,
                                        float(setting("stt_buy_rate", D.stt_buy_rate)), 1e-4, format="%.6f")
                stt_s = c2.number_input("STT sell rate", 0.0, 0.01,
                                        float(setting("stt_sell_rate", D.stt_sell_rate)), 1e-4, format="%.6f")
                exch = c3.number_input("Exchange txn rate", 0.0, 0.01,
                                       float(setting("exchange_txn_rate", D.exchange_txn_rate)), 1e-6, format="%.7f")
                sebi = c1.number_input("SEBI turnover rate", 0.0, 0.01,
                                       float(setting("sebi_turnover_rate", D.sebi_turnover_rate)), 1e-6, format="%.7f")
                stamp = c2.number_input("Stamp duty (buy) rate", 0.0, 0.01,
                                        float(setting("stamp_duty_buy_rate", D.stamp_duty_buy_rate)), 1e-5, format="%.6f")
                gst = c3.number_input("GST rate", 0.0, 1.0,
                                      float(setting("gst_rate", D.gst_rate)), 0.01)
                dp = c1.number_input("DP charge per sell (₹)", 0.0, 500.0,
                                     float(setting("dp_charge_per_sell", D.dp_charge_per_sell)), 0.1)
                idle_rate_pct = c2.number_input(
                    "Idle capital interest (% p.a.)", 0.0, 30.0,
                    float(setting("idle_interest_rate", D.idle_interest_rate)) * 100.0, 0.1,
                    help="Capital that is not invested earns this annual rate, accrued daily (rate / 365): "
                         "undeployed weekly capital for the whole week, and sale proceeds from the sell date "
                         "to the next Monday. Added to weekly PnL.")
                compound_wk_cap = st.checkbox(
                    "Compound weekly capital (rebase on both profit and loss)",
                    value=bool(setting("compound_weekly_capital", D.compound_weekly_capital)),
                    help="When enabled, each week's capital base is rebased to prior base capital + prior week net PnL (both gains and losses).")

            # Keep all four actions on one line. Run Analysis retains the app's
            # primary red styling; the utility actions use Indigo and Emerald.
            st.markdown(
                fasta_css.ACTION_BUTTON,
                unsafe_allow_html=True,
            )
            cache_col, revert_col, save_col, run_col = st.columns(4)
            with cache_col:
                clear_cache_clicked = st.form_submit_button(
                    "🧹 Clear Cache", type="secondary", key="fasta_clear_cache", **_W
                )
            with revert_col:
                revert_clicked = st.form_submit_button(
                    "↩ Revert Settings", type="secondary", key="fasta_revert_settings", **_W
                )
            with save_col:
                save_clicked = st.form_submit_button(
                    "💾 Save Settings", type="secondary", key="fasta_save_settings", **_W
                )
            with run_col:
                submitted = st.form_submit_button(
                    "▶ Run Analysis", type="primary", key="fasta_run_analysis", **_W
                )

    if st.session_state.pop("fasta_revert_notice", False):
        st.success("Reverted to the last saved FASTA settings.")
    if revert_clicked:
        st.session_state["fasta_revert_pending"] = True
        st.rerun()

    p = Params(
        start_date=start.isoformat(),
        cutoff_date="" if cutoff <= NO_CUTOFF else cutoff.isoformat(),
        restrict_tickers=tuple(held) if (restrict and held) else (),
        confidence_cutoff=conf, entry_stoploss_gtt_multiplier=entry_mult,
        exit_day_mult=exit_day_mult, stop_loss_pct=stop, gap_aware_stop=gap,
        equal_weight_basket=eq_basket, strategies=tuple(strategies),
        allocation_decay=alloc_decay, win_ratio_prior=win_prior,
        market_cap_allocation_pct=market_cap_allocation_pct,
        portfolio_halt_pct=halt, min_history=int(min_hist), decay_factor=decay,
        exclude_tickers=_parse_symbols(exclude_text),
        market_caps=tuple(caps_sel),
        min_market_cap_amt=float(amt_val) if amt_on else None,
        min_fii_hold_pct=float(fii_val) if fii_on else None,
        min_public_hold_pct=float(pub_val) if pub_on else None,
        weekly_capital=cap, brokerage_rate=brk, brokerage_cap=brk_cap,
        stt_buy_rate=stt_b, stt_sell_rate=stt_s, exchange_txn_rate=exch,
        sebi_turnover_rate=sebi, stamp_duty_buy_rate=stamp, gst_rate=gst, dp_charge_per_sell=dp, idle_interest_rate=idle_rate_pct / 100.0,
        compound_weekly_capital=bool(compound_wk_cap),
        advance_days=int(advance_days_ui))

    if save_clicked:
        settings_payload = asdict(p)
        settings_payload["restrict_to_portfolio"] = bool(restrict)
        if _save_settings(**settings_payload):
            st.success(f"Settings saved to {SETTINGS_FILE}. You can manually edit this JSON file to reverse or adjust values.")
        else:
            st.error(f"Could not save FASTA settings to {SETTINGS_FILE}.")
    return p, submitted, clear_cache_clicked


def _clear_caches():
    reset_connection()
    st.session_state.pop(RESULT_KEY, None)
    removed = 0
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("cache_*.parquet"):
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    return removed


# ==========================================
# REPORT SECTIONS
# ==========================================
def _bare(ticker):
    return re.sub(r"\.(NS|BO)$", "", ticker, flags=re.IGNORECASE)


def _basket(picks, capital, equal_weight=False, market_cap_caps_pct=None):
    """Build order basket across unique tickers.

    When `equal_weight=True`, capital is split equally across unique tickers.
    When `equal_weight=False`, allocations from `picks` are summed per unique ticker
    so they match the strategy-weighted `allocation` column in the picks table.
    """
    if equal_weight:
        df = picks.drop_duplicates(subset=['ticker'], keep='first').copy()
        df['allocation'] = capital / max(len(df), 1)
    else:
        aggregations = {'price': ('price', 'first'), 'allocation': ('allocation', 'sum')}
        if 'market_cap' in picks.columns:
            aggregations['market_cap'] = ('market_cap', 'first')
        df = picks.groupby('ticker', as_index=False, sort=False).agg(**aggregations)
    if equal_weight and 'market_cap' in picks.columns:
        cap_lookup = picks.drop_duplicates('ticker').set_index('ticker')['market_cap']
        df['market_cap'] = df['ticker'].map(cap_lookup)
    df = _apply_market_cap_ticker_caps(df, 'allocation', capital, market_cap_caps_pct)
    ok = df['price'] > 0
    df['qty'] = np.where(ok, np.floor(df['allocation'] / df['price'].where(ok, 1.0)), 0).astype(int)
    out = df[['ticker', 'price', 'allocation', 'qty']].copy()
    out['invested'] = out['qty'] * out['price']
    denom = capital if capital > 0 else 1.0
    out['alloc_pct'] = out['allocation'] / denom * 100.0
    out['invested_pct'] = out['invested'] / denom * 100.0
    return out


def _show_picks(res):
    st.subheader(res["picks_title"])
    picks = res["picks"]
    if picks.empty:
        st.info("No active buy picks for the current week matching the criteria.")
        return
    split = picks.groupby('strategy').agg(n=('ticker', 'count'), weight=('strategy_weight', 'first'))
    st.caption("Weekly pool split: " + " · ".join(
        f"{STRATEGIES[k].label if k in STRATEGIES else k} {r.weight:.0%} ({r.n} picks)" for k, r in split.iterrows()))
    # CARRY = a pick that is already held in the live portfolio: it can be kept through the
    # Friday exit instead of being sold and re-bought on Monday.
    held_now = set(_portfolio_symbols())
    picks = picks.copy()
    picks.insert(1, "carry", [_bare(str(t)).upper() in held_now for t in picks["ticker"]])
    if held_now:
        n_carry = int(picks["carry"].sum())
        carry_names = ", ".join(_bare(str(t)) for t in picks.loc[picks["carry"], "ticker"].unique())
        st.caption(f"Carry candidates (picked and already held): {n_carry} of {len(picks)} picks"
                   + (f" - {carry_names}" if n_carry else "") + ".")
    else:
        st.caption("Carry check unavailable: holdings not loaded (Setup → Fetch Holdings).")
    c1, c2 = st.columns([5, 2])
    with c1:
        _table(picks, column_config={
            "confidence": st.column_config.NumberColumn("confidence %", format="%.2f"),
            "price": st.column_config.NumberColumn("price", format="%.2f"),
            "strategy_win_ratio": st.column_config.NumberColumn("win ratio", format="%.3f"),
            "strategy_weight": st.column_config.NumberColumn("capital weight", format="%.3f"),
            "allocation": st.column_config.NumberColumn("allocation ₹", format="%.0f")})
    with c2:
        eq_weight = bool(res["params"].get("equal_weight_basket", False))
        default_pick_cap = (
            float(res["kpis"]["rebased_capital_end"])
            if res.get("kpis") and res["params"].get("compound_weekly_capital")
            else float(res["params"]["weekly_capital"])
        )
        capital = st.number_input("Capital (₹)", 0.0, 1e10, default_pick_cap, 10_000.0,
                                  key="fasta_pick_capital",
                                  help=("Equal weight across unique tickers. " if eq_weight else
                                        "Summed strategy allocations per unique ticker (scaled to Capital when changed). ")
                                       + "Quantity = whole shares at the latest close.")
        basket = _basket(
            picks, capital, equal_weight=eq_weight,
            market_cap_caps_pct=res["params"].get("market_cap_allocation_pct", {}),
        )
        deployed = float(basket['invested'].sum())
        deployed_pct = (deployed / capital * 100.0) if capital > 0 else 0.0
        st.caption(f"Available: ₹{capital:,.0f}  \nDeployed: ₹{deployed:,.0f} ({deployed_pct:.1f}%)  \n"
                   f"Unused: ₹{max(capital - deployed, 0.0):,.0f} ({max(100.0 - deployed_pct, 0.0):.1f}%)")
        mode = st.radio("List as", ["Quantity", "Amount ₹", "Percentage %"], horizontal=True, key="fasta_pick_mode",
                        help="Quantity: whole shares at the last close. Amount: the rupee allocation per ticker. "
                             "Percentage: each ticker's allocation as a % of Capital.")
        if mode == "Quantity":
            st.caption("Ticker list (copy): TICKER⇥+qty")
            zero_mask = basket['qty'] == 0
            ordered_basket = pd.concat([basket[~zero_mask], basket[zero_mask]], ignore_index=True)
            st.code("\n".join(f"{_bare(t)}\t+{q}" for t, q in zip(ordered_basket['ticker'], ordered_basket['qty'])), language=None)
            zero_qty = int(zero_mask.sum())
            if zero_qty:
                st.caption(f"Included all {len(basket)} picks; {zero_qty} pick(s) have quantity 0 because the allocation buys less than one share (listed at the end).")
        elif mode == "Amount ₹":
            st.caption("Ticker list (copy): TICKER⇥+amount")
            st.code("\n".join(f"{_bare(t)}\t+{a:.0f}" for t, a in zip(basket['ticker'], basket['allocation'])), language=None)
        else:
            st.caption("Ticker list (copy): TICKER⇥+% of Capital")
            st.code("\n".join(f"{_bare(t)}\t+{pc:.1f}" for t, pc in zip(basket['ticker'], basket['alloc_pct'])), language=None)
    with st.expander("Order basket detail"):
        _table(basket, column_config={
            "price": st.column_config.NumberColumn("last close", format="%.2f"),
            "allocation": st.column_config.NumberColumn("allocation ₹", format="%.0f"),
            "alloc_pct": st.column_config.NumberColumn("allocation %", format="%.1f"),
            "invested": st.column_config.NumberColumn("invested ₹", format="%.0f"),
            "invested_pct": st.column_config.NumberColumn("invested %", format="%.1f")})
        st.caption(f"Invested ₹{basket['invested'].sum():,.0f} of ₹{capital:,.0f} "
                   f"(the rest is whole-share rounding).")

    show_news(picks)


def _show_performance(res, p):
    k = res["kpis"]
    c = st.columns(4)
    c[0].metric("Initial capital (₹)", f"{p.weekly_capital:,.0f}",
                help="Capital the back-test starts with in week 1 (the weekly capital pool).")
    c[1].metric("Invested capital (₹)", f"{k['capital_deployed']:,.0f}",
                help="Total capital put into trades, summed over every week (each week's position capital).")
    c[2].metric("Net PnL (₹)", _signed(k["net_abs"], money=True))
    c[3].metric("Net return on capital traded", _signed(k["net_return_on_capital_pct"], pct=True))
    c = st.columns(5)
    c[0].metric("Completed trades", f"{k['total_trades']:,}")
    c[1].metric("Avg net PnL / trade", _signed(k["avg_net_pnl_pct"], pct=True))
    c[2].metric("Cumulative realised PnL", _signed(k["overall_pnl_pct"], pct=True),
                help="Sum of weekly portfolio returns across all weeks (includes idle-capital interest).")
    c[3].metric("Transaction costs (₹)", f"{k['total_costs']:,.2f}")
    c[4].metric("Idle capital interest (₹)", _signed(k.get("interest_abs", 0.0), money=True),
                help="Interest earned on capital that was not invested (annual rate accrued daily).")
    c = st.columns(4)
    gw, lw = k["max_gain_week"], k["max_loss_week"]
    bq, wq = k["best_quarter"], k["worst_quarter"]
    c[0].metric("Best week", _signed(gw[1], pct=True), f"₹{gw[2]:+,.0f}", help=gw[0])
    c[1].metric("Worst week", _signed(lw[1], pct=True), f"₹{lw[2]:+,.0f}", help=lw[0])
    c[2].metric("Best quarter", _signed(bq[1], pct=True), f"₹{bq[2]:+,.0f}", help=bq[0])
    c[3].metric("Worst quarter", _signed(wq[1], pct=True), f"₹{wq[2]:+,.0f}", help=wq[0])
    st.caption(f"Rebased capital: ₹{k.get('rebased_capital_end', p.weekly_capital):,.2f} · "
               f"Gross PnL ₹{k['gross_abs']:+,.2f} · Avg net PnL/trade ₹{k['avg_net_abs_per_trade']:+,.2f} · "
               f"Best week {gw[0]} · Worst week {lw[0]} · Best quarter {bq[0]} · Worst quarter {wq[0]}")

    st.subheader("Portfolio vs Nifty 50")
    st.line_chart(res["chart"])
    st.caption("Cumulative sum of weekly returns; Nifty is shown over the weeks the strategy traded.")

    st.subheader("Strategies")
    _table(res["strategy_stats"], column_config={
        c: st.column_config.NumberColumn(format="%+.2f" if c in ("avg net PnL %", "net ₹") else "%.2f")
        for c in ("win rate %", "avg net PnL %", "net ₹", "capital ₹", "avg capital share %", "win ratio now %")})
    st.line_chart(res["win_ratio_chart"])
    st.caption("Recency-weighted win ratio per strategy, scored on earlier weeks only. Each week's capital is split "
               "between the strategies that have trades that week in proportion to these ratios.")

    cq, cy = st.columns(2)
    fmt = {"Portfolio PnL %": st.column_config.NumberColumn(format="%+.2f"),
           "Nifty 50 %": st.column_config.NumberColumn(format="%+.2f"),
           "Alpha %": st.column_config.NumberColumn(format="%+.2f")}
    with cq:
        st.subheader("Quarterly")
        _table(res["quarterly"], column_config=fmt)
    with cy:
        st.subheader("Yearly")
        _table(res["yearly"], column_config=fmt)

    st.subheader("Exits")
    ss = res["stop_stats"]
    msg = (f"{ss['n_stops']} stop exits: {ss['at_stop']} filled exactly at {p.stop_loss_pct:+.2f}% "
           f"(opened above the stop, traded down through it), {ss['gaps']} gap-down fills"
           + (f" (average {ss['gap_avg']:+.2f}%)" if ss['gap_avg'] is not None else "")
           + f", {ss['nodata']} without daily data, {ss['below_stop_weekly_exit']} closed below the stop at the weekly exit.")
    st.info(msg + ("" if res["gap_check_on"] else "  Gap-aware fill pricing is OFF."))
    _table(res["exit_stats"], column_config={
        "avg_pnl_pct": st.column_config.NumberColumn(format="%+.2f"),
        "avg_net_pnl_pct": st.column_config.NumberColumn(format="%+.2f"),
        "net_rupees": st.column_config.NumberColumn("net ₹", format="%+,.2f")})


def _set_weeklog_week(week_key, selected_week):
    """Update the selected week in a Streamlit button callback."""
    st.session_state[week_key] = selected_week


def _download_all_trades_button(res, key, compact=False):
    """Download every trade of the complete run (all weeks, no filters) as CSV."""
    all_trades = res["trades"].drop(columns=_DROP_FOR_DISPLAY, errors="ignore")
    label = ("⬇ All trades (CSV)" if compact
             else f"⬇ Download ALL trades - all weeks, complete run (CSV) · {len(all_trades):,} trades")
    st.download_button(
        label,
        all_trades.to_csv(index=False).encode(),
        "fasta_all_trades_complete_run.csv", "text/csv",
        key=key, disabled=all_trades.empty, use_container_width=compact,
        help=f"All {len(all_trades):,} trades across all weeks, complete run",
    )


def _show_weekly(res):
    trades, weekly = res["trades"], res["weekly"].sort_values('week_period', ascending=False)
    weeks = [str(w) for w in weekly['week_period']]
    if not weeks:
        st.info("No weekly performance data is available.")
        return

    # Keep the selected week in session state so the dropdown and navigation
    # buttons remain synchronized across Streamlit reruns.
    week_key = "fasta_weeklog_selected_week"
    if st.session_state.get(week_key) not in weeks:
        st.session_state[week_key] = weeks[0]

    prev_col, week_col, next_col, dl_col = st.columns([1.4, 3.0, 1.4, 1.8], vertical_alignment="center")
    with prev_col:
        current_idx = weeks.index(st.session_state[week_key])
        st.button(
            "← Previous Week",
            key="fasta_weeklog_previous",
            disabled=current_idx >= len(weeks) - 1,
            use_container_width=True,
            on_click=_set_weeklog_week,
            args=(week_key, weeks[current_idx + 1]) if current_idx < len(weeks) - 1 else (),
        )

    with week_col:
        chosen = st.selectbox( "Week:",label_visibility="collapsed",
                               options=weeks, key=week_key,)
    with next_col:
        current_idx = weeks.index(chosen)
        st.button(
            "Next Week →",
            key="fasta_weeklog_next",
            disabled=current_idx <= 0,
            use_container_width=True,
            on_click=_set_weeklog_week,
            args=(week_key, weeks[current_idx - 1]) if current_idx > 0 else (),
        )
    with dl_col:
        _download_all_trades_button(res, "fasta_dl_all_trades_weekly", compact=True)
    row = weekly[weekly['week_period'].astype(str) == chosen].iloc[0]
    c = st.columns(4)
    c[0].metric("Portfolio PnL", _signed(row.portfolio_ret, pct=True))
    c[1].metric("Nifty 50", _signed(row.nifty_ret, pct=True))
    c[2].metric("Net PnL incl. interest (₹)", _signed(row.net_rupees, money=True),
                help=f"Includes idle-capital interest of ₹{row.interest_rupees:,.2f}.")
    c[3].metric("Trades", int(row.trades))
    if int(row.trades) == 0:
        st.info("No completed entry trades were recorded for this week. Trades whose exit window "
                "extends into the current/incomplete week are excluded from historical performance.")
    g = trades[trades['week_period'].astype(str) == chosen].drop(columns=_DROP_FOR_DISPLAY)
    _table(g)
    with st.expander("All weeks - summary"):
        w = weekly[['week_period', 'trades', 'portfolio_ret', 'nifty_ret', 'interest_rupees', 'net_rupees']].rename(columns={
            'week_period': 'week', 'portfolio_ret': 'portfolio %', 'nifty_ret': 'nifty %',
            'interest_rupees': 'interest ₹', 'net_rupees': 'net ₹ (incl. interest)'})
        _table(w)


def _show_trades(res):
    trades = res["trades"].drop(columns=_DROP_FOR_DISPLAY)
    c1, c2 = st.columns(2)
    tickers = c1.multiselect("Ticker", sorted(trades['ticker'].unique()))
    reasons = c2.multiselect("Exit reason", sorted(trades['exit_reason'].unique()))
    if tickers:
        trades = trades[trades['ticker'].isin(tickers)]
    if reasons:
        trades = trades[trades['exit_reason'].isin(reasons)]
    st.caption(f"{len(trades):,} trades")
    _table(trades)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("⬇ Download filtered view (CSV)", trades.to_csv(index=False).encode(),
                           "fasta_trades_filtered.csv", "text/csv", key="fasta_dl_filtered_trades")
    with d2:
        _download_all_trades_button(res, "fasta_dl_all_trades_tab")


def _show_stop_checker(default_entry_mult=0.98, default_exit_day_mult=0.99, default_stop=-2.0):
    st.caption("Shows the daily open/low of one ticker-week and what the stop logic does with it.")
    with st.form("fasta_stopcheck"):
        c1, c2, c3 = st.columns(3)
        ticker = c1.text_input("Ticker", placeholder="MAXHEALTH.NS")
        week = c2.date_input("Week start (Monday)", date.today())
        entry = c3.number_input("Entry price (0 = entry-day average)", 0.0, 1e7, 0.0, 0.5)
        c4, c5, c6 = st.columns(3)
        entry_m = c4.number_input("Entry stoploss GTT multiplier", 0.0, 1.10, float(default_entry_mult), 0.005, format="%.5f")
        exit_day_m = c6.number_input("Exitday stoploss multiplier (0 = off)", 0.0, 1.10, float(default_exit_day_mult), 0.005, format="%.5f")
        go = st.form_submit_button("Check")
    if not go:
        return
    if not ticker.strip():
        st.warning("Enter a ticker.")
        return
    stop = entry_stop_pct_from_mult(entry_m, float(default_stop))
    try:
        out = stop_check(
            get_connection(), ticker.strip(), week, entry or None, stop,
            entry_stoploss_gtt_multiplier=entry_m, exit_day_mult=exit_day_m,
                                                    )
    except Exception as exc:
        st.error(f"Could not run the check: {exc}")
        return
    if out is None:
        st.warning(f"No daily rows for {ticker} in the week of {pd.Timestamp(week).date()} "
                   "(check the ticker spelling / date).")
        return
    st.write(f"**{ticker}** | week of {out['week_start'].date()} | entry {out['entry']:,.2f} | "
             f"stop {stop:+.2f}% = {out['stop_px']:,.2f}")
    _table(out["daily"])
    fill = out["fill"]
    if fill is None:
        st.success("The stop is never reached this week: no stop exit.")
    elif fill[1] in ("STOP_OPENGAP", "STOCK_STOP_LOSS_GAP"):
        st.error(f"{fill[2].date()} ({fill[1]}): it opened at {fill[0]:,.2f}, below the stop, so the exit is at that open "
                 f"({(fill[0] / out['entry'] - 1) * 100:+.2f}%), not at {stop:+.2f}%.")
    else:
        st.warning(f"{fill[2].date()} ({fill[1]}): it opened ABOVE the stop and then traded down through it, so the stop order "
                   f"fills at the stop price {fill[0]:,.2f} ({(fill[0] / out['entry'] - 1) * 100:+.2f}%).")


# ==========================================
# PAGE ENTRY POINT
# ==========================================
def render():
    st.header("⚙️ FASTA ALGO")
    st.caption("Multi-strategy weekly-rotation back-test with Nifty 50 benchmark. "
               "Signals use strictly past, completed weeks (no look-ahead).")

    tab_report, tab_stop = st.tabs(["📊 Back-test report", "🛑 Stop-loss checker"])

    with tab_report:
        p, submitted, clear_cache_clicked = _parameter_form()

        if clear_cache_clicked:
            n = _clear_caches()
            st.toast(f"Cleared {n} cache file(s). Run the analysis again.")
            st.rerun()

        if submitted and not p.strategies:
            st.warning("Select at least one strategy.")
            submitted = False
        if submitted:
            log_lines = []
            bar = st.progress(0.0, text="Loading data…")
            try:
                res = analyze(p, progress=lambda f, t: bar.progress(min(max(f, 0.0), 1.0), text=t),
                              log=log_lines.append)
            except Exception as exc:
                bar.empty()
                st.error(f"Analysis failed: {exc}")
                st.exception(exc)
            else:
                bar.empty()
                res["log"] = log_lines
                st.session_state[RESULT_KEY] = res


        res = st.session_state.get(RESULT_KEY)
        if res is None:
            st.info("Set the parameters and press **Run analysis**. The first run can take a while; "
                    "market data is cached to parquet afterwards.")
        else:
            rp = Params(**res["params"])
            st.caption(f"Window {rp.start_date} → {rp.cutoff_date or 'latest data'} · data through {res['data_through']:%Y-%m-%d} "
                       + f"· strategies: {', '.join(STRATEGIES[k].label if k in STRATEGIES else k for k in rp.strategies)} "
                         f"· confidence ≥ {rp.confidence_cutoff:g}% "
                         f"· entry mult {rp.entry_stoploss_gtt_multiplier:g} ({rp.stop_loss_pct:+.2f}%) "
                       + (f"· exit_day_mult {rp.exit_day_mult:g} " if rp.exit_day_mult > 0 else "· exit_day_mult OFF ")
                       + f"· ₹{rp.weekly_capital:,.0f}/week"
                       + (" · weekly capital compounded (profit & loss)"
                          if rp.compound_weekly_capital else "")
                       + (" · equal-weight basket" if rp.equal_weight_basket else " · strategy-weighted basket")
                       + (f" · portfolio only ({len(rp.restrict_tickers)} tickers)" if rp.restrict_tickers else "")
                       + (f" · {len(rp.exclude_tickers)} excluded" if rp.exclude_tickers else "")
                       + (f" · cap {'/'.join(rp.market_caps)}" if rp.market_caps else "")
                       + (f" · mcap amt > {rp.min_market_cap_amt:g}" if rp.min_market_cap_amt is not None else "")
                       + (f" · FII > {rp.min_fii_hold_pct:g}%" if rp.min_fii_hold_pct is not None else "")
                       + (f" · public > {rp.min_public_hold_pct:g}%" if rp.min_public_hold_pct is not None else ""))

            if res["kpis"] is None:
                st.warning("No completed transactions generated with current rules.")
                _show_picks(res)
            else:
                t_picks, t_perf, t_week, t_trades, t_text = st.tabs(
                    ["📌 Picks", "📈 Performance", "🗓️ Weekly log", "🧾 All trades", "📄 Text report"])
                with t_picks:
                    _show_picks(res)
                with t_perf:
                    _show_performance(res, rp)
                with t_week:
                    _show_weekly(res)
                with t_trades:
                    _show_trades(res)
                with t_text:
                    st.download_button("⬇ Download fasta.txt", res["text_report"].encode(),
                                       "fasta.txt", "text/plain")
                    st.text_area("Report", res["text_report"], height=500, label_visibility="collapsed")

            if res["log"]:
                with st.expander("Run log"):
                    st.code("\n".join(res["log"]), language=None)

    with tab_stop:
        res = st.session_state.get(RESULT_KEY)
        default_stop = res["params"]["stop_loss_pct"] if res else Params().stop_loss_pct
        default_entry_mult = res["params"].get("entry_stoploss_gtt_multiplier", Params().entry_stoploss_gtt_multiplier) if res else Params().entry_stoploss_gtt_multiplier
        default_exit_day_mult = res["params"].get("exit_day_mult", Params().exit_day_mult) if res else Params().exit_day_mult
        _show_stop_checker(default_entry_mult, default_exit_day_mult, default_stop)