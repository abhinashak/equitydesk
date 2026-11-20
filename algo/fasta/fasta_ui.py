"""FASTA ALGO - weekly rotation back-test, rendered as a Streamlit page.

Hook-up (app.py):
    elif page == "algo_fasta":
        from utils.fasta import render; render()

Multi-strategy: the final selection combines several strategies (see STRATEGIES). Every selected strategy
is run on the same market data and, each week, the capital pool is split between them in proportion to
their recency-weighted win ratio. To add one: subclass `Strategy`, implement `visit()`, decorate with `@register`.

The strategy / simulation / cost-model logic is unchanged from the command line
version. What changed is how it is driven and shown:
  * module-level constants  -> `Params` (editable in the UI)
  * print() output          -> on-screen metrics, charts and tables
  * `fasta.py stopcheck ..` -> "Stop-loss checker" tab
  * HTML/text report files  -> on-screen report (text report still written + downloadable)
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# STATIC CONFIGURATION (paths)
# ==========================================
DB_SQL_PATH = "sqls/all.sql"
OUTPUT_FILE = "outputs/fasta/fasta.txt"
CACHE_DIR = Path("outputs/fasta")
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

    # Strategy rules & filters
    confidence_cutoff: float = 72.0
    stop_loss_pct: float = -2.00      # stock stop-loss threshold
    gap_aware_stop: bool = True       # a stock that OPENS below the stop exits at that open, not at the stop
    portfolio_halt_pct: float = -4.0
    min_history: int = 10
    decay_factor: float = 0.98

    # Strategies & capital split (keys from the STRATEGIES registry)
    strategies: tuple = ("trend_shift_off", "trend_shift_on")
    allocation_decay: float = 0.97     # per-week decay applied to a strategy's past trades when scoring its win ratio
    win_ratio_prior: float = 5.0       # pseudo-trades at a 50% win ratio; keeps a strategy's first trades from swinging the split

    # Transaction cost model (Indian cash equity delivery) - verify against your broker/exchange
    weekly_capital: float = 100_000.0  # total capital pool per trading week, split across selected stocks
    brokerage_rate: float = 0.0
    brokerage_cap: float = 0.0         # per order; 0 = no cap
    stt_buy_rate: float = 0.001        # 0.10%
    stt_sell_rate: float = 0.001       # 0.10%
    exchange_txn_rate: float = 0.0000325
    sebi_turnover_rate: float = 0.000001
    stamp_duty_buy_rate: float = 0.00015
    gst_rate: float = 0.18
    dp_charge_per_sell: float = 18.80


def _fasta_sql():
    try:
        from algo.fasta import fasta_sql as mod
    except ImportError:
        import fasta_sql as mod
    return mod


# ==========================================
# TRANSACTION COSTS
# ==========================================
def apply_transaction_costs(trades_df, p: Params):
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


# ==========================================
# DATA ACCESS
# ==========================================
def _run_sql_script(connection, path, _seen=None):
    """Runs a .sql file, following `.read other.sql` includes."""
    path = Path(path).expanduser().resolve()
    _seen = _seen or set()
    if path in _seen:
        return
    _seen = _seen | {path}
    buf = []

    def flush():
        sql = "\n".join(buf).strip()
        buf.clear()
        if sql:
            connection.execute(sql)

    for line in path.read_text().splitlines():
        stripped = line.strip()
        at_boundary = not buf or "\n".join(buf).rstrip().endswith(";")
        if stripped.startswith(".") and at_boundary:
            flush()
            parts = stripped.split(maxsplit=1)
            if parts[0] == ".read" and len(parts) == 2:
                target = Path(parts[1].strip().strip("'\"")).expanduser()
                _run_sql_script(connection, target, _seen)
        else:
            buf.append(line)
    flush()


def initialize_database():
    if not Path(DB_SQL_PATH).exists():
        raise FileNotFoundError(
            f"'{DB_SQL_PATH}' not found (looked relative to {Path.cwd()}). "
            "Start the app from the project root or fix DB_SQL_PATH.")
    con = duckdb.connect()
    _run_sql_script(con, DB_SQL_PATH)
    return con


@st.cache_resource(show_spinner="Initialising database…")
def get_connection():
    return initialize_database()


def _cache_paths(p: Params):
    tag = f"{p.start_date}_{p.cutoff_date}"
    return (CACHE_DIR / f"cache_market_df_{tag}.parquet",
            CACHE_DIR / f"cache_nifty_df_{p.cutoff_date}.parquet",
            CACHE_DIR / "cache_daily_stop_df.parquet")


def load_market_data(con, p: Params, log):
    market_cache, _, _ = _cache_paths(p)
    if market_cache.exists():
        try:
            cached_df = pd.read_parquet(market_cache)
            if {'open_price', 'first_date', 'last_date'}.issubset(cached_df.columns):
                log("[Cache Hit] Loaded market data from parquet cache.")
                return cached_df
            log("[Cache Notice] Stale cache lacking required columns. Rebuilding…")
            market_cache.unlink()
        except Exception:
            pass

    log("[Cache Miss] Executing DuckDB query for market data…")
    sql = _fasta_sql().build_weekly_momentum_sql(start_date=p.start_date, cutoff_date=p.cutoff_date)
    df = con.execute(sql).fetchdf()
    df.columns = [c.lower() for c in df.columns]
    market_cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(market_cache, index=False)
    return df


def compound_pct(returns_pct):
    """Compound a series of percentage returns: (prod(1 + r/100) - 1) * 100."""
    return ((1.0 + pd.Series(returns_pct, dtype=float) / 100.0).prod() - 1.0) * 100.0


def load_nifty_benchmark(con, p: Params, log):
    _, cache_path, _ = _cache_paths(p)
    if cache_path.exists():
        nifty_df = pd.read_parquet(cache_path)
    else:
        nifty_sql = _fasta_sql().build_nifty_sql(cutoff_date=p.cutoff_date)
        try:
            nifty_df = con.execute(nifty_sql).fetchdf()
            if not nifty_df.empty:
                nifty_df.columns = [c.lower() for c in nifty_df.columns]
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                nifty_df.to_parquet(cache_path, index=False)
        except Exception as exc:
            log(f"[Nifty] benchmark unavailable ({exc}) - benchmark shown as 0.")
            return {}, {}, {}

    if not nifty_df.empty:
        nifty_df['w_date'] = pd.to_datetime(nifty_df['w_date'])
        weekly_ret = pd.Series(nifty_df.nifty_ret.values, index=nifty_df['w_date'].dt.to_period('W')).to_dict()
        q_sum = nifty_df.groupby(nifty_df['w_date'].dt.to_period('Q'))['nifty_ret'].apply(compound_pct).to_dict()
        y_sum = nifty_df.groupby(nifty_df['w_date'].dt.to_period('Y'))['nifty_ret'].apply(compound_pct).to_dict()
        return weekly_ret, q_sum, y_sum
    return {}, {}, {}


def load_daily_stop_data(con, p: Params, log):
    """Daily open/low per ticker, used to fill a stop at the OPEN when a stock gaps down through the stop price.
    Returns {ticker: (dates, opens, lows)}, or None when disabled / daily open+low columns are missing."""
    if not p.gap_aware_stop:
        return None
    _, _, daily_cache = _cache_paths(p)
    try:
        cols = {r[0].lower() for r in con.execute("DESCRIBE ticker_prices").fetchall()}
        if not {'ticker', 'date', 'open', 'low'}.issubset(cols):
            log("[Gap-aware stop] ticker_prices has no daily open/low columns - using exact stop fills.")
            return None
        latest = pd.to_datetime(con.execute("SELECT max(date) FROM ticker_prices").fetchone()[0]).normalize()
        daily = None
        if daily_cache.exists():
            cached = pd.read_parquet(daily_cache)
            if not cached.empty and pd.to_datetime(cached['date']).max().normalize() == latest:
                daily = cached
        if daily is None:
            log("[Cache Miss] Loading daily open/low for gap-aware stop fills…")
            daily = con.execute("""
                SELECT ticker, CAST(date AS DATE) AS date, open, low
                FROM ticker_prices
                WHERE open IS NOT NULL AND low IS NOT NULL
                ORDER BY ticker, date
            """).fetchdf()
            daily.columns = [c.lower() for c in daily.columns]
            daily_cache.parent.mkdir(parents=True, exist_ok=True)
            daily.to_parquet(daily_cache, index=False)
        return {ticker: (pd.to_datetime(g['date']).values.astype('datetime64[ns]'),
                         g['open'].to_numpy(dtype=float), g['low'].to_numpy(dtype=float))
                for ticker, g in daily.groupby('ticker', sort=False)}
    except Exception as exc:
        log(f"[Gap-aware stop] unavailable ({exc}) - using exact stop fills.")
        return None


# ==========================================
# STOP-LOSS LOGIC
# ==========================================
def stop_fill_price(daily_stops, ticker, week_start, buy_price, stop_loss_pct):
    """Where a stop order really fills in the week starting `week_start`.

    Finds the first trading day whose low reaches the stop price. If that day OPENED at or below the stop
    (a gap-down) the order fills at that open - a worse price than the stop; otherwise it fills at the stop.
    On the entry day the position was bought at the open, so there is no gap to suffer there.
    Returns (fill_price, gapped, fill_date), or None when there is no daily data for that ticker/week."""
    rec = daily_stops.get(ticker) if daily_stops else None
    if rec is None:
        return None
    dates, opens, lows = rec
    lo = np.searchsorted(dates, week_start.to_datetime64(), side='left')
    hi = np.searchsorted(dates, (week_start + pd.Timedelta(days=7)).to_datetime64(), side='left')
    if hi <= lo:
        return None
    stop_px = buy_price * (1.0 + (stop_loss_pct / 100.0))
    hit = np.nonzero(lows[lo:hi] <= stop_px)[0]
    if hit.size == 0:
        return None
    k = lo + int(hit[0])
    if k > lo and opens[k] <= stop_px:
        return float(opens[k]), True, pd.Timestamp(dates[k])
    return stop_px, False, pd.Timestamp(dates[k])


def stop_check(con, ticker, week_start, buy_price, stop_loss_pct):
    """Daily open/low of one ticker-week and what the stop logic does with it (was `explain_stop`)."""
    week_start = pd.Timestamp(week_start).normalize()
    daily = con.execute(
        "SELECT CAST(date AS DATE) AS date, open, low, close FROM ticker_prices "
        "WHERE ticker = ? AND CAST(date AS DATE) >= ? AND CAST(date AS DATE) < ? ORDER BY date",
        [ticker, week_start.date(), (week_start + pd.Timedelta(days=7)).date()]).fetchdf()
    if daily.empty:
        return None
    daily.columns = [c.lower() for c in daily.columns]
    entry = float(buy_price) if buy_price else float(daily['open'].iloc[0])
    stop_px = entry * (1.0 + stop_loss_pct / 100.0)
    daily['low <= stop'] = daily['low'] <= stop_px
    daily['open <= stop'] = daily['open'] <= stop_px
    rec = {ticker: (pd.to_datetime(daily['date']).values.astype('datetime64[ns]'),
                    daily['open'].to_numpy(float), daily['low'].to_numpy(float))}
    fill = stop_fill_price(rec, ticker, week_start, entry, stop_loss_pct)
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
# The engine (`run_simulation`) walks the market data once. Each selected strategy is a visitor: for every
# eligible ticker-week the engine hands it a `TickerWeek` and gets a `Signal` back. Exits, stop-losses and the
# stock shadow-ban belong to the ticker-week itself, so they are computed once and shared by all strategies.
#
# To add a strategy: subclass `Strategy`, implement `visit()`, decorate the class with `@register`.
# It then appears in the UI's strategy list and takes part in the capital split automatically.
@dataclass(frozen=True)
class TickerWeek:
    """One ticker in one signal week, as a strategy sees it.
    `history` holds only weeks completed before this week began, so a strategy cannot look ahead."""
    ticker: str
    week: pd.Timestamp
    history: np.ndarray      # bucket chars ('L' / 'P' / ...) of past completed weeks, oldest first
    volume: float            # this week's average volume (the engine applies the liquidity filter itself)


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


def run_simulation(df, daily_stops, strategies, p: Params, progress=None):
    """Monday entry uses strictly historical completed-week patterns (no look-ahead).

    Every strategy in `strategies` visits each eligible ticker-week. Returns raw (trades, picks) DataFrames
    with a `strategy` column - before capital allocation and transaction costs."""
    median_volume = df['avg_volume'].median() if 'avg_volume' in df.columns else 0

    ticker_data = {}
    for ticker, grp in df.groupby('ticker'):
        grp = grp.sort_values('w_date').reset_index(drop=True)
        if len(grp) >= p.min_history:
            opens_arr = grp['open_price'].values if 'open_price' in grp.columns else grp['close_price'].values
            lows_arr = grp['low_price'].values if 'low_price' in grp.columns else grp['close_price'].values
            ticker_data[ticker] = {
                'dates': pd.to_datetime(grp['w_date']).values,
                'opens': opens_arr,
                'closes': grp['close_price'].values,
                'lows': lows_arr,
                'volumes': grp['avg_volume'].values if 'avg_volume' in grp.columns else np.full(len(grp), median_volume),
                'chars': grp['bucket_char'].values
            }

    all_weeks = sorted(df['w_date'].unique())
    stock_in_shadow = set()  # depends only on price outcomes, so it is shared by all strategies
    runs = {s.key: _StrategyRun() for s in strategies}

    for w_idx, w_date in enumerate(all_weeks):
        if progress and w_idx % 3 == 0:
            progress(w_idx / max(len(all_weeks), 1), f"Simulating week {w_idx + 1} of {len(all_weeks)}…")
        week_dt = pd.to_datetime(w_date)
        for run in runs.values():
            run.week_pnls.clear()
            run.shadow_pnls.clear()

        for ticker, tdict in ticker_data.items():
            dates = tdict['dates']
            opens = tdict['opens']
            closes = tdict['closes']
            lows = tdict['lows']
            volumes = tdict['volumes']
            chars = tdict['chars']

            idx_match = np.where(dates == week_dt.to_datetime64())[0]
            if len(idx_match) == 0:
                continue
            i = idx_match[0]

            # Require enough history and strictly use past completed weeks up to i-1
            if i < p.min_history or len(chars[:i]) < 5:
                continue

            curr_volume = volumes[i]
            was_in_shadow = ticker in stock_in_shadow  # status BEFORE this week's stop-loss can change it

            # Monday open entry price
            buy_price = opens[i] if not np.isnan(opens[i]) else closes[i]
            buy_date = pd.to_datetime(dates[i])

            week_low = lows[i]
            potential_sl_pnl = ((week_low - buy_price) / buy_price) * 100

            if potential_sl_pnl <= p.stop_loss_pct:
                sell_price = buy_price * (1.0 + (p.stop_loss_pct / 100.0))
                sell_date = buy_date + pd.Timedelta(days=2)
                exit_reason = "STOCK_STOP_LOSS"
                pnl_pct = p.stop_loss_pct
                stock_in_shadow.add(ticker)
                fill = stop_fill_price(daily_stops, ticker, week_dt, buy_price, p.stop_loss_pct)
                if fill is not None:
                    sell_date = fill[2]                                # the day the stop was actually hit
                    if fill[1]:                                        # opened below the stop: exit at that open
                        sell_price = fill[0]
                        pnl_pct = round(((sell_price - buy_price) / buy_price) * 100, 2)
                        exit_reason = "STOCK_STOP_LOSS_GAP"
                elif daily_stops is not None:                          # gap check wanted but no daily rows
                    exit_reason = "STOCK_STOP_LOSS_NODATA"
            else:
                if i + 1 < len(closes):
                    sell_price = closes[i + 1]
                    sell_date = pd.to_datetime(dates[i + 1])
                else:
                    sell_price = closes[i]
                    sell_date = buy_date + pd.Timedelta(days=4)

                pnl_pct = round(((sell_price - buy_price) / buy_price) * 100, 2)
                exit_reason = "WEEKEND_EXIT"
                if pnl_pct <= p.stop_loss_pct:
                    exit_reason = "STOCK_STOP_LOSS"
                    stock_in_shadow.add(ticker)

            if was_in_shadow:
                if pnl_pct > 0.0:
                    stock_in_shadow.discard(ticker)
                else:
                    continue

            # Every selected strategy visits this ticker-week.
            ctx = TickerWeek(ticker, week_dt, chars[:i], curr_volume)
            for strat in strategies:
                run = runs[strat.key]
                if run.halted:
                    run.shadow_pnls.append(pnl_pct)
                    continue
                sig = strat.visit(ctx, p)
                if sig.buy and curr_volume >= median_volume:
                    run.trades.append({
                        'ticker': ticker,
                        'strategy': strat.key,
                        'buy_date': buy_date.strftime('%Y-%m-%d'),
                        'sell_date': sell_date.strftime('%Y-%m-%d'),
                        'buy_price': round(buy_price, 2),
                        'sell_price': round(sell_price, 2),
                        'confidence': sig.confidence,
                        'exit_reason': exit_reason,
                        'pnl_pct': pnl_pct
                    })
                    run.week_pnls.append(pnl_pct)

        # Portfolio halt is judged per strategy, on that strategy's own trades.
        for run in runs.values():
            if run.halted:
                if run.shadow_pnls and sum(run.shadow_pnls) / len(run.shadow_pnls) > 0.0:
                    run.halted = False
            elif run.week_pnls and sum(run.week_pnls) / len(run.week_pnls) <= p.portfolio_halt_pct:
                run.halted = True

    picks = []
    latest_week = all_weeks[-1] if len(all_weeks) > 0 else None
    if latest_week is not None:
        latest_dt = pd.to_datetime(latest_week)
        for strat in strategies:
            if runs[strat.key].halted:
                continue
            for ticker, tdict in ticker_data.items():
                if ticker in stock_in_shadow:
                    continue
                idx_match = np.where(tdict['dates'] == latest_dt.to_datetime64())[0]
                if len(idx_match) == 0:
                    continue
                i = idx_match[0]
                if i < p.min_history:
                    continue
                sig = strat.visit(TickerWeek(ticker, latest_dt, tdict['chars'][:i], tdict['volumes'][i]), p)
                if sig.buy:
                    picks.append({
                        'ticker': ticker,
                        'strategy': strat.key,
                        'entry_date': latest_dt.strftime('%Y-%m-%d'),
                        'price': tdict['closes'][i],
                        'confidence': sig.confidence
                    })

    trades = [t for run in runs.values() for t in run.trades]
    return pd.DataFrame(trades), pd.DataFrame(picks)


# ==========================================
# CAPITAL ALLOCATION ACROSS STRATEGIES
# ==========================================
# Each week the capital pool is split between the strategies that have trades that week, in proportion to their
# recency-weighted win ratio. The win ratio is scored on each strategy's stand-alone trades from EARLIER weeks
# only (no look-ahead). A trade n weeks old counts `allocation_decay ** n`; a small 50% prior
# (`win_ratio_prior` pseudo-trades) stops the first few trades from swinging the split.
_WEEK0 = pd.Timestamp("1970-01-05")  # a Monday; week numbers count Monday-anchored weeks from here
_NO_TRADES = (np.array([], dtype=np.int64), np.array([], dtype=float))


def _week_no(ts):
    return (pd.Timestamp(ts).normalize() - _WEEK0).days // 7


def _week_nos(dates):
    return ((pd.to_datetime(dates).dt.normalize() - _WEEK0).dt.days // 7).astype(np.int64)


def _track_record(trades_df):
    """{strategy: (week numbers, win flags)} from the raw trades. A win is a trade that closed above its entry."""
    if trades_df.empty:
        return {}
    weeks = _week_nos(trades_df['buy_date']).to_numpy()
    wins = (trades_df['pnl_pct'].to_numpy(dtype=float) > 0.0).astype(float)
    record = {}
    for key in trades_df['strategy'].unique():
        mask = (trades_df['strategy'] == key).to_numpy()
        record[key] = (weeks[mask], wins[mask])
    return record


def recency_win_ratio(weeks, wins, as_of, p: Params):
    """Recency-weighted share of winning trades among those from weeks before `as_of` (0.5 with no history)."""
    past = weeks < as_of
    w = p.allocation_decay ** (as_of - weeks[past])
    denom = float(w.sum()) + p.win_ratio_prior
    if denom <= 0:
        return 0.5
    return (float((w * wins[past]).sum()) + 0.5 * p.win_ratio_prior) / denom


def split_capital(win_ratios, active):
    """Share of the pool per active strategy, proportional to its win ratio. Swap this to change the policy."""
    total = sum(win_ratios[k] for k in active)
    return {k: (win_ratios[k] / total if total > 0 else 1.0 / len(active)) for k in active}


def allocate_capital(trades_df, p: Params):
    """Give each trade its slice of the weekly pool: pool x strategy share / that strategy's trades that week.
    Returns (trades with strategy_win_ratio / strategy_weight / position_capital, per-week allocation history)."""
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
    alloc['week'] = _WEEK0 + pd.to_timedelta(alloc['_wk'] * 7, unit='D')
    return df.drop(columns='_wk'), alloc.drop(columns='_wk')


def allocate_picks(picks_df, win_ratios, p: Params):
    """The same split for the live picks, using the win ratios as of the pick week."""
    df = picks_df.copy()
    weights = split_capital(win_ratios, sorted(df['strategy'].unique()))
    df['strategy_win_ratio'] = df['strategy'].map(win_ratios)
    df['strategy_weight'] = df['strategy'].map(weights)
    df['allocation'] = p.weekly_capital * df['strategy_weight'] / df.groupby('strategy')['ticker'].transform('count')
    return df


def pick_window(df):
    """Return the latest Monday-anchored signal week and Thursday-close readiness."""
    if df.empty:
        return pd.NaT, pd.NaT, False

    required = {'w_date', 'last_date'}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"pick_window() requires columns: {sorted(missing)}")

    latest_week = pd.Timestamp(pd.to_datetime(df['w_date']).max()).normalize()
    latest_data_date = pd.Timestamp(pd.to_datetime(df['last_date']).max()).normalize()
    signal_week_end = latest_week + pd.Timedelta(days=3)  # Thursday
    for_next_week = latest_data_date >= signal_week_end
    return latest_week, signal_week_end, for_next_week


# ==========================================
# ANALYSIS (everything the report needs, computed once)
# ==========================================
_DROP_FOR_DISPLAY = ['sort_date', 'week_period', 'quarter_period', 'year_period']


def analyze(p: Params, progress=None, log=lambda m: None):
    con = get_connection().cursor()  # per-run cursor: the cached connection is shared across sessions
    df = load_market_data(con, p, log)
    if df.empty:
        raise ValueError("The market-data query returned no rows for this date window.")
    daily_stops = load_daily_stop_data(con, p, log)
    nifty_weekly, nifty_q, nifty_y = load_nifty_benchmark(con, p, log)

    strategies = resolve_strategies(p.strategies)
    raw_trades, picks_df = run_simulation(df, daily_stops, strategies, p, progress)
    wk_start, wk_end, is_next = pick_window(df)
    picks_title = (f"{'NEXT-WEEK' if is_next else 'CURRENT-WEEK'} PICKS for week {wk_start:%a %Y-%m-%d} to "
                   f"{wk_end:%a %Y-%m-%d} (signals at Thursday close | CARRY = already held)")

    # Capital: each strategy's share of the weekly pool follows its recency-weighted win ratio (past weeks only).
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
        "gap_check_on": daily_stops is not None, "log": [],
    }

    if not trades_df.empty:
        # ---- stop-loss statistics (before trades_df is re-sorted) ----
        n_stops = int(trades_df['exit_reason'].str.startswith('STOCK_STOP_LOSS').sum())
        gaps = trades_df[trades_df['exit_reason'] == 'STOCK_STOP_LOSS_GAP']
        nodata = int((trades_df['exit_reason'] == 'STOCK_STOP_LOSS_NODATA').sum())
        at_stop = int(((trades_df['exit_reason'] == 'STOCK_STOP_LOSS') &
                       (trades_df['pnl_pct'] == p.stop_loss_pct)).sum())
        res["stop_stats"] = {
            "n_stops": n_stops, "at_stop": at_stop, "gaps": len(gaps),
            "gap_avg": float(gaps['pnl_pct'].mean()) if len(gaps) else None,
            "nodata": nodata, "below_stop_weekly_exit": n_stops - at_stop - len(gaps) - nodata,
        }

        trades_df['sort_date'] = pd.to_datetime(trades_df['sell_date'])
        trades_df = trades_df.sort_values('sort_date', kind='stable').reset_index(drop=True)
        trades_df['week_period'] = pd.to_datetime(trades_df['buy_date']).dt.to_period('W')
        trades_df['quarter_period'] = trades_df['sort_date'].dt.to_period('Q')
        trades_df['year_period'] = trades_df['sort_date'].dt.to_period('Y')
        res["trades"] = trades_df

        # ---- weekly ----
        weekly_net = trades_df.groupby('week_period')['net_profit_rupees'].sum()
        weekly_pct = (weekly_net / p.weekly_capital) * 100.0
        weekly_cnt = trades_df.groupby('week_period').size()
        weekly_df = pd.DataFrame({
            'week_period': weekly_pct.index, 'trades': weekly_cnt.reindex(weekly_pct.index).values,
            'portfolio_ret': weekly_pct.values, 'net_rupees': weekly_net.values})
        weekly_df['nifty_ret'] = weekly_df['week_period'].map(lambda w: nifty_weekly.get(w, 0.0))
        weekly_df['quarter_period'] = weekly_df['week_period'].dt.asfreq('Q')
        weekly_df['year_period'] = weekly_df['week_period'].dt.asfreq('Y')
        res["weekly"] = weekly_df

        # ---- quarterly / yearly ----
        q = weekly_df.groupby('quarter_period')['portfolio_ret'].sum()
        res["quarterly"] = pd.DataFrame({
            'Quarter': q.index.astype(str), 'Portfolio PnL %': q.values,
            'Nifty 50 %': [nifty_q.get(k, 0.0) for k in q.index]})
        y = weekly_df.groupby('year_period')['portfolio_ret'].sum()
        res["yearly"] = pd.DataFrame({
            'Year': y.index.astype(str), 'Portfolio PnL %': y.values,
            'Nifty 50 %': [nifty_y.get(k, 0.0) for k in y.index]})
        for key in ("quarterly", "yearly"):
            res[key]['Alpha %'] = res[key]['Portfolio PnL %'] - res[key]['Nifty 50 %']

        # ---- cumulative chart (sum of weekly returns, same convention as the quarterly/yearly roll-ups) ----
        ch = weekly_df.sort_values('week_period')
        res["chart"] = pd.DataFrame({
            'Portfolio (cumulative %)': ch['portfolio_ret'].cumsum().values,
            'Nifty 50 (cumulative %, traded weeks)': ch['nifty_ret'].cumsum().values,
        }, index=ch['week_period'].dt.start_time)

        # ---- exit reasons ----
        res["exit_stats"] = (trades_df.groupby('exit_reason')
                             .agg(trades=('ticker', 'count'), avg_pnl_pct=('pnl_pct', 'mean'),
                                  avg_net_pnl_pct=('net_pnl_pct', 'mean'), net_rupees=('net_profit_rupees', 'sum'))
                             .reset_index().sort_values('trades', ascending=False))

        # ---- per-strategy breakdown ----
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

        # ---- headline numbers ----
        total_trades = len(trades_df)
        overall = float(weekly_pct.sum())
        capital = float(trades_df.groupby('week_period')['position_capital'].sum().sum())
        net_abs = float(trades_df['net_profit_rupees'].sum())
        gw, lw = weekly_pct.idxmax(), weekly_pct.idxmin()
        res["kpis"] = {
            "overall_pnl_pct": overall,
            "avg_pnl_per_trade_pct": overall / total_trades if total_trades else 0.0,
            "total_trades": total_trades,
            "total_costs": float(trades_df['total_transaction_cost'].sum()),
            "gross_abs": float(trades_df['gross_profit_rupees'].sum()),
            "net_abs": net_abs,
            "avg_net_abs_per_trade": float(trades_df['net_profit_rupees'].mean()),
            "capital_deployed": capital,
            "net_return_on_capital_pct": (net_abs / capital * 100.0) if capital else 0.0,
            "avg_net_pnl_pct": float(trades_df['net_pnl_pct'].mean()),
            "max_gain_week": (str(gw), float(weekly_pct.loc[gw]), float(weekly_net.loc[gw])),
            "max_loss_week": (str(lw), float(weekly_pct.loc[lw]), float(weekly_net.loc[lw])),
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
         f"  WEEKLY ROTATION TRADING LOG (Weekly ₹{p.weekly_capital:,.0f} Capital Pool, Strict Past-Only N-Gram Features)",
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
    """Arrow can't serialise Period columns - show them as text."""
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


def _parameter_form():
    """Returns (Params, submitted)."""
    D = Params(cutoff_date=date.today().isoformat())
    open_it = RESULT_KEY not in st.session_state
    with st.expander("⚙️ Parameters", expanded=open_it):
        with st.form("fasta_params"):
            t_strat, t_data, t_cost = st.tabs(["Strategy", "Data window", "Transaction costs"])
            with t_strat:
                c1, c2, c3 = st.columns(3)
                conf = c1.number_input("Confidence cutoff (%)", 0.0, 100.0, D.confidence_cutoff, 1.0)
                stop = c2.number_input("Stock stop-loss (%)", -50.0, 0.0, D.stop_loss_pct, 0.25)
                halt = c3.number_input("Portfolio halt (%)", -50.0, 0.0, D.portfolio_halt_pct, 0.5)
                min_hist = c1.number_input("Min history (weeks)", 5, 200, D.min_history, 1)
                decay = c2.number_input("Recency decay factor", 0.5, 1.0, D.decay_factor, 0.01)
                gap = c3.checkbox("Gap-aware stop fills", D.gap_aware_stop,
                                  help="A stock that opens below the stop price exits at that open, not at the stop.")
                strategies = st.multiselect(
                    "Strategies", list(STRATEGIES), default=list(D.strategies),
                    format_func=lambda k: STRATEGIES[k].label,
                    help="\n".join(f"- **{s.label}**: {s.description}" for s in STRATEGIES.values()))
                c1, c2 = st.columns(2)
                alloc_decay = c1.number_input(
                    "Win-ratio recency decay (per week)", 0.5, 1.0, D.allocation_decay, 0.01,
                    help="A strategy's trade n weeks old counts decay^n when scoring its win ratio. Lower = reacts faster.")
                win_prior = c2.number_input(
                    "Win-ratio prior (pseudo-trades)", 0.0, 100.0, D.win_ratio_prior, 1.0,
                    help="Pseudo-trades at a 50% win ratio. Keeps the split near even until a strategy has a record.")
            with t_data:
                c1, c2 = st.columns(2)
                start = c1.date_input("Start date", date(2017, 4, 1))
                cutoff = c2.date_input("Cutoff date", date.today())
            with t_cost:
                st.caption("Indian cash-equity delivery. Rates are assumptions - verify against your broker/exchange.")
                c1, c2, c3 = st.columns(3)
                cap = c1.number_input("Weekly capital pool (₹)", 1000.0, 1e9, D.weekly_capital, 10_000.0)
                brk = c2.number_input("Brokerage rate", 0.0, 0.05, D.brokerage_rate, 1e-4, format="%.6f")
                brk_cap = c3.number_input("Brokerage cap per order (₹, 0 = none)", 0.0, 1e5, D.brokerage_cap, 10.0)
                stt_b = c1.number_input("STT buy rate", 0.0, 0.01, D.stt_buy_rate, 1e-4, format="%.6f")
                stt_s = c2.number_input("STT sell rate", 0.0, 0.01, D.stt_sell_rate, 1e-4, format="%.6f")
                exch = c3.number_input("Exchange txn rate", 0.0, 0.01, D.exchange_txn_rate, 1e-6, format="%.7f")
                sebi = c1.number_input("SEBI turnover rate", 0.0, 0.01, D.sebi_turnover_rate, 1e-6, format="%.7f")
                stamp = c2.number_input("Stamp duty (buy) rate", 0.0, 0.01, D.stamp_duty_buy_rate, 1e-5, format="%.6f")
                gst = c3.number_input("GST rate", 0.0, 1.0, D.gst_rate, 0.01)
                dp = c1.number_input("DP charge per sell (₹)", 0.0, 500.0, D.dp_charge_per_sell, 0.1)
            submitted = st.form_submit_button("▶ Run analysis", type="primary")

    p = Params(
        start_date=start.isoformat(), cutoff_date=cutoff.isoformat(),
        confidence_cutoff=conf, stop_loss_pct=stop, gap_aware_stop=gap, strategies=tuple(strategies),
        allocation_decay=alloc_decay, win_ratio_prior=win_prior,
        portfolio_halt_pct=halt, min_history=int(min_hist), decay_factor=decay,
        weekly_capital=cap, brokerage_rate=brk, brokerage_cap=brk_cap,
        stt_buy_rate=stt_b, stt_sell_rate=stt_s, exchange_txn_rate=exch,
        sebi_turnover_rate=sebi, stamp_duty_buy_rate=stamp, gst_rate=gst, dp_charge_per_sell=dp)
    return p, submitted


def _clear_caches():
    st.cache_resource.clear()
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
    """Copy-list ticker without the exchange suffix (RELIANCE.NS -> RELIANCE)."""
    return re.sub(r"\.(NS|BO)$", "", ticker, flags=re.IGNORECASE)


def _basket(picks, capital):
    """Whole-share order basket for `capital`: strategies share it by their capital weights, and each strategy's
    slice is split equally across its picks. One row per ticker, in the picks' order (price = latest close)."""
    df = picks.copy()
    df['allocation'] = capital * df['strategy_weight'] / df.groupby('strategy')['ticker'].transform('count')
    ok = df['price'] > 0
    df['qty'] = np.where(ok, np.floor(df['allocation'] / df['price'].where(ok, 1.0)), 0).astype(int)
    out = df.groupby('ticker', sort=False).agg(
        price=('price', 'first'), allocation=('allocation', 'sum'), qty=('qty', 'sum')).reset_index()
    out['invested'] = out['qty'] * out['price']
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
    c1, c2 = st.columns([5, 2])
    with c1:
        _table(picks, column_config={
            "confidence": st.column_config.NumberColumn("confidence %", format="%.2f"),
            "price": st.column_config.NumberColumn("price", format="%.2f"),
            "strategy_win_ratio": st.column_config.NumberColumn("win ratio", format="%.3f"),
            "strategy_weight": st.column_config.NumberColumn("capital weight", format="%.3f"),
            "allocation": st.column_config.NumberColumn("allocation ₹", format="%.0f")})
    with c2:
        capital = st.number_input("Capital (₹)", 0.0, 1e10, float(res["params"]["weekly_capital"]), 10_000.0,
                                  key="fasta_pick_capital",
                                  help="Split across the strategies by their capital weights, then equally across each "
                                       "strategy's picks. Quantity = whole shares at the latest close.")
        basket = _basket(picks, capital)
        mode = st.radio("List as", ["Quantity", "Amount ₹"], horizontal=True, key="fasta_pick_mode",
                        help="Quantity: whole shares at the last close. Amount: the rupee allocation per ticker.")
        if mode == "Quantity":
            buy = basket[basket['qty'] > 0]
            st.caption("Ticker list (copy): TICKER⇥+qty")
            st.code("\n".join(f"{_bare(t)}\t+{q}" for t, q in zip(buy['ticker'], buy['qty'])), language=None)
            if len(buy) < len(basket):
                st.caption(f"{len(basket) - len(buy)} pick(s) left out: their share buys less than one share.")
        else:
            st.caption("Ticker list (copy): TICKER⇥+amount")
            st.code("\n".join(f"{_bare(t)}\t+{a:.0f}" for t, a in zip(basket['ticker'], basket['allocation'])), language=None)
    with st.expander("Order basket detail"):
        _table(basket, column_config={
            "price": st.column_config.NumberColumn("last close", format="%.2f"),
            "allocation": st.column_config.NumberColumn("allocation ₹", format="%.0f"),
            "invested": st.column_config.NumberColumn("invested ₹", format="%.0f")})
        st.caption(f"Invested ₹{basket['invested'].sum():,.0f} of ₹{capital:,.0f} "
                   f"(the rest is whole-share rounding).")


def _show_performance(res, p):
    k = res["kpis"]
    c = st.columns(4)
    c[0].metric("Net PnL (₹)", _signed(k["net_abs"], money=True))
    c[1].metric("Net return on capital traded", _signed(k["net_return_on_capital_pct"], pct=True))
    c[2].metric("Completed trades", f"{k['total_trades']:,}")
    c[3].metric("Avg net PnL / trade", _signed(k["avg_net_pnl_pct"], pct=True))
    c = st.columns(4)
    c[0].metric("Cumulative realised PnL", _signed(k["overall_pnl_pct"], pct=True),
                help="Sum of weekly portfolio returns across all weeks.")
    c[1].metric("Transaction costs (₹)", f"{k['total_costs']:,.2f}")
    gw, lw = k["max_gain_week"], k["max_loss_week"]
    c[2].metric("Best week", _signed(gw[1], pct=True), f"₹{gw[2]:+,.0f}", help=gw[0])
    c[3].metric("Worst week", _signed(lw[1], pct=True), f"₹{lw[2]:+,.0f}", help=lw[0])
    st.caption(f"Capital deployed across trades: ₹{k['capital_deployed']:,.2f} · "
               f"Gross PnL ₹{k['gross_abs']:+,.2f} · Avg net PnL/trade ₹{k['avg_net_abs_per_trade']:+,.2f} · "
               f"Best week {gw[0]} · Worst week {lw[0]}")

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
    st.info(msg + ("" if res["gap_check_on"] else "  Gap check is OFF: daily open/low not loaded."))
    _table(res["exit_stats"], column_config={
        "avg_pnl_pct": st.column_config.NumberColumn(format="%+.2f"),
        "avg_net_pnl_pct": st.column_config.NumberColumn(format="%+.2f"),
        "net_rupees": st.column_config.NumberColumn("net ₹", format="%+,.2f")})


def _show_weekly(res):
    trades, weekly = res["trades"], res["weekly"].sort_values('week_period', ascending=False)
    weeks = [str(w) for w in weekly['week_period']]
    chosen = st.selectbox("Week", weeks, index=0)
    row = weekly[weekly['week_period'].astype(str) == chosen].iloc[0]
    c = st.columns(4)
    c[0].metric("Portfolio PnL", _signed(row.portfolio_ret, pct=True))
    c[1].metric("Nifty 50", _signed(row.nifty_ret, pct=True))
    c[2].metric("Net PnL (₹)", _signed(row.net_rupees, money=True))
    c[3].metric("Trades", int(row.trades))
    g = trades[trades['week_period'].astype(str) == chosen].drop(columns=_DROP_FOR_DISPLAY)
    _table(g)
    with st.expander("All weeks - summary"):
        w = weekly[['week_period', 'trades', 'portfolio_ret', 'nifty_ret', 'net_rupees']].rename(columns={
            'week_period': 'week', 'portfolio_ret': 'portfolio %', 'nifty_ret': 'nifty %', 'net_rupees': 'net ₹'})
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
    st.download_button("⬇ Download trades (CSV)", trades.to_csv(index=False).encode(),
                       "fasta_trades.csv", "text/csv")


def _show_stop_checker(default_stop):
    st.caption("Shows the daily open/low of one ticker-week and what the stop logic does with it.")
    with st.form("fasta_stopcheck"):
        c1, c2, c3 = st.columns(3)
        ticker = c1.text_input("Ticker", placeholder="MAXHEALTH.NS")
        week = c2.date_input("Week start (Monday)", date.today())
        entry = c3.number_input("Entry price (0 = week's first open)", 0.0, 1e7, 0.0, 0.5)
        stop = st.number_input("Stop-loss (%)", -50.0, 0.0, float(default_stop), 0.25)
        go = st.form_submit_button("Check")
    if not go:
        return
    if not ticker.strip():
        st.warning("Enter a ticker.")
        return
    try:
        out = stop_check(get_connection().cursor(), ticker.strip(), week, entry or None, stop)
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
    elif fill[1]:
        st.error(f"{fill[2].date()} GAPPED: it opened at {fill[0]:,.2f}, below the stop, so the exit is at that open "
                 f"({(fill[0] / out['entry'] - 1) * 100:+.2f}%), not at {stop:+.2f}%.")
    else:
        st.warning(f"{fill[2].date()}: it opened ABOVE the stop and then traded down through it, so the stop order "
                   f"fills at the stop price {out['stop_px']:,.2f} = exactly {stop:+.2f}%. A realistic fill, not a gap.")


# ==========================================
# PAGE ENTRY POINT
# ==========================================
def render():
    st.header("⚙️ FASTA ALGO")
    st.caption("Multi-strategy weekly-rotation back-test with Nifty 50 benchmark. "
               "Signals use strictly past, completed weeks (no look-ahead).")

    tab_report, tab_stop = st.tabs(["📊 Back-test report", "🛑 Stop-loss checker"])

    with tab_report:
        p, submitted = _parameter_form()

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

        if st.button("🔄 Clear cached data & reload database"):
            n = _clear_caches()
            st.toast(f"Cleared {n} cache file(s). Run the analysis again.")
            st.rerun()

        res = st.session_state.get(RESULT_KEY)
        if res is None:
            st.info("Set the parameters and press **Run analysis**. The first run can take a while; "
                    "market data is cached to parquet afterwards.")
        else:
            rp = Params(**res["params"])
            st.caption(f"Window {rp.start_date} → {rp.cutoff_date} · data through {res['data_through']:%Y-%m-%d} "
                       f"· strategies: {', '.join(STRATEGIES[k].label if k in STRATEGIES else k for k in rp.strategies)} "
                       f"· confidence ≥ {rp.confidence_cutoff:g}% · stop {rp.stop_loss_pct:+.2f}% "
                       f"· ₹{rp.weekly_capital:,.0f}/week")

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
        _show_stop_checker(default_stop)