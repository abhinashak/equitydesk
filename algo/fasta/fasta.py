import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURATION & PARAMETERS
# ==========================================
DB_SQL_PATH = "sqls/all.sql"
OUTPUT_FILE = "outputs/fasta/fasta.txt"
START_DATE = "2017-04-01"
CUTOFF_DATE = "2026-11-18"


MARKET_CACHE_FILE = "outputs/fasta/cache_market_df.parquet"
NIFTY_CACHE_FILE = "outputs/fasta/cache_nifty_df.parquet"
DAILY_STOP_CACHE_FILE = "outputs/fasta/cache_daily_stop_df.parquet"

# ==========================================
# TRANSACTION COST MODEL (Indian cash equity delivery)
# Rates are configurable; verify broker/exchange rates before relying on results.
# Position sizing is needed to allocate fixed DP charges to a trade.
# ==========================================
WEEKLY_POSITION_CAPITAL = 100_000.0  # ₹ total capital pool per trading week, split across selected stocks
BROKERAGE_RATE = 0.0          # Delivery brokerage rate; 0 for zero-brokerage broker
BROKERAGE_CAP = 0.0           # ₹ cap per order; 0 means no cap (when rate is 0, brokerage is 0)
STT_BUY_RATE = 0.001           # 0.10%
STT_SELL_RATE = 0.001          # 0.10%
EXCHANGE_TXN_RATE = 0.0000325    # 0.00325% (editable assumption)
SEBI_TURNOVER_RATE = 0.000001   # ₹10 per crore = 0.0001%
STAMP_DUTY_BUY_RATE = 0.00015   # 0.015% buy-side only
GST_RATE = 0.18                 # 18% on brokerage + exchange + SEBI charges
DP_CHARGE_PER_SELL = 18.80      # ₹ estimate including GST, per delivery sell transaction

def apply_transaction_costs(trades_df):
    """Apply delivery transaction costs using a ₹100,000 total weekly capital pool.

    Each week's capital is split equally among the executed positions for that week.
    """
    if trades_df.empty:
        return trades_df

    df = trades_df.copy()
    df['buy_date'] = pd.to_datetime(df['buy_date'])
    df['week_period'] = df['buy_date'].dt.to_period('W')
    weekly_trade_count = df.groupby('week_period')['ticker'].transform('count')
    df['position_capital'] = WEEKLY_POSITION_CAPITAL / weekly_trade_count
    buy_value = df['position_capital'].astype(float)
    sell_value = buy_value * (1.0 + df['pnl_pct'].astype(float) / 100.0)
    sell_value = sell_value.clip(lower=0.0)

    brokerage_buy = buy_value * BROKERAGE_RATE
    brokerage_sell = sell_value * BROKERAGE_RATE
    if BROKERAGE_CAP > 0:
        brokerage_buy = brokerage_buy.clip(upper=BROKERAGE_CAP)
        brokerage_sell = brokerage_sell.clip(upper=BROKERAGE_CAP)

    stt_buy = buy_value * STT_BUY_RATE
    stt_sell = sell_value * STT_SELL_RATE
    exchange_buy = buy_value * EXCHANGE_TXN_RATE
    exchange_sell = sell_value * EXCHANGE_TXN_RATE
    sebi_buy = buy_value * SEBI_TURNOVER_RATE
    sebi_sell = sell_value * SEBI_TURNOVER_RATE
    stamp_buy = buy_value * STAMP_DUTY_BUY_RATE
    stamp_sell = 0.0
    gst_buy = (brokerage_buy + exchange_buy + sebi_buy) * GST_RATE
    gst_sell = (brokerage_sell + exchange_sell + sebi_sell) * GST_RATE
    dp_sell = DP_CHARGE_PER_SELL

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

# Strategy Rules & Filters
CONFIDENCE_CUTOFF = 72.0
STOCK_STOP_LOSS_PCT = -2.00     # Stop-loss threshold
GAP_AWARE_STOP = True           # True: a stock that OPENS below the stop price exits at that open (gap-down), not at the stop
PORTFOLIO_HALT_PCT = -4.0
MIN_HISTORY_PERIODS = 10
DECAY_FACTOR = 0.98


def initialize_database():
    con = duckdb.connect()
    if Path(DB_SQL_PATH).exists():
        def run_sql_script(connection, path, _seen=None):
            path = Path(path).expanduser().resolve()
            _seen = _seen or set()
            if path in _seen: return
            _seen = _seen | {path}
            buf = []
            def flush():
                sql = "\n".join(buf).strip()
                buf.clear()
                if sql: connection.execute(sql)
            for line in path.read_text().splitlines():
                stripped = line.strip()
                at_boundary = not buf or "\n".join(buf).rstrip().endswith(";")
                if stripped.startswith(".") and at_boundary:
                    flush()
                    parts = stripped.split(maxsplit=1)
                    if parts[0] == ".read" and len(parts) == 2:
                        target = Path(parts[1].strip().strip("'\"")).expanduser()
                        run_sql_script(connection, target, _seen)
                else:
                    buf.append(line)
            flush()
        run_sql_script(con, DB_SQL_PATH)
    return con


def load_market_data(con):
    if Path(MARKET_CACHE_FILE).exists():
        try:
            cached_df = pd.read_parquet(MARKET_CACHE_FILE)
            if {'open_price', 'first_date', 'last_date'}.issubset(cached_df.columns):
                print("[Cache Hit] Loading market data from parquet cache...")
                return cached_df
            else:
                print("[Cache Notice] Stale cache detected lacking required columns. Rebuilding cache...")
                Path(MARKET_CACHE_FILE).unlink()
        except Exception:
            pass

    print("[Cache Miss] Executing DuckDB query for market data...")
    from fasta_sql import build_weekly_momentum_sql
    sql = build_weekly_momentum_sql(start_date=START_DATE, cutoff_date=CUTOFF_DATE)
    df = con.execute(sql).fetchdf()
    df.columns = [c.lower() for c in df.columns]
    df.to_parquet(MARKET_CACHE_FILE, index=False)
    return df

def compound_pct(returns_pct):
    """Compound a series of percentage returns: (prod(1 + r/100) - 1) * 100."""
    return ((1.0 + pd.Series(returns_pct, dtype=float) / 100.0).prod() - 1.0) * 100.0

def load_nifty_benchmark(con):
    cache_path = Path(NIFTY_CACHE_FILE)
    if cache_path.exists():
        nifty_df = pd.read_parquet(cache_path)
    else:
        from fasta_sql import build_nifty_sql
        nifty_sql = build_nifty_sql(cutoff_date=CUTOFF_DATE)
        try:
            nifty_df = con.execute(nifty_sql).fetchdf()
            if not nifty_df.empty:
                nifty_df.columns = [c.lower() for c in nifty_df.columns]
                nifty_df.to_parquet(NIFTY_CACHE_FILE, index=False)
        except Exception:
            return {}, {}, {}

    if not nifty_df.empty:
        nifty_df['w_date'] = pd.to_datetime(nifty_df['w_date'])
        weekly_ret = pd.Series(nifty_df.nifty_ret.values, index=nifty_df['w_date'].dt.to_period('W')).to_dict()
        q_sum = nifty_df.groupby(nifty_df['w_date'].dt.to_period('Q'))['nifty_ret'].apply(compound_pct).to_dict()
        y_sum = nifty_df.groupby(nifty_df['w_date'].dt.to_period('Y'))['nifty_ret'].apply(compound_pct).to_dict()
        return weekly_ret, q_sum, y_sum
    return {}, {}, {}


def load_daily_stop_data(con):
    """Daily open/low per ticker, used to fill a stop at the OPEN when a stock gaps down through the stop price.
    Returns {ticker: (dates, opens, lows)}, or None when disabled / the daily open+low columns are missing
    (the simulation then falls back to the old exact-stop fill)."""
    if not GAP_AWARE_STOP:
        return None
    try:
        cols = {r[0].lower() for r in con.execute("DESCRIBE ticker_prices").fetchall()}
        if not {'ticker', 'date', 'open', 'low'}.issubset(cols):
            print("[Gap-aware stop] ticker_prices has no daily open/low columns - using exact stop fills.")
            return None
        latest = pd.to_datetime(con.execute("SELECT max(date) FROM ticker_prices").fetchone()[0]).normalize()
        daily = None
        if Path(DAILY_STOP_CACHE_FILE).exists():
            cached = pd.read_parquet(DAILY_STOP_CACHE_FILE)
            if not cached.empty and pd.to_datetime(cached['date']).max().normalize() == latest:
                daily = cached
        if daily is None:
            print("[Cache Miss] Loading daily open/low for gap-aware stop fills...")
            daily = con.execute("""
                SELECT ticker, CAST(date AS DATE) AS date, open, low
                FROM ticker_prices
                WHERE open IS NOT NULL AND low IS NOT NULL
                ORDER BY ticker, date
            """).fetchdf()
            daily.columns = [c.lower() for c in daily.columns]
            daily.to_parquet(DAILY_STOP_CACHE_FILE, index=False)
        return {ticker: (pd.to_datetime(g['date']).values.astype('datetime64[ns]'),
                         g['open'].to_numpy(dtype=float), g['low'].to_numpy(dtype=float))
                for ticker, g in daily.groupby('ticker', sort=False)}
    except Exception as exc:
        print(f"[Gap-aware stop] unavailable ({exc}) - using exact stop fills.")
        return None


def stop_fill_price(daily_stops, ticker, week_start, buy_price):
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
    stop_px = buy_price * (1.0 + (STOCK_STOP_LOSS_PCT / 100.0))
    hit = np.nonzero(lows[lo:hi] <= stop_px)[0]
    if hit.size == 0:
        return None
    k = lo + int(hit[0])
    if k > lo and opens[k] <= stop_px:
        return float(opens[k]), True, pd.Timestamp(dates[k])
    return stop_px, False, pd.Timestamp(dates[k])


def explain_stop(con, ticker, week_start, buy_price=None):
    """Prints the daily open/low of one ticker-week and what the stop logic does with it.
    python fasta.py stopcheck MAXHEALTH.NS 2026-11-16 [buy_price]"""
    week_start = pd.Timestamp(week_start).normalize()
    daily = con.execute(
        "SELECT CAST(date AS DATE) AS date, open, low, close FROM ticker_prices "
        "WHERE ticker = ? AND CAST(date AS DATE) >= ? AND CAST(date AS DATE) < ? ORDER BY date",
        [ticker, week_start.date(), (week_start + pd.Timedelta(days=7)).date()]).fetchdf()
    if daily.empty:
        print(f"No daily rows for {ticker} in the week of {week_start.date()} (check the ticker spelling / date).")
        return
    daily.columns = [c.lower() for c in daily.columns]
    entry = float(buy_price) if buy_price else float(daily['open'].iloc[0])
    stop_px = entry * (1.0 + STOCK_STOP_LOSS_PCT / 100.0)
    print(f"{ticker} | week of {week_start.date()} | entry (week open) {entry:,.2f} | stop {STOCK_STOP_LOSS_PCT:+.2f}% = {stop_px:,.2f}")
    print(f"{'date':<12}{'open':>10}{'low':>10}{'close':>10}   low<=stop   open<=stop")
    for r in daily.itertuples():
        print(f"{str(pd.Timestamp(r.date).date()):<12}{r.open:>10,.2f}{r.low:>10,.2f}{r.close:>10,.2f}   {str(r.low <= stop_px):<10}  {r.open <= stop_px}")
    rec = {ticker: (pd.to_datetime(daily['date']).values.astype('datetime64[ns]'), daily['open'].to_numpy(float), daily['low'].to_numpy(float))}
    fill = stop_fill_price(rec, ticker, week_start, entry)
    if fill is None:
        print("-> The stop is never reached this week: no stop exit.")
    elif fill[1]:
        print(f"-> {fill[2].date()} GAPPED: it opened at {fill[0]:,.2f}, below the stop, so the exit is at that open "
              f"({(fill[0] / entry - 1) * 100:+.2f}%), not at {STOCK_STOP_LOSS_PCT:+.2f}%.")
    else:
        print(f"-> {fill[2].date()}: it opened ABOVE the stop and then traded down through it, so the stop order fills at the "
              f"stop price {stop_px:,.2f} = exactly {STOCK_STOP_LOSS_PCT:+.2f}%. That is a realistic fill, not a gap.")


def get_recency_ngram_prediction(train_str, n, decay_factor=DECAY_FACTOR):
    if len(train_str) >= n:
        last_n = train_str[-n:]
        transitions = {}
        total_weight = 0.0
        max_idx = len(train_str) - n
        for i in range(max_idx):
            if train_str[i:i+n] == last_n:
                nxt = train_str[i+n]
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


def run_simulation(df, daily_stops=None):
    """Simulation ensuring Monday entry uses strictly historical completed week patterns (no look-ahead)."""
    trades = []
    current_week_picks = []

    median_volume = df['avg_volume'].median() if 'avg_volume' in df.columns else 0

    ticker_data = {}
    for ticker, grp in df.groupby('ticker'):
        grp = grp.sort_values('w_date').reset_index(drop=True)
        if len(grp) >= MIN_HISTORY_PERIODS:
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
    stock_in_shadow = set()
    portfolio_halted = False

    for w_date in all_weeks:
        week_dt = pd.to_datetime(w_date)
        week_pnl_accumulator = []
        shadow_pnl_accumulator = []

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
            if i < MIN_HISTORY_PERIODS or len(chars[:i]) < 5:
                continue

            # TRAINING STRING STRICTLY ENDS AT WEEK i-1 (Past completed weeks only)
            #train_str = "".join(chars[:i])
            train_str = "".join(chars[:i]).translate(str.maketrans("LP", "PL"))

            pred_char, prob = get_recency_ngram_prediction(train_str, n=4)
            confidence_pct = round(prob * 100, 2)
            curr_volume = volumes[i]

            was_in_shadow = ticker in stock_in_shadow  # status BEFORE this week's stop-loss can change it

            # Monday Open Entry Price
            buy_price = opens[i] if not np.isnan(opens[i]) else closes[i]
            buy_date = pd.to_datetime(dates[i])

            week_low = lows[i]
            potential_sl_pnl = ((week_low - buy_price) / buy_price) * 100

            if potential_sl_pnl <= STOCK_STOP_LOSS_PCT:
                sell_price = buy_price * (1.0 + (STOCK_STOP_LOSS_PCT / 100.0))
                sell_date = buy_date + pd.Timedelta(days=2)
                exit_reason = "STOCK_STOP_LOSS"
                pnl_pct = STOCK_STOP_LOSS_PCT
                stock_in_shadow.add(ticker)
                fill = stop_fill_price(daily_stops, ticker, week_dt, buy_price)
                if fill is not None:
                    sell_date = fill[2]                                # the day the stop was actually hit
                    if fill[1]:                                        # opened below the stop: exit at that open
                        sell_price = fill[0]
                        pnl_pct = round(((sell_price - buy_price) / buy_price) * 100, 2)
                        exit_reason = "STOCK_STOP_LOSS_GAP"
                elif daily_stops is not None:                          # gap check wanted but no daily rows for this ticker/week
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
                if pnl_pct <= STOCK_STOP_LOSS_PCT:
                    exit_reason = "STOCK_STOP_LOSS"
                    stock_in_shadow.add(ticker)

            if was_in_shadow:
                if pnl_pct > 0.0:
                    stock_in_shadow.discard(ticker)
                else:
                    continue

            if portfolio_halted:
                shadow_pnl_accumulator.append(pnl_pct)
                continue

            if pred_char == 'P' and confidence_pct >= CONFIDENCE_CUTOFF and curr_volume >= median_volume:
                trades.append({
                    'ticker': ticker,
                    'buy_date': buy_date.strftime('%Y-%m-%d'),
                    'sell_date': sell_date.strftime('%Y-%m-%d'),
                    'buy_price': round(buy_price, 2),
                    'sell_price': round(sell_price, 2),
                    'confidence': confidence_pct,
                    'exit_reason': exit_reason,
                    'pnl_pct': pnl_pct
                })
                week_pnl_accumulator.append(pnl_pct)

        if portfolio_halted:
            if shadow_pnl_accumulator:
                avg_shadow_pnl = sum(shadow_pnl_accumulator) / len(shadow_pnl_accumulator)
                if avg_shadow_pnl > 0.0:
                    portfolio_halted = False
        else:
            if week_pnl_accumulator:
                avg_week_portfolio_pnl = sum(week_pnl_accumulator) / len(week_pnl_accumulator)
                if avg_week_portfolio_pnl <= PORTFOLIO_HALT_PCT:
                    portfolio_halted = True

    latest_week = all_weeks[-1] if len(all_weeks) > 0 else None
    if latest_week is not None and not portfolio_halted:
        latest_dt = pd.to_datetime(latest_week)
        for ticker, tdict in ticker_data.items():
            if ticker in stock_in_shadow:
                continue
            dates = tdict['dates']
            closes = tdict['closes']
            chars = tdict['chars']
            idx_match = np.where(dates == latest_dt.to_datetime64())[0]
            if len(idx_match) == 0:
                continue
            i = idx_match[0]
            if i < MIN_HISTORY_PERIODS:
                continue
            train_str = "".join(chars[:i])
            pred_char, prob = get_recency_ngram_prediction(train_str, n=4)
            confidence_pct = round(prob * 100, 2)
            if pred_char == 'P' and confidence_pct >= CONFIDENCE_CUTOFF:
                current_week_picks.append({
                    'ticker': ticker,
                    'entry_date': latest_dt.strftime('%Y-%m-%d'),
                    'price': closes[i],
                    'confidence': confidence_pct
                })

    return apply_transaction_costs(pd.DataFrame(trades)), pd.DataFrame(current_week_picks)


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


def main():
    con = initialize_database()
    df = load_market_data(con)
    daily_stops = load_daily_stop_data(con)
    nifty_weekly_ret, nifty_q_summary, nifty_y_summary = load_nifty_benchmark(con)

    trades_df, picks_df = run_simulation(df, daily_stops)
    if not trades_df.empty:
        n_stops = int(trades_df['exit_reason'].str.startswith('STOCK_STOP_LOSS').sum())
        gaps = trades_df[trades_df['exit_reason'] == 'STOCK_STOP_LOSS_GAP']
        nodata = int((trades_df['exit_reason'] == 'STOCK_STOP_LOSS_NODATA').sum())
        at_stop = int(((trades_df['exit_reason'] == 'STOCK_STOP_LOSS') & (trades_df['pnl_pct'] == STOCK_STOP_LOSS_PCT)).sum())
        print(f"[Stop-loss] {n_stops} stop exits: {at_stop} filled exactly at {STOCK_STOP_LOSS_PCT:+.2f}% "
              f"(opened above the stop, traded down through it), {len(gaps)} gap-down fills"
              + (f" (average {gaps['pnl_pct'].mean():+.2f}%)" if len(gaps) else "")
              + f", {nodata} without daily data, {n_stops - at_stop - len(gaps) - nodata} closed below the stop at the weekly exit"
              + ("" if daily_stops is not None else "  [gap check OFF: daily open/low not loaded]"))

    output_lines = []
    output_lines.append("="*115)
    output_lines.append(f"  WEEKLY ROTATION TRADING LOG (Weekly ₹100,000 Capital Pool, Strict Past-Only N-Gram Features)")
    output_lines.append("="*115)

    if not trades_df.empty:
        trades_df['sort_date'] = pd.to_datetime(trades_df['sell_date'])
        trades_df = trades_df.sort_values('sort_date').reset_index(drop=True)
        trades_df['week_period'] = pd.to_datetime(trades_df['buy_date']).dt.to_period('W')
        trades_df['quarter_period'] = trades_df['sort_date'].dt.to_period('Q')
        trades_df['year_period'] = trades_df['sort_date'].dt.to_period('Y')

        overall_realized_pnl = 0.0
        total_trades = len(trades_df)

        weekly_portfolio_pnl = (trades_df.groupby('week_period')['net_profit_rupees'].sum() / WEEKLY_POSITION_CAPITAL) * 100.0
        weekly_df = weekly_portfolio_pnl.reset_index(name='portfolio_ret')
        weekly_df['quarter_period'] = weekly_df['week_period'].dt.asfreq('Q')
        weekly_df['year_period'] = weekly_df['week_period'].dt.asfreq('Y')

        for week, group in trades_df.groupby('week_period'):
            output_lines.append(f"\n--- Week: {week} ---")
            display_group = group.drop(columns=['sort_date', 'week_period', 'quarter_period', 'year_period'])
            output_lines.append(display_group.to_string(index=False))

            week_pnl = weekly_portfolio_pnl.get(week, 0.0)
            overall_realized_pnl += week_pnl

            nifty_ret_val = nifty_weekly_ret.get(week, 0.0)
            summary_line = f"  -> **End of Week Summary**: Portfolio Average PnL = {week_pnl:+.2f}% | Nifty 50 Return = {nifty_ret_val:+.2f}%"
            output_lines.append(summary_line)
            output_lines.append("_________________________________________________________________________________________________________")

            print(f"--- Week: {week} --- -> Portfolio PnL = {week_pnl:+.2f}% | Nifty 50 = {nifty_ret_val:+.2f}%")

        output_lines.append("\n" + "="*115)
        output_lines.append("  QUARTERLY PERFORMANCE SUMMARY (Portfolio vs Nifty 50)")
        output_lines.append("="*115)
        print("\n" + "="*75 + "\n  QUARTERLY PERFORMANCE SUMMARY\n" + "="*75)

        q_grouped = weekly_df.groupby('quarter_period')['portfolio_ret'].sum()
        for q in q_grouped.index:
            q_pnl = q_grouped[q]
            nifty_q_ret = nifty_q_summary.get(q, 0.0)
            q_line = f"Quarter {q} | Portfolio PnL: {q_pnl:+.2f}% | Nifty 50 Return: {nifty_q_ret:+.2f}%"
            output_lines.append(q_line)
            print(q_line)

        output_lines.append("\n" + "="*115)
        output_lines.append("  YEARLY PERFORMANCE SUMMARY (Portfolio vs Nifty 50)")
        output_lines.append("="*115)
        print("\n" + "="*75 + "\n  YEARLY PERFORMANCE SUMMARY\n" + "="*75)

        y_grouped = weekly_df.groupby('year_period')['portfolio_ret'].sum()
        for yr in y_grouped.index:
            yr_pnl = y_grouped[yr]
            nifty_yr_ret = nifty_y_summary.get(yr, 0.0)
            yr_line = f"Year {yr}    | Portfolio PnL: {yr_pnl:+.2f}% | Nifty 50 Return: {nifty_yr_ret:+.2f}%"
            output_lines.append(yr_line)
            print(yr_line)

        avg_pnl_per_trade = overall_realized_pnl / total_trades if total_trades > 0 else 0.0
        total_gross_absolute_pnl = trades_df['gross_profit_rupees'].sum()
        total_net_absolute_pnl = trades_df['net_profit_rupees'].sum()
        avg_net_absolute_pnl_per_trade = trades_df['net_profit_rupees'].mean()
        total_capital_traded = trades_df.groupby('week_period')['position_capital'].sum().sum()
        net_return_on_total_capital_pct = (total_net_absolute_pnl / total_capital_traded * 100.0) if total_capital_traded else 0.0
        # Best and worst weekly portfolio outcomes (net of transaction costs).
        weekly_net_rupees = trades_df.groupby('week_period')['net_profit_rupees'].sum()
        max_gain_week = weekly_portfolio_pnl.idxmax()
        max_loss_week = weekly_portfolio_pnl.idxmin()
        max_gain_pct = weekly_portfolio_pnl.loc[max_gain_week]
        max_loss_pct = weekly_portfolio_pnl.loc[max_loss_week]
        max_gain_rupees = weekly_net_rupees.loc[max_gain_week]
        max_loss_rupees = weekly_net_rupees.loc[max_loss_week]

        summary_footer = f"""
=========================================================================
  OVERALL CUMULATIVE REALISED PnL ACROSS ALL WEEKS : {overall_realized_pnl:+.2f}%
  AVERAGE PnL PER COMPLETED TRADE                : {avg_pnl_per_trade:+.2f}%
  TOTAL COMPLETED TRADES EXECUTED                : {total_trades}
  TOTAL TRANSACTION COSTS (₹)                    : {trades_df['total_transaction_cost'].sum():,.2f}
  TOTAL GROSS ABSOLUTE PnL (₹)                   : {total_gross_absolute_pnl:+,.2f}
  TOTAL NET ABSOLUTE PnL (₹)                     : {total_net_absolute_pnl:+,.2f}
  AVERAGE NET ABSOLUTE PnL / TRADE (₹)           : {avg_net_absolute_pnl_per_trade:+,.2f}
  TOTAL CAPITAL DEPLOYED ACROSS TRADES (₹)       : {total_capital_traded:,.2f}
  NET RETURN ON TOTAL CAPITAL TRADED             : {net_return_on_total_capital_pct:+.2f}%
  AVERAGE NET PnL PER TRADE                      : {trades_df['net_pnl_pct'].mean():+.2f}%
  MAXIMUM GAIN WEEK                              : {max_gain_week} | {max_gain_pct:+.2f}% | ₹{max_gain_rupees:+,.2f}
  MAXIMUM LOSS WEEK                              : {max_loss_week} | {max_loss_pct:+.2f}% | ₹{max_loss_rupees:+,.2f}
  
=========================================================================
"""
        output_lines.append(summary_footer)
        print(summary_footer)
    else:
        msg = "No completed transactions generated with current rules."
        output_lines.append(msg)
        print(msg)

    wk_start, wk_end, is_next = pick_window(df)
    picks_title = (f"{'NEXT-WEEK' if is_next else 'CURRENT-WEEK'} PICKS for week {wk_start:%a %Y-%m-%d} to {wk_end:%a %Y-%m-%d} "
                   f"(signals at Thursday close | CARRY = already held)")
    print("\n" + "="*75)
    print("  " + picks_title)
    print("="*75)
    output_lines.append("\n" + "="*75 + "\n  " + picks_title + "\n" + "="*75)

    if not picks_df.empty:
        picks_df = picks_df.sort_values('confidence', ascending=False).reset_index(drop=True)
        picks_str = picks_df.to_string(index=False)
        print(picks_str)
        output_lines.append(picks_str)
    else:
        no_picks = "No active buy picks for the current week matching criteria."
        print(no_picks)
        output_lines.append(no_picks)
    print("="*75)
    output_lines.append("="*75)

    pick_tickers = picks_df['ticker'].tolist() if not picks_df.empty else []
    output_lines.extend(pick_tickers)

    Path(OUTPUT_FILE).write_text("\n".join(output_lines))
    print(f"\n[Details successfully saved to {OUTPUT_FILE}]")

    # HTML report (UI lives in fasta_report.py and is built straight from the DataFrames above)
    try:
        from fasta_report import render_html_report
    except ImportError:
        render_html_report = None
        print("[fasta_report.py not found - HTML report skipped]")
    if render_html_report:
        html_file = render_html_report(
            trades_df=trades_df, picks_df=picks_df,
            output_path=str(Path(OUTPUT_FILE).with_suffix('.html')),
            nifty_weekly_ret=nifty_weekly_ret, nifty_q_summary=nifty_q_summary, nifty_y_summary=nifty_y_summary,
            weekly_capital=WEEKLY_POSITION_CAPITAL, title=output_lines[1].strip(), picks_title=picks_title)
        print(f"[HTML report saved to {html_file}]")
    for t in pick_tickers:
        print(t)

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4 and sys.argv[1] == "stopcheck":      # python fasta.py stopcheck MAXHEALTH.NS 2026-11-16 [buy_price]
        explain_stop(initialize_database(), sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    else:
        main()