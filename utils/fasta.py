import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURATION & PARAMETERS
# ==========================================
DB_SQL_PATH = "sqls/all.sql"
OUTPUT_FILE = "fasta.txt"
START_DATE = "2017-04-01"
CUTOFF_DATE = "2026-11-06"

MARKET_CACHE_FILE = "cache_market_df.parquet"
NIFTY_CACHE_FILE = "cache_nifty_df.parquet"

# Strategy Rules & Filters
CONFIDENCE_CUTOFF = 75.0        
STOCK_STOP_LOSS_PCT = -0.20     # Exact stop-loss threshold
PORTFOLIO_HALT_PCT = -5.0       
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
    # If cache exists but lacks 'open_price', remove it so it rebuilds fresh
    if Path(MARKET_CACHE_FILE).exists():
        try:
            cached_df = pd.read_parquet(MARKET_CACHE_FILE)
            if 'open_price' in cached_df.columns:
                print("[Cache Hit] Loading market data from parquet cache...")
                return cached_df
            else:
                print("[Cache Notice] Stale cache detected lacking 'open_price'. Rebuilding cache...")
                Path(MARKET_CACHE_FILE).unlink()
        except Exception:
            pass

    print("[Cache Miss] Executing DuckDB query for market data...")
    sql = f"""
    WITH weekly_data AS (
        SELECT
            ticker,
            date_trunc('week', date) AS w_date,
            arg_min(case when date is not null then close end, date) AS open_price,
            arg_max(close, date) AS close_price,
            min(low) AS low_price,
            avg(volume) AS avg_volume
        FROM ticker_prices 
        WHERE date < DATE '{CUTOFF_DATE}'
        GROUP BY ticker, date_trunc('week', date)
    ),
    with_lag AS (
        SELECT
            ticker,
            w_date,
            open_price,
            close_price,
            low_price,
            avg_volume,
            lag(close_price, 1) OVER (PARTITION BY ticker ORDER BY w_date) AS prev_close
        FROM weekly_data
    ),
    weekly_ret AS (
        SELECT
            ticker,
            w_date,
            open_price,
            close_price,
            low_price,
            avg_volume,
            prev_close,
            (close_price - prev_close) / prev_close * 100 AS ret_1w
        FROM with_lag
        WHERE prev_close IS NOT NULL
    ),
    with_prior_ret AS (
        SELECT
            ticker,
            w_date,
            open_price,
            close_price,
            low_price,
            avg_volume,
            prev_close,
            ret_1w AS current_month_ret,
            lag(ret_1w, 1) OVER (PARTITION BY ticker ORDER BY w_date) AS prior_month_ret
        FROM weekly_ret
    ),
    ranked_momentum AS (
        SELECT
            ticker,
            w_date,
            open_price,
            close_price,
            low_price,
            avg_volume,
            current_month_ret,
            prior_month_ret,
            row_number() OVER (PARTITION BY w_date ORDER BY prior_month_ret DESC) AS rn
        FROM with_prior_ret
    )
    SELECT
        ticker,
        CAST(w_date AS DATE) AS w_date,
        open_price,
        close_price,
        low_price,
        avg_volume,
        current_month_ret,
        CASE WHEN current_month_ret < 0 THEN 'L' ELSE 'P' END AS bucket_char
    FROM ranked_momentum
    WHERE w_date BETWEEN '{START_DATE}' AND '2026-10-01'
      AND prior_month_ret IS NOT NULL
    ORDER BY ticker, w_date ASC;
    """
    df = con.execute(sql).fetchdf()
    df.columns = [c.lower() for c in df.columns]
    df.to_parquet(MARKET_CACHE_FILE, index=False)
    return df


def load_nifty_benchmark(con):
    cache_path = Path(NIFTY_CACHE_FILE)
    if cache_path.exists():
        nifty_df = pd.read_parquet(cache_path)
    else:
        nifty_sql = f"""
        WITH nifty_closes AS (
            SELECT
                date_trunc('week', date) AS w_date,
                arg_max(close, date) AS close_price
            FROM ticker_prices
            WHERE (ticker LIKE '%NIFTY%' OR ticker LIKE '%NSEI%')
              AND date < DATE '{CUTOFF_DATE}'
            GROUP BY date_trunc('week', date)
        ),
        nifty_lag AS (
            SELECT
                w_date,
                close_price,
                lag(close_price, 1) OVER (ORDER BY w_date) AS prev_close
            FROM nifty_closes
        )
        SELECT
            CAST(w_date AS DATE) AS w_date,
            CASE WHEN prev_close > 0 THEN ((close_price - prev_close) / prev_close) * 100 ELSE 0.0 END AS nifty_ret
        FROM nifty_lag
        WHERE prev_close IS NOT NULL;
        """
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
        q_sum = nifty_df.groupby(nifty_df['w_date'].dt.to_period('Q'))['nifty_ret'].sum().to_dict()
        y_sum = nifty_df.groupby(nifty_df['w_date'].dt.to_period('Y'))['nifty_ret'].sum().to_dict()
        return weekly_ret, q_sum, y_sum
    return {}, {}, {}


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


def run_simulation(df):
    """Simulation with intra-week low price check for strict -0.20% stop-loss execution."""
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

            if i < MIN_HISTORY_PERIODS or len(chars[:i]) < 5:
                continue

            train_str = "".join(chars[:i])
            pred_char, prob = get_recency_ngram_prediction(train_str, n=4)
            confidence_pct = round(prob * 100, 2)
            curr_volume = volumes[i]

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

            if ticker in stock_in_shadow:
                if pnl_pct > 0.0:
                    stock_in_shadow.remove(ticker)
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

    return pd.DataFrame(trades), pd.DataFrame(current_week_picks)


def main():
    con = initialize_database()
    df = load_market_data(con)
    nifty_weekly_ret, nifty_q_summary, nifty_y_summary = load_nifty_benchmark(con)
    
    trades_df, picks_df = run_simulation(df)
    
    output_lines = []
    output_lines.append("="*115)
    output_lines.append(f"  WEEKLY ROTATION TRADING LOG (Intra-Week {STOCK_STOP_LOSS_PCT}% Stop-Loss Check)")
    output_lines.append("="*115)
    
    if not trades_df.empty:
        trades_df['sort_date'] = pd.to_datetime(trades_df['sell_date'])
        trades_df = trades_df.sort_values('sort_date').reset_index(drop=True)
        trades_df['week_period'] = trades_df['sort_date'].dt.to_period('W')
        trades_df['quarter_period'] = trades_df['sort_date'].dt.to_period('Q')
        trades_df['year_period'] = trades_df['sort_date'].dt.to_period('Y')
        
        overall_realized_pnl = 0.0
        total_trades = len(trades_df)
        
        weekly_portfolio_pnl = trades_df.groupby('week_period')['pnl_pct'].mean()
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
            
        # Quarterly Summary
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

        # Yearly Summary
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
        
        summary_footer = f"""
=========================================================================
  OVERALL CUMULATIVE REALISED PnL ACROSS ALL WEEKS : {overall_realized_pnl:+.2f}%
  AVERAGE PnL PER COMPLETED TRADE                : {avg_pnl_per_trade:+.2f}%
  TOTAL COMPLETED TRADES EXECUTED                  : {total_trades}
=========================================================================
"""
        output_lines.append(summary_footer)
        print(summary_footer)
    else:
        msg = "No completed transactions generated with current rules."
        output_lines.append(msg)
        print(msg)
        
    print("\n" + "="*75)
    print("  CURRENT WEEK BUY PICKS (Active Open Positions)")
    print("="*75)
    output_lines.append("\n" + "="*75 + "\n  CURRENT WEEK BUY PICKS (Active Open Positions)\n" + "="*75)
    
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
        
    Path(OUTPUT_FILE).write_text("\n".join(output_lines))
    print(f"\n[Details successfully saved to {OUTPUT_FILE}]")

if __name__ == "__main__":
    main()
