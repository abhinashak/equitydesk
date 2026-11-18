import duckdb
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION & PARAMETERS
# ==========================================
DB_SQL_PATH = "sqls/all.sql"
OUTPUT_FILE = "fasta.txt"
START_DATE = "2017-04-01"
CUTOFF_DATE = "2026-11-06"

# Strategy Rules & Filters
CONFIDENCE_CUTOFF = 75.0        # Minimum prediction probability (75%)
STOP_LOSS_PCT = -2.0            # Drop threshold (%) to remove stock from portfolio
MIN_HISTORY_PERIODS = 10        # Minimum weeks required before trading a ticker
DECAY_FACTOR = 0.98             # Recency-weighting decay for N-gram


def initialize_database():
    """Initializes DuckDB and executes setup SQL script if present."""
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
    """Loads normalized weekly close prices and volume metrics from DuckDB."""
    sql = f"""
    WITH weekly_data AS (
        SELECT
            ticker,
            date_trunc('week', date) AS w_date,
            arg_max(close, date) AS close_price,
            avg(volume) AS avg_volume
        FROM ticker_prices 
        WHERE date < DATE '{CUTOFF_DATE}'
        GROUP BY ticker, date_trunc('week', date)
    ),
    with_lag AS (
        SELECT
            ticker,
            w_date,
            close_price,
            avg_volume,
            lag(close_price, 1) OVER (PARTITION BY ticker ORDER BY w_date) AS prev_close
        FROM weekly_data
    ),
    weekly_ret AS (
        SELECT
            ticker,
            w_date,
            close_price,
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
            close_price,
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
            close_price,
            avg_volume,
            current_month_ret,
            prior_month_ret,
            row_number() OVER (PARTITION BY w_date ORDER BY prior_month_ret DESC) AS rn
        FROM with_prior_ret
    )
    SELECT
        ticker,
        CAST(w_date AS DATE) AS w_date,
        close_price,
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
    return df


def load_nifty_benchmark(con):
    """Loads Nifty 50 weekly benchmark performance."""
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
            nifty_df['w_date'] = pd.to_datetime(nifty_df['w_date'])
            weekly_ret = pd.Series(nifty_df.nifty_ret.values, index=nifty_df['w_date'].dt.to_period('W')).to_dict()
            q_sum = nifty_df.groupby(nifty_df['w_date'].dt.to_period('Q'))['nifty_ret'].sum().to_dict()
            y_sum = nifty_df.groupby(nifty_df['w_date'].dt.to_period('Y'))['nifty_ret'].sum().to_dict()
            return weekly_ret, q_sum, y_sum
    except Exception:
        pass
    return {}, {}, {}


def get_recency_ngram_prediction(train_str, n, decay_factor=DECAY_FACTOR):
    """Computes recency-weighted N-gram prediction and confidence probability."""
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
    """Runs weekly rotation simulation incorporating volume filter, stop-loss, and confidence cutoff."""
    trades = []
    current_week_picks = []
    
    # Calculate volume threshold dynamically (e.g., median volume across universe)
    median_volume = df['avg_volume'].median() if 'avg_volume' in df.columns else 0

    for ticker, group in df.groupby('ticker'):
        group = group.sort_values('w_date').reset_index(drop=True)
        if len(group) < MIN_HISTORY_PERIODS:
            continue
            
        history = list(group['bucket_char'])
        dates = list(group['w_date'])
        prices = list(group['close_price'])
        volumes = list(group['avg_volume']) if 'avg_volume' in group.columns else [median_volume] * len(group)
        
        active_buy = None
        
        for i in range(5, len(group)):
            train_str = "".join(history[:i])
            monday_date = pd.to_datetime(dates[i])
            curr_price = prices[i]
            curr_volume = volumes[i]
            
            pred_char, prob = get_recency_ngram_prediction(train_str, n=4)
            confidence_pct = round(prob * 100, 2)
            
            # Check Active Position for Exit (Weekend Exit or Stop-loss trigger)
            if active_buy is not None:
                friday_date = active_buy['buy_date'] + pd.Timedelta(days=4)
                pnl_pct = round(((curr_price - active_buy['buy_price']) / active_buy['buy_price']) * 100, 2)
                
                # Exit condition: normal weekend exit OR hit 2% drop stop-loss rule
                if pnl_pct <= STOP_LOSS_PCT:
                    exit_reason = "STOP_LOSS"
                else:
                    exit_reason = "WEEKEND_EXIT"
                
                trades.append({
                    'ticker': ticker,
                    'buy_date': active_buy['buy_date'].strftime('%Y-%m-%d'),
                    'sell_date': friday_date.strftime('%Y-%m-%d'),
                    'buy_price': round(active_buy['buy_price'], 2),
                    'sell_price': round(curr_price, 2),
                    'confidence': active_buy['confidence'],
                    'exit_reason': exit_reason,
                    'pnl_pct': pnl_pct
                })
                active_buy = None

            # Entry condition: Predicted 'P', confidence >= 75%, and sufficient volume liquidity
            if pred_char == 'P' and confidence_pct >= CONFIDENCE_CUTOFF and curr_volume >= median_volume:
                active_buy = {
                    'buy_date': monday_date,
                    'buy_price': curr_price,
                    'confidence': confidence_pct
                }
                
        if active_buy is not None:
            current_week_picks.append({
                'ticker': ticker,
                'entry_date': active_buy['buy_date'].strftime('%Y-%m-%d'),
                'price': active_buy['buy_price'],
                'confidence': active_buy['confidence']
            })
            
    return pd.DataFrame(trades), pd.DataFrame(current_week_picks)


def main():
    con = initialize_database()
    df = load_market_data(con)
    nifty_weekly_ret, nifty_q_summary, nifty_y_summary = load_nifty_benchmark(con)
    
    trades_df, picks_df = run_simulation(df)
    
    output_lines = []
    output_lines.append("="*115)
    output_lines.append(f"  WEEKLY ROTATION TRADING LOG ({CONFIDENCE_CUTOFF}% Confidence Cutoff + Volume Filter + {STOP_LOSS_PCT}% Stop-Loss)")
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
        msg = "No completed transactions generated with >= 75% confidence and volume filters."
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
