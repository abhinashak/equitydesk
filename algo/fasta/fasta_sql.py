from datetime import date


NIFTY_TABLE = "benchmark_prices"
NIFTY_TICKER = "NIFTYBEES.NS"


def build_nifty_sql(cutoff_date: str | date) -> str:
    """Build the SQL query for weekly returns, prior-week momentum rank, and L/P buckets."""
    return f"""
        WITH nifty_closes AS (
            SELECT
                date_trunc('week', date) AS w_date,
                arg_max(close, date) AS close_price
            FROM {NIFTY_TABLE}
            WHERE ticker = '{NIFTY_TICKER}'
              AND date <= DATE '{cutoff_date}'
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


def build_weekly_momentum_sql(start_date: str | date, cutoff_date: str | date) -> str:
    """Build the SQL query for weekly returns, prior-week momentum rank, and L/P buckets."""
    return f"""
    WITH weekly_data AS (
        SELECT
            ticker,
            date_trunc('week', date) AS w_date,
            min(date) AS first_date,
            max(date) AS last_date,
            arg_min(open, date) AS open_price,
            arg_max(close, date) AS close_price,
            min(low) AS low_price,
            avg(volume) AS avg_volume
        FROM ticker_prices
        WHERE date <= DATE '{cutoff_date}'
        GROUP BY ticker, date_trunc('week', date)
    ),
    with_lag AS (
        SELECT
            ticker,
            w_date,
            first_date,
            last_date,
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
            first_date,
            last_date,
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
            first_date,
            last_date,
            close_price,
            low_price,
            avg_volume,
            prev_close,
            ret_1w AS current_week_ret,
            lag(ret_1w, 1) OVER (PARTITION BY ticker ORDER BY w_date) AS prior_week_ret
        FROM weekly_ret
    ),
    ranked_momentum AS (
        SELECT
            ticker,
            w_date,
            first_date,
            last_date,
            open_price,
            close_price,
            low_price,
            avg_volume,
            current_week_ret,
            prior_week_ret,
            row_number() OVER (PARTITION BY w_date ORDER BY prior_week_ret DESC) AS rn
        FROM with_prior_ret
    )
    SELECT
        ticker,
        CAST(w_date AS DATE) AS w_date,
        first_date,
        last_date,
        open_price,
        close_price,
        low_price,
        avg_volume,
        current_week_ret,
        CASE WHEN current_week_ret < 0 THEN 'L' ELSE 'P' END AS bucket_char
    FROM ranked_momentum
    WHERE w_date BETWEEN '{start_date}' AND '{cutoff_date}'
      AND prior_week_ret IS NOT NULL
    ORDER BY ticker, w_date ASC;
    """


def build_daily_stop_max_date_sql(with_cutoff: bool = True) -> str:
    """Latest available daily price date, optionally bounded by a cutoff."""
    if with_cutoff:
        return "SELECT max(date) FROM ticker_prices WHERE CAST(date AS DATE) <= ?"
    return "SELECT max(date) FROM ticker_prices"


def build_daily_stop_data_sql() -> str:
    """Daily OHLC fields needed by the gap-aware stop-fill model."""
    return """
        SELECT ticker, CAST(date AS DATE) AS date, open, low
        FROM ticker_prices
        WHERE open IS NOT NULL AND low IS NOT NULL
          AND CAST(date AS DATE) <= ?
        ORDER BY ticker, date
    """


def build_stop_check_sql() -> str:
    """Daily prices for the interactive single-ticker stop-loss explanation."""
    return """
        SELECT CAST(date AS DATE) AS date, open, low, close
        FROM ticker_prices
        WHERE ticker = ?
          AND CAST(date AS DATE) >= ?
          AND CAST(date AS DATE) < ?
        ORDER BY date
    """


def build_market_cap_options_sql() -> str:
    return """
        SELECT DISTINCT market_cap
        FROM tickers
        WHERE market_cap IS NOT NULL
        ORDER BY market_cap
    """


def build_universe_symbols_sql(
    market_caps=(),
    min_market_cap_amt=None,
    min_fii_hold_pct=None,
    min_public_hold_pct=None,
):
    """Return parameterized SQL and parameters for the configured ticker universe."""
    where = []
    params = []
    if market_caps:
        where.append(f"market_cap IN ({', '.join('?' for _ in market_caps)})")
        params.extend(market_caps)
    for column, value in (
        ("market_cap_amt", min_market_cap_amt),
        ("fii_hold_pct", min_fii_hold_pct),
        ("public_hold_pct", min_public_hold_pct),
    ):
        if value is not None:
            where.append(f"{column} > ?")
            params.append(float(value))

    sql = "SELECT DISTINCT upper(nse_symbol) FROM tickers WHERE nse_symbol IS NOT NULL"
    if where:
        sql += " AND " + " AND ".join(where)
    return sql, params



def build_describe_ticker_prices_sql() -> str:
    return "DESCRIBE ticker_prices"
