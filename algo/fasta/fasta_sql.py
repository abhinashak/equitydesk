from datetime import date


def build_nifty_sql(cutoff_date: str | date) -> str:
    """Build the SQL query for weekly returns, prior-week momentum rank, and L/P buckets."""
    return f"""
        WITH nifty_closes AS (
            SELECT
                date_trunc('week', date) AS w_date,
                arg_max(close, date) AS close_price
            FROM ticker_prices
            WHERE (ticker LIKE '%NIFTY%' OR ticker LIKE '%NSEI%')
              AND date < DATE '{cutoff_date}'
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
            arg_min(CASE WHEN date IS NOT NULL THEN close END, date) AS open_price,
            arg_max(close, date) AS close_price,
            min(low) AS low_price,
            avg(volume) AS avg_volume
        FROM ticker_prices
        WHERE date < DATE '{cutoff_date}'
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
