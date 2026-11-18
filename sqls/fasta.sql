WITH monthly_closes AS (
    SELECT
        ticker,
        date_trunc('week', date) AS m_date,
        arg_max(close, date) AS close_price
    FROM ticker_prices
    GROUP BY ticker, date_trunc('week', date)
),
with_lag AS (
    SELECT
        ticker,
        m_date,
        close_price,
        lag(close_price, 1) OVER (PARTITION BY ticker ORDER BY m_date) AS prev_close
    FROM monthly_closes
),
monthly_ret AS (
    SELECT
        ticker,
        m_date,
        close_price,
        prev_close,
        (close_price - prev_close) / prev_close * 100 AS ret_1m
    FROM with_lag
    WHERE prev_close IS NOT NULL
),
with_prior_ret AS (
    SELECT
        ticker,
        m_date,
        close_price,
        prev_close,
        ret_1m AS current_month_ret,
        lag(ret_1m, 1) OVER (PARTITION BY ticker ORDER BY m_date) AS prior_month_ret
    FROM monthly_ret
),
ranked_momentum AS (
    SELECT
        ticker,
        m_date,
        current_month_ret,
        prior_month_ret,
        row_number() OVER (PARTITION BY m_date ORDER BY prior_month_ret DESC) AS rn
    FROM with_prior_ret
),
bucketed AS (
    SELECT
        m_date,
        ticker,
        current_month_ret,
        CASE 
            WHEN current_month_ret < -10 THEN 'A'
            WHEN current_month_ret >= -10 AND current_month_ret < -7 THEN 'B'
            WHEN current_month_ret >= -7 AND current_month_ret < -5 THEN 'C'
            WHEN current_month_ret >= -5 AND current_month_ret < -4 THEN 'D'
            WHEN current_month_ret >= -4 AND current_month_ret < -3 THEN 'E'
            WHEN current_month_ret >= -3 AND current_month_ret < -2 THEN 'F'
            WHEN current_month_ret >= -2 AND current_month_ret < -1 THEN 'G'
            WHEN current_month_ret >= -1 AND current_month_ret < 0 THEN 'H'
            WHEN current_month_ret >= 0 AND current_month_ret < 1 THEN 'I'
            WHEN current_month_ret >= 1 AND current_month_ret < 2 THEN 'J'
            WHEN current_month_ret >= 2 AND current_month_ret < 3 THEN 'K'
            WHEN current_month_ret >= 3 AND current_month_ret < 4 THEN 'L'
            WHEN current_month_ret >= 4 AND current_month_ret < 5 THEN 'M'
            WHEN current_month_ret >= 5 AND current_month_ret < 6 THEN 'N'
            WHEN current_month_ret >= 6 AND current_month_ret < 7 THEN 'O'
            WHEN current_month_ret >= 7 AND current_month_ret < 8 THEN 'P'
            WHEN current_month_ret >= 8 AND current_month_ret < 9 THEN 'Q'
            WHEN current_month_ret >= 9 AND current_month_ret < 10 THEN 'R'
            ELSE 'S'
        END AS bucket_char
    FROM ranked_momentum
    WHERE m_date BETWEEN '2017-04-01' AND '2026-10-01'
      AND prior_month_ret IS NOT NULL
)
SELECT
    ticker,
    count(*) AS total_periods,
    -- Aggregates the chronological sequence of buckets into a single string separated by empty space (or change '' to ',' for commas)
    string_agg(bucket_char, '' ORDER BY m_date ASC) AS "bucket_string"
FROM bucketed
GROUP BY ticker
ORDER BY ticker ASC;
