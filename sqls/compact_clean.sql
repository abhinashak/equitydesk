-- General Info
COPY (
SELECT
    ticker,
    TRY_CAST(NULLIF(TRIM(bse_code), '') AS BIGINT)                AS bse_code,
    nse_symbol, company_name, about, broad_sector, sector, broad_industry, industry,
    TRY_CAST(NULLIF(REPLACE(TRIM(market_cap), ',', ''), '') AS DOUBLE)       AS market_cap,
    TRY_CAST(NULLIF(REPLACE(TRIM(current_price), ',', ''), '') AS DOUBLE)    AS current_price,
    TRY_CAST(NULLIF(REPLACE(TRIM(high___low), ',', ''), '') AS DOUBLE)       AS high_low,
    TRY_CAST(NULLIF(REPLACE(TRIM(stock_p_e), ',', ''), '') AS DOUBLE)        AS stock_p_e,
    TRY_CAST(NULLIF(REPLACE(TRIM(book_value), ',', ''), '') AS DOUBLE)       AS book_value,
    TRY_CAST(NULLIF(REPLACE(TRIM(dividend_yield), ',', ''), '') AS DOUBLE)   AS dividend_yield,
    TRY_CAST(NULLIF(REPLACE(TRIM(roce), ',', ''), '') AS DOUBLE)             AS roce,
    TRY_CAST(NULLIF(REPLACE(TRIM(roe), ',', ''), '') AS DOUBLE)              AS roe,
    TRY_CAST(NULLIF(REPLACE(TRIM(face_value), ',', ''), '') AS DOUBLE)       AS face_value
FROM general_info_raw
) TO 'data/fundamental_clean/general_info.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);


-- Pros & Cons (one row per bullet point per ticker)

COPY ( SELECT * FROM pros_raw ) TO 'data/fundamental_clean/pros.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);

COPY ( SELECT * FROM cons_raw ) TO 'data/fundamental_clean/cons.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);

-- Financial tables (long format: ticker | period | metric | value)
COPY (
    SELECT ticker, metric,
        CASE WHEN period = 'TTM' THEN CURRENT_DATE ELSE strptime(left(period, 8), '%b %Y') END AS dt,
        TRY_CAST( NULLIF( REPLACE( REPLACE(TRIM(value), '%', ''), ',', '' ), '' ) AS DOUBLE ) AS val
    FROM quarterly_results_raw
)
TO 'data/fundamental_clean/quarterly_results.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);


COPY (
    SELECT ticker, metric,
        CASE WHEN period = 'TTM' THEN CURRENT_DATE ELSE strptime(left(period, 8), '%b %Y') END AS dt,
        TRY_CAST( NULLIF( REPLACE( REPLACE(TRIM(value), '%', ''), ',', '' ), '' ) AS DOUBLE ) AS val
    FROM profit_loss_raw
)
TO 'data/fundamental_clean/profit_loss.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);


COPY (
    SELECT ticker, metric,
        CASE WHEN period = 'TTM' THEN CURRENT_DATE ELSE strptime(left(period, 8), '%b %Y') END AS dt,
        TRY_CAST( NULLIF( REPLACE( REPLACE(TRIM(value), '%', ''), ',', '' ), '' ) AS DOUBLE ) AS val
    FROM balance_sheet_raw
)
TO 'data/fundamental_clean/balance_sheet.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);



COPY (
    SELECT ticker, metric,
        CASE WHEN period = 'TTM' THEN CURRENT_DATE ELSE strptime(left(period, 8), '%b %Y') END AS dt,
        TRY_CAST( NULLIF( REPLACE( REPLACE(TRIM(value), '%', ''), ',', '' ), '' ) AS DOUBLE ) AS val
    FROM cash_flows_raw
)
TO 'data/fundamental_clean/cash_flows.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);



COPY (
    SELECT ticker, metric,
        CASE WHEN period = 'TTM' THEN CURRENT_DATE ELSE strptime(left(period, 8), '%b %Y') END AS dt,
        TRY_CAST( NULLIF( REPLACE( REPLACE(TRIM(value), '%', ''), ',', '' ), '' ) AS DOUBLE ) AS val
    FROM ratios_raw
)
TO 'data/fundamental_clean/ratios.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);


-- CAGR / growth tables (ticker | horizon | value)
COPY (
    SELECT
        ticker, horizon,
        CASE WHEN horizon = 'TTM' THEN 0 ELSE TRY_CAST(REPLACE(horizon, ' Years', '') AS INTEGER) END AS years,
        TRY_CAST(  NULLIF( REPLACE( REPLACE(TRIM(value), '%', ''), ',', '' ), '' ) AS DOUBLE ) AS val
    FROM compounded_sales_growth_raw
)
TO 'data/fundamental_clean/compounded_sales_growth.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);


COPY (
    SELECT
        ticker, horizon,
        CASE WHEN horizon = 'TTM' THEN 0 ELSE TRY_CAST(REPLACE(horizon, ' Years', '') AS INTEGER) END AS years,
        TRY_CAST(  NULLIF( REPLACE( REPLACE(TRIM(value), '%', ''), ',', '' ), '' ) AS DOUBLE ) AS val
    FROM compounded_profit_growth_raw
)
TO 'data/fundamental_clean/compounded_profit_growth.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);

COPY (
    SELECT
        ticker, horizon,
        CASE WHEN horizon = 'TTM' THEN 0 ELSE TRY_CAST(REPLACE(horizon, ' Years', '') AS INTEGER) END AS years,
        TRY_CAST(  NULLIF( REPLACE( REPLACE(TRIM(value), '%', ''), ',', '' ), '' ) AS DOUBLE ) AS val
    FROM stock_price_cagr_raw
)
TO 'data/fundamental_clean/stock_price_cagr.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);

COPY (
    SELECT
        ticker, horizon,
        CASE WHEN horizon = 'TTM' THEN 0 ELSE TRY_CAST(REPLACE(horizon, ' Years', '') AS INTEGER) END AS years,
        TRY_CAST(  NULLIF( REPLACE( REPLACE(TRIM(value), '%', ''), ',', '' ), '' ) AS DOUBLE ) AS val
    FROM return_on_equity_raw
)
TO 'data/fundamental_clean/return_on_equity.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);

-- Shareholding pattern (ticker | period | category | value)
COPY (
    SELECT
        ticker,category,
        CASE WHEN period = 'TTM' THEN CURRENT_DATE ELSE strptime(left(period, 8), '%b %Y') END AS dt,
        TRY_CAST( NULLIF( REPLACE( REPLACE(TRIM(value), '%', ''), ',', '' ), '' ) AS DOUBLE ) AS val
    FROM shareholding_quarterly_raw
)
TO 'data/fundamental_clean/shareholding_quarterly.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);

COPY (
    SELECT
        ticker,category,
        CASE WHEN period = 'TTM' THEN CURRENT_DATE ELSE strptime(left(period, 8), '%b %Y') END AS dt,
        TRY_CAST( NULLIF( REPLACE( REPLACE(TRIM(value), '%', ''), ',', '' ), '' ) AS DOUBLE ) AS val
    FROM shareholding_yearly_raw
)
TO 'data/fundamental_clean/shareholding_yearly.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);

