CREATE OR REPLACE VIEW general_info AS
SELECT ticker, bse_code, nse_symbol, company_name, about, broad_sector, sector, broad_industry, industry, market_cap, current_price, high_low, stock_p_e, book_value, dividend_yield, roce, roe, face_value
FROM 'data/fundamental_clean/general_info.parquet';

CREATE OR REPLACE VIEW pros AS SELECT ticker,item
FROM 'data/fundamental_clean/pros.parquet';

CREATE OR REPLACE VIEW cons AS SELECT ticker,item
FROM 'data/fundamental_clean/cons.parquet';

-- metric : Tax %,Sales,EPS in Rs,Other Income,Revenue,Financing Margin %,Gross NPA %,Operating Profit,OPM %,Net NPA %,Interest,Depreciation,Profit before tax,Expenses,Net Profit,Financing Profit
CREATE OR REPLACE VIEW quarterly_results AS
    SELECT ticker, metric, dt, val
FROM 'data/fundamental_clean/quarterly_results.parquet';

-- metric : Tax %,Sales,EPS in Rs,Other Income,Revenue,Financing Margin %,Gross NPA %,Operating Profit,OPM %,Net NPA %,Interest,Depreciation,Profit before tax,Expenses,Net Profit,Financing Profit
CREATE OR REPLACE VIEW quarterly_results AS
    SELECT ticker, metric, dt, val
FROM 'data/fundamental_clean/quarterly_results.parquet';

-- metric : Profit before tax,Financing Margin %,Sales,EPS in Rs,Other Income,Revenue,Expenses,Net Profit,Financing Profit,Operating Profit,Tax %,Interest,Depreciation,Dividend Payout %,OPM %
CREATE OR REPLACE VIEW profit_loss AS
    SELECT ticker, metric, dt, val
FROM 'data/fundamental_clean/profit_loss.parquet';

-- metric Deposits,Equity Capital,Fixed Assets,Total Assets,Borrowings,Reserves,Other Assets,Borrowing,Other Liabilities,Total Liabilities,Investments,CWIP
CREATE OR REPLACE VIEW balance_sheet AS
    SELECT ticker, metric, dt, val
FROM 'data/fundamental_clean/balance_sheet.parquet';

-- metric Free Cash Flow,Cash from Investing Activity,Cash from Financing Activity,Cash from Operating Activity,CFO/OP,Net Cash Flow
CREATE OR REPLACE VIEW cash_flows AS
    SELECT ticker, metric, dt, val
FROM 'data/fundamental_clean/cash_flows.parquet';

-- metric : ROE %,ROCE %,Debtor Days,Working Capital Days,Days Payable,Inventory Days,Cash Conversion Cycle
CREATE OR REPLACE VIEW ratios AS
    SELECT ticker, metric, dt, val
FROM 'data/fundamental_clean/ratios.parquet';

-- horizon: 10 Years,3 Years,TTM,5 Years
-- year : 3,5,0,10
CREATE OR REPLACE VIEW compounded_sales_growth AS
    SELECT ticker ,horizon ,years ,val
FROM 'data/fundamental_clean/compounded_sales_growth.parquet';

-- horizon: 10 Years,3 Years,TTM,5 Years
-- year : 3,5,0,10
CREATE OR REPLACE VIEW compounded_profit_growth AS
    SELECT ticker ,horizon ,years ,val
FROM 'data/fundamental_clean/compounded_profit_growth.parquet';

-- horizon: 1 Year, 5 Years, 3 Years, 10 Years
CREATE OR REPLACE VIEW stock_price_cagr AS
    SELECT ticker ,horizon ,years ,val
FROM 'data/fundamental_clean/stock_price_cagr.parquet';

-- horizon : 10 Years,3 Years,Last Year,5 Years
CREATE OR REPLACE VIEW return_on_equity AS
SELECT ticker ,horizon ,years ,val
FROM 'data/fundamental_clean/return_on_equity.parquet';

-- category = DIIs,Government,No. of Shareholders,FIIs,Public,Others,Promoters
CREATE OR REPLACE VIEW shareholding_quarterly AS
    SELECT ticker,category, dt, val
FROM 'data/fundamental_clean/shareholding_quarterly.parquet';

CREATE OR REPLACE VIEW shareholding_quarterly_metric AS
    SELECT ticker,category as metric, dt, val from shareholding_quarterly;

-- category = Public,DIIs,Government,No. of Shareholders,Promoters,Others,FIIs
CREATE OR REPLACE VIEW shareholding_yearly AS
    SELECT ticker,category, dt, val
FROM 'data/fundamental_clean/shareholding_yearly.parquet';

CREATE OR REPLACE VIEW ticker_prices AS
SELECT Date, Close, High, Low, Open, Volume, Ticker, split_part(Ticker, '.', 1) AS nse_symbol, year
FROM "data/ticker/year=*/*.parquet";

CREATE OR REPLACE VIEW benchmark_prices AS
SELECT Date, Close, High, Low, Open, Volume, Ticker, split_part(Ticker, '.', 1) AS nse_symbol, year
FROM "data/benchmark/year=*/*.parquet";

CREATE OR REPLACE VIEW ticker_momentum AS
SELECT
    date,     close,     "1Y",     "6M",     "3M",     "2M",     "1M",     "14D",
    "7D",     "5D",     "4D",     "3D",     "2D",     "1D",     sma20_dist,
    sma50_dist,     sma200_dist,     sma50,     sma200,     trend_stack,     vol_ratio,
     volume_trend,     trend_consistency,     dist_52w_high,     volatility,
     ulcer,     drawdown,     rsi,     breakout_strength,     gap,     acceleration,
     volume_price_signal,     kalman_dist,     vol_compression,     macd,     macd_bullish,
      atr,     atr_expansion,     adx,     strong_trend,     bb_width,     bb_squeeze,
    vwap_dist,     momentum_quality,     days_in_trend,     momentum_decay,     year,
    sma20_low,     sma20_mid,     sma20_high,     sma50_pos,     sma200_pos,     vol_low,
     vol_mid,     vol_high,     volume_trend_up,     volatility_low,     volatility_mid,
      volatility_high,     compression,     rsi_low,     rsi_mid,     rsi_high,
      ulcer_low,     ulcer_high,     drawdown_shallow,     trend_consistent,     breakout,
   kalman_positive,     gap_up,     near_52w_high,     at_52w_high,     trend_fresh,
   trend_mature,     momentum_slowing,     vps_bullish,     vps_bearish,
   momentum_accelerating,     momentum_aligned,     ticker,
   split_part(ticker, '.', 1) AS nse_symbol,     sector
FROM "data/signal_momentum/year=*/ticker_momentum.parquet";

CREATE OR REPLACE VIEW benchmark_momentum AS
SELECT
    date,     close,     "1Y",     "6M",     "3M",     "2M",     "1M",     "14D",
    "7D",     "5D",     "4D",     "3D",     "2D",     "1D",     sma20_dist,
    sma50_dist,     sma200_dist,     sma50,     sma200,     trend_stack,     vol_ratio,
     volume_trend,     trend_consistency,     dist_52w_high,     volatility,
     ulcer,     drawdown,     rsi,     breakout_strength,     gap,     acceleration,
     volume_price_signal,     kalman_dist,     vol_compression,     macd,     macd_bullish,
      atr,     atr_expansion,     adx,     strong_trend,     bb_width,     bb_squeeze,
    vwap_dist,     momentum_quality,     days_in_trend,     momentum_decay,     year,
    sma20_low,     sma20_mid,     sma20_high,     sma50_pos,     sma200_pos,     vol_low,
     vol_mid,     vol_high,     volume_trend_up,     volatility_low,     volatility_mid,
      volatility_high,     compression,     rsi_low,     rsi_mid,     rsi_high,
      ulcer_low,     ulcer_high,     drawdown_shallow,     trend_consistent,     breakout,
   kalman_positive,     gap_up,     near_52w_high,     at_52w_high,     trend_fresh,
   trend_mature,     momentum_slowing,     vps_bullish,     vps_bearish,
   momentum_accelerating,     momentum_aligned,     ticker, name as sector
FROM "data/signal_momentum/year=*/benchmark_momentum.parquet";

CREATE OR REPLACE VIEW ticker_sector_momentum AS
SELECT
    -- Sector benchmark columns
    b.date                  AS s_date,
    b.close                 AS s_close,
    b."1Y"                  AS s_1Y,
    b."6M"                  AS s_6M,
    b."3M"                  AS s_3M,
    b."2M"                  AS s_2M,
    b."1M"                  AS s_1M,
    b."14D"                 AS s_14D,
    b."7D"                  AS s_7D,
    b."5D"                  AS s_5D,
    b."4D"                  AS s_4D,
    b."3D"                  AS s_3D,
    b."2D"                  AS s_2D,
    b."1D"                  AS s_1D,
    b.sma20_dist            AS s_sma20_dist,
    b.sma50_dist            AS s_sma50_dist,
    b.sma200_dist           AS s_sma200_dist,
    b.sma50                 AS s_sma50,
    b.sma200                AS s_sma200,
    b.trend_stack           AS s_trend_stack,
    b.vol_ratio             AS s_vol_ratio,
    b.volume_trend          AS s_volume_trend,
    b.trend_consistency     AS s_trend_consistency,
    b.dist_52w_high         AS s_dist_52w_high,
    b.volatility            AS s_volatility,
    b.ulcer                 AS s_ulcer,
    b.drawdown              AS s_drawdown,
    b.rsi                   AS s_rsi,
    b.breakout_strength     AS s_breakout_strength,
    b.gap                   AS s_gap,
    b.acceleration          AS s_acceleration,
    b.volume_price_signal   AS s_volume_price_signal,
    b.kalman_dist           AS s_kalman_dist,
    b.vol_compression       AS s_vol_compression,
    b.macd                  AS s_macd,
    b.macd_bullish          AS s_macd_bullish,
    b.atr                   AS s_atr,
    b.atr_expansion         AS s_atr_expansion,
    b.adx                   AS s_adx,
    b.strong_trend          AS s_strong_trend,
    b.bb_width              AS s_bb_width,
    b.bb_squeeze            AS s_bb_squeeze,
    b.vwap_dist             AS s_vwap_dist,
    b.momentum_quality      AS s_momentum_quality,
    b.days_in_trend         AS s_days_in_trend,
    b.momentum_decay        AS s_momentum_decay,

    -- Ticker columns
    t.date                  AS t_date,
    t.close                 AS t_close,
    t."1Y"                  AS t_1Y,
    t."6M"                  AS t_6M,
    t."3M"                  AS t_3M,
    t."2M"                  AS t_2M,
    t."1M"                  AS t_1M,
    t."14D"                 AS t_14D,
    t."7D"                  AS t_7D,
    t."5D"                  AS t_5D,
    t."4D"                  AS t_4D,
    t."3D"                  AS t_3D,
    t."2D"                  AS t_2D,
    t."1D"                  AS t_1D,
    t.sma20_dist            AS t_sma20_dist,
    t.sma50_dist            AS t_sma50_dist,
    t.sma200_dist           AS t_sma200_dist,
    t.sma50                 AS t_sma50,
    t.sma200                AS t_sma200,
    t.trend_stack           AS t_trend_stack,
    t.vol_ratio             AS t_vol_ratio,
    t.volume_trend          AS t_volume_trend,
    t.trend_consistency     AS t_trend_consistency,
    t.dist_52w_high         AS t_dist_52w_high,
    t.volatility            AS t_volatility,
    t.ulcer                 AS t_ulcer,
    t.drawdown              AS t_drawdown,
    t.rsi                   AS t_rsi,
    t.breakout_strength     AS t_breakout_strength,
    t.gap                   AS t_gap,
    t.acceleration          AS t_acceleration,
    t.volume_price_signal   AS t_volume_price_signal,
    t.kalman_dist           AS t_kalman_dist,
    t.vol_compression       AS t_vol_compression,
    t.macd                  AS t_macd,
    t.macd_bullish          AS t_macd_bullish,
    t.atr                   AS t_atr,
    t.atr_expansion         AS t_atr_expansion,
    t.adx                   AS t_adx,
    t.strong_trend          AS t_strong_trend,
    t.bb_width              AS t_bb_width,
    t.bb_squeeze            AS t_bb_squeeze,
    t.vwap_dist             AS t_vwap_dist,
    t.momentum_quality      AS t_momentum_quality,
    t.days_in_trend         AS t_days_in_trend,
    t.momentum_decay        AS t_momentum_decay,

    t.ticker,
    t.nse_symbol,
    t.sector

FROM ticker_momentum t
LEFT JOIN benchmark_momentum b
    ON t.date = b.date
   AND t.sector = b.sector;

CREATE OR REPLACE VIEW gate_score AS
     (WITH company AS (
        SELECT nse_symbol AS ticker, company_name, sector, market_cap, stock_p_e, roce, roe, book_value, current_price
        FROM general_info
        QUALIFY ROW_NUMBER() OVER (PARTITION BY nse_symbol ORDER BY nse_symbol) = 1
    ),
    sector_pe AS (
        SELECT sector, MEDIAN(stock_p_e) AS sector_median_pe
        FROM company WHERE stock_p_e > 0 GROUP BY sector
    ),
    qr_ranked AS (
        SELECT *, DENSE_RANK() OVER (PARTITION BY ticker ORDER BY dt DESC) AS rk
        FROM (SELECT DISTINCT ticker, dt, metric, val FROM quarterly_results)
    ),
    qr_pivot AS (
        SELECT ticker,
            MAX(CASE WHEN rk=1 AND metric='Sales'      THEN val END) AS sales_q0,
            MAX(CASE WHEN rk=2 AND metric='Sales'      THEN val END) AS sales_q1,
            MAX(CASE WHEN rk=3 AND metric='Sales'      THEN val END) AS sales_q2,
            MAX(CASE WHEN rk=4 AND metric='Sales'      THEN val END) AS sales_q3,
            MAX(CASE WHEN rk=1 AND metric='OPM %'      THEN val END) AS opm_q0,
            MAX(CASE WHEN rk=2 AND metric='OPM %'      THEN val END) AS opm_q1,
            MAX(CASE WHEN rk=1 AND metric='Net Profit' THEN val END) AS profit_q0,
            MAX(CASE WHEN rk=2 AND metric='Net Profit' THEN val END) AS profit_q1,
            MAX(CASE WHEN rk=3 AND metric='Net Profit' THEN val END) AS profit_q2,
            MAX(CASE WHEN rk=4 AND metric='Net Profit' THEN val END) AS profit_q3,
            MAX(CASE WHEN rk=1 AND metric='EPS in Rs'  THEN val END) AS eps_q0,
            MAX(CASE WHEN rk=2 AND metric='EPS in Rs'  THEN val END) AS eps_q1
        FROM qr_ranked WHERE rk <= 4 GROUP BY ticker
    ),
    pl_years AS (
        SELECT ticker, dt, DENSE_RANK() OVER (PARTITION BY ticker ORDER BY dt DESC) AS yr_rk
        FROM (SELECT DISTINCT ticker, dt FROM profit_loss WHERE dt < CURRENT_DATE)
    ),
    pl_latest AS (
        SELECT a.ticker,
            MAX(CASE WHEN metric='Net Profit' THEN val END) AS profit_now,
            MAX(CASE WHEN metric='Sales'      THEN val END) AS sales_now,
            MAX(CASE WHEN metric='Interest'   THEN val END) AS interest
        FROM profit_loss a JOIN pl_years y ON a.ticker = y.ticker AND a.dt = y.dt
        WHERE y.yr_rk = 1 GROUP BY a.ticker
    ),
    pl_3y AS (
        SELECT a.ticker,
            MAX(CASE WHEN metric='Net Profit' THEN val END) AS profit_3y,
            MAX(CASE WHEN metric='Sales'      THEN val END) AS sales_3y
        FROM profit_loss a JOIN pl_years y ON a.ticker = y.ticker AND a.dt = y.dt
        WHERE y.yr_rk = 4 GROUP BY a.ticker
    ),
    cf_latest AS (
        SELECT a.ticker,
            MAX(CASE WHEN metric='Cash from Operating Activity' THEN val END) AS cfo,
            MAX(CASE WHEN metric='Free Cash Flow'               THEN val END) AS fcf
        FROM cash_flows a
        WHERE a.dt = (SELECT MAX(dt) FROM cash_flows WHERE ticker = a.ticker)
        GROUP BY a.ticker
    ),
    bs_latest AS (
        SELECT a.ticker,
            MAX(CASE WHEN metric IN ('Borrowing','Borrowings') THEN val END) AS debt,
            MAX(CASE WHEN metric='Reserves'                    THEN val END) AS reserves,
            MAX(CASE WHEN metric='Equity Capital'              THEN val END) AS equity_capital
        FROM balance_sheet a
        WHERE a.dt = (SELECT MAX(dt) FROM balance_sheet WHERE ticker = a.ticker)
        GROUP BY a.ticker
    ),
    sh_ranked AS (
        SELECT ticker, dt, category, val,
            DENSE_RANK() OVER (PARTITION BY ticker ORDER BY dt DESC) AS rk
        FROM shareholding_quarterly
    ),
    sh_latest AS (
        SELECT ticker,
            MAX(CASE WHEN category='Promoters' THEN val END) AS promoter,
            MAX(CASE WHEN category='FIIs'      THEN val END) AS fii,
            MAX(CASE WHEN category='DIIs'      THEN val END) AS dii
        FROM sh_ranked WHERE rk = 1 GROUP BY ticker
    ),
    sh_prev AS (
        SELECT ticker,
            MAX(CASE WHEN category='Promoters' THEN val END) AS promoter_prev,
            MAX(CASE WHEN category='FIIs'      THEN val END) AS fii_prev,
            MAX(CASE WHEN category='DIIs'      THEN val END) AS dii_prev
        FROM sh_ranked WHERE rk = 2 GROUP BY ticker
    ),
    daily_ind AS (
        SELECT nse_symbol AS Ticker, Date, Close, Volume,
            LAG(Close, 21)  OVER (PARTITION BY nse_symbol ORDER BY Date) AS close_1m,
            LAG(Close, 63)  OVER (PARTITION BY nse_symbol ORDER BY Date) AS close_3m,
            AVG(Close)      OVER (PARTITION BY nse_symbol ORDER BY Date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS ma_20,
            STDDEV(Close)   OVER (PARTITION BY nse_symbol ORDER BY Date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS std_20,
            AVG(Close)      OVER (PARTITION BY nse_symbol ORDER BY Date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS ma_50,
            MAX(Close)      OVER (PARTITION BY nse_symbol ORDER BY Date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW) AS high_52w,
            AVG(Volume)     OVER (PARTITION BY nse_symbol ORDER BY Date ROWS BETWEEN 4  PRECEDING AND CURRENT ROW) AS vol_ma_5,
            AVG(Volume)     OVER (PARTITION BY nse_symbol ORDER BY Date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) AS vol_ma_50
        FROM ticker_prices
    ),
    tech_states AS (
        SELECT *,
            ((Close - close_1m) / NULLIF(close_1m,0)) * 100 AS return_1m,
            ((Close - close_3m) / NULLIF(close_3m,0)) * 100 AS return_3m,
            (4.0 * std_20) / NULLIF(ma_20,0)                AS bb_width,
            ((Close - high_52w) / NULLIF(high_52w,0)) * 100 AS pct_from_52w,
            vol_ma_5 / NULLIF(vol_ma_50,0)                   AS volume_ratio,
            ((Close - ma_50) / NULLIF(ma_50,0)) * 100        AS dist_from_ma50
        FROM daily_ind
    ),
    tech_latest AS (
        SELECT *,
            MIN(bb_width) OVER (PARTITION BY Ticker ORDER BY Date ROWS BETWEEN 62 PRECEDING AND CURRENT ROW) AS min_bb_63d
        FROM tech_states
        QUALIFY ROW_NUMBER() OVER (PARTITION BY Ticker ORDER BY Date DESC) = 1
    )
        SELECT
            c.ticker, c.company_name, c.sector,
            c.market_cap, c.stock_p_e, c.roce, c.roe, c.book_value, c.current_price,
            sp.sector_median_pe,
            q.sales_q0, q.sales_q1, q.sales_q2, q.sales_q3,
            q.opm_q0, q.opm_q1,
            q.profit_q0, q.profit_q1, q.profit_q2, q.profit_q3,
            q.eps_q0, q.eps_q1,
            pl.profit_now, pl.sales_now, pl.interest,
            py.profit_3y, py.sales_3y,
            cf.cfo, cf.fcf,
            bs.debt, bs.reserves, bs.equity_capital,
            sh.promoter, sh.fii, sh.dii,
            sp2.promoter_prev, sp2.fii_prev, sp2.dii_prev,
            t.Close AS price, t.return_1m, t.return_3m,
            t.bb_width, t.min_bb_63d, t.pct_from_52w,
            t.volume_ratio, t.dist_from_ma50
        FROM company c
        JOIN sector_pe   sp  ON c.sector = sp.sector
        JOIN tech_latest t   ON c.ticker = t.Ticker
        LEFT JOIN qr_pivot  q   ON c.ticker = q.ticker
        LEFT JOIN pl_latest pl  ON c.ticker = pl.ticker
        LEFT JOIN pl_3y     py  ON c.ticker = py.ticker
        LEFT JOIN cf_latest cf  ON c.ticker = cf.ticker
        LEFT JOIN bs_latest bs  ON c.ticker = bs.ticker
        LEFT JOIN sh_latest sh  ON c.ticker = sh.ticker
        LEFT JOIN sh_prev   sp2 ON c.ticker = sp2.ticker
    ) ;