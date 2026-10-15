-- ============================================================================
-- fundamental_spread_signal.sql
-- Command-line version of page_algo_momentum.py (Fundamental Spread Signal).
--
-- Composite = mean of four sector-relative scores (z-score or percentile rank):
--   z_roe   ROE momentum  : latest minus earliest quarterly ROE (last 5 qtrs)   higher = better
--   z_accr  Accruals      : (Net Profit - CFO) / Total Assets (annual)           lower  = better (negated)
--   z_rev   Revenue CV    : stdev / |mean| of Sales over last 4 quarters         lower  = better (negated)
--   z_de    D/E trend     : latest minus earliest D/E over last 4 annual points  lower  = better (negated)
-- Grouping is broad_industry. Long/short book = top and bottom top_pct per industry.
--
-- USAGE (run from the project root so the parquet paths in init.sql resolve):
--   cat sqls/init.sql sqls/compact_clean.sql fundamental_spread_signal.sql | duckdb
--   cat sqls/init.sql sqls/compact_clean.sql fundamental_spread_signal.sql | duckdb -markdown
--   cat sqls/init.sql sqls/compact_clean.sql fundamental_spread_signal.sql | duckdb -csv > signal.csv
--
-- Change parameters in an open session without re-creating anything, e.g.
--   UPDATE fss_params SET method = 'rank', weighting = 'score', top_pct = 0.30
-- then re-run any SELECT at the bottom (all views read fss_params live).
--
-- Input views : general_info, quarterly_results, balance_sheet, profit_loss, cash_flows
--               (long format: ticker, dt, metric, val)
-- Output views: fss_roe_momentum, fss_accruals, fss_rev_cv, fss_de_trend,
--               fss_metrics, fss_scored, fss_book
-- ============================================================================


-- ----------------------------------------------------------------------------
-- Parameters (the sidebar controls of the app)
-- ----------------------------------------------------------------------------
CREATE OR REPLACE TABLE fss_params AS
SELECT
    'zscore'::VARCHAR AS method,          -- 'zscore' or 'rank'
    4::INTEGER        AS min_per_sector,  -- min scored stocks per industry (app: 2..20)
    0.25::DOUBLE      AS top_pct,         -- long/short fraction            (app: 0.10..0.40)
    'equal'::VARCHAR  AS weighting;       -- 'equal' or 'score'


-- ----------------------------------------------------------------------------
-- Universe (one row per ticker)
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW fss_general_info AS
SELECT ticker, company_name, sector, broad_sector, broad_industry, industry,
       roe, roce, stock_p_e, market_cap, current_price
FROM general_info
QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY ticker) = 1;


-- ----------------------------------------------------------------------------
-- 1) ROE momentum (quarterly)
--    Quarterly ROE = Net Profit / Equity, Equity = Equity Capital + Reserves from
--    the most recent annual balance sheet on or before that quarter.
--    Momentum = ROE of latest quarter minus ROE of earliest of the last 5 quarters.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW fss_roe_momentum AS
WITH np_q AS (
    SELECT ticker, dt, MAX(val) AS net_profit
    FROM quarterly_results
    WHERE metric = 'Net Profit' AND val IS NOT NULL
    GROUP BY ticker, dt
),
ranked AS (
    SELECT ticker, dt, net_profit,
           DENSE_RANK() OVER (PARTITION BY ticker ORDER BY dt DESC) AS rk
    FROM np_q
),
last5 AS (
    SELECT *, COUNT(*) OVER (PARTITION BY ticker) AS cnt
    FROM ranked
    WHERE rk <= 5
),
eq_all AS (
    SELECT ticker, dt,
           COALESCE(MAX(CASE WHEN metric = 'Equity Capital' THEN val END), 0)
         + COALESCE(MAX(CASE WHEN metric = 'Reserves'       THEN val END), 0) AS equity
    FROM balance_sheet
    GROUP BY ticker, dt
),
eq AS (
    SELECT ticker, dt, equity FROM eq_all WHERE equity > 0
),
roe_q AS (
    SELECT q.ticker, q.dt, q.rk, q.cnt, q.net_profit / e.equity AS roe_q
    FROM last5 q
    ASOF LEFT JOIN eq e
      ON q.ticker = e.ticker AND q.dt >= e.dt
)
SELECT ticker,
       MAX(CASE WHEN rk = 1   THEN roe_q END)
     - MAX(CASE WHEN rk = cnt THEN roe_q END) AS roe_momentum
FROM roe_q
GROUP BY ticker
HAVING MAX(cnt) >= 2;


-- ----------------------------------------------------------------------------
-- 2) Accruals (annual) = (Net Profit - CFO) / Total Assets
--    Net Profit from the latest profit_loss year before today, CFO and Total
--    Assets from the latest available date of their tables.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW fss_accruals AS
WITH pl_latest AS (
    SELECT ticker, MAX(dt) AS dt FROM profit_loss WHERE dt < CURRENT_DATE GROUP BY ticker
),
ni AS (
    SELECT p.ticker,
           MAX(CASE WHEN p.metric = 'Net Profit' THEN p.val END) AS net_income
    FROM profit_loss p
    JOIN pl_latest l ON p.ticker = l.ticker AND p.dt = l.dt
    GROUP BY p.ticker
),
cf_latest AS (
    SELECT ticker, MAX(dt) AS dt FROM cash_flows GROUP BY ticker
),
cfo AS (
    SELECT c.ticker,
           MAX(CASE WHEN c.metric = 'Cash from Operating Activity' THEN c.val END) AS cfo
    FROM cash_flows c
    JOIN cf_latest l ON c.ticker = l.ticker AND c.dt = l.dt
    GROUP BY c.ticker
),
bs_latest AS (
    SELECT ticker, MAX(dt) AS dt FROM balance_sheet GROUP BY ticker
),
assets AS (
    SELECT b.ticker,
           MAX(CASE WHEN b.metric = 'Total Assets' THEN b.val END) AS total_assets
    FROM balance_sheet b
    JOIN bs_latest l ON b.ticker = l.ticker AND b.dt = l.dt
    GROUP BY b.ticker
),
joined AS (
    SELECT ni.ticker,
           (ni.net_income - cfo.cfo) / NULLIF(assets.total_assets, 0) AS accruals
    FROM ni
    JOIN cfo    ON ni.ticker = cfo.ticker
    JOIN assets ON ni.ticker = assets.ticker
)
SELECT ticker, accruals FROM joined WHERE accruals IS NOT NULL;


-- ----------------------------------------------------------------------------
-- 3) Revenue coefficient of variation (last 4 quarters of Sales / Revenue)
--    Sample stdev / |mean|
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW fss_rev_cv AS
WITH ranked AS (
    SELECT ticker, dt, val,
           DENSE_RANK() OVER (PARTITION BY ticker ORDER BY dt DESC) AS rk
    FROM quarterly_results
    WHERE metric IN ('Sales', 'Revenue') AND val IS NOT NULL
),
cv AS (
    SELECT ticker,
           STDDEV_SAMP(val) / NULLIF(ABS(AVG(val)), 0) AS rev_cv
    FROM ranked
    WHERE rk <= 4
    GROUP BY ticker
)
SELECT ticker, rev_cv FROM cv WHERE rev_cv IS NOT NULL;


-- ----------------------------------------------------------------------------
-- 4) D/E trend (annual) = latest D/E minus earliest D/E over the last 4 annual
--    balance-sheet periods before today (periods with no D/E are skipped).
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW fss_de_trend AS
WITH bs AS (
    SELECT ticker, dt,
           MAX(CASE WHEN metric IN ('Borrowing', 'Borrowings') THEN val END) AS debt,
           MAX(CASE WHEN metric = 'Reserves'                   THEN val END) AS reserves,
           MAX(CASE WHEN metric = 'Equity Capital'             THEN val END) AS eq_cap
    FROM balance_sheet
    GROUP BY ticker, dt
),
ranked AS (
    SELECT *, DENSE_RANK() OVER (PARTITION BY ticker ORDER BY dt DESC) AS rk
    FROM bs
    WHERE dt < CURRENT_DATE
),
de AS (
    SELECT ticker, dt,
           debt / NULLIF(COALESCE(eq_cap, 0) + COALESCE(reserves, 0), 0) AS de_ratio
    FROM ranked
    WHERE rk <= 4
)
SELECT ticker,
       arg_max(de_ratio, dt) - arg_min(de_ratio, dt) AS de_trend
FROM de
WHERE de_ratio IS NOT NULL
GROUP BY ticker
HAVING COUNT(*) >= 2;


-- ----------------------------------------------------------------------------
-- Metrics table: outer join of the four metrics + broad_industry
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW fss_metrics AS
SELECT ticker, r.roe_momentum, a.accruals, v.rev_cv, d.de_trend, g.broad_industry
FROM fss_roe_momentum r
FULL OUTER JOIN fss_accruals a USING (ticker)
FULL OUTER JOIN fss_rev_cv   v USING (ticker)
FULL OUTER JOIN fss_de_trend d USING (ticker)
LEFT JOIN fss_general_info   g USING (ticker);


-- ----------------------------------------------------------------------------
-- Scoring: sector-relative z-scores (or percentile ranks), composite = mean
--   zscore : (x - mean) / sample stdev within industry (0 if stdev is 0)
--   rank   : average-rank percentile - 0.5 (pandas rank(pct=True) - 0.5)
--   Accruals, Rev CV and D/E trend are negated (lower is better).
-- Only stocks with all four metrics and a broad_industry are scored, and only
-- industries with at least min_per_sector such stocks.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW fss_scored AS
WITH elig AS (
    SELECT ticker, broad_industry AS industry, roe_momentum, accruals, rev_cv, de_trend
    FROM fss_metrics
    WHERE broad_industry IS NOT NULL
      AND roe_momentum IS NOT NULL AND accruals IS NOT NULL
      AND rev_cv IS NOT NULL       AND de_trend IS NOT NULL
),
stats AS (
    SELECT e.*,
           COUNT(*) OVER w AS n,
           AVG(roe_momentum)         OVER w AS mu_roe,
           STDDEV_SAMP(roe_momentum) OVER w AS sd_roe,
           AVG(accruals)             OVER w AS mu_accr,
           STDDEV_SAMP(accruals)     OVER w AS sd_accr,
           AVG(rev_cv)               OVER w AS mu_rev,
           STDDEV_SAMP(rev_cv)       OVER w AS sd_rev,
           AVG(de_trend)             OVER w AS mu_de,
           STDDEV_SAMP(de_trend)     OVER w AS sd_de,
           (RANK() OVER r1 + COUNT(*) OVER r1) / 2.0 AS ar_roe,
           (RANK() OVER r2 + COUNT(*) OVER r2) / 2.0 AS ar_accr,
           (RANK() OVER r3 + COUNT(*) OVER r3) / 2.0 AS ar_rev,
           (RANK() OVER r4 + COUNT(*) OVER r4) / 2.0 AS ar_de
    FROM elig e
    WINDOW w  AS (PARTITION BY industry),
           r1 AS (PARTITION BY industry ORDER BY roe_momentum),
           r2 AS (PARTITION BY industry ORDER BY accruals),
           r3 AS (PARTITION BY industry ORDER BY rev_cv),
           r4 AS (PARTITION BY industry ORDER BY de_trend)
),
tr AS (
    SELECT s.ticker, s.industry, s.n,
           s.roe_momentum, s.accruals, s.rev_cv, s.de_trend,
           CASE WHEN p.method = 'zscore'
                THEN COALESCE((s.roe_momentum - s.mu_roe) / NULLIF(s.sd_roe, 0), 0.0)
                WHEN s.n < 2 THEN 0.0
                ELSE s.ar_roe / s.n - 0.5 END AS t_roe,
           CASE WHEN p.method = 'zscore'
                THEN COALESCE((s.accruals - s.mu_accr) / NULLIF(s.sd_accr, 0), 0.0)
                WHEN s.n < 2 THEN 0.0
                ELSE s.ar_accr / s.n - 0.5 END AS t_accr,
           CASE WHEN p.method = 'zscore'
                THEN COALESCE((s.rev_cv - s.mu_rev) / NULLIF(s.sd_rev, 0), 0.0)
                WHEN s.n < 2 THEN 0.0
                ELSE s.ar_rev / s.n - 0.5 END AS t_rev,
           CASE WHEN p.method = 'zscore'
                THEN COALESCE((s.de_trend - s.mu_de) / NULLIF(s.sd_de, 0), 0.0)
                WHEN s.n < 2 THEN 0.0
                ELSE s.ar_de / s.n - 0.5 END AS t_de
    FROM stats s
    CROSS JOIN fss_params p
    WHERE s.n >= p.min_per_sector
)
SELECT ticker, industry, n AS industry_size,
       t_roe   AS z_roe,
       -t_accr AS z_accr,
       -t_rev  AS z_rev,
       -t_de   AS z_de,
       (t_roe - t_accr - t_rev - t_de) / 4.0 AS composite,
       roe_momentum, accruals, rev_cv, de_trend
FROM tr;


-- ----------------------------------------------------------------------------
-- Long / short book
--   k = max(1, round_half_even(n * top_pct)) per industry
--   Longs = top k composite, shorts = bottom k (ties broken by ticker)
--   equal : weight = 1/k on each side
--   score : composite (shorts: -composite) shifted to be positive, normalised
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW fss_book AS
WITH g AS (
    SELECT s.*,
           ROW_NUMBER() OVER (PARTITION BY industry ORDER BY composite DESC, ticker) AS rn
    FROM fss_scored s
),
sized AS (
    SELECT g.*,
           GREATEST(1, ROUND_EVEN(g.industry_size * p.top_pct, 0))::INTEGER AS k,
           p.weighting
    FROM g CROSS JOIN fss_params p
),
sided AS (
    SELECT ticker, industry, composite, k, weighting,
           CASE WHEN rn <= k THEN 'L' ELSE 'S' END AS side,
           CASE WHEN rn <= k THEN composite ELSE -composite END AS val
    FROM sized
    WHERE rn <= k OR rn > industry_size - k
),
shifted AS (
    SELECT *,
           val - MIN(val) OVER b
               + 0.1 * (MAX(val) OVER b - MIN(val) OVER b + 1e-9) AS sh
    FROM sided
    WINDOW b AS (PARTITION BY industry, side)
)
SELECT ticker, industry, side, composite,
       CASE WHEN weighting = 'score'
            THEN sh / SUM(sh) OVER (PARTITION BY industry, side)
            ELSE 1.0 / k END AS weight
FROM shifted;


-- ============================================================================
-- OUTPUT 1: Coverage (how many tickers each metric covers)
-- ============================================================================
SELECT
    (SELECT COUNT(DISTINCT ticker) FROM fss_general_info) AS universe,
    (SELECT COUNT(DISTINCT ticker) FROM fss_roe_momentum) AS roe_momentum,
    (SELECT COUNT(DISTINCT ticker) FROM fss_accruals)     AS accruals,
    (SELECT COUNT(DISTINCT ticker) FROM fss_rev_cv)       AS rev_cv,
    (SELECT COUNT(DISTINCT ticker) FROM fss_de_trend)     AS de_trend,
    (SELECT COUNT(*)               FROM fss_scored)       AS scored,
    (SELECT COUNT(DISTINCT industry) FROM fss_scored)     AS industries;


-- ============================================================================
-- OUTPUT 2: Scored universe (best composite first)
-- ============================================================================
SELECT s.ticker,
       LEFT(g.company_name, 24)       AS company,
       s.industry,
       ROUND(s.composite, 3)          AS composite,
       ROUND(s.z_roe, 2)              AS z_roe,
       ROUND(s.z_accr, 2)             AS z_accr,
       ROUND(s.z_rev, 2)              AS z_rev,
       ROUND(s.z_de, 2)               AS z_de,
       ROUND(s.roe_momentum, 3)       AS roe_mom,
       ROUND(s.accruals, 3)           AS accruals,
       ROUND(s.rev_cv, 3)             AS rev_cv,
       ROUND(s.de_trend, 3)           AS de_trend,
       g.market_cap, g.current_price, g.stock_p_e, g.roe, g.roce
FROM fss_scored s
LEFT JOIN fss_general_info g USING (ticker)
ORDER BY s.composite DESC;


-- ============================================================================
-- OUTPUT 3: Long book
-- ============================================================================
SELECT b.ticker, LEFT(g.company_name, 24) AS company, b.industry,
       ROUND(b.composite, 3) AS composite, ROUND(b.weight, 3) AS weight,
       g.market_cap, g.current_price
FROM fss_book b
LEFT JOIN fss_general_info g USING (ticker)
WHERE b.side = 'L'
ORDER BY b.composite DESC;


-- ============================================================================
-- OUTPUT 4: Short book
-- ============================================================================
SELECT b.ticker, LEFT(g.company_name, 24) AS company, b.industry,
       ROUND(b.composite, 3) AS composite, ROUND(b.weight, 3) AS weight,
       g.market_cap, g.current_price
FROM fss_book b
LEFT JOIN fss_general_info g USING (ticker)
WHERE b.side = 'S'
ORDER BY b.composite ASC;


-- ============================================================================
-- OUTPUT 5: Industry exposure (sum of weights per side)
-- ============================================================================
SELECT industry,
       ROUND(SUM(CASE WHEN side = 'L' THEN weight ELSE 0 END), 3) AS L,
       ROUND(SUM(CASE WHEN side = 'S' THEN weight ELSE 0 END), 3) AS S,
       ROUND(SUM(CASE WHEN side = 'L' THEN weight ELSE -weight END), 3) AS net
FROM fss_book
GROUP BY industry
ORDER BY industry;
