-- =============================================================================
-- dcf_valuation.sql   (DuckDB)         v3 - FCF Corrected & Time-Aligned
-- =============================================================================

-- 0. MANUAL OVERRIDES
CREATE TABLE IF NOT EXISTS dcf_assumptions (
    ticker         VARCHAR,
    g1             DOUBLE,    -- profit growth yr 1-5   (%)
    g2             DOUBLE,    -- profit growth yr 6-10  (%)
    tg             DOUBLE,    -- terminal growth 10+    (%)
    dr             DOUBLE,    -- discount rate          (%)
    mos            DOUBLE,    -- margin of safety       (%)
    zscore_thresh  DOUBLE,    -- other-income spike threshold
    oi_structural  BOOLEAN,   -- TRUE = other income is recurring, skip spike removal
    decay_lambda   DOUBLE     -- recency decay per day
);

-- 0b. POLICY
CREATE TABLE IF NOT EXISTS dcf_policy (
    rf DOUBLE, erp DOUBLE,
    anchor_growth DOUBLE, shrink DOUBLE,
    g1_cap DOUBLE, tg_max DOUBLE, fade DOUBLE,
    dr_min DOUBLE, dr_max DOUBLE,
    mos_base DOUBLE, mos_k DOUBLE, mos_min DOUBLE, mos_max DOUBLE, mos_sd_default DOUBLE,
    zscore_thresh DOUBLE, decay_lambda DOUBLE,
    beta_window_days INTEGER, benchmark_ticker VARCHAR
);

CREATE OR REPLACE VIEW v_dcf_policy AS
SELECT COALESCE(p.rf, 7.0)              AS rf,
       COALESCE(p.erp, 5.5)             AS erp,
       COALESCE(p.anchor_growth, 10.0)  AS anchor_growth,
       COALESCE(p.shrink, 0.5)          AS shrink,
       COALESCE(p.g1_cap, 25.0)         AS g1_cap,
       COALESCE(p.tg_max, 5.0)          AS tg_max,
       COALESCE(p.fade, 0.5)            AS fade,
       COALESCE(p.dr_min, 10.0)         AS dr_min,
       COALESCE(p.dr_max, 16.0)         AS dr_max,
       COALESCE(p.mos_base, 15.0)       AS mos_base,
       COALESCE(p.mos_k, 0.5)           AS mos_k,
       COALESCE(p.mos_min, 15.0)        AS mos_min,
       COALESCE(p.mos_max, 40.0)        AS mos_max,
       COALESCE(p.mos_sd_default, 15.0) AS mos_sd_default,
       COALESCE(p.zscore_thresh, 2.0)   AS zscore_thresh,
       COALESCE(p.decay_lambda, 0.001)  AS decay_lambda,
       COALESCE(p.beta_window_days, 1095) AS beta_window_days,
       COALESCE(p.benchmark_ticker, '^NSEI') AS benchmark_ticker
FROM (SELECT 1) AS _one
LEFT JOIN (SELECT * FROM dcf_policy LIMIT 1) p ON TRUE;

-- 1. DECAY RATE per ticker
CREATE OR REPLACE VIEW v_dcf_lambda AS
WITH cg AS (SELECT * FROM dcf_assumptions WHERE COALESCE(TRIM(ticker), '') = '' LIMIT 1),
     ct AS (SELECT * FROM dcf_assumptions WHERE COALESCE(TRIM(ticker), '') <> ''
            QUALIFY ROW_NUMBER() OVER (PARTITION BY UPPER(TRIM(ticker))) = 1)
SELECT g.ticker, COALESCE(ct.decay_lambda, cg.decay_lambda, pol.decay_lambda) AS lam
FROM general_info g
CROSS JOIN v_dcf_policy pol
LEFT JOIN ct ON UPPER(TRIM(ct.ticker)) = UPPER(TRIM(g.ticker))
LEFT JOIN cg ON TRUE;

-- 2. RECENCY-WEIGHTED HISTORY
CREATE OR REPLACE VIEW v_dcf_history AS
WITH
lam AS (SELECT ticker, lam FROM v_dcf_lambda),
np_year AS (
    SELECT ticker, YEAR(dt) AS yr, MAX(dt) AS dt_end, SUM(val) AS net_profit
    FROM profit_loss WHERE metric = 'Net Profit'
    GROUP BY ticker, YEAR(dt)
),
np_pos AS (
    SELECT *, LAG(net_profit) OVER (PARTITION BY ticker ORDER BY yr) AS prev_np,
              LAG(yr)         OVER (PARTITION BY ticker ORDER BY yr) AS prev_yr
    FROM np_year WHERE net_profit > 0
),
g_obs AS (
    SELECT ticker, yr, dt_end,
           LN(LEAST(GREATEST(net_profit / prev_np, 0.5), 2.0)) AS ln_g
    FROM np_pos WHERE prev_np > 0 AND yr - prev_yr = 1
),
g_w AS (
    SELECT o.ticker, o.ln_g, o.ln_g * 100 AS gp,
           ROW_NUMBER() OVER (PARTITION BY o.ticker ORDER BY o.yr DESC)                  AS rk,
           EXP(-l.lam * GREATEST(DATE_DIFF('day', o.dt_end, CURRENT_DATE), 0))           AS w
    FROM g_obs o JOIN lam l USING (ticker)
),
growth AS (
    SELECT ticker,
        COUNT(*) FILTER (WHERE rk <= 3)                                                  AS n_g3,
        COUNT(*) FILTER (WHERE rk <= 5)                                                  AS n_g5,
        (EXP(SUM(w * ln_g) FILTER (WHERE rk <= 3) / SUM(w) FILTER (WHERE rk <= 3)) - 1) * 100 AS wgrowth_3y,
        (EXP(SUM(w * ln_g) FILTER (WHERE rk <= 5) / SUM(w) FILTER (WHERE rk <= 5)) - 1) * 100 AS wgrowth_5y,
        CASE WHEN COUNT(*) FILTER (WHERE rk <= 5) >= 3 THEN
          SQRT(GREATEST(
            (SUM(w*gp*gp) FILTER (WHERE rk <= 5) - POW(SUM(w*gp) FILTER (WHERE rk <= 5), 2) / SUM(w) FILTER (WHERE rk <= 5))
            / NULLIF(SUM(w) FILTER (WHERE rk <= 5) - SUM(w*w) FILTER (WHERE rk <= 5) / SUM(w) FILTER (WHERE rk <= 5), 0), 0))
        END AS growth_wsd
    FROM g_w GROUP BY ticker
),
simple_g AS (
    SELECT ticker,
           MAX(val) FILTER (WHERE horizon IN ('3 Years','3Y')) AS cagr_3y,
           MAX(val) FILTER (WHERE horizon IN ('5 Years','5Y')) AS cagr_5y
    FROM compounded_profit_growth GROUP BY ticker
),
oi_year AS (
    SELECT ticker, YEAR(dt) AS yr, MAX(dt) AS dt_end, SUM(val) AS oi
    FROM profit_loss WHERE metric = 'Other Income'
    GROUP BY ticker, YEAR(dt)
),
oi_w AS (
    SELECT y.ticker, y.oi, y.rn,
           EXP(-l.lam * GREATEST(DATE_DIFF('day', y.dt_end, CURRENT_DATE), 0)) AS w
    FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY yr DESC) AS rn
          FROM oi_year WHERE oi IS NOT NULL) y
    JOIN lam l USING (ticker) WHERE y.rn <= 5
),
oi_stats AS (
    SELECT ticker,
        COUNT(*) FILTER (WHERE rn <= 5)                                                   AS n,
        SUM(w*oi) FILTER (WHERE rn <= 5) / SUM(w) FILTER (WHERE rn <= 5)                  AS oi_wmean,
        COUNT(*) FILTER (WHERE rn BETWEEN 2 AND 5)                                        AS n_prior,
        SUM(w*oi) FILTER (WHERE rn BETWEEN 2 AND 5)
          / SUM(w) FILTER (WHERE rn BETWEEN 2 AND 5)                                      AS oi_prior_wmean,
        SQRT(GREATEST(
            (SUM(w*oi*oi) FILTER (WHERE rn BETWEEN 2 AND 5)
             - POW(SUM(w*oi) FILTER (WHERE rn BETWEEN 2 AND 5), 2) / SUM(w) FILTER (WHERE rn BETWEEN 2 AND 5))
            / NULLIF(SUM(w) FILTER (WHERE rn BETWEEN 2 AND 5)
                     - SUM(w*w) FILTER (WHERE rn BETWEEN 2 AND 5) / SUM(w) FILTER (WHERE rn BETWEEN 2 AND 5), 0),
            0))                                                                           AS oi_prior_wsd
    FROM oi_w GROUP BY ticker
),
tax AS (
    SELECT t.ticker, SUM(t.w * t.val) / SUM(t.w) / 100.0 AS wtax
    FROM (SELECT p.ticker, p.val,
                 EXP(-l.lam * GREATEST(DATE_DIFF('day', p.dt, CURRENT_DATE), 0)) AS w,
                 ROW_NUMBER() OVER (PARTITION BY p.ticker ORDER BY p.dt DESC)      AS rn
          FROM profit_loss p JOIN lam l USING (ticker) WHERE p.metric = 'Tax %') t
    WHERE t.rn <= 3 GROUP BY t.ticker
)
SELECT l.ticker, l.lam,
       ROUND(LN(2) / l.lam, 0)             AS half_life_days,
       ROUND(gr.wgrowth_3y, 1)             AS wgrowth_3y,   gr.n_g3,
       ROUND(gr.wgrowth_5y, 1)             AS wgrowth_5y,   gr.n_g5,
       ROUND(gr.growth_wsd, 1)             AS growth_wsd,
       ROUND(sg.cagr_3y, 1)                AS cagr_3y_simple,
       ROUND(sg.cagr_5y, 1)                AS cagr_5y_simple,
       os.n AS oi_n,  os.oi_wmean,  os.n_prior AS oi_n_prior,  os.oi_prior_wmean,  os.oi_prior_wsd,
       tx.wtax
FROM lam l
LEFT JOIN growth   gr ON gr.ticker = l.ticker
LEFT JOIN simple_g sg ON sg.ticker = l.ticker
LEFT JOIN oi_stats os ON os.ticker = l.ticker
LEFT JOIN tax      tx ON tx.ticker = l.ticker;

-- 2b. RECENCY-WEIGHTED BETA
CREATE OR REPLACE VIEW v_dcf_beta AS
WITH pol AS (SELECT * FROM v_dcf_policy),
px AS (
    SELECT REPLACE(Ticker, '.NS', '') AS ticker, Date, Close FROM ticker_prices
    WHERE Date >= CURRENT_DATE - (SELECT beta_window_days FROM pol)
),
ret AS (
    SELECT ticker, Date, LN(Close / LAG(Close) OVER (PARTITION BY ticker ORDER BY Date)) AS r FROM px
),
mkt AS (
    SELECT Date, r AS rm FROM ret
    WHERE ticker = REPLACE((SELECT benchmark_ticker FROM pol), '.NS', '') AND r IS NOT NULL
),
j AS (
    SELECT s.ticker, s.r AS rs, m.rm,
           EXP(-l.lam * GREATEST(DATE_DIFF('day', s.Date, CURRENT_DATE), 0)) AS w
    FROM ret s JOIN mkt m USING (Date) JOIN v_dcf_lambda l ON l.ticker = s.ticker
    WHERE s.r IS NOT NULL
),
b AS (
    SELECT ticker, COUNT(*) AS n_obs,
           (SUM(w*rs*rm)/SUM(w) - (SUM(w*rs)/SUM(w)) * (SUM(w*rm)/SUM(w)))
             / NULLIF(SUM(w*rm*rm)/SUM(w) - POW(SUM(w*rm)/SUM(w), 2), 0)  AS beta_raw
    FROM j GROUP BY ticker HAVING COUNT(*) >= 100
),
vol AS (
    SELECT r.ticker,
           SQRT(GREATEST(SUM(w*r*r)/SUM(w) - POW(SUM(w*r)/SUM(w), 2), 0)) AS wvol
    FROM (SELECT x.ticker, x.r, EXP(-l.lam * GREATEST(DATE_DIFF('day', x.Date, CURRENT_DATE), 0)) AS w
          FROM ret x JOIN v_dcf_lambda l ON l.ticker = x.ticker WHERE x.r IS NOT NULL) r
    GROUP BY r.ticker HAVING COUNT(*) >= 100
),
med AS (SELECT MEDIAN(wvol) AS m FROM vol)
SELECT g.ticker,
       COALESCE(b.beta_raw, LEAST(GREATEST(v.wvol / NULLIF(med.m, 0), 0.7), 1.4))          AS beta_raw,
       CASE WHEN b.beta_raw IS NOT NULL THEN 'benchmark regression'
            WHEN v.wvol IS NOT NULL      THEN 'volatility proxy'
            ELSE 'default 1.0' END                                                         AS beta_source,
       LEAST(GREATEST(0.67 * COALESCE(b.beta_raw, LEAST(GREATEST(v.wvol / NULLIF(med.m, 0), 0.7), 1.4), 1.0)
                      + 0.33, 0.5), 1.8)                                                   AS beta_adj
FROM general_info g
LEFT JOIN b   ON b.ticker = g.ticker
LEFT JOIN vol v ON v.ticker = g.ticker
CROSS JOIN med;

-- 2c. AUTOMATIC ASSUMPTIONS
CREATE OR REPLACE VIEW v_dcf_assumptions_auto AS
WITH pol AS (SELECT * FROM v_dcf_policy),
base AS (
    SELECT g.ticker,
           COALESCE(h.wgrowth_3y, h.wgrowth_5y, h.cagr_3y_simple, 12.0)            AS g_recent,
           h.growth_wsd,
           CASE WHEN g.roe IS NOT NULL AND g.stock_p_e IS NOT NULL
                THEN g.roe * (1 - LEAST(GREATEST(COALESCE(g.dividend_yield, 0) / 100.0 * g.stock_p_e, 0), 1))
           END                                                                     AS g_sustainable,
           COALESCE(b.beta_adj, 1.0)                                               AS beta_adj,
           COALESCE(b.beta_source, 'default 1.0')                                  AS beta_source
    FROM general_info g
    LEFT JOIN v_dcf_history h ON h.ticker = g.ticker
    LEFT JOIN v_dcf_beta    b ON b.ticker = g.ticker
),
c1 AS (
    SELECT base.*, pol.*,
           GREATEST(LEAST(pol.shrink * g_recent + (1 - pol.shrink) * pol.anchor_growth,
                          pol.g1_cap, COALESCE(g_sustainable, 1e9)), 0)            AS g1
    FROM base CROSS JOIN pol
),
c2 AS (SELECT c1.*, GREATEST(LEAST(tg_max, g1), 2.0) AS tg FROM c1)
SELECT ticker,
       ROUND(g1, 1)                                                                AS g1,
       ROUND(tg + fade * (g1 - tg), 1)                                             AS g2,
       ROUND(tg, 1)                                                                AS tg,
       ROUND(LEAST(GREATEST(rf + beta_adj * erp, dr_min), dr_max), 1)              AS dr,
       LEAST(GREATEST(ROUND((mos_base + mos_k * COALESCE(growth_wsd, mos_sd_default)) / 5) * 5,
                      mos_min), mos_max)                                           AS mos,
       ROUND(g_recent, 1) AS hist_growth_w3y, ROUND(g_sustainable, 1) AS sustainable_growth,
       ROUND(growth_wsd, 1) AS growth_volatility, ROUND(beta_adj, 2) AS beta, beta_source
FROM c2;

-- 2d. RESOLVED ASSUMPTIONS
CREATE OR REPLACE VIEW v_dcf_cfg AS
WITH cg AS (SELECT * FROM dcf_assumptions WHERE COALESCE(TRIM(ticker), '') = '' LIMIT 1),
     ct AS (SELECT * FROM dcf_assumptions WHERE COALESCE(TRIM(ticker), '') <> ''
            QUALIFY ROW_NUMBER() OVER (PARTITION BY UPPER(TRIM(ticker))) = 1)
SELECT g.ticker,
       COALESCE(ct.g1,  au.g1,  cg.g1,  12.0)                          AS g1,
       COALESCE(ct.g2,  au.g2,  cg.g2,  10.0)                          AS g2,
       COALESCE(ct.tg,  au.tg,  cg.tg,   5.0)                          AS tg,
       COALESCE(ct.dr,  au.dr,  cg.dr,  12.0)                          AS dr,
       COALESCE(ct.mos, au.mos, cg.mos, 25.0)                          AS mos,
       COALESCE(ct.zscore_thresh, cg.zscore_thresh, pol.zscore_thresh) AS z_thresh,
       COALESCE(ct.oi_structural, cg.oi_structural, FALSE)             AS oi_structural,
       lm.lam,
       CASE WHEN ct.ticker IS NOT NULL THEN 'manual override (ticker row) + auto'
            ELSE 'auto' END                                            AS assump_source
FROM general_info g
CROSS JOIN v_dcf_policy pol
JOIN v_dcf_lambda lm ON lm.ticker = g.ticker
LEFT JOIN v_dcf_assumptions_auto au ON au.ticker = g.ticker
LEFT JOIN ct ON UPPER(TRIM(ct.ticker)) = UPPER(TRIM(g.ticker))
LEFT JOIN cg ON TRUE;

-- 3. INPUTS
CREATE OR REPLACE VIEW v_dcf_inputs AS
WITH
np_year AS (
    SELECT ticker, YEAR(dt) AS yr, SUM(val) AS net_profit
    FROM profit_loss WHERE metric = 'Net Profit' GROUP BY ticker, YEAR(dt)
),
np_latest AS (
    SELECT ticker, yr, net_profit AS reported_profit FROM np_year WHERE net_profit > 0
    QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY yr DESC) = 1
),
cf_ratio AS (
    SELECT c.ticker,
           LEAST(GREATEST(SUM(c.val) / NULLIF(SUM(p.val), 0), 0.1), 1.5) AS raw_fcf_conversion
    FROM (SELECT ticker, YEAR(dt) AS yr, SUM(val) AS val FROM cash_flows WHERE metric = 'Free Cash Flow' GROUP BY ticker, YEAR(dt)) c
    JOIN (SELECT ticker, YEAR(dt) AS yr, SUM(val) AS val FROM profit_loss WHERE metric = 'Net Profit' GROUP BY ticker, YEAR(dt)) p
      ON p.ticker = c.ticker AND p.yr = c.yr
    WHERE c.yr >= YEAR(CURRENT_DATE) - 3
    GROUP BY c.ticker
),
oi_latest AS (
    SELECT ticker, YEAR(dt) AS yr, SUM(val) AS oi FROM profit_loss
    WHERE metric = 'Other Income' GROUP BY ticker, YEAR(dt)
),
exc_year AS (
    SELECT ticker, YEAR(dt) AS yr, SUM(val) AS exc FROM profit_loss
    WHERE metric IN ('Exceptional Items','Exceptional Item','Extraordinary Items')
    GROUP BY ticker, YEAR(dt)
),
eq AS (
    SELECT ticker, val AS equity_cr FROM balance_sheet WHERE metric = 'Equity Capital'
    QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY dt DESC) = 1
),
eps_latest AS (
    SELECT ticker, eps FROM (SELECT ticker, YEAR(dt) AS yr, AVG(val) AS eps FROM profit_loss
                             WHERE metric = 'EPS in Rs' GROUP BY ticker, YEAR(dt))
    QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY yr DESC) = 1
),
px AS (
    SELECT REPLACE(Ticker,'.NS','') AS ticker, Close AS last_close FROM ticker_prices
    QUALIFY ROW_NUMBER() OVER (PARTITION BY REPLACE(Ticker,'.NS','') ORDER BY Date DESC) = 1
),
n AS (
    SELECT c.ticker, c.tg, c.dr, c.mos, c.z_thresh, c.oi_structural, c.lam,
           GREATEST(COALESCE(c.g1, h.wgrowth_3y, h.cagr_3y_simple, 12.0), 0)  AS g1,
           COALESCE(c.g2, h.wgrowth_5y, h.cagr_5y_simple, 10.0)               AS g2_raw,
           gi.current_price, gi.face_value, gi.sector,
           l.reported_profit,
           COALESCE(oy.oi, 0)                                   AS reported_oi,
           h.oi_n, h.oi_wmean, h.oi_n_prior, h.oi_prior_wmean, h.oi_prior_wsd,
           COALESCE(ey.exc, 0)                                  AS exc_latest,
           COALESCE(h.wtax, 0.25)                               AS tax_rate,
           CASE WHEN gi.sector = 'Financial Services' THEN 1.0
                ELSE COALESCE(cfr.raw_fcf_conversion, 0.7) 
           END AS fcf_conversion
    FROM v_dcf_cfg c
    JOIN general_info gi      ON gi.ticker = c.ticker
    JOIN np_latest l          ON l.ticker  = c.ticker
    LEFT JOIN v_dcf_history h ON h.ticker  = c.ticker
    LEFT JOIN oi_latest oy    ON oy.ticker = c.ticker AND oy.yr = l.yr
    LEFT JOIN exc_year  ey    ON ey.ticker = c.ticker AND ey.yr = l.yr
    LEFT JOIN cf_ratio cfr    ON cfr.ticker = c.ticker
),
n2 AS (
    SELECT n.*,
        (NOT oi_structural AND oi_n_prior >= 3 AND oi_prior_wsd > 0
         AND ABS((reported_oi - oi_prior_wmean) / oi_prior_wsd) > z_thresh)              AS oi_spike
    FROM n
),
n3 AS (
    SELECT n2.*,
        CASE WHEN oi_spike         THEN oi_prior_wmean
             WHEN oi_structural    THEN reported_oi
             WHEN oi_n >= 2        THEN oi_wmean
             ELSE reported_oi END                                                        AS norm_oi
    FROM n2
),
n4 AS (
    SELECT n3.*,
           GREATEST(reported_oi - norm_oi, 0) * (CASE WHEN oi_spike THEN 1 ELSE 0 END) AS abnormal_oi,
           -( (norm_oi - reported_oi) - exc_latest ) * tax_rate                          AS tax_adj
    FROM n3
)
SELECT
    n4.ticker, n4.reported_profit, n4.reported_oi, n4.norm_oi, n4.oi_spike, n4.abnormal_oi,
    n4.exc_latest, n4.tax_adj, n4.tax_rate, n4.fcf_conversion,
    CASE WHEN (reported_profit + (norm_oi - reported_oi) - exc_latest + tax_adj) > 0
         THEN  reported_profit + (norm_oi - reported_oi) - exc_latest + tax_adj
         ELSE  reported_profit END                                                       AS norm_profit,
    (CASE WHEN (reported_profit + (norm_oi - reported_oi) - exc_latest + tax_adj) > 0
          THEN reported_profit + (norm_oi - reported_oi) - exc_latest + tax_adj
          ELSE reported_profit END) * n4.fcf_conversion                                  AS norm_fcf,
    COALESCE(eq.equity_cr / COALESCE(NULLIF(n4.face_value, 0), 10.0),
             ABS(n4.reported_profit) / NULLIF(ABS(el.eps), 0))                           AS shares_cr,
    COALESCE(px.last_close, n4.current_price)                                            AS price,
    n4.g1, LEAST(n4.g2_raw, n4.g1) AS g2, n4.tg, n4.dr, n4.mos, n4.lam
FROM n4
LEFT JOIN eq            ON eq.ticker = n4.ticker
LEFT JOIN eps_latest el ON el.ticker = n4.ticker
LEFT JOIN px            ON px.ticker = n4.ticker;

-- 4. PROJECTION
CREATE OR REPLACE VIEW v_dcf_projection AS
WITH scn(scenario, g_mult, dr_shift) AS (
    VALUES ('bear', 0.6, 1.0), ('base', 1.0, 0.0), ('bull', 1.3, -1.0)
),
sp AS (
    SELECT i.ticker, s.scenario, i.norm_fcf, i.tg,
           i.g1 * s.g_mult AS g1s, i.g2 * s.g_mult AS g2s,
           CASE WHEN s.dr_shift < 0 THEN GREATEST(i.dr + s.dr_shift, i.tg + 0.5)
                ELSE i.dr + s.dr_shift END AS drs
    FROM v_dcf_inputs i CROSS JOIN scn s
)
SELECT sp.ticker, sp.scenario, sp.g1s, sp.g2s, sp.tg, sp.drs, y.yr,
       sp.norm_fcf * POW(1 + sp.g1s/100, LEAST(y.yr, 5))
                   * POW(1 + sp.g2s/100, GREATEST(y.yr - 5, 0))                  AS projected_fcf,
       sp.norm_fcf * POW(1 + sp.g1s/100, LEAST(y.yr, 5))
                   * POW(1 + sp.g2s/100, GREATEST(y.yr - 5, 0))
                   / POW(1 + sp.drs/100, y.yr)                                   AS pv
FROM sp CROSS JOIN generate_series(1, 10) AS y(yr);

-- 4b. RECENCY-WEIGHTED HISTORICAL P/E
CREATE OR REPLACE VIEW v_dcf_pe_history AS
WITH
lam AS (SELECT ticker, lam FROM v_dcf_lambda),
px AS (
    SELECT REPLACE(Ticker, '.NS', '') AS ticker, Date, Close FROM ticker_prices
),
px_month AS ( 
    SELECT ticker, DATE_TRUNC('month', Date) AS ym, Close AS close_price FROM px
    QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker, DATE_TRUNC('month', Date) ORDER BY Date DESC) = 1
),
eps_annual AS (
    SELECT ticker, CAST(dt AS DATE) AS dt, SUM(val) AS eps
    FROM profit_loss WHERE metric = 'EPS in Rs'
    GROUP BY ticker, CAST(dt AS DATE) HAVING SUM(val) > 0
),
annual_pe AS (
    SELECT e.ticker, e.dt, AVG(p.Close) / e.eps AS pe
    FROM eps_annual e
    JOIN px p ON p.ticker = e.ticker AND p.Date BETWEEN e.dt - INTERVAL '1 year' AND e.dt
    GROUP BY e.ticker, e.dt, e.eps
),
q_ttm AS (
    SELECT ticker, CAST(dt AS DATE) AS dt,
           SUM(val) OVER w AS ttm_eps, COUNT(*) OVER w AS n4,
           DATE_DIFF('day', CAST(LAG(dt, 3) OVER w_lag AS DATE), CAST(dt AS DATE)) AS days_diff
    FROM quarterly_results WHERE metric = 'EPS in Rs'
    WINDOW w AS (PARTITION BY ticker ORDER BY dt ROWS BETWEEN 3 PRECEDING AND CURRENT ROW),
           w_lag AS (PARTITION BY ticker ORDER BY dt)
),
quarterly_pe AS (
    SELECT q.ticker, q.dt, m.close_price / q.ttm_eps AS pe
    FROM q_ttm q
    JOIN px_month m ON m.ticker = q.ticker AND m.ym = DATE_TRUNC('month', q.dt)
    WHERE q.n4 = 4 AND q.ttm_eps > 0 AND q.days_diff BETWEEN 350 AND 380
),
cutoff AS (SELECT ticker, MIN(dt) AS qdt FROM quarterly_pe GROUP BY ticker),
combined AS (
    SELECT u.ticker, u.dt, u.pe, u.src,
           EXP(-l.lam * GREATEST(DATE_DIFF('day', u.dt, CURRENT_DATE), 0)) AS w
    FROM (SELECT a.ticker, a.dt, a.pe, 'annual' AS src
          FROM annual_pe a LEFT JOIN cutoff c ON c.ticker = a.ticker
          WHERE c.qdt IS NULL OR a.dt < c.qdt
          UNION ALL
          SELECT ticker, dt, pe, 'quarterly' FROM quarterly_pe) u
    JOIN lam l USING (ticker)
    WHERE u.pe > 3 AND u.pe < 300
),
wmed AS (
    SELECT ticker, pe AS wmedian_pe
    FROM (SELECT ticker, pe,
                 SUM(w) OVER (PARTITION BY ticker ORDER BY pe, dt ROWS UNBOUNDED PRECEDING)
                   / SUM(w) OVER (PARTITION BY ticker) AS cum_w, dt
          FROM combined)
    WHERE cum_w >= 0.5
    QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY pe, dt) = 1
),
agg AS (
    SELECT ticker,
           SUM(pe * w) / SUM(w)                         AS wmean_pe,
           MEDIAN(pe) AS median_pe, MIN(pe) AS min_pe, MAX(pe) AS max_pe,
           COUNT(*) FILTER (WHERE src = 'annual')       AS n_annual,
           COUNT(*) FILTER (WHERE src = 'quarterly')    AS n_quarterly
    FROM combined GROUP BY ticker
),
eps_now AS ( 
    SELECT ticker, eps AS latest_eps
    FROM (SELECT ticker, YEAR(dt) AS yr, AVG(val) AS eps FROM profit_loss
          WHERE metric = 'EPS in Rs' GROUP BY ticker, YEAR(dt))
    QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY yr DESC) = 1
)
SELECT a.ticker, m.wmedian_pe AS wmedian_pe, ROUND(a.wmean_pe, 1) AS wmean_pe,
       ROUND(a.median_pe, 1) AS median_pe, ROUND(a.min_pe, 1) AS min_pe, ROUND(a.max_pe, 1) AS max_pe,
       a.n_annual, a.n_quarterly, e.latest_eps
FROM agg a
LEFT JOIN wmed    m ON m.ticker = a.ticker
LEFT JOIN eps_now e ON e.ticker = a.ticker;

-- 5. VALUATION
CREATE OR REPLACE VIEW v_dcf_valuation AS
WITH tv AS (
    SELECT ticker, scenario, SUM(pv) AS pv_explicit,
           MAX(projected_fcf) FILTER (WHERE yr = 10) AS fcf_y10,
           ANY_VALUE(tg) AS tg, ANY_VALUE(drs) AS drs
    FROM v_dcf_projection GROUP BY ticker, scenario
),
tot AS (
    SELECT ticker, scenario, pv_explicit,
           CASE WHEN drs > tg THEN fcf_y10 * (1 + tg/100) / ((drs - tg)/100) / POW(1 + drs/100, 10)
                ELSE 0 END AS pv_terminal
    FROM tv
),
ps AS (
    SELECT t.ticker,
           MAX((t.pv_explicit + t.pv_terminal)/i.shares_cr) FILTER (WHERE t.scenario='bear') AS iv_bear,
           MAX((t.pv_explicit + t.pv_terminal)/i.shares_cr) FILTER (WHERE t.scenario='base') AS iv_base,
           MAX((t.pv_explicit + t.pv_terminal)/i.shares_cr) FILTER (WHERE t.scenario='bull') AS iv_bull,
           MAX(t.pv_terminal/(t.pv_explicit + t.pv_terminal)) FILTER (WHERE t.scenario='base') AS tv_share
    FROM tot t JOIN v_dcf_inputs i USING (ticker) GROUP BY t.ticker
)
SELECT i.ticker,
       ROUND(i.price, 2) AS price,  
       ROUND(i.norm_profit, 0) AS norm_profit_cr,  
       ROUND(i.norm_fcf, 0) AS norm_fcf_cr,
       ROUND(i.shares_cr, 2) AS shares_cr,
       i.g1, i.g2, i.tg, i.dr, i.mos, i.lam AS decay_lambda,
       ROUND(ps.iv_bear, 0) AS iv_bear,  ROUND(ps.iv_base, 0) AS iv_base,  ROUND(ps.iv_bull, 0) AS iv_bull,
       ROUND(ps.iv_base * (1 - i.mos/100), 0)                AS mos_price,
       ROUND(pe.wmedian_pe, 1)                               AS pe_wmedian,
       pe.latest_eps                                         AS latest_eps,
       ROUND(pe.wmedian_pe * pe.latest_eps, 0)               AS pe_fair_value,
       ROUND((i.price - pe.wmedian_pe * pe.latest_eps)
             / NULLIF(pe.wmedian_pe * pe.latest_eps, 0) * 100, 1) AS pe_premium_pct, 
       CASE WHEN pe.wmedian_pe * pe.latest_eps IS NULL THEN NULL
            WHEN i.price <= pe.wmedian_pe * pe.latest_eps THEN 'CHEAP vs own P/E history'
            ELSE 'RICH vs own P/E history' END               AS pe_verdict,
       ROUND((i.price - ps.iv_base) / ps.iv_base * 100, 1)   AS premium_to_base_pct,
       ROUND((ps.iv_base - i.price) / ps.iv_base * 100, 1)   AS margin_of_safety_pct,
       ROUND(ps.tv_share * 100, 0)                           AS terminal_value_pct,
       CASE WHEN i.price <= ps.iv_base * (1 - i.mos/100) THEN 'BUY (inside MoS)'
            WHEN i.price <= ps.iv_base                   THEN 'UNDERVALUED'
            WHEN i.price <= ps.iv_bull                   THEN 'OVERVALUED vs base'
            ELSE 'OVERVALUED even vs bull' END             AS verdict
FROM v_dcf_inputs i JOIN ps USING (ticker)
LEFT JOIN v_dcf_pe_history pe ON pe.ticker = i.ticker
WHERE i.shares_cr > 0 AND i.price IS NOT NULL;
