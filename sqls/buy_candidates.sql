-- ============================================================================
-- abinash_recommendations.sql
-- A deliberately LIBERAL shortlist of stocks worth a closer look, built by
-- joining four lenses on ticker:
--
--   quality_gates   -> view gate_results            (quality_gates.sql)
--   momentum_algo   -> views fss_scored / fss_book  (fundamental_spread_signal.sql)
--   ticker_momentum -> latest technical row per ticker
--   v_dcf_valuation -> DCF intrinsic value vs price
--
-- LENSES (each is PASS / FAIL / not-available; not-available never counts against)
--   1 Quality   : quality_score >= 40                       (app's "Watch" floor, Q>=70 = strong)
--   2 Value     : DCF base value (iv_base) within 10% of price, or price <= mos_price,
--                 or the DCF verdict says BUY; with no DCF row, gate valuation_score >= 40
--   3 Momentum  : at least 2 of 4 -> above 200DMA, 6M return > 0, RSI 40..75, MACD bullish
--   4 Algo      : sector-relative composite > 0, or in the algo's long book
--
-- SHORTLIST RULE : Quality passes AND at least 2 lenses pass in total.
-- TIERS          : A = 3+ lenses and Quality >= 70 | B = 3+ lenses | C = exactly 2 (watchlist)
-- conviction     : 0..100 average of the available lens scores (sorting aid, not a forecast)
--
-- USAGE (from the project root; the first three scripts print a lot, so their
-- output is silenced and only this script's tables are shown):
--
--   { cat sqls/init.sql sqls/compact_clean.sql
--     echo ".output /dev/null"
--     cat quality_gates.sql fundamental_spread_signal.sql
--     echo ".output"
--     cat abinash_recommendations.sql; } | duckdb -markdown
--
--   ... | duckdb -csv > abinash_recommendations.csv     (export the shortlist)
--
-- Afterwards, in the same session:
--   SELECT * FROM abinash_recommendations;
--   SELECT * FROM abinash_screen WHERE ticker = 'NATIONALUM';   -- every lens for one stock
--   UPDATE abinash_params SET quality_min = 60, lenses_min = 3;  -- make it stricter
--
-- IF A COLUMN IS NOT FOUND: all source-column names live in the four
-- abinash_src_* views right below. Edit only those lines.
-- ============================================================================


-- ----------------------------------------------------------------------------
-- Thresholds ("liberal" defaults). Tighten with UPDATE abinash_params ...
-- ----------------------------------------------------------------------------
CREATE OR REPLACE TABLE abinash_params AS
SELECT
    40::INTEGER    AS quality_min,        -- lens 1: min quality_score
    70::INTEGER    AS quality_strong,     -- tier A needs this
    -10.0::DOUBLE  AS dcf_upside_min_pct, -- lens 2: price may be up to 10% above DCF value
    40::INTEGER    AS valuation_min,      -- lens 2 fallback when no DCF row
    2::INTEGER     AS momentum_tests_min, -- lens 3: of the 4 tests
    40.0::DOUBLE   AS rsi_lo,
    75.0::DOUBLE   AS rsi_hi,
    0.0::DOUBLE    AS algo_composite_min, -- lens 4
    75.0::DOUBLE   AS tv_pct_caution,     -- caution if DCF value is mostly terminal value
    2::INTEGER     AS lenses_min;         -- shortlist: lenses that must pass


-- ============================================================================
-- SOURCE ADAPTERS  (one place to map your column names; ticker is normalised
-- to upper-case without '.NS' so all sources join cleanly)
-- ============================================================================

-- quality_gates (output of quality_gates.sql)
CREATE OR REPLACE VIEW abinash_src_quality AS
SELECT REPLACE(UPPER(ticker), '.NS', '') AS ticker,
       company_name, sector, market_cap, stock_p_e, current_price,
       roce, de, profit_cagr, pe_discount_pct,
       quality_score, valuation_score, timing_score, technical_score,
       recommendation AS gate_rec,
       g1b_fcf_fail, g1e_debt_fail, g3d_promoter_fail
FROM gate_results;

-- ticker_momentum: latest row per ticker
CREATE OR REPLACE VIEW abinash_src_momentum AS
SELECT REPLACE(UPPER(nse_symbol), '.NS', '') AS ticker,
       date        AS mom_date,
       close       AS mom_close,
       "6M"        AS ret_6m,
       "1Y"        AS ret_1y,
       sma50_dist, sma200_dist, rsi, macd_bullish, adx
FROM ticker_momentum
QUALIFY ROW_NUMBER() OVER (PARTITION BY REPLACE(UPPER(nse_symbol), '.NS', '')
                           ORDER BY date DESC) = 1;

-- v_dcf_valuation (iv_base = base-case intrinsic value, price = current price)
CREATE OR REPLACE VIEW abinash_src_dcf AS
SELECT REPLACE(UPPER(ticker), '.NS', '') AS ticker,
       price             AS dcf_price,
       iv_bear           AS dcf_bear,
       iv_base           AS dcf_value,
       iv_bull           AS dcf_bull,
       mos_price         AS dcf_mos_price,
       terminal_value_pct AS dcf_tv_pct,
       verdict           AS dcf_verdict,
       pe_verdict        AS dcf_pe_verdict
FROM v_dcf_valuation
QUALIFY ROW_NUMBER() OVER (PARTITION BY REPLACE(UPPER(ticker), '.NS', '')
                           ORDER BY ticker) = 1;

-- momentum_algo = Fundamental Spread Signal (fss_scored + long/short side)
CREATE OR REPLACE VIEW abinash_src_algo AS
SELECT REPLACE(UPPER(s.ticker), '.NS', '') AS ticker,
       s.composite AS algo_composite,
       s.industry  AS algo_industry,
       b.side      AS algo_side
FROM fss_scored s
LEFT JOIN fss_book b ON b.ticker = s.ticker;


-- ============================================================================
-- SCREEN: every ticker with all four lenses, flags, scores and notes
-- ============================================================================
CREATE OR REPLACE VIEW abinash_screen AS
WITH j AS (
    SELECT q.*,
           d.dcf_price, d.dcf_bear, d.dcf_value, d.dcf_bull, d.dcf_mos_price,
           d.dcf_tv_pct, d.dcf_verdict, d.dcf_pe_verdict,
           m.mom_date, m.mom_close, m.ret_6m, m.ret_1y,
           m.sma50_dist, m.sma200_dist, m.rsi, m.macd_bullish, m.adx,
           a.algo_composite, a.algo_industry, a.algo_side
    FROM abinash_src_quality q
    LEFT JOIN abinash_src_dcf      d ON d.ticker = q.ticker
    LEFT JOIN abinash_src_momentum m ON m.ticker = q.ticker
    LEFT JOIN abinash_src_algo     a ON a.ticker = q.ticker
),
t AS (
    SELECT j.*,
           CASE WHEN dcf_price > 0 AND dcf_value IS NOT NULL
                THEN (dcf_value / dcf_price - 1) * 100 END AS dcf_upside_pct,
           CASE WHEN mom_date IS NOT NULL THEN
                COALESCE(sma200_dist > 0, FALSE)::INT
              + COALESCE(ret_6m > 0, FALSE)::INT
              + COALESCE(rsi BETWEEN p.rsi_lo AND p.rsi_hi, FALSE)::INT
              + COALESCE(macd_bullish = 1, FALSE)::INT
           END AS mom_tests
    FROM j CROSS JOIN abinash_params p
),
l AS (
    SELECT t.*,
           -- lens 1
           (quality_score >= p.quality_min)                          AS l_quality,
           -- lens 2
           CASE WHEN dcf_upside_pct IS NOT NULL
                THEN (dcf_upside_pct >= p.dcf_upside_min_pct
                      OR COALESCE(dcf_price <= dcf_mos_price, FALSE)
                      OR UPPER(COALESCE(dcf_verdict, '')) LIKE '%BUY%')
                ELSE (valuation_score >= p.valuation_min) END        AS l_value,
           -- lens 3
           CASE WHEN mom_tests IS NOT NULL
                THEN mom_tests >= p.momentum_tests_min END           AS l_momentum,
           -- lens 4
           CASE WHEN algo_composite IS NOT NULL
                THEN (algo_composite > p.algo_composite_min OR COALESCE(algo_side = 'L', FALSE)) END AS l_algo
    FROM t CROSS JOIN abinash_params p
),
c AS (
    SELECT l.*,
           COALESCE(l_quality, FALSE)::INT + COALESCE(l_value, FALSE)::INT
         + COALESCE(l_momentum, FALSE)::INT + COALESCE(l_algo, FALSE)::INT AS lenses_passed,
           (l_quality IS NOT NULL)::INT + (l_value IS NOT NULL)::INT
         + (l_momentum IS NOT NULL)::INT + (l_algo IS NOT NULL)::INT        AS lenses_available
    FROM l
)
SELECT
    c.ticker, c.company_name, c.sector, c.market_cap, c.stock_p_e, c.current_price,
    c.roce, c.de, c.profit_cagr, c.pe_discount_pct,
    c.quality_score, c.valuation_score, c.timing_score, c.technical_score, c.gate_rec,
    c.dcf_price, c.dcf_bear, c.dcf_value, c.dcf_bull, c.dcf_mos_price, c.dcf_upside_pct,
    c.dcf_tv_pct, c.dcf_verdict, c.dcf_pe_verdict,
    c.mom_date, c.ret_6m, c.ret_1y, c.sma50_dist, c.sma200_dist, c.rsi, c.macd_bullish, c.adx,
    c.mom_tests, c.algo_composite, c.algo_side, c.algo_industry,
    c.l_quality, c.l_value, c.l_momentum, c.l_algo,
    c.lenses_passed, c.lenses_available,
    (c.l_quality AND c.lenses_passed >= p.lenses_min)                AS shortlisted,
    CASE WHEN NOT (c.l_quality AND c.lenses_passed >= p.lenses_min) THEN NULL
         WHEN c.lenses_passed >= 3 AND c.quality_score >= p.quality_strong THEN 'A'
         WHEN c.lenses_passed >= 3 THEN 'B'
         ELSE 'C' END                                                AS tier,
    -- 0..100 average of whichever lens scores exist (NULLs are ignored by list_avg)
    list_avg([
        c.quality_score::DOUBLE,
        CASE WHEN c.dcf_upside_pct IS NOT NULL
             THEN LEAST(100.0, GREATEST(0.0, 50 + c.dcf_upside_pct))
             ELSE c.valuation_score::DOUBLE END,
        c.mom_tests * 25.0,
        LEAST(100.0, GREATEST(0.0, 50 + 25 * c.algo_composite))
    ])                                                               AS conviction,
    -- why it passed (concat_ws skips NULL parts)
    NULLIF(concat_ws('; ',
        CASE WHEN c.l_quality  THEN 'Quality ' || c.quality_score END,
        CASE WHEN c.l_value AND c.dcf_upside_pct IS NOT NULL
             THEN 'DCF ' || printf('%+.0f%%', c.dcf_upside_pct)
                  || CASE WHEN c.dcf_bear >= c.dcf_price THEN ' (even bear case >= price)' ELSE '' END
             WHEN c.l_value THEN 'Valuation score ' || c.valuation_score END,
        CASE WHEN c.l_momentum THEN 'Momentum ' || c.mom_tests || '/4' END,
        CASE WHEN c.l_algo     THEN 'Algo ' || printf('%+.2f', c.algo_composite)
                                    || COALESCE(' (' || c.algo_side || ')', '') END
    ), '')                                                           AS why,
    -- things to look at before buying
    NULLIF(concat_ws('; ',
        CASE WHEN c.rsi > p.rsi_hi                         THEN 'RSI overbought' END,
        CASE WHEN c.sma200_dist < 0                        THEN 'below 200DMA' END,
        CASE WHEN UPPER(COALESCE(c.dcf_verdict, '')) LIKE '%OVERVALUED%' THEN 'DCF says overvalued' END,
        CASE WHEN UPPER(COALESCE(c.dcf_pe_verdict, '')) LIKE '%OVERVALUED%' THEN 'P/E says overvalued' END,
        CASE WHEN c.dcf_tv_pct > p.tv_pct_caution          THEN 'DCF mostly terminal value' END,
        CASE WHEN c.g1b_fcf_fail                           THEN 'negative FCF' END,
        CASE WHEN c.g1e_debt_fail                          THEN 'high debt' END,
        CASE WHEN c.g3d_promoter_fail                      THEN 'promoter selling' END,
        CASE WHEN c.algo_side = 'S'                        THEN 'algo short-side' END
    ), '')                                                           AS cautions
FROM c CROSS JOIN abinash_params p;


-- ============================================================================
-- THE SHORTLIST
-- ============================================================================
CREATE OR REPLACE VIEW abinash_recommendations AS
SELECT
    tier, ticker, company_name, sector,
    ROUND(conviction, 1)        AS conviction,
    lenses_passed, lenses_available,
    why, cautions,
    quality_score, valuation_score, timing_score, technical_score, gate_rec,
    dcf_price, dcf_bear, dcf_value, dcf_bull, ROUND(dcf_upside_pct, 1) AS dcf_upside_pct,
    dcf_verdict, dcf_pe_verdict,
    ret_6m, ret_1y, ROUND(sma200_dist, 1) AS sma200_dist, ROUND(rsi, 1) AS rsi, mom_tests,
    ROUND(algo_composite, 3) AS algo_composite, algo_side, algo_industry,
    market_cap, current_price, stock_p_e
FROM abinash_screen
WHERE shortlisted
ORDER BY tier, conviction DESC, ticker;


-- ============================================================================
-- OUTPUT 1: how big is the shortlist
-- ============================================================================
SELECT
    COUNT(*)                                        AS universe,
    COUNT(*) FILTER (WHERE shortlisted)             AS shortlisted,
    COUNT(*) FILTER (WHERE tier = 'A')              AS tier_A,
    COUNT(*) FILTER (WHERE tier = 'B')              AS tier_B,
    COUNT(*) FILTER (WHERE tier = 'C')              AS tier_C,
    COUNT(*) FILTER (WHERE dcf_value IS NOT NULL)   AS with_dcf,
    COUNT(*) FILTER (WHERE mom_date IS NOT NULL)    AS with_momentum,
    COUNT(*) FILTER (WHERE algo_composite IS NOT NULL) AS with_algo
FROM abinash_screen;


-- ============================================================================
-- OUTPUT 2: abinash_recommendations (best first)
-- ============================================================================
SELECT tier, ticker, LEFT(company_name, 22) AS company, sector,
       conviction, lenses_passed AS passed, lenses_available AS avail,
       why, cautions
FROM abinash_recommendations
ORDER BY tier, conviction DESC, ticker;
