-- ============================================================================
-- quality_gates.sql
-- Command-line version of the 4-gate engine (one_ticker_common.compute_scores
-- + one_ticker_gates._recommendation).
--
--   Gate 1  Quality     (5 tests, +20 per pass, -15 per fail)
--   Gate 2  Valuation   (3 tests, +33 per pass, -20 per fail)
--   Gate 3  Timing      (5 tests, +20 per pass, -15 per fail)
--   Gate 4  Technical   (4 tests, +25 per pass, -15 per fail)
--   Each score is clamped to 0..100.
--
-- USAGE (run from the project root so init.sql parquet paths resolve):
--   cat sqls/init.sql quality_gates.sql | duckdb
--   cat sqls/init.sql quality_gates.sql | duckdb -markdown
--   cat sqls/init.sql quality_gates.sql | duckdb -csv > gates.csv
--
-- After it runs, the view gate_results stays available in that session, e.g.
--   SELECT * FROM gate_results WHERE ticker = 'NATIONALUM'
--
-- Input : view gate_score (columns used are listed in the "derived" CTE below)
-- Output: view gate_results (flags, scores, recommendation) + a summary table
--
-- Notes
--   * A test that cannot be evaluated (missing data) is neither PASS nor FAIL,
--     exactly like the grey cell in the app.
--   * Gate 4c: set dip_mode to TRUE below for the "10-40% off high" window.
-- ============================================================================

CREATE OR REPLACE VIEW gate_results AS
WITH
params AS (
    SELECT FALSE AS dip_mode                      -- <<< toggle Gate 4c mode here
),

-- ---------------------------------------------------------------------------
-- Derived metrics
-- ---------------------------------------------------------------------------
derived AS (
    SELECT
        g.*,
        p.dip_mode,

        -- 3Y profit CAGR (%), only when both profits are positive
        CASE WHEN g.profit_now > 0 AND g.profit_3y > 0
             THEN (POWER(g.profit_now / g.profit_3y, 1.0 / 3) - 1) * 100
        END AS profit_cagr,

        -- Debt / Equity, only when equity (reserves + capital) is positive
        CASE WHEN g.reserves + g.equity_capital > 0
             THEN g.debt / (g.reserves + g.equity_capital)
        END AS de,

        -- Financial sectors are exempt from the debt test
        (   LOWER(COALESCE(g.sector, '')) LIKE '%banks%'
         OR LOWER(COALESCE(g.sector, '')) LIKE '%finance%'
         OR LOWER(COALESCE(g.sector, '')) LIKE '%nbfc%'
         OR LOWER(COALESCE(g.sector, '')) LIKE '%insurance%') AS is_financial,

        -- P/E discount to sector median (%), positive = cheaper than sector
        CASE WHEN g.stock_p_e > 0 AND g.sector_median_pe > 0
             THEN (g.sector_median_pe - g.stock_p_e) / g.sector_median_pe * 100
        END AS pe_discount_pct,

        -- Price / Book
        CASE WHEN g.book_value > 0
             THEN g.current_price / g.book_value
        END AS pb
    FROM gate_score g
    CROSS JOIN params p
),

derived2 AS (
    SELECT
        d.*,
        -- PEG = P/E divided by profit CAGR, only when CAGR > 0
        CASE WHEN d.profit_cagr > 0 THEN d.stock_p_e / d.profit_cagr END AS peg
    FROM derived d
),

-- ---------------------------------------------------------------------------
-- Pass / Fail flags (NULL data -> both FALSE)
-- ---------------------------------------------------------------------------
flags AS (
    SELECT
        d.*,

        -- ===== GATE 1: QUALITY ==============================================
        COALESCE(roce >= 18, FALSE)                                   AS g1a_roce_pass,
        COALESCE(roce < 8,  FALSE)                                    AS g1a_roce_fail,

        COALESCE(fcf > 0, FALSE)                                      AS g1b_fcf_pass,
        COALESCE(fcf < 0, FALSE)                                      AS g1b_fcf_fail,

        COALESCE(profit_cagr >= 15, FALSE)                            AS g1c_cagr_pass,
        COALESCE(profit_cagr < 0,   FALSE)                            AS g1c_cagr_fail,

        COALESCE(profit_now > 0 AND cfo > profit_now, FALSE)          AS g1d_cfo_pass,
        COALESCE(profit_now > 0 AND cfo < profit_now * 0.5, FALSE)    AS g1d_cfo_fail,

        (NOT is_financial) AND COALESCE(de < 0.5, FALSE)              AS g1e_debt_pass,
        (NOT is_financial) AND COALESCE(de > 1.5, FALSE)              AS g1e_debt_fail,

        -- ===== GATE 2: VALUATION ============================================
        COALESCE(pe_discount_pct >= 15,  FALSE)                       AS g2a_pe_pass,
        COALESCE(pe_discount_pct < -30,  FALSE)                       AS g2a_pe_fail,

        COALESCE(peg < 1.0,  FALSE)                                   AS g2b_peg_pass,
        COALESCE(peg > 2.5,  FALSE)                                   AS g2b_peg_fail,

        COALESCE(pb < 3.0,  FALSE)                                    AS g2c_pb_pass,
        COALESCE(pb > 8.0,  FALSE)                                    AS g2c_pb_fail,

        -- ===== GATE 3: TIMING ===============================================
        COALESCE(sales_q0 > sales_q1 AND sales_q1 > sales_q2, FALSE)  AS g3a_sales_pass,
        (sales_q0 IS NOT NULL AND sales_q1 IS NOT NULL AND sales_q2 IS NOT NULL
         AND sales_q0 < sales_q1)                                     AS g3a_sales_fail,

        COALESCE(profit_q0 > profit_q1, FALSE)                        AS g3b_profit_pass,
        COALESCE(profit_q0 < profit_q1 AND profit_q0 > 0, FALSE)      AS g3b_profit_fail,

        COALESCE(opm_q0 > opm_q1, FALSE)                              AS g3c_opm_pass,
        COALESCE(opm_q0 < opm_q1 - 3, FALSE)                          AS g3c_opm_fail,

        COALESCE(promoter >= promoter_prev, FALSE)                    AS g3d_promoter_pass,
        COALESCE(promoter < promoter_prev - 2, FALSE)                 AS g3d_promoter_fail,

        (fii IS NOT NULL AND fii_prev IS NOT NULL
         AND dii IS NOT NULL AND dii_prev IS NOT NULL
         AND (fii >= fii_prev OR dii >= dii_prev))                    AS g3e_inst_pass,
        (fii IS NOT NULL AND fii_prev IS NOT NULL
         AND dii IS NOT NULL AND dii_prev IS NOT NULL
         AND fii < fii_prev AND dii < dii_prev)                       AS g3e_inst_fail,

        -- ===== GATE 4: TECHNICAL ============================================
        COALESCE(bb_width <= min_bb_63d * 1.15, FALSE)                AS g4a_bb_pass,
        COALESCE(bb_width > 0.35, FALSE)                              AS g4a_bb_fail,

        COALESCE(dist_from_ma50 > 0 AND dist_from_ma50 < 12, FALSE)   AS g4b_ma50_pass,
        COALESCE(dist_from_ma50 > 25 OR dist_from_ma50 < -8, FALSE)   AS g4b_ma50_fail,

        CASE WHEN dip_mode
             THEN COALESCE(pct_from_52w BETWEEN -40 AND -10, FALSE)
             ELSE COALESCE(pct_from_52w >= -5, FALSE)
        END                                                           AS g4c_52w_pass,
        COALESCE(pct_from_52w < -40, FALSE)                           AS g4c_52w_fail,

        COALESCE(volume_ratio >= 1.1, FALSE)                          AS g4d_vol_pass,
        COALESCE(volume_ratio < 0.6,  FALSE)                          AS g4d_vol_fail
    FROM derived2 d
),

-- ---------------------------------------------------------------------------
-- Scores (0..100)
-- ---------------------------------------------------------------------------
scored AS (
    SELECT
        f.*,
        GREATEST(0, LEAST(100,
              (g1a_roce_pass::INT + g1b_fcf_pass::INT + g1c_cagr_pass::INT
               + g1d_cfo_pass::INT + g1e_debt_pass::INT) * 20
            - (g1a_roce_fail::INT + g1b_fcf_fail::INT + g1c_cagr_fail::INT
               + g1d_cfo_fail::INT + g1e_debt_fail::INT) * 15))   AS quality_score,

        GREATEST(0, LEAST(100,
              (g2a_pe_pass::INT + g2b_peg_pass::INT + g2c_pb_pass::INT) * 33
            - (g2a_pe_fail::INT + g2b_peg_fail::INT + g2c_pb_fail::INT) * 20))
                                                                  AS valuation_score,

        GREATEST(0, LEAST(100,
              (g3a_sales_pass::INT + g3b_profit_pass::INT + g3c_opm_pass::INT
               + g3d_promoter_pass::INT + g3e_inst_pass::INT) * 20
            - (g3a_sales_fail::INT + g3b_profit_fail::INT + g3c_opm_fail::INT
               + g3d_promoter_fail::INT + g3e_inst_fail::INT) * 15)) AS timing_score,

        GREATEST(0, LEAST(100,
              (g4a_bb_pass::INT + g4b_ma50_pass::INT + g4c_52w_pass::INT
               + g4d_vol_pass::INT) * 25
            - (g4a_bb_fail::INT + g4b_ma50_fail::INT + g4c_52w_fail::INT
               + g4d_vol_fail::INT) * 15))                        AS technical_score
    FROM flags f
)

SELECT
    s.*,
    -- Recommendation ladder (same as one_ticker_gates._recommendation)
    CASE
        WHEN quality_score >= 70 AND valuation_score >= 70
             AND (timing_score >= 60 OR technical_score >= 60)    THEN 'Strong Buy'
        WHEN quality_score >= 70 AND valuation_score >= 40
             AND (timing_score >= 40 OR technical_score >= 40)    THEN 'Buy'
        WHEN quality_score >= 70 AND valuation_score >= 40        THEN 'Accumulate'
        WHEN quality_score >= 70 AND valuation_score <  40        THEN 'Overvalued'
        WHEN quality_score >= 40                                  THEN 'Watch'
        ELSE 'Avoid'
    END AS recommendation
FROM scored s;


-- Helper: P = pass, F = fail, . = not available
CREATE OR REPLACE MACRO gate_sym(p, n) AS
    CASE WHEN p THEN 'P' WHEN n THEN 'F' ELSE '.' END;


-- ============================================================================
-- OUTPUT 1: Gate matrix for every ticker (best quality first)
-- ============================================================================
SELECT
    ticker,
    LEFT(company_name, 22)                                  AS company,
    quality_score                                           AS Q,
    valuation_score                                         AS V,
    timing_score                                            AS T,
    technical_score                                         AS Tech,
    recommendation                                          AS rec,
    gate_sym(g1a_roce_pass,     g1a_roce_fail)              AS "1a ROCE",
    gate_sym(g1b_fcf_pass,      g1b_fcf_fail)               AS "1b FCF",
    gate_sym(g1c_cagr_pass,     g1c_cagr_fail)              AS "1c CAGR",
    gate_sym(g1d_cfo_pass,      g1d_cfo_fail)               AS "1d CFO",
    gate_sym(g1e_debt_pass,     g1e_debt_fail)              AS "1e Debt",
    gate_sym(g2a_pe_pass,       g2a_pe_fail)                AS "2a PE",
    gate_sym(g2b_peg_pass,      g2b_peg_fail)               AS "2b PEG",
    gate_sym(g2c_pb_pass,       g2c_pb_fail)                AS "2c PB",
    gate_sym(g3a_sales_pass,    g3a_sales_fail)             AS "3a Sales",
    gate_sym(g3b_profit_pass,   g3b_profit_fail)            AS "3b Profit",
    gate_sym(g3c_opm_pass,      g3c_opm_fail)               AS "3c OPM",
    gate_sym(g3d_promoter_pass, g3d_promoter_fail)          AS "3d Promo",
    gate_sym(g3e_inst_pass,     g3e_inst_fail)              AS "3e Inst",
    gate_sym(g4a_bb_pass,       g4a_bb_fail)                AS "4a BB",
    gate_sym(g4b_ma50_pass,     g4b_ma50_fail)              AS "4b MA50",
    gate_sym(g4c_52w_pass,      g4c_52w_fail)               AS "4c 52W",
    gate_sym(g4d_vol_pass,      g4d_vol_fail)               AS "4d Vol"
FROM gate_results
ORDER BY quality_score DESC, valuation_score DESC, ticker;


-- ============================================================================
-- OPTIONAL QUERIES (uncomment to use, or run them in the same session)
-- ============================================================================

-- A) Quality gate only, with the underlying numbers
-- SELECT ticker, ROUND(roce,1) AS roce, ROUND(profit_cagr,1) AS profit_cagr_3y,
--        ROUND(de,2) AS d_e, fcf, cfo, profit_now, quality_score
-- FROM gate_results
-- WHERE quality_score >= 70
-- ORDER BY quality_score DESC, roce DESC;

-- B) Single-ticker breakdown
-- SELECT * FROM gate_results WHERE ticker = 'NATIONALUM';

-- C) Only actionable names
-- SELECT ticker, quality_score, valuation_score, timing_score, technical_score, recommendation
-- FROM gate_results
-- WHERE recommendation IN ('Strong Buy', 'Buy')
-- ORDER BY quality_score DESC, valuation_score DESC;
