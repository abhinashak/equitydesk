-- ============================================================
-- Creates DuckDB virtual views over the per-ticker parquet files.
-- ============================================================

-- General company info (one row per ticker)
CREATE OR REPLACE VIEW general_info_raw AS
SELECT * FROM 'data/fundamental/*/general_info.parquet';

-- Pros & Cons (one row per bullet point per ticker)
CREATE OR REPLACE VIEW pros_raw AS
SELECT * FROM 'data/fundamental/*/pros.parquet';

CREATE OR REPLACE VIEW cons_raw AS
SELECT * FROM 'data/fundamental/*/cons.parquet';

-- Financial tables (long format: ticker | period | metric | value)
CREATE OR REPLACE VIEW quarterly_results_raw AS
SELECT * FROM 'data/fundamental/*/quarterly_results.parquet';

CREATE OR REPLACE VIEW profit_loss_raw AS
SELECT * FROM 'data/fundamental/*/profit_loss.parquet';

CREATE OR REPLACE VIEW balance_sheet_raw AS
SELECT * FROM 'data/fundamental/*/balance_sheet.parquet';

CREATE OR REPLACE VIEW cash_flows_raw AS
SELECT * FROM 'data/fundamental/*/cash_flows.parquet';

CREATE OR REPLACE VIEW ratios_raw AS
SELECT * FROM 'data/fundamental/*/ratios.parquet';

-- CAGR / growth tables (ticker | horizon | value)
CREATE OR REPLACE VIEW compounded_sales_growth_raw AS
SELECT * FROM 'data/fundamental/*/compounded_sales_growth.parquet';

CREATE OR REPLACE VIEW compounded_profit_growth_raw AS
SELECT * FROM 'data/fundamental/*/compounded_profit_growth.parquet';

CREATE OR REPLACE VIEW stock_price_cagr_raw AS
SELECT * FROM 'data/fundamental/*/stock_price_cagr.parquet';

CREATE OR REPLACE VIEW return_on_equity_raw AS
SELECT * FROM 'data/fundamental/*/return_on_equity.parquet';

-- Shareholding pattern (ticker | period | category | value)
CREATE OR REPLACE VIEW shareholding_quarterly_raw AS
SELECT * FROM 'data/fundamental/*/shareholding_quarterly.parquet';

CREATE OR REPLACE VIEW shareholding_yearly_raw AS
SELECT * FROM 'data/fundamental/*/shareholding_yearly.parquet';

