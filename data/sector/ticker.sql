

CREATE OR REPLACE VIEW my_tickers AS
         SELECT
             Name AS company_name,
             "Yahoo Symbol" AS ticker_yahoo,
             Sector AS my_sector,
             UPPER(TRIM(REGEXP_REPLACE("Yahoo Symbol", '\.(NS|BO|BS)$', ''))) AS ticker
         FROM read_csv_auto('../../config/tickers.csv', header = true);


CREATE OR REPLACE VIEW nifty_500 AS
             SELECT
                 Instrument as ticker, Sector as kite_sector
             FROM read_csv_auto('NIFTY500.csv', header = true);
         


CREATE OR REPLACE VIEW sector_hierarchy AS 
             SELECT
                 "Macro Sector" as macro_sector, "Sector Group" as "sector_group", "Sub-Sector / Industry" AS kite_sector
             FROM read_csv_auto('nifty_500_sectors_hierarchy.csv', header = true);
         

CREATE OR REPLACE VIEW nifty_500_sectors as Select n.*,sector_group,macro_sector, from nifty_500 n, sector_hierarchy s where n. kite_sector = s. kite_sector;

select * from my_tickers m, nifty_500_sectors ns where m.ticker = ns.ticker limit 5;


CREATE OR REPLACE VIEW ticker_midcap150 as select Company as company_name, SYMBOL as ticker, 'midcap150' as market_cap_segment, "Mar Cap Rs.Cr." as market_cap_cr, "FII Hold %" as fii_hold_pct, "Public Hold %" as public_hold_pct  FROM read_csv_auto('midcap150_with_symbols.csv', header = true);
CREATE OR REPLACE VIEW ticker_smallcap250 as select Company as company_name, SYMBOL as ticker, 'smallcap250' as market_cap_segment, "Mar Cap Rs.Cr." as market_cap_cr, "FII Hold %" as fii_hold_pct, "Public Hold %" as public_hold_pct  FROM read_csv_auto('smallcap250_with_symbols.csv', header = true);
CREATE OR REPLACE VIEW ticker_nifty50 as select Company as company_name, SYMBOL as ticker, 'nifty50' as market_cap_segment, "Mar Cap Rs.Cr." as market_cap_cr, "FII Hold %" as fii_hold_pct, "Public Hold %" as public_hold_pct  FROM read_csv_auto('nifty50_with_symbols.csv', header = true);

create view ticker_screener as
Select * from ticker_midcap150 
Union
Select * from ticker_nifty50 
Union
Select * from ticker_smallcap250;

CREATE OR REPLACE VIEW enriched_tickers AS
WITH ns_dedup AS (
    SELECT
        UPPER(TRIM(ticker)) AS ticker,
        kite_sector,
        sector_group,
        macro_sector
    FROM nifty_500_sectors
    WHERE ticker IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (PARTITION BY UPPER(TRIM(ticker)) ORDER BY kite_sector NULLS LAST) = 1
),
ts_dedup AS (
    SELECT
        UPPER(TRIM(ticker)) AS ticker,
        company_name,
        market_cap_segment,
        market_cap_cr,
        fii_hold_pct,
        public_hold_pct
    FROM ticker_screener
    WHERE ticker IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (PARTITION BY UPPER(TRIM(ticker)) ORDER BY market_cap_cr DESC NULLS LAST) = 1
)
SELECT
    COALESCE(ts.company_name, m.company_name) AS company_name,
    m.company_name AS raw_company_name,
    m.ticker_yahoo,
    UPPER(TRIM(m.ticker)) AS ticker,
    m.my_sector,
    ns.kite_sector,
    ns.sector_group,
    ns.macro_sector,
    ts.market_cap_segment,
    ts.market_cap_cr,
    ts.fii_hold_pct,
    ts.public_hold_pct
FROM my_tickers AS m
LEFT JOIN ns_dedup AS ns
    ON UPPER(TRIM(m.ticker)) = ns.ticker
LEFT JOIN ts_dedup AS ts
    ON UPPER(TRIM(m.ticker)) = ts.ticker;


CREATE OR REPLACE VIEW delta_tickers AS
WITH existing_tickers AS (
    SELECT DISTINCT UPPER(TRIM(ticker)) AS ticker
    FROM my_tickers
    WHERE ticker IS NOT NULL
    UNION
    SELECT DISTINCT UPPER(TRIM(REGEXP_REPLACE(ticker_yahoo, '\.(NS|BO|BS)$', ''))) AS ticker
    FROM my_tickers
    WHERE ticker_yahoo IS NOT NULL
),
ns_dedup AS (
    SELECT
        UPPER(TRIM(ticker)) AS ticker,
        kite_sector,
        sector_group,
        macro_sector
    FROM nifty_500_sectors
    WHERE ticker IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (PARTITION BY UPPER(TRIM(ticker)) ORDER BY kite_sector NULLS LAST) = 1
),
ts_dedup AS (
    SELECT
        UPPER(TRIM(ticker)) AS ticker,
        company_name,
        market_cap_segment ,
        market_cap_cr,
        fii_hold_pct,
        public_hold_pct
    FROM ticker_screener
    WHERE ticker IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (PARTITION BY UPPER(TRIM(ticker)) ORDER BY market_cap_cr DESC NULLS LAST) = 1
)

SELECT
    ts.company_name,
    ts.ticker || '.NS' AS ticker_yahoo,
    ts.ticker,
    ns.kite_sector,
    ns.sector_group,
    ns.macro_sector,
    ts.market_cap_segment,
    ts.market_cap_cr,
    ts.fii_hold_pct,
    ts.public_hold_pct
FROM ts_dedup AS ts
LEFT JOIN ns_dedup AS ns
    ON ts.ticker = ns.ticker
WHERE ts.ticker NOT IN (
    SELECT ticker FROM existing_tickers WHERE ticker IS NOT NULL
);


COPY (
    WITH combined AS (
        SELECT
            company_name AS Name,
            ticker_yahoo AS "Yahoo Symbol",
            kite_sector AS Sector,
            UPPER(TRIM(ticker)) AS ticker,
            sector_group,
            macro_sector,
            market_cap_segment  as market_cap,
            market_cap_cr,
            fii_hold_pct,
            public_hold_pct,
            1 AS priority
        FROM enriched_tickers

        UNION ALL

        SELECT
            company_name AS Name,
            ticker_yahoo AS "Yahoo Symbol",
            kite_sector AS Sector,
            UPPER(TRIM(ticker)) AS ticker,
            sector_group,
            macro_sector,
            market_cap_segment  as market_cap,
            market_cap_cr,
            fii_hold_pct,
            public_hold_pct,
            2 AS priority
        FROM delta_tickers
    )
    SELECT
        Name,
        "Yahoo Symbol",
        Sector,
        ticker,
        sector_group,
        macro_sector,
        market_cap,
        market_cap_cr,
        fii_hold_pct,
        public_hold_pct
    FROM combined
    WHERE ticker IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY ticker
        ORDER BY priority ASC, market_cap_cr DESC NULLS LAST
    ) = 1
    ORDER BY ticker DESC
) TO 'tickers_updated.csv' (HEADER, DELIMITER ',');
