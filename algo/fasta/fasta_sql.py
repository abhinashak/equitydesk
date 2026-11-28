from datetime import date


NIFTY_TABLE = "benchmark_prices"
NIFTY_TICKER = "NIFTYBEES.NS"


def build_nifty_sql(cutoff_date: str | date) -> str:
    """Build the SQL query for weekly returns, prior-week momentum rank, and L/P buckets."""
    return f"""
        WITH nifty_closes AS (
            SELECT
                date_trunc('week', date) AS w_date,
                arg_max(close, date) AS close_price
            FROM {NIFTY_TABLE}
            WHERE ticker = '{NIFTY_TICKER}'
              AND date <= DATE '{cutoff_date}'
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
            arg_min(open, date) AS open_price,
            arg_max(close, date) AS close_price,
            min(low) AS low_price,
            avg(volume) AS avg_volume
        FROM ticker_prices
        WHERE date <= DATE '{cutoff_date}'
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


def build_daily_stop_max_date_sql(with_cutoff: bool = True) -> str:
    """Latest available daily price date, optionally bounded by a cutoff."""
    if with_cutoff:
        return "SELECT max(date) FROM ticker_prices WHERE CAST(date AS DATE) <= ?"
    return "SELECT max(date) FROM ticker_prices"


def build_daily_stop_data_sql() -> str:
    """Daily OHLC fields needed by the gap-aware stop-fill model."""
    return """
        SELECT ticker, CAST(date AS DATE) AS date, open, low
        FROM ticker_prices
        WHERE open IS NOT NULL AND low IS NOT NULL
          AND CAST(date AS DATE) <= ?
        ORDER BY ticker, date
    """


def build_stop_check_sql() -> str:
    """Daily prices for the interactive single-ticker stop-loss explanation."""
    return """
        SELECT CAST(date AS DATE) AS date, open, low, close
        FROM ticker_prices
        WHERE ticker = ?
          AND CAST(date AS DATE) >= ?
          AND CAST(date AS DATE) < ?
        ORDER BY date
    """


def build_market_cap_options_sql() -> str:
    return """
        SELECT DISTINCT market_cap
        FROM tickers
        WHERE market_cap IS NOT NULL
        ORDER BY market_cap
    """


def build_universe_symbols_sql(
        market_caps=(),
        min_market_cap_amt=None,
        min_fii_hold_pct=None,
        min_public_hold_pct=None,
):
    """Return parameterized SQL and parameters for the configured ticker universe."""
    where = []
    params = []
    if market_caps:
        where.append(f"market_cap IN ({', '.join('?' for _ in market_caps)})")
        params.extend(market_caps)
    for column, value in (
            ("market_cap_amt", min_market_cap_amt),
            ("fii_hold_pct", min_fii_hold_pct),
            ("public_hold_pct", min_public_hold_pct),
    ):
        if value is not None:
            where.append(f"{column} > ?")
            params.append(float(value))

    sql = "SELECT DISTINCT upper(nse_symbol) FROM tickers WHERE nse_symbol IS NOT NULL"
    if where:
        sql += " AND " + " AND ".join(where)
    return sql, params



def build_describe_ticker_prices_sql() -> str:
    return "DESCRIBE ticker_prices"


# ---------------------------------------------------------------------------
# Market-data access and caching (FASTA data layer)
# ---------------------------------------------------------------------------
from pathlib import Path
import pandas as pd


def _cache_paths(cache_dir, p):
    tag = f"{p.start_date}_{p.cutoff_date}"
    root = Path(cache_dir)
    return (
        root / f"cache_market_df_pit_v3_{tag}.parquet",
        root / f"cache_nifty_df_{p.cutoff_date}.parquet",
        root / f"cache_daily_stop_df_{p.cutoff_date}.parquet",
    )


def compound_pct(returns_pct):
    """Compound a series of percentage returns."""
    return ((1.0 + pd.Series(returns_pct, dtype=float) / 100.0).prod() - 1.0) * 100.0


def load_market_data(con, p, log, cache_dir):
    market_cache, _, _ = _cache_paths(cache_dir, p)
    if market_cache.exists():
        try:
            cached_df = pd.read_parquet(market_cache)
            if {'open_price', 'first_date', 'last_date'}.issubset(cached_df.columns):
                log("[Cache Hit] Loaded market data from parquet cache.")
                return cached_df
            log("[Cache Notice] Stale cache lacking required columns. Rebuilding…")
            market_cache.unlink()
        except Exception:
            pass

    log("[Cache Miss] Executing DuckDB query for market data…")
    df = con.execute(
        build_weekly_momentum_sql(start_date=p.start_date, cutoff_date=p.cutoff_date)
    ).fetchdf()
    df.columns = [c.lower() for c in df.columns]
    if 'open_price' not in df.columns:
        raise KeyError("Weekly SQL must return open_price (first trading day's actual open).")
    market_cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(market_cache, index=False)
    return df


def load_nifty_benchmark(con, p, log, cache_dir):
    _, cache_path, _ = _cache_paths(cache_dir, p)
    if cache_path.exists():
        nifty_df = pd.read_parquet(cache_path)
    else:
        try:
            nifty_df = con.execute(build_nifty_sql(cutoff_date=p.cutoff_date)).fetchdf()
            if not nifty_df.empty:
                nifty_df.columns = [c.lower() for c in nifty_df.columns]
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                nifty_df.to_parquet(cache_path, index=False)
        except Exception as exc:
            log(f"[Nifty] benchmark unavailable ({exc}) - benchmark shown as 0.")
            return {}, {}, {}

    if not nifty_df.empty:
        nifty_df['w_date'] = pd.to_datetime(nifty_df['w_date'])
        weekly_ret = pd.Series(
            nifty_df.nifty_ret.values, index=nifty_df['w_date'].dt.to_period('W')
        ).to_dict()
        q_sum = nifty_df.groupby(nifty_df['w_date'].dt.to_period('Q'))['nifty_ret'].apply(compound_pct).to_dict()
        y_sum = nifty_df.groupby(nifty_df['w_date'].dt.to_period('Y'))['nifty_ret'].apply(compound_pct).to_dict()
        return weekly_ret, q_sum, y_sum
    return {}, {}, {}


def load_daily_stop_data(con, p, log, cache_dir):
    """Load daily open/low arrays used by the GTT trigger simulation."""
    _, _, daily_cache = _cache_paths(cache_dir, p)
    try:
        cols = {r[0].lower() for r in con.execute(build_describe_ticker_prices_sql()).fetchall()}
        if not {'ticker', 'date', 'open', 'low'}.issubset(cols):
            log("[GTT] ticker_prices has no daily open/low columns - using weekly fallback.")
            return None

        if p.cutoff_date:
            cutoff_day = pd.Timestamp(p.cutoff_date).normalize()
            max_avail = pd.to_datetime(
                con.execute(build_daily_stop_max_date_sql(with_cutoff=True), [cutoff_day.date()]).fetchone()[0]
            ).normalize()
        else:
            max_avail = pd.to_datetime(
                con.execute(build_daily_stop_max_date_sql(with_cutoff=False)).fetchone()[0]
            ).normalize()
            cutoff_day = max_avail

        daily = None
        if daily_cache.exists():
            cached = pd.read_parquet(daily_cache)
            if not cached.empty:
                c_dates = pd.to_datetime(cached['date']).dt.normalize()
                if pd.notna(max_avail) and c_dates.max() == max_avail and c_dates.max() <= cutoff_day:
                    daily = cached

        if daily is None:
            log("[Cache Miss] Loading daily open/low for GTT triggers…")
            daily = con.execute(build_daily_stop_data_sql(), [cutoff_day.date()]).fetchdf()
            daily.columns = [c.lower() for c in daily.columns]
            daily_cache.parent.mkdir(parents=True, exist_ok=True)
            daily.to_parquet(daily_cache, index=False)

        daily['date'] = pd.to_datetime(daily['date']).dt.normalize()
        return {
            ticker: (
                group['date'].values.astype('datetime64[ns]'),
                group['open'].to_numpy(dtype=float),
                group['low'].to_numpy(dtype=float),
            )
            for ticker, frame in daily.groupby('ticker', sort=False)
            for group in [frame.sort_values('date', kind='stable').reset_index(drop=True)]
        }
    except Exception as exc:
        log(f"[GTT] daily data unavailable ({exc}) - using weekly fallback.")
        return None


def load_market_cap_options(con):
    rows = con.execute(build_market_cap_options_sql()).fetchall()
    return [r[0] for r in rows]


def ticker_market_cap_map(con):
    """Load ticker classifications from the canonical tickers table."""
    rows = con.execute(
        "SELECT upper(nse_symbol), market_cap FROM tickers WHERE nse_symbol IS NOT NULL"
    ).fetchall()
    return {str(symbol).strip().upper(): market_cap for symbol, market_cap in rows}


def universe_symbols(con, p):
    sql, args = build_universe_symbols_sql(
        market_caps=p.market_caps,
        min_market_cap_amt=p.min_market_cap_amt,
        min_fii_hold_pct=p.min_fii_hold_pct,
        min_public_hold_pct=p.min_public_hold_pct,
    )
    return {r[0] for r in con.execute(sql, args).fetchall()}


def load_stop_check_data(con, ticker, week_start):
    """Fetch one ticker's daily OHLC rows for the stop/GTT diagnostic."""
    start = pd.Timestamp(week_start).normalize()
    return con.execute(
        build_stop_check_sql(),
        [ticker, start.date(), (start + pd.Timedelta(days=7)).date()],
    ).fetchdf()
