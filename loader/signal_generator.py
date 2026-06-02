import yfinance as yf
import pandas as pd
import numpy as np
import sys
import argparse
from datetime import datetime, timedelta
import json
import duckdb
import os

# -----------------------------
# CONFIG
# -----------------------------
TIMEFRAMES = {
    "1Y":252, "6M":126, "3M":63, "2M":42,
    "1M":21, "14D":14, "7D":7, "5D":5,
    "4D":4, "3D":3, "2D":2, "1D":1
}

BENCHMARK_CSV   = "config/benchmarks.csv"
TICKERS_CSV     = "config/tickers.csv"

# Parquet paths
DATA_TICKER_GLOB    = "./data/ticker/**/*.parquet"
DATA_BENCHMARK_GLOB = "./data/benchmark/**/*.parquet"
DATA_TICKER_LATEST  = "./data/ticker/**/*.parquet"   # used for DISTINCT ticker scan
OUTPUT_ROOT         = "./data/signal_momentum"

# The reference index used for RS / market-trend calculations
INDEX_TICKER = "^NSEI"

# How many extra calendar days to pull before anchor_date
LOOKBACK_DAYS = 1100

# Offset applied when deriving anchor_date from the earliest parquet date
ANCHOR_OFFSET_DAYS = 1111


def load_benchmark_name_map(csv_path: str = BENCHMARK_CSV) -> dict:
    """
    Read config/benchmarks.csv -> {Yahoo Symbol: Name}.
    E.g.  "^NSEI" -> "NIFTY_50",  "NIFTYBEES.NS" -> "NIFTYBEES"
    Falls back to empty dict if the file is missing.
    """
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        mapping = dict(zip(df["Yahoo Symbol"].str.strip(), df["Name"].str.strip()))
        print(f"INFO  Loaded {len(mapping)} benchmark symbols from {csv_path}")
        return mapping
    except Exception as e:
        print(f"WARN  Could not load benchmark map from {csv_path}: {e}")
        return {}


# Populated at runtime inside __main__
BENCHMARK_NAME_MAP = {}

# -----------------------------
# INDICATORS
# -----------------------------
def compute_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)

    if len(gain) < period:
        return np.nan

    avg_gain = np.mean(gain[-period:])
    avg_loss = np.mean(loss[-period:])

    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def ulcer_index(prices, window=14):
    if len(prices) < window:
        return np.nan
    window_prices = prices[-window:]
    max_price = np.max(window_prices)
    drawdowns = ((window_prices - max_price) / max_price) * 100
    return np.sqrt(np.mean(drawdowns**2))


def kalman_filter(prices):
    n = len(prices)
    x = np.zeros(n)
    p = np.zeros(n)

    x[0] = prices[0]
    p[0] = 1

    Q = 0.0001
    R = 0.01

    for i in range(1, n):
        x_pred = x[i-1]
        p_pred = p[i-1] + Q

        K = p_pred / (p_pred + R)
        x[i] = x_pred + K * (prices[i] - x_pred)
        p[i] = (1 - K) * p_pred

    return x

def fetch_ticker_data(ticker: str, start_date: str = '2019-05-07', is_benchmark: bool = False):
    """
    Fetch OHLCV for ticker.
    - Equity tickers    -> read from DATA_TICKER_GLOB   (no index join needed;
                          rs / market_trend computed later in the SQL final join)
    - Benchmark tickers -> read from DATA_BENCHMARK_GLOB
    """
    source_glob = DATA_BENCHMARK_GLOB if is_benchmark else DATA_TICKER_GLOB
    con = duckdb.connect(database=':memory:')
    query = f"""
    SELECT *
    FROM read_parquet('{source_glob}', hive_partitioning = true)
    WHERE Ticker = '{ticker}'
      AND "Date" >= DATE '{start_date}'
    ORDER BY "Date" ASC
    """
    print(query)
    return con.execute(query).df()

# -----------------------------
# MAIN FEATURE ENGINE
# -----------------------------
def generate_features(ticker, start="2020-01-01", is_benchmark=False):
    print(f"Querying {ticker} (benchmark={is_benchmark})...")
    df = fetch_ticker_data(ticker, start, is_benchmark)

    # Standardize column names to lowercase for consistency
    df.columns = [c.lower() for c in df.columns]

    # We need 'open', 'close', and 'volume'
    df = df.dropna()

    closes = df['close'].values
    vols   = df['volume'].values
    opens  = df['open'].values
    dates  = df['date'].values
    highs  = df['high'].values
    lows   = df['low'].values

    kalman = kalman_filter(closes)

    # =============================
    # GLOBAL INDICATORS
    # =============================
    ema12 = pd.Series(closes).ewm(span=12).mean().values
    ema26 = pd.Series(closes).ewm(span=26).mean().values
    macd = ema12 - ema26
    macd_signal = pd.Series(macd).ewm(span=9).mean().values
    macd_hist = macd - macd_signal

    tr = np.maximum.reduce([
        highs - lows,
        abs(highs - np.roll(closes, 1)),
        abs(lows - np.roll(closes, 1))
    ])

    atr = pd.Series(tr).rolling(14).mean().values

    up_move = np.diff(highs, prepend=highs[0])
    down_move = -np.diff(lows, prepend=lows[0])

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / (pd.Series(tr).rolling(14).mean() + 1e-9)
    minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / (pd.Series(tr).rolling(14).mean() + 1e-9)

    adx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)).rolling(14).mean() * 100
    adx = adx.values

    bb_mid = pd.Series(closes).rolling(20).mean().values
    bb_std = pd.Series(closes).rolling(20).std().values
    bb_width = (2 * bb_std) / (bb_mid + 1e-9)

    vwap = (df['close'] * df['volume']).cumsum() / (df['volume'].cumsum() + 1e-9)
    vwap = vwap.values

    min_hist = max(TIMEFRAMES.values()) + 200
    print(f"DEBUG: Processing {ticker}")
    print(f"DEBUG: Total rows in df: {len(df)}")
    print(f"DEBUG: Starting range at: {min_hist}")

    if len(df) <= min_hist:
        print(f"⚠️ Skipping {ticker}: Not enough data. Need > {min_hist}, got {len(df)}")
        return pd.DataFrame() # Return empty to avoid downstream KeyErrors[cite: 1]

    # Now run the loop
    rows = []
    for i in range(min_hist, len(df)):
        row = {"date": dates[i], "close": closes[i]}

        # MOMENTUM
        for tf, back in TIMEFRAMES.items():
            row[tf] = (closes[i] - closes[i-back]) / closes[i-back] * 100

        # MOVING AVERAGES
        sma20  = np.mean(closes[i-20:i])
        sma50  = np.mean(closes[i-50:i])
        sma200 = np.mean(closes[i-200:i])

        row['sma20_dist']  = (closes[i] - sma20) / sma20
        row['sma50_dist']  = (closes[i] - sma50) / sma50
        row['sma200_dist'] = (closes[i] - sma200) / sma200
        row['sma50']       = sma50    # raw values kept for regime detection
        row['sma200']      = sma200
        row['trend_stack'] = int(sma20 > sma50 > sma200)

        # VOLUME
        vol_avg20 = np.mean(vols[i-20:i])
        vol_avg5  = np.mean(vols[i-5:i])
        row['vol_ratio']   = vols[i] / (vol_avg20 + 1e-9)
        row['volume_trend'] = vol_avg5 / (vol_avg20 + 1e-9)

        # --- Trend Consistency (The missing column) ---
        ret20 = np.diff(closes[i-20:i+1])
        row['trend_consistency'] = np.sum(ret20 > 0) / len(ret20)

        high_52w = np.max(closes[i-252:i])
        row['dist_52w_high'] = (closes[i] - high_52w) / (high_52w + 1e-9)

        # VOLATILITY
        returns = np.diff(closes[i-10:i+1]) / (closes[i-10:i] + 1e-9)
        row['volatility'] = np.std(returns)

        # ULCER & RSI - RISKS and DRAWDOWNS
        row['ulcer'] = ulcer_index(closes[i-14:i+1], 14)
        high_50 = np.max(closes[i-50:i+1])
        row['drawdown'] = (closes[i] - high_50) / (high_50 + 1e-9)
        row['rsi'] = compute_rsi(closes[:i+1])
        high20 = np.max(closes[i-20:i])
        row['breakout_strength'] = (closes[i] - high20) / (high20 + 1e-9)

        # GAP
        row['gap'] = (opens[i] - closes[i-1]) / (closes[i-1] + 1e-9)

        row['acceleration'] = row.get('1M', 0) - row.get('3M', 0)
        row['volume_price_signal'] = row['vol_ratio'] * np.sign(row['acceleration'])
        row['kalman_dist'] = (closes[i] - kalman[i]) / kalman[i]

        short_vol = np.std(np.diff(closes[i-5:i+1]))
        long_vol  = np.std(np.diff(closes[i-20:i+1]))
        row['vol_compression'] = short_vol / (long_vol + 1e-9)

        # MACD
        row['macd'] = macd[i]
        row['macd_bullish'] = int(macd[i] > macd_signal[i])

        # ATR
        row['atr'] = atr[i]
        row['atr_expansion'] = int(atr[i] > np.mean(atr[i-20:i]))

        # ADX
        row['adx'] = adx[i]
        row['strong_trend'] = int(adx[i] > 25)

        # Bollinger
        row['bb_width'] = bb_width[i]
        row['bb_squeeze'] = int(bb_width[i] < np.mean(bb_width[i-50:i]))

        # VWAP
        row['vwap_dist'] = (closes[i] - vwap[i]) / vwap[i]

        # Momentum quality
        row['momentum_quality'] = row['1M'] / (row['volatility'] + 1e-9)

        rows.append(row)

    # Convert to DataFrame AFTER the loop
    feature_df = pd.DataFrame(rows)

    # -----------------------------
    # POST-PROCESSING (Trend Flags)
    # -----------------------------
    if not feature_df.empty:
        feature_df['trend_flag'] = (feature_df['trend_stack'] == 1).astype(int)
        feature_df['days_in_trend'] = feature_df['trend_flag'].groupby(
            (feature_df['trend_flag'] == 0).cumsum()
        ).cumsum()
        feature_df['days_in_trend'] = np.log1p(feature_df['days_in_trend'])
        feature_df = feature_df.drop(columns=['trend_flag'])
        feature_df['momentum_decay'] = feature_df['1M'] - feature_df['5D']


    return feature_df

def create_slabs(df):

    print ( df.head(10) )
    # Price vs SMAs
    df['sma20_low']   = (df['sma20_dist'] < 0).astype(float)
    df['sma20_mid']   = (df['sma20_dist'].between(0, 0.03)).astype(float)
    df['sma20_high']  = (df['sma20_dist'] >= 0.03).astype(float)
    df['sma50_pos']   = (df['sma50_dist'] > 0).astype(float)
    df['sma200_pos']  = (df['sma200_dist'] > 0).astype(float)

    # Trend & Volume
    df['trend_stack'] = (df['trend_stack'] == 1).astype(float)
    df['vol_low']     = (df['vol_ratio'] < 1.0).astype(float)
    df['vol_mid']     = (df['vol_ratio'].between(1.0, 1.5)).astype(float)
    df['vol_high']    = (df['vol_ratio'] >= 1.5).astype(float)
    df['volume_trend_up'] = (df['volume_trend'] > 1.0).astype(float)

    # Volatility
    df['volatility_low']  = (df['volatility'] < 0.03).astype(float)
    df['volatility_mid']  = (df['volatility'].between(0.03, 0.06)).astype(float)
    df['volatility_high'] = (df['volatility'] >= 0.06).astype(float)
    df['compression']     = (df['vol_compression'] < 1.0).astype(float)

    # RSI & Risk
    df['rsi_low']         = (df['rsi'] < 50).astype(float)
    df['rsi_mid']         = (df['rsi'].between(50, 70)).astype(float)
    df['rsi_high']        = (df['rsi'] >= 70).astype(float)
    df['ulcer_low']       = (df['ulcer'] < 8).astype(float)
    df['ulcer_high']      = (df['ulcer'] >= 15).astype(float)
    df['drawdown_shallow'] = (df['drawdown'] > -0.05).astype(float)

    # Trend Quality & Indicators
    df['trend_consistent'] = (df['trend_consistency'] > 0.6).astype(float)
    df['breakout']         = (df['breakout_strength'] >= 0).astype(float)
    df['acceleration']     = (df['acceleration'] > 0).astype(float)
    df['kalman_positive']  = (df['kalman_dist'] > 0).astype(float)

    # RS & Market
    df['gap_up']           = (df['gap'] > 0.005).astype(float)

    # 52W High & Maturity
    df['near_52w_high']    = (df['dist_52w_high'] > -0.05).astype(float)
    df['at_52w_high']      = (df['dist_52w_high'] >= -0.01).astype(float)
    df['trend_fresh']      = (df['days_in_trend'].between(1, 21)).astype(float)
    df['trend_mature']     = (df['days_in_trend'] > 63).astype(float)
    df['momentum_slowing'] = (df['momentum_decay'] < 0).astype(float)

    # Momentum Alignment
    df['vps_bullish']      = (df['volume_price_signal'] > 1.5).astype(float)
    df['vps_bearish']      = (df['volume_price_signal'] < -1.5).astype(float)
    df['momentum_accelerating'] = (df['acceleration'] > 0).astype(float)
    df['momentum_aligned'] = ((df['1M'] > df['3M']) & (df['3M'] > df['6M'])).astype(float)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 2 : Sector momentum helpers
# ─────────────────────────────────────────────────────────────────────────────

# Columns aggregated (mean) for both sector-level and market-wide (NSEI) views.
# sector_* prefix  -> mean across tickers sharing the same sector on that date
# nsei_*   prefix  -> mean across ALL equity tickers on that date
AGG_COLS = [
    "sma200_dist",  "sma200_pos",
    "dist_52w_high","near_52w_high",
    "momentum_accelerating",
    "ulcer",        "atr",
    "bb_width",     "bb_squeeze",
    "drawdown_shallow",
    "vwap_dist",
]


# ─────────────────────────────────────────────────────────────────────────────
# SAVE HELPER
# ─────────────────────────────────────────────────────────────────────────────

def save_by_year(df: pd.DataFrame, output_root: str, filename: str) -> None:
    """Partition df by 'year' column and write one parquet per year."""
    for year, chunk in df.groupby("year"):
        year_dir = os.path.join(output_root, f"year={year}")
        os.makedirs(year_dir, exist_ok=True)
        out_file = os.path.join(year_dir, filename)
        chunk.to_parquet(out_file, index=False)
        print(f"✅ Wrote {len(chunk)} rows -> {out_file}")


def get_sector_mapping(csv_path: str = TICKERS_CSV):
    """
    Load ticker -> sector mapping from config/tickers.csv.
    Columns expected: Name, Yahoo Symbol, Sector
    Returns DataFrame with [ticker, sector]; Sector values equal benchmark Names
    (e.g. "NIFTY_Pharma_Index") from config/benchmarks.csv.
    """
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        mapping = (
            df[["Yahoo Symbol", "Sector"]]
            .rename(columns={"Yahoo Symbol": "ticker", "Sector": "sector"})
            .dropna(subset=["sector"])
            .drop_duplicates()
        )
        print(f"INFO  Loaded sector mapping for {len(mapping)} tickers from {csv_path}")
        return mapping
    except Exception as e:
        print(f"WARN  Could not load sector mapping from {csv_path}: {e}")
        return pd.DataFrame(columns=["ticker", "sector"])




# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Momentum Feature Generator")
    parser.add_argument("--start", type=str, default=None, help="Optional start date (YYYY-MM-DD)")

    args = parser.parse_args()

    # Load dynamic benchmark map from CSV
    BENCHMARK_NAME_MAP = load_benchmark_name_map()

    con = duckdb.connect(database=':memory:')

    if args.start:
        anchor_date = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        query = f"""
            SELECT min(date) + {ANCHOR_OFFSET_DAYS} as start_date_derived
            FROM read_parquet('{DATA_TICKER_GLOB}', hive_partitioning = true)
            WHERE ticker != '{INDEX_TICKER}'
        """
        result = con.execute(query).fetchone()[0]
        anchor_date = pd.to_datetime(result)

    print(f"Computed Anchor Date: {anchor_date.strftime('%Y-%m-%d')}")
    fetch_start = (anchor_date - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    print(f"Fetch start date: {fetch_start}")

    # Regular equity tickers from data_ticker parquet
    all_tickers_query = f"""
        SELECT DISTINCT Ticker AS ticker
        FROM read_parquet('{DATA_TICKER_GLOB}', hive_partitioning = true)
    """
    regular_tickers = con.execute(all_tickers_query).df()['ticker'].tolist()

    # Benchmark tickers from config/benchmarks.csv (already loaded)
    benchmark_tickers = list(BENCHMARK_NAME_MAP.keys())

    print(f"Found {len(regular_tickers)} equity tickers and {len(benchmark_tickers)} benchmarks.")
    print(f"  Benchmarks : {benchmark_tickers}")

    # ── Process & save benchmark momentum ───────────────────────────────────
    benchmark_features = []
    for ticker in benchmark_tickers:
        print(f"\n--- [BENCHMARK] Processing {ticker} ---")
        feature_df = generate_features(ticker, start=fetch_start, is_benchmark=True)
        if feature_df.empty:
            print(f"⚠️ Skipping {ticker}: Not enough data.")
            continue
        feature_df['year'] = feature_df['date'].dt.year
        X = create_slabs(feature_df)
        X['ticker'] = ticker
        X['name']   = BENCHMARK_NAME_MAP.get(ticker, ticker.lstrip("^"))
        benchmark_features.append(X)

    if benchmark_features:
        bench_df = pd.concat(benchmark_features, ignore_index=True)
        save_by_year(bench_df, OUTPUT_ROOT, "benchmark_momentum.parquet")

    # ── Regular equity loop ─────────────────────────────────────────────────
    sector_map  = get_sector_mapping()          # reads config/tickers.csv
    all_features = []
    for ticker in regular_tickers:
        print(f"\n--- Processing {ticker} ---")
        feature_df = generate_features(ticker, start=fetch_start)
        if feature_df.empty:
            print(f"⚠️ Skipping {ticker}: Not enough data to calculate features.")
            continue
        feature_df['year'] = feature_df['date'].dt.year
        X = create_slabs(feature_df)
        X['ticker'] = ticker
        # Attach sector label so the DuckDB join can use it
        X = X.merge(sector_map, on="ticker", how="left")
        all_features.append(X)

    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        save_by_year(final_df, OUTPUT_ROOT, "ticker_momentum.parquet")

        # ── Build ticker_momentum_normalized via DuckDB join ─────────────────
        # Join ticker_momentum with benchmark_momentum on (sector = name, date)
        # to bring every benchmark's AGG_COLS across as sector_* columns,
        # then aggregate all benchmarks market-wide as nsei_* columns.
        norm_query = f"""
        WITH ticker AS (
            SELECT *
            FROM read_parquet('{OUTPUT_ROOT}/**/ticker_momentum.parquet',
                              hive_partitioning = true)
        ),
        bench AS (
            SELECT *
            FROM read_parquet('{OUTPUT_ROOT}/**/benchmark_momentum.parquet',
                              hive_partitioning = true)
        ),
        -- ^NSEI close per date for rs / market_trend calculations
        nsei_close AS (
            SELECT date, close AS nsei_close, sma200_dist
            FROM bench
            WHERE  ticker = '{INDEX_TICKER}'
        ),
        -- sector-level: join each ticker to its sector benchmark by (name, date)
        sector_joined AS (
            SELECT
                t.*,
                {", ".join(f"b.{c} AS sector_{c}" for c in AGG_COLS)}
            FROM ticker t
            LEFT JOIN bench b
                ON  t.sector = b.name
                AND t.date   = b.date
        ),
        -- market-wide (NSEI): mean of AGG_COLS across all benchmarks per date
        nsei_agg AS (
            SELECT
                date,
                {", ".join(f"AVG({c}) AS nsei_{c}" for c in AGG_COLS)}
            FROM bench
            GROUP BY date
        ),
        -- compute rs, rs_momentum, market_trend from NSEI close
        with_rs AS (
            SELECT
                s.*,
                {", ".join(f"n.nsei_{c}" for c in AGG_COLS)},
                s.close / nc.nsei_close                                      AS rs,
                (s.close / nc.nsei_close)
                    / LAG(s.close / nc.nsei_close, 20)
                        OVER (PARTITION BY s.ticker ORDER BY s.date) - 1     AS rs_momentum,
                CASE WHEN nc.sma200_dist > 0 THEN 1 ELSE 0 END               AS market_trend
            FROM sector_joined s
            LEFT JOIN nsei_agg  n  ON s.date = n.date
            LEFT JOIN nsei_close nc ON s.date = nc.date
        )
        SELECT
            *,
            CASE WHEN rs_momentum >    0 THEN 1 ELSE 0 END  AS rs_improving,
            CASE WHEN rs_momentum > 0.05 THEN 1 ELSE 0 END  AS rs_strong
        FROM with_rs
        """

        print ( norm_query )

        print("ℹ️  Building ticker_momentum_normalized via DuckDB join...")
        norm_con = duckdb.connect(database=':memory:')
        norm_df  = norm_con.execute(norm_query).df()
        norm_df['year'] = pd.to_datetime(norm_df['date']).dt.year
        save_by_year(norm_df, OUTPUT_ROOT, "ticker_momentum_normalized.parquet")
        print(f"✅ ticker_momentum_normalized written ({len(norm_df)} rows).")