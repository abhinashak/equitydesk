Git a"""
Exit-Timing Analysis from Past FASTA WEEKEND_EXIT Trades
========================================================
Companion to entry_timing.py (same architecture, parameters, verbosity levels, Past/Recent split,
confidence score, sector-level selection and FASTA cost model).

Given tickers and the CSV of all back-test trades, find which EXIT TIME would have produced the best
NET PROFIT AFTER TRANSACTION COSTS on trades that were closed by the weekend rule
(`exit_reason == 'WEEKEND_EXIT'` only; stop-loss exits are excluded).

The objective is average net return per trade after costs (also reported in rupees).  Win rate is shown
for information but is never used to choose an exit.

Exit modes compared on every historical weekend exit (daily bars, FASTA cost model).  The entry price is
the historical `buy_price` from the CSV, so only the exit varies:
    fri_open    sell at the exit day's (Friday's) open
    fri_twap    sell at OHLC-average proxy of Friday, 7 slices
    fri_close   sell at Friday's close                       (what the CSV trades assume)
    mon_open    hold over the weekend, sell at the next session's (Monday's) open
    mon_twap    hold over the weekend, sell at OHLC-average proxy of Monday, 7 slices
    mon_close   hold over the weekend, sell at Monday's close
Holding past Friday keeps the stop-loss alive (stop = buy_price * (1 + stop_loss%)): a Monday gap through
the stop fills at Monday's open, an intraday touch fills at the stop price.  The baseline for all "edge"
numbers is the mode that best reproduces the CSV's own sell prices (normally fri_close).

Movement features
    Friday-known (decision taken on the exit day, same as the entry study; tradable for every mode):
        <x>_gap   exit-day open vs previous close, in %
        <x>_prev  previous session's close-to-close return, in %
    MONDAY-MORNING movement (new):
        mon_<x>_gap   Monday's open vs Friday's close, in %  (the weekend move + opening reaction)
  for x in: stk (the stock), nifty (^NSEI), and the equal-weighted average of peers in the stock's
  ind (industry) / sg (sector_group) / ms (macro_sector).
  Monday-morning features are only known AFTER the weekend, so a rule built on them is a "hold, then
  decide on Monday morning" policy: it may only choose between the Monday exits (mon_open / mon_twap /
  mon_close), never back-fill a Friday exit.  It is still scored against the best fixed mode (which may
  be a Friday exit), so committing to the hold has to pay for itself.

Statistics
    1. overall comparison of the exit modes (paired vs baseline)
    2. single-factor breakdown: for LOW / MID / HIGH terciles of each feature, which mode wins
    3. Past vs Recent: trades split chronologically (default: newest 30% = Recent); tests whether the
       stock's Friday / weekend / Monday behaviour and the value of each exit mode have SHIFTED.
    4. A rule learned on Past only, scored on Recent, plus a 0-100 confidence score for the recommendation.

Usage:
    from exit_timing import analyze_exit_timing
    df = analyze_exit_timing("RELIANCE,TCS", trades="all_trades.csv", verbose=1)
    df = analyze_exit_timing(["RELIANCE", "TCS.NS"], trades="all_trades.csv")

    python exit_timing.py RELIANCE,TCS --trades all_trades.csv --verbose 2
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from typing import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from utils.db_utils import db_cursor, initialize_database
except ImportError:
    from db_utils import db_cursor, initialize_database  # type: ignore


FRI_MODES = ("fri_open", "fri_twap", "fri_close")
MON_MODES = ("mon_open", "mon_twap", "mon_close")
MODES = FRI_MODES + MON_MODES
N_SLICES = {"fri_open": 1, "fri_twap": 7, "fri_close": 1, "mon_open": 1, "mon_twap": 7, "mon_close": 1}
MODE_DESC = {
    "fri_open": "sell Friday open", "fri_twap": "sell Friday TWAP", "fri_close": "sell Friday close",
    "mon_open": "hold, sell Monday open", "mon_twap": "hold, sell Monday TWAP",
    "mon_close": "hold, sell Monday close",
}
DEFAULT_BASE = "fri_close"
MON_DEFAULT = "mon_open"     # default exit inside a Monday-morning rule
LABELS = ("low", "mid", "high")
EXIT_REASON = "WEEKEND_EXIT"

# Sector levels from the `tickers` table (column name -> short prefix).
LEVELS = {"industry": "ind", "sector_group": "sg", "macro_sector": "ms"}

FACTORS = {
    "stock gap": "stk_gap", "stock prev-day": "stk_prev",
    "NIFTY gap": "nifty_gap", "NIFTY prev-day": "nifty_prev",
    "industry gap": "ind_gap", "industry prev-day": "ind_prev",
    "sector_group gap": "sg_gap", "sector_group prev-day": "sg_prev",
    "macro_sector gap": "ms_gap", "macro_sector prev-day": "ms_prev",
    # Monday-morning movement (open vs Friday close) -> Monday-only exit choices
    "Mon stock gap": "mon_stk_gap", "Mon NIFTY gap": "mon_nifty_gap",
    "Mon industry gap": "mon_ind_gap", "Mon sector_group gap": "mon_sg_gap",
    "Mon macro_sector gap": "mon_ms_gap",
}
MON_GAP_COLS = ("stk_gap", "nifty_gap", "ind_gap", "sg_gap", "ms_gap")


# --------------------------------------------------------------------------------------
# FASTA cost model and price loader
# --------------------------------------------------------------------------------------
def net_trade_return_pct(
        buy_px: float,
        sell_px: float,
        capital: float = 100_000.0,
        n_slices: int = 1,
        n_sell_slices: int = 1,
        brokerage_rate: float = 0.0,
        brokerage_cap: float = 0.0,
        stt_buy_rate: float = 0.001,
        stt_sell_rate: float = 0.001,
        exchange_txn_rate: float = 0.0000325,
        sebi_turnover_rate: float = 0.000001,
        stamp_duty_buy_rate: float = 0.00015,
        gst_rate: float = 0.18,
        dp_charge_per_sell: float = 18.80,
) -> float:
    """Net % return on allocated capital after FASTA-style transaction costs.

    Identical to the entry study's cost model; `n_sell_slices` spreads the SELL order (TWAP exits) the
    same way `n_slices` spreads the buy order (only matters if a per-order brokerage cap is set)."""
    if not (np.isfinite(buy_px) and np.isfinite(sell_px) and buy_px > 0 and capital > 0):
        return np.nan

    qty = np.floor(capital / buy_px)
    if qty <= 0:
        return 0.0

    buy_val = qty * buy_px
    sell_val = qty * sell_px

    slice_val = buy_val / max(n_slices, 1)
    buy_brok_per_slice = slice_val * brokerage_rate
    if brokerage_cap > 0:
        buy_brok_per_slice = min(buy_brok_per_slice, brokerage_cap)
    buy_brok = buy_brok_per_slice * max(n_slices, 1)

    sell_slice_val = sell_val / max(n_sell_slices, 1)
    sell_brok_per_slice = sell_slice_val * brokerage_rate
    if brokerage_cap > 0:
        sell_brok_per_slice = min(sell_brok_per_slice, brokerage_cap)
    sell_brok = sell_brok_per_slice * max(n_sell_slices, 1)

    stt = buy_val * stt_buy_rate + sell_val * stt_sell_rate
    exch = (buy_val + sell_val) * exchange_txn_rate
    sebi = (buy_val + sell_val) * sebi_turnover_rate
    stamp = buy_val * stamp_duty_buy_rate
    gst = (buy_brok + sell_brok + exch + sebi) * gst_rate
    total_cost = buy_brok + sell_brok + stt + exch + sebi + stamp + gst + dp_charge_per_sell

    net_pnl = (sell_val - buy_val) - total_cost
    return (net_pnl / capital) * 100.0


def _fetch_series_from_views(cur, tickers: tuple[str, ...], candidate_tables: tuple[str, ...]) -> pd.DataFrame:
    """OHLCV rows for `tickers` from the first of `candidate_tables` that exists and has data."""
    existing = {
        r[0]
        for r in cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_catalog = current_database() AND table_schema = 'main'"
        ).fetchall()
    }
    placeholders = ", ".join("?" for _ in tickers)
    for tbl in candidate_tables:
        if tbl not in existing:
            continue
        sql = f"""
            SELECT
                UPPER(TRIM(COALESCE(nse_symbol, ticker))) AS sym,
                CAST(Date AS DATE) AS dt,
                CAST(Open AS DOUBLE) AS open,
                CAST(High AS DOUBLE) AS high,
                CAST(Low AS DOUBLE) AS low,
                CAST(Close AS DOUBLE) AS close,
                CAST(Volume AS DOUBLE) AS volume
            FROM {tbl}
            WHERE (UPPER(TRIM(Ticker)) IN ({placeholders})
                   OR UPPER(TRIM(COALESCE(nse_symbol, ''))) IN ({placeholders}))
              AND Open > 0 AND Close > 0
        """
        df = cur.execute(sql, [*tickers, *tickers]).fetchdf()
        if not df.empty:
            return df.sort_values("dt").drop_duplicates("dt").reset_index(drop=True)
    return pd.DataFrame(columns=["sym", "dt", "open", "high", "low", "close", "volume"])


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def _bare(ticker: str) -> str:
    t = str(ticker).strip().upper()
    return t.rsplit(".", 1)[0] if t.endswith((".NS", ".BO")) else t


def _variants(ticker: str) -> tuple[str, ...]:
    b = _bare(ticker)
    return tuple(dict.fromkeys([b, f"{b}.NS", str(ticker).strip().upper()]))


def load_trades(path: str, ticker: str, strategy: str | None) -> pd.DataFrame:
    """Weekend-exit trades only (`exit_reason == 'WEEKEND_EXIT'`) for one ticker."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    need = {"ticker", "buy_date", "sell_date", "exit_reason", "buy_price"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Trades CSV is missing columns: {sorted(missing)}")

    df = df[df["ticker"].map(_bare) == _bare(ticker)].copy()
    if strategy:
        df = df[df["strategy"] == strategy]
    if df.empty:
        raise ValueError(f"No trades for {ticker}" + (f" / strategy {strategy}" if strategy else "") + f" in {path}")

    n_all = len(df)
    df = df[df["exit_reason"].astype(str).str.strip().str.upper() == EXIT_REASON].copy()
    if df.empty:
        raise ValueError(f"No {EXIT_REASON} trades for {ticker} ({n_all} trades in other exit categories excluded).")

    df["buy_date"] = pd.to_datetime(df["buy_date"]).dt.normalize()
    df["sell_date"] = pd.to_datetime(df["sell_date"]).dt.normalize()
    if "position_capital" not in df.columns:
        df["position_capital"] = 100_000.0

    n0 = len(df)
    df = df.sort_values("buy_date").drop_duplicates("buy_date", keep="first").reset_index(drop=True)
    if len(df) < n0:
        print(f"note: {n0 - len(df)} duplicate buy_date rows (several strategies bought the same day) "
              f"collapsed to one trade per day.")
    return df


def _ticker_meta(cur, variants: tuple[str, ...]) -> dict:
    ph = ", ".join("?" for _ in variants)
    row = cur.execute(
        f'SELECT industry, sector_group, macro_sector FROM tickers '
        f'WHERE UPPER(TRIM(nse_symbol)) IN ({ph}) OR UPPER(TRIM("Yahoo Symbol")) IN ({ph}) LIMIT 1',
        [*variants, *variants],
    ).fetchone()
    if not row:
        return {}
    return {k: (str(v).strip() if v is not None and str(v).strip() else None)
            for k, v in zip(("industry", "sector_group", "macro_sector"), row)}


def _peer_movement(cur, level: str, value: str, exclude: tuple[str, ...]) -> pd.DataFrame:
    """Equal-weighted average gap / close-to-close return of all peers sharing `level == value`.

    Per-stock moves are clipped to +/-20% so one bad print or unadjusted split cannot swamp the average.
    """
    assert level in LEVELS  # column name is interpolated below, so keep it whitelisted
    ph = ", ".join("?" for _ in exclude)
    sql = f"""
        WITH peers AS (
            SELECT UPPER(TRIM("Yahoo Symbol")) AS ysym, UPPER(TRIM(nse_symbol)) AS nsym
            FROM tickers WHERE TRIM({level}) = ?
        ),
        px AS (
            SELECT UPPER(TRIM(COALESCE(p.nse_symbol, p.Ticker))) AS sym,
                   CAST(p.Date AS DATE) AS dt,
                   CAST(p.Open AS DOUBLE) AS o, CAST(p.Close AS DOUBLE) AS c
            FROM ticker_prices p
            WHERE (UPPER(TRIM(p.Ticker)) IN (SELECT ysym FROM peers)
                   OR UPPER(TRIM(COALESCE(p.nse_symbol, ''))) IN (SELECT nsym FROM peers))
              AND p.Open > 0 AND p.Close > 0
            QUALIFY ROW_NUMBER() OVER (PARTITION BY UPPER(TRIM(COALESCE(p.nse_symbol, p.Ticker))),
                                                    CAST(p.Date AS DATE) ORDER BY p.Date) = 1
        ),
        mv AS (
            SELECT sym, dt, o, c, LAG(c) OVER (PARTITION BY sym ORDER BY dt) AS pc FROM px
        ),
        r AS (
            SELECT dt,
                   LEAST(GREATEST((o / pc - 1) * 100, -20), 20) AS gap,
                   LEAST(GREATEST((c / pc - 1) * 100, -20), 20) AS ret1d
            FROM mv
            WHERE pc IS NOT NULL AND sym NOT IN ({ph})
        )
        SELECT dt, AVG(gap) AS gap, AVG(ret1d) AS ret1d, COUNT(*) AS n
        FROM r GROUP BY dt ORDER BY dt
    """
    df = cur.execute(sql, [value, *exclude]).fetchdf()
    df["dt"] = pd.to_datetime(df["dt"])
    df["prev"] = df["ret1d"].shift(1)  # previous session's average return
    return df.set_index("dt")[["gap", "prev", "n"]]


def fetch_market_data(variants, index_tickers, sql_path):
    if sql_path:
        con = initialize_database(sql_path)
        cur, ctx, close_root = con.cursor(), None, True
    else:
        ctx = db_cursor()
        cur, con, close_root = ctx.__enter__(), None, False
    try:
        stk = _fetch_series_from_views(cur, variants, ("ticker_prices", "ticker_prices_extended"))
        idx = _fetch_series_from_views(cur, index_tickers, ("ticker_prices", "benchmark_prices"))
        meta, peers = {}, {}
        try:
            meta = _ticker_meta(cur, variants)
            for level, value in meta.items():
                if value:
                    peers[level] = _peer_movement(cur, level, value, variants)
        except Exception as exc:  # tickers table missing / schema differs
            print(f"warning: sector features unavailable ({exc}); using stock + NIFTY only.")
    finally:
        if close_root:
            cur.close()
            con.close()
        else:
            ctx.__exit__(None, None, None)

    if stk.empty:
        raise ValueError(f"No price rows found for {variants}.")
    if idx.empty:
        raise ValueError(f"No price rows found for index {index_tickers}.")
    if not meta:
        print("warning: ticker not found in `tickers` table; sector features skipped.")
    return stk, idx, meta, peers


# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------
def _gap_prev(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    d = df.copy()
    d["dt"] = pd.to_datetime(d["dt"])
    d = d.sort_values("dt").set_index("dt")
    pc = d["close"].shift(1)
    out = pd.DataFrame(index=d.index)
    out[f"{prefix}_gap"] = (d["open"] / pc - 1.0) * 100.0
    out[f"{prefix}_prev"] = (pc / d["close"].shift(2) - 1.0) * 100.0
    return out


def build_features(stk, idx, peers, min_peers) -> pd.DataFrame:
    feats = _gap_prev(stk, "stk").join(_gap_prev(idx, "nifty"), how="left")
    for level, prefix in LEVELS.items():
        if level in peers:
            p = peers[level].rename(columns={"gap": f"{prefix}_gap", "prev": f"{prefix}_prev", "n": f"{prefix}_n"})
            feats = feats.join(p, how="left")
            thin = feats[f"{prefix}_n"].fillna(0) < min_peers
            feats.loc[thin, [f"{prefix}_gap", f"{prefix}_prev"]] = np.nan
        else:
            feats[f"{prefix}_gap"] = np.nan
            feats[f"{prefix}_prev"] = np.nan
            feats[f"{prefix}_n"] = np.nan
    return feats


def score_levels(feats: pd.DataFrame, meta: dict, tol: float = 0.02):
    """Rank sector levels by how closely their peer average tracks THIS stock (full price history,
    independent of trade P&L, so choosing a level this way adds no look-ahead to the exit tests).
    Score = mean of corr(stock gap, peer gap) and corr(stock prev-day ret, peer prev-day ret).
    A narrower level wins ties (within `tol`) because it is the more specific peer group."""
    rows = []
    for level, prefix in LEVELS.items():
        if not meta.get(level) or feats[f"{prefix}_gap"].notna().sum() < 30:
            continue
        g = feats[["stk_gap", f"{prefix}_gap"]].dropna()
        p = feats[["stk_prev", f"{prefix}_prev"]].dropna()
        cg, cp = g.corr().iloc[0, 1], p.corr().iloc[0, 1]
        rows.append({"level": level, "value": meta[level], "median_peers": feats[f"{prefix}_n"].median(),
                     "days": len(g), "corr_gap": cg, "corr_prev_day": cp, "score": np.nanmean([cg, cp])})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, None
    best = df["score"].max()
    chosen = next(lv for lv in LEVELS if lv in set(df["level"]) and df.loc[df["level"] == lv, "score"].iloc[0] >= best - tol)
    return df, chosen


def factors_for(level_choice: str | None) -> dict:
    """Stock + NIFTY factors always; sector factors only for the chosen level ('all' keeps every level).
    Applies to both the Friday-known and the Monday-morning factors."""
    if level_choice in (None, "all"):
        return dict(FACTORS)
    keep = ("stk_", "nifty_", LEVELS[level_choice] + "_",
            "mon_stk_", "mon_nifty_", "mon_" + LEVELS[level_choice] + "_")
    return {k: v for k, v in FACTORS.items() if v.startswith(keep)}


def is_monday_factor(col: str) -> bool:
    return col.startswith("mon_")


def eligible_modes(col: str) -> tuple[str, ...]:
    """A Monday-morning feature is unknown on Friday, so a rule on it may only pick Monday exits."""
    return MON_MODES if is_monday_factor(col) else MODES


def default_mode(col: str, base: str) -> str:
    return MON_DEFAULT if is_monday_factor(col) else base


# --------------------------------------------------------------------------------------
# Re-simulating each exit mode
# --------------------------------------------------------------------------------------
def simulate_exit(o, h, l, c, i_sell, mode, buy_px, stop_pct):
    """Returns (sell_px, stopped) or (nan, False) if the mode cannot be executed.

    fri_*: the trade already survived to the exit day under the CSV's own engine, so no stop is re-applied.
    mon_*: stop stays live on the next session: gap-through fills at the open, intraday touch at the stop."""
    if mode.startswith("fri_"):
        kind, i = mode[4:], i_sell
        stopped = False
    else:
        kind, i = mode[4:], i_sell + 1
        if i >= len(o):
            return np.nan, False
        stop = buy_px * (1.0 + stop_pct / 100.0)
        if o[i] <= stop:
            return o[i], True
        if kind != "open" and l[i] <= stop:
            return stop, True
        stopped = False

    if kind == "open":
        px = o[i]
    elif kind == "twap":
        px = (o[i] + h[i] + l[i] + c[i]) / 4.0
    else:
        px = c[i]
    return px, stopped


def simulate_all(trades, stk, feats, stop_pct) -> pd.DataFrame:
    stk = stk.copy()
    stk["dt"] = pd.to_datetime(stk["dt"])
    stk = stk.sort_values("dt").reset_index(drop=True)
    dates = stk["dt"].to_numpy(dtype="datetime64[ns]")
    o, h, l, c = (stk[k].to_numpy(float) for k in ("open", "high", "low", "close"))

    rows, skipped = [], 0
    for r in trades.itertuples(index=False):
        i_sell = int(np.searchsorted(dates, np.datetime64(r.sell_date), side="left"))
        if i_sell >= len(dates) or dates[i_sell] != np.datetime64(r.sell_date):
            skipped += 1
            continue
        buy_px = float(getattr(r, "buy_price", np.nan))
        if not np.isfinite(buy_px) or buy_px <= 0:
            i_buy = int(np.searchsorted(dates, np.datetime64(r.buy_date), side="left"))
            if i_buy >= len(dates) or dates[i_buy] != np.datetime64(r.buy_date):
                skipped += 1
                continue
            buy_px = o[i_buy]
        has_mon = i_sell + 1 < len(dates)
        capital = float(r.position_capital)

        row = {
            "buy_date": r.buy_date, "sell_date": r.sell_date, "buy_px": buy_px, "capital": capital,
            "mon_date": pd.Timestamp(dates[i_sell + 1]) if has_mon else pd.NaT,
            "exit_reason_csv": r.exit_reason,
            "csv_sell_price": getattr(r, "sell_price", np.nan),
            "csv_net_pnl_pct": getattr(r, "net_pnl_pct", np.nan),
            "fri_open": o[i_sell], "fri_high": h[i_sell], "fri_low": l[i_sell], "fri_close": c[i_sell],
            "mon_open": o[i_sell + 1] if has_mon else np.nan, "mon_high": h[i_sell + 1] if has_mon else np.nan,
            "mon_low": l[i_sell + 1] if has_mon else np.nan, "mon_close": c[i_sell + 1] if has_mon else np.nan,
        }
        for m in MODES:
            sell, stopped = simulate_exit(o, h, l, c, i_sell, m, buy_px, stop_pct)
            row[f"{m}_sell"], row[f"{m}_stopped"] = sell, stopped
            row[f"{m}_ret"] = net_trade_return_pct(buy_px, sell, capital=capital, n_sell_slices=N_SLICES[m])
            row[f"{m}_rs"] = row[f"{m}_ret"] / 100.0 * capital
        rows.append(row)

    if skipped:
        print(f"note: {skipped} trades skipped (sell_date missing from the price history).")
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("No trades could be matched to price history.")

    # Friday-known features on the exit day; Monday-morning gaps on the following session.
    fri = feats.reset_index().rename(columns={"dt": "sell_date"})
    mon = feats[[c for c in MON_GAP_COLS if c in feats.columns]].add_prefix("mon_").reset_index().rename(
        columns={"dt": "mon_date"})
    detail = detail.merge(fri, on="sell_date", how="left").merge(mon, on="mon_date", how="left")
    return detail.sort_values("buy_date").reset_index(drop=True)


def detect_baseline(d: pd.DataFrame) -> tuple[str, dict]:
    """Which Friday mode reproduces the CSV's own sell price?  Returns (baseline, match rates %)."""
    rates = {}
    if "csv_sell_price" in d and d["csv_sell_price"].notna().any():
        ok = d["csv_sell_price"].notna()
        for m in ("fri_open", "fri_close"):
            rates[m] = float(((d.loc[ok, "csv_sell_price"] / d.loc[ok, f"{m}_sell"] - 1).abs() < 0.001).mean() * 100)
    if rates and max(rates.values()) >= 50:
        return max(rates, key=rates.get), rates
    return DEFAULT_BASE, rates


# --------------------------------------------------------------------------------------
# Statistics (profit-based: every comparison uses net % after costs)
# --------------------------------------------------------------------------------------
def mode_summary(d: pd.DataFrame, base: str) -> pd.DataFrame:
    rows = []
    for m in MODES:
        r, diff = d[f"{m}_ret"], d[f"{m}_ret"] - d["base_ret"]
        sd = diff.std(ddof=1)
        rows.append({
            "mode": m, "trades": len(r),
            "avg_net_%": r.mean(), "median_net_%": r.median(),
            "total_net_%": r.sum(), "avg_net_rs": d[f"{m}_rs"].mean(), "total_net_rs": d[f"{m}_rs"].sum(),
            "win_rate_%": (r > 0).mean() * 100.0,
            "stop_hit_%": d[f"{m}_stopped"].mean() * 100.0,
            "avg_vs_base_%": diff.mean(),
            "t_vs_base": (diff.mean() / (sd / np.sqrt(len(diff)))) if m != base and sd > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def tercile_cuts(s: pd.Series):
    s = s.dropna()
    if len(s) < 9 or s.nunique() < 3:
        return None
    return float(s.quantile(1 / 3)), float(s.quantile(2 / 3))


def bucket(s: pd.Series, cuts) -> pd.Series:
    lab = np.select([s <= cuts[0], s >= cuts[1]], ["low", "high"], default="mid")
    return pd.Series(lab, index=s.index).where(s.notna())


def _range_text(lab, cuts):
    return {"low": f"<= {cuts[0]:+.2f}", "mid": f"({cuts[0]:+.2f}, {cuts[1]:+.2f})", "high": f">= {cuts[1]:+.2f}"}[lab]


def factor_breakdown(d: pd.DataFrame, factors: dict) -> pd.DataFrame:
    rows = []
    for name, col in factors.items():
        cuts = tercile_cuts(d[col])
        if cuts is None:
            continue
        b = bucket(d[col], cuts)
        modes = eligible_modes(col)
        for lab in LABELS:
            sub = d[b == lab]
            if sub.empty:
                continue
            means = {m: sub[f"{m}_ret"].mean() for m in MODES}
            best = max(modes, key=means.get)
            rows.append({
                "factor": name, "family": "monday" if is_monday_factor(col) else "friday",
                "bucket": lab, "range_%": _range_text(lab, cuts), "n": len(sub),
                **{f"avg_{m}_%": means[m] for m in MODES},
                "best": best, "edge_vs_base_%": means[best] - sub["base_ret"].mean(),
            })
    return pd.DataFrame(rows)


def learn_rule(train, col, cuts, min_bucket, min_edge, base):
    modes, default = eligible_modes(col), default_mode(col, base)
    b = bucket(train[col], cuts)
    rule = {}
    for lab in LABELS:
        sub = train[b == lab]
        rule[lab] = default
        if len(sub) >= min_bucket:
            means = {m: sub[f"{m}_ret"].mean() for m in modes}
            best = max(means, key=means.get)
            if means[best] - means[default] >= min_edge:
                rule[lab] = best
    return rule


def apply_rule(df, col, cuts, rule, base) -> np.ndarray:
    b = bucket(df[col], cuts)
    pos = {m: i for i, m in enumerate(MODES)}
    chosen = b.map(rule).fillna(default_mode(col, base)).map(pos).to_numpy(dtype=int)  # missing feature -> default
    stack = df[[f"{m}_ret" for m in MODES]].to_numpy(float)
    return stack[np.arange(len(df)), chosen]


def walk_forward(d, factors, n_train, min_bucket, min_edge, base):
    """Learn rules on the first `n_train` trades (Past), score them on the rest (Recent)."""
    train, test = d.iloc[:n_train], d.iloc[n_train:]
    fixed_mode = max(MODES, key=lambda m: train[f"{m}_ret"].mean())  # best single mode on Past
    rows = []
    for name, col in factors.items():
        cuts = tercile_cuts(train[col])
        if cuts is None:
            continue
        rule = learn_rule(train, col, cuts, min_bucket, min_edge, base)
        tr, te = apply_rule(train, col, cuts, rule, base), apply_rule(test, col, cuts, rule, base)
        rows.append({
            "factor": name, "family": "monday" if is_monday_factor(col) else "friday",
            "rule (low/mid/high)": "/".join(rule[x] for x in LABELS),
            "past_n": len(train), "recent_n": len(test),
            "past_rule_%": np.nanmean(tr), "past_base_%": train["base_ret"].mean(),
            "recent_rule_%": np.nanmean(te), "recent_base_%": test["base_ret"].mean(),
            "recent_gain_vs_base_%": np.nanmean(te) - test["base_ret"].mean(),
            "fixed_mode": fixed_mode,
            "recent_fixed_%": test[f"{fixed_mode}_ret"].mean(),
            "recent_gain_vs_fixed_%": np.nanmean(te) - test[f"{fixed_mode}_ret"].mean(),
            "_past_gain_vs_fixed": np.nanmean(tr) - train[f"{fixed_mode}_ret"].mean(),
            # same mode in every bucket = just a fixed mode, not a movement-dependent rule
            "_trivial": len(set(rule.values())) == 1,
            "_rule": rule, "_cuts": cuts, "_col": col,
        })
    return pd.DataFrame(rows), train, test


# --------------------------------------------------------------------------------------
# Past vs Recent: has behaviour changed?
# --------------------------------------------------------------------------------------
SHIFT_Z = 2.0  # |z| above this = statistically notable change


def _finite(x) -> np.ndarray:
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def _welch_z(past, recent) -> float:
    x, y = _finite(past), _finite(recent)
    if len(x) < 5 or len(y) < 5:
        return np.nan
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    return float((y.mean() - x.mean()) / se) if se > 0 else np.nan


def _prop_z(past_bool, recent_bool) -> float:
    x, y = _finite(past_bool), _finite(recent_bool)
    if len(x) < 5 or len(y) < 5:
        return np.nan
    p = (x.sum() + y.sum()) / (len(x) + len(y))
    se = np.sqrt(p * (1 - p) * (1 / len(x) + 1 / len(y)))
    return float((y.mean() - x.mean()) / se) if se > 0 else np.nan


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), int(ok.sum())


def _corr_z(r1, n1, r2, n2) -> float:
    if not (np.isfinite(r1) and np.isfinite(r2)) or n1 < 8 or n2 < 8:
        return np.nan
    r1, r2 = np.clip(r1, -0.999, 0.999), np.clip(r2, -0.999, 0.999)
    return float((np.arctanh(r2) - np.arctanh(r1)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3)))


def mode_shift_table(past: pd.DataFrame, recent: pd.DataFrame, base: str) -> pd.DataFrame:
    """Value of each exit mode in Past vs Recent, and whether its edge over the baseline changed."""
    rows = []
    for m in MODES:
        rp, rr = past[f"{m}_ret"].to_numpy(float), recent[f"{m}_ret"].to_numpy(float)
        dp, dr = rp - past["base_ret"].to_numpy(float), rr - recent["base_ret"].to_numpy(float)
        sd = dr.std(ddof=1) if len(dr) > 1 else 0.0
        z = _welch_z(dp, dr) if m != base else np.nan
        rows.append({
            "mode": m, "past_avg_%": rp.mean(), "recent_avg_%": rr.mean(),
            "past_edge_vs_base_%": dp.mean(), "recent_edge_vs_base_%": dr.mean(),
            "recent_edge_t": (dr.mean() / (sd / np.sqrt(len(dr)))) if m != base and sd > 0 else np.nan,
            "change_z": z, "shift": "SHIFT" if np.isfinite(z) and abs(z) > SHIFT_Z else "",
        })
    return pd.DataFrame(rows)


def behavior_shift_table(past: pd.DataFrame, recent: pd.DataFrame, sector_col: str | None) -> pd.DataFrame:
    """Has the stock's Friday / weekend / Monday behaviour changed?  Each row is one measurable behaviour."""
    def drift(d, p):   # open -> close move on that day, %
        return ((d[f"{p}_close"] / d[f"{p}_open"] - 1.0) * 100.0).to_numpy(float)

    def rng(d, p):     # high-low range, % of open
        return ((d[f"{p}_high"] - d[f"{p}_low"]) / d[f"{p}_open"] * 100.0).to_numpy(float)

    def wkend(d):      # Friday close -> Monday open, %
        return ((d["mon_open"] / d["fri_close"] - 1.0) * 100.0).to_numpy(float)

    rows = []

    def add(name, pv, rv, z):
        rows.append({"behaviour": name, "past": pv, "recent": rv, "change": rv - pv, "z": z,
                     "shift": "SHIFT" if np.isfinite(z) and abs(z) > SHIFT_Z else ""})

    for name, fn in (("avg Fri open->close drift %", lambda d: drift(d, "fri")),
                     ("avg Fri day range %", lambda d: rng(d, "fri")),
                     ("avg weekend gap % (Fri close->Mon open)", wkend),
                     ("avg |weekend gap| % (volatility)", lambda d: np.abs(wkend(d))),
                     ("avg Mon open->close drift %", lambda d: drift(d, "mon")),
                     ("avg Mon day range %", lambda d: rng(d, "mon"))):
        add(name, np.nanmean(fn(past)), np.nanmean(fn(recent)), _welch_z(fn(past), fn(recent)))
    add("avg |Fri gap| % (volatility)", np.nanmean(np.abs(past["stk_gap"])), np.nanmean(np.abs(recent["stk_gap"])),
        _welch_z(np.abs(past["stk_gap"]), np.abs(recent["stk_gap"])))
    add("stop-hit rate % (held over weekend)", past["mon_close_stopped"].mean() * 100,
        recent["mon_close_stopped"].mean() * 100,
        _prop_z(past["mon_close_stopped"].astype(float), recent["mon_close_stopped"].astype(float)))
    add("win rate % (baseline exit)", (past["base_ret"] > 0).mean() * 100, (recent["base_ret"] > 0).mean() * 100,
        _prop_z((past["base_ret"] > 0).astype(float), (recent["base_ret"] > 0).astype(float)))

    # Does Monday's drift fade or continue the Monday-morning gap?  Negative = fade, positive = continue.
    rel = [("corr(Mon drift, Mon stock gap)", "mon_stk_gap"), ("corr(Mon drift, Mon NIFTY gap)", "mon_nifty_gap")]
    if sector_col:
        rel.append((f"corr(Mon drift, Mon {sector_col.replace('_gap', '')} gap)", f"mon_{sector_col}"))
    for name, col in rel:
        r1, n1 = _corr(drift(past, "mon"), past[col])
        r2, n2 = _corr(drift(recent, "mon"), recent[col])
        if np.isfinite(r1) and np.isfinite(r2):
            add(name, r1, r2, _corr_z(r1, n1, r2, n2))
    return pd.DataFrame(rows)


def confidence_score(past, recent, policy_fn, mode_tbl, behav_tbl, base):
    """0-100 confidence that the Past-learned exit policy still describes how the stock behaves now.

    persistence 35%  recent edge over the baseline relative to the edge seen in Past (capped 0..1)
    stability   25%  how statistically large the change in that edge is (|z| >= 3 -> 0)
    behaviour   20%  share of the tracked behaviours that did NOT shift
    sample      20%  enough trades on both sides (Recent >= 30, Past >= 90 for full marks)
    If the policy is the plain baseline exit (no edge claimed), persistence/stability instead check that no
    alternative exit has become clearly better in Recent.
    """
    e_past = policy_fn(past) - past["base_ret"].to_numpy(float)
    e_rec = policy_fn(recent) - recent["base_ret"].to_numpy(float)
    alt = mode_tbl[mode_tbl["mode"] != base]

    if np.nanmax(np.abs(e_past)) > 1e-12 or np.nanmax(np.abs(e_rec)) > 1e-12:   # policy differs from baseline
        mp, mr = np.nanmean(e_past), np.nanmean(e_rec)
        persistence = float(np.clip(mr / mp, 0, 1)) if mp > 0 else 0.0
        z = _welch_z(e_past, e_rec)
        stability = 1.0 - float(np.clip(abs(z) / 3.0, 0, 1)) if np.isfinite(z) else 0.5
    else:                                                                         # policy == baseline exit
        mp = mr = 0.0
        best_alt_t = np.nanmax(alt["recent_edge_t"].to_numpy(float)) if alt["recent_edge_t"].notna().any() else 0.0
        persistence = 1.0 - float(np.clip(max(best_alt_t, 0.0) / 3.0, 0, 1))
        mean_abs_z = np.nanmean(np.abs(alt["change_z"].to_numpy(float))) if alt["change_z"].notna().any() else 0.0
        stability = 1.0 - float(np.clip(mean_abs_z / 3.0, 0, 1))

    tested = behav_tbl["z"].notna()
    shifted = int((behav_tbl["shift"] == "SHIFT").sum())
    behaviour = 1.0 - (shifted / int(tested.sum())) if tested.any() else 0.5
    sample = 0.5 * min(1.0, len(recent) / 30.0) + 0.5 * min(1.0, len(past) / 90.0)

    comps = pd.DataFrame([
        {"component": "persistence (recent edge vs past edge)", "score": persistence, "weight": 0.35},
        {"component": "stability (change in edge not significant)", "score": stability, "weight": 0.25},
        {"component": "behaviour (tracked metrics unchanged)", "score": behaviour, "weight": 0.20},
        {"component": "sample size (past/recent trades)", "score": sample, "weight": 0.20},
    ])
    comps["points"] = comps["score"] * comps["weight"] * 100.0
    total = float(comps["points"].sum())
    label = "HIGH" if total >= 70 else "MEDIUM" if total >= 45 else "LOW"
    return total, label, comps, float(mp), float(mr)


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def _fmt(df: pd.DataFrame, nd=3) -> str:
    return df.to_string(index=False, float_format=lambda x: f"{x:.{nd}f}")


def _period(d: pd.DataFrame) -> str:
    return f"{d['sell_date'].min():%Y-%m-%d} to {d['sell_date'].max():%Y-%m-%d}, {len(d)} trades"


def _analyze_one(a):
    """Run the exit analysis for one ticker and return its recommendation row."""
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    bare = _bare(a.ticker)

    trades = load_trades(a.trades, a.ticker, a.strategy)
    stk, idx, meta, peers = fetch_market_data(_variants(a.ticker), tuple(a.index), a.sql_path)
    feats = build_features(stk, idx, peers, a.min_peers)
    d = simulate_all(trades, stk, feats, a.stop_loss)

    print("=" * 100)
    print(f"EXIT-TIMING ANALYSIS: {bare}   ({len(d)} {EXIT_REASON} trades, "
          f"{d['sell_date'].min():%Y-%m-%d} to {d['sell_date'].max():%Y-%m-%d})")
    if meta:
        print("industry = {industry} | sector_group = {sector_group} | macro_sector = {macro_sector}".format(**meta))
        used = {LEVELS[k]: int(feats[f"{LEVELS[k]}_n"].median()) for k in peers if feats[f"{LEVELS[k]}_n"].notna().any()}
        print("median peer count: " + ", ".join(f"{k}={v}" for k, v in used.items()))
    print("=" * 100)

    print("\n--- 0. Which sector level tracks this stock best? ---")
    lvl_scores, auto_level = score_levels(feats, meta)
    if lvl_scores.empty:
        print("No sector level had enough peer data; using stock + NIFTY features only.")
        level_choice = None
    else:
        print(_fmt(lvl_scores))
        level_choice = auto_level if a.sector_level == "auto" else a.sector_level
        if level_choice not in (None, "all") and level_choice not in set(lvl_scores["level"]):
            print(f"warning: {level_choice} has too little peer data; falling back to {auto_level}.")
            level_choice = auto_level
        print(f"Best tracker: {auto_level}.  Using sector level: {level_choice}.")
    factors = factors_for(level_choice)
    sector_level_for_behavior = level_choice if level_choice in LEVELS else auto_level
    sector_col = f"{LEVELS[sector_level_for_behavior]}_gap" if sector_level_for_behavior else None

    base, rates = detect_baseline(d)
    if rates:
        print("Replication check vs CSV: sell price matches Friday "
              + " / ".join(f"{k[4:]} {v:.0f}%" for k, v in rates.items()) + f" of trades (within 0.1%) -> baseline = {base}.")
    ok = d["csv_net_pnl_pct"].notna()
    if ok.any():
        diff = (d.loc[ok, f"{base}_ret"] - d.loc[ok, "csv_net_pnl_pct"]).abs()
        print(f"Baseline net % within 0.1 of CSV for {(diff < 0.1).mean() * 100:.0f}% of trades "
              f"(mean abs diff {diff.mean():.3f}).")

    # Compare modes only on trades where every mode could be executed (Monday session must exist).
    full = d.dropna(subset=[f"{m}_ret" for m in MODES]).reset_index(drop=True)
    if len(full) < len(d):
        print(f"note: {len(d) - len(full)} trades dropped from the comparison (a Monday exit could not execute).")
    if len(full) < 10:
        raise SystemExit("Too few usable trades for a meaningful analysis.")
    full["base_ret"] = full[f"{base}_ret"]

    print("\n--- 1. Exit modes, all trades (net % of capital per trade, after costs) ---")
    summ = mode_summary(full, base)
    print(_fmt(summ))
    best_fixed = summ.loc[summ["avg_net_%"].idxmax(), "mode"]
    print(f"Best fixed mode by net profit: {best_fixed}   (|t_vs_base| below ~2 means the gap to `{base}` is within noise)")

    print("\n--- 2. By movement regime, all trades (terciles; which mode wins in each bucket) ---")
    fb = factor_breakdown(full, factors)
    if fb.empty:
        print("Not enough feature data.")
    else:
        print(_fmt(fb.sort_values("edge_vs_base_%", ascending=False)))
        print("(family=monday: Monday-morning gap known only after the weekend -> only Monday exits compete.)")

    # ---- Past vs Recent --------------------------------------------------------------
    if a.recent_since is not None:
        n_past = int((full["sell_date"] < a.recent_since).sum())
        split_desc = f"date boundary {a.recent_since:%Y-%m-%d} (--recent-since)"
    else:
        n_recent = max(1, int(round(len(full) * a.recent_frac)))
        n_past = len(full) - n_recent
        split_desc = f"newest {a.recent_frac:.1%} of trades (--recent-frac)"
    full = full.sort_values("sell_date", kind="stable").reset_index(drop=True)  # chronological by exit
    past, recent = full.iloc[:n_past], full.iloc[n_past:]
    print(f"Past/Recent split: {split_desc} -> {len(past)} Past / {len(recent)} Recent")

    mode_tbl = behav_tbl = wf = pd.DataFrame()
    conf = None
    policy_lines: list[str] = []
    policy_desc = f"always {best_fixed} ({MODE_DESC[best_fixed]})"
    print("\n--- 3. Past vs Recent: has the stock's behaviour changed? ---")
    if len(past) < a.min_past or len(recent) < a.min_recent:
        print(f"Need >= {a.min_past} Past and >= {a.min_recent} Recent trades "
              f"(have {len(past)} / {len(recent)}); skipping. "
              f"Adjust --min-past / --min-recent and the split using --recent-frac / --recent-since.")
    else:
        print(f"Past  : {_period(past)}\nRecent: {_period(recent)}")
        mode_tbl = mode_shift_table(past, recent, base)
        print("\nValue of each exit mode (net % per trade):")
        print(_fmt(mode_tbl))
        bp, br = (mode_tbl.loc[mode_tbl[c].idxmax(), "mode"] for c in ("past_avg_%", "recent_avg_%"))
        print(f"Best mode in Past: {bp}   |   best mode in Recent: {br}"
              + ("   -> ranking changed" if bp != br else "   -> same winner"))
        behav_tbl = behavior_shift_table(past, recent, sector_col)
        print("\nBehaviour around the weekend exit (SHIFT = |z| > 2, i.e. unlikely to be chance):")
        print(_fmt(behav_tbl))

        print("\n--- 4. Rules learned on Past only, scored on Recent ---")
        wf, _, _ = walk_forward(full, factors, n_past, a.min_bucket, a.min_edge, base)
        if not wf.empty:
            print(_fmt(wf[[c for c in wf.columns if not c.startswith("_")]]))

        # Policy to recommend = learned from Past only; objective = net profit.
        fixed_mode = max(MODES, key=lambda m: past[f"{m}_ret"].mean())
        cand = wf[(~wf["_trivial"]) & (wf["_past_gain_vs_fixed"] >= a.min_edge)] if not wf.empty else wf
        if not cand.empty:
            pick = cand.loc[cand["_past_gain_vs_fixed"].idxmax()]
            rule, cuts, col = pick["_rule"], pick["_cuts"], pick["_col"]
            policy_fn = lambda df, _c=col, _u=cuts, _r=rule: apply_rule(df, _c, _u, _r, base)
            if is_monday_factor(col):
                policy_desc = f"hold over weekend, then Monday-morning rule on {pick['factor']}"
            else:
                policy_desc = f"movement rule on {pick['factor']}"
            policy_lines = [f"   {pick['factor']} {lab:<4} ({_range_text(lab, cuts)} %)  ->  {rule[lab]} ({MODE_DESC[rule[lab]]})"
                            for lab in LABELS]
        else:
            policy_fn = lambda df, _m=fixed_mode: df[f"{_m}_ret"].to_numpy(float)
            policy_desc = f"always {fixed_mode} ({MODE_DESC[fixed_mode]})" + \
                          (" (no movement rule beat it by the minimum edge on Past)" if not wf.empty else "")
        conf = confidence_score(past, recent, policy_fn, mode_tbl, behav_tbl, base)

    print("\n--- Verdict ---")
    if conf is None:
        print(f"Best fixed mode by net profit on all trades: {best_fixed}. Not enough history for the "
              f"Past/Recent check, so no confidence score.")
    else:
        total, label, comps, mp, mr = conf
        print(f"Exit policy learned on PAST trades: {policy_desc}")
        for ln in policy_lines:
            print(ln)
        print(f"Net-profit edge over baseline exit ({base}): Past {mp:+.3f}%  ->  Recent {mr:+.3f}% per trade")
        shifted = list(behav_tbl.loc[behav_tbl["shift"] == "SHIFT", "behaviour"]) + \
                  [f"{m} edge" for m in mode_tbl.loc[mode_tbl["shift"] == "SHIFT", "mode"]]
        print("Deviations in Recent: " + (", ".join(shifted) if shifted else "none detected"))
        print(f"\nCONFIDENCE: {total:.0f} / 100  ({label})")
        print(_fmt(comps.assign(score=comps["score"].round(2), points=comps["points"].round(1))))
        if label == "HIGH":
            print("=> Behaviour in Recent trades is consistent with Past. The policy can be used as is.")
        elif label == "MEDIUM":
            print("=> Mostly consistent, with some drift or thin data. Use it, but re-check as new trades arrive.")
        else:
            print("=> Recent behaviour deviates from Past (or too little data). Don't rely on the learned policy; "
                  f"stay with the baseline {base} exit until more trades confirm it.")
        print("(Heuristic score: weights are judgement, not statistics. Treat it as a guide, not a probability.)")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / f"{bare}_exit_timing_detail.csv", index=False)
    summ.to_csv(out / f"{bare}_mode_summary.csv", index=False)
    fb.to_csv(out / f"{bare}_factor_breakdown.csv", index=False)
    if not mode_tbl.empty:
        mode_tbl.to_csv(out / f"{bare}_past_vs_recent_modes.csv", index=False)
        behav_tbl.to_csv(out / f"{bare}_past_vs_recent_behaviour.csv", index=False)
    if not wf.empty:
        wf[[c for c in wf.columns if not c.startswith("_")]].to_csv(out / f"{bare}_past_rules_on_recent.csv", index=False)
    if conf is not None:
        conf[2].to_csv(out / f"{bare}_confidence.csv", index=False)
    print(f"\nCSVs written to {out.resolve()}")
    confidence_value = conf[0] if conf is not None else np.nan
    confidence_label = conf[1] if conf is not None else "INSUFFICIENT_DATA"
    exit_details = " | ".join([policy_desc, *policy_lines]) if conf is not None else policy_desc
    return {
        "ticker": bare,
        "exit_details": exit_details,
        "CONFIDENCE": confidence_value,
        "confidence_label": confidence_label,
    }


def _verbose_output(output: str, level: int) -> str:
    """Keep analysis sections up to the requested verbosity (1..4)."""
    if level >= 4:
        return output
    kept = []
    active_section = 0
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("--- ") and ". " in stripped:
            try:
                active_section = int(stripped.split("--- ", 1)[1].split(".", 1)[0])
            except ValueError:
                active_section = 0
        elif stripped == "--- Verdict ---":
            active_section = 99
        if active_section == 0 or active_section == 99 or (1 <= active_section <= level):
            kept.append(line)
    return "\n".join(kept)


def analyze_exit_timing(
        tickers: str | Sequence[str],
        *,
        trades: str = "fasta_all_trades_complete_run.csv",
        strategy: str | None = None,
        stop_loss: float = -2.0,
        index: Sequence[str] = ("^NSEI", "NIFTYBEES", "NIFTYBEES.NS"),
        sql_path: str | None = None,
        sector_level: str = "auto",
        min_peers: int = 3,
        recent_frac: float = 0.3,
        recent_since: str | None = None,
        min_past: int = 30,
        min_recent: int = 10,
        min_bucket: int = 8,
        min_edge: float = 0.10,
        out_dir: str = "exit_timing_out",
        verbose: int = 0,
) -> pd.DataFrame:
    """Analyze WEEKEND_EXIT trades of one or more tickers; return ticker, exit details and confidence.

    `tickers` may be a comma-separated string or any sequence of ticker symbols.
    Verbosity: 0=silent, 1=overall exit-mode results, 2=also movement regimes,
    3=also Past/Recent behaviour, 4=full learning/rule diagnostics.
    """
    if isinstance(tickers, str):
        ticker_list = [x.strip() for x in tickers.split(",") if x.strip()]
    else:
        ticker_list = [str(x).strip() for x in tickers if str(x).strip()]
    if not ticker_list:
        raise ValueError("At least one ticker is required.")
    if not 0 <= int(verbose) <= 4:
        raise ValueError("verbose must be an integer from 0 to 4.")
    if not 0.0 < recent_frac < 1.0:
        raise ValueError("recent_frac must be greater than 0 and less than 1.")
    if min_past < 1 or min_recent < 1:
        raise ValueError("min_past and min_recent must each be at least 1.")
    if sector_level not in {"auto", "industry", "sector_group", "macro_sector", "all"}:
        raise ValueError("Invalid sector_level.")
    if recent_since is not None:
        try:
            recent_since = pd.to_datetime(recent_since, format="%Y-%m-%d", errors="raise").normalize()
        except (TypeError, ValueError):
            raise ValueError("recent_since must be a valid date in YYYY-MM-DD format.")

    results = []
    for ticker in ticker_list:
        args = argparse.Namespace(
            ticker=ticker, trades=trades, strategy=strategy, stop_loss=stop_loss,
            index=list(index), sql_path=sql_path, sector_level=sector_level,
            min_peers=min_peers, recent_frac=recent_frac, recent_since=recent_since,
            min_past=min_past, min_recent=min_recent, min_bucket=min_bucket,
            min_edge=min_edge, out_dir=out_dir,
        )
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                row = _analyze_one(args)
        except (Exception, SystemExit) as exc:
            row = {
                "ticker": _bare(ticker),
                "exit_details": f"Analysis failed: {exc}",
                "CONFIDENCE": np.nan,
                "confidence_label": "ERROR",
            }
        results.append(row)
        if verbose:
            rendered = _verbose_output(buffer.getvalue(), int(verbose))
            if rendered.strip():
                print(rendered)
    return pd.DataFrame(results, columns=["ticker", "exit_details", "CONFIDENCE", "confidence_label"])


def main():
    ap = argparse.ArgumentParser(description="Which exit time would have maximised net profit on weekend exits "
                                             "for one or more tickers?")
    ap.add_argument("ticker", help="ticker or comma-separated tickers, e.g. RELIANCE,TCS,INFY")
    ap.add_argument("--trades", default="fasta_all_trades_complete_run.csv", help="all-trades CSV from the FASTA UI")
    ap.add_argument("--strategy", default=None, help="only use trades from this strategy")
    ap.add_argument("--stop-loss", type=float, default=-2.0)
    ap.add_argument("--index", nargs="+", default=["^NSEI", "NIFTYBEES", "NIFTYBEES.NS"])
    ap.add_argument("--sql-path", default=None, help="override for sqls/all.sql")
    ap.add_argument("--sector-level", default="auto", choices=["auto", "industry", "sector_group", "macro_sector", "all"])
    ap.add_argument("--min-peers", type=int, default=3)
    ap.add_argument("--recent-frac", type=float, default=0.3)
    ap.add_argument("--recent-since", default=None, help="YYYY-MM-DD; overrides --recent-frac")
    ap.add_argument("--min-past", type=int, default=30)
    ap.add_argument("--min-recent", type=int, default=10)
    ap.add_argument("--min-bucket", type=int, default=8)
    ap.add_argument("--min-edge", type=float, default=0.10)
    ap.add_argument("--out-dir", default="exit_timing_out")
    ap.add_argument("--verbose", type=int, choices=range(0, 5), default=4,
                    help="0=silent, 1=exit modes, 2=movement regimes, 3=Past/Recent, 4=full learning output")
    a = ap.parse_args()
    result = analyze_exit_timing(
        a.ticker, trades=a.trades, strategy=a.strategy, stop_loss=a.stop_loss,
        index=a.index, sql_path=a.sql_path, sector_level=a.sector_level,
        min_peers=a.min_peers, recent_frac=a.recent_frac, recent_since=a.recent_since,
        min_past=a.min_past, min_recent=a.min_recent, min_bucket=a.min_bucket,
        min_edge=a.min_edge, out_dir=a.out_dir, verbose=a.verbose,
    )
    print("\nRecommendation dataframe:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
