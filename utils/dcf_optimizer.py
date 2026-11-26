"""
utils/dcf_optimizer.py
──────────────────────
Fit ticker-specific DCF assumptions (g1, g2, tg, dr, mos) to each stock's own price
history with the Adam optimizer, and store them in a table (`config/dcf_optimized.csv`)
that the DCF page reads as per-ticker defaults.

Run from the project root (so `sqls/all.sql` resolves):

    python -m utils.dcf_optimizer --tickers RELIANCE,INFY
    python -m utils.dcf_optimizer --all
    python -m utils.dcf_optimizer --all --horizon-months 24 --out config/dcf_optimized.csv

How it works
────────────
Point-in-time samples. At every `--step-months` date d, rebuild what an investor could
have known: annual profit / other income / exceptional items that were published by d
(fiscal-year end + `--lag-days`), run them through the SAME `normalize_earnings` the page
uses, and divide by the shares implied that year (net profit / EPS). That gives a
per-share base earnings x_d.

Because the DCF is linear in base profit, the model value is  IV_d = x_d * f(θ),
f(θ) = dcf_multiple(g1, g2, tg, dr).  Valuation target: the price `--horizon-months`
later, discounted back at the cost of capital being fitted:

    r_d(θ) = ln IV_d − [ ln P_{d+h} − (h/12)·ln(1 + dr/100) ]

(h = 0 fits today's price instead.)  Loss, with recency weights w_d (half-life
`--half-life-years`):

    L(θ) = Σ w_d·r_d²                                  how well DCF explains price
         + w_growth · mean((g1 − G_d)² / 15²)          g1 vs realised 3-yr forward profit CAGR
         + w_prior  · Σ ((θ_k − prior_k)/σ_k)²         pull toward historical growth / 5% / 12%

θ is reparameterised through sigmoids so every Adam step stays inside the page's slider
ranges (g1 0-40, g2 ≤ min(g1, 30), tg 2-8, dr 8-20, dr ≥ tg + 1).  Gradients are central
finite differences (4 parameters, so this is cheap and needs no autodiff dependency);
the update rule is standard Adam with bias correction.  The optimum is then projected onto
the page's 0.5-step slider grid (best of the 81 neighbouring grid points).  The margin of
safety is set from the fit's residual error:  mos = 100·(1 − e^(−rmse)), rounded to 5 and clipped to 0-50.

Cash-flow basis. dcf.sql values `norm_fcf = norm_profit * fcf_conversion` (conversion =
clip(sum FCF / sum profit over the last 4 years, 0.1, 1.5); 1.0 for Financial Services).
With `--basis fcf` (default) the optimizer applies the same point-in-time conversion to x_d,
so the fitted assumptions reproduce prices through dcf.sql's own formula. The Streamlit page
values net profit directly, so use `--basis profit` if the CSV is meant for the page only.

Prior. Taken from the precomputed `v_dcf_assumptions_auto` view (CAPM discount rate, shrunk
weighted growth, sustainable-growth cap) when available, else from reported 3Y/5Y growth.

Data guard. Samples whose implied P/E (price / EPS) is outside 3-300 are dropped (the same
filter as the P/E history); this removes split/bonus mismatches between prices and EPS.

Fit mode. Prices pin down only the effective multiple f(θ), so fitting g1, g2, tg and dr
together drives many tickers to the box corners (dr at 8%, tg at 7%, g1 at the cap). The
default `--fit growth` therefore keeps dr and tg at the precomputed auto values (CAPM discount
rate, policy terminal growth) and fits only the growth rates g1, g2, i.e. the growth the market
is pricing in. `--fit all` restores the free four-parameter fit.

Confidence. Every row gets `confidence` (high / medium / low) and `confidence_reason`.
Low: fit error > 1.0, fewer than 10 samples, price more than 100% away from the DCF, growth at
the cap with a weak fit, or an outlier fit versus the rest of the universe (robust z > 2).
Medium: fit error > 0.6, growth near the cap, fewer than 16 samples, price 50-100% away, no
better than the prior, or FCF conversion clipped. Otherwise high. `--rescore` re-rates an
existing CSV without refitting (use it to tune the thresholds above).

Identifiability caveat: the price data pins down the *effective valuation multiple*
f(θ)·(1+dr)^(h/12) very well, but not g1, g2, tg and dr individually. The growth and
prior terms decide how that multiple is split between them, so treat the individual
numbers as a regularised decomposition, and compare `rmse_fit` with `rmse_prior`.
Dividends are ignored in the return term (slightly biases dr downwards).
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils import dcf_core
from utils.db_utils import run_readonly_query

# ── parameter bounds (mirror the page's sliders) ─────────────────────────────
G1_MAX, G2_MAX = 40.0, 30.0
TG_LO, TG_HI   = 2.0, 8.0
DR_LO, DR_HI   = 8.0, 20.0
DR_MIN_SPREAD  = 1.0           # dr >= tg + 1 keeps the terminal value finite and sane

DEFAULT_PRIOR  = {"g1": 12.0, "g2": 10.0, "tg": 5.0, "dr": 12.0}
PRIOR_SIGMA    = {"g1": 10.0, "g2": 10.0, "tg": 1.5, "dr": 2.5}
GROWTH_SIGMA   = 15.0

MOS_LO, MOS_HI = 15, 40        # mirrors dcf_policy mos_min / mos_max
FCF_CONV_LO, FCF_CONV_HI, FCF_CONV_DEFAULT = 0.1, 1.5, 0.7   # mirrors dcf.sql cf_ratio
PE_LO, PE_HI = 3.0, 300.0                                      # mirrors v_dcf_pe_history filter

OUT_COLUMNS = ["ticker", "g1", "g2", "tg", "dr", "mos",
               "rmse_fit", "rmse_prior", "n_samples", "horizon_months", "fitted_on",
               "fit_mode", "basis", "fcf_conv", "price_latest", "iv_latest", "gap_pct",
               "confidence", "confidence_reason"]

# ── confidence rating (written to the CSV, carried through dcf.sql and the page) ──────────
CONF_RMSE_MEDIUM, CONF_RMSE_LOW = 0.60, 1.00   # weighted log-error: 0.6 ~ x1.8 off, 1.0 ~ x2.7 off
CONF_G_CAP       = G1_MAX - 5                  # g1 this close to the cap = growth the data can't pin
CONF_N_MEDIUM, CONF_N_LOW = 16, 10             # fewer usable samples -> weaker evidence
CONF_GAP_MEDIUM, CONF_GAP_LOW = 50.0, 100.0    # |price / DCF - 1| in % at the latest date
CONF_Z_LOW       = 2.0                         # robust z of rmse_fit vs the other tickers
CONF_Z_MIN_N     = 30                          # need this many tickers for the relative test


@dataclass
class FitConfig:
    horizon_months: int   = 12
    step_months:    int   = 3
    lag_days:       int   = 90      # results are published ~this long after fiscal year-end
    half_life_years: float = 3.0
    w_growth:       float = 0.3
    w_prior:        float = 0.05
    iters:          int   = 500
    lr:             float = 0.03
    min_samples:    int   = 8
    basis:          str   = "fcf"   # "fcf" (matches dcf.sql) or "profit" (matches the page)
    fit:            str   = "growth"  # "growth": fit g1,g2 only, dr/tg fixed at the prior (auto view)
    # "all": fit g1,g2,tg,dr (price pins only the multiple -> corners)


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERISATION  (unconstrained u  <->  bounded assumptions θ)
# ══════════════════════════════════════════════════════════════════════════════

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 0.02), 0.98)
    return float(np.log(p / (1 - p)))


def theta_from_u(u: np.ndarray) -> tuple[float, float, float, float]:
    g1 = G1_MAX * _sigmoid(u[0])
    g2 = min(g1, G2_MAX) * _sigmoid(u[1])
    tg = TG_LO + (TG_HI - TG_LO) * _sigmoid(u[2])
    dr = max(DR_LO + (DR_HI - DR_LO) * _sigmoid(u[3]), tg + DR_MIN_SPREAD)
    return g1, g2, tg, dr


def u_from_theta(g1: float, g2: float, tg: float, dr: float) -> np.ndarray:
    g1 = min(max(g1, 0.0), G1_MAX)
    g2 = min(max(g2, 0.0), min(g1, G2_MAX))
    return np.array([
        _logit(g1 / G1_MAX),
        _logit(g2 / max(min(g1, G2_MAX), 1e-9)),
        _logit((tg - TG_LO) / (TG_HI - TG_LO)),
        _logit((dr - DR_LO) / (DR_HI - DR_LO)),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# LOSS
# ══════════════════════════════════════════════════════════════════════════════

def residuals(theta, s: pd.DataFrame) -> np.ndarray:
    """r_d(θ) for every sample row (columns: x, target, years_fwd)."""
    g1, g2, tg, dr = theta
    f = dcf_core.dcf_multiple(g1, g2, tg, dr)
    if not np.isfinite(f) or f <= 0:
        return np.full(len(s), np.inf)
    return (np.log(s["x"].values * f) - np.log(s["target"].values)
            + s["years_fwd"].values * np.log1p(dr / 100))


def loss(theta, s: pd.DataFrame, prior: dict, cfg: FitConfig) -> float:
    r = residuals(theta, s)
    if not np.all(np.isfinite(r)):
        return 1e9
    w = s["w"].values
    l_fit = float(np.sum(w * r ** 2))

    g1, g2, tg, dr = theta
    valid = s["g_real"].notna().values
    l_growth = 0.0
    if valid.any():
        wv = w[valid] / w[valid].sum()
        l_growth = float(np.sum(wv * ((g1 - s["g_real"].values[valid]) / GROWTH_SIGMA) ** 2))

    l_prior = sum(((v - prior[k]) / PRIOR_SIGMA[k]) ** 2
                  for k, v in zip(("g1", "g2", "tg", "dr"), theta))
    return l_fit + cfg.w_growth * l_growth + cfg.w_prior * l_prior


# ══════════════════════════════════════════════════════════════════════════════
# ADAM
# ══════════════════════════════════════════════════════════════════════════════

def _num_grad(fn, u: np.ndarray, h: float = 1e-4) -> np.ndarray:
    g = np.zeros_like(u)
    for i in range(len(u)):
        up, dn = u.copy(), u.copy()
        up[i] += h
        dn[i] -= h
        g[i] = (fn(up) - fn(dn)) / (2 * h)
    return g


def adam_minimize(fn, u0: np.ndarray, iters: int = 500, lr: float = 0.03,
                  beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                  patience: int = 40, tol: float = 1e-8) -> tuple[np.ndarray, list[float]]:
    """Plain Adam (bias-corrected). Returns the best u seen and the loss history."""
    u = u0.astype(float).copy()
    m = np.zeros_like(u)
    v = np.zeros_like(u)
    best_u, best_l = u.copy(), fn(u)
    history, stall = [best_l], 0
    for t in range(1, iters + 1):
        g = _num_grad(fn, u)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        u = u - lr * (m / (1 - beta1 ** t)) / (np.sqrt(v / (1 - beta2 ** t)) + eps)
        cur = fn(u)
        history.append(cur)
        if cur < best_l - tol:
            best_u, best_l, stall = u.copy(), cur, 0
        else:
            stall += 1
            if stall >= patience:
                break
    return best_u, history


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE BUILDER  (point-in-time, no look-ahead in the inputs)
# ══════════════════════════════════════════════════════════════════════════════

def _annual(df: pd.DataFrame, how: str, name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["year", name])
    s = df.set_index("dt")["val"]
    s = s.resample("YE").sum() if how == "sum" else s.resample("YE").mean()
    return s.rename(name).reset_index().rename(columns={"dt": "year"})


def _price_at(prices: pd.Series, when: pd.Timestamp) -> float | None:
    """Last close on or before `when`; None if `when` is past the data or before its start."""
    if when > prices.index[-1] or when < prices.index[0]:
        return None
    return float(prices.loc[:when].iloc[-1])


def _fcf_conversion(fcf_df: pd.DataFrame, np_all_d: pd.DataFrame, latest_yr: int,
                    sector: str | None, lag: pd.Timedelta, d: pd.Timestamp) -> float:
    """Point-in-time version of dcf.sql's cf_ratio: sum(FCF)/sum(profit) over the last 4 fiscal years."""
    if sector == "Financial Services":
        return 1.0
    if fcf_df is None or fcf_df.empty or np_all_d.empty:
        return FCF_CONV_DEFAULT
    f = fcf_df[fcf_df["dt"] + lag <= d]
    if f.empty:
        return FCF_CONV_DEFAULT
    f_y = f.groupby(f["dt"].dt.year)["val"].sum()
    p_y = np_all_d.groupby(np_all_d["year"].dt.year)["net_profit"].sum()
    j = pd.concat([f_y.rename("fcf"), p_y.rename("np")], axis=1, join="inner")
    j = j[(j.index >= latest_yr - 3) & (j.index <= latest_yr)]
    if j.empty or j["np"].sum() == 0:
        return FCF_CONV_DEFAULT
    return float(min(max(j["fcf"].sum() / j["np"].sum(), FCF_CONV_LO), FCF_CONV_HI))


class _Ctx:
    """Everything fetched once per ticker."""
    def __init__(self, tkr: str, cfg: FitConfig):
        self.np_df, self.eps_df = dcf_core.get_net_profit(tkr), dcf_core.get_eps(tkr)
        self.px_df = dcf_core.get_price_history(tkr)
        self.ok = not (self.np_df.empty or self.eps_df.empty or self.px_df.empty)
        if not self.ok:
            return
        self.oi_df, self.exc_df = dcf_core.get_other_income(tkr), dcf_core.get_exceptional_items(tkr)
        self.tax_r  = dcf_core.get_tax_rate(tkr)
        self.sector = dcf_core.get_sector(tkr)
        self.fcf_df = dcf_core.get_free_cash_flow(tkr) if cfg.basis == "fcf" else pd.DataFrame()
        self.np_all  = _annual(self.np_df, "sum", "net_profit")     # full series (forward-growth target)
        self.eps_ann = _annual(self.eps_df, "mean", "eps")
        self.prices  = self.px_df.drop_duplicates("dt").set_index("dt")["price"].sort_index()
        self.lag     = pd.Timedelta(days=cfg.lag_days)


def _state_at(c: _Ctx, d: pd.Timestamp, cfg: FitConfig) -> dict | None:
    """What an investor could know at `d`: per-share base cash earnings x, implied P/E, fcf conversion."""
    # Publication lag is applied to the raw period-end dates BEFORE resampling:
    # resample("YE") labels a March fiscal year as 31 Dec, which would hide results
    # for ~9 extra months.
    np_d = c.np_df[c.np_df["dt"] + c.lag <= d]
    if np_d.empty or (d - np_d["dt"].max()).days > 730:
        return None                                    # no usable / only stale results
    all_d = _annual(np_d, "sum", "net_profit")
    avail = all_d[all_d["net_profit"] > 0].tail(10)
    if avail.empty:
        return None
    oi_d  = c.oi_df[c.oi_df["dt"] + c.lag <= d]   if not c.oi_df.empty  else c.oi_df
    exc_d = c.exc_df[c.exc_df["dt"] + c.lag <= d] if not c.exc_df.empty else c.exc_df

    reported = float(avail["net_profit"].iloc[-1])
    norm, _  = dcf_core.normalize_earnings(avail, oi_d, exc_d, c.tax_r)
    if norm <= 0:
        norm = reported

    yr = avail["year"].iloc[-1]
    eps_row = c.eps_ann[c.eps_ann["year"].dt.year == yr.year]["eps"]
    if eps_row.empty or float(eps_row.iloc[0]) <= 0:
        return None
    shares_cr = reported / float(eps_row.iloc[0])      # implied shares that year (crores)
    x_earn = norm / shares_cr

    px_d = _price_at(c.prices, d)
    pe = px_d / x_earn if px_d else np.nan
    conv = (_fcf_conversion(c.fcf_df, all_d, yr.year, c.sector, c.lag, d)
            if cfg.basis == "fcf" else 1.0)
    return {"x": x_earn * conv, "pe": pe, "conv": conv, "reported": reported, "yr": yr}


def build_samples(tkr: str, cfg: FitConfig, ctx: _Ctx | None = None) -> pd.DataFrame:
    """
    One row per sample date: x (normalised cash earnings per share, ₹, as known then), target
    (price at d+h), years_fwd, g_real (realised 3-yr forward profit CAGR, %), w (recency weight).
    """
    c = ctx or _Ctx(tkr, cfg)
    if not c.ok:
        return pd.DataFrame()

    dates = pd.date_range(c.prices.index[0], c.prices.index[-1], freq=f"{cfg.step_months}ME")
    rows = []
    for d in dates:
        st = _state_at(c, d, cfg)
        if st is None:
            continue
        if not (PE_LO < st["pe"] < PE_HI):             # unadjusted split/bonus or distorted EPS
            continue
        target = _price_at(c.prices, d + pd.DateOffset(months=cfg.horizon_months)) \
            if cfg.horizon_months > 0 else _price_at(c.prices, d)
        if not target or target <= 0:
            continue

        fwd = c.np_all[c.np_all["year"].dt.year == st["yr"].year + 3]["net_profit"]
        g_real = np.nan
        if not fwd.empty and float(fwd.iloc[0]) > 0:
            g_real = ((float(fwd.iloc[0]) / st["reported"]) ** (1 / 3) - 1) * 100

        rows.append({"date": d, "x": st["x"], "target": target, "pe": st["pe"],
                     "years_fwd": cfg.horizon_months / 12, "g_real": g_real})

    s = pd.DataFrame(rows)
    if s.empty:
        return s
    age_years = (s["date"].max() - s["date"]).dt.days / 365.25
    w = 0.5 ** (age_years / cfg.half_life_years)
    s["w"] = w / w.sum()
    return s


def _auto_prior(tkr: str) -> dict | None:
    """Precomputed assumptions from dcf.sql (CAPM dr, shrunk weighted growth, sustainable cap)."""
    try:
        df = run_readonly_query("""
            SELECT g1, g2, tg, dr FROM v_dcf_assumptions_auto
            WHERE UPPER(TRIM(ticker)) = ? LIMIT 1
        """, [tkr])
    except Exception:
        return None
    if df.empty or df.iloc[0].isna().any():
        return None
    r = df.iloc[0]
    g1 = float(np.clip(r["g1"], 0, G1_MAX))
    g2 = float(np.clip(r["g2"], 0, min(g1, G2_MAX)))
    tg = float(np.clip(r["tg"], TG_LO, TG_HI))
    dr = float(np.clip(max(r["dr"], tg + DR_MIN_SPREAD), DR_LO, DR_HI))
    return {"g1": g1, "g2": g2, "tg": tg, "dr": dr}


def prior_for(tkr: str) -> dict:
    """Prior: v_dcf_assumptions_auto if available, else reported 3Y / 5Y profit growth, 5% terminal, 12% WACC."""
    auto = _auto_prior(tkr)
    if auto:
        return auto
    growth = dcf_core.get_growth_rates(tkr)
    g3 = growth.get("3 Years", growth.get("3Y"))
    g5 = growth.get("5 Years", growth.get("5Y"))
    g1 = float(np.clip(g3 if g3 is not None else DEFAULT_PRIOR["g1"], 0, G1_MAX))
    g2 = float(np.clip(g5 if g5 is not None else DEFAULT_PRIOR["g2"], 0, min(g1, G2_MAX)))
    return {"g1": g1, "g2": g2, "tg": DEFAULT_PRIOR["tg"], "dr": DEFAULT_PRIOR["dr"]}


# ══════════════════════════════════════════════════════════════════════════════
# FIT
# ══════════════════════════════════════════════════════════════════════════════

def _snap(v: float, step: float, lo: float, hi: float) -> float:
    return float(min(max(round(v / step) * step, lo), hi))


def _grid_ok(g1: float, g2: float, tg: float, dr: float) -> bool:
    return (0.0 <= g1 <= G1_MAX and 0.0 <= g2 <= min(g1, G2_MAX)
            and TG_LO <= tg <= TG_HI and DR_LO <= dr <= DR_HI and dr >= tg + DR_MIN_SPREAD)


def _project_to_grid(theta, s: pd.DataFrame, prior: dict, cfg: FitConfig,
                     free: tuple[int, ...] = (0, 1, 2, 3)) -> tuple[float, float, float, float]:
    """
    The page's sliders move in 0.5 steps, so store values on that grid. Rather than just
    rounding, try every point within one grid step of the optimum (3^k, k = free dims) and keep
    the one with the lowest full loss. Dimensions not in `free` stay at their snapped value.
    """
    base = [_snap(v, 0.5, 0.0, 100.0) for v in theta]
    steps = [(-0.5, 0.0, 0.5) if i in free else (0.0,) for i in range(4)]
    best, best_l = None, np.inf
    for a in steps[0]:
        for b in steps[1]:
            for c in steps[2]:
                for d in steps[3]:
                    cand = (base[0] + a, base[1] + b, base[2] + c, base[3] + d)
                    if not _grid_ok(*cand):
                        continue
                    l = loss(cand, s, prior, cfg)
                    if l < best_l:
                        best, best_l = cand, l
    return best if best is not None else tuple(base)


def fit_assumptions(samples: pd.DataFrame, prior: dict, cfg: FitConfig) -> dict:
    """Run Adam from the prior and return assumptions plus fit diagnostics."""
    free = (0, 1) if cfg.fit == "growth" else (0, 1, 2, 3)
    if cfg.fit == "growth":
        # dr / tg are not fitted: put them on the slider grid once so prior == stored values
        tg0 = _snap(prior["tg"], 0.5, TG_LO, TG_HI)
        dr0 = _snap(max(prior["dr"], tg0 + DR_MIN_SPREAD), 0.5, DR_LO, DR_HI)
        prior = {**prior, "tg": tg0, "dr": dr0}
    u0 = u_from_theta(prior["g1"], prior["g2"], prior["tg"], prior["dr"])

    def fn(uf):
        u = u0.copy()
        u[list(free)] = uf
        return loss(theta_from_u(u), samples, prior, cfg)

    uf_best, history = adam_minimize(fn, u0[list(free)], iters=cfg.iters, lr=cfg.lr)
    u_best = u0.copy()
    u_best[list(free)] = uf_best

    w = samples["w"].values
    rmse = lambda th: float(np.sqrt(np.sum(w * residuals(th, samples) ** 2)))
    g1, g2, tg, dr = _project_to_grid(theta_from_u(u_best), samples, prior, cfg, free)
    if cfg.fit == "growth":
        tg, dr = prior["tg"], prior["dr"]
    rmse_fit   = rmse((g1, g2, tg, dr))                   # of the values actually stored
    rmse_prior = rmse(theta_from_u(u0))                   # baseline: just using the prior
    mos = int(_snap(100 * (1 - np.exp(-rmse_fit)), 5, MOS_LO, MOS_HI))

    return {"g1": g1, "g2": g2, "tg": tg, "dr": dr, "mos": mos,
            "rmse_fit": round(rmse_fit, 4), "rmse_prior": round(rmse_prior, 4),
            "n_samples": int(len(samples)), "iterations": len(history) - 1,
            "loss_start": history[0], "loss_end": min(history)}


def optimize_ticker(tkr: str, cfg: FitConfig) -> dict | None:
    ctx = _Ctx(tkr, cfg)
    samples = build_samples(tkr, cfg, ctx)
    if len(samples) < cfg.min_samples:
        return None
    res = fit_assumptions(samples, prior_for(tkr), cfg)

    # Diagnostic: what the fitted assumptions say about *today's* price (same maths as dcf.sql).
    d_now = ctx.prices.index[-1]
    st = _state_at(ctx, d_now, cfg)
    price_latest = float(ctx.prices.iloc[-1])
    iv_latest = gap = conv = np.nan
    if st is not None:
        f = dcf_core.dcf_multiple(res["g1"], res["g2"], res["tg"], res["dr"])
        iv_latest = st["x"] * f
        gap = (price_latest / iv_latest - 1) * 100 if iv_latest > 0 else np.nan
        conv = st["conv"]
    res.update(ticker=tkr, horizon_months=cfg.horizon_months, basis=cfg.basis, fit_mode=cfg.fit,
               fitted_on=dt.date.today().isoformat(),
               fcf_conv=round(conv, 3) if pd.notna(conv) else np.nan,
               price_latest=round(price_latest, 2),
               iv_latest=round(iv_latest, 2) if pd.notna(iv_latest) else np.nan,
               gap_pct=round(gap, 1) if pd.notna(gap) else np.nan)
    return res


# ══════════════════════════════════════════════════════════════════════════════
# STORAGE + CLI
# ══════════════════════════════════════════════════════════════════════════════

def _f(v) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def score_confidence(table: pd.DataFrame) -> pd.DataFrame:
    """Add `confidence` / `confidence_reason` to every row (relative test uses the whole table)."""
    t = table.copy()
    rm = pd.to_numeric(t.get("rmse_fit"), errors="coerce")
    med = rm.median()
    mad = (rm - med).abs().median() * 1.4826
    z = (rm - med) / mad if (rm.notna().sum() >= CONF_Z_MIN_N and mad and mad > 0) \
        else pd.Series(0.0, index=t.index)

    levels, reasons = [], []
    for i, r in t.iterrows():
        low, medm = [], []
        rmse, rprior = _f(r.get("rmse_fit")), _f(r.get("rmse_prior"))
        n, g1, gap, conv = _f(r.get("n_samples")), _f(r.get("g1")), _f(r.get("gap_pct")), _f(r.get("fcf_conv"))
        if np.isnan(rmse):
            low.append("no fit error")
        elif rmse > CONF_RMSE_LOW:
            low.append(f"poor fit (rmse {rmse:.2f})")
        elif rmse > CONF_RMSE_MEDIUM:
            medm.append(f"weak fit (rmse {rmse:.2f})")
        if not np.isnan(z.get(i, np.nan)) and z.get(i, 0) > CONF_Z_LOW:
            low.append(f"outlier fit vs universe (z {z[i]:.1f})")
        if not np.isnan(n):
            if n < CONF_N_LOW:
                low.append(f"few samples ({int(n)})")
            elif n < CONF_N_MEDIUM:
                medm.append(f"limited samples ({int(n)})")
        if not np.isnan(g1) and g1 >= CONF_G_CAP:
            (low if (not np.isnan(rmse) and rmse > CONF_RMSE_MEDIUM) else medm).append(f"growth at cap (g1 {g1:g})")
        if not np.isnan(gap):
            if abs(gap) > CONF_GAP_LOW:
                low.append(f"price {gap:+.0f}% vs DCF")
            elif abs(gap) > CONF_GAP_MEDIUM:
                medm.append(f"price {gap:+.0f}% vs DCF")
        if not np.isnan(rmse) and not np.isnan(rprior) and rprior > 0 and rmse > 0.9 * rprior:
            medm.append("fit barely beats prior")
        if not np.isnan(conv) and (conv <= FCF_CONV_LO + 1e-9 or conv >= FCF_CONV_HI - 1e-9):
            medm.append("FCF conversion clipped")
        levels.append("low" if low else "medium" if medm else "high")
        reasons.append("; ".join(low + medm))
    t["confidence"] = levels
    t["confidence_reason"] = reasons
    return t


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop schema placeholder / blank rows and make sure every output column exists."""
    df = df.copy()
    df["ticker"] = df["ticker"].fillna("").astype(str).str.strip().str.upper()
    df = df[(df["ticker"] != "") & (~df["ticker"].str.startswith("__"))]
    return df.reindex(columns=OUT_COLUMNS)


def _write(table: pd.DataFrame, p) -> pd.DataFrame:
    table = score_confidence(_clean(table)).sort_values("ticker").reset_index(drop=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(p, index=False)
    return table


def save_results(results: list[dict], path: str) -> pd.DataFrame:
    """Upsert `results` into the CSV table at `path` (one row per ticker); re-rate confidence."""
    from pathlib import Path
    new = pd.DataFrame(results).reindex(columns=OUT_COLUMNS)
    p = Path(path)
    if p.exists():
        old = _clean(pd.read_csv(p))
        old = old[~old["ticker"].isin(new["ticker"])]
        new = pd.concat([old, new], ignore_index=True)
    return _write(new, p)


def rescore(path: str) -> pd.DataFrame:
    """Re-rate confidence for the CSV at `path` without refitting."""
    from pathlib import Path
    return _write(pd.read_csv(path), Path(path))


def _confidence_summary(table: pd.DataFrame) -> str:
    c = table["confidence"].value_counts()
    return "confidence: " + ", ".join(f"{k}={int(c.get(k, 0))}" for k in ("high", "medium", "low"))


def _universe() -> list[str]:
    return run_readonly_query("SELECT DISTINCT ticker FROM gate_score ORDER BY ticker")["ticker"].tolist()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fit per-ticker DCF assumptions with Adam.")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--rescore", action="store_true",
                     help="re-rate confidence for the existing --out CSV without refitting")
    grp.add_argument("--tickers", help="comma-separated tickers, e.g. RELIANCE,INFY")
    grp.add_argument("--all", action="store_true", help="every ticker in the gate_score view")
    ap.add_argument("--out", default="config/dcf_optimized.csv")
    d = FitConfig()
    ap.add_argument("--horizon-months", type=int, default=d.horizon_months)
    ap.add_argument("--step-months", type=int, default=d.step_months)
    ap.add_argument("--lag-days", type=int, default=d.lag_days)
    ap.add_argument("--half-life-years", type=float, default=d.half_life_years)
    ap.add_argument("--w-growth", type=float, default=d.w_growth)
    ap.add_argument("--w-prior", type=float, default=d.w_prior)
    ap.add_argument("--iters", type=int, default=d.iters)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--min-samples", type=int, default=d.min_samples)
    ap.add_argument("--fit", choices=("growth", "all"), default=d.fit,
                    help="growth = fit g1,g2 only (dr/tg from the auto view); all = fit g1,g2,tg,dr")
    ap.add_argument("--basis", choices=("fcf", "profit"), default=d.basis,
                    help="fcf = match dcf.sql (profit x fcf_conversion); profit = match the Streamlit page")
    a = ap.parse_args(argv)
    if a.rescore:
        table = rescore(a.out)
        print(f"Re-rated {len(table)} ticker(s) in {a.out}. {_confidence_summary(table)}")
        return

    cfg = FitConfig(a.horizon_months, a.step_months, a.lag_days, a.half_life_years,
                    a.w_growth, a.w_prior, a.iters, a.lr, a.min_samples, a.basis, a.fit)
    tickers = _universe() if a.all else [t.strip().upper().removesuffix(".NS")
                                         for t in a.tickers.split(",") if t.strip()]
    results, skipped = [], []
    for i, tkr in enumerate(tickers, 1):
        try:
            res = optimize_ticker(tkr, cfg)
        except Exception as exc:                       # one bad ticker must not stop the batch
            print(f"[{i}/{len(tickers)}] {tkr}: ERROR {exc}")
            skipped.append(tkr)
            continue
        if res is None:
            print(f"[{i}/{len(tickers)}] {tkr}: skipped (fewer than {cfg.min_samples} usable samples)")
            skipped.append(tkr)
            continue
        results.append(res)
        print(f"[{i}/{len(tickers)}] {tkr}: g1={res['g1']} g2={res['g2']} tg={res['tg']} "
              f"dr={res['dr']} mos={res['mos']}  rmse {res['rmse_prior']:.3f} -> {res['rmse_fit']:.3f} "
              f"(n={res['n_samples']})  price {res['price_latest']} vs DCF {res['iv_latest']} ({res['gap_pct']}%)")

    if results:
        table = save_results(results, a.out)
        print(f"\nSaved {len(results)} ticker(s) to {a.out} ({len(table)} rows total). "
              f"{_confidence_summary(table)}")
    print(f"Skipped: {len(skipped)}" + (f" -> {', '.join(skipped)}" if skipped else ""))


if __name__ == "__main__":
    main()