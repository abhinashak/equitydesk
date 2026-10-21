
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

TICKERS = [
    "COALINDIA.NS", "ONGC.NS", "PFC.NS", "SBIN.NS",
    "POWERGRID.NS", "HINDZINC.NS", "HINDCOPPER.NS",
    "JSWSTEEL.NS", "JINDALSTEL.NS", "TATASTEEL.NS",
    "HINDALCO.NS", "NATIONALUM.NS", "VEDL.NS", "NMDC.NS"
]

START = "2001-01-01"
END = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

INITIAL_CAPITAL = 5.14e7   # ₹5.14 crore; change as required
MIN_ANNUAL_DIVIDEND = 12e5 # ₹12 lakh gross annual dividends
TRAINING_YEARS = 3
REBALANCE_MONTHS = 12

MAX_WEIGHT = 0.20
RISK_AVERSION = 3.0
TURNOVER_PENALTY = 0.001
ADAM_STEPS = 2000
LEARNING_RATE = 0.02

torch.set_default_dtype(torch.float64)
DEVICE = "cpu"

# ============================================================
# DOWNLOAD PRICES AND DIVIDENDS
# ============================================================

raw = yf.download(
    TICKERS,
    start=START,
    end=END,
    auto_adjust=False,
    actions=True,
    progress=False,
    group_by="column"
)

def extract_field(field):
    x = raw[field]
    if isinstance(x, pd.Series):
        x = x.to_frame()
    x.columns = [
        c if str(c).endswith(".NS") else f"{c}.NS"
        for c in x.columns
    ]
    return x

close = extract_field("Close").reindex(columns=TICKERS)
adj_close = extract_field("Adj Close").reindex(columns=TICKERS)

dividends = extract_field("Dividends").reindex(
    columns=TICKERS
).fillna(0.0)

# Keep stocks with sufficient price history.
valid = [
    t for t in TICKERS
    if close[t].notna().sum() > 500
]
close = close[valid].ffill()
adj_close = adj_close[valid].ffill()
dividends = dividends[valid].fillna(0.0)

# Total returns include dividends through adjusted prices.
returns = adj_close.pct_change(fill_method=None)
returns = returns.replace([np.inf, -np.inf], np.nan)

# Annual cash dividend per share, using trailing 365 calendar days.
trailing_dividends = dividends.rolling("365D").sum()
dividend_yields = trailing_dividends / close

# ============================================================
# ADAM OPTIMIZER
# ============================================================

def optimise_weights(train_returns, yields, capital, previous=None):
    """
    Maximise expected log-growth approximation, penalising
    variance and turnover, subject to portfolio constraints.

    Dividend constraint is a soft penalty. Always inspect
    feasibility and verify the final dividend estimate.
    """
    mu = train_returns.mean().values * 252
    cov = train_returns.cov().values * 252
    y = np.nan_to_num(yields.values, nan=0.0)

    n = len(mu)
    mu_t = torch.tensor(mu)
    cov_t = torch.tensor(cov)
    y_t = torch.tensor(y)

    previous = (
        np.ones(n) / n if previous is None
        else np.asarray(previous)
    )
    prev_t = torch.tensor(previous)

    # Softmax ensures weights are non-negative and sum to 1.
    logits = torch.nn.Parameter(torch.zeros(n))
    optimizer = torch.optim.Adam([logits], lr=LEARNING_RATE)

    # Scaling the shortfall makes the constraint numerically useful.
    target_yield = MIN_ANNUAL_DIVIDEND / max(capital, 1.0)

    for _ in range(ADAM_STEPS):
        optimizer.zero_grad()
        w = torch.softmax(logits, dim=0)

        expected_return = torch.dot(w, mu_t)
        variance = w @ cov_t @ w
        turnover = torch.abs(w - prev_t).sum()

        estimated_income = capital * torch.dot(w, y_t)
        shortfall = torch.relu(
            torch.tensor(MIN_ANNUAL_DIVIDEND) - estimated_income
        ) / max(capital, 1.0)

        # Reward growth; penalise risk, trading and dividend shortfall.
        loss = (
            -expected_return
            + RISK_AVERSION * variance
            + TURNOVER_PENALTY * turnover
            + 100.0 * shortfall**2
        )

        loss.backward()
        optimizer.step()

    w = torch.softmax(logits, dim=0).detach().numpy()

    # Enforce maximum position weight by capped-simplex projection.
    # Bisection finds a shift whose clipped weights sum to one.
    lo, hi = -1.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        projected = np.clip(w - mid, 0, MAX_WEIGHT)
        if projected.sum() > 1:
            lo = mid
        else:
            hi = mid

    w = np.clip(w - (lo + hi) / 2, 0, MAX_WEIGHT)
    w /= w.sum()

    income = capital * np.dot(w, y)
    return w, income

# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================

dates = returns.index
portfolio_value = INITIAL_CAPITAL
weights = np.ones(len(valid)) / len(valid)
history = []
weight_history = []

# Start after enough historical data is available.
first_date = pd.Timestamp(START) + pd.DateOffset(
    years=TRAINING_YEARS
)
rebalance_dates = dates[dates >= first_date]

last_rebalance = None

for date in rebalance_dates:
    if last_rebalance is not None:
        if (date - last_rebalance).days < 350:
            continue

    train_start = date - pd.DateOffset(years=TRAINING_YEARS)
    train = returns.loc[train_start:date].iloc[:-1].dropna()

    if len(train) < 252:
        continue

    y = dividend_yields.loc[:date].iloc[-1]
    weights, estimated_income = optimise_weights(
        train,
        y,
        portfolio_value,
        previous=weights
    )

    # Hold the weights until the next annual rebalance.
    next_dates = dates[dates > date]
    if len(next_dates) == 0:
        break

    next_date = next_dates[
        min(251, len(next_dates) - 1)
    ]

    period_returns = returns.loc[
        (returns.index > date) &
        (returns.index <= next_date), valid
    ].fillna(0.0)

    # Daily portfolio total return, including reinvested dividends.
    daily_portfolio_returns = period_returns.values @ weights

    for dt, r in zip(period_returns.index, daily_portfolio_returns):
        portfolio_value *= (1 + r)
        history.append({
            "date": dt,
            "portfolio_value": portfolio_value
        })

    weight_history.append(
        pd.Series(weights, index=valid, name=date)
    )
    last_rebalance = date

# ============================================================
# RESULTS
# ============================================================

equity_curve = pd.DataFrame(history).drop_duplicates("date")
equity_curve = equity_curve.set_index("date")

if not equity_curve.empty:
    total_return = (
        equity_curve["portfolio_value"].iloc[-1]
        / INITIAL_CAPITAL - 1
    )
    years = (
        equity_curve.index[-1] - equity_curve.index[0]
    ).days / 365.25
    cagr = (
        (equity_curve["portfolio_value"].iloc[-1]
         / INITIAL_CAPITAL) ** (1 / years) - 1
        if years > 0 else np.nan
    )

    daily_ret = equity_curve["portfolio_value"].pct_change().dropna()
    max_drawdown = (
        equity_curve["portfolio_value"]
        / equity_curve["portfolio_value"].cummax() - 1
    ).min()

    print(f"Initial capital: ₹{INITIAL_CAPITAL:,.0f}")
    print(f"Final value: ₹{equity_curve['portfolio_value'].iloc[-1]:,.0f}")
    print(f"Total return: {total_return:.2%}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Max drawdown: {max_drawdown:.2%}")

    if weight_history:
        print("\nLatest optimised weights:")
        print(weight_history[-1].sort_values(ascending=False).round(4))

    equity_curve["portfolio_value"].plot(
        title="Dividend-Constrained Adam Portfolio (Total Return)"
    )
    plt.ylabel("Portfolio value (₹)")
    plt.tight_layout()
    plt.show()
else:
    print("Insufficient valid history to run the backtest.")


# ============================================================
# YEARLY PORTFOLIO vs NIFTYBEES + DIVIDEND YIELD
# ============================================================

# Download benchmark with dividends included in adjusted prices
benchmark = yf.download(
    "NIFTYBEES.NS",
    start=START,
    end=END,
    auto_adjust=False,
    progress=False
)

bench_adj = benchmark["Adj Close"]
if isinstance(bench_adj, pd.DataFrame):
    bench_adj = bench_adj.iloc[:, 0]

bench_adj = bench_adj.dropna()

# Year-end portfolio value
portfolio_year_end = (
    equity_curve["portfolio_value"]
    .resample("YE")
    .last()
    .dropna()
)

# Align benchmark to the portfolio's actual backtest start
common_start = equity_curve.index.min()
bench_adj = bench_adj.loc[bench_adj.index >= common_start]

benchmark_value = (
    bench_adj / bench_adj.iloc[0] * INITIAL_CAPITAL
).resample("YE").last().dropna()

# Annual dividend yield for the stocks
# Calendar-year dividends per share / average share price.
annual_dividends = dividends.resample("YE").sum()
annual_average_price = close.resample("YE").mean()

annual_stock_yields = (
    annual_dividends / annual_average_price
).replace([np.inf, -np.inf], np.nan).fillna(0)

rows = []

for date, portfolio_val in portfolio_year_end.items():
    year = date.year

    # Latest weights known at or before this year-end.
    eligible_weights = [
        w for d, w in zip(
            [pd.Timestamp(d) for d in
             [s.name for s in weight_history]],
            weight_history
        )
        if d <= date
    ]

    if eligible_weights:
        w = eligible_weights[-1].reindex(valid).fillna(0)
    else:
        w = pd.Series(1 / len(valid), index=valid)

    # Use that year's actual dividend-per-share data.
    year_end_date = annual_stock_yields.index[
        annual_stock_yields.index.year == year
    ]

    if len(year_end_date):
        yields_for_year = annual_stock_yields.loc[
            year_end_date[-1]
        ].reindex(valid).fillna(0)

        portfolio_yield = float(
            (w * yields_for_year).sum()
        )
    else:
        portfolio_yield = np.nan

    # Estimated dividend income based on year-end portfolio value.
    estimated_dividend = portfolio_val * portfolio_yield

    # Benchmark value for the same calendar year
    bench_years = benchmark_value[
        benchmark_value.index.year == year
    ]
    nifty_value = (
        float(bench_years.iloc[-1])
        if len(bench_years) else np.nan
    )

    rows.append({
        "Year": year,
        "Adam Portfolio (₹)": portfolio_val,
        "NIFTYBEES (₹)": nifty_value,
        "Outperformance (₹)": (
            portfolio_val - nifty_value
            if pd.notna(nifty_value) else np.nan
        ),
        "Dividend Yield (%)": portfolio_yield * 100,
        "Est. Annual Dividend (₹)": estimated_dividend
    })

yearly_comparison = pd.DataFrame(rows).set_index("Year")

print("\nYEARLY PORTFOLIO COMPARISON")
print(
    yearly_comparison.to_string(
        formatters={
            "Adam Portfolio (₹)": "{:,.0f}".format,
            "NIFTYBEES (₹)": lambda x: (
                f"{x:,.0f}" if pd.notna(x) else "N/A"
            ),
            "Outperformance (₹)": lambda x: (
                f"{x:,.0f}" if pd.notna(x) else "N/A"
            ),
            "Dividend Yield (%)": "{:.2f}%".format,
            "Est. Annual Dividend (₹)": "{:,.0f}".format
        }
    )
)

yearly_comparison.to_csv("yearly_adam_vs_niftybees.csv")

# Plot both portfolio values
yearly_comparison[
    ["Adam Portfolio (₹)", "NIFTYBEES (₹)"]
].plot(
    kind="line",
    marker="o",
    title="Adam Dividend Portfolio vs NIFTYBEES"
)

plt.ylabel("Portfolio value (₹)")
plt.xlabel("Year")
plt.grid(True)
plt.tight_layout()
plt.show()



# ============================================================
# YEARLY COMPARISON — RESET TO ₹5 CRORE EACH YEAR
# ============================================================

BASE_CAPITAL = 5e7  # ₹5 crore

rows = []

for year in sorted(returns.index.year.unique()):
    year_start = pd.Timestamp(f"{year}-01-01")
    year_end = pd.Timestamp(f"{year}-12-31")

    # Data available within the calendar year
    year_returns = returns.loc[
        (returns.index >= year_start) &
        (returns.index <= year_end)
    ].dropna()

    if year_returns.empty:
        continue

    # Adam weights selected using only data before the year starts.
    train_start = year_start - pd.DateOffset(years=TRAINING_YEARS)
    train = returns.loc[
        (returns.index >= train_start) &
        (returns.index < year_start)
    ].dropna()

    if len(train) >= 252:
        y = dividend_yields.loc[
            dividend_yields.index < year_start
        ].iloc[-1]

        year_weights, _ = optimise_weights(
            train,
            y,
            BASE_CAPITAL,
            previous=np.ones(len(valid)) / len(valid)
        )
    else:
        # Insufficient prior history: use equal weights.
        year_weights = np.ones(len(valid)) / len(valid)

    # Reset both portfolios to ₹5 crore at the start of the year.
    daily_adam_returns = year_returns[valid].values @ year_weights

    adam_value = BASE_CAPITAL * np.prod(1 + daily_adam_returns)

    # NIFTYBEES: buy-and-hold total return for the same calendar year.
    b = bench_adj.loc[
        (bench_adj.index >= year_start) &
        (bench_adj.index <= year_end)
    ].dropna()

    if len(b) < 2:
        continue

    nifty_value = BASE_CAPITAL * (b.iloc[-1] / b.iloc[0])

    # Trailing dividend yield available at the beginning of the year.
    prior_yields = dividend_yields.loc[
        dividend_yields.index < year_start
    ]

    if len(prior_yields):
        annual_yield = float(
            np.dot(year_weights, prior_yields.iloc[-1][valid].fillna(0))
        )
    else:
        annual_yield = np.nan

    estimated_dividend = (
        BASE_CAPITAL * annual_yield
        if pd.notna(annual_yield) else np.nan
    )

    rows.append({
        "Year": year,
        "Starting Capital (₹)": BASE_CAPITAL,
        "Adam Year-End Value (₹)": adam_value,
        "NIFTYBEES Year-End Value (₹)": nifty_value,
        "Adam Profit/Loss (₹)": adam_value - BASE_CAPITAL,
        "NIFTYBEES Profit/Loss (₹)": nifty_value - BASE_CAPITAL,
        "Outperformance (₹)": adam_value - nifty_value,
        "Dividend Yield (%)": annual_yield * 100,
        "Estimated Annual Dividend (₹)": estimated_dividend
    })

yearly_reset = pd.DataFrame(rows).set_index("Year")

print("\nYEARLY COMPARISON — ₹5 CRORE RESET EACH YEAR")
print(yearly_reset.to_string(
    formatters={
        c: "{:,.0f}".format
        for c in [
            "Starting Capital (₹)",
            "Adam Year-End Value (₹)",
            "NIFTYBEES Year-End Value (₹)",
            "Adam Profit/Loss (₹)",
            "NIFTYBEES Profit/Loss (₹)",
            "Outperformance (₹)",
            "Estimated Annual Dividend (₹)"
        ]
    } | {
        "Dividend Yield (%)": lambda x:
            f"{x:.2f}%" if pd.notna(x) else "N/A"
    }
))

yearly_reset.to_csv("yearly_reset_5cr_comparison.csv")

