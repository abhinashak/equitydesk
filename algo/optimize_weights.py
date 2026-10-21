import duckdb
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# 1. Define the asset universe
tickers = [
    'TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'VAML', 'VEDL', 
    'JINDALSTEL', 'NATIONALUM', 'HINDZINC', 'NMDC', 
    'LLOYDSME', 'HINDCOPPER', 'GOLDBEES', 'SILVERBEES','NIFTYBEES'
]

# 2. Connect to DuckDB and initialize views using init.sql
conn = duckdb.connect()
try:
    with open("init.sql", "r") as f:
        conn.execute(f.read())
except FileNotFoundError:
    with open("sqls/init.sql", "r") as f:
        conn.execute(f.read())

query = f"""
    SELECT Date, nse_symbol AS ticker, Close
    FROM ticker_prices
    WHERE nse_symbol IN ({str(tickers)[1:-1]})
      AND Date >= '2013-01-01'
    UNION
    SELECT Date, nse_symbol AS ticker, Close
    FROM benchmark_prices WHERE nse_symbol IN ({str(tickers)[1:-1]})
    AND Date >= '2023-01-01'
    ORDER BY Date ASC
"""

df = conn.execute(query).fetchdf()
conn.close()

# 3. Pivot data & compute daily returns
pivot_df = df.pivot(index='Date', columns='ticker', values='Close').dropna()
prices = pivot_df[tickers].values
returns = np.diff(prices, axis=0) / prices[:-1]

R = torch.tensor(returns, dtype=torch.float32)
T, N = R.shape

# Equal weight benchmark returns & cumulative benchmark
equal_weights = torch.ones(N, dtype=torch.float32) / N
benchmark_returns = R @ equal_weights

# 4. Initialize weights for Adam Optimizer
weights_logit = torch.zeros(N, requires_grad=True)
optimizer = optim.Adam([weights_logit], lr=0.05)

# 5. Optimization Loop with Diversification & Outperformance Objective
epochs = 1500
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Apply softmax for base weights, then cap max weight via scaled mixing or penalty
    w = torch.softmax(weights_logit, dim=0)
    
    # Portfolio returns
    port_returns = R @ w
    excess_returns = port_returns - benchmark_returns
    
    # Objective 1: Maximize mean excess return over equal weight
    mean_excess = torch.mean(excess_returns)
    
    # Objective 2: Reward consistency (percentage of days beating benchmark via smooth approximation)
    win_rate_proxy = torch.mean(torch.sigmoid(excess_returns * 50))
    
    # Penalty 3: Herfindahl-Hirschman Index (HHI) concentration penalty to prevent 99% corner solutions
    hhi = torch.sum(w ** 2)
    
    # Loss: Minimize negative excess return, penalize concentration (hhi), encourage win rate
    loss = -mean_excess * 20.0 - win_rate_proxy * 5.0 + hhi * 2.0
    
    loss.backward()
    optimizer.step()

# 6. Extract Final Optimized Weights & apply hard cap (e.g., max 25% per asset for safety)
raw_weights = torch.softmax(weights_logit, dim=0).detach().numpy()
capped_weights = np.minimum(raw_weights, 0.25)  # Cap individual asset weight at 25%
final_weights = capped_weights / np.sum(capped_weights) # Re-normalize to sum to 1

result_df = pd.DataFrame({'ticker': tickers, 'optimized_weight': final_weights})
result_df['optimized_weight_pct'] = (result_df['optimized_weight'] * 100).round(2)
result_df = result_df.sort_values(by='optimized_weight_pct', ascending=False)

print("\n--- Balanced & Constrained Weight Distribution ---")
print(result_df[['ticker', 'optimized_weight_pct']].to_string(index=False))
