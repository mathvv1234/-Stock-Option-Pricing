import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

# Parameters
ticker = "SAP.DE"  # Stock ticker
start_date = "2023-04-30"  # Start of historical data
end_date = "2025-01-18"  # End of historical data (today's date)
num_simulations = 100  # Number of price paths
max_retries = 3  # Retry attempts for rate limit
retry_delay = 10  # Seconds to wait between retries

# Fetch historical stock data with retry mechanism
df = None
for attempt in range(max_retries):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        if df.empty:
            raise ValueError(f"No data retrieved for {ticker}")
        break  # Success, exit retry loop
    except Exception as e:
        if "Too Many Requests" in str(e):
            print(f"Rate limit hit. Retrying ({attempt + 1}/{max_retries}) after {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            print(f"Error fetching data: {e}")
            exit(1)
if df is None:
    print(f"Failed to fetch data for {ticker} after {max_retries} attempts.")
    exit(1)

# Calculate daily returns
prices = df['Close']
returns = prices.pct_change().dropna()

# Calculate number of trading days in historical data
num_days = len(prices)
dt = 1 / num_days  # Time step

# Calculate drift (mean return) and volatility (std of returns)
mu = returns.mean() * 252  # Annualized drift (assuming 252 trading days per year)
sigma = returns.std() * np.sqrt(252)  # Annualized volatility
S0 = prices[0]  # First closing price in historical data

# Simulate stock prices using Geometric Brownian Motion
np.random.seed(18)  # For reproducibility
simulations = np.zeros((num_days, num_simulations))
simulations[0] = S0

for t in range(1, num_days):
    # Generate random increments (Wiener process)
    Z = np.random.standard_normal(num_simulations)
    # GBM formula
    simulations[t] = simulations[t-1] * np.exp(
        (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    )

# Create time index for plotting (use historical data's dates)
dates = df.index

# Plot actual stock prices and simulated price paths
plt.figure(figsize=(10, 6))

# Plot simulated price paths
for i in range(num_simulations):
    plt.plot(dates, simulations[:, i], label='Simulated Stock Prices' if i == 0 else None, linewidth = 1, color='gray', alpha=0.5)

# Plot actual prices
plt.plot(dates, prices, label='Actual Stock Price', color='black', linewidth=2)

plt.title(f"{ticker} Actual vs. Simulated Stock Prices (Apr 2023 - Jan 2025)")
plt.xlabel("Date")
plt.ylabel("Stock Price ($)")
plt.grid(True)
plt.legend()
plt.savefig('stock_actual_vs_simulated.png')
plt.show()

print(f"Simulation completed. Plot saved as 'stock_actual_vs_simulated.png'")
print(f"Drift (μ): {mu:.4f}, Volatility (σ): {sigma:.4f}, Initial Price (S0): {S0:.2f}")
