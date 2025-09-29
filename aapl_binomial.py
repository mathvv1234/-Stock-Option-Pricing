import yfinance as yf
import numpy as np
import datetime
import math # Import math module

def binomial_american(S, K, T, r, sigma, N, option_type='call'):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    tree = np.zeros((N + 1, N + 1))

    # Terminal values
    for j in range(N + 1):
        ST = S * (u ** (N - j)) * (d ** j)
        tree[j, N] = max(0, ST - K if option_type == 'call' else K - ST)

    # Backward induction
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            ST = S * (u ** (i - j)) * (d ** j)
            exercise = max(0, ST - K if option_type == 'call' else K - ST)
            hold = np.exp(-r * dt) * (p * tree[j, i + 1] + (1 - p) * tree[j + 1, i + 1])
            tree[j, i] = max(exercise, hold)

    return tree[0, 0]

# To approximate prob exercise, we can use the European equivalent from BS, or simulate paths, but here approximate with sum of probabilities where exercised at end
def approx_prob_exercise_binomial(S, K, T, r, sigma, N, option_type='call'):  # Approximate as European prob
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    prob = 0
    for j in range(N + 1):
        binom_coeff = math.comb(N, j) # Use math.comb
        path_prob = binom_coeff * (p ** (N - j)) * ((1 - p) ** j)
        ST = S * (u ** (N - j)) * (d ** j)
        if (option_type == 'call' and ST > K) or (option_type == 'put' and ST < K):
            prob += path_prob
    return prob

# Parameters
ticker = 'AAPL'
# Ensure S is a scalar float
S = float(yf.Ticker(ticker).info['currentPrice'])
r = 0.0385  # Risk-free rate
N = 100  # Steps
expiration = '2025-08-15'

# Volatility
data = yf.download(ticker, period='1y')
returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
sigma = returns.std() * np.sqrt(252)
# Ensure sigma is a scalar float
sigma = float(sigma.iloc[0])

# Time to expiration
today = datetime.date.today()
exp_date = datetime.date(2025, 8, 15)
T = (exp_date - today).days / 365.0

# Option chain
chain = yf.Ticker(ticker).option_chain(expiration)
calls = chain.calls
puts = chain.puts

# Compare prices and calculate weighted exercise percent
total_oi = 0
weighted_prob = 0

print("Calls:")
for _, row in calls.iterrows():
    K = row['strike']
    bid = row['bid']
    ask = row['ask']
    mid = (bid + ask) / 2
    oi = row['openInterest']
    bin_price = binomial_american(S, K, T, r, sigma, N, 'call')
    prob = approx_prob_exercise_binomial(S, K, T, r, sigma, N, 'call')  # Approx prob
    print(f"Strike: {K}, Market Mid: {mid:.2f}, Binomial Price: {bin_price:.2f}")
    if oi > 0:
        weighted_prob += prob * oi
        total_oi += oi

print("\nPuts:")
for _, row in puts.iterrows():
    K = row['strike']
    bid = row['bid']
    ask = row['ask']
    mid = (bid + ask) / 2
    oi = row['openInterest']
    bin_price = binomial_american(S, K, T, r, sigma, N, 'put')
    prob = approx_prob_exercise_binomial(S, K, T, r, sigma, N, 'put')
    print(f"Strike: {K}, Market Mid: {mid:.2f}, Binomial Price: {bin_price:.2f}")
    if oi > 0:
        weighted_prob += prob * oi
        total_oi += oi

percent_exercised = (weighted_prob / total_oi) * 100 if total_oi > 0 else 0
print(f"\nEstimated percent of options exercised (weighted by open interest): {percent_exercised:.2f}%")
