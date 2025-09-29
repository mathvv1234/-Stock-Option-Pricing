import yfinance as yf
import numpy as np
from scipy.stats import norm
import pandas as pd
from datetime import datetime, timedelta

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Calculate Black-Scholes option price for a European call or put.

    Parameters:
    S (float): Current stock price
    K (float): Strike price
    T (float): Time to expiration in years
    r (float): Risk-free interest rate
    sigma (float): Volatility of the underlying asset
    option_type (str): "call" or "put"

    Returns:
    float: Option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return max(price, 0)  # Ensure non-negative price

def fetch_option_data(ticker_symbol):
    """
    Fetch option chain data for the given ticker using yfinance.

    Parameters:
    ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL')

    Returns:
    tuple: (calls, puts, current_price, expiration_date)
    """
    ticker = yf.Ticker(ticker_symbol)
    # Get nearest expiration date
    expiration_dates = ticker.options
    if not expiration_dates:
        raise ValueError("No option chain data available")

    nearest_expiry = expiration_dates[0]  # Use the first (nearest) expiration
    option_chain = ticker.option_chain(nearest_expiry)
    calls, puts = option_chain.calls, option_chain.puts

    # Get current stock price
    stock_data = ticker.history(period='1d')
    current_price = stock_data['Close'].iloc[-1]

    return calls, puts, current_price, nearest_expiry

def estimate_volatility(ticker_symbol):
    """
    Estimate annualized volatility from historical daily returns.

    Parameters:
    ticker_symbol (str): Stock ticker symbol

    Returns:
    float: Annualized volatility
    """
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period='1y')
    returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    volatility = returns.std() * np.sqrt(252)  # Annualize volatility
    return volatility

def analyze_option_chain():
    """
    Analyze AAPL option chain, compute Black-Scholes prices, and calculate percent errors.
    """
    ticker = "SAP"
    risk_free_rate = 0.0423  # 10-year Treasury yield as of July 2025 (approximate)

    try:
        # Fetch data
        calls, puts, current_price, expiry_date = fetch_option_data(ticker)
        volatility = estimate_volatility(ticker)

        # Calculate time to expiration
        expiry_datetime = datetime.strptime(expiry_date, '%Y-%m-%d')
        today = datetime.now()
        T = (expiry_datetime - today).days / 365.0
        if T <= 0:
            raise ValueError("Expiration date must be in the future")

        # Initialize results list
        results = []

        # Process call options
        for _, row in calls.iterrows():
            strike = row['strike']
            actual_price = row['lastPrice']  # Use last traded price
            if pd.isna(actual_price) or actual_price <= 0:
                continue  # Skip invalid prices

            bs_price = black_scholes(current_price, strike, T, risk_free_rate, volatility, "call")
            percent_error = abs(bs_price - actual_price)

            results.append({
                'Type': 'Call',
                'Strike': strike,
                'Actual Price': actual_price,
                'BS Price': bs_price,
                'Absolute Error': percent_error
            })

        # Process put options
        for _, row in puts.iterrows():
            strike = row['strike']
            actual_price = row['lastPrice']
            if pd.isna(actual_price) or actual_price <= 0:
                continue

            bs_price = black_scholes(current_price, strike, T, risk_free_rate, volatility, "put")
            percent_error = abs(bs_price - actual_price)

            results.append({
                'Type': 'Put',
                'Strike': strike,
                'Actual Price': actual_price,
                'BS Price': bs_price,
                'Absolute Error': percent_error
            })

        # Create DataFrame
        df = pd.DataFrame(results)
        df = df.round({'Actual Price': 2, 'BS Price': 2, 'Error': 2})
        df = df.sort_values(['Type', 'Strike'])

        # Print summary
        print(f"\nSAP Option Chain Analysis (Expiration: {expiry_date})")
        print(f"Current Stock Price: ${current_price:.2f}")
        print(f"Estimated Volatility: {volatility:.2%}")
        print(f"Risk-Free Rate: {risk_free_rate:.2%}")
        print(f"Time to Expiration: {T:.4f} years")
        print("\nResults:")
        print(df.to_string(index=False))

        # Summary statistics
        print("\nSummary Statistics:")
        print(df.groupby('Type')[['Absolute Error']].agg(['mean', 'min', 'max']).round(2))

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    analyze_option_chain()
