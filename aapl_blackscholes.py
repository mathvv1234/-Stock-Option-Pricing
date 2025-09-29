import yfinance as yf
import numpy as np
from scipy.stats import norm, pearsonr
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Calculate Black-Scholes option price.

    Parameters:
    S (float): Stock price
    K (float): Strike price
    T (float): Time to expiration (years)
    r (float): Risk-free rate
    sigma (float): Volatility
    option_type (str): "call" or "put"

    Returns:
    float: Option price
    """
    if T <= 0 or sigma <= 0:
        return 0 if option_type == "call" else max(K - S, 0) if S > 0 else K
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return max(price, 0)

def fetch_option_data(ticker_symbol):
    """
    Fetch option chain data.

    Returns:
    tuple: (calls, puts, current_price, expiration_date)
    """
    ticker = yf.Ticker(ticker_symbol)
    expiration_dates = ticker.options
    if not expiration_dates:
        raise ValueError("No option chain data available")

    nearest_expiry = expiration_dates[0]  # Nearest expiration
    option_chain = ticker.option_chain(nearest_expiry)
    calls, puts = option_chain.calls, option_chain.puts
    stock_data = ticker.history(period='1d')
    current_price = stock_data['Close'].iloc[-1]

    return calls, puts, current_price, nearest_expiry

def estimate_volatility(ticker_symbol):
    """
    Estimate annualized volatility from 1-year daily returns.

    Returns:
    float: Volatility
    """
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period='1y')
    returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
    return returns.std() * np.sqrt(252)

def analyze_option_chain():
    """
    Analyze ASML option chain using Black-Scholes, compute correlation coefficients,
    and generate a scatter plot of actual vs. Black-Scholes prices.
    """
    ticker = "AAPL"
    risk_free_rate = 0.0423  # 10-year Treasury yield (Dec 2023, assumed for July 2025)

    try:
        # Fetch data
        calls, puts, current_price, expiry_date = fetch_option_data(ticker)
        volatility = estimate_volatility(ticker)

        # Time to expiration
        expiry_datetime = datetime.strptime(expiry_date, '%Y-%m-%d')
        today = datetime.now()
        T = (expiry_datetime - today).days / 365.0
        if T <= 0:
            raise ValueError("Expiration date in the past")

        # Initialize results
        results = []

        # Process calls
        for _, row in calls.iterrows():
            strike = row['strike']
            actual_price = row['lastPrice']
            if pd.isna(actual_price) or actual_price <= 0:
                continue

            bs_price = black_scholes(current_price, strike, T, risk_free_rate, volatility, "call")
            ae = abs(bs_price - actual_price)

            # Determine moneyness
            moneyness = current_price / strike
            if moneyness > 1.05:
                money_type = "ITM"
            elif moneyness < 0.95:
                money_type = "OTM"
            else:
                money_type = "ATM"

            results.append({
                'Type': 'Call',
                'Moneyness': money_type,
                'Strike': strike,
                'Actual Price': actual_price,
                'BS Price': bs_price,
                'Absolute Error': ae
            })

        # Process puts
        for _, row in puts.iterrows():
            strike = row['strike']
            actual_price = row['lastPrice']
            if pd.isna(actual_price) or actual_price <= 0:
                continue

            bs_price = black_scholes(current_price, strike, T, risk_free_rate, volatility, "put")
            ae = abs(bs_price - actual_price)

            # Determine moneyness
            moneyness = current_price / strike
            if moneyness < 0.95:
                money_type = "ITM"
            elif moneyness > 1.05:
                money_type = "OTM"
            else:
                money_type = "ATM"

            results.append({
                'Type': 'Put',
                'Moneyness': money_type,
                'Strike': strike,
                'Actual Price': actual_price,
                'BS Price': bs_price,
                'Absolute Error': ae
            })

        # Create DataFrame
        df = pd.DataFrame(results)
        df = df.round({'Actual Price': 2, 'BS Price': 2, 'Absolute Error': 2})
        df = df.sort_values(['Type', 'Moneyness', 'Strike'])

        # Compute correlation coefficients
        call_df = df[df['Type'] == 'Call']
        put_df = df[df['Type'] == 'Put']

        call_corr = pearsonr(call_df['Actual Price'], call_df['BS Price'])[0] if len(call_df) >= 2 else np.nan
        put_corr = pearsonr(put_df['Actual Price'], put_df['BS Price'])[0] if len(put_df) >= 2 else np.nan

        # Correlation by moneyness
        corr_by_moneyness = {}
        for money_type in ['ITM', 'ATM', 'OTM']:
            for opt_type in ['Call', 'Put']:
                subset = df[(df['Type'] == opt_type) & (df['Moneyness'] == money_type)]
                if len(subset) >= 2:
                    corr = pearsonr(subset['Actual Price'], subset['BS Price'])[0]
                    corr_by_moneyness[f"{opt_type} {money_type}"] = corr
                else:
                    corr_by_moneyness[f"{opt_type} {money_type}"] = np.nan

        # Create scatter plot
        plt.figure(figsize=(10, 6))

        # Plot calls
        calls = df[df['Type'] == 'Call']
        plt.scatter(calls['Actual Price'], calls['BS Price'], color='blue', label='Call Options', alpha=0.6)

        # Plot puts
        puts = df[df['Type'] == 'Put']
        plt.scatter(puts['Actual Price'], puts['BS Price'], color='red', label='Put Options', alpha=0.6)

        # Add y=x line
        max_price = max(df['Actual Price'].max(), df['BS Price'].max())
        plt.plot([0, max_price], [0, max_price], color='gray', linestyle='--', label='Perfect Fit (y=x)')

        # Customize plot
        plt.xlabel('Actual Option Price ($)')
        plt.ylabel('Black-Scholes Predicted Price ($)')
        plt.title('Black-Scholes Model vs. Actual Option Prices for AAPL')
        plt.legend()
        plt.grid(True)
        plt.xlim(0, max_price * 1.1)
        plt.ylim(0, max_price * 1.1)

        # Save and show plot

        plt.show()

        # Print summary
        print(f"\nAAPL Option Chain Analysis (Expiration: {expiry_date})")
        print(f"Current Stock Price: ${current_price:.2f}")
        print(f"Estimated Volatility: {volatility:.2%}")
        print(f"Risk-Free Rate: {risk_free_rate:.2%}")
        print(f"Time to Expiration: {T:.4f} years")
        print("\nResults:")
        print(df.to_string(index=False))

        # Summary statistics
        print("\nSummary Statistics:")
        summary = df.groupby(['Type', 'Moneyness'])['Absolute Error'].agg(['mean', 'min', 'max']).round(2)
        print(summary)

        print("\nCorrelation Coefficients:")
        print(f"Call Options: {call_corr:.4f}")
        print(f"Put Options: {put_corr:.4f}")
        print("\nCorrelation by Moneyness:")
        for key, value in corr_by_moneyness.items():
            print(f"{key}: {value:.4f}" if not np.isnan(value) else f"{key}: Insufficient data")

        print("\nScatter plot saved as 'option_plot.png' and displayed.")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    analyze_option_chain()
