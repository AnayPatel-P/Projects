import yfinance as yf
import pandas as pd

def fetch_price_data(tickers, start_date="2020-01-01", end_date=None):
    print("[INFO] Fetching price data...")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    # If downloading multiple tickers, data is multi-level; else it's single-level
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close']  # Use Close since auto_adjust=True already adjusts
    else:
        data = data.to_frame(name='Close')

    # Drop tickers that failed to download
    data = data.dropna(axis=1, how='all')
    print(f"[INFO] Downloaded data for {len(data.columns)} tickers.")
    return data

def calculate_return_stats(price_df, frequency="biweekly"):
    if frequency == "biweekly":
        # Resample to biweekly using last Friday of each 2-week period
        resampled = price_df.resample('2W-FRI').last()
        factor = 26  # 26 biweekly periods in a year
    else:
        resampled = price_df
        factor = 252  # Trading days in a year

    returns = resampled.pct_change().dropna()

    if returns.empty:
        raise ValueError("No valid return data after resampling.")

    annualized_returns = returns.mean() * factor
    annualized_volatility = returns.std() * (factor ** 0.5)
    cov_matrix = returns.cov() * factor

    return {
        f"{frequency}_returns": returns,
        "annualized_returns": annualized_returns,
        "annualized_volatility": annualized_volatility,
        "cov_matrix": cov_matrix
    }

def export_returns_to_csv(returns_df, filename="biweekly_returns.csv"):
    returns_df.index = returns_df.index.strftime('%Y-%m-%d')  # Format index as string dates
    returns_df.to_csv(filename)
    print(f"[INFO] Exported biweekly returns to '{filename}'")
