# src/data_fetch.py

import os
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

def fetch_sp500_tickers() -> list:
    """
    Scrape the S&P 500 ticker list from Wikipedia.
    """
    table = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )[0]
    return table["Symbol"].sort_values().tolist()

def download_price_data(
    tickers: list,
    start_date: str,
    end_date: str,
    out_dir: str = "data/raw"
):
    """
    Download daily price data for each ticker and save as CSV.
    """
    os.makedirs(out_dir, exist_ok=True)
    for ticker in tickers:
        print(f"Downloading {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date)
        if not df.empty:
            df.to_csv(f"{out_dir}/{ticker}.csv")
        else:
            print(f"  ⚠️ No data for {ticker}, skipping.")

if __name__ == "__main__":
    # Define date range: last 5 years
    end_dt   = datetime.today()
    start_dt = end_dt - timedelta(days=5*365)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str   = end_dt.strftime("%Y-%m-%d")

    # Run
    tickers = fetch_sp500_tickers()
    print(f"Found {len(tickers)} tickers. Fetching data from {start_str} to {end_str}...")
    download_price_data(tickers, start_str, end_str)
    print("✅ Data download complete.")
