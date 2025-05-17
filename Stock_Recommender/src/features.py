# src/features.py

import os
import pandas as pd
import numpy as np
from glob import glob


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling features for stock price data:
      - return: daily percent change
      - volatility: annualized rolling std of returns
      - momentum: percent change over 60 days
      - sharpe: annualized rolling Sharpe ratio
    """
    df = df.copy()
    # ensure Close is numeric, drop invalid rows
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])

    # compute raw returns
    returns = df["Close"].pct_change()
    # annualized volatility
    volatility = returns.rolling(60).std() * np.sqrt(252)
    # momentum over 60 days
    momentum = df["Close"].pct_change(periods=60)
    # annualized Sharpe ratio
    sharpe = (returns.rolling(60).mean() / returns.rolling(60).std()) * np.sqrt(252)

    # assemble features DataFrame
    feats = pd.concat(
        [returns, volatility, momentum, sharpe],
        axis=1,
        keys=["return", "volatility", "momentum", "sharpe"]
    )
    return feats.dropna()


def main():
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    all_feats = []
    for fp in glob(f"{raw_dir}/*.csv"):
        ticker = os.path.basename(fp).replace('.csv', '')
        # read first column as datetime index
        df = pd.read_csv(fp, index_col=0, parse_dates=True)
        df.index.name = "Date"

        feats = compute_features(df)
        feats["ticker"] = ticker
        feats["date"] = feats.index
        all_feats.append(feats)

    # combine all tickers
    features_df = pd.concat(all_feats, ignore_index=True)
    features_df.to_csv(f"{processed_dir}/features.csv", index=False)

    # extract latest features per ticker
    latest = (
        features_df.sort_values("date")
                   .groupby("ticker")
                   .last()
                   .reset_index()
                   .set_index("ticker")
    )
    latest[["return", "volatility", "momentum", "sharpe"]] \
        .to_csv(f"{processed_dir}/latest_features.csv")

    print("✅ Saved features.csv and latest_features.csv in data/processed/")


if __name__ == "__main__":
    main()