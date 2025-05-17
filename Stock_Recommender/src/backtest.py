# src/backtest.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from recommend import recommend


def backtest(alpha: float = 0.5, top_n: int = 10):
    """
    Monthly backtest:
    - At each month-end, pick top_n tickers by score for given alpha
    - Track realized returns & volatility over next 21 trading days
    - Compute average performance and cumulative return curve
    """
    # Load and prepare features
    feats = pd.read_csv('data/processed/features.csv', parse_dates=['date'])
    feats.sort_values('date', inplace=True)

    # Determine month-end rebalance dates
    month_ends = (
        feats['date']
             .dt.to_period('M')
             .drop_duplicates()
             .dt.to_timestamp('M')
    )
    results = []

    for date in month_ends:
        # snapshot features on rebalance date
        snapshot = feats[feats['date'] == date]
        if snapshot.empty:
            continue

        # recommendations for that snapshot
        picks = recommend(
            snapshot.set_index('ticker'), alpha=alpha, top_n=top_n
        )
        tickers = picks['ticker'].tolist()

        rets, vols = [], []
        for ticker in tickers:
            csv_path = os.path.join('data/raw', f'{ticker}.csv')
            if not os.path.exists(csv_path):
                continue

                        # Read raw price and clean index
            raw = pd.read_csv(csv_path, index_col=0)
            raw.index = pd.to_datetime(raw.index, errors='coerce')
            raw = raw[~raw.index.isna()]
            raw.sort_index(inplace=True)

            # Ensure Close prices are numeric
            raw['Close'] = pd.to_numeric(raw['Close'], errors='coerce')
            raw = raw.dropna(subset=['Close'])

            # locate the first trading day >= rebalance date
            pos = raw.index.searchsorted(date)
            if pos >= len(raw):
                continue
            # take next 21 trading days
            window = raw['Close'].iloc[pos : pos + 21]
            if len(window) < 2:
                continue

            # compute realized return & volatility
            ret = window.iloc[-1] / window.iloc[0] - 1
            daily_ret = window.pct_change().dropna()
            vol = daily_ret.std() * np.sqrt(252)

            rets.append(ret)
            vols.append(vol)

        if rets:
            results.append({
                'date': date,
                'avg_return': np.mean(rets),
                'avg_volatility': np.mean(vols)
            })

    # assemble backtest results
    bt = pd.DataFrame(results)
    bt.sort_values('date', inplace=True)
    bt['cumulative_return'] = (1 + bt['avg_return']).cumprod()

    # save and plot
    os.makedirs('data/backtest', exist_ok=True)
    bt.to_csv('data/backtest/backtest_results.csv', index=False)

    plt.figure()
    plt.plot(bt['date'], bt['cumulative_return'], marker='o')
    plt.title(f'Backtest Cumulative Return (alpha={alpha}, top_n={top_n})')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    backtest(alpha=0.5, top_n=10)
