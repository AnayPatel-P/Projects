# backend/optimizer.py
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions

def optimize_portfolio(price_df, risk_level="medium", max_assets=10):
    # Compute expected returns and sample covariance
    mu = expected_returns.mean_historical_return(price_df)
    S = risk_models.sample_cov(price_df)

    ef = EfficientFrontier(mu, S)

    # Add regularization: keep portfolios simple
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)

    # Adjust optimization objective based on risk level
    if risk_level == "low":
        weights = ef.min_volatility()
    elif risk_level == "high":
        weights = ef.max_sharpe()
    else:
        try:
            weights = ef.efficient_risk(target_volatility=0.15)
        except ValueError as e:
            # If requested volatility is too low, fallback to min volatility
            print(f"[WARN] {e}")
            weights = ef.min_volatility()


    cleaned_weights = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False)

    return {
        "weights": cleaned_weights,
        "expected_return": perf[0],
        "expected_volatility": perf[1],
        "sharpe_ratio": perf[2]
    }



def export_weights_to_csv(weights_dict, filename="optimized_weights.csv"):
    df = pd.DataFrame(list(weights_dict.items()), columns=["Ticker", "Weight"])
    df["Weight"] = (df["Weight"] * 100).round(2)  # Convert to %
    df.to_csv(filename, index=False)
    print(f"[INFO] Exported optimized weights to '{filename}'")
