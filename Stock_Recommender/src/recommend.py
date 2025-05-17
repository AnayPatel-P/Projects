# src/recommend.py

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

def recommend(features_df: pd.DataFrame, alpha: float, top_n: int = 10) -> pd.DataFrame:
    """
    Recommend top_n tickers based on risk preference alpha.
    alpha=0 -> conservative (minimize volatility), alpha=1 -> aggressive (maximize return).
    Returns a DataFrame with tickers, predicted return, predicted volatility, and score.
    """
    # Load trained model without compiling (inference only)
    model = load_model("models/risk_model.h5", compile=False)

    # Predict return and volatility
    preds = model.predict(
        features_df[["return", "volatility", "momentum", "sharpe"]].values,
        verbose=0
    )
    pred_return, pred_volatility = preds

    # Compute composite score
    scores = alpha * pred_return.flatten() - (1 - alpha) * pred_volatility.flatten()

    # Build result DataFrame
    results = features_df.reset_index().copy()
    results["pred_return"] = pred_return.flatten()
    results["pred_volatility"] = pred_volatility.flatten()
    results["score"] = scores

    # Select top N
    best = results.nlargest(top_n, "score")
    return best[["ticker", "score", "pred_return", "pred_volatility"]]



