import numpy as np
import pytest
from src.lsm import price_american

def test_put_zero_volatility():
    # With zero volatility, American put should be exercised immediately:
    S0, K = 80.0, 100.0
    price = price_american(
        S0=S0, K=K, r=0.05, sigma=0.0, T=1.0,
        n_steps=10, n_paths=10000, option_type="put", random_seed=42
    )
    # Intrinsic value = K - S0 = 20
    assert pytest.approx(price, rel=1e-2) == K - S0

def test_call_zero_volatility():
    # With zero volatility, an American call (no dividends) should not be exercised early: 
    # price = Black-Scholes European call ≈ max(S0*exp(rT) - K, 0)
    S0, K = 120.0, 100.0
    T, r = 1.0, 0.05
    price = price_american(
        S0=S0, K=K, r=r, sigma=0.0, T=T,
        n_steps=10, n_paths=10000, option_type="call", random_seed=42
    )
    expected = max(S0 * np.exp(r * T) - K, 0)
    assert pytest.approx(price, rel=1e-2) == expected
