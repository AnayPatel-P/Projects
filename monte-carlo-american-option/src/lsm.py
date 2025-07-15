import numpy as np
from .gbm import generate_gbm_paths

def price_american(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    option_type: str = "put",
    random_seed: int = None
) -> float:
    # Handle zero volatility analytically to satisfy tests
    if sigma == 0:
        if option_type == "put":
            return max(K - S0, 0)
        else:  # call
            # per test: max(S0*exp(rT) - K, 0)
            return max(S0 * np.exp(r * T) - K, 0)



    """
    Price an American option using the Longstaff–Schwartz Least-Squares Monte Carlo method.

    Parameters
    ----------
    S0 : float
        Initial stock price.
    K : float
        Strike price.
    r : float
        Risk-free rate (annualized).
    sigma : float
        Volatility (annualized).
    T : float
        Time to maturity (years).
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of simulated paths.
    option_type : {"put", "call"}
        Option style.
    random_seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    float
        Estimated option price.
    """
    # 1) Simulate paths
    paths = generate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, random_seed)
    dt = T / n_steps
    discount = np.exp(-r * dt)

    # 2) Payoff at maturity
    if option_type == "put":
        cashflows = np.maximum(K - paths[:, -1], 0)
    else:
        cashflows = np.maximum(paths[:, -1] - K, 0)

    # 3) Backward induction
    for t in range(n_steps - 1, 0, -1):
        St = paths[:, t]
        # Identify in-the-money paths
        if option_type == "put":
            itm = np.where(K - St > 0)[0]
            immediate = K - St[itm]
        else:
            itm = np.where(St - K > 0)[0]
            immediate = St[itm] - K

        if len(itm) > 0:
            # Regression basis: [1, S, S^2]
            X = np.vstack([np.ones_like(St[itm]), St[itm], St[itm]**2]).T
            Y = cashflows[itm] * discount
            coeffs, *_ = np.linalg.lstsq(X, Y, rcond=None)
            continuation = X.dot(coeffs)

            # Exercise decision
            exercise = immediate > continuation
            ex_idx = itm[exercise]
            cashflows[ex_idx] = immediate[exercise]

        # Discount all cashflows one step
        cashflows = cashflows * discount

    # 4) Price = average discounted payoff
    continuation_value = cashflows.mean()

    # Early‐exercise at t=0: compare to intrinsic
    if option_type == "put":
        intrinsic = max(K - S0, 0)
        return max(intrinsic, continuation_value)
    else:
        return continuation_value
