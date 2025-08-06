import numpy as np
import matplotlib.pyplot as plt
from .lsm import price_american


def plot_exercise_boundary(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    option_type: str = "put",
    random_seed: int = None
) -> plt.Figure:
    """
    Estimate and return a Figure of the early exercise boundary for American options.

    Parameters and behavior as before, but returns the matplotlib Figure instead of showing.
    """
    dt = T / n_steps
    rng = np.random.default_rng(random_seed)
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal((n_paths, n_steps))
    log_paths = np.concatenate([np.full((n_paths, 1), np.log(S0)), increments], axis=1).cumsum(axis=1)
    paths = np.exp(log_paths)

    exercise_boundary = []
    discount = np.exp(-r * dt)

    if option_type == "put":
        cashflows = np.maximum(K - paths[:, -1], 0)
    else:
        cashflows = np.maximum(paths[:, -1] - K, 0)

    for t in range(n_steps - 1, 0, -1):
        St = paths[:, t]
        if option_type == "put":
            itm = np.where(K - St > 0)[0]
            immediate = K - St[itm]
        else:
            itm = np.where(St - K > 0)[0]
            immediate = St[itm] - K

        if len(itm) > 0:
            X = np.vstack([np.ones_like(St[itm]), St[itm], St[itm]**2]).T
            continuation = X.dot(np.linalg.lstsq(X, cashflows[itm] * discount, rcond=None)[0])
            diff = immediate - continuation
            idx = np.argsort(np.abs(diff))[:2]
            boundary = St[itm][idx].mean()
        else:
            boundary = np.nan

        exercise_boundary.append(boundary)
        cashflows = cashflows * discount

    times = np.linspace(dt, T - dt, n_steps - 1)[::-1]

    fig, ax = plt.subplots()
    ax.plot(times, exercise_boundary, marker='o')
    ax.set_xlabel('Time to maturity')
    ax.set_ylabel('Estimated early-exercise boundary')
    ax.set_title('American Option Exercise Boundary Over Time')
    fig.tight_layout()
    return fig


def plot_convergence(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    path_counts: list,
    option_type: str = "put",
    random_seed: int = None
) -> plt.Figure:
    """
    Return a Figure plotting estimated price vs. number of paths for Monte Carlo convergence.
    """
    estimates = []
    for n in path_counts:
        estimates.append(price_american(S0, K, r, sigma, T, n_steps, n, option_type, random_seed))

    fig, ax = plt.subplots()
    ax.plot(path_counts, estimates, marker='x')
    ax.set_xscale('log')
    ax.set_xlabel('Number of paths (log scale)')
    ax.set_ylabel('Estimated option price')
    ax.set_title('Monte Carlo Convergence')
    fig.tight_layout()
    return fig
