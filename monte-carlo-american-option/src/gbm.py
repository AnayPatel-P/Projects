import numpy as np

def generate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    random_seed: int = None
) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion paths for an underlying asset.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    r : float
        Risk-free rate (annualized drift).
    sigma : float
        Volatility (annualized).
    T : float
        Time to maturity (in years).
    n_steps : int
        Number of discrete time steps.
    n_paths : int
        Number of simulated paths.
    random_seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_paths, n_steps+1) with simulated price paths.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    dt = T / n_steps
    # pre-allocate array and set initial price
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0

    # generate log returns
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    increments = drift + diffusion * np.random.randn(n_paths, n_steps)

    # build price paths
    paths[:, 1:] = S0 * np.exp(np.cumsum(increments, axis=1))
    return paths
