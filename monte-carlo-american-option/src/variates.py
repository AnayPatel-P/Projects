import numpy as np
from .gbm import generate_gbm_paths

def generate_antithetic_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    random_seed: int = None
) -> np.ndarray:
    """
    Simulate GBM paths using antithetic variates and return averaged pairs to reduce variance.

    Parameters
    ----------
    S0, r, sigma, T, n_steps : as in generate_gbm_paths
    n_paths : int
        Must be even; total number of raw samples before pairing.
    random_seed : int, optional

    Returns
    -------
    np.ndarray of shape (n_paths//2, n_steps+1)
        Averaged antithetic path pairs.
    """
    if n_paths % 2 != 0:
        raise ValueError("n_paths must be even for antithetic variates")
    if random_seed is not None:
        np.random.seed(random_seed)

    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    half = n_paths // 2
    # draw half the normals
    normals = np.random.randn(half, n_steps)
    inc_plus = drift + diffusion * normals
    inc_minus = drift - diffusion * normals

    def build_paths(increments):
        log_S0 = np.log(S0)
        # prepend initial log price
        log_incs = np.c_[np.full((increments.shape[0], 1), log_S0), increments]
        log_paths = np.cumsum(log_incs, axis=1)
        return np.exp(log_paths)

    paths_plus = build_paths(inc_plus)
    paths_minus = build_paths(inc_minus)

    # Average each antithetic pair
    averaged_paths = (paths_plus + paths_minus) / 2
    return averaged_paths


def control_variate_price(
    mc_payoffs: np.ndarray,
    control_payoffs: np.ndarray,
    control_true_mean: float
) -> float:
    """
    Adjust Monte Carlo estimate using a control variate.

    Let Y be mc_payoffs, X be control_payoffs, and mu = control_true_mean.
    Then optimal b = Cov(Y,X)/Var(X), and estimator is
      Y_adj = Y - b*(X - mu),  price = mean(Y_adj).

    Parameters
    ----------
    mc_payoffs : np.ndarray
        Raw payoff samples from the target estimator.
    control_payoffs : np.ndarray
        Samples of the control variate with known expectation.
    control_true_mean : float
        Exact (analytic) mean of the control variate.

    Returns
    -------
    float
        Control‐variate‐adjusted price.
    """
    cov = np.cov(mc_payoffs, control_payoffs, ddof=1)[0, 1]
    var_x = np.var(control_payoffs, ddof=1)
    b = cov / var_x if var_x > 0 else 0.0
    adjusted = mc_payoffs - b * (control_payoffs - control_true_mean)
    return adjusted.mean()
