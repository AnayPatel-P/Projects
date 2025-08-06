import numpy as np
import pytest

from src.variates import generate_antithetic_gbm_paths, control_variate_price
from src.gbm import generate_gbm_paths

def test_antithetic_variance_reduction():
    # setup
    S0, r, sigma, T, n_steps = 100.0, 0.05, 0.2, 1.0, 50
    n_paths = 1000  # even
    seed = 2025

    # plain MC
    paths_plain = generate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, random_seed=seed)
    payoffs_plain = np.maximum(paths_plain[:, -1] - 100, 0)
    var_plain = np.var(payoffs_plain, ddof=1)

    # antithetic MC
    paths_anti = generate_antithetic_gbm_paths(S0, r, sigma, T, n_steps, n_paths, random_seed=seed)
    payoffs_anti = np.maximum(paths_anti[:, -1] - 100, 0)
    var_anti = np.var(payoffs_anti, ddof=1)

    assert var_anti < var_plain

def test_control_variate_centering():
    # simple check: if control_payoffs already at its true mean, no adjustment
    mc = np.array([1.0, 2.0, 3.0, 4.0])
    ctrl = np.array([10.0, 10.0, 10.0, 10.0])
    true_mean = 10.0
    adjusted = control_variate_price(mc, ctrl, true_mean)
    assert pytest.approx(adjusted) == mc.mean()

def test_control_variate_effect():
    # create a control variate perfectly correlated with mc_payoffs,
    # so the variance should drop to zero after adjustment.
    mc = np.array([1.0, 2.0, 3.0, 4.0])
    ctrl = mc * 2.0          # perfect correlation
    true_mean = ctrl.mean()
    adjusted = control_variate_price(mc, ctrl, true_mean)
    # since ctrl perfectly tracks mc, variance adj = 0, so adjusted samples all equal something => variance zero
    # and mean should equal the true mc mean
    assert pytest.approx(adjusted) == mc.mean()
