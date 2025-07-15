import numpy as np
from src.gbm import generate_gbm_paths

def test_gbm_reproducibility_and_shape():
    paths_a = generate_gbm_paths(100, 0.03, 0.2, T=1.0, n_steps=50, n_paths=500, random_seed=123)
    paths_b = generate_gbm_paths(100, 0.03, 0.2, T=1.0, n_steps=50, n_paths=500, random_seed=123)
    # Same shape
    assert paths_a.shape == (500, 51)
    # Reproducible given same seed
    assert np.allclose(paths_a, paths_b)
    # All values positive
    assert np.all(paths_a > 0)
