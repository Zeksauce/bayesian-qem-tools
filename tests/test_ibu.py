import pytest
import numpy as np
from bayesian_qem import iterative_bayesian_unfolding


@pytest.fixture
def basic_setup():
    response_matrix = np.array([
        [0.9, 0.2],
        [0.1, 0.8]
    ])
    noisy_counts = np.array([60.0, 40.0])
    return response_matrix, noisy_counts

def test_ibu_basic_convergence(basic_setup):
    response_matrix, noisy_counts = basic_setup
    mitigated = iterative_bayesian_unfolding(noisy_counts, response_matrix)
    
    assert mitigated.shape == noisy_counts.shape
    assert np.isclose(np.sum(mitigated), np.sum(noisy_counts))
    assert np.all(mitigated >= 0)

@pytest.mark.parametrize("iterations", [1, 5, 10, 20])
def test_ibu_varying_iterations(basic_setup, iterations):
    response_matrix, noisy_counts = basic_setup
    mitigated = iterative_bayesian_unfolding(noisy_counts, response_matrix, max_iterations=iterations)
    assert np.all(mitigated >= 0)