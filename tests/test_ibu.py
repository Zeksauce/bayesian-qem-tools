"""
test_ibu.py

File that contains the tests for the ibu algorithm
"""

import pytest
import numpy as np
from bayesian_qem import iterative_bayesian_unfolding


@pytest.fixture
def basic_setup():
    response_matrix = np.array([[0.9, 0.2], [0.1, 0.8]])
    noisy_counts = np.array([55.0, 45.0])
    expected_output = np.array([50.0, 50.0])
    return response_matrix, noisy_counts, expected_output


def test_ibu_basic_convergence(basic_setup):
    """Verify that IBU runs successfully with default parameters,
    preserving shape, total counts, and non-negativity."""
    response_matrix, noisy_counts, expected_output = basic_setup
    mitigated = iterative_bayesian_unfolding(noisy_counts, response_matrix)

    assert mitigated.shape == noisy_counts.shape
    assert np.isclose(np.sum(mitigated), np.sum(noisy_counts))
    assert np.all(mitigated >= 0)
    assert np.isclose(mitigated, expected_output)


@pytest.mark.parametrize("iterations", [1, 5, 10, 20])
def test_ibu_varying_iterations(basic_setup, iterations):
    """Ensure the unfolding algorithm maintains valid non-negative counts
    across different iteration counts."""
    response_matrix, noisy_counts = basic_setup
    mitigated = iterative_bayesian_unfolding(
        noisy_counts, response_matrix, max_iterations=iterations
    )
    assert np.all(mitigated >= 0)
