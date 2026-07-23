"""
test_ibu.py

File that contains the tests for the ibu algorithm
"""

from collections.abc import Callable
from typing import Any
from functools import partial
import re
import pytest
import numpy as np
from bayesian_qem import iterative_bayesian_unfolding
from bayesian_qem.exceptions import (
    DimensionError,
    ShapeError,
    InvalidResponseMatrixError,
)


@pytest.fixture
def basic_setup():
    """Verifiable 1 qubit system with known output"""
    response_matrix = np.array([[0.9, 0.2], [0.1, 0.8]])
    noisy_counts = np.array([55.0, 45.0])
    expected_output = np.array([50.0, 50.0])
    return response_matrix, noisy_counts, expected_output


def _is_valid_solution(
    mitigated: np.ndarray[tuple[int], np.dtype[np.float64]],
    noisy_counts: np.ndarray[tuple[int], np.dtype[np.float64]],
):
    assert mitigated.shape == noisy_counts.shape
    assert np.isclose(np.sum(mitigated), np.sum(noisy_counts))
    assert np.all(mitigated >= 0)


def _is_valid_expected_solution(
    mitigated: np.ndarray[tuple[int], np.dtype[np.float64]],
    noisy_counts: np.ndarray[tuple[int], np.dtype[np.float64]],
    expected_output: np.ndarray[tuple[int], np.dtype[np.float64]],
):
    _is_valid_solution(mitigated, noisy_counts)
    assert np.allclose(mitigated, expected_output)


def _validate_error_raised(
    function: Callable[..., Any],
    error: type[Exception],
    msg: str,
    *args: Any,
    **kwargs: Any,
):
    """Helper to validate raised errors, forwarding
    any positional or keyword arguments directly to the function"""
    with pytest.raises(error, match=re.escape(msg)):
        function(*args, **kwargs)


_validate_error_raised_in_ibu = partial(
    _validate_error_raised, iterative_bayesian_unfolding
)


def test_ibu_basic_convergence(basic_setup):
    """Verify that IBU runs successfully with default parameters,
    preserving shape, total counts, and non-negativity."""
    response_matrix, noisy_counts, expected_output = basic_setup
    mitigated = iterative_bayesian_unfolding(noisy_counts, response_matrix)

    _is_valid_expected_solution(mitigated, noisy_counts, expected_output)


@pytest.mark.parametrize("iterations", [1, 5, 10, 20])
def test_ibu_varying_iterations(basic_setup, iterations):
    """Ensure the unfolding algorithm maintains valid non-negative counts
    across different iteration counts."""
    response_matrix, noisy_counts, expected_output = basic_setup
    mitigated = iterative_bayesian_unfolding(
        noisy_counts, response_matrix, max_iterations=iterations
    )
    _is_valid_expected_solution(mitigated, noisy_counts, expected_output)



@pytest.mark.parametrize(
    "response_matrix", [ValueError, TypeError]
)
def test_ibu_varying_non_array_response_matrix(basic_setup, response_matrix):
    """Ensure algorithm raises ValueErrors for Response Matrices that are not array-like"""
    _, noisy_counts, _ = basic_setup
    msg = f"Response matrix must be an array-like, but received Type {type(response_matrix)}."
    _validate_error_raised_in_ibu(
        ValueError,
        msg,
        noisy_counts,
        response_matrix,
    )
@pytest.mark.parametrize(
    "noisy_counts", [ValueError, TypeError]
)
def test_ibu_varying_non_array_noisy_counts(basic_setup, noisy_counts):
    """Ensure algorithm raises ValueErrors for noisy counts that are not array-like"""
    response_matrix, _, _ = basic_setup
    msg = f"Noisy counts must be an array-like, but received Type {type(noisy_counts)}."
    _validate_error_raised_in_ibu(
        ValueError,
        msg,
        noisy_counts,
        response_matrix,
    )
@pytest.mark.parametrize(
    "iterations", [-1.1, -1.0, -0.1, 0.0, 0.1, 1.0, 1.1, "1", "0", "test", None]
)
def test_ibu_varying_non_int_iterations(basic_setup, iterations):
    """Ensure algorithm raises ValueErrors for max_iterations that are not integers"""
    response_matrix, noisy_counts, _ = basic_setup
    msg = f"Max iterations must be a positive integer, but received {iterations}."
    _validate_error_raised_in_ibu(
        ValueError,
        msg,
        noisy_counts,
        response_matrix,
        max_iterations=iterations,
    )


@pytest.mark.parametrize("iterations", [-20, -10, -1, 0])
def test_ibu_varying_invalid_int_iterations(basic_setup, iterations):
    """Ensure algorithm raises ValueErrors for max_iterations that are invalid integers"""
    response_matrix, noisy_counts, _ = basic_setup
    msg = f"Max iterations must be a positive integer, but received {iterations}."
    _validate_error_raised_in_ibu(
        ValueError,
        msg,
        noisy_counts,
        response_matrix,
        max_iterations=iterations,
    )


def test_ibu_mismatched_dimensions(basic_setup):
    """Ensure algorithm raises error when response matrix dimensions don't match counts."""
    _, noisy_counts, _ = basic_setup
    bad_matrix = np.array([[0.9, 0.2, 0.1], [0.1, 0.8, 0.9]])
    msg = f"Response matrix {bad_matrix.shape} must be a 2D square matrix."
    _validate_error_raised_in_ibu(
        DimensionError,
        msg,
        noisy_counts,
        bad_matrix,
    )


def test_ibu_negative_noisy_counts(basic_setup):
    """Ensure algorithm raises error when noisy counts contain negative values."""
    response_matrix, _, _ = basic_setup
    bad_counts = np.array([-55.0, 45.0])
    msg = "Noisy counts cannot be negative"
    _validate_error_raised_in_ibu(
        ValueError,
        msg,
        bad_counts,
        response_matrix,
    )


def test_ibu_negative_response_matrix(basic_setup):
    """Ensure algorithm raises error when response matrix contains negative values."""
    _, noisy_counts, _ = basic_setup
    bad_matrix = np.array([[-0.9, 0.2], [0.1, 0.8]])
    msg = "Response matrix entries cannot be negative"
    _validate_error_raised_in_ibu(
        ValueError,
        msg,
        noisy_counts,
        bad_matrix,
    )


def test_ibu_negative_prior(basic_setup):
    """Ensure algorithm raises error when initial prior contains negative values."""
    response_matrix, noisy_counts, _ = basic_setup
    bad_prior = np.array([-50.0, 50.0])
    msg = "Prior probabilities cannot be negative."
    _validate_error_raised_in_ibu(
        ValueError,
        msg,
        noisy_counts,
        response_matrix,
        initial_prior=bad_prior,
    )


@pytest.mark.parametrize("bad_tolerance", [-1e-6, 0.0, "1e-6"])
def test_ibu_invalid_tolerance(basic_setup, bad_tolerance):
    """Ensure algorithm raises error for invalid tolerance values."""
    response_matrix, noisy_counts, _ = basic_setup
    msg = f"Tolerance must be a positive number, but received {bad_tolerance}."
    _validate_error_raised_in_ibu(
        ValueError,
        msg,
        noisy_counts,
        response_matrix,
        tolerance=bad_tolerance,
    )


def test_ibu_invalid_response_matrix_normalization(basic_setup):
    """Ensure algorithm raises error when response matrix columns/rows do not sum to expected constraints."""
    _, noisy_counts, _ = basic_setup
    # Invalid sum matrix
    bad_matrix = np.array([[0.5, 0.2], [0.1, 0.8]])
    msg = (
        "Response matrix columns must sum to 1 (representing valid transition probabilities). "
        f"Found column sums: {np.sum(bad_matrix, axis=0)}"
    )
    _validate_error_raised_in_ibu(
        InvalidResponseMatrixError,
        msg,
        noisy_counts,
        bad_matrix,
    )


def test_ibu_unnormalized_prior(basic_setup):
    """Ensure algorithm runs when initial prior is not properly normalized."""
    response_matrix, noisy_counts, expected_output = basic_setup
    prior = np.array([50, 50])
    mitigated = iterative_bayesian_unfolding(
        noisy_counts, response_matrix, initial_prior=prior
    )
    _is_valid_expected_solution(mitigated, noisy_counts, expected_output)
