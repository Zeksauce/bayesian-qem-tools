"""
ibu.py

File that contains the function iterative_bayesian_unfolding.
"""

from typing import Callable
from functools import partial
import numbers
import logging
import numpy as np
from .exceptions import DimensionError, InvalidResponseMatrixError, ShapeError
from . import backend as ibu_backend

# Initialize a module-level logger
logger = logging.getLogger(__name__)


def iterative_bayesian_unfolding(
    noisy_counts: np.ndarray,
    response_matrix: np.ndarray | None = None,
    local_matrices: list[np.ndarray] | None = None,
    initial_prior: np.ndarray | None = None,
    max_iterations: int = 10,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """
    Solves for likely physically accurate true outputs from
    a quantum circuit's noisy outputs and calibration classifications.

    Args:
        noisy_counts: The observed output counts from the quantum circuit.
        response_matrix: The square calibration response matrix where rows
            correspond to measured states and columns correspond to true states
            (entry [i, j] is the probability of measuring state i given true state j).
        local_matrices: The collection of clustered correlated qubit systems and their
            square calibration responses.
        initial_prior: Optional starting prior distribution.
            Defaults to using an uninformed uniform prior.
        max_iterations: Optional maximum number of IBU iterations to run.
            Defaults to 10.
        tolerance: Optional early stopping condition for convergence based on L1 norm.
            Defaults to 1e-6.

    Returns:
        The estimated true state counts of the quantum circuit.

    Example:
        >>> import numpy as np
        >>> response = np.array([[0.9, 0.2], [0.1, 0.8]])
        >>> counts = np.array([55.0, 45.0])
        >>> iterative_bayesian_unfolding(counts, response)
        array([50., 50.])
    """
    _validate_inputs(max_iterations, tolerance)
    noisy_counts = _validate_array_like(noisy_counts, "Noisy counts")
    _validate_non_negative(noisy_counts, "Noisy counts")

    # Wrapper / Dispatcher logic: Selects the backend operations
    matmul_forward, matmul_transpose = _select_backend(
        noisy_counts, response_matrix, local_matrices
    )

    current_prior = _initialize_prior(initial_prior, noisy_counts)

    # Run the decoupled core loop
    return _run_ibu_loop(
        noisy_counts,
        current_prior,
        matmul_forward,
        matmul_transpose,
        max_iterations,
        tolerance,
    )


def _select_backend(noisy_counts, response_matrix, local_matrices):
    """Dispatcher wrapper to choose between dense or tensor-product backends."""
    if response_matrix is not None and local_matrices is not None:
        raise ValueError(
            "Provide either `response_matrix` or `local_matrices`, not both."
        )

    if response_matrix is not None:
        response_matrix = _initialize_response_matrix(response_matrix)
        # Validate shape match for dense mode
        if noisy_counts.shape[0] != response_matrix.shape[0]:
            raise DimensionError(
                f"Response matrix of shape: {response_matrix.shape} and "
                f"observed counts of shape: {noisy_counts.shape} are mismatched dimensions."
            )

        matmul_forward = partial(
            ibu_backend.apply_dense_response, response_matrix=response_matrix
        )
        matmul_transpose = partial(
            ibu_backend.apply_dense_transpose, response_matrix=response_matrix
        )

    elif local_matrices is not None:
        expected_size = 2 ** len(local_matrices)
        if noisy_counts.shape[0] != expected_size:
            raise DimensionError(
                f"Noisy counts size {noisy_counts.shape[0]} "
                f"does not match tensor product system size {expected_size}."
            )

        matmul_forward = partial(
            ibu_backend.apply_tensor_response, local_matrices=local_matrices
        )
        matmul_transpose = partial(
            ibu_backend.apply_tensor_transpose, local_matrices=local_matrices
        )

    else:
        raise ValueError("Must provide either `response_matrix` or `local_matrices`.")
    return matmul_forward, matmul_transpose


def _run_ibu_loop(
    noisy_counts: np.ndarray[tuple[int], np.dtype[np.float64]],
    current_prior: np.ndarray[tuple[int], np.dtype[np.float64]],
    matmul_forward: Callable[[np.ndarray], np.ndarray],
    matmul_transpose: Callable[[np.ndarray], np.ndarray],
    max_iterations: int,
    tolerance: float,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Runs the core iterative expectation-maximization loop using backend operators."""
    estimated_true_counts = noisy_counts

    logger.info(
        "Iteration initialized with max_iterations=%d, tolerance=%e",
        max_iterations,
        tolerance,
    )

    # Iterate through max_iteration times
    for i in range(max_iterations):
        # Normalizing constants are given by the forward operation under current prior
        normalizing_constants = matmul_forward(current_prior)[:, np.newaxis]

        # Prevent division by zero
        normalizing_constants = np.where(
            normalizing_constants == 0, 1e-12, normalizing_constants
        )

        # Estimated counts are computed using the transpose/backward operator
        adjusted_counts = noisy_counts / normalizing_constants[:, 0]
        estimated_true_counts = matmul_transpose(adjusted_counts) * current_prior

        # Prior is updated to the normalized estimated counts
        new_prior = estimated_true_counts / estimated_true_counts.sum()
        diff = np.linalg.norm(new_prior - current_prior, ord=1)

        logger.debug("Iteration %d: L1 difference = %e", i + 1, diff)

        # Stop iterating if desired convergence is reached
        if diff < tolerance:
            logger.info(
                "Reached convergence tolerance (%e) using %d iterations",
                tolerance,
                i + 1,
            )
            break

        current_prior = new_prior
    else:
        logger.warning(
            "Reached max_iterations without reaching convergence tolerance (%e)",
            tolerance,
        )

    logger.debug(
        "Iterations complete. Final estimated true counts: %s", estimated_true_counts
    )
    return estimated_true_counts


def _validate_array_like(array, object_str):
    """Validates that an object can be transformed into a float 64 np.array"""
    try:
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{object_str} must be a numerical array-like, "
            f"but received Type {type(array)}. Details: {e}"
        ) from e
    return array


def _initialize_response_matrix(
    response_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Initialize the response matrix if it is valid"""
    # Validate response matrix is a numpy array
    response_matrix = _validate_array_like(response_matrix, "Response matrix")
    # Validate non-negative response matrix
    _validate_non_negative(response_matrix, "Response matrix entries")

    # Validate response matrix is 2 dimensional and square
    if (
        response_matrix.ndim != 2
        or response_matrix.shape[0] != response_matrix.shape[1]
    ):
        raise DimensionError(
            f"Response matrix {response_matrix.shape} must be a 2D square matrix."
        )

    # Validate response matrix columns sum to 1
    col_sums = np.sum(response_matrix, axis=0)
    if not np.allclose(col_sums, 1.0, atol=1e-3):
        raise InvalidResponseMatrixError(
            "Response matrix columns must sum to 1 (representing valid transition probabilities). "
            f"Found column sums: {col_sums}"
        )
    return response_matrix


def _validate_inputs(
    max_iterations: int,
    tolerance: float,
) -> None:
    """Validates shapes, dimensions, and physical constraints of all inputs."""
    # Validate max_iterations is an integer type (excluding booleans)
    if not isinstance(max_iterations, int) or isinstance(tolerance, bool):
        raise ValueError(
            f"Max iterations must be a positive integer, but received {max_iterations}."
        )
    # Validate positive max iterations
    if max_iterations <= 0:
        raise ValueError(
            f"Max iterations must be a positive integer, but received {max_iterations}."
        )

    # Validate tolerance is a numeric type (excluding booleans)
    if not isinstance(tolerance, numbers.Number) or isinstance(tolerance, bool):
        raise ValueError(
            f"Tolerance must be a positive number, but received {tolerance}."
        )

    # Validate positive tolerance
    if tolerance <= 0:
        raise ValueError(
            f"Tolerance must be a positive number, but received {tolerance}."
        )


def _initialize_prior(
    initial_prior: np.ndarray[tuple[int], np.dtype[np.float64]] | None,
    noisy_counts: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """Initializes a uniform or normalized user-inputted prior."""
    # Use inputted initial prior if given
    if initial_prior is not None:
        initial_prior = _validate_array_like(initial_prior, "Inputted prior")
        # Validate equal shape of prior and observed counts
        if initial_prior.shape != noisy_counts.shape:
            raise ShapeError(
                f"Initial prior shape {initial_prior.shape} and "
                f"observed counts shape {noisy_counts.shape} must be the same."
            )

        # Validate prior probabilities are non-negative
        _validate_non_negative(initial_prior, "Prior probabilities")
        prior_sum = initial_prior.sum()
        if not np.allclose(prior_sum, 1, 1e-3):
            logger.warning("Inputted prior is not normalized")

            # Normalize inputted prior
            current_prior = initial_prior / prior_sum
        else:
            current_prior = initial_prior
        logger.info("Using normalized custom initial prior.")
        logger.debug("Custom prior values: %s", current_prior)
    else:
        # Start with a uniform prior
        current_prior = np.ones(len(noisy_counts)) / len(noisy_counts)
        logger.info("Using a uniform initial prior.")
        logger.debug("Uniform prior values: %s", current_prior)
    return current_prior


def _validate_non_negative(array: np.ndarray, object_name: str) -> None:
    """Raises ValueError for negative values."""
    if np.any(array < 0):
        raise ValueError(f"{object_name} cannot be negative.")
