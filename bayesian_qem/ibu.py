"""
ibu.py

File that contains the function iterative_bayesian_unfolding.
"""

import logging
import numpy as np
from .exceptions import DimensionError, InvalidResponseMatrixError, ShapeError

# Initialize a module-level logger
logger = logging.getLogger(__name__)


def iterative_bayesian_unfolding(
    noisy_counts: np.ndarray[tuple[int], np.dtype[np.float64]],
    response_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    initial_prior: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
    max_iterations: int = 10,
    tolerance: float = 1e-6,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """
    Solves for likely physically accurate true outputs from
    a quantum circuit's noisy outputs and calibration classifications.

    Args:
        noisy_counts: The observed output counts from the quantum circuit.
        response_matrix: The square calibration response matrix where rows
            correspond to measured states and columns correspond to true states
            (entry [i, j] is the probability of measuring state i given true state j).
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
    # Validate the noisy counts and response matrix
    _validate_inputs(noisy_counts, response_matrix, max_iterations, tolerance)

    current_prior = _initialize_prior(initial_prior, noisy_counts)

    # Initialize the best estimate for counts
    estimated_true_counts = noisy_counts
    logger.info(
        "Iteration initialized with max_iterations=%d, tolerance=%e",
        max_iterations,
        tolerance,
    )
    # Iterate through max_iteration times
    for i in range(max_iterations):
        # Normalizing constants are given by the sum of responses under prior
        normalizing_constants = (response_matrix @ current_prior)[:, np.newaxis]

        # Posterior matrix is the likelihood * prior / normalizing constants
        posterior_matrix = (
            response_matrix * current_prior[np.newaxis, :] / normalizing_constants
        )

        # Estimated counts are the posterior matrix times noisy counts
        estimated_true_counts = posterior_matrix.T @ noisy_counts

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


def _validate_inputs(
    noisy_counts: np.ndarray[tuple[int], np.dtype[np.float64]],
    response_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    max_iterations: int,
    tolerance: float,
) -> None:
    """Validates shapes, dimensions, and physical constraints of all inputs."""
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

    # Validate observed counts match the response matrix
    if noisy_counts.shape[0] != response_matrix.shape[0]:
        raise DimensionError(
            f"Response matrix of shape: {response_matrix.shape} and "
            f"observed counts of shape: {noisy_counts.shape} are mismatched dimensions."
        )

    # Validate non-negative response matrix
    _validate_non_negative(response_matrix, "Response matrix entries")

    # Validate non-negative noisy counts
    _validate_non_negative(noisy_counts, "Noisy counts")

    # Validate max_iterations is an integer type
    if not isinstance(max_iterations, int):
        raise ValueError(
            f"Max iterations must be an integer, but received {max_iterations}."
        )
    # Validate positive max iterations
    if max_iterations <= 0:
        raise ValueError(
            f"Max iterations must be a positive integer, but received {max_iterations}."
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
