"""
ibu.py

File that contains the function iterative_bayesian_unfolding.
"""

import numpy as np


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
            Defaults to None.
        max_iterations: Optional maximum number of IBU iterations to run.
            Defaults to 10.
        tolerance: Optional early stopping condition for convergence based on L1 norm.
            Defaults to 1e-6.

    Returns:
        The estimated true state counts of the quantum circuit.

    Example:
        >>> import numpy as np
        >>> response = np.array([[0.9, 0.1], [0.2, 0.8]])
        >>> counts = np.array([55.0, 45.0])
        >>> iterative_bayesian_unfolding(counts, response)
        array([67.64677211, 32.35322789])
    """
    # Use inputted initial prior if given
    if initial_prior is not None:
        # Normalize inputted prior
        current_prior = initial_prior / initial_prior.sum()
    else:
        # Start with a uniform prior
        current_prior = np.ones(len(noisy_counts)) / len(noisy_counts)

    # Initialize the best estimate for counts
    estimated_true_counts = noisy_counts

    # Iterate through max_iteration times
    for _ in range(max_iterations):
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

        # Stop iterating if desired convergence is reached
        if diff < tolerance:
            break

        current_prior = new_prior

    return estimated_true_counts
