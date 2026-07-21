import numpy as np


def iterative_bayesian_unfolding(
    noisy_counts: np.ndarray[tuple[int], np.dtype[np.float64]],
    response_matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    max_iterations: int = 10,
    tolerance: float = 1e-6,
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:

    current_prior: np.ndarray = np.ones(len(noisy_counts)) / len(noisy_counts)
    estimated_true_counts: np.ndarray = noisy_counts

    for _ in range(max_iterations):
        normalizing_constants: np.ndarray = (
            response_matrix.dot(current_prior)[:, np.newaxis]
        )
        posterior_matrix: np.ndarray = (
            response_matrix * current_prior[np.newaxis, :] / normalizing_constants
        )
        estimated_true_counts = posterior_matrix.T.dot(noisy_counts)
        new_prior: np.ndarray = estimated_true_counts / estimated_true_counts.sum()
        diff: float = np.linalg.norm(new_prior - current_prior, ord=1)
        if diff < tolerance:
            break
        current_prior = new_prior
    return estimated_true_counts
