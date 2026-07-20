import numpy as np

def iterative_bayesian_unfolding(noisy_counts: np.ndarray, response_matrix: np.ndarray, max_iterations: int = 10, tol: float = 1e-6) -> np.ndarray:
    """
    Performs Iterative Bayesian Unfolding (IBU) / D'Agostini unfolding 
    to mitigate readout noise in quantum measurement data.
    
    Parameters:
    - noisy_counts: 1D numpy array of observed counts for each state.
    - response_matrix: 2D numpy array where R[i, j] = P(measured i | true j).
    - max_iterations: Maximum number of update steps.
    - tol: Convergence tolerance threshold.
    
    Returns:
    - estimated_true_counts: 1D numpy array of mitigated counts.
    """
    num_states = len(noisy_counts)
    
    # 1. Initialize prior with a uniform distribution over the states
    # (or start with the normalized noisy counts)
    current_prior = np.ones(num_states) / num_states
    
    total_counts = np.sum(noisy_counts)
    
    for iteration in range(max_iterations):
        old_prior = current_prior.copy()
        
        # 2. Compute P(true_j | measured_i) using Bayes' Theorem
        # R is shape (states, states): R[i, j] = P(i | j)
        # We need denominator: P(i) = sum_k R[i, k] * prior[k]
        denominator = np.dot(response_matrix, current_prior)
        
        # Avoid division by zero
        denominator = np.where(denominator == 0, 1e-12, denominator)
        
        # P(j | i) matrix calculation
        # We want a matrix M[j, i] = P(true = j | measured = i)
        # M[j, i] = (R[i, j] * prior[j]) / denominator[i]
        likelihood_matrix = response_matrix * current_prior[np.newaxis, :] # Broadcasting along columns
        posterior_matrix = likelihood_matrix / denominator[:, np.newaxis]
        
        # 3. Estimate new true counts: n_hat_j = sum_i (noisy_counts_i * P(j | i))
        estimated_true_counts = np.dot(posterior_matrix.T, noisy_counts)
        
        # 4. Re-normalize to get the new prior probability distribution
        current_prior = estimated_true_counts / np.sum(estimated_true_counts)
        
        # Check for convergence
        diff = np.linalg.norm(current_prior - old_prior, ord=1)
        if diff < tol:
            break
            
    # Scale back to total counts
    return estimated_true_counts