import numpy as np
from bayesian_qem.ibu import iterative_bayesian_unfolding

def main():
    # Define a noisy response matrix for a 1-qubit system
    # R[i, j] = P(measured i | true j)
    # State 0 is index 0, State 1 is index 1
    response_matrix = np.array([
        [0.9, 0.2],  # P(meas 0|true 0) = 0.9, P(meas 0|true 1) = 0.2
        [0.1, 0.8]   # P(meas 1|true 0) = 0.1, P(meas 1|true 1) = 0.8
    ])
    
    # Simulated noisy output counts: e.g., 60 counts of '0', 40 counts of '1'
    noisy_counts = np.array([60.0, 40.0])
    
    print(f"Raw Noisy Counts: {noisy_counts}")
    
    # Run IBU
    mitigated_counts = iterative_bayesian_unfolding(noisy_counts, response_matrix, max_iterations=5)
    
    print(f"Mitigated (True) Counts: {mitigated_counts}")

if __name__ == "__main__":
    main()
