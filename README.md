# Bayesian QEM Tools (`bayesian-qem-tools`)

A modular Python toolkit implementing Iterative Bayesian Unfolding (IBU) and related Bayesian post-processing methods for **Quantum Error Mitigation (QEM)**. 

Designed for researchers and developers looking to recover physically consistent probability distributions from noisy NISQ-era quantum hardware without running into the pitfalls of traditional linear inversion.

---

## Why Bayesian Unfolding?

In current quantum processors, measurement readout noise and gate errors distort experimental results. Standard mitigation approaches (like direct matrix inversion) often fail because:
* **Non-Physical Results:** They frequently produce negative probabilities or values greater than 1.
* **Ill-Conditioned Amplification:** Small statistical fluctuations in your shot counts are magnified by inverse matrices, leading to wild oscillations.

**Iterative Bayesian Unfolding (IBU)**—also known in high-energy physics as D'Agostini unfolding—treats noise correction as a statistical inference problem. By iteratively applying Bayes' theorem, this toolkit ensures that your mitigated results naturally respect fundamental physical constraints (probabilities always sum to 1 and remain non-negative).

---

## Project Structure

```text
bayesian-qem-tools/
│
├── bayesian_qem/         # Core mitigation library
│   ├── __init__.py
│   └── ibu.py            # Iterative Bayesian Unfolding (IBU) implementation
│
├── examples/             # Executable demonstrations
│   └── single_qubit_demo.py
│
├── requirements.txt      # Project dependencies
├── LICENSE               # Apache 2.0 License
└── README.md
```
## Installation & Setup
Clone the repository:

```Bash
git clone https://github.com/Zeksauce/bayesian-qem-tools.git
cd bayesian-qem-tools
```
Install dependencies:

```Bash
pip install -r requirements.txt
```
## Quick Start Example
Here is how you can use the IBU module to correct a noisy single-qubit measurement distribution:

```Python
import numpy as np
from ibu_qem.ibu import iterative_bayesian_unfolding

# 1. Define your noise response (confusion) matrix: R[i, j] = P(measured i | true j)
response_matrix = np.array([
    [0.9, 0.2],  # P(meas 0 | true 0) = 0.9, P(meas 0 | true 1) = 0.2
    [0.1, 0.8]   # P(meas 1 | true 0) = 0.1, P(meas 1 | true 1) = 0.8
])

# 2. Raw noisy output counts from your quantum circuit execution
noisy_counts = np.array([60.0, 40.0])

print(f"Raw Noisy Counts: {noisy_counts}")

# 3. Apply Iterative Bayesian Unfolding
mitigated_counts = iterative_bayesian_unfolding(
    noisy_counts=noisy_counts, 
    response_matrix=response_matrix, 
    max_iterations=5
)

print(f"Mitigated (True) Counts: {mitigated_counts}")
```
To run the full demonstration script:

```Bash
python examples/single_qubit_demo.py
```
## Roadmap
[ ] Core Iterative Bayesian Unfolding (IBU) engine for readout mitigation

[ ] Support for multi-qubit tensor product response matrices

[ ] Integration wrappers for Qiskit / Mitiq output formats

[ ] Bayesian Parameter Estimation modules for gate-level noise characterization

## Contributing
Contributions, suggestions, and feature requests are welcome! If you are working on advanced error mitigation or hybrid quantum-classical workflows, feel free to open an issue or submit a pull request.

## License
This project is open-source and available under the Apache License 2.0.
