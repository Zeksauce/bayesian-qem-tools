"""
backends.py

File that contains the logic to decide which backend operations to use for IBU
"""
import numpy as np


def apply_dense_response(vector: np.ndarray, response_matrix: np.ndarray) -> np.ndarray:
    """Standard dense matrix multiplication."""
    return response_matrix @ vector

def apply_dense_transpose(vector: np.ndarray, response_matrix: np.ndarray) -> np.ndarray:
    """Standard dense transpose multiplication."""
    return response_matrix.T @ vector

def apply_tensor_response(vector: np.ndarray, local_matrices: list[np.ndarray]) -> np.ndarray:
    """Matrix-free tensor-product multiplication for uncorrelated qubits."""
    matrix_size = len(local_matrices)
    tensor = vector.reshape((2,) * matrix_size)
    subscripts_in = list(range(matrix_size))
    
    for i, response_matrix in enumerate(local_matrices):
        out_subscripts = subscripts_in.copy()
        out_subscripts[i] = matrix_size + i
        ein_str = f"ab, {','.join(map(str, subscripts_in))} -> {','.join(map(str, out_subscripts))}"
        tensor = np.einsum(ein_str, response_matrix, tensor)
        subscripts_in[i] = matrix_size + i
    return tensor.reshape(-1)

def apply_tensor_transpose(vector: np.ndarray, local_matrices: list[np.ndarray]) -> np.ndarray:
    """Matrix-free tensor-product transpose multiplication using transposed local matrices."""
    transposed_locals = [response_matrix.T for response_matrix in local_matrices]
    return apply_tensor_response(vector, transposed_locals)