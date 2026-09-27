"""Observable helpers for Assignment 3."""

import numpy as np

_PAULI = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

def pauli_string_matrix(labels: str) -> np.ndarray:
    """Optionally construct a dense Pauli-string matrix. Students implement."""
    if not labels:
        raise ValueError("labels must be non-empty.")
    if any(c not in _PAULI for c in labels):
        raise ValueError("labels must contain only I, X, Y, Z.")

    matrix = np.array([[1.0]], dtype=complex)
    for c in labels:
        matrix = np.kron(matrix, _PAULI[c])
    return matrix


def is_hermitian(matrix: np.ndarray, atol: float = 1e-9) -> bool:
    """Return whether matrix equals its adjoint within atol."""
    matrix = np.asarray(matrix, dtype=complex)
    return matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and np.allclose(
        matrix,
        matrix.conj().T,
        rtol=0.0,
        atol=atol,
    )
