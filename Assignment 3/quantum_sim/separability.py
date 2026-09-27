"""Pure-state bipartite separability interface for Assignment 3."""

import numpy as np


def schmidt_rank(
    statevector: np.ndarray,
    subsystem_a: list[int] | tuple[int, ...],
    num_qubits: int,
    atol: float = 1e-9,
) -> int:
    """Return the Schmidt rank across the A|B bipartition. Students implement.

    Reshape/reorder amplitudes consistently with the selected Qiskit-style
    indexing convention, compute singular values, and count values > atol.
    """
    if statevector.size != 2**num_qubits:
        raise ValueError("statevector length does not match num_qubits.")

    subsystem_a = list(subsystem_a)
    if len(set(subsystem_a)) != len(subsystem_a):
        raise ValueError("subsystem_a must contain distinct qubits.")
    if any(not (0 <= q < num_qubits) for q in subsystem_a):
        raise ValueError("subsystem_a qubits out of range.")
    if not subsystem_a or len(subsystem_a) == num_qubits:
        raise ValueError("subsystem_a must be a proper, non-empty subset.")

    subsystem_b = [q for q in range(num_qubits) if q not in subsystem_a]

    dim_a = 2 ** len(subsystem_a)
    dim_b = 2 ** len(subsystem_b)
    coeff = np.zeros((dim_a, dim_b), dtype=complex)

    for i, amp in enumerate(statevector):
        a_index = 0
        for pos, q in enumerate(sorted(subsystem_a, reverse=True)):
            bit = (i >> q) & 1
            a_index = (a_index << 1) | bit
        b_index = 0
        for pos, q in enumerate(sorted(subsystem_b, reverse=True)):
            bit = (i >> q) & 1
            b_index = (b_index << 1) | bit
        coeff[a_index, b_index] = amp

    singular_values = np.linalg.svd(coeff, compute_uv=False)
    return int(np.sum(singular_values > atol))


def is_separable(
    statevector: np.ndarray,
    subsystem_a: list[int] | tuple[int, ...],
    num_qubits: int,
    atol: float = 1e-9,
) -> bool:
    """Return True exactly when the Schmidt rank is one."""
    return schmidt_rank(
        statevector,
        subsystem_a,
        num_qubits,
        atol=atol,
    ) == 1
