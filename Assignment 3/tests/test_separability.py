import numpy as np
import pytest

from quantum_sim.separability import schmidt_rank, is_separable


# Deleted placeholder 

"""Tests for Assignment 3 Schmidt-rank separability."""

ATOL = 1e-9


def sv(*amps):
    return np.array(amps, dtype=complex)


CASES = {
    "basis_00": (sv(1, 0, 0, 0), 1, True),
    "basis_01": (sv(0, 1, 0, 0), 1, True),
    "basis_10": (sv(0, 0, 1, 0), 1, True),
    "basis_11": (sv(0, 0, 0, 1), 1, True),
    "psi_plus": (sv(0, 1, 1, 0) / np.sqrt(2), 2, False),
    "psi_minus": (sv(0, 1, -1, 0) / np.sqrt(2), 2, False),
    "phi_plus": (sv(1, 0, 0, 1) / np.sqrt(2), 2, False),
    "phi_minus": (sv(1, 0, 0, -1) / np.sqrt(2), 2, False),
}


@pytest.mark.parametrize("name", CASES.keys())
def test_schmidt_rank_basis_and_bell_states(name):
    state, expected_rank, expected_separable = CASES[name]
    assert schmidt_rank(state, subsystem_a=[0], num_qubits=2, atol=ATOL) == expected_rank
    assert is_separable(state, subsystem_a=[0], num_qubits=2, atol=ATOL) == expected_separable


def test_complex_product_1():
    # |c1> = (|0>+i|1>)/sqrt(2) ⊗ (sqrt(3)|0>+i2|1>)/2   [check normalization below]
    q1 = sv(1, 1j) / np.sqrt(2)
    q0 = sv(np.sqrt(3), 2j) / 2  # NOTE: verify against handout, this may not be normalized
    state = np.kron(q1, q0)
    assert schmidt_rank(state, subsystem_a=[0], num_qubits=2, atol=ATOL) == 1
    assert is_separable(state, subsystem_a=[0], num_qubits=2, atol=ATOL) is True


def test_complex_entangled_1():
    # |c4> = (|00> + i|11>) / sqrt(2)
    state = sv(1, 0, 0, 1j) / np.sqrt(2)
    assert schmidt_rank(state, subsystem_a=[0], num_qubits=2, atol=ATOL) == 2
    assert is_separable(state, subsystem_a=[0], num_qubits=2, atol=ATOL) is False


def test_complex_entangled_2():
    # |c5> = (|00> + (1+i)|11>) / sqrt(3)
    state = sv(1, 0, 0, 1 + 1j) / np.sqrt(3)
    assert schmidt_rank(state, subsystem_a=[0], num_qubits=2, atol=ATOL) == 2
    assert is_separable(state, subsystem_a=[0], num_qubits=2, atol=ATOL) is False


def test_schmidt_rank_rejects_empty_subsystem():
    bell = sv(1, 0, 0, 1) / np.sqrt(2)
    with pytest.raises(ValueError):
        schmidt_rank(bell, subsystem_a=[], num_qubits=2)


def test_schmidt_rank_rejects_full_subsystem():
    bell = sv(1, 0, 0, 1) / np.sqrt(2)
    with pytest.raises(ValueError):
        schmidt_rank(bell, subsystem_a=[0, 1], num_qubits=2)


def test_schmidt_rank_rejects_duplicate_qubits():
    bell = sv(1, 0, 0, 1) / np.sqrt(2)
    with pytest.raises(ValueError):
        schmidt_rank(bell, subsystem_a=[0, 0], num_qubits=2)


def test_schmidt_rank_rejects_out_of_range_qubit():
    bell = sv(1, 0, 0, 1) / np.sqrt(2)
    with pytest.raises(ValueError):
        schmidt_rank(bell, subsystem_a=[5], num_qubits=2)