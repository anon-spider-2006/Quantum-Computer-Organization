"""Tests for Assignment 3 local two-qubit gates.

Convention
----------
Statevectors use Qiskit-style ordering:

    |q_(n-1) ... q_1 q_0>

For two qubits, vector entries are ordered:

    |00>, |01>, |10>, |11>

For apply_two_qubit_unitary(U, control, target), the local 4x4
matrix U is interpreted in local basis order:

    |control target> = |00>, |01>, |10>, |11>.

Thus the standard local CX matrix flips `target` iff `control` is 1.
"""

import numpy as np
import pytest

from quantum_sim.gates import CX, H, I, SWAP, X
from quantum_sim.statevector import Statevector, is_unitary


ATOL = 1e-9


def basis_state(
    index: int,
    num_qubits: int = 2,
) -> Statevector:
    """Return the computational-basis state with the given integer index."""
    data = np.zeros(2**num_qubits, dtype=complex)
    data[index] = 1.0

    return Statevector(data)


def expected_basis_state(
    index: int,
    num_qubits: int = 2,
) -> np.ndarray:
    """Return a computational-basis statevector as a NumPy array."""
    data = np.zeros(2**num_qubits, dtype=complex)
    data[index] = 1.0

    return data


def test_cx_is_defined():
    assert CX is not None, (
        "Define CX in quantum_sim/gates.py."
    )


def test_swap_is_defined():
    assert SWAP is not None, (
        "Define SWAP in quantum_sim/gates.py."
    )

def test_cx_has_expected_shape():
    """CX must be a 4x4 two-qubit matrix."""
    assert CX.shape == (4, 4)


def test_swap_has_expected_shape():
    """SWAP must be a 4x4 two-qubit matrix."""
    assert SWAP.shape == (4, 4)


def test_cx_is_unitary():
    """CX dagger times CX must equal identity."""
    assert is_unitary(CX, atol=ATOL)


def test_swap_is_unitary():
    """SWAP dagger times SWAP must equal identity."""
    assert is_unitary(SWAP, atol=ATOL)


@pytest.mark.parametrize(
    ("input_index", "expected_index"),
    [
        (0, 0),  # CX |00> = |00>
        (1, 1),  # CX |01> = |01>
        (2, 3),  # CX |10> = |11>
        (3, 2),  # CX |11> = |10>
    ],
)
def test_cx_matrix_action_on_two_qubit_basis(
    input_index: int,
    expected_index: int,
):
    """The CX matrix has the required basis-state action."""
    state = expected_basis_state(input_index)
    expected = expected_basis_state(expected_index)

    obtained = CX @ state

    assert np.allclose(
        obtained,
        expected,
        rtol=0.0,
        atol=ATOL,
    )


@pytest.mark.parametrize(
    ("input_index", "expected_index"),
    [
        (0, 0),  # SWAP |00> = |00>
        (1, 2),  # SWAP |01> = |10>
        (2, 1),  # SWAP |10> = |01>
        (3, 3),  # SWAP |11> = |11>
    ],
)
def test_swap_matrix_action_on_two_qubit_basis(
    input_index: int,
    expected_index: int,
):
    """The SWAP matrix exchanges the two computational-basis bits."""
    state = expected_basis_state(input_index)
    expected = expected_basis_state(expected_index)

    obtained = SWAP @ state

    assert np.allclose(
        obtained,
        expected,
        rtol=0.0,
        atol=ATOL,
    )


@pytest.mark.parametrize(
    ("input_index", "expected_index"),
    [
        (0, 0),  # |00> -> |00>
        (1, 1),  # |01> -> |01>
        (2, 3),  # |10> -> |11>
        (3, 2),  # |11> -> |10>
    ],
)
def test_apply_cx_control_q1_target_q0(
    input_index: int,
    expected_index: int,
):
    """Apply CX with q1 as control and q0 as target."""
    state = basis_state(input_index)

    state.apply_two_qubit_unitary(
        CX,
        control=1,
        target=0,
    )

    assert np.allclose(
        state.data,
        expected_basis_state(expected_index),
        rtol=0.0,
        atol=ATOL,
    )


@pytest.mark.parametrize(
    ("input_index", "expected_index"),
    [
        (0, 0),  # q0 = 0: |00> -> |00>
        (1, 3),  # q0 = 1: |01> -> |11>
        (2, 2),  # q0 = 0: |10> -> |10>
        (3, 1),  # q0 = 1: |11> -> |01>
    ],
)
def test_apply_cx_control_q0_target_q1(
    input_index: int,
    expected_index: int,
):
    """Apply the same local CX matrix with q0 as control and q1 as target."""
    state = basis_state(input_index)

    state.apply_two_qubit_unitary(
        CX,
        control=0,
        target=1,
    )

    assert np.allclose(
        state.data,
        expected_basis_state(expected_index),
        rtol=0.0,
        atol=ATOL,
    )


@pytest.mark.parametrize(
    ("input_index", "expected_index"),
    [
        (0, 0),  # |00> -> |00>
        (1, 2),  # |01> -> |10>
        (2, 1),  # |10> -> |01>
        (3, 3),  # |11> -> |11>
    ],
)
def test_apply_swap_on_two_qubits(
    input_index: int,
    expected_index: int,
):
    """SWAP exchanges q0 and q1."""
    state = basis_state(input_index)

    state.apply_two_qubit_unitary(
        SWAP,
        control=1,
        target=0,
    )

    assert np.allclose(
        state.data,
        expected_basis_state(expected_index),
        rtol=0.0,
        atol=ATOL,
    )


def test_swap_is_its_own_inverse():
    """Applying SWAP twice restores every state."""
    state = Statevector(
        np.array(
            [
                1.0 + 1.0j,
                2.0 - 1.0j,
                -1.0 + 3.0j,
                0.5 - 2.0j,
            ],
            dtype=complex,
        )
    )

    original = state.data.copy()

    state.apply_two_qubit_unitary(
        SWAP,
        control=1,
        target=0,
    )
    state.apply_two_qubit_unitary(
        SWAP,
        control=1,
        target=0,
    )

    assert np.allclose(
        state.data,
        original,
        rtol=0.0,
        atol=ATOL,
    )


def test_cx_is_its_own_inverse():
    """Applying CX twice restores every state."""
    state = Statevector(
        np.array(
            [
                1.0 + 1.0j,
                2.0 - 1.0j,
                -1.0 + 3.0j,
                0.5 - 2.0j,
            ],
            dtype=complex,
        )
    )

    original = state.data.copy()

    state.apply_two_qubit_unitary(
        CX,
        control=1,
        target=0,
    )
    state.apply_two_qubit_unitary(
        CX,
        control=1,
        target=0,
    )

    assert np.allclose(
        state.data,
        original,
        rtol=0.0,
        atol=ATOL,
    )


def test_h_then_cx_prepares_phi_plus():
    """H on q1 followed by CX(q1, q0) prepares |Phi+>."""
    state = Statevector.zero(num_qubits=2)

    state.apply_unitary(H, target=1)
    state.apply_two_qubit_unitary(
        CX,
        control=1,
        target=0,
    )

    phi_plus = np.array(
        [
            1.0 / np.sqrt(2.0),
            0.0,
            0.0,
            1.0 / np.sqrt(2.0),
        ],
        dtype=complex,
    )

    assert np.allclose(
        state.data,
        phi_plus,
        rtol=0.0,
        atol=ATOL,
    )


def test_h_on_q0_then_cx_q0_q1_prepares_phi_plus():
    """H on q0 followed by CX(q0, q1) also prepares |Phi+>."""
    state = Statevector.zero(num_qubits=2)

    state.apply_unitary(H, target=0)
    state.apply_two_qubit_unitary(
        CX,
        control=0,
        target=1,
    )

    phi_plus = np.array(
        [
            1.0 / np.sqrt(2.0),
            0.0,
            0.0,
            1.0 / np.sqrt(2.0),
        ],
        dtype=complex,
    )

    assert np.allclose(
        state.data,
        phi_plus,
        rtol=0.0,
        atol=ATOL,
    )


def test_two_qubit_gate_inside_three_qubit_state():
    """CX must affect only the selected qubits of a larger state.

    Begin with |010>, for which q2=0, q1=1, q0=0.
    Apply CX(control=q1, target=q0), giving |011>.
    """
    state = basis_state(
        index=2,
        num_qubits=3,
    )

    state.apply_two_qubit_unitary(
        CX,
        control=1,
        target=0,
    )

    assert np.allclose(
        state.data,
        expected_basis_state(
            index=3,
            num_qubits=3,
        ),
        rtol=0.0,
        atol=ATOL,
    )


def test_three_qubit_cx_nonadjacent_qubits():
    """CX must work when control and target are not adjacent.

    Begin with |001>, where q0=1, q1=0, q2=0.
    Apply CX(control=q0, target=q2), giving |101>.
    """
    state = basis_state(
        index=1,
        num_qubits=3,
    )

    state.apply_two_qubit_unitary(
        CX,
        control=0,
        target=2,
    )

    assert np.allclose(
        state.data,
        expected_basis_state(
            index=5,
            num_qubits=3,
        ),
        rtol=0.0,
        atol=ATOL,
    )


def test_three_qubit_swap_nonadjacent_qubits():
    """SWAP must exchange selected nonadjacent qubits.

    Begin with |001>, where q2q1q0 = 001.
    Swapping q0 and q2 gives |100>.
    """
    state = basis_state(
        index=1,
        num_qubits=3,
    )

    state.apply_two_qubit_unitary(
        SWAP,
        control=0,
        target=2,
    )

    assert np.allclose(
        state.data,
        expected_basis_state(
            index=4,
            num_qubits=3,
        ),
        rtol=0.0,
        atol=ATOL,
    )


@pytest.mark.parametrize(
    "invalid_gate",
    [
        np.eye(2, dtype=complex),
        np.eye(3, dtype=complex),
        np.eye(5, dtype=complex),
        np.ones((4, 3), dtype=complex),
    ],
)
def test_rejects_gate_with_wrong_shape(
    invalid_gate: np.ndarray,
):
    """A local two-qubit gate must have shape (4, 4)."""
    state = Statevector.zero(num_qubits=2)

    with pytest.raises(ValueError):
        state.apply_two_qubit_unitary(
            invalid_gate,
            control=1,
            target=0,
        )


def test_rejects_nonunitary_two_qubit_gate():
    """A 4x4 nonunitary matrix must be rejected."""
    state = Statevector.zero(num_qubits=2)

    nonunitary = np.eye(4, dtype=complex)
    nonunitary[0, 1] = 1.0

    with pytest.raises(ValueError):
        state.apply_two_qubit_unitary(
            nonunitary,
            control=1,
            target=0,
        )


def test_rejects_same_control_and_target():
    """The two selected qubits must be distinct."""
    state = Statevector.zero(num_qubits=2)

    with pytest.raises(ValueError):
        state.apply_two_qubit_unitary(
            CX,
            control=0,
            target=0,
        )


@pytest.mark.parametrize(
    ("control", "target"),
    [
        (-1, 0),
        (0, -1),
        (2, 0),
        (0, 2),
        (3, 0),
        (0, 3),
    ],
)
def test_rejects_out_of_range_qubit_index(
    control: int,
    target: int,
):
    """Both selected qubits must lie in 0 through num_qubits - 1."""
    state = Statevector.zero(num_qubits=2)

    with pytest.raises(ValueError):
        state.apply_two_qubit_unitary(
            CX,
            control=control,
            target=target,
        )


def test_gate_application_preserves_normalization():
    """A unitary gate must preserve total probability."""
    state = Statevector(
        np.array(
            [
                1.0 + 2.0j,
                -2.0 + 1.0j,
                0.5 - 1.5j,
                3.0 + 0.5j,
            ],
            dtype=complex,
        )
    )

    state.apply_two_qubit_unitary(
        CX,
        control=1,
        target=0,
    )

    assert np.isclose(
        np.sum(state.probabilities()),
        1.0,
        rtol=0.0,
        atol=ATOL,
    )


def test_controlled_not_acts_linearly_on_superposition():
    """CX must act on all amplitudes, not only on basis-state inputs.

    Start with:
        (|00> + |10>) / sqrt(2)

    CX(control=q1, target=q0) must produce:
        (|00> + |11>) / sqrt(2).
    """
    state = Statevector(
        np.array(
            [
                1.0 / np.sqrt(2.0),
                0.0,
                1.0 / np.sqrt(2.0),
                0.0,
            ],
            dtype=complex,
        )
    )

    state.apply_two_qubit_unitary(
        CX,
        control=1,
        target=0,
    )

    expected = np.array(
        [
            1.0 / np.sqrt(2.0),
            0.0,
            0.0,
            1.0 / np.sqrt(2.0),
        ],
        dtype=complex,
    )

    assert np.allclose(
        state.data,
        expected,
        rtol=0.0,
        atol=ATOL,
    )

