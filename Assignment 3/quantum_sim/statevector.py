"""Statevector interface for Assignment 3.

Convention: q0 is the least-significant bit of a basis index.  For two
qubits, data entries are ordered |00>, |01>, |10>, |11>.
"""

import numpy as np


def is_unitary(matrix: np.ndarray, atol: float = 1e-9) -> bool:
    """Return whether matrix is square and unitary within atol."""
    matrix = np.asarray(matrix, dtype=complex)
    return matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and np.allclose(
        matrix.conj().T @ matrix,
        np.eye(matrix.shape[0], dtype=complex),
        rtol=0.0,
        atol=atol,
    )

_PAULI = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

class Statevector:
    """Normalized pure n-qubit statevector using Qiskit-style ordering."""

    def __init__(self, data: np.ndarray):
        data = np.asarray(data, dtype=complex).reshape(-1)
        if data.size < 2 or data.size & (data.size - 1):
            raise ValueError("Statevector length must be a power of two and at least 2.")
        norm = np.linalg.norm(data)
        if np.isclose(norm, 0.0):
            raise ValueError("Statevector cannot have zero norm.")
        self.data = data / norm
        self.num_qubits = int(np.log2(data.size))

    @classmethod
    def zero(cls, num_qubits: int = 1) -> "Statevector":
        """Return |0...0>."""
        if num_qubits < 1:
            raise ValueError("num_qubits must be at least 1.")
        data = np.zeros(2**num_qubits, dtype=complex)
        data[0] = 1.0
        return cls(data)

    def copy(self) -> "Statevector":
        """Return an independent copy."""
        return Statevector(self.data.copy())

    def probabilities(self) -> np.ndarray:
        """Return computational-basis probabilities."""
        return np.abs(self.data) ** 2

    def _check_target(self, target: int) -> None:
            if not (0 <= target < self.num_qubits):
                raise ValueError(f"target {target} out of range for {self.num_qubits} qubits.")

    def apply_unitary(self, unitary: np.ndarray, target: int = 0) -> None:
        """Apply a one-qubit 2x2 unitary to target."""
        unitary = np.asarray(unitary, dtype=complex)
        if unitary.shape != (2, 2):
            raise ValueError("apply_unitary requires a 2x2 matrix.")
        if not is_unitary(unitary):
            raise ValueError("apply_unitary requires a unitary matrix.")
        self._check_target(target)

        n = self.data.size
        mask = 1 << target
        for i in range(n):
            if i & mask:
                continue  # only the bit=0 rep
            j = i | mask
            a0 = self.data[i]
            a1 = self.data[j]
            self.data[i] = unitary[0, 0] * a0 + unitary[0, 1] * a1
            self.data[j] = unitary[1, 0] * a0 + unitary[1, 1] * a1


    def apply_two_qubit_unitary(\
        self,
        unitary: np.ndarray,
        control: int,
        target: int,
    ) -> None:
        """Apply a 4x4 two-qubit unitary.

        Local basis: for the ordered pair (control, target), local index
        = control_bit * 2 + target_bit, i.e. control is the higher local
        bit and target is the lower local bit -- matching |00>,|01>,|10>,|11>
        with control on the left. This matches gates.CX/CZ/SWAP as defined
        in the handout.
        """
        unitary = np.asarray(unitary, dtype=complex)
        if unitary.shape != (4, 4):
            raise ValueError("apply_two_qubit_unitary requires a 4x4 matrix.")
        if not is_unitary(unitary):
            raise ValueError("apply_two_qubit_unitary requires a unitary matrix.")
        if control == target:
            raise ValueError("control and target must be distinct.")
        self._check_target(control)
        self._check_target(target)

        n = self.data.size
        cmask = 1 << control
        tmask = 1 << target
        both = cmask | tmask

        for i in range(n):
            if i & both:
                continue  # only process the representative with both bits = 0
            # four global indices for local (c,t) in {0,1}x{0,1}
            i00 = i
            i01 = i | tmask
            i10 = i | cmask
            i11 = i | both
            amps = np.array(
                [self.data[i00], self.data[i01], self.data[i10], self.data[i11]],
                dtype=complex,
            )
            new_amps = unitary @ amps
            self.data[i00] = new_amps[0]
            self.data[i01] = new_amps[1]
            self.data[i10] = new_amps[2]
            self.data[i11] = new_amps[3]

    def measure(self, targets: list[int] | tuple[int, ...], seed: int | None = None) -> tuple[str, "Statevector"]:
        """Measure selected qubits and return (Qiskit-order bit string, collapsed state). """

        targets = list(targets)
        if not targets:
            raise ValueError("targets must be non-empty.")
        if len(set(targets)) != len(targets):
            raise ValueError("targets must be distinct.")
        for t in targets:
            self._check_target(t)

        probs = self.probabilities()
        n = self.data.size
        sorted_targets = sorted(targets, reverse=True)  # highest qubit index first

        # group basis indices by the outcome pattern on the measured qubits
        outcome_probs: dict[int, float] = {}
        for i in range(n):
            pattern = 0
            for t in sorted_targets:
                bit = (i >> t) & 1
                pattern = (pattern << 1) | bit
            outcome_probs[pattern] = outcome_probs.get(pattern, 0.0) + probs[i]

        patterns = sorted(outcome_probs.keys())
        weights = np.array([outcome_probs[p] for p in patterns])
        weights = weights / weights.sum()  # accounting for small fp drift

        rng = np.random.default_rng(seed)
        chosen_idx = rng.choice(len(patterns), p=weights)
        chosen_pattern = patterns[chosen_idx]

        outcome = "".join(
            str((chosen_pattern >> (len(sorted_targets) - 1 - k)) & 1)
            for k in range(len(sorted_targets))
        )

        new_data = np.zeros_like(self.data)
        for i in range(n):
            pattern = 0
            for t in sorted_targets:
                bit = (i >> t) & 1
                pattern = (pattern << 1) | bit
            if pattern == chosen_pattern:
                new_data[i] = self.data[i]

        collapsed = Statevector(new_data)
        return outcome, collapsed

    def expectation_pauli(self, pauli: np.ndarray, target: int = 0) -> float:
        """Return <psi|P|psi> for a one-qubit observable. Students implement."""
        pauli = np.asarray(pauli, dtype=complex)
        if pauli.shape != (2, 2):
            raise ValueError("expectation_pauli requires a 2x2 matrix.")
        self._check_target(target)

        n = self.data.size
        mask = 1 << target
        total = 0.0 + 0.0j
        for i in range(n):
            if i & mask:
                continue
            j = i | mask
            a0 = self.data[i]
            a1 = self.data[j]
            # <a0,a1| P |a0,a1>
            b0 = pauli[0, 0] * a0 + pauli[0, 1] * a1
            b1 = pauli[1, 0] * a0 + pauli[1, 1] * a1
            total += np.conj(a0) * b0 + np.conj(a1) * b1

        if abs(total.imag) > 1e-6:
            raise ValueError(f"Expectation value has large imaginary part: {total}")
        
        return float(total.real)

    def expectation_pauli_string(self, labels: str) -> float:
        """Return the expectation of a Pauli string. Students implement.

        Example: labels='ZI' means Z on q1 and I on q0.
        """
        if len(labels) != self.num_qubits:
            raise ValueError(
                f"labels length {len(labels)} does not match num_qubits {self.num_qubits}."
            )
        if any(c not in _PAULI for c in labels):
            raise ValueError("labels must contain only I, X, Y, Z.")

        n = self.data.size
        # psi_out = (P_{n-1}  ...  P_0) |psi>
        out = self.data.copy()
        for qubit in range(self.num_qubits):
            # labels[0] is qubit num_qubits-1, labels[-1] is qubit 0
            label_char = labels[self.num_qubits - 1 - qubit]
            p = _PAULI[label_char]
            if label_char == "I":
                continue
            mask = 1 << qubit
            new_out = out.copy()
            for i in range(n):
                if i & mask:
                    continue
                j = i | mask
                a0 = out[i]
                a1 = out[j]
                new_out[i] = p[0, 0] * a0 + p[0, 1] * a1
                new_out[j] = p[1, 0] * a0 + p[1, 1] * a1
            out = new_out

        total = np.vdot(self.data, out)  # <psi|psi_out>
        if abs(total.imag) > 1e-6:
            raise ValueError(f"Expectation value has large imaginary part: {total}")
        return float(total.real)
