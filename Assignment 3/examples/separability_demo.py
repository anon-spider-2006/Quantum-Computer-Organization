"""Assignment 3 demonstration template: Schmidt rank and separability."""

"""Demonstrate Schmidt-rank separability tests for two-qubit pure states.

Run from the project root:
    python examples/separability_demo.py
"""

import json
from pathlib import Path

import numpy as np

from quantum_sim.separability import is_separable, schmidt_rank


def complex_to_json(value: complex) -> dict[str, float]:
    """Convert one complex number to assignment JSON format."""
    return {
        "real": float(np.real(value)),
        "imag": float(np.imag(value)),
    }


def vector_to_json(
    vector: np.ndarray,
) -> list[dict[str, float]]:
    """Convert a complex vector to assignment JSON format."""
    return [
        complex_to_json(value)
        for value in np.asarray(vector, dtype=complex).reshape(-1)
    ]


def make_cases() -> list[dict]:
    """Return fixed product and entangled two-qubit test states."""
    zero = np.array([1.0, 0.0], dtype=complex)
    one = np.array([0.0, 1.0], dtype=complex)

    psi_plus = np.array(
        [0.0, 1.0, 1.0, 0.0],
        dtype=complex,
    ) / np.sqrt(2.0)

    psi_minus = np.array(
        [0.0, 1.0, -1.0, 0.0],
        dtype=complex,
    ) / np.sqrt(2.0)

    phi_plus = np.array(
        [1.0, 0.0, 0.0, 1.0],
        dtype=complex,
    ) / np.sqrt(2.0)

    phi_minus = np.array(
        [1.0, 0.0, 0.0, -1.0],
        dtype=complex,
    ) / np.sqrt(2.0)

    return [
        {
            "name": "basis_00",
            "state": np.kron(zero, zero),
            "expected_schmidt_rank": 1,
            "expected_separable": True,
        },
        {
            "name": "basis_01",
            "state": np.kron(zero, one),
            "expected_schmidt_rank": 1,
            "expected_separable": True,
        },
        {
            "name": "basis_10",
            "state": np.kron(one, zero),
            "expected_schmidt_rank": 1,
            "expected_separable": True,
        },
        {
            "name": "basis_11",
            "state": np.kron(one, one),
            "expected_schmidt_rank": 1,
            "expected_separable": True,
        },
        {
            "name": "psi_plus",
            "state": psi_plus,
            "expected_schmidt_rank": 2,
            "expected_separable": False,
        },
        {
            "name": "psi_minus",
            "state": psi_minus,
            "expected_schmidt_rank": 2,
            "expected_separable": False,
        },
        {
            "name": "phi_plus",
            "state": phi_plus,
            "expected_schmidt_rank": 2,
            "expected_separable": False,
        },
        {
            "name": "phi_minus",
            "state": phi_minus,
            "expected_schmidt_rank": 2,
            "expected_separable": False,
        },
        {
            "name": "complex_product_1",
            "state": np.kron(
                np.array([1.0, 1j]) / np.sqrt(2.0),
                np.array(
                    [
                        np.sqrt(3.0) / 2.0,
                        0.5j,
                    ],
                    dtype=complex,
                ),
            ),
            "expected_schmidt_rank": 1,
            "expected_separable": True,
        },
        {
            "name": "complex_product_2",
            "state": np.kron(
                np.array(
                    [
                        2.0 - 1j,
                        1.0 + 2j,
                    ],
                    dtype=complex,
                ) / np.sqrt(10.0),
                np.array(
                    [
                        1.0,
                        -1j,
                    ],
                    dtype=complex,
                ) / np.sqrt(2.0),
            ),
            "expected_schmidt_rank": 1,
            "expected_separable": True,
        },
        {
            "name": "complex_product_3",
            "state": np.kron(
                np.array(
                    [
                        np.sqrt(3.0) / 2.0,
                        0.5,
                    ],
                    dtype=complex,
                ),
                np.array(
                    [
                        1.0 + 1j,
                        1.0 - 1j,
                    ],
                    dtype=complex,
                ) / 2.0,
            ),
            "expected_schmidt_rank": 1,
            "expected_separable": True,
        },
        {
            "name": "complex_entangled_1",
            "state": np.array(
                [
                    1.0,
                    0.0,
                    0.0,
                    1j,
                ],
                dtype=complex,
            ) / np.sqrt(2.0),
            "expected_schmidt_rank": 2,
            "expected_separable": False,
        },
        {
            "name": "complex_entangled_2",
            "state": np.array(
                [
                    1.0,
                    0.0,
                    0.0,
                    1.0 + 1j,
                ],
                dtype=complex,
            ) / np.sqrt(3.0),
            "expected_schmidt_rank": 2,
            "expected_separable": False,
        },
    ]


def main() -> None:
    """Evaluate the required two-qubit separability cases."""
    results = {
        "assignment": "assignment3",
        "experiment": "separability",
        "qiskit_ordering": (
            "basis entries are |00>, |01>, |10>, |11>; "
            "displayed bit strings are q1q0"
        ),
        "bipartition": {
            "subsystem_a": [0],
            "subsystem_b": [1],
        },
        "states": [],
    }

    for case in make_cases():
        state = case["state"]

        obtained_rank = schmidt_rank(
            state,
            subsystem_a=[0],
            num_qubits=2,
            atol=1e-9,
        )

        obtained_separable = is_separable(
            state,
            subsystem_a=[0],
            num_qubits=2,
            atol=1e-9,
        )

        passed = (
            obtained_rank
            == case["expected_schmidt_rank"]
            and obtained_separable
            == case["expected_separable"]
        )

        results["states"].append(
            {
                "name": case["name"],
                "statevector": vector_to_json(state),
                "expected_schmidt_rank": int(
                    case["expected_schmidt_rank"]
                ),
                "obtained_schmidt_rank": int(
                    obtained_rank
                ),
                "expected_separable": bool(
                    case["expected_separable"]
                ),
                "obtained_separable": bool(
                    obtained_separable
                ),
                "passed": bool(passed),
            }
        )

    output_path = Path(__file__).with_name(
        "separability_results.json"
    )

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            results,
            handle,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")

    failures = [
        entry["name"]
        for entry in results["states"]
        if not entry["passed"]
    ]

    print("Schmidt-rank separability demonstration")
    print(f"Cases: {len(results['states'])}")
    print(f"Failures: {failures}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

