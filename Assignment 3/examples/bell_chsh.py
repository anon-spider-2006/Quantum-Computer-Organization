"""Assignment 3 demonstration template: Bell state and CHSH game.

Demonstrate Bell-state correlations using the CHSH game.

Run from the project root:
    python examples/bell_chsh.py

Assignment requirements
-----------------------
- Prepare the Bell state |Phi+> = (|00> + |11>) / sqrt(2).
- Use 1000 shots for each of the four input pairs (x, y).
- Use seed = 42 for reproducibility.
- Alice and Bob receive x, y in {0, 1}.
- They produce output bits a, b in {0, 1}.
- A trial wins when:

      a XOR b == x AND y

The best classical strategy wins with probability at most 3/4.
The supplied optimal quantum strategy has target probability:

    cos(pi / 8)**2 ~= 0.853553

Qubit-ordering convention
--------------------------
This project uses Qiskit-style ordering:

- q0 is the least-significant qubit.
- Two-qubit statevector entries are ordered:
      |00>, |01>, |10>, |11>.
- Displayed measurement bit strings are q1q0.
- In an outcome string such as "10":
      outcome[0] is q1 (Bob),
      outcome[1] is q0 (Alice).

This file intentionally provides the output schema and experiment
constants. Students implement the state preparation, basis rotations,
measurement sampling, and game statistics.
"""

import json
from pathlib import Path
from quantum_sim.statevector import Statevector
from quantum_sim.gates import H, CX, ry
from quantum_sim.measurement import sample_counts

import numpy as np


SHOTS_PER_SETTING = 1000
SEED = 42

CLASSICAL_WIN_BOUND = 0.75
QUANTUM_TARGET_WIN_PROBABILITY = float(
    np.cos(np.pi / 8.0) ** 2
)

# Alice's observables:
#     A0 = Z
#     A1 = X
#
# Bob's observables:
#     B0 = (Z + X) / sqrt(2)
#     B1 = (Z - X) / sqrt(2)
#
# With |Phi+>, these give the optimal CHSH value S = 2 sqrt(2)
# and CHSH-game win probability cos(pi / 8)^2.
SETTING_DESCRIPTIONS = {
    (0, 0): {
        "alice_observable": "Z",
        "bob_observable": "(Z + X) / sqrt(2)",
        "alice_basis_rotation": "I",
        "bob_basis_rotation": "Ry(-pi / 4)",
    },
    (0, 1): {
        "alice_observable": "Z",
        "bob_observable": "(Z - X) / sqrt(2)",
        "alice_basis_rotation": "I",
        "bob_basis_rotation": "Ry(+pi / 4)",
    },
    (1, 0): {
        "alice_observable": "X",
        "bob_observable": "(Z + X) / sqrt(2)",
        "alice_basis_rotation": "H",
        "bob_basis_rotation": "Ry(-pi / 4)",
    },
    (1, 1): {
        "alice_observable": "X",
        "bob_observable": "(Z - X) / sqrt(2)",
        "alice_basis_rotation": "H",
        "bob_basis_rotation": "Ry(+pi / 4)",
    },
}


def complex_to_json(value: complex) -> dict[str, float]:
    """Convert one complex number to the required JSON representation."""
    return {
        "real": float(np.real(value)),
        "imag": float(np.imag(value)),
    }


def vector_to_json(
    vector: np.ndarray,
) -> list[dict[str, float]]:
    """Convert a complex vector to a list of JSON complex objects."""
    return [
        complex_to_json(value)
        for value in np.asarray(vector, dtype=complex).reshape(-1)
    ]


def counts_with_all_outcomes(
    counts: dict[str, int],
) -> dict[str, int]:
    """Return deterministic q1q0 ordering for all two-qubit outcomes."""
    return {
        outcome: int(counts.get(outcome, 0))
        for outcome in ("00", "01", "10", "11")
    }


def wins_chsh_game(
    x: int,
    y: int,
    a: int,
    b: int,
) -> bool:
    """Return whether outputs a, b win the CHSH game for inputs x, y."""
    return (a ^ b) == (x & y)


def outcome_to_alice_bob_bits(
    outcome: str,
) -> tuple[int, int]:
    """Convert Qiskit-order bit string q1q0 to CHSH bits (a, b).

    Alice owns q0, the rightmost displayed bit.
    Bob owns q1, the leftmost displayed bit.

    For example:
        "00" -> (a=0, b=0)
        "01" -> (a=1, b=0)
        "10" -> (a=0, b=1)
        "11" -> (a=1, b=1)
    """
    if outcome not in {"00", "01", "10", "11"}:
        raise ValueError(
            "Expected a two-bit Qiskit-order outcome string."
        )

    bob_bit = int(outcome[0])
    alice_bit = int(outcome[1])

    return alice_bit, bob_bit


def count_wins(
    counts: dict[str, int],
    x: int,
    y: int,
) -> int:
    """Count CHSH wins from a dictionary of Qiskit-order outcomes."""
    wins = 0

    for outcome, count in counts.items():
        a, b = outcome_to_alice_bob_bits(outcome)

        if wins_chsh_game(x, y, a, b):
            wins += int(count)

    return wins


def prepare_phi_plus() -> np.ndarray:
    """Return the statevector for |Phi+>.

    Implement this by starting from |00>, applying H to q1,
    then applying CX with q1 as control and q0 as target.

    In Qiskit-order basis entries |00>, |01>, |10>, |11>, the expected
    statevector is:

        [1 / sqrt(2), 0, 0, 1 / sqrt(2)]

    Raises
    ------
    NotImplementedError
        Until students implement state preparation using their simulator.
    """
    state = Statevector.zero(2)
    state.apply_unitary(H, target=1)
    state.apply_two_qubit_unitary(CX, control=1, target=0)
    return state.data


def sample_chsh_setting(
    x: int,
    y: int,
    shots: int,
    seed: int,
) -> dict[str, int]:
    """Return q1q0 measurement counts for one CHSH input pair.

    1. Prepare a new |Phi+> state.
    2. Apply Alice's basis rotation to q0.
    3. Apply Bob's basis rotation to q1.
    4. Measure both qubits in the computational basis.
    5. Return a dictionary of counts for q1q0 strings.

    The setting-dependent rotations are specified in SETTING_DESCRIPTIONS.
    For reproducibility, use the supplied integer seed.

    Raises
    ------
    NotImplementedError
        Until students implement the experiment with their simulator.
    """
    state = Statevector(prepare_phi_plus())

    # Alice owns q0.
    if x == 1:
        state.apply_unitary(H, target=0)
    # x == 0 -> "I"

    # Bob owns q1.
    if y == 0:
        state.apply_unitary(ry(-np.pi / 4), target=1)
    else:
        state.apply_unitary(ry(np.pi / 4), target=1)

    return sample_counts(state, shots=shots, seed=seed)


def setting_result(
    x: int,
    y: int,
    shots: int,
    seed: int,
) -> dict:
    """Run one CHSH setting and construct its required JSON record."""
    counts = counts_with_all_outcomes(
        sample_chsh_setting(
            x=x,
            y=y,
            shots=shots,
            seed=seed,
        )
    )

    total_counts = int(sum(counts.values()))

    if total_counts != shots:
        raise ValueError(
            f"Expected {shots} samples, but received {total_counts}."
        )

    wins = count_wins(counts, x=x, y=y)
    probability = wins / shots

    return {
        "x": int(x),
        "y": int(y),
        **SETTING_DESCRIPTIONS[(x, y)],
        "shots": int(shots),
        "seed": int(seed),
        "counts": counts,
        "total_counts": total_counts,
        "wins": int(wins),
        "win_probability": float(probability),
        "passed": bool(
            total_counts == shots
        ),
    }


def main() -> None:
    """Run all four CHSH input settings and write JSON results."""
    phi_plus = prepare_phi_plus()

    results = {
        "assignment": "assignment3",
        "experiment": "bell_chsh",
        "seed": SEED,
        "shots_per_setting": SHOTS_PER_SETTING,
        "total_shots": 4 * SHOTS_PER_SETTING,
        "qiskit_ordering": (
            "Statevector entries and outcome strings use "
            "|00>, |01>, |10>, |11>; displayed strings are q1q0; "
            "Alice is q0 and Bob is q1."
        ),
        "initial_state": {
            "name": "phi_plus",
            "statevector": vector_to_json(phi_plus),
        },
        "win_condition": "a XOR b == x AND y",
        "classical_win_bound": CLASSICAL_WIN_BOUND,
        "quantum_target_win_probability": (
            QUANTUM_TARGET_WIN_PROBABILITY
        ),
        "optimal_angle_radians": float(np.pi / 8.0),
        "settings": [],
    }

    # Separate seeds make each setting reproducible while avoiding
    # accidental reuse of a random-number-generator state.
    for setting_index, (x, y) in enumerate(
        ((0, 0), (0, 1), (1, 0), (1, 1))
    ):
        results["settings"].append(
            setting_result(
                x=x,
                y=y,
                shots=SHOTS_PER_SETTING,
                seed=SEED + setting_index,
            )
        )

    total_wins = int(
        sum(
            setting["wins"]
            for setting in results["settings"]
        )
    )

    total_shots = int(
        sum(
            setting["total_counts"]
            for setting in results["settings"]
        )
    )

    observed_probability = total_wins / total_shots

    results["total_wins"] = total_wins
    results["observed_win_probability"] = float(
        observed_probability
    )
    results["classical_bound_exceeded"] = bool(
        observed_probability > CLASSICAL_WIN_BOUND
    )
    results["passed"] = bool(
        results["classical_bound_exceeded"]
        and all(
            setting["passed"]
            for setting in results["settings"]
        )
    )

    output_path = Path(__file__).with_name(
        "assignment3_results.json"
    )

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            results,
            handle,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")

    print("CHSH game demonstration")
    print(f"Shots per setting: {SHOTS_PER_SETTING}")
    print(f"Total shots: {total_shots}")
    print(
        "Classical winning bound: "
        f"{CLASSICAL_WIN_BOUND:.6f}"
    )
    print(
        "Quantum target probability: "
        f"{QUANTUM_TARGET_WIN_PROBABILITY:.6f}"
    )
    print(
        "Observed winning probability: "
        f"{observed_probability:.6f}"
    )
    print(
        "Classical bound exceeded: "
        f"{results['classical_bound_exceeded']}"
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
