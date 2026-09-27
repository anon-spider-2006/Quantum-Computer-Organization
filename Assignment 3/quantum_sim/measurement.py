"""Sampling helpers for Assignment 3."""

from .statevector import Statevector
import numpy as np


def sample_counts(state: Statevector, shots: int, seed: int | None = None) -> dict[str, int]:
    """Sample all qubits in the computational basis. Students implement."""
    if shots <= 0:
        raise ValueError("shots must be positive.")

    probs = state.probabilities()
    n = state.num_qubits
    rng = np.random.default_rng(seed)
    outcomes = rng.choice(len(probs), size=shots, p=probs)

    counts: dict[str, int] = {}
    for outcome in outcomes:
        key = format(int(outcome), f"0{n}b")
        counts[key] = counts.get(key, 0) + 1

    return counts