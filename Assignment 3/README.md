# Assignment 3 Starter: N-Qubit Statevector Simulation

This template extends Assignment 2 from a single-qubit statevector to
an n-qubit simulator.  Copy your working Assignment 2 implementation
into this layout where appropriate, then extend it rather than
replacing it wholesale.

## Running the programs and tests

Run all commands from the project root directory, which is the directory that
contains the `quantum_sim/`, `tests/`, and `examples/` directories.

As with Assignment 2, the project requires Python, NumPy, and pytest.
Numpy should already be installed along with Qiskit, and pytest was
installed in Assignment 2.  If you need to install it:

On CCI Power PC install pytest from conda-forge
```bash
conda install conda-forge::pytest --yes
```

On a laptop use pip.
```bash
python -m pip install pytest
```

## Required convention

Use the Qiskit-style little-endian convention throughout.

- Qubit `0` is the least-significant bit of an integer basis index.

- For two qubits, the statevector entries are ordered
`|00>, |01>, |10>, |11>`.

- A displayed bit string is `q_(n-1)...q_0`; therefore qubit `0` is
the rightmost displayed bit.

- For a Pauli label such as `"ZI"`, the left character acts on the
highest-index qubit: `Z` acts on `q1` and `I` on `q0`.


Document any local two-qubit-gate convention you choose and apply it
consistently in code, tests, examples, and JSON results.

## Layout

```text
assignment3_starter/
├── quantum_sim/
│   ├── __init__.py
│   ├── gates.py
│   ├── measurement.py
│   ├── observables.py
│   ├── separability.py
│   └── statevector.py
├── examples/
│   ├── bell_chsh.py
│   └── separability_demo.py
├── tests/
│   ├── test_measurement.py
│   ├── test_separability.py
│   ├── test_statevector.py
│   └── test_two_qubit_gates.py
├── pyproject.toml
└── README.md
```

## Assignment milestones

1. **Reuse Assignment 2:** retain normalization, one-qubit gates,
   single-qubit expectation values, and seeded sampling where they
   remain useful.
   
2. **Generalize statevectors:** support `2**n` amplitudes and
   `Statevector.zero(num_qubits)`.
   
3. **Apply local gates:** implement one-qubit local gates and 4-by-4
   two-qubit gates without constructing a full `2**n` by `2**n` matrix
   for routine operations.
   
4. **Measurements:** implement computational-basis measurements of all
   qubits and selected qubits. A partial measurement must return an
   outcome and a normalized collapsed n-qubit state.
   
5. **Expectation values:** implement one-qubit Pauli and Pauli-string
   expectation values.
   
6. **Separable states:** implement Schmidt rank with an SVD-based
   method and use it to decide whether a pure bipartite state is
   separable across a supplied bipartition.
   
7. **Bell/CHSH example:** prepare an entangled state, implement the
   requested measurement settings, and report the CHSH-game results in
   the supplied JSON schema.
   '
8. **Scaling:** benchmark time and memory as the number of qubits
   grows. Discuss why dense statevectors require exponentially many
   amplitudes.
   

## JSON rule

The example scripts own the output format. Do not redesign the
top-level schema. Convert all NumPy scalar values to Python `int`,
`float`, or `bool` before `json.dump`. Convert a complex value `z` as:

```python
{"real": float(z.real), "imag": float(z.imag)}
```

Never write `NaN` or `Infinity`; use `allow_nan=False` when dumping
JSON. The supplied templates intentionally contain placeholder values
and therefore will not produce valid result files until implemented.

## Setup and checks

From the project root:

```bash
python -m pip install -e .
pytest -q
python examples/bell_chsh.py
python examples/separability_demo.py
```

The current tests are scaffolding tests: they confirm the starter
interfaces and expected unfinished methods. Replace placeholder tests
with behavioral tests as the implementation is completed.

### Two-qubit gate convention

For `apply_two_qubit_unitary(unitary, control, target)`, the local 4x4
basis order is |00>, |01>, |10>, |11>, where the first local bit is
`control` and the second local bit is `target`. Under this convention,
the local CX matrix flips `target` iff `control` is 1:

    CX = [[1,0,0,0],
          [0,1,0,0],
          [0,0,0,1],
          [0,0,1,0]]

This convention is applied consistently in `gates.py`, `statevector.py`
(`apply_two_qubit_unitary`), `test_two_qubit_gates.py`, and `bell_chsh.py`

### Required separability cases

Your `separability_demo.py` must test:

- `|00>`, `|01>`, `|10>`, and `|11>`.
- All four Bell states: `Psi+`, `Psi-`, `Phi+`, and `Phi-`.
- At least five fixed complex two-qubit states.
- At least two of the five complex states must be entangled.

Use Schmidt rank across the `q0 | q1` bipartition. Product states have
Schmidt rank 1. The Bell states have Schmidt rank 2.

### Required CHSH experiment

Use `shots_per_setting = 1000` and `seed = 42`.

Evaluate all four setting pairs `(x, y)` in `{0, 1} x {0, 1}`. A trial wins
when:

    a XOR b == x AND y

The best classical win probability is `0.75`. The target quantum win
probability for the supplied optimal strategy is:

    cos(pi / 8)**2 ~= 0.853553

Your output must report individual setting counts, total wins, total shots,
the observed win probability, and whether the observed result exceeds the
classical bound.

## Suggested benchmark table

For selected values of `n`, record:

- `n`
- number of amplitudes, `2**n`
- statevector bytes, approximately `16 * 2**n` for NumPy `complex128`
- time to apply a representative one-qubit gate
- time to apply a representative two-qubit gate
- time and memory environment used for the measurement

Avoid benchmarking machines with insufficient RAM. Dense statevectors scale exponentially.
