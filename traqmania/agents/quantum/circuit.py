"""Canonical Qiskit circuit definition for traQmania (single source of truth).

Circuit spec (n qubits, L data re-uploading blocks, acting on |0...0>):

    block l = 0..L-1:
        for qubit i: RY(x[l, i]) on qubit i        # x[l, i] = lam[l, i] * s[i]
        for qubit i: RY(theta[l, i, 0]) then RZ(theta[l, i, 1]) on qubit i
        CZ ring: CZ(0, 1), CZ(1, 2), ..., CZ(n-1, 0)
    readout: E_a = <Z_a>, Q_a = w[a] * E_a + b[a]   # output head lives outside

The encoding angle is exposed as a separate input Parameter ``x[l*n + i]``;
the product ``x[l, i] = lam[l, i] * s[i]`` is computed OUTSIDE the circuit
(the input-scaling parameters ``lam`` are trainable but the circuit itself is
linear in the bound angle). Parameter order matches the flat layout:
``x`` indexed ``l*n + i``, ``theta`` indexed ``(l*n + i)*2 + k`` (k=0 RY, k=1 RZ).

Gate conventions: RY(phi) = exp(-i phi Y/2), RZ(phi) = exp(-i phi Z/2)
(Qiskit's). Qubit a maps to Qiskit qubit a; Qiskit statevectors are
little-endian, so Z on qubit a is the Pauli string "I"*(n-1-a) + "Z" + "I"*a.

Qiskit is imported lazily inside functions so importing this module stays cheap
and ``circuit_spec`` works without qiskit at all.
"""

from __future__ import annotations


def build_circuit(n_qubits: int = 4, n_layers: int = 4):
    """Build the canonical parameterized QuantumCircuit.

    Input parameters: ParameterVector "x" of length L*n (encoding angles,
    x[l*n + i] = lam[l, i] * s[i], computed outside). Weight parameters:
    ParameterVector "theta" of length L*n*2, flat order (l, i, k) with k=0 the
    RY angle and k=1 the RZ angle — the same order as theta.ravel() in the
    numpy fast path. Use :func:`split_parameters` to recover both orderings.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    x = ParameterVector("x", n_layers * n_qubits)
    theta = ParameterVector("theta", n_layers * n_qubits * 2)

    qc = QuantumCircuit(n_qubits, name="traqmania_qcircuit")
    for layer in range(n_layers):
        for i in range(n_qubits):
            qc.ry(x[layer * n_qubits + i], i)
        for i in range(n_qubits):
            qc.ry(theta[(layer * n_qubits + i) * 2], i)
            qc.rz(theta[(layer * n_qubits + i) * 2 + 1], i)
        for i in range(n_qubits):
            qc.cz(i, (i + 1) % n_qubits)
    return qc


def split_parameters(qc) -> tuple[list, list]:
    """Return (input_params, weight_params) of a built circuit, in flat order.

    input_params: the "x" ParameterVector elements ordered by index (l*n + i).
    weight_params: the "theta" elements ordered by index ((l*n + i)*2 + k).
    """
    by_name: dict[str, list] = {"x": [], "theta": []}
    for p in qc.parameters:
        by_name[p.vector.name].append(p)
    return (
        sorted(by_name["x"], key=lambda p: p.index),
        sorted(by_name["theta"], key=lambda p: p.index),
    )


def observables(n_qubits: int = 4) -> list:
    """[Z_0, ..., Z_{n-1}] as SparsePauliOp, little-endian string convention."""
    from qiskit.quantum_info import SparsePauliOp

    return [
        SparsePauliOp("I" * (n_qubits - 1 - a) + "Z" + "I" * a) for a in range(n_qubits)
    ]


def circuit_spec(config: dict) -> dict:
    """JSON-serializable, layer-by-layer description of the circuit (no qiskit).

    ``config`` may be a full traQmania config dict (with a "circuit" section)
    or the [circuit] section itself. Intended for the browser circuit diagram.
    """
    cfg = config.get("circuit", config) if isinstance(config.get("circuit"), dict) else config
    n = int(cfg.get("n_qubits", 4))
    layers = int(cfg.get("n_layers", 4))

    gates: list[dict] = []
    for layer in range(layers):
        for i in range(n):
            gates.append({"type": "ry_enc", "qubit": i, "layer": layer})
        for i in range(n):
            gates.append({"type": "ry", "qubit": i, "layer": layer})
            gates.append({"type": "rz", "qubit": i, "layer": layer})
        for i in range(n):
            gates.append({"type": "cz", "q0": i, "q1": (i + 1) % n, "layer": layer})

    return {
        "n_qubits": n,
        "n_layers": layers,
        "n_actions": n,
        "gates": gates,
        "counts": {
            "ry_enc": layers * n,
            "ry": layers * n,
            "rz": layers * n,
            "cz": layers * n,
            "total": 4 * layers * n,
        },
        "n_params": {
            "lam": layers * n,
            "theta": layers * n * 2,
            "w": n,
            "b": n,
            "total": layers * n * 3 + 2 * n,
        },
        "param_layout": ["lam", "theta", "w", "b"],
        "readout": [f"Z_{a}" for a in range(n)],
    }
