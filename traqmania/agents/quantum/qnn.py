"""Qiskit execution paths for the canonical circuit via EstimatorQNN (V2 primitives).

``QiskitQFunction`` implements the same ``QFunction`` contract (and the same
flat parameter layout ``[lam(L*n), theta(L*n*2), w(A), b(A)]``) as the numpy
fast path in ``qdqn.py``, but evaluates the circuit through
qiskit-machine-learning's ``EstimatorQNN`` on a choice of Aer backends:

- ``aer_statevector``: exact expectation values (Aer EstimatorV2, precision 0).
- ``aer_shots``: shot-based sampling (precision = 1/sqrt(shots)).
- ``aer_noisy``: shot-based with a noise model — built from a
  ``qiskit_ibm_runtime`` fake backend when that package is installed, else a
  simple depolarizing model (so this path works without qiskit-ibm-runtime).

The trainable input scalings ``lam`` are handled analytically outside the
circuit: the circuit's input parameters are the encoding angles
``x[l*n + i] = lam[l, i] * s[i]``, and with ``input_gradients=True``

    dE/dlam[l, i] = s[i] * dE/dx[l, i].

Qiskit is imported lazily (inside ``__init__``) so importing this module stays
cheap; the package ``__init__`` additionally loads this submodule lazily.
"""

from __future__ import annotations

import numpy as np

BACKENDS = ("aer_statevector", "aer_shots", "aer_noisy")


def _depolarizing_noise_model():
    """Simple depolarizing noise on the circuit's native gates (no transpilation)."""
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    noise_model = NoiseModel(basis_gates=["ry", "rz", "cz"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(1e-3, 1), ["ry", "rz"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(1e-2, 2), ["cz"])
    return noise_model


def _fake_backend(name: str | None):
    """A fake IBM backend instance, or None if qiskit-ibm-runtime is unavailable."""
    try:
        from qiskit_ibm_runtime import fake_provider
    except ImportError:
        return None
    cls = getattr(fake_provider, name or "FakeManilaV2", None)
    return cls() if cls is not None else None


class QiskitQFunction:
    """Q-function on Aer via EstimatorQNN; drop-in twin of ``QuantumQFunction``."""

    def __init__(
        self,
        circuit_cfg: dict,
        backend: str = "aer_statevector",
        shots: int = 1024,
        noise_backend_name: str | None = None,
        seed: int | None = None,
    ):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        cfg = circuit_cfg.get("circuit", circuit_cfg)
        self.n_qubits = int(cfg.get("n_qubits", 4))
        self.n_layers = int(cfg.get("n_layers", 4))
        if seed is None:
            seed = int(cfg.get("seed", 7))
        self.seed = int(seed)
        self.backend = backend
        self.shots = int(shots)

        self.n_features = self.n_qubits
        self.n_actions = self.n_qubits  # one Z_a readout per action

        # Same initialization (and rng stream) as the numpy fast path.
        rng = np.random.default_rng(self.seed)
        self.lam = np.full((self.n_layers, self.n_qubits), np.pi, dtype=np.float64)
        self.theta = rng.uniform(-0.1, 0.1, size=(self.n_layers, self.n_qubits, 2))
        self.w = np.ones(self.n_actions, dtype=np.float64)
        self.b = np.zeros(self.n_actions, dtype=np.float64)

        self._qnn = self._build_qnn(noise_backend_name)

    def _build_qnn(self, noise_backend_name: str | None):
        from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
        from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
        from qiskit_machine_learning.neural_networks import EstimatorQNN

        from traqmania.agents.quantum import circuit as circuit_mod

        qc = circuit_mod.build_circuit(self.n_qubits, self.n_layers)
        input_params, weight_params = circuit_mod.split_parameters(qc)
        obs = circuit_mod.observables(self.n_qubits)

        backend_options: dict = {"seed_simulator": self.seed}
        pass_manager = None
        exact = self.backend == "aer_statevector"
        # Aer EstimatorV2 computes exact expectation values at precision 0.
        precision = 0.0 if exact else 1.0 / float(np.sqrt(self.shots))
        if self.backend == "aer_noisy":
            fake = _fake_backend(noise_backend_name)
            if fake is not None:
                from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
                from qiskit_aer import AerSimulator

                sim = AerSimulator.from_backend(fake)
                backend_options["noise_model"] = sim.options.noise_model
                backend_options["basis_gates"] = sim.configuration().basis_gates
                pm = generate_preset_pass_manager(optimization_level=1, backend=sim)
                # Transpile once here and map the observables onto the physical
                # qubits ourselves: EstimatorQNN's own pass_manager path runs the
                # circuit through the pass manager but leaves the observables on
                # the virtual qubits, which Aer's EstimatorV2 rejects when the
                # fake backend has more qubits than the circuit.
                qc = pm.run(qc)
                obs = [o.apply_layout(qc.layout) for o in obs]
            else:
                backend_options["noise_model"] = _depolarizing_noise_model()

        # backend_options seeds the simulation itself; run_options seeds the
        # N(expval, precision) sampling Aer's EstimatorV2 applies at precision > 0.
        estimator = AerEstimatorV2(
            options={
                "backend_options": backend_options,
                "run_options": {"seed_simulator": self.seed},
            }
        )
        gradient = ParamShiftEstimatorGradient(estimator, pass_manager=pass_manager)
        return EstimatorQNN(
            circuit=qc,
            estimator=estimator,
            observables=obs,
            input_params=input_params,
            weight_params=weight_params,
            gradient=gradient,
            input_gradients=True,
            default_precision=precision,
            pass_manager=pass_manager,
        )

    def _encoding_angles(self, obs: np.ndarray) -> np.ndarray:
        """x (B, L*n) with x[b, l*n + i] = lam[l, i] * obs[b, i]."""
        return (self.lam[None, :, :] * obs[:, None, :]).reshape(
            obs.shape[0], self.n_layers * self.n_qubits
        )

    def _expectations(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self._qnn.forward(self._encoding_angles(obs), self.theta.ravel()))

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """Q-values for a batch of observations: (B, F) -> (B, A)."""
        obs = np.asarray(obs, dtype=np.float64)
        return self._expectations(obs) * self.w + self.b

    def grad_selected(
        self, obs: np.ndarray, action_idx: np.ndarray, upstream: np.ndarray
    ) -> np.ndarray:
        """Gradient of ``sum_b upstream[b] * Q[b, action_idx[b]]`` wrt the flat params."""
        obs = np.asarray(obs, dtype=np.float64)
        action_idx = np.asarray(action_idx, dtype=np.int64)
        upstream = np.asarray(upstream, dtype=np.float64)
        batch = obs.shape[0]
        rows = np.arange(batch)

        x = self._encoding_angles(obs)
        expectations = np.asarray(self._qnn.forward(x, self.theta.ravel()))
        input_grads, weight_grads = self._qnn.backward(x, self.theta.ravel())
        input_grads = np.asarray(input_grads)  # (B, A, L*n) d<Z_a>/dx
        weight_grads = np.asarray(weight_grads)  # (B, A, L*n*2) d<Z_a>/dtheta

        # Chain rule through the output head: dQ_sel/dE_sel = w[a_b].
        coeff = upstream * self.w[action_idx]  # (B,)
        dx_sel = input_grads[rows, action_idx, :].reshape(batch, self.n_layers, self.n_qubits)
        # x[l, i] = lam[l, i] * s[i]  =>  dE/dlam[l, i] = s[i] * dE/dx[l, i].
        dlam = np.einsum("b,bli,bi->li", coeff, dx_sel, obs)
        dtheta = coeff @ weight_grads[rows, action_idx, :]

        dw = np.zeros(self.n_actions, dtype=np.float64)
        db = np.zeros(self.n_actions, dtype=np.float64)
        np.add.at(dw, action_idx, upstream * expectations[rows, action_idx])
        np.add.at(db, action_idx, upstream)
        return np.concatenate([dlam.ravel(), dtheta.ravel(), dw, db])

    @property
    def n_params(self) -> int:
        return self.lam.size + self.theta.size + self.w.size + self.b.size

    def get_params(self) -> np.ndarray:
        """Copy of the flat parameter vector, shape (P,)."""
        return np.concatenate([self.lam.ravel(), self.theta.ravel(), self.w, self.b])

    def set_params(self, params: np.ndarray) -> None:
        """Load a flat parameter vector, shape (P,)."""
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.n_params,):
            raise ValueError(f"params must have shape ({self.n_params},), got {params.shape}")
        n_lam = self.lam.size
        n_theta = self.theta.size
        n_w = self.w.size
        self.lam = params[:n_lam].reshape(self.lam.shape).copy()
        self.theta = params[n_lam : n_lam + n_theta].reshape(self.theta.shape).copy()
        self.w = params[n_lam + n_theta : n_lam + n_theta + n_w].copy()
        self.b = params[n_lam + n_theta + n_w :].copy()
