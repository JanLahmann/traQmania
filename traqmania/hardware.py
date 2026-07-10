"""IBM Quantum hardware execution path: real/fake backends, inference, SPSA sprints.

This module is the bridge from traQmania's numpy fast path to actual quantum
processors (or their local fake twins) via ``qiskit-ibm-runtime``:

- :func:`get_backend` — a real IBMBackend through ``QiskitRuntimeService`` or a
  local noise-model twin from the fake provider.
- :class:`HardwareQFunction` — the same ``QFunction`` contract (and the same
  flat 56-parameter layout ``[lam, theta, w, b]``) as ``QuantumQFunction``,
  but INFERENCE ONLY: expectation values come from ``EstimatorV2`` PUBs on an
  ISA-transpiled circuit. Gradients are deliberately not implemented — a
  param-shift gradient would cost 2 * 48 circuit evaluations per batch, which
  is why hardware fine-tuning uses SPSA (two evaluations, period).
- :func:`run_hardware_lap` — greedy rollout of one car with every steering
  decision made by the quantum backend, inside a runtime ``Session``.
- :func:`spsa_sprint` — a short TD-loss fine-tune: replay batch and double-DQN
  targets are computed ONCE in the exact fastsim simulator, then SPSA descends
  the hardware-evaluated MSE loss at 2 Estimator jobs per iteration.

Import hygiene: importing this module must NOT import qiskit — all qiskit /
qiskit-ibm-runtime imports live inside functions. Run the CLI with
``python -m traqmania.hardware lap|sprint --track oval [--fake] ...``.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from traqmania.agents.training import spsa

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

# Known-good small fake backends to fall back through (5 qubits, V2 API).
_FAKE_FALLBACKS = ("FakeManilaV2", "FakeLimaV2", "FakeBelemV2", "FakeQuitoV2")

_SERVICE_HELP = (
    "Could not reach IBM Quantum. To run on real hardware:\n"
    "  1. Create an account at https://quantum.cloud.ibm.com and copy your API token.\n"
    "  2. Either export QISKIT_IBM_TOKEN=<token>, or save it once with\n"
    "     QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token=...).\n"
    "  3. Re-run. Or pass --fake to use a local noise-model twin instead."
)


# --------------------------------------------------------------------- backends


def ensure_qiskit_imported() -> None:
    """Import qiskit from the calling thread — which must be a LONG-LIVED one.

    qiskit's compiled extension pins process-wide lazy state to the thread
    that first imports it; if that thread exits, the next primitives job on
    any other thread segfaults (observed deterministically with qiskit 2.5.0:
    SIGSEGV in ``SparseObservable.to_sparse_list`` via
    ``ObservablesArray.__array__`` on the SECOND hardware job of a process
    whose first job ran on a since-exited thread). The demo server runs each
    hardware job on a short-lived worker thread, so it calls this first from
    the session thread, which outlives every worker. Idempotent; the first
    call pays the one-off qiskit import cost (~1 s).
    """
    import qiskit  # noqa: F401
    import qiskit_ibm_runtime  # noqa: F401


def _fake_class_name(fake_name: str) -> str:
    """'fake_manila' -> 'FakeManilaV2' (already-camel-cased names pass through)."""
    if fake_name.startswith("Fake"):
        return fake_name
    name = "".join(part.capitalize() for part in fake_name.split("_"))
    return name if name.endswith("V2") else name + "V2"


def _fake_backend(fake_name: str):
    from qiskit_ibm_runtime import fake_provider

    candidates = [_fake_class_name(fake_name)] if fake_name else []
    candidates += [n for n in _FAKE_FALLBACKS if n not in candidates]
    # Last resort: anything in the fake provider that looks like a V2 backend.
    candidates += sorted(
        n for n in dir(fake_provider) if n.startswith("Fake") and n.endswith("V2")
    )
    tried: list[str] = []
    for cls_name in candidates:
        cls = getattr(fake_provider, cls_name, None)
        if cls is None or cls_name in tried:
            continue
        tried.append(cls_name)
        try:
            backend = cls()
        except Exception:  # noqa: BLE001 - some snapshots fail to load; keep trying
            continue
        if getattr(backend, "num_qubits", 0) >= 5:
            return backend
    raise RuntimeError(
        f"no working 5+ qubit FakeBackendV2 found in qiskit_ibm_runtime.fake_provider "
        f"(tried {tried[:8]}...)"
    )


def _real_backend(name: str | None):
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("QISKIT_IBM_TOKEN")
    try:
        service = QiskitRuntimeService(token=token) if token else QiskitRuntimeService()
    except Exception as exc:
        raise RuntimeError(f"{_SERVICE_HELP}\n(underlying error: {exc})") from exc
    try:
        if name:
            return service.backend(name)
        return service.least_busy(operational=True, simulator=False, min_num_qubits=5)
    except Exception as exc:
        raise RuntimeError(
            f"could not get a backend from IBM Quantum "
            f"({'name=' + name if name else 'least busy'}): {exc}\n{_SERVICE_HELP}"
        ) from exc


def get_backend(name: str | None = None, use_fake: bool = False, fake_name: str = "fake_manila"):
    """A qiskit backend: real via ``QiskitRuntimeService`` or a local fake twin.

    ``use_fake=True`` returns a ``FakeBackendV2`` (noise model + coupling map of
    a retired 5-qubit device, simulated locally — no account needed). Otherwise
    the runtime service is reached with a token from ``QISKIT_IBM_TOKEN`` or the
    saved account; ``name`` picks a device, empty means least busy. Raises
    ``RuntimeError`` with setup instructions when the service is unavailable.
    """
    if use_fake:
        return _fake_backend(fake_name)
    return _real_backend(name)


def _make_session(backend) -> tuple[Any, str | None]:
    """Try to open a runtime Session on ``backend``; (session, note-on-fallback).

    Fake backends run Sessions in local testing mode; if a backend rejects
    Sessions entirely we fall back to ``mode=backend`` (each Estimator job is
    then scheduled independently) and say so in the note.
    """
    try:
        from qiskit_ibm_runtime import Session

        return Session(backend=backend), None
    except Exception as exc:  # noqa: BLE001
        name = getattr(backend, "name", backend)
        return None, f"Session unsupported for {name} ({exc}); using mode=backend"


# ---------------------------------------------------------------- Q-function


class HardwareQFunction:
    """Inference-only ``QFunction`` on an IBM (or fake) backend via EstimatorV2.

    Same flat 56-parameter layout as ``QuantumQFunction``:
    ``[lam (L*n), theta (L*n*2), w (A), b (A)]`` with
    ``Q_a = w[a] * <Z_a> + b[a]``. The circuit is transpiled to ISA form ONCE
    at construction; every ``q_values`` call is a single Estimator job whose
    PUB batches all observation rows (parameter bindings of shape ``(B, ...)``)
    against all four ``Z_a`` observables.
    """

    def __init__(self, circuit_cfg: dict, backend, shots: int = 1024, session=None):
        cfg = circuit_cfg.get("circuit", circuit_cfg)
        self.n_qubits = int(cfg.get("n_qubits", 4))
        self.n_layers = int(cfg.get("n_layers", 4))
        self.seed = int(cfg.get("seed", 7))
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

        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import EstimatorV2

        from traqmania.agents.quantum import circuit as circuit_mod

        qc = circuit_mod.build_circuit(self.n_qubits, self.n_layers)
        self._input_params, self._weight_params = circuit_mod.split_parameters(qc)
        pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
        self._isa_circuit = pass_manager.run(qc)
        self._isa_observables = [
            obs.apply_layout(self._isa_circuit.layout)
            for obs in circuit_mod.observables(self.n_qubits)
        ]
        self._estimator = EstimatorV2(mode=session if session is not None else backend)
        self._estimator.options.default_shots = self.shots

    def expectations(self, obs: np.ndarray) -> np.ndarray:
        """Raw readout expectations <Z_a> from the backend: (B, F) -> (B, A)."""
        obs = np.atleast_2d(np.asarray(obs, dtype=np.float64))
        batch = obs.shape[0]
        # x[b, l*n + i] = lam[l, i] * obs[b, i] — encoding angles bound per row.
        x = (self.lam[None, :, :] * obs[:, None, :]).reshape(batch, -1)
        theta = np.repeat(self.theta.reshape(1, -1), batch, axis=0)
        bindings = {tuple(self._input_params): x, tuple(self._weight_params): theta}
        # Observables shaped (A, 1) broadcast against (B,) bindings -> evs (A, B).
        pub = (self._isa_circuit, [[o] for o in self._isa_observables], bindings)
        result = self._estimator.run([pub]).result()[0]
        evs = np.asarray(result.data.evs, dtype=np.float64).reshape(self.n_actions, batch)
        return evs.T

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """Q-values for a batch of observations: (B, F) -> (B, A)."""
        return self.expectations(obs) * self.w + self.b

    def grad_selected(
        self, obs: np.ndarray, action_idx: np.ndarray, upstream: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError(
            "HardwareQFunction is inference-only: a parameter-shift gradient needs "
            f"2 evaluations per circuit parameter = {2 * self.theta.size} extra Estimator "
            "jobs per batch on real hardware (minutes of QPU time per DQN update). "
            "Fine-tune on hardware with traqmania.hardware.spsa_sprint instead, which "
            "needs exactly 2 jobs per iteration regardless of parameter count."
        )

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


# -------------------------------------------------------------------- rollout


def _build_env(track_name: str, config: dict, n_envs: int, seed: int):
    from traqmania.env.racing_env import RacingEnv
    from traqmania.env.track import Track

    track = Track.load(track_name, config["track"]["resample_spacing"])
    return RacingEnv(track, config, n_envs=n_envs, seed=seed)


def run_hardware_lap(
    track_name: str,
    weights_path: str | Path,
    backend,
    shots: int = 1024,
    max_decisions: int | None = None,
    on_decision: Callable[[int, dict], None] | None = None,
    stop_event: Any = None,
) -> dict:
    """Drive ONE car greedily with every decision evaluated on ``backend``.

    Opens a runtime ``Session`` (local testing mode on fake backends; falls
    back to ``mode=backend`` if the backend rejects Sessions) and rolls out
    until the first completed lap, the episode ending, or ``max_decisions``.
    ``on_decision(i, info)`` fires after each decision with the action taken,
    the Q-values, the car state and per-decision latency. ``stop_event``
    (optional): ``threading.Event``-like; when set, the rollout stops between
    decisions (cooperative cancellation) and ``aborted`` is True. Returns
    ``{lapped, best_lap_s, decisions, seconds_per_decision, trajectory,
    aborted, note}``.
    """
    from traqmania.config import load_config

    config = load_config()
    seed = int(config["training"]["seed"])
    env = _build_env(track_name, config, n_envs=1, seed=seed)
    if max_decisions is None:
        max_decisions = env.max_decisions
    params = np.load(weights_path)["params"]

    session, note = _make_session(backend)
    try:
        qfunc = HardwareQFunction(config["circuit"], backend, shots=shots, session=session)
        qfunc.set_params(params)

        obs = env.reset()
        trajectory: list[list[float]] = [env.state[0].tolist()]
        lapped = False
        best_lap_s: float | None = None
        decisions = 0
        aborted = False
        t0 = time.perf_counter()

        for i in range(int(max_decisions)):
            if stop_event is not None and stop_event.is_set():
                aborted = True
                break
            t_dec = time.perf_counter()
            q = qfunc.q_values(obs)
            action = int(np.argmax(q[0]))
            obs, _reward, done, info = env.step(np.array([action]))
            decisions += 1

            if info["lap"][0] >= 1:
                lapped = True
                best_lap_s = float(info["last_lap_time"][0])
            if not done[0]:  # after done the env auto-respawns; don't record that pose
                trajectory.append(env.state[0].tolist())
            if on_decision is not None:
                on_decision(
                    i,
                    {
                        "action": action,
                        "q_values": q[0].tolist(),
                        "state": trajectory[-1],
                        "off_track": bool(info["off_track"][0]),
                        "lap": int(info["lap"][0]),
                        "seconds": time.perf_counter() - t_dec,
                    },
                )
            if lapped or done[0]:
                break

        elapsed = time.perf_counter() - t0
    finally:
        if session is not None:
            session.close()

    return {
        "lapped": lapped,
        "best_lap_s": best_lap_s,
        "decisions": decisions,
        "seconds_per_decision": elapsed / max(1, decisions),
        "trajectory": trajectory,
        "aborted": aborted,
        "note": note,
    }


# ---------------------------------------------------------------- SPSA sprint


def _greedy_return(qfunc, track_name: str, config: dict, seed: int, max_steps: int = 600) -> float:
    """Total reward of one greedy episode (n_envs=1, fixed seed) under ``qfunc``."""
    env = _build_env(track_name, config, n_envs=1, seed=seed)
    obs = env.reset()
    total = 0.0
    for _ in range(max_steps):
        action = np.argmax(qfunc.q_values(obs), axis=1)
        obs, reward, done, _info = env.step(action)
        total += float(reward[0])
        if done[0]:
            break
    return total


def _collect_batch(
    qfunc, track_name: str, config: dict, batch: int, seed: int, epsilon: float = 0.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay batch from a fastsim rollout: (obs, action, double-DQN TD target).

    States come from a greedy rollout with epsilon-greedy noise mixed in (so the
    batch covers more than the on-policy tube); targets use the same parameter
    snapshot as both online and target network — fine for a short sprint.
    """
    n_envs = 4
    env = _build_env(track_name, config, n_envs=n_envs, seed=seed)
    rng = np.random.default_rng(seed)
    gamma = float(config["training"]["gamma"])

    obs_list, act_list, rew_list, next_list, done_list = [], [], [], [], []
    obs = env.reset()
    while len(obs_list) * n_envs < 4 * batch:
        greedy = np.argmax(qfunc.q_values(obs), axis=1)
        random_a = rng.integers(qfunc.n_actions, size=n_envs)
        explore = rng.random(n_envs) < epsilon
        actions = np.where(explore, random_a, greedy)
        next_obs, reward, done, _info = env.step(actions)
        obs_list.append(obs)
        act_list.append(actions)
        rew_list.append(reward)
        next_list.append(next_obs)
        done_list.append(done)
        obs = next_obs

    all_obs = np.concatenate(obs_list)
    all_act = np.concatenate(act_list)
    all_rew = np.concatenate(rew_list)
    all_next = np.concatenate(next_list)
    all_done = np.concatenate(done_list).astype(np.float64)

    idx = rng.choice(all_obs.shape[0], size=batch, replace=False)
    obs_b, act_b = all_obs[idx], all_act[idx]

    # Double-DQN targets, computed ONCE with the exact simulator.
    rows = np.arange(batch)
    a_star = np.argmax(qfunc.q_values(all_next[idx]), axis=1)
    q_next = qfunc.q_values(all_next[idx])[rows, a_star]
    target_b = all_rew[idx] + gamma * (1.0 - all_done[idx]) * q_next
    return obs_b, act_b, target_b


def spsa_sprint(
    track_name: str,
    init_weights_path: str | Path,
    backend,
    iterations: int = 30,
    shots: int = 1024,
    batch: int = 16,
    on_iter: Callable[[int, dict], None] | None = None,
    step_target: float = 0.01,
    stop_event: Any = None,
) -> dict:
    """Short TD-loss SPSA fine-tune of trained weights ON the backend.

    ``stop_event`` (optional): ``threading.Event``-like; when set, the SPSA
    loop stops between iterations (cooperative cancellation) and the result
    reflects the iterations completed so far.

    The expensive-but-exact parts run ONCE in fastsim (replay batch collection
    and double-DQN targets); the hardware only evaluates the MSE TD loss at the
    two SPSA probe points per iteration — 2 Estimator jobs each of ``batch``
    parameter bindings x 4 observables (plus TWO up-front calibration jobs).
    The SPSA gain ``a`` is CALIBRATED from one probe pair (Spall's rule):
    trained output heads have |w| in the hundreds, so the raw TD-loss gradient
    magnitude varies over orders of magnitude between weight files — the
    calibration picks ``a`` such that the first iteration moves each parameter
    by about ``step_target`` (0.01 rad by default), whatever that scale is.
    Greedy-eval returns (fastsim) before and after quantify what the sprint
    did to the policy. Note the deliberate asymmetry: the loss is evaluated on
    the NOISY backend while the returns use the exact simulator, so a sprint
    that compensates hardware noise (loss goes down) can trade away fastsim
    return — the parameters have specialized to the device. Returns
    ``{loss_history, return_before, return_after, params, iterations,
    seconds, note}``.
    """
    from traqmania.config import load_config

    config = load_config()
    seed = int(config["training"]["seed"])
    params0 = np.load(init_weights_path)["params"]

    from traqmania.agents.quantum.qdqn import QuantumQFunction

    fast = QuantumQFunction(config["circuit"], seed=seed)
    fast.set_params(params0)

    obs_b, act_b, target_b = _collect_batch(fast, track_name, config, batch, seed)
    return_before = _greedy_return(fast, track_name, config, seed)

    rows = np.arange(batch)
    session, note = _make_session(backend)
    t0 = time.perf_counter()
    try:
        hw = HardwareQFunction(config["circuit"], backend, shots=shots, session=session)

        def loss(theta: np.ndarray) -> float:
            hw.set_params(theta)
            q_sel = hw.q_values(obs_b)[rows, act_b]
            return float(np.mean((q_sel - target_b) ** 2))

        # Gain calibration (Spall): one probe pair estimates the per-coordinate
        # gradient magnitude |g0|; choose `a` so the FIRST step moves each
        # parameter by ~step_target regardless of the loss scale (|w| in the
        # hundreds makes the raw TD-loss gradient enormous for trained heads).
        c = 0.1
        stability = iterations / 10.0
        rng = np.random.default_rng(seed)
        delta0 = rng.choice(np.array([-1.0, 1.0]), size=params0.shape)
        g0 = abs(loss(params0 + c * delta0) - loss(params0 - c * delta0)) / (2.0 * c)
        a = step_target * (stability + 1.0) ** 0.602 / max(g0, 1e-12)

        result = spsa.minimize(
            loss,
            params0,
            iterations=iterations,
            a=a,
            c=c,
            A=stability,
            seed=seed,
            callback=on_iter,
            stop_event=stop_event,
        )
    finally:
        if session is not None:
            session.close()
    seconds = time.perf_counter() - t0

    fast.set_params(result["x"])
    return_after = _greedy_return(fast, track_name, config, seed)

    return {
        "loss_history": result["loss_history"],
        "params": result["x"],
        "return_before": return_before,
        "return_after": return_after,
        "iterations": int(iterations),
        "seconds": seconds,
        "note": note,
    }


# ------------------------------------------------------------------------ CLI


def _default_weights(track: str) -> Path:
    return WEIGHTS_DIR / f"quantum_{track}.npz"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m traqmania.hardware",
        description="Run traQmania's quantum policy on IBM hardware (or a local fake twin).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("lap", "greedy rollout of one car with decisions made on the backend"),
        ("sprint", "SPSA TD-loss fine-tune on the backend"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--track", default="oval", help="track name (oval | chicane | gp)")
        p.add_argument("--fake", action="store_true", help="use a local fake backend")
        p.add_argument("--fake-name", default="fake_manila", help="fake backend to prefer")
        p.add_argument("--backend", default=None, help="real backend name (default: least busy)")
        p.add_argument("--weights", default=None, help="weights .npz (default: bundled)")
        p.add_argument("--shots", type=int, default=1024)
        if name == "lap":
            p.add_argument("--max-decisions", type=int, default=None,
                           help="stop the rollout after N decisions")
        else:
            p.add_argument("--iterations", type=int, default=30, help="SPSA iterations")
            p.add_argument("--batch", type=int, default=16, help="replay batch size")
    args = parser.parse_args(argv)

    weights = Path(args.weights) if args.weights else _default_weights(args.track)
    backend = get_backend(args.backend, use_fake=args.fake, fake_name=args.fake_name)
    print(f"backend: {getattr(backend, 'name', backend)}"
          f"{' (fake, local simulation)' if args.fake else ''}")
    print(f"weights: {weights}")

    if args.command == "lap":
        def on_decision(i: int, info: dict) -> None:
            q = " ".join(f"{v:+.2f}" for v in info["q_values"])
            print(f"decision {i + 1:>3}  action={info['action']}  Q=[{q}]  "
                  f"lap={info['lap']}  {info['seconds']:.2f}s")

        result = run_hardware_lap(args.track, weights, backend, shots=args.shots,
                                  max_decisions=args.max_decisions, on_decision=on_decision)
        if result["note"]:
            print(f"note: {result['note']}")
        lap_txt = f"{result['best_lap_s']:.2f}s" if result["lapped"] else "no (rollout ended)"
        print(f"\nlap completed: {lap_txt}")
        print(f"decisions: {result['decisions']}  "
              f"({result['seconds_per_decision']:.2f}s per decision)")
    else:
        def on_iter(k: int, info: dict) -> None:
            print(f"iter {k + 1:>3}/{args.iterations}  loss={info['loss']:.4f}  "
                  f"(f+={info['f_plus']:.4f} f-={info['f_minus']:.4f})")

        result = spsa_sprint(args.track, weights, backend, iterations=args.iterations,
                             shots=args.shots, batch=args.batch, on_iter=on_iter)
        if result["note"]:
            print(f"note: {result['note']}")
        print(f"\nSPSA sprint done in {result['seconds']:.1f}s "
              f"({args.iterations} iterations, 2 Estimator jobs each)")
        print(f"loss: {result['loss_history'][0]:.4f} -> {result['loss_history'][-1]:.4f}")
        print(f"greedy return (fastsim): {result['return_before']:.1f} -> "
              f"{result['return_after']:.1f}")


if __name__ == "__main__":
    main()
