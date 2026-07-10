# traQmania 🏎️⚛️

**A quantum reinforcement learning demo, Trackmania-style.** Watch a variational
quantum circuit learn to race — then grab the keyboard and try to beat it.

![traQmania demo: attract mode, live quantum training, and race mode](docs/traqmania-hero.gif)

- Quantum Deep Q-Learning (4 qubits / 56 trainable parameters by default; a
  trained 6-qubit / 80-parameter variant ships behind `--profile q6`) built on
  [Qiskit](https://www.ibm.com/quantum/qiskit) and
  [qiskit-machine-learning](https://github.com/qiskit-community/qiskit-machine-learning),
  anchored in [Chen et al., *Variational Quantum Circuits for Deep Reinforcement
  Learning* (IEEE Access 2020)](https://research.ibm.com/publications/variational-quantum-circuits-for-deep-reinforcement-learning).
- Classical numpy baseline with comparable parameter count, trained side-by-side.
- Live training you can watch in minutes, bundled pre-trained weights,
  human-vs-quantum race mode, optional laps on real IBM Quantum hardware.
- Runs on a laptop or a Raspberry Pi; environments based on
  [QuBins](https://qubins.org) images.

## Quick start

```sh
./run.sh                    # venv + install + launch, opens http://127.0.0.1:8000
./run.sh --profile pi5      # Raspberry Pi 5 profile
./run.sh --profile q6       # 6-qubit circuit: 5 lidar rays, 80 parameters
```

Or with Docker (multi-arch, works on a Pi):

```sh
docker run --rm -p 8000:8000 ghcr.io/janlahmann/traqmania
```

## Notebooks

Five teaching notebooks build the whole stack up from scratch — no local install
needed, each badge launches on Binder (via [QuBins](https://qubins.org) `xl`
images with Qiskit preinstalled):

| Notebook | What it covers | Launch |
|---|---|---|
| [01 — The racing environment](notebooks/01_the_racing_env.ipynb) | tracks, car physics (why you must brake for hairpins), lidar, reward, a scripted lap | [![Binder](https://mybinder.org/badge_logo.svg)](https://qubins.org/launch/?image=latest-xl&repo=https://github.com/JanLahmann/traQmania&branch=main&path=notebooks/01_the_racing_env.ipynb) |
| [02 — Q-learning from scratch](notebooks/02_q_learning_from_scratch.ipynb) | MDPs, double DQN in pure numpy, a 76-parameter MLP learns to lap in seconds | [![Binder](https://mybinder.org/badge_logo.svg)](https://qubins.org/launch/?image=latest-xl&repo=https://github.com/JanLahmann/traQmania&branch=main&path=notebooks/02_q_learning_from_scratch.ipynb) |
| [03 — Quantum circuits as Q-functions](notebooks/03_quantum_circuits_as_q_functions.ipynb) | the data re-uploading VQC, expressivity, fastsim ≡ `EstimatorQNN`, dead parameters, adjoint vs param-shift | [![Binder](https://mybinder.org/badge_logo.svg)](https://qubins.org/launch/?image=latest-xl&repo=https://github.com/JanLahmann/traQmania&branch=main&path=notebooks/03_quantum_circuits_as_q_functions.ipynb) |
| [04 — Training the quantum driver](notebooks/04_training_the_quantum_driver.ipynb) | live quantum DQN training, quantum-vs-classical curves over seeds, lap-time table, honest takeaways | [![Binder](https://mybinder.org/badge_logo.svg)](https://qubins.org/launch/?image=latest-xl&repo=https://github.com/JanLahmann/traQmania&branch=main&path=notebooks/04_training_the_quantum_driver.ipynb) |
| [05 — Real quantum hardware](notebooks/05_real_quantum_hardware.ipynb) | shots, device noise, SPSA hardware sprints, and laps on IBM Quantum devices | [![Binder](https://mybinder.org/badge_logo.svg)](https://qubins.org/launch/?image=latest-xl&repo=https://github.com/JanLahmann/traQmania&branch=main&path=notebooks/05_real_quantum_hardware.ipynb) |

## Measured results (Apple Silicon laptop, seed-robust)

| Track | Quantum first clean lap | Best lap (greedy, verified 6/6) | Classical MLP best lap |
|---|---|---|---|
| oval | ~11 s training (ep ≈ 300) | 14.4 s | 13.9–15.2 s |
| chicane | ~11 s training (ep ≈ 294) | 14.1 s | 14.7–15.8 s |
| gp | ~47 s training (ep ≈ 686) | 20.4 s | 19.8–20.4 s |

Warm-start live demo: from the bundled pre-first-lap checkpoint, the quantum agent
gets its first clean lap in **1.4–2.8 s** of training (oval, 3/3 seeds).

**4 vs 6 qubits** (oval; `--profile q6` senses with 5 lidar rays instead of 3):
80 vs 56 parameters (+43 %), first clean lap at episode 286/311/352 across seeds
vs ≈ 300 — the same sample efficiency — and best lap 14.7 s vs 14.4 s. Honest
takeaway: the extra qubits neither help nor hurt learning on this track; treat it
as a real scaling data point, not a success story. (Wall-clock to a first lap is
~28 s vs ~11 s, but that is per-episode compute, not learning speed: the 6-qubit
statevector is 4× larger — 64 vs 16 amplitudes — and the 6-qubit runs also shared
the machine three-at-once.)

Why we train on a simulator and run inference on hardware: one double-DQN update is
**~3.4 ms** with the numpy statevector + adjoint path vs **~20.5 s** with
parameter-shift gradients through `EstimatorQNN` — a ~6,000× gap, before any queue
time. The trained policy still laps under 1024-shot sampling and simulated device
noise (`aer_noisy`).

## Modes

- **Watch** (attract): the trained 4-qubit agent drives; live ⟨Z⟩ gauges, Q-values,
  and the circuit diagram update as it decides.
- **Train**: watch quantum and classical agents learn side-by-side (warm-start mode
  reaches a first clean lap in seconds).
- **Race**: arrow keys / WASD or a gamepad (analog steering, trigger
  throttle/brake) — race the quantum agent.
- **Evolution**: four training snapshots of the same quantum agent (episodes
  350/450/550/700, each verified better than the last) race each other — watch
  the policy improve across checkpoints.
- **Hardware**: run a lap or a bounded SPSA fine-tune sprint on an IBM Quantum
  backend (or a local fake backend) with live status, then replay the hardware
  lap as a ghost. Needs the `[hardware]` extra and a saved IBM Quantum account —
  see the [exhibition runbook](docs/EXHIBITION.md).

## Documentation

- [Exhibition runbook](docs/EXHIBITION.md) — laptop/Pi/kiosk setups, a scripted
  5-minute demo, per-mode talking points, hardware-mode prerequisites,
  troubleshooting.
- [Architecture](docs/ARCHITECTURE.md) — system overview, module map, the
  complete WebSocket protocol reference, data-flow rates, and how to add a
  track / agent / mode.
- [The science](docs/SCIENCE.md) — the circuit, algorithm lineage, training
  and hardware approach, measured results, and what this demo does *not* claim.
- [Notebooks](#notebooks) — the five-part build-it-from-scratch course above.
