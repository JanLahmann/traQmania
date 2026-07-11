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

Six teaching notebooks build the whole stack up from scratch — no local install
needed, each badge launches on Binder (via [QuBins](https://qubins.org) `xl`
images with Qiskit preinstalled):

| Notebook | What it covers | Launch |
|---|---|---|
| [01 — The racing environment](notebooks/01_the_racing_env.ipynb) | tracks, car physics (why you must brake for hairpins), lidar, reward, a scripted lap | [![Binder](https://mybinder.org/badge_logo.svg)](https://qubins.org/launch/?image=latest-xl&repo=https://github.com/JanLahmann/traQmania&branch=main&path=notebooks/01_the_racing_env.ipynb) |
| [02 — Q-learning from scratch](notebooks/02_q_learning_from_scratch.ipynb) | MDPs, double DQN in pure numpy, a 76-parameter MLP learns to lap in seconds | [![Binder](https://mybinder.org/badge_logo.svg)](https://qubins.org/launch/?image=latest-xl&repo=https://github.com/JanLahmann/traQmania&branch=main&path=notebooks/02_q_learning_from_scratch.ipynb) |
| [03 — Quantum circuits as Q-functions](notebooks/03_quantum_circuits_as_q_functions.ipynb) | the data re-uploading VQC, expressivity, fastsim ≡ `EstimatorQNN`, dead parameters, adjoint vs param-shift | [![Binder](https://mybinder.org/badge_logo.svg)](https://qubins.org/launch/?image=latest-xl&repo=https://github.com/JanLahmann/traQmania&branch=main&path=notebooks/03_quantum_circuits_as_q_functions.ipynb) |
| [04 — Training the quantum driver](notebooks/04_training_the_quantum_driver.ipynb) | live quantum DQN training, quantum-vs-classical curves over seeds, lap-time table, honest takeaways | [![Binder](https://mybinder.org/badge_logo.svg)](https://qubins.org/launch/?image=latest-xl&repo=https://github.com/JanLahmann/traQmania&branch=main&path=notebooks/04_training_the_quantum_driver.ipynb) |
| [05 — Real quantum hardware](notebooks/05_real_quantum_hardware.ipynb) | shots, device noise, SPSA hardware sprints, and laps on IBM Quantum devices | [![Binder](https://mybinder.org/badge_logo.svg)](https://qubins.org/launch/?image=latest-xl&repo=https://github.com/JanLahmann/traQmania&branch=main&path=notebooks/05_real_quantum_hardware.ipynb) |
| [06 — More qubits or better features?](notebooks/06_scaling_and_features.ipynb) | sensor scaling 4→10 qubits vs engineered observations, learning curves over seeds, matched MLP baselines, the gp failure-and-rescue | [![Binder](https://mybinder.org/badge_logo.svg)](https://qubins.org/launch/?image=latest-xl&repo=https://github.com/JanLahmann/traQmania&branch=main&path=notebooks/06_scaling_and_features.ipynb) |

## Measured results (Apple Silicon laptop, seed-robust)

| Track | Quantum first clean lap | Best lap (greedy, verified 6/6) | Classical MLP best lap |
|---|---|---|---|
| oval | ~11 s training (ep ≈ 300) | 14.4 s | 13.9–15.2 s |
| chicane | ~11 s training (ep ≈ 294) | 14.1 s | 14.7–15.8 s |
| gp | ~47 s training (ep ≈ 686) | 20.4 s | 19.8–20.4 s |
| combo | ~61 s training (ep ≈ 1130) | 21.7 s (2/3 seeds lap) | 23.1 s |

Warm-start live demo: from the bundled pre-first-lap checkpoint, the quantum agent
gets its first clean lap in **1.4–2.8 s** of training (oval, 3/3 seeds).

**Scaling qubits vs engineering features** (oval; greedy eval, 6 standing-start
episodes per seed, best seed shown — full seed spreads in
[docs/SCIENCE.md](docs/SCIENCE.md)). Extra qubits widen the observation register:
either more lidar rays ("rays + speed") or hand-engineered features (track
curvature ahead, corner-speed ratio, lateral offset, heading error):

| Qubits (profile) | Params | Best oval lap, rays + speed | Best oval lap, engineered features |
|---|---|---|---|
| 4 (default) | 56 | 14.4 s | — (register full: 3 rays + speed) |
| 6 (`q6`) | 80 | 14.7 s | 13.7 s |
| 8 (`q8`) | 104 | 14.0 s | **13.6 s** |
| 10 (`q10`) | 128 | 13.7 s | 14.2 s |

Sample efficiency stays roughly flat from 4 to 10 qubits (first clean lap between
episode ~190 and ~475 everywhere); what grows is per-decision compute, as the
statevector goes 16 → 1024 amplitudes (fastsim greedy: ≲1 ms per decision at 4–6
qubits, ~1.2–2 ms at 8, ~5–9 ms at 10). More rays alone actually *regressed* the
hard gp track — the 5-ray `q6` profile fails greedy gp eval on all seeds (0–1/6
laps at 2000 episodes), which is why no `quantum_gp_q6.npz` ships — while trading
rays for engineered features rescued it (two features, curvature ahead +
corner-speed ratio: 4–5/6 laps on 2 of 3 seeds) and produced our fastest quantum
laps on the oval. Those same features do **not** help
the matched MLP baselines (92–124 params: 13.5–14.1 s rays-only vs 14.1–14.6 s
with features, ~0.01 ms/decision) — yet the MLP still matches or beats every
quantum lap time, so this stays parity, not advantage.

**One driver, every track**: training a single 4-qubit circuit on all four
tracks round-robin (3000 episodes) yields a bundled **universal** driver that
laps oval (15.1 s), chicane (15.4 s), gp (21.7 s), combo (26.8 s) *and*
10/10 unseen generated tracks at medium difficulty (best 14.4 s) — and this
recipe is seed-robust: all 3 seeds tried produce fully universal drivers.
Specialists stay faster at home; the egocentric lidar-and-speed observation
is what makes the transfer work. Two honest history notes: a v1 trained on
only three tracks was defeated zero-shot by combo's blind inward hairpins
(transfer has limits), and with the hard tracks missing from the mix only 1
of 3 seeds generalized.

Why we train on a simulator and run inference on hardware: one double-DQN update is
**~3.4 ms** with the numpy statevector + adjoint path vs **~20.5 s** with
parameter-shift gradients through `EstimatorQNN` — a ~6,000× gap, before any queue
time. The trained policy still laps under 1024-shot sampling and simulated device
noise (`aer_noisy`).

## Modes

- **Watch** (attract): the trained 4-qubit agent drives; live ⟨Z⟩ gauges, Q-values,
  and the circuit diagram update as it decides. A driver picker swaps in any
  bundled training — watch the gp-trained specialist lap the oval zero-shot, or
  the **universal** driver (trained on all three tracks at once) take on any of
  them.
- **Surprise tracks**: pick 🎲 random in the track menu for a procedurally
  generated track with real hairpins and chicanes — fresh every roll, or type a
  seed to reload a favourite, with short/medium/long size presets. The universal
  driver laps unseen generated tracks zero-shot (10/10 at medium difficulty).
- **Train**: watch quantum and classical agents learn side-by-side (warm-start mode
  reaches a first clean lap in seconds).
- **Race**: arrow keys / WASD or a gamepad (analog steering, trigger
  throttle/brake) — race the quantum agent.
- **Evolution**: training snapshots of the same quantum agent race each other
  (three mid-training checkpoints plus the shipped best driver) — watch the
  policy improve across checkpoints.
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
- [Notebooks](#notebooks) — the six-part build-it-from-scratch course above.
