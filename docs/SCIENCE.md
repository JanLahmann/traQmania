# The science behind traQmania

What the quantum agent actually is, how it is trained, what runs on real
hardware, and — importantly — what this demo does and does not show. The five
[notebooks](../notebooks/) build all of this up from scratch with runnable
code; this page is the condensed reference.

## The circuit

A data re-uploading variational quantum circuit (VQC) acting as the Q-function
of a DQN agent. Canonical definition: `traqmania/agents/quantum/circuit.py`
(single source of truth for the numpy fast path, the Qiskit/`EstimatorQNN`
path, and the hardware path).

- **n qubits** (default **4**), **L = 4 re-uploading blocks** on |0…0⟩.
  Block *l*:
  1. encoding: `RY(λ[l,i] · s[i])` on each qubit *i* — the n observation
     features `s` (n − 1 lidar rays evenly spaced over [−60°, +60°] plus
     speed, each normalized to [0, 1]; 3 rays + speed at the default size)
     are re-uploaded in **every** block;
  2. variational: `RY(θ[l,i,0])` then `RZ(θ[l,i,1])` on each qubit;
  3. entanglement: a CZ ring `CZ(0,1) … CZ(n−1,0)`.
- **Readout:** one Pauli-Z expectation per action, `E_a = ⟨Z_a⟩` on the
  first four qubits — the **4 actions stay fixed at any n**; extra qubits
  only widen the feature register — mapped through a trainable classical
  head `Q_a = w[a]·E_a + b[a]`. The output scaling is essential: useful
  Q-values for this task reach the hundreds while ⟨Z⟩ ∈ [−1, 1].
- **Trainable parameters, P = 3·L·n + 8** (**56** at n = 4, **80** at
  n = 6), one flat vector `[λ, θ, w, b]`: input scalings λ (L×n, init π),
  variational angles θ (L×n×2, init U(−0.1, 0.1)), output weights w
  (4, init 1) and biases b (4, init 0).

**The dead-parameter teaching point:** in the final block each RZ is followed
only by diagonal operations (the CZ ring) before the Z measurement. RZ is
itself diagonal, so it commutes through to the observable and only shifts
phases a Z measurement can never see — the gradient of those n final-RZ
parameters (4 at the default size) is *exactly* zero for every input, at any
qubit count; they can never train. Kept
deliberately (the canonical Chen/Skolik-style block structure) as a lesson in
how ansatz structure interacts with the observable. Notebook 03 verifies it
by taking the derivative rather than trusting the commutator argument.

## Algorithm lineage

- **Chen, Yang, Qi, Chen, Ma & Goan, “Variational Quantum Circuits for Deep
  Reinforcement Learning”, IEEE Access 8 (2020)** — the IBM-anchored paper
  this demo follows: a VQC as the Q-function of experience-replay Q-learning.
- **Skolik, Jerbi & Dunjko, “Quantum agents in the Gym”, Quantum 6, 720
  (2022)** — data re-uploading, trainable input/output scaling, and the
  observable-design considerations adopted here.
- **Meyer et al., “A Survey on Quantum Reinforcement Learning”
  (arXiv:2211.03464)** — situates this family of value-based QRL methods in
  the broader landscape.

## Training

**Double DQN in pure numpy** (`agents/training/dqn.py`): experience replay
(ring buffer, 10 000 transitions), epsilon-greedy exploration with linear
decay, Adam, and a periodically-synced target network — which is just a
second flat parameter vector, so the identical loop trains the quantum
circuit and the classical baseline. The environment is vectorized (8 parallel
cars by default); reward is signed centerline progress plus checkpoint/lap
bonuses minus an off-track penalty. The classical baseline is a 76-parameter
MLP (4-8-4, tanh) — chosen for comparable parameter count, not tuned to win.

**Two executions of one circuit, verified equal.** The training path is
`fastsim`, a hand-written numpy statevector simulator with **adjoint**
(backprop-style) gradients: one forward plus one backward sweep for the whole
gradient. The same circuit runs through Qiskit / qiskit-machine-learning's
`EstimatorQNN` on Aer; `tests/test_fastsim_vs_qiskit.py` and
`tests/test_gradients.py` pin forward values and gradients against each other
and against finite differences. Measured cost of one batch-32 double-DQN
update: **~3.4 ms** (fastsim + adjoint) vs **~20.5 s** (`EstimatorQNN` +
parameter-shift, 2 evaluations per parameter) — a ~6 000× gap, which is *the*
practical reason training runs on the simulator.

## Hardware

(`traqmania/hardware.py`, via `qiskit-ibm-runtime`; a `FakeBackendV2` twin —
noise model + coupling map of a real small device, simulated locally — makes
everything below runnable without an account. Backend selection is
qubit-aware, `min_qubits = max(5, n_qubits)`: the default circuit lands on a
5-qubit fake, while `--profile q6` skips those and picks the 7-qubit
`fake_lagos`.)

- **Inference laps** (`run_hardware_lap`): drive one car greedily with every
  10 Hz-equivalent decision evaluated on the backend. `HardwareQFunction`
  implements the same `QFunction` contract and parameter layout as the
  simulator: the circuit is transpiled to ISA form once, then each decision
  is one `EstimatorV2` job (all observation rows × all four Z_a observables
  in a single PUB) inside a runtime `Session`. Gradients are deliberately
  `NotImplementedError` — parameter-shift would cost 2 × 48 circuit
  evaluations per batch at the default 4 qubits (2 × 3·L·n in general).
- **SPSA sprint** (`spsa_sprint`): the only sensible way to *update*
  parameters on hardware today. The replay batch and double-DQN TD targets
  are computed **once** in the exact simulator; the hardware then only
  evaluates the MSE TD loss at the two SPSA probe points — **2 Estimator jobs
  per iteration regardless of parameter count** (plus 2 up-front jobs that
  calibrate the SPSA gain via Spall's rule, needed because trained output
  heads with |w| in the hundreds make the raw loss scale vary wildly between
  weight files). Greedy fastsim returns before/after quantify the effect.
  Deliberate asymmetry: the loss is evaluated on the noisy backend but the
  returns on the exact simulator — a sprint that adapts to device noise can
  trade away simulator return.
- **Why full training stays simulated:** a typical run is ~20 000 gradient
  steps. At param-shift pricing that is ~5 days of pure compute before queue
  time; even SPSA's 2 jobs/iteration makes hardware training a demonstration
  of mechanics, not a competitive training method at this scale.

## Measured results

Apple Silicon laptop, `fastsim`, seed-robust (multiple seeds per cell; see
README, `data/histories/`, and notebook 04 for the underlying runs):

| Track | Quantum first clean lap | Quantum best lap (greedy, verified 6/6) | Classical MLP best lap |
|---|---|---|---|
| oval | ~11 s wall-clock training (ep ≈ 300) | 14.4 s | 13.9–15.2 s |
| chicane | ~11 s (ep ≈ 294) | 14.1 s | 14.7–15.8 s |
| gp | ~47 s (ep ≈ 686) | 20.4 s | 19.8–20.4 s |

Warm-start live demo: from the bundled pre-first-lap checkpoint the quantum
agent reaches its first clean lap in **1.4–2.8 s** of training (oval, 3/3
seeds). One double-DQN update: **~3.4 ms** fastsim/adjoint vs **~20.5 s**
`EstimatorQNN`/param-shift.

### Scaling the qubit count

The circuit generalizes over `n_qubits` (config profiles `q6`, `q8`, `q10`;
the default stays 4 and is bit-identical to the pre-scaling stack, pinned by
a regression test). Extra qubits widen the *feature* register — n − 1 lidar
rays evenly spaced over [−60°, +60°] plus speed, one feature per qubit —
while the 4 actions and the Z_0…Z_3 readout stay fixed. Trained 6-qubit
weights ship for the oval (`quantum_oval_q6.npz`: seed 42, 600 episodes,
best greedy-eval lap 14.7 s over 24 laps); `q8`/`q10` are untrained options.
Measured head-to-head on the oval:

| | 4 qubits (3 rays) | 6 qubits (5 rays) |
|---|---|---|
| Trainable parameters | 56 | 80 (+43 %) |
| Episodes to first clean lap | ≈ 300 | 286 / 311 / 352 (seeds 42/1/0) |
| Wall-clock to first clean lap | ~11 s | ~28 s |
| Best lap (greedy) | 14.4 s | 14.7 s |

The honest reading: on the oval, 5 rays instead of 3 buys **no lap-time
win** — the extra qubits neither help nor hurt learning on this task. This
is a real scaling data point, not a success story. Sample efficiency (per
episode) is the same; the wall-clock gap is per-episode compute cost, not
learning speed — a 6-qubit statevector is 4× larger (64 vs 16 amplitudes),
and the 6-qubit runs additionally shared the machine three-at-once. The
fastsim/`EstimatorQNN` parity and gradient checks run at 6 qubits too
(forward agreement ≤ 1e-9; all 80 gradients verified against finite
differences), and the dead-parameter teaching point above holds at any n.

## Honest claims

Being honest matters more than being exciting:

- **Parity at tiny scale, not speedup.** A 56-parameter VQC and a
  76-parameter MLP learn the same task with broadly similar sample
  efficiency and comparable lap times (quantum edges ahead on oval/chicane,
  the MLP on gp — a few percent, within seed noise). That a VQC *can* do this
  is the point, echoing Chen et al. (2020) and Skolik et al. (2022). The
  6-qubit variant tells the same story: +43 % parameters, same sample
  efficiency, no faster laps.
- **No quantum advantage is claimed — or possible here.** Four qubits — or
  ten — are trivially simulable; training literally ran on a classical
  simulation of the circuit. An advantage claim would need problem classes
  where a quantum model provably captures structure a comparably-sized
  classical model cannot, not a racing toy.
- **Watch the denominators.** “Similar sample efficiency” is per *episode*;
  per *second* the MLP trains far faster (cheaper gradients). Parameter count
  is an imperfect fairness measure — expressivity per parameter differs.
- **Noise robustness is an observation, not a theorem.** Under 1024-shot
  sampling the trained policy laps at reference pace (Q-value gaps at visited
  states exceed the ~0.03·|w| shot noise). Under a real device's noise model
  it is on the ragged edge — sometimes lapping, often crashing, varying
  between runs with the stochastic transpilation layout. Noise robustness of
  RL *policies* (not just expectation values) is a genuinely open question
  this testbed lets you poke at.

## References

1. S. Y.-C. Chen, C.-H. H. Yang, J. Qi, P.-Y. Chen, X. Ma, H.-S. Goan,
   *Variational Quantum Circuits for Deep Reinforcement Learning*, IEEE
   Access 8, 141007–141024 (2020).
   <https://research.ibm.com/publications/variational-quantum-circuits-for-deep-reinforcement-learning>
2. A. Skolik, S. Jerbi, V. Dunjko, *Quantum agents in the Gym: a variational
   quantum algorithm for deep Q-learning*, Quantum 6, 720 (2022).
   <https://quantum-journal.org/papers/q-2022-05-24-720/>
3. N. Meyer, C. Ufrecht, M. Periyasamy, D. D. Scherer, A. Plinge,
   C. Mutschler, *A Survey on Quantum Reinforcement Learning*,
   arXiv:2211.03464. <https://arxiv.org/abs/2211.03464>
4. J. C. Spall, *Multivariate stochastic approximation using a simultaneous
   perturbation gradient approximation*, IEEE Trans. Autom. Control 37(3)
   (1992) — the SPSA method and gain-selection rule used by the hardware
   sprint.
