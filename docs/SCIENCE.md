# The science behind traQmania

What the quantum agent actually is, how it is trained, what runs on real
hardware, and — importantly — what this demo does and does not show. The six
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
a regression test). Extra qubits widen the *feature* register — by default
n − 1 lidar rays evenly spaced over [−60°, +60°] plus speed, one feature per
qubit — while the 4 actions and the Z_0…Z_3 readout stay fixed. Trained
6-qubit weights ship for the oval (`quantum_oval_q6.npz`: seed 42, 600
episodes, best greedy-eval lap 14.7 s over 24 laps) and the chicane
(`quantum_chicane_q6.npz`: 14.8 s, 6/6 greedy laps), plus a pre-first-lap
oval warm-start checkpoint (`quantum_oval_warmstart_q6.npz`); `q8`/`q10` are
config options whose training histories live in `data/histories/` but whose
weights are not bundled. Measured on the oval (greedy eval, 6 standing-start
episodes per seed; campaign of 3–5 seeds per variant):

| | 4 qubits | 6 qubits | 8 qubits | 10 qubits |
|---|---|---|---|---|
| Lidar rays (plain profile) | 3 | 5 | 7 | 9 |
| Trainable parameters | 56 | 80 | 104 | 128 |
| First clean lap (episode, seed range) | ≈ 300 | 286–352 | 226–465 | 190–259 |
| Best greedy lap | 14.4 s | 14.7 s | 14.0 s | 13.7 s |
| Greedy-eval robustness | bundled, robust | 3/3 seeds lap (14.7/15.4/16.0 s) | 4/5 lap 6/6 (14.0/14.5/16.2/16.3 s); one seed 0/6 | 3/4 lap 6/6 (13.7/14.8/16.2 s); one seed 3/6 |
| ms per greedy decision (fastsim) | < 1 | 0.6–1 | 1.2–2 | 4.6–8.7 |

The honest reading: sample efficiency per *episode* is roughly flat from 4
to 10 qubits — a real scaling data point, not a success story. The best q8
and q10 laps edge ahead of the 4-qubit 14.4 s, but with a wide seed spread
(one q8 seed never laps the greedy eval at all); what grows reliably is
per-decision compute, since the statevector goes 16 → 1024 amplitudes.
Under 1024-shot Aer sampling the evaluated 8-qubit policy still laps 6/6
(14.5 s, ~18 ms/decision); the evaluated 10-qubit seed — already the least
robust in fastsim at 3/6 — stays at 3/6 (16.1 s, ~24 ms/decision). The
fastsim/`EstimatorQNN` parity and gradient checks run at 6 qubits too
(forward agreement ≤ 1e-9; all 80 gradients verified against finite
differences), and the dead-parameter teaching point above holds at any n.

Scaling also produced an honest failure: on the hard gp track, the plain
5-ray `q6` profile **fails greedy eval on every seed** (0/6, 0/6, 1/6 laps
after 2000 episodes) even though the 4-qubit default laps gp 6/6 at 20.4 s —
more rays alone *regressed* the hard track. `quantum_gp_q6.npz` is therefore
deliberately not bundled. Engineered observations restored it — next
subsection.

### Observation engineering

The observation registry (`[observation] features` in the config;
`traqmania/env/racing_env.py`) lets any qubit count trade lidar rays for
engineered scalars, one feature per qubit, all normalized to [0, 1]:

- `rays` — lidar distances (the default), `speed` — car speed;
- `curvature_ahead` — max centerline |κ| over a lookahead window;
- `lateral_offset` — signed centerline offset / half-width;
- `heading_error` — wrapped angle to the track tangent / π;
- `corner_speed_ratio` — v / v_safe(R) with R = 1/max(|κ_ahead|, 1e-6) and
  v_safe = √(max(0, 2·k_steer·v_turn·R − v_turn²)), derived from the car
  model's steering kinematics (the same “why you must brake for hairpins”
  analysis as notebook 01; derivation in the `racing_env.py` docstring) —
  “am I too fast for the corner coming up?”

Measured variants (same DQN hyperparameters as the plain runs, 3 seeds,
greedy eval with 6 standing-start episodes):

- **`q6feat`** (80 params: 3 rays + speed + curvature_ahead +
  corner_speed_ratio), oval: best laps **13.7/13.8/13.8 s** across seeds
  (6/6, 4/6, 6/6 episodes lapped) — faster than plain q6 (14.7 s) *and* the
  4-qubit default (14.4 s). On the chicane: 13.9–15.8 s across seeds. On gp
  (`featgp`): laps 5/6 at 23.3 s and 4/6 at 23.1 s on two seeds, 0/6 on the
  third — restoring the track that plain 5-ray q6 lost entirely, though
  slower than the 4-qubit default (20.4 s) or the MLP (19.2 s).
- **`q8feat`** (104: adds lateral_offset + heading_error), oval:
  **13.6/13.8/13.9 s, all seeds 6/6** — 13.6 s is the best quantum lap in
  this project.
- **`q10feat`** (128: 5 rays + speed + all four features), oval: 14.2 s
  (one seed 6/6, one 1/6; only 2 seeds run) — the trend does not continue.

The curious part is an **asymmetry**: identical features do *not* help the
matched MLP baselines. On the oval the rays-only MLPs run 13.5–14.1 s
(92/108/124 params at the q6/q8/q10 observation widths, all seeds 6/6,
~0.01 ms/decision) while the feature-observation MLP (`mlp_featoval_q6`)
runs 14.1–14.6 s — the features made the classical baseline *slower* and
the quantum circuit *faster*. We flag this explicitly as an open
observation, not a claim: it rests on one environment, 3 seeds per variant,
and no per-variant hyperparameter search; a plausible mundane explanation
(smooth low-dimensional scalars may simply suit the RY-angle encoding
better than ray geometry) is untested. Notebook 06 has the learning curves
and the full comparison.

### One driver, every track (cross-track generalization)

Because the observation is egocentric (lidar rays + speed, no absolute
position), a trained policy is not tied to its training track. Measured
zero-shot (greedy, no fine-tuning): the gp-trained 4-qubit specialist laps
oval (15.8 s), chicane (16.1 s) and 10/10 unseen procedurally generated
tracks at difficulty 0.5 (best 11.3 s, median 25.9 s); oval/chicane
specialists lap each other's tracks but fail gp (0/6). Training one circuit
on all three tracks round-robin (`--track multi`, 2500 episodes) produced
the bundled **`quantum_universal.npz`** (seed 42): oval 14.3 s, chicane
14.8 s, gp 25.7 s (vs the specialist's 20.4 s), 10/10 generated tracks at
difficulty 0.5, 7/10 at 0.8. Seed-honesty: seeds 0 and 1 of the same recipe
failed to generalize to gp (0/6); training purely on pools of generated
tracks (`--track random`) was strictly worse on the bundled tracks. All
learning curves are in `data/histories/` (`multi_*`, `random_*`).

### The ceiling: a model-based reference driver

To know how good the learned drivers actually are, the expert demo includes a
**hero** driver that is not learned at all: it builds a family of candidate
racing lines from the track geometry (curve-shortening flow, blended wide
through slow corners), derives brake/accelerate-feasible speed profiles from
the `[physics]` constants, *simulates itself* on each candidate with the real
car physics, and drives the fastest provably crash-free combination with
continuous steering (pure pursuit). Measured: oval 13.8 s, chicane 13.9 s,
gp 18.2 s, combo 20.3 s — ahead of every learned agent wherever braking and
line choice matter (on the flat-out oval the big trained MLP ties it to
within 0.03 s: that track is pure path geometry). Two useful facts follow. First, the RL agents' gap to this ceiling is
mostly their action set: they steer with 4 bang-bang actions at 10 Hz, the
hero steers continuously — on gp that is worth 2 s per lap (18.2 vs 20.4).
Second, capacity is not the constraint — an MLP with 8× the parameters
(hidden width 64, 580 params) trained on the same recipe laps the oval in
14.2 s, no better than the 76-parameter one. The expert menu's **pro**
driver takes that question to its limit: the same double-DQN recipe as every
agent in the demo, with a wide MLP (hidden 128, 2,436 params) and the rich
14-feature observation (9 rays + speed + 4 track scalars), trained on all
four tracks at once (`mlp_pro.npz`, seed 42 of 3 — the other seeds crash
gp). Measured: 13.8 / 13.9 / 18.9 / 21.2 s and 10/10 generated tracks — the
strongest learned driver we have, and still a second-plus behind the hero
wherever control quality matters. The remaining gap is dominated by the
4-action interface, not model size. Hero and pro
laps are excluded from ghost records — records stay with the standard demo
agents (and humans).

## Honest claims

Being honest matters more than being exciting:

- **Parity at tiny scale, not speedup.** A 56-parameter VQC and a
  76-parameter MLP learn the same task with broadly similar sample
  efficiency and comparable lap times (quantum edges ahead on oval/chicane,
  the MLP on gp — a few percent, within seed noise). That a VQC *can* do this
  is the point, echoing Chen et al. (2020) and Skolik et al. (2022). The
  scaled variants tell the same story: 4 → 10 qubits keeps sample efficiency
  roughly flat, and even the best engineered-feature quantum lap (13.6 s) is
  matched by a rays-only MLP (13.5–13.7 s) at ~1/100th the decision cost.
- **No quantum advantage is claimed — or possible here.** Four qubits — or
  ten — are trivially simulable; training literally ran on a classical
  simulation of the circuit. An advantage claim would need problem classes
  where a quantum model provably captures structure a comparably-sized
  classical model cannot, not a racing toy.
- **We do not claim engineered features prove quantum advantage.** On the
  oval, hand-engineered observations made the VQC faster and the matched MLP
  slower — an intriguing asymmetry, but observed in one environment, with 3
  seeds per variant and no per-variant hyperparameter search. It is a thing
  to poke at (notebook 06), not evidence of anything quantum.
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
