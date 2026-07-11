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

**Physics v2 (2026-07).** All numbers in this section are measured under the
current physics constants (`accel` 11, `brake` 16, `v_max` 25 — faster
straights, unchanged hairpin discipline). The change was made so speed
visibly *varies*: the model-based hero driver's max/min speed ratio is 1.81
on gp and 1.79 on combo (≈1.5 under v1), while the oval/chicane corners are
gentle enough to stay flat-out at any of these speeds. It was tuned as far
as the 4-qubit circuit could follow — `v_max` 28–30 variants left gp and
combo unlearnable at 4 qubits even when warm-started — and it still has a
real cost, reported below: the hard tracks got harder for the tiny circuit.
Unless noted, v2 numbers are seed 42 with spot checks on 1–2 further seeds
(the v1 campaign's full seed spreads have not been re-run yet); histories in
`data/histories/`.

Apple Silicon laptop, `fastsim`, greedy best-snapshot eval:

| Track | Quantum first clean lap | Quantum best lap (greedy) | Classical MLP best lap |
|---|---|---|---|
| oval | ~18 s wall-clock training (ep ≈ 381) | 14.1 s | 13.2 s |
| chicane | ~25 s (ep ≈ 450) | 12.5 s | 13.5 s |
| gp | ~107 s (ep ≈ 1451) | 27.8 s | 20.3 s |
| combo | ~79 s (ep ≈ 1034, warm-started) | 27.5 s | 30.8 s |

Two honest v2 regressions. **gp:** the greedy policy peaks *early* (the
bundled snapshot is from episode ≈ 250, reached during high-epsilon
exploration) and continued training degrades it rather than improving it —
recipe grids over epsilon decay, gamma, and warm-starting from the v1
weights all failed to beat that peak, so the quantum gp lap is now well
behind the MLP's (27.8 vs 20.3 s; under v1 they were at parity). **combo:**
fresh 4-qubit training never laps at all under v2; the bundled driver is a
warm-started migration of the v1 weights (6/6 greedy at 27.5 s), which
still beats the fresh-trained MLP (30.8 s).

Warm-start live demo: from the bundled pre-first-lap checkpoints the quantum
agent reaches its first clean lap in **~2.2 s** of training on the oval,
**~2.7 s** on the chicane, and **~9 s** on combo (seed-0 verification runs);
gp's warm recipe is currently *not* reliable under v2 (no seed lapped within
500 warm episodes — the fresh run's high-epsilon peak cannot be retraced at
low epsilon), which the config notes honestly. One double-DQN update:
**~3.4 ms** fastsim/adjoint vs **~20.5 s** `EstimatorQNN`/param-shift.

### Scaling the qubit count

The circuit generalizes over `n_qubits` (config profiles `q6`, `q8`, `q10`;
the default stays 4 and is bit-identical to the pre-scaling stack, pinned by
a regression test). Extra qubits widen the *feature* register — by default
n − 1 lidar rays evenly spaced over [−60°, +60°] plus speed, one feature per
qubit — while the 4 actions and the Z_0…Z_3 readout stay fixed. Trained
weights now ship for **oval and chicane at every size** (`quantum_<track>_q6/
q8/q10.npz`), plus a pre-first-lap oval warm-start checkpoint and evolution
stages at q6, so the live Qubits selector works out of the box on those
tracks. Measured on the oval (greedy best-snapshot eval, seed 42, physics
v2):

| | 4 qubits | 6 qubits | 8 qubits | 10 qubits |
|---|---|---|---|---|
| Lidar rays (plain profile) | 3 | 5 | 7 | 9 |
| Trainable parameters | 56 | 80 | 104 | 128 |
| First clean lap (episode) | ≈ 381 | ≈ 392 | ≈ 202 | ≈ 358 |
| Best greedy lap | 14.1 s | 13.3 s | 13.3 s | **12.0 s** |
| ms per greedy decision (fastsim) | < 1 | 0.6–1 | 1.2–2 | 4.6–8.7 |

The honest reading: sample efficiency per *episode* stays roughly flat from
4 to 10 qubits (first clean lap between episode ~200 and ~515 across all
sizes and both tracks) — a real scaling data point, not a success story.
Under v2 the wider registers do convert into faster laps on the easy tracks
(the 9-ray q10 driver's 12.0 s is the fastest quantum lap in the demo), but
what grows reliably is per-decision compute, since the statevector goes
16 → 1024 amplitudes. The fastsim/`EstimatorQNN` parity and gradient checks
run at 6 qubits too (forward agreement ≤ 1e-9; all 80 gradients verified
against finite differences), and the dead-parameter teaching point above
holds at any n. gp and combo remain deliberately untrained above 4 qubits
(the v1 campaign showed more rays alone *regressing* gp; under v2 gp is
fragile even at 4 qubits, so we have not spent the compute).

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

Measured variants on the oval (same DQN hyperparameters as the plain runs;
physics v2, seed 42, greedy best-snapshot eval):

- **`q6feat`** (80 params: 3 rays + speed + curvature_ahead +
  corner_speed_ratio): 13.2 s — a hair faster than plain q6 (13.3 s).
- **`q8feat`** (104: adds lateral_offset + heading_error): **12.5 s** vs
  plain q8's 13.3 s — the clearest feature win at v2.
- **`q10feat`** (128: 5 rays + speed + all four features): 13.5 s vs plain
  q10's 12.0 s — at ten qubits the 9-ray observation wins; the feature
  advantage does not continue up the scale (same qualitative finding as
  under v1 physics).

The **asymmetry** we observed under v1 persists in weakened form: the same
features do not help the matched MLP baselines (92/108/124 params at the
q6/q8/q10 observation widths, rays-only greedy laps 12.5/12.3/11.9 s,
~0.01 ms/decision; the feature-observation MLP at the q6 width runs 13.0 s —
slower than its 12.5 s rays-only twin). We flag this as an open observation,
not a claim: it rests on one environment, few seeds, and no per-variant
hyperparameter search; a plausible mundane explanation (smooth
low-dimensional scalars may simply suit the RY-angle encoding better than
ray geometry) is untested. Note the parity headline hiding in those numbers:
at the q10 observation the MLP's 11.9 s vs quantum's 12.0 s is the closest
the two stacks have ever been. Notebook 06 has the learning curves and the
full comparison (v1-physics campaign; re-execution under v2 refreshes the
plotted histories).

### One driver, every track (cross-track generalization)

Because the observation is egocentric (lidar rays + speed, no absolute
position), a trained policy is not tied to its training track. Measured
zero-shot under v2 (greedy, no fine-tuning): the gp-trained 4-qubit
specialist laps the oval (14.1 s, 23 laps in 6 episodes) — though under v2
it no longer transfers to the chicane (0/6). The bundled
**`quantum_universal.npz`** is now a *warm migration*: the v1 universal
driver fine-tuned on the four-track round-robin under the new physics
(3000 episodes, seed 42). It laps everything — oval 13.0 s, chicane 13.3 s,
gp 30.3 s, combo 52.3 s — and 10/10 unseen generated tracks at difficulty
0.5 (best 23.0 s). Seed-honesty: *fresh* multi-track training under v2 does
not produce a fully universal driver anymore — four runs tried (seeds 42,
0, 1, 7, one of them 5000 episodes); the best of them laps oval/chicane
and 9–10/10 generated tracks but fails gp and combo outright (the same gp
fragility as the specialist story above). Knowledge transfer across a
physics change is what preserved universality, which is a nice RL lesson in
itself. All learning curves are in `data/histories/` (`uni2_*`).

### The ceiling: a model-based reference driver

To know how good the learned drivers actually are, the expert demo includes a
**hero** driver that is not learned at all: it builds a family of candidate
racing lines from the track geometry (curve-shortening flow, blended wide
through slow corners), derives brake/accelerate-feasible speed profiles from
the `[physics]` constants, *simulates itself* on each candidate with the real
car physics, and drives the fastest provably crash-free combination with
continuous steering (pure pursuit). Because everything derives from the
`[physics]` constants, the hero adapted to the v2 physics with no retraining.
Measured: oval 12.1 s, chicane 12.1 s, gp 16.4 s, combo 19.0 s — ahead of
every learned agent wherever braking and line choice matter. Two useful
facts follow. First, the RL agents' gap to this ceiling is mostly their
action set: they steer with 4 bang-bang actions at 10 Hz, the hero steers
continuously — on gp that is now worth over 10 s per lap for the 4-qubit
circuit (16.4 vs 27.8) and ~1.4 s for the strongest MLP. Second, capacity
alone is not the constraint at baseline scale, but the expert menu's **pro**
driver shows what capacity plus rich sensing buys: the same double-DQN
recipe as every agent in the demo, with a wide MLP (hidden 128, 2,436
params) and the 14-feature observation (9 rays + speed + 4 track scalars),
trained fresh on all four tracks at once for 5000 episodes (`mlp_pro.npz`,
seed 0; the 3000-episode recipe that sufficed under v1 fails gp under v2).
Measured: 11.9 / 12.4 / 17.8 / 20.4 s and 10/10 generated tracks — the
strongest learned driver we have. It even edges the hero on the flat-out
oval by 0.2 s (that track is pure path geometry, and the bang-bang zigzag
traces a fractionally shorter path than smooth line-tracking — documented as
a near-tie, not chased further), while trailing by 1–1.4 s everywhere
control quality matters. The remaining gap is dominated by the 4-action
interface, not model size. Hero and pro laps are excluded from ghost
records — records stay with the standard demo agents (and humans).

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
