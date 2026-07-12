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
Unless noted, headline numbers are seed 42; the v2 seed spreads (seeds
42/0/1 per variant, quoted inline below) were re-run in the overnight
campaign of 2026-07-11; histories in `data/histories/`.

Apple Silicon laptop, `fastsim`, greedy best-snapshot eval:

| Track | Quantum first clean lap | Quantum best lap (greedy) | Classical MLP best lap |
|---|---|---|---|
| oval | ~18 s wall-clock training (ep ≈ 381) | 14.1 s (spread 13.5–14.1) | 13.2 s |
| chicane | ~25 s (ep ≈ 450) | 12.5 s (spread 12.5–14.8) | 13.5 s |
| gp | ~123 s (ep ≈ 1794) | 23.2 s | 20.3 s |
| combo | ~79 s (ep ≈ 1034, warm-started) | 27.5 s (fresh s0: 30.7 s) | 30.8 s |

The gp story improved after an overnight recipe sweep (8 epsilon-decay ×
gamma combinations plus learning-rate, target-sync, curriculum and
warm-start probes). The winning recipe — epsilon decayed over 2000 of 3000
episodes, gamma 0.99, now the shipped `[training_presets.gp]` — moves the
best snapshot from an early high-epsilon fluke (episode ≈ 250 under the old
recipe) to a genuinely learned late-training policy (episode 1950,
epsilon 0.07): greedy eval 7 laps at 23.2 s, and over a 36-episode
evaluation it laps in a mean of 24.0 s (best 20.4 s) with the same 7/36
no-lap episode rate as the old 27.8 s driver. It is also the first v2 gp
recipe that is seed-robust: all three seeds lap (best-snapshot 23.2 /
31.0 / 24.8 s at seeds 42/0/1) where the old recipe lapped at one seed of
three. That closes most, not all, of the gap to the MLP's 20.3 s (under v1
the two were at parity). One honest
regression stands: **combo** — fresh 4-qubit training never laps under v2;
every lapping combo driver is a warm migration of the v1 weights (seeds
42/0/1/7 reach 27.5 / 30.7 / 37.7 / 33.6 s snapshots — the shipped driver
is the seed-42 run), still ahead of the fresh-trained MLP (30.8 s).

Warm-start live demo: from the bundled pre-first-lap checkpoints the quantum
agent reaches its first clean lap in **~2.2 s** of training on the oval,
**~2.7 s** on the chicane, and **~9 s** on combo (seed-0 verification runs).
gp's warm training improved with the new checkpoint (episode 1100 of the
winning recipe) but stays honestly a coin flip: the shipped 900-episode
schedule laps on 2 of 3 seeds tried (~20–40 s wall); when it misses it
misses outright, and every shorter or hotter schedule probed did worse —
gp's braking discovery remains exploration-luck-dependent even
warm-started, which the config notes honestly. One double-DQN update:
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
tracks. Measured on the oval (greedy best-snapshot eval, physics v2; the
headline row is the bundled driver's seed — 42 everywhere except q10,
whose bundled driver is the seed-7 run):

| | 4 qubits | 6 qubits | 8 qubits | 10 qubits |
|---|---|---|---|---|
| Lidar rays (plain profile) | 3 | 5 | 7 | 9 |
| Trainable parameters | 56 | 80 | 104 | 128 |
| First clean lap (episode) | ≈ 381 | ≈ 392 | ≈ 202 | ≈ 358 |
| Best greedy lap (bundled driver) | 14.1 s | 13.3 s | 13.3 s | **12.0 s** |
| Spread over 3 seeds | 13.5–14.1 s | 12.9–13.3 s | 12.5–13.3 s | 12.0–14.2 s |
| ms per greedy decision (fastsim) | < 1 | 0.6–1 | 1.2–2 | 4.6–8.7 |

The honest reading: sample efficiency per *episode* stays roughly flat from
4 to 10 qubits (first clean lap between episode ~200 and ~515 across all
sizes and both tracks) — a real scaling data point, not a success story.
Under v2 the wider registers still trend faster on the easy tracks, but the
seed spreads overlap heavily: the q10 driver's 12.0 s headline is the best
of three seeds (its spread 12.0–14.2 s is the *widest* of any size), so
"more qubits = faster laps" is a tendency, not a clean ordering. Chicane
tells the same story (q6 12.6–13.4 s, q8 12.2–13.1 s, q10 13.2–13.6 s
across seeds). What grows reliably is per-decision compute, since the
statevector goes 16 → 1024 amplitudes. The fastsim/`EstimatorQNN` parity and gradient checks
run at 6 qubits too (forward agreement ≤ 1e-9; all 80 gradients verified
against finite differences), and the dead-parameter teaching point above
holds at any n. The hard tracks above 4 qubits went through two campaigns
with opposite outcomes, and the difference was the recipe, not the circuit.
The first campaign trained gp at q6 (plain and feature observations), q8
and q10 (features) plus combo at q6/q8/q10 — all with the *old* fast
epsilon decay (1200 of 2000 episodes) — and every run finished with zero
greedy laps (a few lapped during exploration only). A follow-up campaign
re-ran the same configurations with the swept winning recipe (decay over
2000 of 3000 episodes), and gp now laps greedily at **every** qubit count.
36-episode reliability evals (12 episodes × 3 env seeds, same protocol as
the 4-qubit numbers): q6-feat laps in 27/36 episodes but slowly
(mean 41.5 s), q8-feat 18/36 (mean 40.2 s), and **q10-feat 25/36 with
mean 22.3 s / best 20.0 s — pace-competitive with the bundled 4-qubit
driver's 29/36 / mean 23.9 s / best 20.4 s** (one training seed vs the
4-qubit driver's three, so treat it as an existence proof, not a spread).
q6-plain is the cautionary tale: its snapshot posted an 18.1 s lap on the
4-episode training-time eval, but the full check shows 5/36, and seeds 0/1
give 1 and 0 greedy laps — a fluke, reported as such. Combo moved too:
with the slow decay, *fresh* 4-qubit combo training laps for the first
time under v2 (23/36, mean 38.1 s — the warm-migrated bundled driver
stays much faster at 21/36, mean 28.7 s), while combo above 4 qubits
still never laps greedily (fresh slow-decay and warm-from-chicane both
fail at q6/q10; cross-track warm starts have never worked in this
project). Net verdict, strengthened: the braking problem was a
recipe/exploration problem, not a capacity one — with the right decay,
capacity is fine all the way to 10 qubits. The q10 gp winner ships as
`quantum_gp_q10.npz`: weight resolution is observation-aware — each weights
file's `.meta.json` records the `[observation]` it was trained with
(`runtime.weights_observation`), and the server overlays it on the profile
when that driver is active, so a feature-observation driver coexists with
the plain-rays oval/chicane weights at the same qubit count. The q6/q8 gp
candidates stay unbundled (they lap, but ~17 s off the pace), as does
every combo candidate above 4 qubits.

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
  corner_speed_ratio): 13.2 s at seed 42; spread 13.0–13.6 s vs plain q6's
  12.9–13.3 s — indistinguishable.
- **`q8feat`** (104: adds lateral_offset + heading_error): **12.5 s** at
  seed 42, which looked like the clearest feature win at v2 — but the seed
  spread (12.5–13.8 s) overlaps plain q8's (12.5–13.3 s), so it does not
  survive as a claim.
- **`q10feat`** (128: 5 rays + speed + all four features): 13.5 s at seed
  42, spread 12.5–13.5 s vs plain q10's 12.0–14.2 s — overlapping again;
  neither the v1 "features stop helping at ten qubits" reading nor its
  opposite survives the seeds.

The **asymmetry** we observed under v1 is now, with three seeds per variant
under v2, best described as *gone at this sample size*: engineered features
neither reliably help the VQC (spreads overlap at every width) nor reliably
hurt the matched MLP baselines (92/108/124 params at the q6/q8/q10
observation widths, rays-only greedy laps 12.1–12.5 / 12.2–12.3 /
11.9–12.1 s over seeds, ~0.01 ms/decision; the q6-width feature MLP runs
12.9–13.0 s — slightly slower than its rays-only twin, the one remnant of
the v1 finding). The v1 asymmetry rested on single seeds; the v2 spreads
absorb it. What does survive is the parity headline: at the q10 observation
the MLP's 11.9 s vs quantum's 12.0 s is the closest the two stacks have
ever been. Notebook 06 has the learning curves and the
full comparison (v1-physics campaign; re-execution under v2 refreshes the
plotted histories).

### One driver, every track (cross-track generalization)

Because the observation is egocentric (lidar rays + speed, no absolute
position), a trained policy is not tied to its training track. Measured
zero-shot under v2 (greedy, no fine-tuning): the gp-trained 4-qubit
specialist laps the oval (13.5 s, 24 laps in 6 episodes) and even combo
(40.6 s) — though under v2 it does not transfer to the chicane (0/6). The bundled
**`quantum_universal.npz`** is now a *warm migration*: the v1 universal
driver fine-tuned on the four-track round-robin under the new physics
(3000 episodes, seed 42). It laps everything — oval 13.0 s, chicane 13.3 s,
gp 30.3 s, combo 52.3 s — and 10/10 unseen generated tracks at difficulty
0.5 (best 23.0 s). Seed-honesty, two layers of it: *fresh* multi-track
training under v2 does not produce a fully universal driver — four runs
tried (seeds 42, 0, 1, 7, one of them 5000 episodes); the best laps
oval/chicane and 9–10/10 generated tracks but fails gp and combo outright.
And the warm migration itself is seed-sensitive: re-running the same v1 →
v2 fine-tune at seeds 0 and 1 *collapses to an oval specialist* (24/24
greedy oval laps at 12.1 s, zero laps on chicane/gp/combo/generated) —
only the seed-42 run kept all-track coverage, and that is the one that
ships. Knowledge transfer across a physics change preserved universality
once, not reproducibly — a nice RL lesson in itself. All learning curves
are in `data/histories/` (`uni2_*`).

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
continuously — on gp that is worth ~7 s per lap for the 4-qubit
circuit (16.4 vs 23.2) and ~1.4 s for the strongest MLP. Second, capacity
alone is not the constraint at baseline scale, but the expert menu's **pro**
driver shows what capacity plus rich sensing buys: the same double-DQN
recipe as every agent in the demo, with a wide MLP (hidden 128, 2,436
params) and the 14-feature observation (9 rays + speed + 4 track scalars),
trained fresh on all four tracks at once for 5000 episodes (`mlp_pro.npz`,
seed 0; the 3000-episode recipe that sufficed under v1 fails gp under v2).
Measured: 11.9 / 12.4 / 17.8 / 20.4 s and 10/10 generated tracks — the
strongest learned driver we have, and unlike the quantum universal driver
it is seed-robust: an independent seed-1 run of the same recipe lands at
11.9 / 12.5 / 17.3 / 20.0 s with 10/10 generated tracks. It even edges the hero on the flat-out
oval by 0.2 s (that track is pure path geometry, and the bang-bang zigzag
traces a fractionally shorter path than smooth line-tracking — documented as
a near-tie, not chased further), while trailing by 1–1.4 s everywhere
control quality matters. The remaining gap is dominated by the 4-action
interface, not model size. Hero and pro laps are excluded from ghost
records — records stay with the standard demo agents (and humans).

### Making qubits matter: scaled actions, longer sensing, a pace objective

Through ten qubits, the scaling story above kept its uncomfortable shape:
more qubits widen the *observation*, yet lap times barely move. Analyzing
the trained drivers against the hero ceiling identified four concrete
bottlenecks — none of them circuit capacity — and each now has a mechanism
in the stack (measured single-seed results from the follow-up campaign are
inline below; the standard eval is 36 greedy episodes):

1. **The action set caps the racing line.** Every driver picked from the
   same 4 bang-bang actions, and crucially could not steer while braking, so
   hairpin entries alternate brake/steer decisions at 10 Hz. The readout now
   scales with the register: `[circuit] n_actions` reads Q_a = ⟨Z_a⟩ off the
   first *k* qubits — 6 actions add trail braking (full steer + brake), 8
   add half-steer (`traqmania/agents/base.py`; prefix-compatible, so the
   default stays bit-identical and every existing weights file keeps its
   meaning). A pure coast action was evaluated and rejected: at drag 0.35
   it is dominated by brake/throttle nearly everywhere. Weights record
   their action count in the `.meta.json` sidecar and the server adopts it
   per driver, exactly like the recorded observation.
   *Measured (campaign E, seed 42):* the mechanism works, the recipe
   doesn't yet. During exploration the 8-action q10 driver drove **18.9 s**
   on gp — the fastest quantum lap ever recorded in this project — but no
   scaled-action lane converged to greedy lapping under the campaign-D
   recipe (q8 6-action 0/36 where its 4-action same-observation control
   lapped 29/36; q10 8-action 0/36, partially rescued to 7/36 @ best
   20.0 s by a pace fine-tune). On the flat-out oval, 6 actions lap
   reliably (36/36) but *slower* (14.8 s vs ~13 s plain q8): extra actions
   only have value where braking matters, and everywhere they enlarge the
   exploration problem. Scaled action sets need their own recipe — the
   promising untried route is warm-starting the widened head from a
   trained 4-action policy (the prefix property makes the padding
   well-defined).
2. **The sensing horizon was shorter than the braking distance.** Slowing
   from v_max 25 to hairpin speed 9 takes ≈(25²−9²)/(2·16) ≈ 17 m, but
   `curvature_ahead` looked only 15 m ahead — optimal braking was literally
   invisible. Curvature kinds now take a per-feature horizon suffix
   (`"curvature_ahead:30"`, `"curvature_ahead:50"`), so a 10-qubit register
   can afford a *braking ladder* of near/mid/far curvature instead of more
   rays.
   *Measured:* honest null-to-negative so far. The ladder observation
   (rays + speed + κ@15/30/50 + corner-speed) is learnable at q8 (29/36,
   but slow at 36.4 s mean) and converged *worse* at q10 (1/36) than
   campaign D's mixed-feature observation (25/36) under the same recipe —
   seeing farther did not make braking easier to learn at this seed.
3. **The reward never asked for speed.** Progress reward pays the same
   total per lap however slow the lap; discounting was the only pace
   pressure. `[reward] time_penalty` (default 0 — bit-identical) charges
   each decision, so a lap's net value rises ~5 per second saved at the
   `[training_pace]` setting. The intended use is two-phase:
   `train_headless --pace --init <lapping snapshot>` runs a low-epsilon
   fine-tune whose objective is lap time, on top of a reliability-trained
   policy.
   *Measured:* **the clear win of the campaign.** A 600-episode pace
   fine-tune of the bundled 10-qubit gp driver moved it from 25/36 @
   22.3 s mean / 20.0 s best to 14/36 @ **20.2 s mean / 17.5 s best** —
   the first greedy quantum gp lap under 18 s, closing a third of the gap
   to the hero's 16.4 s, at a real reliability cost (the bundled driver
   is unchanged; the pace variant lives in the campaign archive). The
   same fine-tune on the 4-qubit gp driver went the other way: **36/36**
   lapped (perfect, up from 30/36) but 28.1 s mean — see (4) for why.
4. **Snapshot selection rewarded lucky laps.** The shipped driver used to
   be the best *4-episode* greedy eval by (laps, best-lap) — the recipe that
   crowned an 18.1 s gp headline which a 36-episode recheck put at 5/36
   lapped episodes. Selection now runs 12 greedy episodes and ranks by
   (lapped episodes, mean lap): reliability first, average pace second, one
   lucky lap never decides.
   *Measured:* selection did what it promises — every campaign-E lane that
   lapped shipped a snapshot whose 36-episode recheck matches its
   12-episode eval (no more mirages). It also exposed a design tension:
   *inside a pace fine-tune*, reliability-first ranking keeps the most
   reliable snapshot even when a slightly less reliable one is seconds
   faster (that is how the 4-qubit pace run "won" 36/36 at 28.1 s). A
   pace-phase selection rule — best mean lap above a reliability floor —
   is the natural follow-up.

The model-based hero (gp 16.4 s) remains the pace target; the bundled
4-qubit gp driver stands at 23.7 s mean under the 36-episode protocol, and
the pace-tuned 10-qubit driver has touched 17.5 s. All campaign-E numbers
are one seed (42) — the same caveat as every headline in this document
until a seed spread says otherwise.

## Honest claims

Being honest matters more than being exciting:

- **Parity at tiny scale, not speedup.** A 56-parameter VQC and a
  76-parameter MLP learn the same task with broadly similar sample
  efficiency and comparable lap times (quantum edges ahead on chicane and
  warm combo, the MLP on oval and gp — mostly within seed noise). That a
  VQC *can* do this is the point, echoing Chen et al. (2020) and Skolik
  et al. (2022). The scaled variants tell the same story: 4 → 10 qubits
  keeps sample efficiency roughly flat, and every quantum lap is matched
  or beaten by a matched-observation MLP (11.9–12.5 s rays-only) at
  ~1/100th the decision cost.
- **No quantum advantage is claimed — or possible here.** Four qubits — or
  ten — are trivially simulable; training literally ran on a classical
  simulation of the circuit. An advantage claim would need problem classes
  where a quantum model provably captures structure a comparably-sized
  classical model cannot, not a racing toy.
- **We do not claim engineered features prove quantum advantage.** Under v1
  physics, hand-engineered observations appeared to make the VQC faster and
  the matched MLP slower — an intriguing asymmetry. The v2 three-seed
  spreads dissolved it (feature and rays-only ranges overlap for the VQC;
  the MLP penalty shrank to ~0.5 s at one width). We report the dissolution
  as prominently as we reported the observation: single-seed asymmetries in
  small RL benchmarks usually are noise (notebook 06 walks through both).
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
