# Exhibiting traQmania

A practical runbook for running traQmania at a booth, in a classroom, or on a
museum kiosk. For what the science means, see [SCIENCE.md](SCIENCE.md); for
how the system works, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Setups

### Laptop (the simple case)

```sh
./run.sh          # venv + install + launch, opens http://127.0.0.1:8000
```

Requires Python ≥ 3.11. Everything (training included) runs locally on CPU.
Pass server flags straight through, e.g. `./run.sh --port 8010`
(set `TRAQMANIA_PORT=8010` too so the auto-opened browser URL matches).

### Raspberry Pi (QuBins)

The demo runs on a Pi 4 or Pi 5 — e.g. on [QuBins](https://qubins.org) images,
which ship Qiskit preinstalled. Use the matching profile:

```sh
./run.sh --profile pi5     # or pi4
```

The Pi profiles lower the broadcast rate to 15 Hz and telemetry to 5 Hz and
shrink training batches (pi4 also drops default episodes to 150). For live
training on a Pi, always tick **Warm start** — cold training on a Pi 4 is a
coffee break, warm-start is seconds.

Or containerized (the image is multi-arch and built on a QuBins base):

```sh
docker run --rm -p 8000:8000 ghcr.io/janlahmann/traqmania
# podman works identically:
podman run --rm -p 8000:8000 ghcr.io/janlahmann/traqmania
```

To use a profile inside the container, override the command:

```sh
docker run --rm -p 8000:8000 ghcr.io/janlahmann/traqmania \
  python -m traqmania --host 0.0.0.0 --port 8000 --profile pi5
```

### Kiosk / exhibition mode

```sh
./run.sh --profile exhibition
```

The `exhibition` profile (`traqmania/config/exhibition.toml`):

- binds `0.0.0.0` — the UI is reachable from other devices on the LAN
  (visitors' phones can watch);
- `attract_idle_seconds = 20` — the client returns to attract mode after 20 s
  without keyboard/mouse activity, so the exhibit never sits on a stale
  screen (default is 45 s; set `0` to disable);
- `kiosk = true` — larger captions, hidden mouse cursor.

Profiles stack with an extra overlay via `--config <file.toml>`, and any
`./config/<name>.toml` in the working directory shadows the packaged profile
of the same name — so a Pi kiosk is `./run.sh --profile pi5 --config
traqmania/config/exhibition.toml`. Run the browser fullscreen, e.g.
`chromium-browser --kiosk http://localhost:8000`.

### The 6-qubit variant

```sh
./run.sh --profile q6
```

What visibly changes: the car senses with **5 lidar rays** instead of 3, the
circuit diagram shows **6 qubits**, and the parameter count reads **80**
(actions stay 4). Bundled 6-qubit weights exist for the **oval** only, so
attract, race-vs-quantum, and hardware laps work there; **Evolution** and the
**MLP opponent** are 4-qubit-only, and the UI reports a clear error if
selected rather than crashing. Live training works on any track, but there is
no 6-qubit warm-start checkpoint — training is cold (~28 s to a first clean
lap on the oval). Hardware mode automatically picks a big-enough fake backend
(the 7-qubit `fake_lagos` instead of the 5-qubit devices).

Talking point — the honest scaling result: *"Same circuit family at 6 qubits:
43 % more parameters, the same number of episodes to a first lap, and no
faster laps (14.7 s vs 14.4 s best). At this scale extra qubits neither help
nor hurt — a real scaling data point is worth more than a hype slide."*

## The 5-minute demo

A narrative that works cold, in order. Controls for the race segment:
**arrow keys or WASD** (up/W throttle, down/S brake, left/right steer).

1. **Attract — "this driver is a quantum circuit"** (~1 min).
   The screen already shows it: a car lapping, four wobbling gauges, a
   circuit diagram. Say: *"Every tenth of a second, the car's three distance
   sensors and its speed are encoded into rotations on 4 qubits; the four
   ⟨Z⟩ readouts you see on the gauges become the four action values — the
   biggest one steers the car."* Point at a corner: watch the brake action
   win just before the hairpin.

2. **Train — "watch it learn its first lap in seconds"** (~1 min).
   Mode **Train** → agent *Quantum* → tick **Warm start** → *Start
   training*. Eight cars flail, the return curve climbs, and the first clean
   lap lands in a couple of seconds (measured 1.4–2.8 s on oval); the
   best-lap banner fires as laps keep improving. Mention: *"This is real
   double-DQN training against a simulated version of the circuit — the same
   math IBM's 2020 quantum-RL paper used."* (Without warm start a full cold
   run on oval is ~11 s to the first clean lap — still demoable; on a Pi,
   warm only.)

3. **Evolution — "the same circuit at four ages"** (~30 s).
   Mode **Evolution**: four cars drive weights snapshotted at increasing
   points of one training run — the labels show how many episodes each has
   trained ("ep N"). The youngest car wobbles and crashes; the oldest is
   smooth. One picture of what training buys.

4. **Race — "beat the quantum driver"** (~1.5 min).
   Mode **Race**, opponent *Quantum*, hand over the keyboard. Going off
   track freezes the visitor's car for a second, then respawns it; the ghost
   car is the all-time best lap on this machine. Most first-timers lose —
   that lands the point better than any slide.

5. **Hardware — "and now on a (fake) quantum device"** (~1 min).
   Mode **Hardware**, backend *fake*, run a **lap**: every steering decision
   is now an `EstimatorV2` job against a local twin of a real 5-qubit IBM
   device — noise model, coupling map, per-decision latency and all. Watch
   the status pill walk through connecting → transpiling → running, then the
   lap replays next to a simulator car driving the same weights. Say:
   *"Same circuit, same weights — the only change is who executes it. With
   an IBM Quantum token this exact flow runs on a physical device."*

## Per-mode talking points

- **Watch (attract):** 4 qubits, 4 layers, 56 trainable parameters — of which
  4 are provably dead (a fun aside for physicists: the final RZ commutes with
  the Z measurement). Gauges are live ⟨Z_a⟩; bars are Q-values after the
  trained output head. The circuit diagram is the actual gate sequence.
  The **Driver** dropdown swaps which training drives — put the gp-trained
  specialist on the oval to show zero-shot transfer, or pick *universal* (one
  circuit trained on all three tracks at once).
- **Surprise tracks (🎲 random):** every roll is a fresh procedurally
  generated circuit with hairpins and chicanes; the universal weights drive
  it. Type a seed (shown in the track label) to reload a favourite; the size
  dropdown gives short/medium/long layouts. Long tracks are for driving and
  watching, not training.
- **Train:** double DQN, epsilon-greedy, replay buffer — the classical RL
  recipe, with the neural network swapped for a quantum circuit. Choosing
  *Both* races quantum vs MLP learning curves live. Honest line: *"similar
  learning, no speedup — the interesting part is that a 56-parameter quantum
  model does this at all."*
- **Evolution:** all cars run the identical architecture; only the training
  amount differs. Labels show "ep N" for mid-training checkpoints and "best"
  for the shipped driver. Tracks without stage snapshots show just two cars:
  warm-start vs best.
- **Race:** the agent decides 10 times per second and gets exactly the same
  observation a human gets from the screen: three distance rays and speed.
  Clean laps that beat the record become the new ghost.
- **Hardware:** inference on hardware, training on simulator — one gradient
  step is ~3.4 ms simulated vs ~20.5 s with circuit-evaluation gradients
  (before queue time!). The *sprint* action shows the middle path: SPSA
  fine-tuning at 2 quantum jobs per iteration, whatever the parameter count.

## Expert mode: the hero driver

Open the demo with `#expert` in the URL (`http://127.0.0.1:8000/#expert`) and
the Watch-mode Driver dropdown gains **hero — racing line**: a cyan car driven
by a model-based controller, *not* a learned agent. It computes the
curvature-minimizing racing line and a physics-derived braking/speed profile
straight from the track geometry and tracks it with continuous steering — the
"perfect drive" ceiling for this car model. Measured: oval 14.0 s, chicane
14.0 s, gp 19.8 s, combo 22.6 s, and it handles every generated track. Two talking points:
the RL agents are surprisingly close to this ceiling on the simple tracks
(their gap is mostly the 4-action bang-bang control, not intelligence — an
8×-bigger trained MLP gets *no* faster), and the hero's line visibly differs
(apex-cutting, earlier braking). It is near-optimal, not provably optimal: on
combo the trained quantum specialist actually beats it (21.7 vs 22.6 s). Its clean laps become ghosts labelled
"racing line (model-based)"; delete `traqmania/data/ghosts/<track>.json` (or
your `--ghosts-dir` equivalent) to reset a track's record before a show.

## Hardware-mode prerequisites

- Install the hardware extra (not needed for anything else):

  ```sh
  pip install -e ".[hardware]"     # qiskit-ibm-runtime
  ```

- **Fake backend: nothing else.** It is a local `FakeBackendV2` — the noise
  model and coupling map of a retired small IBM device (5-qubit at the
  default circuit size; the pick is qubit-aware, e.g. 7-qubit at
  `--profile q6`), simulated on your machine. No account, no network,
  exhibition-safe.
- **Real backend:** an IBM Quantum account. Create a token at
  [quantum.cloud.ibm.com](https://quantum.cloud.ibm.com), then either
  `export QISKIT_IBM_TOKEN=<token>` before starting the server, or save it
  once with `QiskitRuntimeService.save_account(channel="ibm_quantum_platform",
  token=...)`. Expect **minutes-to-hours of queue time** and seconds per
  decision — for a booth, run the fake backend live and mention the real one;
  or pre-run the CLI and show the transcript:

  ```sh
  python -m traqmania.hardware lap --track oval --fake        # dry-run flow
  python -m traqmania.hardware lap --track oval               # real device
  python -m traqmania.hardware sprint --track oval --fake --iterations 20
  python -m traqmania.hardware lap --fake --profile q6        # 6-qubit variant
  ```

- Never paste a token into a notebook or config file that might get
  committed.

## Troubleshooting

**Port already in use** (`[Errno 48] address already in use`): a previous
instance is still running. `lsof -ti:8000 | xargs kill`, or start on another
port: `./run.sh --port 8010`.

**Status pill stuck on "reconnecting…":** the browser lost the websocket. The
client auto-reconnects with backoff (0.5 s → 8 s), so if the server is up it
recovers by itself — if it stays stuck, the server process died: check the
terminal, `curl http://127.0.0.1:8000/health`, and restart `./run.sh`. When
serving other devices, remember plain `./run.sh` binds `127.0.0.1` only — use
the exhibition profile or `--host 0.0.0.0`.

**Choppy rendering / sluggish Pi:** use the right profile (`--profile pi4` or
`pi5` — lower broadcast/telemetry rates, smaller training batches). Train
warm-start only, keep episodes modest, and prefer oval/chicane; gp is the
heavyweight track (cold training takes ~2000 episodes). Close other browser
tabs — the canvas renderer is cheap but not free.

**"training is already running" / "cannot change track while training is
running":** press **Stop** on the Training tab first; track switches and new
runs are blocked while a job is live.

**Resetting ghosts:** the best-lap ghost per track lives in
`traqmania/data/ghosts/<track>.json` and is overwritten whenever anyone —
including a talented visitor — beats it with a clean lap. To reset to the
bundled records in a git checkout: `git checkout -- traqmania/data/ghosts/`.
To simply clear one: delete the file and restart (no ghost is shown until a
new clean lap is driven). In a container, ghosts reset with the container.

**Hardware mode fails immediately:** with backend *real*, the server needs
`QISKIT_IBM_TOKEN` (see prerequisites); the error message contains the setup
steps. With *fake*, check that `qiskit-ibm-runtime` is installed
(`pip install -e ".[hardware]"`).
