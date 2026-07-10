# Concept: traQmania × real TrackMania 2020 (via tmrl)

*Status: concept only — not scheduled. Captured 2026-07-10 so we can pick it up later.*

## Goal

An optional mode where the traQmania quantum agent drives the **real TrackMania 2020
game** — "a 56-parameter quantum circuit drives a real video game" — as a show-stopper
for talks. The browser demo remains the always-works exhibit; this is additive.

## Building blocks

- [tmrl](https://github.com/trackmania-rl/tmrl) — actively maintained RL framework for
  TM2020 via OpenPlanet: real-time gymnasium-style env (`rtgym`), observations as
  **19-beam lidar × 4 stacked frames + speed** (or 64×64×4 grayscale vision), game
  input injection, single-server/multi-worker training.
- TM2020 free Starter Access + OpenPlanet plugin. **Windows only.** One game instance,
  real-time — no vectorization, no fastsim: data collection is ~1,000× slower than our
  simulator.
- traQmania's `QFunction` contract: any adapter env that yields our observation shape
  can reuse `QuantumQFunction`, `DQNTrainer`, and the weights format unchanged.

## Parameter reality check

| Pipeline | Params |
|---|---|
| tmrl default lidar SAC (MLP 256×256, ~80-dim input) | ~90k (actor), ~270k with critics |
| tmrl vision CNN | millions |
| Yosh's record-setting IQN agents | millions, months of training |
| traQmania VQC | **56** (4q) / ~120 (6q) |

A VQC cannot and should not compete on lap times. The story is **parameter frugality**:
competent driving at 3–4 orders of magnitude fewer trained parameters.

## Phased approach (recommended order)

1. **Pure quantum policy** — fixed, non-learned downsampling: latest lidar frame,
   19 beams → 3–5 rays + speed (matches our obs design; 6-qubit variant fits 5 rays).
   Adapter env implements our env protocol over `rtgym`. Train overnight on a Windows
   box with our DQN (decisions ~10 Hz real-time). Deliverable: quantum ghost drives a
   simple TM2020 track. Cleanest narrative; zero attribution ambiguity.
2. **Frozen hybrid** — distill a classical encoder from a trained tmrl SAC agent
   (or use its trunk), freeze it, train only the VQC head on its 4–8 features.
   Attribution stays clean ("the quantum part is the only thing that learned").
3. **Joint hybrid (research)** — classical encoder + VQC head trained jointly
   (gradients through the quantum block via adjoint/param-shift; TorchConnector
   pattern). **Mandatory ablation control**: identical pipeline with a same-parameter-
   count classical head, else no claim about the quantum contribution is defensible.
   Only worth doing as a publishable study.

## Constraints & risks

- Windows + game + plugin fragility → never the primary exhibit.
- Live on-stage *training* infeasible (real-time env); show inference + pre-trained.
- Sim-trained traQmania weights do NOT transfer (different dynamics) — training must
  happen in-game (overnight is realistic for phase 1's tiny model).
- tmrl's default algorithms are continuous-action SAC; our DQN is discrete — either
  map 4 discrete actions to game inputs (fine for phase 1) or extend to a small
  continuous policy later.

## Effort estimates

Phase 1: 2–4 days (adapter + overnight training + tuning) on a Windows machine.
Phase 2: +2–3 days given a trained SAC baseline. Phase 3: research-project scale.
