"""DemoSession tests: attract ticking, race input mapping, training-mode smoke.

Everything drives the session via direct ``tick()`` calls (no asyncio, no
wall-clock sleeps beyond a 1 ms GIL-yield in the training smoke test).
"""

import time

import numpy as np

from traqmania.config import load_config
from traqmania.server import protocol as P
from traqmania.server.runtime import load_agent, resolve_training_cfg, track_payload
from traqmania.server.session import DemoSession, keys_to_controls


def make_config(**sections):
    """load_config() with per-section dict overrides, e.g. reward={'max_decisions': 50}."""
    config = load_config()
    for name, overrides in sections.items():
        merged = dict(config[name])
        merged.update(overrides)
        config[name] = merged
    return config


# ------------------------------------------------------------------- runtime


def test_load_agent_bundled_weights():
    quantum = load_agent("quantum", "oval")
    mlp = load_agent("mlp", "oval")
    obs = np.zeros((2, 4))
    assert quantum.q_values(obs).shape == (2, 4)
    assert quantum.expectations(obs).shape == (2, 4)
    assert mlp.q_values(obs).shape == (2, 4)
    warm = load_agent("quantum", "oval", warm=True)
    assert not np.allclose(warm.get_params(), quantum.get_params())


def test_resolve_training_cfg_presets_and_warm():
    config = load_config()
    base = resolve_training_cfg(config, "oval")
    assert base == config["training"]

    gp = resolve_training_cfg(config, "gp")
    assert (gp["episodes"], gp["epsilon_decay_episodes"], gp["gamma"]) == (2000, 1200, 0.99)

    warm = resolve_training_cfg(config, "oval", warm=True)
    assert warm["episodes"] == 150
    assert warm["epsilon_start"] == 0.25
    assert warm["epsilon_decay_episodes"] == 40

    warm_gp = resolve_training_cfg(config, "gp", warm=True)
    assert warm_gp["episodes"] == 500
    assert warm_gp["epsilon_start"] == 0.35
    assert warm_gp["epsilon_end"] == 0.05  # inherited from [training_warm]
    assert warm_gp["epsilon_decay_episodes"] == 150
    assert warm_gp["gamma"] == 0.99


def test_track_payload_shape():
    session = DemoSession(load_config())
    payload = track_payload(session.track)
    assert set(payload) == {"name", "half_width", "total_length", "checkpoints", "theme",
                            "start", "centerline", "left", "right"}
    assert set(payload["start"]) == {"x", "y", "theta"}
    assert len(payload["left"]) == len(payload["centerline"]) == len(payload["right"])
    # boundaries sit half_width away from the centerline
    gap = np.linalg.norm(np.asarray(payload["left"]) - np.asarray(payload["centerline"]), axis=1)
    assert np.allclose(gap, payload["half_width"])


# ------------------------------------------------------------------- attract


CAR_KEYS = {"id", "kind", "x", "y", "theta", "v", "lap", "progress",
            "last_lap_time", "off_track"}


def test_attract_ticks_broadcast_and_move():
    session = DemoSession(load_config())
    assert session.mode == "attract"

    for _ in range(30):
        session.tick()
    msgs = session.drain_outbox()

    states = [m for m in msgs if m["type"] == "state"]
    assert len(states) >= 5  # 30 substeps at broadcast_hz=20 -> every 3rd substep
    for state in states:
        assert P.parse_server(state).mode == "attract"  # validates the payload shape
        assert len(state["cars"]) == 1
        assert set(state["cars"][0]) >= CAR_KEYS

    first, last = states[0]["cars"][0], states[-1]["cars"][0]
    assert last["kind"] == "quantum"
    assert last["v"] > 0.0
    assert (first["x"], first["y"]) != (last["x"], last["y"])
    assert states[-1]["t"] > states[0]["t"] > 0.0

    quantum = [m for m in msgs if m["type"] == "quantum"]
    assert quantum  # decisions at 10 Hz -> several within 30 substeps
    assert len(quantum[0]["expectations"]) == 4
    assert len(quantum[0]["q_values"]) == 4
    assert 0 <= quantum[0]["action"] < 4
    assert quantum[0]["car_id"] == "quantum"


def test_welcome_payload():
    session = DemoSession(load_config())
    welcome = session.welcome_payload()
    assert welcome["type"] == "welcome"
    assert welcome["mode"] == "attract"
    assert set(welcome["tracks"]) == {"oval", "chicane", "gp"}
    assert welcome["circuit_spec"]["n_qubits"] == 4
    assert "attract_idle_seconds" in welcome["ui"]
    assert welcome["track"]["name"] == "oval"


# ---------------------------------------------------------------------- race


def test_keys_to_controls_mapping():
    T, B, L, R = P.KEY_THROTTLE, P.KEY_BRAKE, P.KEY_LEFT, P.KEY_RIGHT
    assert keys_to_controls(0) == (0.0, 0.0, 0.0)
    assert keys_to_controls(T) == (0.0, 1.0, 0.0)
    assert keys_to_controls(B) == (0.0, 0.0, 1.0)
    assert keys_to_controls(T | B) == (0.0, 0.0, 1.0)  # brake overrides throttle
    assert keys_to_controls(L) == (-1.0, 0.0, 0.0)
    assert keys_to_controls(R) == (1.0, 0.0, 0.0)
    assert keys_to_controls(T | L) == (-1.0, 1.0, 0.0)
    assert keys_to_controls(T | R) == (1.0, 1.0, 0.0)
    assert keys_to_controls(L | R) == (0.0, 0.0, 0.0)  # opposite steering cancels
    assert keys_to_controls(T | B | L | R) == (0.0, 0.0, 1.0)


def test_race_mode_human_input_drives_car():
    session = DemoSession(load_config())
    session.handle_message(P.Race(action="start", opponent="mlp"))
    assert session.mode == "race"
    assert {car.kind for car in session.cars} == {"human", "mlp"}

    session.handle_message(P.Input(keys=P.KEY_THROTTLE))
    for _ in range(12):
        session.tick()

    human = next(car for car in session.cars if car.kind == "human")
    assert human.controls == (0.0, 1.0, 0.0)
    assert human.state[3] > 0.0  # throttle accelerated the car

    states = [m for m in session.drain_outbox() if m["type"] == "state"]
    assert states and len(states[-1]["cars"]) == 2
    kinds = {car["kind"] for car in states[-1]["cars"]}
    assert kinds == {"human", "mlp"}


# ------------------------------------------------------------------ training


def test_training_mode_smoke():
    config = make_config(
        reward={"max_decisions": 50},
        training={"n_parallel_envs": 4, "replay_size": 2000, "batch_size": 16,
                  "epsilon_decay_episodes": 6, "target_sync_every": 50},
    )
    session = DemoSession(config)
    session.handle_message(P.Train(action="start", agent="mlp", track="oval", episodes=8))
    assert session.mode == "train"
    assert "mlp" in session.jobs

    telemetry, done_events, train_states = [], [], []
    deadline = time.time() + 60.0
    try:
        while time.time() < deadline and not done_events:
            session.tick()
            for msg in session.drain_outbox():
                if msg["type"] == "telemetry":
                    telemetry.append(msg)
                elif msg["type"] == "event" and msg["kind"] == "training_done":
                    done_events.append(msg)
                elif msg["type"] == "state" and msg["cars"]:
                    train_states.append(msg)
            time.sleep(0.001)  # yield the GIL to the training thread
    finally:
        session.shutdown()

    assert done_events, "no training_done event within 60s"
    assert done_events[0]["agent"] == "mlp"
    assert len(telemetry) >= 1
    sample = telemetry[-1]
    assert P.parse_server(sample).agent == "mlp"  # validates full telemetry shape
    assert len(sample["returns_tail"]) <= 100
    assert sample["episode"] == 7  # 8 episodes, 0-indexed last callback
    history = session.jobs["mlp"].history
    assert history is not None and "best_eval" in history
    assert train_states, "no live training car states broadcast"
    assert train_states[-1]["cars"][0]["kind"] == "mlp"


def test_train_stop_ends_training():
    config = make_config(
        reward={"max_decisions": 400},
        training={"n_parallel_envs": 4, "replay_size": 2000, "batch_size": 16},
    )
    session = DemoSession(config)
    session.handle_message(P.Train(action="start", agent="mlp", episodes=100_000))
    job = session.jobs["mlp"]
    session.handle_message(P.Train(action="stop", agent="mlp"))
    assert job.stop_event.is_set()
    job.thread.join(timeout=30.0)
    assert not job.thread.is_alive(), "training thread did not stop"
    # the reaper announces the stopped run on a later tick
    deadline = time.time() + 5.0
    done = []
    while time.time() < deadline and not done:
        session.tick()
        done = [m for m in session.drain_outbox()
                if m["type"] == "event" and m["kind"] == "training_done"]
    assert done
