"""DemoSession tests: attract ticking, race input mapping, training-mode smoke.

Everything drives the session via direct ``tick()`` calls (no asyncio, no
wall-clock sleeps beyond a 1 ms GIL-yield in the training smoke test).
"""

import json
import re
import time

import numpy as np

from traqmania.config import load_config
from traqmania.server import protocol as P
from traqmania.server.runtime import (
    evolution_stage_specs,
    load_agent,
    load_ghost,
    resolve_training_cfg,
    save_ghost,
    track_payload,
)
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


def test_track_payload_shape(tmp_path):
    session = DemoSession(load_config(), ghosts_dir=tmp_path)
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


def test_attract_ticks_broadcast_and_move(tmp_path):
    session = DemoSession(load_config(), ghosts_dir=tmp_path)
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


def test_welcome_payload(tmp_path):
    session = DemoSession(load_config(), ghosts_dir=tmp_path)
    welcome = session.welcome_payload()
    assert welcome["type"] == "welcome"
    assert welcome["mode"] == "attract"
    assert set(welcome["tracks"]) == {"oval", "chicane", "gp", "combo"}
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
    # car steer +1 = theta increase = counterclockwise = LEFT on screen
    assert keys_to_controls(L) == (1.0, 0.0, 0.0)
    assert keys_to_controls(R) == (-1.0, 0.0, 0.0)
    assert keys_to_controls(T | L) == (1.0, 1.0, 0.0)
    assert keys_to_controls(T | R) == (-1.0, 1.0, 0.0)
    assert keys_to_controls(L | R) == (0.0, 0.0, 0.0)  # opposite steering cancels
    assert keys_to_controls(T | B | L | R) == (0.0, 0.0, 1.0)


def test_race_mode_human_input_drives_car(tmp_path):
    session = DemoSession(load_config(), ghosts_dir=tmp_path)
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


def test_analog_input_overrides_keys(tmp_path):
    session = DemoSession(load_config(), ghosts_dir=tmp_path)
    session.handle_message(P.Race(action="start", opponent="quantum"))

    # any analog field present -> keys bitmask ignored (brake bit set but not used)
    session.handle_message(P.Input(keys=P.KEY_BRAKE, steer=-0.5, throttle=0.7))
    for _ in range(12):
        session.tick()
    human = next(car for car in session.cars if car.kind == "human")
    # stick steer -0.5 (leftward) flips to car steer +0.5 (theta+ = screen-left)
    assert human.controls == (0.5, 0.7, 0.0)  # absent brake axis defaults to 0.0
    assert human.state[3] > 0.0  # analog throttle accelerated the car

    # a keys-only input clears the analog override
    session.handle_message(P.Input(keys=P.KEY_THROTTLE))
    session.tick()
    assert human.controls == (0.0, 1.0, 0.0)


# ----------------------------------------------------------------- evolution


def test_evolution_stage_specs_oval_and_fallback():
    specs = evolution_stage_specs("oval")  # bundled stage snapshots
    assert len(specs) == 4
    labels = [label for label, _ in specs]
    # early stages carry their episode label; the finale is the shipped driver
    assert all(re.fullmatch(r"ep \d+", label) for label in labels[:-1])
    assert labels[-1] == "best"
    assert specs[-1][1].name == "quantum_oval.npz"
    episodes = [int(label.split()[1]) for label in labels[:-1]]
    assert episodes == sorted(episodes) and len(set(episodes)) == 3
    assert all(path.is_file() for _, path in specs)
    # chicane has no stage files -> [warm-start, best], no duplicated cars
    fallback = evolution_stage_specs("chicane")
    assert len(fallback) == 2
    assert fallback[0][0].startswith("warm-start")
    assert fallback[1][0].startswith("best")
    assert all(path.is_file() for _, path in fallback)


def test_evolution_mode_tick_shape(tmp_path):
    session = DemoSession(load_config(), ghosts_dir=tmp_path)
    session.handle_message(P.SetMode(mode="evolution"))
    assert session.mode == "evolution"
    assert len(session.cars) == 4

    for _ in range(30):
        session.tick()
    states = [m for m in session.drain_outbox() if m["type"] == "state"]
    assert states
    for state in states:
        assert P.parse_server(state).mode == "evolution"  # validates the payload shape
        assert len(state["cars"]) == 4
    last = states[-1]["cars"]
    assert [c["id"] for c in last] == ["stage1", "stage2", "stage3", "stage4"]
    assert all(c["kind"] == "quantum" for c in last)
    assert [c["label"] for c in last] == [label for label, _ in evolution_stage_specs("oval")]
    assert all("ghost" not in c for c in last)
    assert any(c["v"] > 0.0 for c in last)


# --------------------------------------------------------------------- ghost


def make_fake_ghost(track, n_points: int = 40) -> dict:
    """Circular fake best-lap trajectory roughly following the track start pose."""
    x0, y0, theta0 = track.start_pose()
    pts = [[x0 + i * 0.5, y0, theta0] for i in range(n_points)]
    return {"lap_time": 12.3, "kind": "quantum", "points": pts}


def test_ghost_save_load_round_trip(tmp_path):
    assert load_ghost("oval", tmp_path) is None
    ghost = {"lap_time": 21.5, "kind": "mlp", "driver": "mlp (oval-trained)",
             "points": [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]}
    path = save_ghost("oval", ghost, tmp_path)
    assert path == tmp_path / "oval.json"
    assert load_ghost("oval", tmp_path) == ghost
    # records from before driver provenance load with driver=None
    (tmp_path / "chicane.json").write_text(json.dumps(
        {"lap_time": 3.0, "kind": "human", "points": [[0, 0, 0], [1, 1, 1]]}))
    assert load_ghost("chicane", tmp_path)["driver"] is None
    # invalid records are rejected
    (tmp_path / "gp.json").write_text(json.dumps({"lap_time": 5.0, "points": []}))
    assert load_ghost("gp", tmp_path) is None


def test_attract_streams_injected_ghost_car(tmp_path):
    config = load_config()
    session = DemoSession(config, ghosts_dir=tmp_path)  # no ghost yet
    session.tick()
    state = [m for m in session.drain_outbox() if m["type"] == "state"]
    assert all(c["id"] != "ghost" for s in state for c in s["cars"])

    save_ghost("oval", make_fake_ghost(session.track), tmp_path)
    session = DemoSession(config, ghosts_dir=tmp_path)
    for _ in range(30):
        session.tick()
    states = [m for m in session.drain_outbox() if m["type"] == "state"]
    assert states
    ghosts = []
    for s in states:
        P.parse_server(s)  # validates ghost/label fields on the wire
        ghost = next(c for c in s["cars"] if c["id"] == "ghost")
        assert ghost["ghost"] is True
        assert ghost["kind"] == "quantum"
        assert ghost["label"] == "best 12.3s"
        assert ghost["last_lap_time"] == 12.3
        ghosts.append(ghost)
    # the replay interpolates along the stored points, so the ghost moves
    assert ghosts[-1]["x"] != ghosts[0]["x"]

    # race mode also streams the ghost alongside human + opponent
    session.handle_message(P.Race(action="start", opponent="quantum"))
    session.tick()
    session.tick()
    session.tick()
    race_states = [m for m in session.drain_outbox() if m["type"] == "state"]
    ids = {c["id"] for c in race_states[-1]["cars"]}
    assert ids == {"human", "quantum", "ghost"}


# ------------------------------------------------------------------ training


def test_training_mode_smoke(tmp_path):
    config = make_config(
        reward={"max_decisions": 50},
        training={"n_parallel_envs": 4, "replay_size": 2000, "batch_size": 16,
                  "epsilon_decay_episodes": 6, "target_sync_every": 50},
    )
    session = DemoSession(config, ghosts_dir=tmp_path)
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
    assert "best_lap_s" in sample and "lap_times" in sample
    assert isinstance(sample["lap_times"], list) and len(sample["lap_times"]) <= 50
    assert all(isinstance(e, int) and isinstance(t, float) for e, t in sample["lap_times"])
    assert sample["episode"] == 7  # 8 episodes, 0-indexed last callback
    history = session.jobs["mlp"].history
    assert history is not None and "best_eval" in history
    assert train_states, "no live training car states broadcast"
    assert train_states[-1]["cars"][0]["kind"] == "mlp"


def test_train_stop_ends_training(tmp_path):
    config = make_config(
        reward={"max_decisions": 400},
        training={"n_parallel_envs": 4, "replay_size": 2000, "batch_size": 16},
    )
    session = DemoSession(config, ghosts_dir=tmp_path)
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
