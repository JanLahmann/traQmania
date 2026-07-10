"""DemoSession hardware mode, exercised entirely on LOCAL fake backends.

Same setup as tests/test_hardware.py: no network, no IBM account — the worker
thread calls ``get_backend(use_fake=True)`` and runs on the Aer-simulated fake
twin. Skipped wholesale when qiskit-ibm-runtime is not installed. The session
is driven via direct ``tick()`` calls (no asyncio); ticking loops sleep 1 ms
to yield the GIL to the hardware thread, exactly like the training smoke test.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("qiskit_ibm_runtime")

from traqmania.config import load_config  # noqa: E402
from traqmania.server import protocol as P  # noqa: E402
from traqmania.server.session import DemoSession  # noqa: E402

DEADLINE_S = 180.0


def run_until(session: DemoSession, phases: set[str], deadline_s: float = DEADLINE_S):
    """Tick until a hardware_status with a phase in ``phases`` arrives.

    Returns (statuses, states) collected along the way; every message is
    round-tripped through ``parse_server`` to validate its wire shape.
    """
    statuses: list[dict] = []
    states: list[dict] = []
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        session.tick()
        for msg in session.drain_outbox():
            if msg["type"] == "hardware_status":
                P.parse_server(msg)
                statuses.append(msg)
            elif msg["type"] == "state":
                P.parse_server(msg)
                states.append(msg)
        if any(s["phase"] in phases for s in statuses):
            return statuses, states
        time.sleep(0.001)  # yield the GIL to the hardware thread
    raise AssertionError(
        f"no hardware_status phase in {phases} within {deadline_s}s; "
        f"saw {[s['phase'] for s in statuses]}"
    )


def test_hardware_mode_idle_then_lap_and_replay(tmp_path):
    session = DemoSession(load_config(), ghosts_dir=tmp_path)
    session.handle_message(P.SetMode(mode="hardware"))
    assert session.mode == "hardware"
    assert len(session.cars) == 1 and session.cars[0].kind == "quantum"

    # entering emits an idle status; the fastsim car waits at the start line
    for _ in range(30):
        session.tick()
    msgs = session.drain_outbox()
    statuses = [m for m in msgs if m["type"] == "hardware_status"]
    assert statuses and statuses[0]["phase"] == "idle"
    states = [m for m in msgs if m["type"] == "state"]
    assert states
    assert all(m["mode"] == "hardware" for m in states)
    assert states[-1]["cars"][0]["v"] == 0.0  # idle: not driving yet

    try:
        session.handle_message(
            P.HardwareMsg(action="lap", backend="fake", shots=128, max_decisions=3))
        assert session.hw_job is not None

        # only one hardware job at a time
        session.handle_message(P.HardwareMsg(action="sprint", backend="fake"))
        busy = [m for m in session.drain_outbox() if m["type"] == "hardware_status"]
        assert busy and busy[-1]["phase"] == "error"
        assert "already running" in busy[-1]["message"]

        statuses, _states = run_until(session, {"replay", "error"})
        phases = [s["phase"] for s in statuses]
        assert "error" not in phases, statuses
        for expected in ("connecting", "transpiling", "running", "done", "replay"):
            assert expected in phases, phases
        # phases arrive in order
        order = [phases.index(p) for p in ("connecting", "transpiling", "running",
                                           "done", "replay")]
        assert order == sorted(order)

        running = [s for s in statuses if s["phase"] == "running"]
        assert running[0]["decision"] >= 1
        assert running[0]["seconds_per_decision"] > 0.0
        assert running[0]["backend_name"]
        done = next(s for s in statuses if s["phase"] == "done")
        assert done["seconds_per_decision"] > 0.0  # 3 decisions: no lap_time expected

        # replay: pinned hardware ghost loops alongside a live fastsim car
        for _ in range(30):
            session.tick()
        replay_states = [m for m in session.drain_outbox() if m["type"] == "state"]
        assert replay_states
        last = replay_states[-1]
        P.parse_server(last)
        assert {c["id"] for c in last["cars"]} == {"quantum", "hardware"}
        hw = next(c for c in last["cars"] if c["id"] == "hardware")
        assert hw["ghost"] is True
        assert hw["kind"] == "quantum"
        assert hw["label"] == "hardware lap"
        first_hw = next(c for c in replay_states[0]["cars"] if c["id"] == "hardware")
        assert (first_hw["x"], first_hw["y"]) != (hw["x"], hw["y"])  # ghost moves
        fastsim = next(c for c in last["cars"] if c["id"] == "quantum")
        assert "ghost" not in fastsim
    finally:
        session.shutdown()


def test_hardware_sprint_smoke(tmp_path):
    session = DemoSession(load_config(), ghosts_dir=tmp_path)
    try:
        # a hardware message outside hardware mode switches into it
        session.handle_message(
            P.HardwareMsg(action="sprint", backend="fake", iterations=2, shots=128))
        assert session.mode == "hardware"

        statuses, states = run_until(session, {"done", "error"})
        phases = [s["phase"] for s in statuses]
        assert "error" not in phases, statuses
        assert "connecting" in phases and "transpiling" in phases

        running = [s for s in statuses if s["phase"] == "running"]
        assert [s["iteration"] for s in running] == [1, 2]
        assert all("loss" in s for s in running)

        done = next(s for s in statuses if s["phase"] == "done")
        assert "eval_return_before" in done and "eval_return_after" in done
        assert states  # idle car broadcast throughout; sprint has no replay
        assert session._hw_replay is None
    finally:
        session.shutdown()


def test_hardware_abort_and_mode_switch(tmp_path):
    session = DemoSession(load_config(), ghosts_dir=tmp_path)
    try:
        session.handle_message(P.SetMode(mode="hardware"))
        # abort with nothing running -> idle status, no error
        session.handle_message(P.HardwareMsg(action="abort", backend="fake"))
        msgs = [m for m in session.drain_outbox() if m["type"] == "hardware_status"]
        assert msgs[-1]["phase"] == "idle"

        session.handle_message(
            P.HardwareMsg(action="lap", backend="fake", shots=128, max_decisions=500))
        run_until(session, {"running", "error"})
        session.handle_message(P.HardwareMsg(action="abort", backend="fake"))
        assert session.hw_job.stop_event.is_set()
        statuses, _ = run_until(session, {"idle", "done", "error"})
        assert statuses[-1]["phase"] == "idle"
        assert "aborted" in statuses[-1]["message"]
        assert session.hw_job is None
        assert session._hw_replay is None  # aborted laps do not replay

        # switching modes while a job runs abandons it silently
        session.handle_message(
            P.HardwareMsg(action="lap", backend="fake", shots=128, max_decisions=500))
        job = session.hw_job
        session.handle_message(P.SetMode(mode="attract"))
        assert session.mode == "attract"
        assert job.stop_event.is_set() and job.abandoned
        job.thread.join(timeout=60.0)
        assert not job.thread.is_alive()
        for _ in range(5):
            session.tick()
        leaked = [m for m in session.drain_outbox() if m["type"] == "hardware_status"]
        assert leaked == []
        assert session.hw_job is None
    finally:
        session.shutdown()
