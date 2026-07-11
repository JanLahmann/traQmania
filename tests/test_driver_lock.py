"""Driver lock: exclusive control of the shared session (first interacting
client drives, others spectate), idle takeover, release on disconnect, and
the per-client ``control`` status message."""

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from traqmania.config import load_config  # noqa: E402
from traqmania.server import protocol as P  # noqa: E402
from traqmania.server.app import create_app  # noqa: E402
from traqmania.server.ws import DriverLock  # noqa: E402

# ------------------------------------------------------------------ unit: lock


def test_lock_first_come_and_refresh():
    lock = DriverLock(idle_s=90.0, turn_s=120.0)
    a, b = object(), object()
    assert not lock.locked
    assert lock.try_control(a, now=0.0)   # free wheel: a takes it
    assert lock.locked and lock.driving(a)
    assert not lock.try_control(b, now=10.0)  # a is active: b joins the line
    assert lock.queue_pos(b) == 1
    assert lock.try_control(a, now=50.0)  # a keeps refreshing its idle timer
    assert not lock.tick(now=120.0)       # 70 s since refresh, turn not over
    assert lock.tick(now=141.0)           # 91 s idle: the line advances to b
    assert lock.driving(b) and not lock.driving(a)


def test_lock_release_on_disconnect():
    lock = DriverLock(idle_s=90.0)
    a, b = object(), object()
    assert lock.try_control(a, now=0.0)
    assert not lock.release(b)   # non-driver release is a no-op
    assert lock.release(a)
    assert not lock.locked
    assert lock.try_control(b, now=1.0)  # immediately available


# ------------------------------------------------------------------ unit: queue


def test_queue_join_order_and_turn_expiry():
    lock = DriverLock(idle_s=90.0, turn_s=120.0)
    a, b, c = object(), object(), object()
    assert lock.try_control(a, now=0.0)
    assert not lock.try_control(b, now=1.0)  # joins the line
    assert not lock.try_control(c, now=2.0)
    assert lock.queue_pos(b) == 1 and lock.queue_pos(c) == 2
    assert lock.waiting == 2
    assert lock.turn_ends_in(now=10.0) == 110  # countdown runs: someone waits

    assert lock.try_control(a, now=80.0)     # a stays active
    assert not lock.tick(now=100.0)          # inside its turn: a keeps driving
    assert lock.try_control(a, now=110.0)    # refreshes idle, NOT the turn
    assert lock.tick(now=121.0)              # 120 s turn over: b takes over
    assert lock.driving(b)
    assert lock.queue_pos(c) == 1            # c moved up
    assert lock.queue_pos(b) is None

    # b drives its full turn too, then c gets the wheel
    assert lock.tick(now=242.0)
    assert lock.driving(c)
    assert lock.waiting == 0
    assert lock.turn_ends_in(now=243.0) is None  # nobody waits: no countdown


def test_no_turn_limit_without_queue_and_idle_release():
    lock = DriverLock(idle_s=90.0, turn_s=120.0)
    a = object()
    assert lock.try_control(a, now=0.0)
    a_active = 0.0
    for now in (60.0, 500.0, 1000.0):  # solo driver far beyond turn_s
        assert not lock.tick(now=a_active + 60.0)
        assert lock.try_control(a, now=now)
        a_active = now
    assert lock.tick(now=a_active + 91.0)  # idled out: wheel frees up
    assert not lock.locked


def test_remove_hands_wheel_to_next_in_line():
    lock = DriverLock(idle_s=90.0, turn_s=120.0)
    a, b, c = object(), object(), object()
    assert lock.try_control(a, now=0.0)
    assert not lock.try_control(b, now=1.0)
    assert not lock.try_control(c, now=2.0)
    assert not lock.remove(c)                # waiter leaves: no driver change
    assert lock.waiting == 1
    assert lock.remove(a, now=3.0)           # driver leaves: b takes over
    assert lock.driving(b) and lock.waiting == 0


# ------------------------------------------------------------------- protocol


def test_control_message_round_trip():
    msg = P.parse_server({"type": "control", "driving": True, "locked": True,
                          "watchers": 2, "waiting": 1, "queue_pos": None,
                          "turn_ends_in_s": 87})
    assert isinstance(msg, P.Control)
    assert msg.driving and msg.locked and msg.watchers == 2
    assert msg.waiting == 1 and msg.queue_pos is None and msg.turn_ends_in_s == 87
    with pytest.raises(P.ProtocolError):
        P.parse_server({"type": "control", "driving": 1, "locked": True,
                        "watchers": 0})


# ------------------------------------------------------------------ e2e: ws


def recv_until(ws, type_tag, pred=None, limit=50):
    for _ in range(limit):
        msg = ws.receive_json()
        if msg["type"] == type_tag and (pred is None or pred(msg)):
            return msg
    raise AssertionError(f"no matching '{type_tag}' message within {limit} messages")


def test_two_clients_one_wheel():
    client = fastapi_testclient.TestClient(create_app(load_config()))
    with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
        recv_until(a, "welcome")
        recv_until(b, "welcome")

        # a interacts first and takes the wheel
        a.send_json({"type": "set_mode", "mode": "train"})
        ctl_a = recv_until(a, "control", lambda m: m["driving"])
        assert ctl_a["locked"]
        P.parse_server(ctl_a)

        # b's action is dropped: it joins the line and learns its place
        b.send_json({"type": "set_mode", "mode": "race"})
        ctl_b = recv_until(b, "control", lambda m: m["queue_pos"] is not None)
        assert not ctl_b["driving"] and ctl_b["locked"]
        assert ctl_b["queue_pos"] == 1
        assert isinstance(ctl_b["turn_ends_in_s"], int)
        P.parse_server(ctl_b)

        # ... and the driver sees a countdown start
        ctl_a2 = recv_until(a, "control", lambda m: m["turn_ends_in_s"] is not None)
        assert ctl_a2["driving"] and ctl_a2["waiting"] == 1

    # both sockets closed: a fresh client finds the wheel free
    with client.websocket_connect("/ws") as c:
        recv_until(c, "welcome")
        c.send_json({"type": "set_mode", "mode": "attract"})
        recv_until(c, "control", lambda m: m["driving"])
