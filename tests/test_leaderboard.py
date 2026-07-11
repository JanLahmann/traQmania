"""Leaderboard: named human race laps rank; AI laps are unranked reference
rows; boards persist per bundled track and reset for ephemeral tracks."""

import numpy as np
import pytest

from traqmania.config import load_config
from traqmania.server import protocol as P
from traqmania.server.runtime import load_leaderboard, save_leaderboard
from traqmania.server.session import DemoSession, _Car


@pytest.fixture()
def session(tmp_path):
    session = DemoSession(load_config(), ghosts_dir=tmp_path)
    session.drain_outbox()
    return session


def human(session):
    x, y, theta = session.track.start_pose()
    return _Car(id="human", kind="human", state=np.array([x, y, theta, 0.0]))


def quantum(session):
    x, y, theta = session.track.start_pose()
    return _Car(id="quantum", kind="quantum", state=np.array([x, y, theta, 0.0]))


def board_msgs(session):
    return [m for m in session.drain_outbox() if m["type"] == "leaderboard"]


# ------------------------------------------------------------------ recording


def test_named_human_race_lap_ranks(session):
    session.mode = "race"
    session.handle_message(P.parse_client({"type": "set_name", "name": "  Ada "}))
    assert session.racer_name == "Ada"
    session._record_leaderboard(human(session), 14.2)
    session._record_leaderboard(human(session), 13.1)
    msgs = board_msgs(session)
    assert msgs, "leaderboard updates broadcast"
    entries = msgs[-1]["entries"]
    assert [e["name"] for e in entries] == ["Ada", "Ada"]
    assert entries[0]["lap_s"] == 13.1  # fastest first
    P.parse_server(msgs[-1])

    # persisted: a fresh session on the same dirs sees the board
    reloaded = load_leaderboard(session.track_name, session._leaderboard_dir)
    assert len(reloaded["entries"]) == 2


def test_anonymous_or_nonrace_human_laps_do_not_rank(session):
    session.mode = "race"
    session._record_leaderboard(human(session), 12.0)  # no name set
    session.racer_name = "Bo"
    session.mode = "attract"
    session._record_leaderboard(human(session), 12.0)  # not in race mode
    assert not board_msgs(session)
    assert session._board["entries"] == []


def test_agent_laps_are_references_not_entries(session):
    session.mode = "race"
    session.racer_name = "Cy"
    session._record_leaderboard(quantum(session), 15.0)
    session._record_leaderboard(quantum(session), 13.5)  # improves the reference
    session._record_leaderboard(quantum(session), 14.9)  # slower: ignored
    msgs = board_msgs(session)
    board = msgs[-1]
    assert board["entries"] == []  # AI never ranks
    assert board["references"] == [
        {"kind": "quantum", "driver": f"{session.track_name}-trained", "lap_s": 13.5}
    ]
    P.parse_server(board)


def test_board_caps_at_ten_ranked_entries(session):
    session.mode = "race"
    session.racer_name = "Dee"
    for i in range(12):
        session._record_leaderboard(human(session), 20.0 - i * 0.1)
    assert len(session._board["entries"]) == 10
    assert session._board["entries"][0]["lap_s"] == pytest.approx(18.9)


def test_ephemeral_tracks_get_fresh_unpersisted_boards(session):
    session.mode = "race"
    session.racer_name = "Eve"
    session._record_leaderboard(human(session), 13.0)
    session.handle_message(P.parse_client({"type": "set_track", "track": "random",
                                           "seed": 5}))
    board = board_msgs(session)[-1]
    assert board["entries"] == []  # fresh board for the generated track
    session._record_leaderboard(human(session), 30.0)
    # nothing persisted for random tracks; the oval board survives untouched
    stored = load_leaderboard(session.track_name, session._leaderboard_dir)
    assert stored["entries"] == []
    oval = load_leaderboard("oval", session._leaderboard_dir)
    assert [e["name"] for e in oval["entries"]] == ["Eve"]


# ---------------------------------------------------------------- persistence


def test_load_leaderboard_validates_and_sorts(tmp_path):
    save_leaderboard("oval", {
        "entries": [{"name": "slow", "lap_s": 20.0}, {"name": "fast", "lap_s": 10.0},
                    {"name": "", "lap_s": 5.0}],
        "references": {"quantum": {"driver": "oval-trained", "lap_s": 12.0}},
    }, tmp_path)
    board = load_leaderboard("oval", tmp_path)
    assert [e["name"] for e in board["entries"]] == ["fast", "slow"]  # nameless dropped
    assert board["references"]["quantum"]["lap_s"] == 12.0
    assert load_leaderboard("nope", tmp_path) == {"entries": [], "references": {}}


def test_set_name_protocol_validation():
    assert P.parse_client({"type": "set_name", "name": "Jan"}).name == "Jan"
    with pytest.raises(P.ProtocolError):
        P.parse_client({"type": "set_name", "name": "x" * 25})
    with pytest.raises(P.ProtocolError):
        P.parse_client({"type": "set_name"})
