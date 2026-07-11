"""Draw-a-track: the draw_track protocol message, the drawing -> Track
builder (rescale, smooth, validate, reject with user-facing reasons), and the
session handler that installs the drawn track with random-track fallbacks."""

import numpy as np
import pytest

from traqmania.config import load_config
from traqmania.env.trackgen import DRAWN_HALF_WIDTH, track_from_drawing
from traqmania.server import protocol as P
from traqmania.server.session import DemoSession


def stroke_circle(n=120, r=40.0, gap=4, noise=0.8, seed=3):
    """A hand-drawn-ish loop: wobbly circle with pointer jitter and a small
    closing gap, like a real mouse stroke."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = r + 6 * np.sin(3 * t) + rng.normal(0, noise, n)
    pts = np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1)
    return pts[: len(pts) - gap].tolist()


# ------------------------------------------------------------------- protocol


def test_parse_draw_track_round_trip():
    pts = stroke_circle()
    msg = P.parse_client({"type": "draw_track", "points": pts})
    assert isinstance(msg, P.DrawTrack)
    assert len(msg.points) == len(pts)


@pytest.mark.parametrize("bad", [
    {"type": "draw_track"},                                  # missing points
    {"type": "draw_track", "points": [[0, 0]] * 4},          # too short
    {"type": "draw_track", "points": [[0, 0, 0]] * 20},      # not [x, y]
    {"type": "draw_track", "points": [[0, "a"]] * 20},       # not numbers
    {"type": "draw_track", "points": [[0, 0]] * 20, "x": 1}, # extra field
])
def test_parse_draw_track_rejects_garbage(bad):
    with pytest.raises(P.ProtocolError):
        P.parse_client(bad)


# -------------------------------------------------------------------- builder


def test_drawing_builds_valid_track():
    track = track_from_drawing(stroke_circle(), name="drawn #1")
    assert track.name == "drawn #1"
    assert track.half_width <= DRAWN_HALF_WIDTH
    assert 200.0 <= track.total_length <= 700.0
    assert track.checkpoints == [0.0, 0.25, 0.5, 0.75]


def test_tiny_drawing_is_scaled_up_to_drivable_size():
    small = (np.asarray(stroke_circle()) * 0.05).tolist()
    assert track_from_drawing(small).total_length >= 200.0


def test_sharp_corners_are_smoothed_not_rejected():
    square = []
    corners = [(-30, -30), (30, -30), (30, 30), (-30, 30)]
    for i, a in enumerate(corners):
        b = corners[(i + 1) % 4]
        for u in np.linspace(0, 1, 30, endpoint=False):
            square.append([a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u])
    track = track_from_drawing(square)
    assert track.total_length > 200.0


def test_open_stroke_rejected():
    with pytest.raises(ValueError, match="finish the loop"):
        track_from_drawing(stroke_circle()[:40])


def test_self_crossing_rejected():
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    fig8 = np.stack([40 * np.sin(t), 25 * np.sin(2 * t)], axis=1).tolist()
    with pytest.raises(ValueError, match="overlaps itself"):
        track_from_drawing(fig8)


# -------------------------------------------------------------------- session


def test_session_installs_drawn_track(tmp_path):
    session = DemoSession(load_config(), ghosts_dir=tmp_path)
    session.drain_outbox()
    session.handle_message(P.parse_client({"type": "draw_track",
                                           "points": stroke_circle()}))
    msgs = session.drain_outbox()
    tracks = [m for m in msgs if m["type"] == "track"]
    assert tracks and tracks[0]["track"]["name"] == "drawn #1"
    assert session.track_name == "drawn #1"
    assert session.track_is_random  # generated-track fallbacks (ghosts, weights)
    assert session._ghost is None
    # a second drawing gets a fresh number
    session.handle_message(P.parse_client({"type": "draw_track",
                                           "points": stroke_circle(seed=5)}))
    assert session.track_name == "drawn #2"


def test_session_reports_bad_drawing_as_error(tmp_path):
    session = DemoSession(load_config(), ghosts_dir=tmp_path)
    session.drain_outbox()
    t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    fig8 = np.stack([40 * np.sin(t), 25 * np.sin(2 * t)], axis=1).tolist()
    session.handle_message(P.parse_client({"type": "draw_track", "points": fig8}))
    msgs = session.drain_outbox()
    errors = [m for m in msgs if m["type"] == "error"]
    assert errors and "overlaps itself" in errors[0]["message"]
    assert session.track_name == "oval"  # unchanged
