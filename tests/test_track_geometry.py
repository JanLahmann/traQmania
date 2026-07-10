"""Tests for track loading, validation and geometric queries on all tracks."""

import numpy as np
import pytest

from traqmania.env.track import Track

TRACK_NAMES = ["oval", "chicane", "gp"]


@pytest.fixture(params=TRACK_NAMES)
def track(request):
    return Track.load(request.param)


def _straightest_index(track):
    """Centerline index where the local tangent changes least (on a straight)."""
    turn = np.linalg.norm(np.roll(track.tangents, -1, axis=0) - track.tangents, axis=1)
    return int(np.argmin(turn + np.roll(turn, 1) + np.roll(turn, -1)))


def test_load_and_validate(track):
    assert track.total_length > 0
    assert len(track.centerline) >= 8
    assert track.half_width >= 3.0
    cps = np.asarray(track.checkpoints)
    assert np.all((cps >= 0) & (cps < 1)) and np.all(np.diff(cps) > 0)
    x, y, heading = track.start_pose()
    assert np.hypot(*(track.centerline[0] - (x, y))) < 1e-9
    assert np.isfinite(heading)


def test_project_on_centerline(track):
    # Segment midpoints lie exactly on the centerline polyline.
    mids = 0.5 * (track.centerline + np.roll(track.centerline, -1, axis=0))
    s_vals, lateral = track.project(mids)
    assert np.all(np.abs(lateral) < 1e-6)
    # s increases monotonically along the loop.
    assert np.all(np.diff(s_vals) > 0)


def test_project_lateral_offset(track):
    idx = np.arange(0, len(track.centerline), 5)
    pts = track.centerline[idx] + 2.0 * track.normals[idx]
    _, lateral = track.project(pts)
    assert np.all(np.abs(lateral - 2.0) < 0.15)


def test_is_inside(track):
    hw = track.half_width
    center = track.centerline[::7]
    normals = track.normals[::7]
    assert np.all(track.is_inside(center))
    assert np.all(track.is_inside(center + (hw - 0.5) * normals))
    assert not np.any(track.is_inside(center + (hw + 1.0) * normals))
    assert not np.any(track.is_inside(center - (hw + 1.0) * normals))


def test_raycast_hits_near_wall(track):
    i = _straightest_index(track)
    origin = track.centerline[i][None, :]
    for normal in (track.normals[i], -track.normals[i]):
        angle = np.array([np.arctan2(normal[1], normal[0])])
        dist = track.raycast(origin, angle, max_dist=30.0)
        assert dist[0] == pytest.approx(track.half_width, abs=0.3)


def test_raycast_capped_along_straight(track):
    i = _straightest_index(track)
    origin = track.centerline[i][None, :]
    tangent = track.tangents[i]
    angle = np.array([np.arctan2(tangent[1], tangent[0])])
    max_dist = 6.0
    dist = track.raycast(origin, angle, max_dist=max_dist)
    assert dist[0] == pytest.approx(max_dist)


def test_validation_rejects_narrow_track(tmp_path):
    import json

    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    circle = np.stack([30 * np.cos(theta), 30 * np.sin(theta)], axis=1)
    bad = {
        "name": "narrow",
        "id": "narrow",
        "centerline": circle.tolist(),
        "half_width": 2.0,
        "checkpoints": [0.0, 0.5],
        "theme": {"surface": "asphalt", "edge": "kerb"},
    }
    path = tmp_path / "narrow.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="half_width"):
        Track.load(str(path))


def test_validation_rejects_tight_corner(tmp_path):
    import json

    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    circle = np.stack([4 * np.cos(theta), 4 * np.sin(theta)], axis=1)
    bad = {
        "name": "tight",
        "id": "tight",
        "centerline": circle.tolist(),
        "half_width": 3.5,
        "checkpoints": [0.0, 0.5],
        "theme": {"surface": "asphalt", "edge": "kerb"},
    }
    path = tmp_path / "tight.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="radius"):
        Track.load(str(path))


def test_validation_rejects_bad_checkpoints(tmp_path):
    import json

    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    circle = np.stack([30 * np.cos(theta), 30 * np.sin(theta)], axis=1)
    bad = {
        "name": "cps",
        "id": "cps",
        "centerline": circle.tolist(),
        "half_width": 5.0,
        "checkpoints": [0.0, 0.5, 0.25],
        "theme": {"surface": "asphalt", "edge": "kerb"},
    }
    path = tmp_path / "cps.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="checkpoints"):
        Track.load(str(path))


def test_validation_rejects_open_loop(tmp_path):
    import json

    theta = np.linspace(0, np.pi, 100)  # half circle: endpoints 60 units apart
    arc = np.stack([30 * np.cos(theta), 30 * np.sin(theta)], axis=1)
    bad = {
        "name": "open",
        "id": "open",
        "centerline": arc.tolist(),
        "half_width": 5.0,
        "checkpoints": [0.0, 0.5],
        "theme": {"surface": "asphalt", "edge": "kerb"},
    }
    path = tmp_path / "open.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="closed"):
        Track.load(str(path))
