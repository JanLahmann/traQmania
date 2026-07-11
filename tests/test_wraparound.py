"""Arc-length wraparound behaviour of Track.project near the start line."""

import numpy as np
import pytest

from traqmania.env.track import Track

TRACK_NAMES = ["oval", "chicane", "gp", "combo"]


@pytest.fixture(params=TRACK_NAMES)
def track(request):
    return Track.load(request.param)


def test_s_small_just_past_start(track):
    p = track.centerline[0] + 0.3 * track.tangents[0]
    s_vals, lateral = track.project(p[None, :])
    assert 0.0 <= s_vals[0] < 2.0  # small, NOT ~total_length
    assert abs(lateral[0]) < 0.1


def test_s_wraps_just_before_start(track):
    p = track.centerline[0] - 0.3 * track.tangents[0]
    s_vals, _ = track.project(p[None, :])
    assert track.total_length - 2.0 < s_vals[0] < track.total_length


def test_s_in_range_everywhere(track):
    rng = np.random.default_rng(3)
    idx = rng.integers(0, len(track.centerline), size=64)
    offsets = rng.uniform(-track.half_width, track.half_width, size=64)
    pts = track.centerline[idx] + offsets[:, None] * track.normals[idx]
    s_vals, _ = track.project(pts)
    assert np.all((s_vals >= 0.0) & (s_vals < track.total_length))
