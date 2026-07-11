"""Author the bundled "combo" track: a chicane plus one hard gp-style turn.

Deterministic straights-and-arcs construction, matching the design language
of the bundled tracks (oval/chicane/gp are stadium-style circuits): a
rounded-rectangle base, a gp-style out-and-back finger on the right whose
180-degree cap is the hard turn (radius ~9.5, gp's is ~9), and a chicane
S-flick on the top straight (like the bundled chicane track's).  The two
closing straight lengths are solved so the loop closes exactly.  Run from
the repo root to (re)generate traqmania/env/tracks/combo.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from traqmania.env.track import Track

HALF_WIDTH = 6.0  # gp-like
STEP = 1.0  # sampling step along the path, world units
OUT = Path(__file__).resolve().parent.parent / "traqmania" / "env" / "tracks" / "combo.json"

# (kind, *args): ("s", length) straight, ("a", radius, degrees) arc where
# positive degrees turn left (CCW) and negative turn right. "BOTTOM"/"LEFT"
# are the closure straights solved below.
LAYOUT = [
    ("s", "BOTTOM"),        # bottom straight, heading +x (start line lives here)
    ("a", 20.0, 90.0),      # bottom-right corner
    ("s", 12.0),            # right side, heading +y
    ("a", 12.0, -90.0),     # into the finger (heading +x)
    ("s", 26.0),            # finger out
    ("a", 9.5, 180.0),      # THE hard gp turn: tight 180 cap
    ("s", 26.0),            # finger back
    ("a", 12.0, -90.0),     # out of the finger (heading +y again)
    ("s", 12.0),            # right side continues
    ("a", 20.0, 90.0),      # top-right corner (heading -x)
    ("s", 30.0),            # top straight, first part
    ("a", 14.0, 45.0),      # chicane: flick out...
    ("a", 14.0, -45.0),
    ("s", 14.0),            # ...short middle...
    ("a", 14.0, -45.0),     # ...flick back
    ("a", 14.0, 45.0),
    ("s", 30.0),            # top straight, second part
    ("a", 20.0, 90.0),      # top-left corner (heading -y)
    ("s", "LEFT"),          # left side down
    ("a", 20.0, 90.0),      # bottom-left corner, closing onto the start
]


def trace(bottom: float, left: float) -> np.ndarray:
    pos = np.zeros(2)
    heading = 0.0  # +x
    pts = []

    def straight(length: float) -> None:
        nonlocal pos
        n = max(2, int(round(length / STEP)))
        d = np.array([np.cos(heading), np.sin(heading)])
        for i in range(1, n + 1):
            pts.append(pos + d * (length * i / n))
        pos = pos + d * length

    def arc(radius: float, degrees: float) -> None:
        nonlocal pos, heading
        ang = np.radians(degrees)
        sign = 1.0 if ang >= 0 else -1.0
        center = pos + radius * np.array(
            [np.cos(heading + sign * np.pi / 2), np.sin(heading + sign * np.pi / 2)]
        )
        start = np.arctan2(pos[1] - center[1], pos[0] - center[0])
        n = max(3, int(round(abs(ang) * radius / STEP)))
        for i in range(1, n + 1):
            a = start + ang * i / n
            pts.append(center + radius * np.array([np.cos(a), np.sin(a)]))
        pos = pts[-1].copy()
        heading += ang

    for seg in LAYOUT:
        if seg[0] == "s":
            straight(bottom if seg[1] == "BOTTOM" else left if seg[1] == "LEFT" else seg[1])
        else:
            arc(seg[1], seg[2])
    return np.array(pts)


def main() -> None:
    # solve the two closure straights: residual with zero-length placeholders
    resid = trace(0.0, 0.0)[-1]  # loop end - start(0,0)
    bottom, left = -resid[0], resid[1]  # bottom adds +x, left adds -y
    assert bottom > 20 and left > 20, (bottom, left)
    pts = trace(bottom, left)[:-1]  # drop the duplicated closing point
    track = Track("combo", pts, HALF_WIDTH, (0.0, 0.25, 0.5, 0.75), 1.5)  # validates
    print(f"combo: length {track.total_length:.1f}, "
          f"min radius {1.0 / track.max_abs_curvature:.2f}, hw {HALF_WIDTH}, "
          f"closure straights bottom={bottom:.1f} left={left:.1f}")
    OUT.write_text(json.dumps({
        "name": "combo",
        "id": "combo",
        "centerline": [[round(float(x), 3), round(float(y), 3)] for x, y in pts],
        "half_width": HALF_WIDTH,
        "checkpoints": [0.0, 0.25, 0.5, 0.75],
        "theme": {"surface": "asphalt", "edge": "kerb-blue"},
    }, indent=None) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
