"""Procedural track generation for traQmania.

``generate_track(seed)`` builds a random closed-loop :class:`~traqmania.env.track.Track`
that is deterministic per ``(seed, difficulty)`` and guaranteed to pass the
same load-time validation as the bundled JSON tracks (it is constructed via
the exact code path ``Track.load`` uses after parsing: resample + validation
+ precompute).

The centerline is a star-shaped polar curve around the origin: gentle random
harmonics ``sum_k a_k cos(k theta + phi_k)`` give the overall layout, and a
handful of localized Gaussian radius features carve distinct corners into it —
inward dents read as hairpins, outward bumps as sweepers, and a dent next to a
bump as a chicane.  A random ellipse stretch elongates the layout so tracks
are not all round.  ``difficulty`` in [0, 1] steers how tight the track is:
the whole shape is rescaled so the minimum corner radius lands near a
difficulty-dependent target (oval-like at 0 tightening to just above the
validator's floor at 1, always above the car's kinematic minimum turn
radius), and the track narrows from wide to gp-like.  Candidates whose track
surfaces would pinch into each other (a dent slot narrower than the track is
wide) are rejected, as is anything failing ``Track``'s own load-time
validation; the generator retries with fresh rng-derived jitter.
"""

from __future__ import annotations

import numpy as np

from traqmania.env.track import Track

# Car steering kinematics at the default [physics] config (default.toml):
# minimum turn radius = v_turn / (2 * k_steer) ~= 1.73 units.
_V_TURN = 9.0
_K_STEER = 2.6
CAR_MIN_TURN_RADIUS = _V_TURN / (2.0 * _K_STEER)

# Same convention as the bundled JSON tracks (oval/chicane/gp).
CHECKPOINTS = (0.0, 0.25, 0.5, 0.75)

MAX_ATTEMPTS = 50

_N_RAW_POINTS = 512  # dense polar samples handed to Track (it resamples anyway)


def _target_min_radius(difficulty: float) -> float:
    """Minimum-corner-radius target: 12 units at difficulty 0 tightening to
    6.6 at 1, always above the validator's 6.0-unit floor and far above the
    car's ~1.73-unit kinematic minimum turn radius (the bundled gp is ~9, so
    difficulty 0.5 lands right at gp-like corners)."""
    kinematic = (3.0 - 1.4 * difficulty) * CAR_MIN_TURN_RADIUS
    drivable = 12.0 - 5.4 * difficulty
    return max(kinematic, drivable)


def _min_self_clearance(pts: np.ndarray, min_arc: float) -> float:
    """Minimum euclidean distance between any two points of the closed
    polyline that are more than ``min_arc`` apart along the loop — small
    values mean two *different* parts of the track run so close that their
    surfaces would overlap.  Neighbours within ``min_arc`` are excluded
    because points along one smooth wall (or around one valid-radius apex)
    are legitimately close."""
    seg = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    total = float(seg.sum())
    arc = np.abs(s[:, None] - s[None, :])
    arc = np.minimum(arc, total - arc)
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
    d2[arc <= min_arc] = np.inf
    return float(np.sqrt(d2.min()))


def _min_corner_radius(pts: np.ndarray) -> float:
    """Min three-point circumradius over consecutive triples of a closed
    polyline (the same formula ``Track._validate_curvature`` uses)."""
    p0, p1, p2 = pts, np.roll(pts, -1, axis=0), np.roll(pts, -2, axis=0)
    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p2 - p0, axis=1)
    area2 = np.abs((p1 - p0)[:, 0] * (p2 - p0)[:, 1] - (p1 - p0)[:, 1] * (p2 - p0)[:, 0])
    with np.errstate(divide="ignore"):
        radius = np.where(area2 > 1e-12, a * b * c / (2.0 * area2), np.inf)
    return float(radius.min())


def _sample_shape(
    rng: np.random.Generator, difficulty: float,
) -> tuple[np.ndarray | None, float]:
    """One random (centerline points, half_width) candidate; points are None
    when the candidate pinched into itself and the caller should retry.

    The radius profile is gentle global harmonics plus a few localized
    Gaussian features (inward dents = hairpins, outward bumps = sweepers,
    adjacent opposite-sign pairs = chicanes), stretched into an ellipse.  The
    measured polyline radius is then corrected by a uniform upscale so the
    tightest corner lands at (or, when the upscale saturates, near) the
    difficulty target — every track keeps at least one genuinely hard corner.
    """
    target = _target_min_radius(difficulty)
    half_width = max(3.5, 7.0 - 3.5 * difficulty)  # wide -> narrow (gp is 6.0)

    base_radius = rng.uniform(24.0, 34.0)
    theta = np.arange(_N_RAW_POINTS) * (2.0 * np.pi / _N_RAW_POINTS)

    # gentle global harmonics: overall irregularity, not the corners
    k_max = 3 + int(round(3.0 * difficulty))  # highest harmonic: 3 (easy) .. 6 (hard)
    ks = np.arange(2, k_max + 1)
    raw = rng.uniform(0.2, 0.8, size=len(ks)) / ks
    amps = raw * (rng.uniform(0.08, 0.2) / float(raw.sum()))
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(ks))
    wave = (amps[:, None] * np.cos(ks[:, None] * theta[None, :] + phases[:, None])).sum(axis=0)

    # localized corner features: mostly dents (hairpins), some bumps; these
    # are deep on purpose — they, not the loop's global curvature, must be
    # the tight corners, or the track degenerates into a rounded blob
    n_feat = int(rng.integers(3, 6 + int(round(3.0 * difficulty))))
    centers = rng.uniform(0.0, 2.0 * np.pi, size=n_feat)
    widths = rng.uniform(0.15 + 0.1 * (1.0 - difficulty), 0.5, size=n_feat)
    depths = rng.uniform(0.10, 0.25 + 0.20 * difficulty, size=n_feat)
    signs = np.where(rng.random(n_feat) < 0.6, -1.0, 1.0)
    # guaranteed hairpin: the first feature is always a narrow deep dent...
    widths[0] = rng.uniform(0.14, 0.22)
    depths[0] = rng.uniform(0.22, 0.30) + 0.15 * difficulty
    signs[0] = -1.0
    # ...and half the time a bump right beside it turns it into an S-chicane
    if n_feat > 1 and rng.random() < 0.5:
        centers[1] = centers[0] + 1.6 * (widths[0] + widths[1]) * (1 if rng.random() < 0.5 else -1)
        signs[1] = 1.0
    d_theta = np.angle(np.exp(1j * (theta[None, :] - centers[:, None])))  # wrapped
    feat = (
        signs[:, None] * depths[:, None]
        * np.exp(-0.5 * (d_theta / widths[:, None]) ** 2)
    ).sum(axis=0)
    stretch = rng.uniform(1.0, 1.5)  # elongate: not every track is round

    # upscale-only correction: blow the shape up until the tightest corner
    # sits at the difficulty target.  Never downscale — shrinking a big
    # shape until its global curvature hits the target is exactly what turns
    # tracks into small rounded blobs.  A deep dent often has a razor tip the
    # capped upscale can't fix; rather than burning a retry, soften all
    # feature depths a step at a time until the target is reachable.  Only a
    # shape so gentle that it has no corner anywhere near the target (or one
    # that stays razor-sharp even fully softened) is rejected.
    for _ in range(8):
        wave_full = np.maximum(wave + feat, -0.62)  # r >= 0.38 R0: star-shaped
        r = base_radius * (1.0 + wave_full)
        pts = np.stack([stretch * r * np.cos(theta), r * np.sin(theta)], axis=1)
        measured = _min_corner_radius(pts)
        scale = float(np.clip(target / measured, 1.0, 3.0))
        if measured * scale >= 0.8 * target:
            break
        feat = feat * 0.75
    else:
        return None, half_width
    if measured * scale > 1.4 * target:
        return None, half_width
    pts = pts * scale
    # keep laps finishable: the episode cap is 60 s (600 decisions at 10 Hz),
    # so a lap must fit comfortably inside it at typical racing speeds
    perimeter = float(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1).sum())
    if perimeter > 650.0:
        return None, half_width
    # a dent slot narrower than the track surface would fold onto itself
    min_clear = 2.35 * half_width
    if _min_self_clearance(pts, min_arc=1.7 * min_clear) < min_clear:
        return None, half_width
    return pts, half_width


def generate_track(seed: int, resample_spacing: float = 1.5, difficulty: float = 0.5,
                   name: str | None = None) -> Track:
    """Generate a random valid Track, deterministic per ``(seed, difficulty)``.

    ``difficulty`` is clipped to [0, 1]: 0 gives wide tracks with gentle
    corners, 1 gives narrow tracks slightly tighter than the bundled gp.
    The returned track is named ``random-<seed>`` unless ``name`` is given,
    uses the bundled tracks' checkpoint convention, and carries the number of
    generation attempts in ``track.generation_attempts``.  Raises
    ``ValueError`` if no candidate passes validation in ``MAX_ATTEMPTS``.
    """
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    track_name = f"random-{seed}" if name is None else str(name)
    rng = np.random.default_rng(seed)
    last_error: ValueError | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        pts, half_width = _sample_shape(rng, difficulty)
        if pts is None:
            last_error = ValueError("track surface would pinch into itself")
            continue
        try:
            # Same post-JSON construction path as Track.load: resample the
            # centerline, run load-time validation, precompute geometry.
            track = Track(track_name, pts, half_width, CHECKPOINTS, resample_spacing)
        except ValueError as err:
            last_error = err
            continue
        track.generation_attempts = attempt
        return track
    raise ValueError(
        f"generate_track(seed={seed}, difficulty={difficulty}): no valid track "
        f"after {MAX_ATTEMPTS} attempts (last error: {last_error})"
    )
