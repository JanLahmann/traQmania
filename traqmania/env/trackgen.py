"""Procedural track generation for traQmania.

``generate_track(seed)`` builds a random closed-loop :class:`~traqmania.env.track.Track`
that is deterministic per ``(seed, difficulty)`` and guaranteed to pass the
same load-time validation as the bundled JSON tracks (it is constructed via
the exact code path ``Track.load`` uses after parsing: resample + validation
+ precompute).

The centerline is a star-shaped polar curve around the origin,
``r(theta) = R0 * (1 + sum_k a_k cos(k theta + phi_k))``, with random harmonic
amplitudes/phases.  ``difficulty`` in [0, 1] steers how tight the track is:
the amplitudes are normalized so the minimum corner radius lands near a
difficulty-dependent target (gentle, oval-like corners at 0 tightening to
slightly under the bundled gp track at 1, always comfortably above the
validator's floor and the car's kinematic minimum turn radius), and the track
narrows from wide to gp-like.  If an attempt still fails validation the
generator retries with fresh rng-derived jitter.
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
    """Minimum-corner-radius target: ~3x the car's kinematic minimum turn
    radius at difficulty 0 tightening toward ~1.6x at 1, but never below a
    drivability floor that keeps the track above the validator's 6.0-unit
    minimum with margin (15 units at 0 -> 7.5 at 1; the bundled gp is ~9)."""
    kinematic = (3.0 - 1.4 * difficulty) * CAR_MIN_TURN_RADIUS
    drivable = 15.0 - 7.5 * difficulty
    return max(kinematic, drivable)


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


def _sample_shape(rng: np.random.Generator, difficulty: float) -> tuple[np.ndarray, float]:
    """One random (centerline points, half_width) candidate.

    Harmonic amplitudes are normalized so the small-amplitude curvature
    estimate ``kappa ~= (1 + sum a_k (k^2 - 1)) / R0`` lands near the target
    minimum radius, then the measured polyline radius is corrected by a
    uniform upscale if the higher-order terms made a corner too tight.
    """
    target = _target_min_radius(difficulty)
    half_width = max(3.5, 7.0 - 3.5 * difficulty)  # wide -> narrow (gp is 6.0)

    base_radius = rng.uniform(32.0, 42.0)
    k_max = 3 + int(round(3.0 * difficulty))  # highest harmonic: 3 (easy) .. 6 (hard)
    ks = np.arange(2, k_max + 1)
    raw = rng.uniform(0.35, 1.0, size=len(ks)) / ks**1.1
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(ks))

    perturbation = max(base_radius / (1.15 * target) - 1.0, 0.05)
    amps = raw * (perturbation / float(np.sum(raw * (ks**2 - 1))))
    total = float(amps.sum())
    if total > 0.5:  # keep r(theta) >= R0 / 2: star-shaped, no self-intersection
        amps *= 0.5 / total

    theta = np.arange(_N_RAW_POINTS) * (2.0 * np.pi / _N_RAW_POINTS)
    wave = (amps[:, None] * np.cos(ks[:, None] * theta[None, :] + phases[:, None])).sum(axis=0)
    r = base_radius * (1.0 + wave)
    pts = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    measured = _min_corner_radius(pts)
    if measured < target:
        pts = pts * min(target / measured, 3.0)
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
