"""Track geometry for traQmania: closed-loop centerline with arc-length
parameterization, signed lateral projection, inside/outside tests and lidar
raycasts against the track boundaries.

A track is loaded from JSON (see ``traqmania/env/tracks/``), its centerline is
resampled to ~uniform arc-length spacing, and unit tangents/normals plus
left/right boundary polylines are precomputed.  Two uniform spatial hash grids
(cell size ~= 2 * half_width) accelerate the hot-path queries: one maps cells
to nearby centerline segments for O(1) ``project()``, the other maps cells to
boundary segments so ``raycast()`` only tests segments near each ray.  All
queries are vectorized over the batch axis.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

TRACKS_DIR = Path(__file__).resolve().parent / "tracks"

MIN_CORNER_RADIUS = 6.0
MIN_HALF_WIDTH = 3.0


def _cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """z-component of the 2D cross product, broadcasting over leading axes."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class _SegmentGrid:
    """Uniform spatial hash mapping grid cells to nearby segment indices.

    Each segment is inserted into every cell overlapping its axis-aligned
    bounding box inflated by ``inflate``.  Lookups return a fixed-width padded
    index table (pad value -1); out-of-grid points map to an empty sentinel
    row, so lookups are pure numpy gathers.
    """

    def __init__(self, seg_a: np.ndarray, seg_b: np.ndarray, cell: float, inflate: float = 0.0):
        self.cell = float(cell)
        lo = np.minimum(seg_a, seg_b).min(axis=0) - inflate - 1e-9
        hi = np.maximum(seg_a, seg_b).max(axis=0) + inflate + 1e-9
        self.origin = lo
        dims = np.ceil((hi - lo) / self.cell).astype(int) + 1
        self.nx, self.ny = int(dims[0]), int(dims[1])
        n_cells = self.nx * self.ny
        self.sentinel = n_cells  # empty row for out-of-grid lookups

        buckets: list[list[int]] = [[] for _ in range(n_cells)]
        lo_seg = np.minimum(seg_a, seg_b) - inflate
        hi_seg = np.maximum(seg_a, seg_b) + inflate
        i0 = np.clip(np.floor((lo_seg - lo) / self.cell).astype(int), 0, dims - 1)
        i1 = np.clip(np.floor((hi_seg - lo) / self.cell).astype(int), 0, dims - 1)
        for k in range(len(seg_a)):  # load time only, not a hot path
            for ix in range(i0[k, 0], i1[k, 0] + 1):
                for iy in range(i0[k, 1], i1[k, 1] + 1):
                    buckets[ix * self.ny + iy].append(k)
        width = max((len(b) for b in buckets), default=1) or 1
        table = np.full((n_cells + 1, width), -1, dtype=np.int64)
        for c, b in enumerate(buckets):
            table[c, : len(b)] = b
        self.table = table

    def cell_ids(self, points: np.ndarray) -> np.ndarray:
        """(..., 2) points -> (...,) flat cell ids (sentinel when off-grid)."""
        ij = np.floor((points - self.origin) / self.cell).astype(np.int64)
        ix, iy = ij[..., 0], ij[..., 1]
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return np.where(ok, ix * self.ny + iy, self.sentinel)

    def cell_ids_3x3(self, points: np.ndarray) -> np.ndarray:
        """(..., 2) points -> (..., 9) flat ids of each point's 3x3 cell block."""
        ij = np.floor((points - self.origin) / self.cell).astype(np.int64)
        offs = np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)], dtype=np.int64)
        ij9 = ij[..., None, :] + offs  # (..., 9, 2)
        ix, iy = ij9[..., 0], ij9[..., 1]
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return np.where(ok, ix * self.ny + iy, self.sentinel)


class Track:
    """Closed-loop race track with vectorized geometric queries.

    Attributes after :meth:`load`: ``name`` (str), ``half_width`` (float),
    ``centerline`` (N,2) resampled to ~uniform spacing, ``s`` (N,) cumulative
    arc length per point, ``total_length`` (float), ``checkpoints`` (list of
    fractions in [0,1)), ``tangents``/``normals`` (N,2) unit vectors (normal
    points left of the direction of travel), ``curvature`` (N,) unsigned
    |kappa| per point and its maximum ``max_abs_curvature``.
    """

    def __init__(self, name: str, centerline, half_width: float, checkpoints,
                 resample_spacing: float = 1.5):
        self.name = str(name)
        self.half_width = float(half_width)
        self.checkpoints = [float(c) for c in checkpoints]
        self._validate_scalars()
        self._resample(np.asarray(centerline, dtype=np.float64), float(resample_spacing))
        self._validate_curvature()
        self._precompute()

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, name_or_path: str, resample_spacing: float = 1.5) -> Track:
        """Load a track JSON by bare name (resolved against
        ``traqmania/env/tracks/<name>.json``) or by explicit path.

        Raises ``ValueError`` if the track fails validation: loop not closed
        (a gap between the raw endpoints of up to 25% of the loop length is
        auto-closed with a straight segment; a larger gap is rejected), min
        corner radius < 6.0, half_width < 3.0, or checkpoints not strictly
        increasing in [0, 1).
        """
        path = Path(name_or_path)
        if not path.suffix:
            path = TRACKS_DIR / f"{name_or_path}.json"
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            name=data.get("name", path.stem),
            centerline=data["centerline"],
            half_width=data["half_width"],
            checkpoints=data.get("checkpoints", []),
            resample_spacing=resample_spacing,
        )

    # ------------------------------------------------------- validation/build

    def _validate_scalars(self) -> None:
        if self.half_width < MIN_HALF_WIDTH:
            raise ValueError(
                f"track '{self.name}': half_width {self.half_width:.2f} < {MIN_HALF_WIDTH}"
            )
        cps = np.asarray(self.checkpoints, dtype=np.float64)
        if cps.size and (
            np.any(cps < 0.0) or np.any(cps >= 1.0) or np.any(np.diff(cps) <= 0.0)
        ):
            raise ValueError(
                f"track '{self.name}': checkpoints must be strictly increasing in [0, 1), "
                f"got {self.checkpoints}"
            )

    def _resample(self, pts: np.ndarray, spacing: float) -> None:
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            raise ValueError(f"track '{self.name}': centerline must be an (N>=3, 2) point list")
        # Drop a duplicated closing point, then close the loop with a final
        # segment back to the start (auto-close, documented in load()).
        if np.linalg.norm(pts[0] - pts[-1]) < 1e-9:
            pts = pts[:-1]
        closed = np.vstack([pts, pts[:1]])
        seg_len = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        total = float(seg_len.sum())
        gap = float(np.linalg.norm(pts[0] - pts[-1]))
        if total <= 0.0 or gap > 0.25 * total:
            raise ValueError(
                f"track '{self.name}': centerline loop is not closed "
                f"(endpoint gap {gap:.1f} of loop length {total:.1f})"
            )
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        n = max(int(round(total / spacing)), 8)
        s_grid = np.arange(n) * (total / n)
        self.centerline = np.stack(
            [np.interp(s_grid, cum, closed[:, 0]), np.interp(s_grid, cum, closed[:, 1])], axis=1
        )
        self.s = s_grid
        self.total_length = total

    def _validate_curvature(self) -> None:
        p0 = self.centerline
        p1 = np.roll(p0, -1, axis=0)
        p2 = np.roll(p0, -2, axis=0)
        a = np.linalg.norm(p1 - p0, axis=1)
        b = np.linalg.norm(p2 - p1, axis=1)
        c = np.linalg.norm(p2 - p0, axis=1)
        area2 = np.abs(_cross2(p1 - p0, p2 - p0))  # twice the triangle area
        with np.errstate(divide="ignore"):
            radius = np.where(area2 > 1e-12, a * b * c / (2.0 * area2), np.inf)
        min_radius = float(radius.min())
        if min_radius < MIN_CORNER_RADIUS:
            raise ValueError(
                f"track '{self.name}': min corner radius {min_radius:.2f} "
                f"< {MIN_CORNER_RADIUS} units"
            )

    def _precompute(self) -> None:
        cl = self.centerline
        # Unit tangents (central difference over the closed loop) and left normals.
        diff = np.roll(cl, -1, axis=0) - np.roll(cl, 1, axis=0)
        self.tangents = diff / np.linalg.norm(diff, axis=1, keepdims=True)
        self.normals = np.stack([-self.tangents[:, 1], self.tangents[:, 0]], axis=1)

        # Unsigned curvature |kappa| per centerline point: three-point
        # circumradius formula (same as _validate_curvature), attributed to
        # the middle point of each consecutive triple.
        p1 = np.roll(cl, -1, axis=0)
        p2 = np.roll(cl, -2, axis=0)
        a = np.linalg.norm(p1 - cl, axis=1)
        b = np.linalg.norm(p2 - p1, axis=1)
        c = np.linalg.norm(p2 - cl, axis=1)
        area2 = np.abs(_cross2(p1 - cl, p2 - cl))  # twice the triangle area
        kappa = np.where(area2 > 1e-12, 2.0 * area2 / (a * b * c), 0.0)
        self.curvature = np.roll(kappa, 1)
        self.max_abs_curvature = float(self.curvature.max())

        # Centerline segments i -> (i+1) mod N.
        self._seg_a = cl
        self._seg_d = np.roll(cl, -1, axis=0) - cl
        self._seg_len = np.linalg.norm(self._seg_d, axis=1)
        self._seg_len2 = np.maximum(self._seg_len**2, 1e-12)
        self._seg_unit = self._seg_d / self._seg_len[:, None]

        # Boundary polylines (centerline +/- normal * half_width) and segments.
        left = cl + self.normals * self.half_width
        right = cl - self.normals * self.half_width
        self._bnd_a = np.vstack([left, right])
        self._bnd_d = np.vstack([np.roll(left, -1, axis=0) - left,
                                 np.roll(right, -1, axis=0) - right])

        cell = 2.0 * self.half_width
        self._proj_reach = 2.0 * self.half_width
        self._cl_grid = _SegmentGrid(cl, np.roll(cl, -1, axis=0), cell,
                                     inflate=self._proj_reach)
        self._bnd_grid = _SegmentGrid(self._bnd_a, self._bnd_a + self._bnd_d, cell)

    # ---------------------------------------------------------------- queries

    def start_pose(self) -> tuple[float, float, float]:
        """(x, y, heading) at s = 0, heading along the track tangent."""
        x, y = self.centerline[0]
        heading = math.atan2(self.tangents[0, 1], self.tangents[0, 0])
        return float(x), float(y), heading

    def curvature_ahead(self, s_vals, lookahead: float) -> np.ndarray:
        """(B,) arc lengths -> (B,) max |kappa| of the centerline over the
        arc-length window [s, s + lookahead] ahead of each query (wraps around
        the loop; sampled at the ~uniform resampled points)."""
        s = np.atleast_1d(np.asarray(s_vals, dtype=np.float64))
        n = len(self.centerline)
        ds = self.total_length / n
        first = np.floor(s / ds).astype(np.int64)
        count = int(math.ceil(float(lookahead) / ds)) + 1
        idx = (first[:, None] + np.arange(count)[None, :]) % n
        return self.curvature[idx].max(axis=1)

    def tangent_angle(self, s_vals) -> np.ndarray:
        """(B,) arc lengths -> (B,) heading (radians, in (-pi, pi]) of the
        centerline tangent at the nearest resampled point."""
        s = np.atleast_1d(np.asarray(s_vals, dtype=np.float64))
        n = len(self.centerline)
        ds = self.total_length / n
        idx = np.floor(s / ds + 0.5).astype(np.int64) % n
        t = self.tangents[idx]
        return np.arctan2(t[:, 1], t[:, 0])

    def _project_onto(self, p: np.ndarray, cand: np.ndarray):
        """Project points (B,2) onto candidate segments (B,K) (pad = -1).

        Returns (s_vals, lateral, dist) each (B,); dist is +inf where no
        candidate was valid.
        """
        idx = np.maximum(cand, 0)
        a = self._seg_a[idx]                      # (B,K,2)
        d = self._seg_d[idx]
        ap = p[:, None, :] - a
        t = np.clip(np.sum(ap * d, axis=-1) / self._seg_len2[idx], 0.0, 1.0)
        closest = a + t[..., None] * d
        delta = p[:, None, :] - closest
        dist2 = np.sum(delta * delta, axis=-1)
        dist2 = np.where(cand >= 0, dist2, np.inf)
        j = np.argmin(dist2, axis=1)
        rows = np.arange(len(p))
        seg = idx[rows, j]
        tb = t[rows, j]
        s_vals = (self.s[seg] + tb * self._seg_len[seg]) % self.total_length
        lateral = _cross2(self._seg_unit[seg], p - closest[rows, j])
        return s_vals, lateral, np.sqrt(dist2[rows, j])

    def project(self, points):
        """(B,2) points -> (s_vals (B,), lateral (B,)).

        ``s_vals`` is the arc length of the nearest centerline point (wraps
        modulo ``total_length``); ``lateral`` is the signed offset, positive
        to the left of the direction of travel.
        """
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        cand = self._cl_grid.table[self._cl_grid.cell_ids(p)]
        s_vals, lateral, dist = self._project_onto(p, cand)
        # The grid is only guaranteed within _proj_reach of the centerline;
        # fall back to a full search for far-away points.
        far = ~(dist <= self._proj_reach)
        if np.any(far):
            all_segs = np.broadcast_to(np.arange(len(self._seg_a)),
                                       (int(far.sum()), len(self._seg_a)))
            s_far, lat_far, _ = self._project_onto(p[far], all_segs)
            s_vals = s_vals.copy()
            lateral = lateral.copy()
            s_vals[far] = s_far
            lateral[far] = lat_far
        return s_vals, lateral

    def is_inside(self, points):
        """(B,2) -> bool (B,): whether |lateral| <= half_width."""
        _, lateral = self.project(points)
        return np.abs(lateral) <= self.half_width

    def raycast(self, origins, angles, max_dist):
        """March rays against both boundary polylines.

        origins (B,2), angles (B,) radians, max_dist float -> (B,) distance
        to the first boundary crossing, capped at max_dist.  Candidate
        segments are gathered from the boundary grid along each ray (3x3 cell
        blocks around samples spaced one cell apart, which conservatively
        covers every cell the ray passes through).
        """
        o = np.atleast_2d(np.asarray(origins, dtype=np.float64))
        ang = np.atleast_1d(np.asarray(angles, dtype=np.float64))
        d = np.stack([np.cos(ang), np.sin(ang)], axis=-1)  # (B,2)

        cell = self._bnd_grid.cell
        n_steps = int(math.ceil(max_dist / cell)) + 1
        ts = np.arange(n_steps + 1) * cell
        samples = o[:, None, :] + ts[None, :, None] * d[:, None, :]      # (B,S,2)
        ids = self._bnd_grid.cell_ids_3x3(samples)                       # (B,S,9)
        cand = self._bnd_grid.table[ids].reshape(len(o), -1)             # (B,C)

        idx = np.maximum(cand, 0)
        a = self._bnd_a[idx]                                             # (B,C,2)
        e = self._bnd_d[idx]
        ao = a - o[:, None, :]
        denom = _cross2(np.broadcast_to(d[:, None, :], e.shape), e)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = _cross2(ao, e) / denom
            u = _cross2(ao, np.broadcast_to(d[:, None, :], ao.shape)) / denom
        valid = (
            (cand >= 0)
            & (np.abs(denom) > 1e-12)
            & (t > 1e-9)
            & (u >= -1e-9)
            & (u <= 1.0 + 1e-9)
        )
        t = np.where(valid, t, np.inf)
        return np.minimum(t.min(axis=1), max_dist)
