"""3D voxel traversal (Amanatides-Woo) for the global cell-traversal layer.

The global algorithm is::

    3D DDA over voxels  +  analytic per-cell optical-depth accumulation  +  <= 1 inversion

This module supplies the traversal.  A grid voxel is the axis-aligned unit cell
[i, i+1) x [j, j+1) x [k, k+1) (grid spacing 1; scale the ray to change resolution).
``traverse`` yields, for each visited voxel, the integer index and the (t_enter, t_exit)
sub-interval, so that adjacent exits/enters match exactly (partition-of-unity of the
ray parameter) -- the property the boundary test suite checks.

Tie-breaking: when the ray crosses a voxel corner/edge (two or three t_max components
coincide), all tied axes step together in a single event, so the traversal never emits
a spurious zero-length voxel.  Rays exactly parallel to an axis (w_i == 0) never step in
that axis.
"""
from __future__ import annotations

import math
from typing import Iterator, List, Tuple

import numpy as np

Voxel = Tuple[int, int, int]
Segment = Tuple[Voxel, float, float]


def _voxel_index(x: float, eps: float = 0.0) -> int:
    return int(math.floor(x + eps))


def traverse(p, w, t_start: float, t_end: float,
             eps: float = 1e-12) -> List[Segment]:
    """Return [(voxel, t_enter, t_exit), ...] for the ray p + t w over [t_start, t_end].

    Assumes t_end > t_start.  The ray need not start inside the grid; traversal begins at
    the voxel containing p + t_start w.
    """
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)
    if t_end <= t_start:
        return []

    entry = p + t_start * w
    voxel = [_voxel_index(entry[0]), _voxel_index(entry[1]), _voxel_index(entry[2])]

    step = [0, 0, 0]
    t_max = [math.inf, math.inf, math.inf]
    t_delta = [math.inf, math.inf, math.inf]

    for a in range(3):
        if w[a] > 0.0:
            step[a] = 1
            next_boundary = voxel[a] + 1
            t_max[a] = t_start + (next_boundary - entry[a]) / w[a]
            t_delta[a] = 1.0 / w[a]
        elif w[a] < 0.0:
            step[a] = -1
            next_boundary = voxel[a]
            t_max[a] = t_start + (next_boundary - entry[a]) / w[a]
            t_delta[a] = -1.0 / w[a]
        # w[a] == 0 -> never crosses a boundary on this axis

    segments: List[Segment] = []
    t_cur = t_start
    # guard against pathological infinite loops
    max_steps = 3 * (int(abs(t_end - t_start) * (abs(w).max() + 1.0)) + 8) + 64
    for _ in range(max_steps):
        t_next = min(t_max)
        seg_exit = min(t_next, t_end)
        if seg_exit > t_cur + 0.0:
            segments.append(((voxel[0], voxel[1], voxel[2]), t_cur, seg_exit))
        if t_next >= t_end:
            break
        # step every axis whose t_max ties the minimum (corner/edge crossing)
        for a in range(3):
            if t_max[a] <= t_next + eps * (1.0 + abs(t_next)):
                voxel[a] += step[a]
                t_max[a] += t_delta[a]
        t_cur = t_next
    return segments


def iter_traverse(p, w, t_start: float, t_end: float, eps: float = 1e-12
                  ) -> Iterator[Segment]:
    """Generator form of :func:`traverse`."""
    for seg in traverse(p, w, t_start, t_end, eps):
        yield seg
