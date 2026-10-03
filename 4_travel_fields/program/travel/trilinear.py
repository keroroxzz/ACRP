"""Trilinear density cell and its restriction to a ray.

A cell is the unit voxel [0,1]^3 with 8 corner densities ``v[i,j,k]``.  Trilinear
interpolation gives

    V(x,y,z) = c000 + c100 x + c010 y + c001 z
             + c110 xy + c101 xz + c011 yz + c111 xyz .

Restricting to a ray r(t) = p + t*w yields a cubic in t (Cellwise Polynomial Travel
theorem, degree d = 3), whose antiderivative -- the optical depth -- is a quartic
(deg A = d + 1 = 4).

Coordinates are expressed in the cell's local frame, so callers must map world
positions into [0,1]^3 before using these routines.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

# Order of the 8 packed coefficients.
COEF_NAMES = ("c000", "c100", "c010", "c001", "c110", "c101", "c011", "c111")


def trilinear_coefs(v: np.ndarray) -> np.ndarray:
    """Pack 8 corner densities ``v[i,j,k]`` into monomial coefficients.

    ``v`` has shape (2,2,2) indexed as v[x][y][z].
    """
    v = np.asarray(v, dtype=float)
    c000 = v[0, 0, 0]
    c100 = v[1, 0, 0] - v[0, 0, 0]
    c010 = v[0, 1, 0] - v[0, 0, 0]
    c001 = v[0, 0, 1] - v[0, 0, 0]
    c110 = v[1, 1, 0] - v[1, 0, 0] - v[0, 1, 0] + v[0, 0, 0]
    c101 = v[1, 0, 1] - v[1, 0, 0] - v[0, 0, 1] + v[0, 0, 0]
    c011 = v[0, 1, 1] - v[0, 1, 0] - v[0, 0, 1] + v[0, 0, 0]
    c111 = (v[1, 1, 1] - v[1, 1, 0] - v[1, 0, 1] - v[0, 1, 1]
            + v[1, 0, 0] + v[0, 1, 0] + v[0, 0, 1] - v[0, 0, 0])
    return np.array([c000, c100, c010, c001, c110, c101, c011, c111])


def evaluate(coefs: Sequence[float], x: float, y: float, z: float) -> float:
    """Evaluate V(x,y,z) from packed coefficients."""
    c000, c100, c010, c001, c110, c101, c011, c111 = coefs
    return (c000 + c100 * x + c010 * y + c001 * z
            + c110 * x * y + c101 * x * z + c011 * y * z
            + c111 * x * y * z)


def ray_density_poly(p: np.ndarray, w: np.ndarray, coefs: Sequence[float]) -> List[float]:
    """Coefficients (ascending) of the cubic V(p + t*w) in t.

    Returns [a0, a1, a2, a3] with V(p + t w) = a0 + a1 t + a2 t^2 + a3 t^3.
    This is the raw (tau-free) density restricted to the ray.
    """
    px, py, pz = float(p[0]), float(p[1]), float(p[2])
    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    c000, c100, c010, c001, c110, c101, c011, c111 = coefs

    a0 = evaluate(coefs, px, py, pz)
    a1 = (c100 * wx + c010 * wy + c001 * wz
          + c110 * (px * wy + py * wx)
          + c101 * (px * wz + pz * wx)
          + c011 * (py * wz + pz * wy)
          + c111 * (px * py * wz + px * pz * wy + py * pz * wx))
    a2 = (c110 * wx * wy
          + c101 * wx * wz
          + c011 * wy * wz
          + c111 * (px * wy * wz + py * wx * wz + pz * wx * wy))
    a3 = c111 * wx * wy * wz
    return [a0, a1, a2, a3]


def random_corners(rng: np.random.RandomState, low: float = -1.0,
                   high: float = 2.0) -> np.ndarray:
    """Random 2x2x2 corner densities in [low, high]."""
    return rng.uniform(low, high, size=(2, 2, 2))
