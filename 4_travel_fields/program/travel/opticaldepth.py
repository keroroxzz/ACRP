"""Analytic optical depth along a ray through one density cell.

The physical extinction is the iso-threshold-clamped density

    d(t) = max(0, V(t) - tau),

where V(t) is the (cubic) trilinear density restricted to the ray and ``tau`` is an
iso-surface threshold that carves the cloud out of the lattice field.  The optical
depth accumulated from the segment start ``t_a`` is

    A(t) = sigma_t * integral_{t_a}^{t} d(s) ds .

Because d >= 0, A is monotone non-decreasing (the Monotone Optical-Depth clock), and
free-path sampling reduces to its generalized inverse (see ``travel.inversion``).

This module implements the *analytic* method under test: closed-form cubic roots
partition the segment into sign-constant pieces and each positive piece is integrated
exactly with the quartic antiderivative.  ``travel.reference`` provides an independent
adaptive-quadrature check.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from . import poly


# --------------------------------------------------------------------------------------
# Real-root finding for the clamped cubic  g(t) = V(t) - tau
# --------------------------------------------------------------------------------------
def real_roots(coeffs: Sequence[float], newton_polish: bool = True,
               imag_tol: float = 1e-6) -> List[float]:
    """Robust real roots of a polynomial of any degree (ascending coefficients).

    Uses the companion-matrix eigenvalues (``numpy.roots``), which stay accurate for the
    grazing / near-degenerate cases that break the normalized closed-form solve: when
    ``a3 = c111 wx wy wz -> 0`` the cubic is effectively a quadratic, and Cardano on the
    a3-normalized form loses all precision, whereas the companion solver returns the
    finite roots correctly (the spurious ~1/a3 root simply appears as a large real value
    we ignore downstream by interval filtering).

    Roots with a non-negligible imaginary part are dropped; the survivors are
    Newton-polished against the original polynomial.
    """
    c = poly.trim(coeffs, tol=0.0)
    if len(c) <= 1:
        return []
    comp = np.roots(np.asarray(c[::-1], dtype=float))  # numpy wants descending order
    scale = max((abs(x) for x in c), default=1.0) or 1.0
    d = poly.derivative(coeffs)
    deg = len(c) - 1

    # A repeated root emerges from numpy.roots as a near-real conjugate pair whose
    # imaginary part is ~sqrt(eps); use a *relative* imaginary tolerance, then confirm
    # each survivor is a genuine real root by its residual after Newton polishing.
    roots = []  # type: List[float]
    for z in comp:
        z = complex(z)
        if abs(z.imag) > imag_tol * (1.0 + abs(z.real)):
            continue
        r = z.real
        best_r = r
        best_res = abs(poly.eval_horner(coeffs, r))
        if newton_polish:
            # Residual-monotone polishing: Newton sharpens simple roots but must never
            # worsen the (already accurate) companion-matrix value at a multiple root
            # where f'(r) -> 0 makes the step blow up, so we keep the best iterate.
            for _ in range(6):
                f = poly.eval_horner(coeffs, r)
                fp = poly.eval_horner(d, r)
                if fp == 0.0:
                    break
                step = f / fp
                r -= step
                res = abs(poly.eval_horner(coeffs, r))
                if res < best_res:
                    best_r, best_res = r, res
                if abs(step) < 1e-15 * (1.0 + abs(r)):
                    break
        # residual guard: reject real parts of genuinely complex roots
        if best_res <= 1e-6 * scale * (1.0 + abs(best_r)) ** deg:
            roots.append(best_r)

    return sorted(roots)


def cubic_real_roots(coeffs: Sequence[float], newton_polish: bool = True) -> List[float]:
    """Real roots of a polynomial of degree <= 3 (robust companion-matrix path)."""
    return real_roots(coeffs, newton_polish=newton_polish)


def cubic_real_roots_cardano(coeffs: Sequence[float],
                             newton_polish: bool = True) -> List[float]:
    """Naive closed-form (Cardano + deflation) solver -- mirrors the GPU prototype.

    Kept for the documented fidelity/robustness comparison: this path is *not* robust
    for grazing rays (leading coefficient -> 0) and is the counter-example the Phase A
    loop flags.  Do not use it in the hot path.
    """
    c = poly.trim(coeffs, tol=0.0)
    deg = len(c) - 1
    if deg <= 0:
        roots = []  # type: List[float]
    elif deg == 1:
        roots = [-c[0] / c[1]]
    elif deg == 2:
        roots = _quadratic_real_roots(c[2], c[1], c[0])
    else:
        roots = _cubic_roots_deflate(c)

    if newton_polish and roots:
        d = poly.derivative(coeffs)
        polished = []
        for r in roots:
            for _ in range(4):
                f = poly.eval_horner(coeffs, r)
                fp = poly.eval_horner(d, r)
                if fp == 0.0:
                    break
                step = f / fp
                r -= step
                if abs(step) < 1e-15 * (1.0 + abs(r)):
                    break
            polished.append(r)
        roots = polished
    return sorted(roots)


def _quadratic_real_roots(a: float, b: float, c: float) -> List[float]:
    """Real roots of a t^2 + b t + c, numerically stable, with a possible 0 leading."""
    if a == 0.0:
        return [] if b == 0.0 else [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return []
    if disc == 0.0:
        return [-b / (2.0 * a)]
    sq = np.sqrt(disc)
    q = -0.5 * (b + np.copysign(sq, b)) if b != 0.0 else -0.5 * sq
    r1 = q / a
    r2 = c / q if q != 0.0 else -r1
    return [r1, r2]


def _cubic_closed_form_real(a0: float, a1: float, a2: float, a3: float) -> List[float]:
    """All real roots of a3 t^3 + a2 t^2 + a1 t + a0 (a3 != 0) via Cardano / trig."""
    p2, p1, p0 = a2 / a3, a1 / a3, a0 / a3
    shift = p2 / 3.0
    P = p1 - p2 * p2 / 3.0
    Q = 2.0 * p2 ** 3 / 27.0 - p2 * p1 / 3.0 + p0
    disc = (Q * Q) / 4.0 + (P ** 3) / 27.0
    if disc > 0.0:
        s = np.sqrt(disc)
        u = np.cbrt(-Q / 2.0 + s)
        v = np.cbrt(-Q / 2.0 - s)
        return [u + v - shift]
    if P == 0.0 and Q == 0.0:
        return [-shift]
    # disc <= 0: three real roots (casus irreducibilis)
    r = np.sqrt(-(P ** 3) / 27.0)
    phi = np.arccos(np.clip(-Q / (2.0 * r), -1.0, 1.0))
    m = 2.0 * np.sqrt(-P / 3.0)
    return [m * np.cos((phi + 2.0 * np.pi * k) / 3.0) - shift for k in range(3)]


def _cubic_roots_deflate(c: Sequence[float]) -> List[float]:
    """Real roots of a genuine cubic via one closed-form root + quadratic deflation.

    The pivot root is the smallest-magnitude closed-form root: forward deflation is
    numerically stable when dividing out the smallest root, which avoids the
    cancellation that corrupts the quotient when a large spurious root is used.
    """
    a0, a1, a2, a3 = c
    cand = _cubic_closed_form_real(a0, a1, a2, a3)
    r0 = min(cand, key=abs)
    # synthetic division of a3 t^3 + a2 t^2 + a1 t + a0 by (t - r0)
    b2 = a3
    b1 = a2 + b2 * r0
    b0 = a1 + b1 * r0
    return [r0] + _quadratic_real_roots(b2, b1, b0)


# --------------------------------------------------------------------------------------
# Optical-depth profile over a single ray segment
# --------------------------------------------------------------------------------------
class OpticalDepthProfile:
    """Analytic A(t) for d(t) = max(0, V(t) - tau) on a segment [t_a, t_b].

    Precomputes the sign-constant partition so that A(t), A'(t), the total optical
    depth and the generalized inverse are all available in closed form.
    """

    def __init__(self, rho_coeffs: Sequence[float], tau: float, sigma_t: float,
                 t_a: float, t_b: float):
        if t_b < t_a:
            raise ValueError("t_b must be >= t_a")
        self.sigma_t = float(sigma_t)
        self.t_a = float(t_a)
        self.t_b = float(t_b)

        # g(t) = V(t) - tau  (ascending coefficients)
        g = list(rho_coeffs)
        g[0] = g[0] - tau
        self.g = g
        self.G = poly.antiderivative(g)  # antiderivative, G(0)=0

        # breakpoints = segment ends + interior roots of g
        roots = [r for r in cubic_real_roots(g) if t_a < r < t_b]
        pts = [t_a] + sorted(roots) + [t_b]

        # positive sub-intervals (where g >= 0) and their cumulative optical depth
        self.pos_intervals = []  # type: List[Tuple[float, float]]
        self._cum = [0.0]        # cumulative A at the *right* end of each pos interval
        cum = 0.0
        for i in range(len(pts) - 1):
            l, r = pts[i], pts[i + 1]
            if r <= l:
                continue
            mid = 0.5 * (l + r)
            if poly.eval_horner(g, mid) > 0.0:
                area = self.sigma_t * (poly.eval_horner(self.G, r) - poly.eval_horner(self.G, l))
                cum += area
                self.pos_intervals.append((l, r))
                self._cum.append(cum)
        self.total = cum

    def density(self, t: float) -> float:
        """Clamped density d(t) = max(0, V(t) - tau)."""
        return max(0.0, poly.eval_horner(self.g, t))

    def A(self, t: float) -> float:
        """Optical depth accumulated from t_a to t (clamped to the segment)."""
        if t <= self.t_a:
            return 0.0
        if t >= self.t_b:
            return self.total
        acc = 0.0
        for (l, r) in self.pos_intervals:
            if t <= l:
                break
            hi = min(r, t)
            acc += self.sigma_t * (poly.eval_horner(self.G, hi) - poly.eval_horner(self.G, l))
        return acc

    def A_prime(self, t: float) -> float:
        """dA/dt = sigma_t * d(t) >= 0."""
        return self.sigma_t * self.density(t)
