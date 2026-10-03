"""Independent high-precision reference for optical depth and free path.

This module deliberately shares *no code path* with the analytic method under test:

* density is evaluated straight from the trilinear coefficients and the ray
  (``trilinear.evaluate`` on p + t w), which also cross-checks the ray-cubic
  construction in ``trilinear.ray_density_poly``;
* kinks of the clamped density max(0, V - tau) are located by a numeric sign scan +
  bisection, **not** by the closed-form cubic solver;
* each sign-constant panel is integrated by high-order Gauss-Legendre quadrature, and
  the cumulative optical depth is precomputed once so that ``A(t)`` and the inverse are
  both cheap.

A 64-node Gauss rule integrates polynomials up to degree 127 exactly, so the reference
stays exact for the cubic trilinear medium *and* the forthcoming degree-10 cubic-fade
medium.  Double precision yields errors ~1e-13, enough to catch the algorithmic
failures (root isolation, tie-breaking, inversion bracketing) this loop targets.  For
sub-ULP certification, swap ``_gauss_panel`` for an ``mpmath.quad`` backend.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from . import trilinear

# Gauss-Legendre nodes/weights on [-1, 1], computed once.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(64)

INF = float("inf")


def _find_kinks(coefs, p, w, tau, t_a, t_b, scan=512) -> List[float]:
    """Sign changes of g(t) = V(p + t w) - tau in (t_a, t_b) by scan + bisection."""
    def g(t):
        x = p + t * w
        return trilinear.evaluate(coefs, x[0], x[1], x[2]) - tau

    ts = np.linspace(t_a, t_b, scan)
    vals = np.array([g(t) for t in ts])
    kinks = []
    for i in range(len(ts) - 1):
        a, b = vals[i], vals[i + 1]
        if a == 0.0:
            kinks.append(ts[i])
        elif a * b < 0.0:
            lo, hi, flo = ts[i], ts[i + 1], a
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                fmid = g(mid)
                if fmid == 0.0:
                    lo = hi = mid
                    break
                if flo * fmid < 0.0:
                    hi = mid
                else:
                    lo, flo = mid, fmid
            kinks.append(0.5 * (lo + hi))
    return kinks


class ReferenceProfile:
    """Reference A(t) for d(t) = max(0, V(p + t w) - tau) on [t_a, t_b]."""

    def __init__(self, coefs: Sequence[float], p: np.ndarray, w: np.ndarray,
                 tau: float, sigma_t: float, t_a: float, t_b: float):
        self.coefs = coefs
        self.p = np.asarray(p, dtype=float)
        self.w = np.asarray(w, dtype=float)
        self.tau = float(tau)
        self.sigma_t = float(sigma_t)
        self.t_a = float(t_a)
        self.t_b = float(t_b)

        kinks = [k for k in _find_kinks(coefs, self.p, self.w, tau, t_a, t_b)
                 if t_a < k < t_b]
        self.breaks = [t_a] + sorted(kinks) + [t_b]  # panel boundaries

        # cumulative optical depth at each panel boundary
        self.cum = [0.0]
        acc = 0.0
        for i in range(len(self.breaks) - 1):
            acc += self._panel_integral(self.breaks[i], self.breaks[i + 1])
            self.cum.append(acc)
        self.total = acc

    def _d(self, t: float) -> float:
        x = self.p + t * self.w
        return max(0.0, trilinear.evaluate(self.coefs, x[0], x[1], x[2]) - self.tau)

    def _panel_integral(self, l: float, r: float) -> float:
        """sigma_t * integral_l^r d dt via 64-node Gauss-Legendre (exact for polys)."""
        if r <= l:
            return 0.0
        half = 0.5 * (r - l)
        mid = 0.5 * (r + l)
        nodes = mid + half * _GL_NODES
        vals = np.array([self._d(t) for t in nodes])
        return self.sigma_t * half * np.dot(_GL_WEIGHTS, vals)

    def _panel_of(self, t: float) -> int:
        i = 0
        while i < len(self.breaks) - 1 and self.breaks[i + 1] <= t:
            i += 1
        return i

    def A(self, t: float) -> float:
        """Optical depth accumulated from t_a to t (clamped to the segment)."""
        if t <= self.t_a:
            return 0.0
        if t >= self.t_b:
            return self.total
        i = self._panel_of(t)
        return self.cum[i] + self._panel_integral(self.breaks[i], t)

    def A_prime(self, t: float) -> float:
        return self.sigma_t * self._d(t)

    def free_path(self, xi: float, tol: float = 1e-12, max_iter: int = 60) -> float:
        """T_xi = inf{t in [t_a,t_b]: A(t) >= xi}; +inf if never reached.

        Safeguarded Newton inside the bracketing panel: A'(t)=sigma_t*d(t) drives the
        step, but every step is clamped to the running [lo,hi] bracket and demoted to
        bisection if it would escape, so grazing panels (A'->0) stay robust.
        """
        if xi <= 0.0:
            return self.t_a
        if xi > self.total:
            return INF
        # locate bracketing panel via cumulative sums
        i = 0
        while i < len(self.cum) - 1 and self.cum[i + 1] < xi:
            i += 1
        left = self.breaks[i]
        cum_before = self.cum[i]
        lo, hi = self.breaks[i], self.breaks[i + 1]

        def F(t):
            return cum_before + self._panel_integral(left, t) - xi

        if F(lo) >= 0.0:
            return lo
        t = 0.5 * (lo + hi)
        for _ in range(max_iter):
            ft = F(t)
            if ft < 0.0:
                lo = t
            else:
                hi = t
            if abs(ft) <= tol or (hi - lo) <= tol * (1.0 + abs(t)):
                return t
            fp = self.A_prime(t)
            if fp > 0.0:
                t_newton = t - ft / fp
                if lo < t_newton < hi:
                    t = t_newton
                    continue
            t = 0.5 * (lo + hi)
        return t


# --- convenience functional wrappers -------------------------------------------------
def optical_depth_ref(coefs, p, w, tau, sigma_t, t_a, t_b) -> float:
    return ReferenceProfile(coefs, p, w, tau, sigma_t, t_a, t_b).total


def free_path_ref(coefs, p, w, tau, sigma_t, t_a, t_b, xi) -> float:
    return ReferenceProfile(coefs, p, w, tau, sigma_t, t_a, t_b).free_path(xi)
