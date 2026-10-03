"""Generalized inverse of the monotone optical-depth clock (free-path sampling).

Given a target budget xi (typically xi = -ln(1-U) for U ~ Uniform(0,1)), the free path
is

    T_xi = inf{ t in [t_a, t_b] : A(t) >= xi } .

Since A is piecewise-quartic, monotone non-decreasing, and *strictly* increasing only
on the positive sub-intervals of the clamped density, the inverse is found by:

1. locating the positive sub-interval whose cumulative optical depth brackets xi
   (an O(#intervals) walk over precomputed cumulative sums -- no multi-root isolation,
   thanks to the Monotone Optical-Depth Inversion result);
2. a safeguarded Newton iteration inside that sub-interval, bisection-bracketed so it
   can never leave the bracket even when A'(t) -> 0 at a grazing endpoint.
"""
from __future__ import annotations

from typing import Optional

from . import poly
from .opticaldepth import OpticalDepthProfile

INF = float("inf")


def free_path(profile: OpticalDepthProfile, xi: float,
              tol: float = 1e-13, max_iter: int = 80) -> float:
    """Solve A(T) = xi for T; returns +inf if the segment never accumulates xi."""
    if xi <= 0.0:
        return profile.t_a
    if xi > profile.total:
        return INF

    cum = profile._cum
    intervals = profile.pos_intervals

    # smallest k with cumulative-at-right-end (cum[k+1]) >= xi
    k = 0
    while k < len(intervals) and cum[k + 1] < xi:
        k += 1
    # xi <= total guarantees k is valid
    l, r = intervals[k]
    target = xi - cum[k]  # optical depth to accumulate *within* interval k

    if target <= tol:
        # xi is reached at (or before) the left end of interval k: the true inf is
        # the right end of the previous positive interval / the segment start.
        return intervals[k - 1][1] if k > 0 else profile.t_a

    return _safeguarded_newton(profile, l, r, cum[k], xi, tol, max_iter)


def _safeguarded_newton(profile: OpticalDepthProfile, l: float, r: float,
                        cum_before: float, xi: float, tol: float, max_iter: int) -> float:
    """Find t in [l, r] with cum_before + sigma_t*(G(t)-G(l)) = xi.

    F(t) := profile.A_within(l, t) is increasing on [l, r]; Newton uses F'(t)=A'(t),
    but every step is clamped to the current [lo, hi] bracket and demoted to a
    bisection step if it would escape or fails to make progress.
    """
    sigma_t = profile.sigma_t
    Gl = poly.eval_horner(profile.G, l)

    def F(t: float) -> float:
        return cum_before + sigma_t * (poly.eval_horner(profile.G, t) - Gl) - xi

    lo, hi = l, r
    flo = F(lo)
    fhi = F(hi)
    # F(lo) <= 0 <= F(hi) by construction of the bracket
    if flo > 0.0:
        return lo
    if fhi < 0.0:
        return hi

    t = 0.5 * (lo + hi)
    for _ in range(max_iter):
        ft = F(t)
        if ft < 0.0:
            lo = t
        else:
            hi = t
        if abs(ft) <= tol or (hi - lo) <= tol * (1.0 + abs(t)):
            return t
        fp = profile.A_prime(t)  # = sigma_t * d(t) >= 0
        if fp > 0.0:
            step = ft / fp
            t_newton = t - step
            if lo < t_newton < hi:
                t = t_newton
                continue
        # fall back to bisection
        t = 0.5 * (lo + hi)
    return t
