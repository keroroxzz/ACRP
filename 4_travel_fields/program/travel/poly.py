"""Monomial-basis polynomial utilities.

A polynomial is represented as a 1-D array of coefficients in *ascending* order:

    p(t) = sum_k coeffs[k] * t**k          (coeffs[0] is the constant term)

Ascending order is chosen because it makes the antiderivative and coefficient
construction (trilinear -> ray) read naturally.  Convert to numpy.poly1d order
(descending) only at the boundary if needed.

All routines are pure-Python / numpy and double precision; the high-precision
reference lives in ``travel.reference``.
"""
from __future__ import annotations

from typing import Sequence, List

import numpy as np


def eval_horner(coeffs: Sequence[float], t: float) -> float:
    """Evaluate p(t) via Horner's scheme (ascending coefficients)."""
    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * t + c
    return acc


def eval_horner_array(coeffs: Sequence[float], t: np.ndarray) -> np.ndarray:
    """Vectorised Horner evaluation over an array of parameters."""
    t = np.asarray(t, dtype=float)
    acc = np.zeros_like(t)
    for c in reversed(coeffs):
        acc = acc * t + c
    return acc


def derivative(coeffs: Sequence[float]) -> List[float]:
    """Coefficients of p'(t) (ascending)."""
    if len(coeffs) <= 1:
        return [0.0]
    return [k * coeffs[k] for k in range(1, len(coeffs))]


def antiderivative(coeffs: Sequence[float], c0: float = 0.0) -> List[float]:
    """Coefficients of the antiderivative P(t) with P(0) = c0 (ascending)."""
    out = [c0]
    for k, c in enumerate(coeffs):
        out.append(c / (k + 1))
    return out


def definite_integral(coeffs: Sequence[float], a: float, b: float) -> float:
    """Exact definite integral of p over [a, b]."""
    P = antiderivative(coeffs)
    return eval_horner(P, b) - eval_horner(P, a)


def add(p: Sequence[float], q: Sequence[float]) -> List[float]:
    """Sum of two polynomials (ascending)."""
    n = max(len(p), len(q))
    out = [0.0] * n
    for k in range(len(p)):
        out[k] += p[k]
    for k in range(len(q)):
        out[k] += q[k]
    return out


def scale(coeffs: Sequence[float], s: float) -> List[float]:
    """Scalar multiple s * p."""
    return [s * c for c in coeffs]


def trim(coeffs: Sequence[float], tol: float = 0.0) -> List[float]:
    """Drop leading (highest-order) coefficients whose magnitude is <= tol.

    Keeps at least the constant term so an all-zero polynomial returns [0.0].
    """
    out = list(coeffs)
    while len(out) > 1 and abs(out[-1]) <= tol:
        out.pop()
    return out


def degree(coeffs: Sequence[float], tol: float = 0.0) -> int:
    """Effective degree after trimming near-zero leading coefficients."""
    return len(trim(coeffs, tol)) - 1
