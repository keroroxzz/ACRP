
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ============================================================
# Core cubic-fade medium
# ============================================================

def fade(x: np.ndarray | float) -> np.ndarray | float:
    """Cubic fade f(x)=3x^2-2x^3."""
    return x * x * (3.0 - 2.0 * x)


def fade_poly(a: float, b: float) -> np.ndarray:
    """
    Coefficients of fade(a + b s) in ascending power order:
        p[0] + p[1] s + p[2] s^2 + p[3] s^3
    """
    return np.array(
        [
            3.0 * a * a - 2.0 * a * a * a,
            6.0 * a * b * (1.0 - a),
            3.0 * b * b * (1.0 - 2.0 * a),
            -2.0 * b * b * b,
        ],
        dtype=np.float64,
    )


def pad(p: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    out[: min(n, len(p))] = p[: min(n, len(p))]
    return out


def corner_differences(c: np.ndarray) -> Tuple[float, ...]:
    """
    c uses c[x,y,z], each index in {0,1}.
    Returns multilinear coefficients in faded coordinates.
    """
    d000 = float(c[0, 0, 0])
    d100 = float(c[1, 0, 0])
    d010 = float(c[0, 1, 0])
    d110 = float(c[1, 1, 0])
    d001 = float(c[0, 0, 1])
    d101 = float(c[1, 0, 1])
    d011 = float(c[0, 1, 1])
    d111 = float(c[1, 1, 1])

    a = d000
    ax = d100 - d000
    ay = d010 - d000
    az = d001 - d000
    axy = d110 - d100 - d010 + d000
    axz = d101 - d100 - d001 + d000
    ayz = d011 - d010 - d001 + d000
    axyz = (
        d111
        - d110
        - d101
        - d011
        + d100
        + d010
        + d001
        - d000
    )
    return a, ax, ay, az, axy, axz, ayz, axyz


def build_density_poly(
    local_entry: np.ndarray,
    direction: np.ndarray,
    corners: np.ndarray,
) -> np.ndarray:
    """
    Build degree <= 9 polynomial rho(s) for cubic-fade value noise
    restricted to a ray segment inside one unit cell.

    Returns 10 coefficients in ascending power order.
    """
    fx = fade_poly(float(local_entry[0]), float(direction[0]))
    fy = fade_poly(float(local_entry[1]), float(direction[1]))
    fz = fade_poly(float(local_entry[2]), float(direction[2]))

    fxy = np.polynomial.polynomial.polymul(fx, fy)      # degree <= 6
    fxz = np.polynomial.polynomial.polymul(fx, fz)
    fyz = np.polynomial.polynomial.polymul(fy, fz)
    fxyz = np.polynomial.polynomial.polymul(fxy, fz)    # degree <= 9

    a, ax, ay, az, axy, axz, ayz, axyz = corner_differences(corners)

    rho = np.zeros(10, dtype=np.float64)
    rho[0] = a
    rho += ax * pad(fx, 10)
    rho += ay * pad(fy, 10)
    rho += az * pad(fz, 10)
    rho += axy * pad(fxy, 10)
    rho += axz * pad(fxz, 10)
    rho += ayz * pad(fyz, 10)
    rho += axyz * pad(fxyz, 10)
    return rho


def build_tau_poly(rho: np.ndarray, sigma_t: float = 1.0) -> np.ndarray:
    """
    If rho(s)=sum_{k=0}^9 c_k s^k, returns degree <= 10:
        tau(s)=sigma_t * integral_0^s rho(u) du.
    """
    tau = np.zeros(11, dtype=np.float64)
    for k, ck in enumerate(rho):
        tau[k + 1] = sigma_t * ck / float(k + 1)
    return tau


def horner_ascending(coeff: np.ndarray, x: float) -> float:
    """Evaluate ascending-power coefficients with Horner's method."""
    y = 0.0
    for ck in coeff[::-1]:
        y = y * x + float(ck)
    return y


def direct_density(
    s: float,
    local_entry: np.ndarray,
    direction: np.ndarray,
    corners: np.ndarray,
) -> float:
    """
    Direct cubic-fade interpolation. Assumes local_entry+s*direction
    remains in [0,1]^3.
    """
    q = local_entry + s * direction
    f = fade(q)

    x00 = (1.0 - f[0]) * corners[0, 0, 0] + f[0] * corners[1, 0, 0]
    x10 = (1.0 - f[0]) * corners[0, 1, 0] + f[0] * corners[1, 1, 0]
    x01 = (1.0 - f[0]) * corners[0, 0, 1] + f[0] * corners[1, 0, 1]
    x11 = (1.0 - f[0]) * corners[0, 1, 1] + f[0] * corners[1, 1, 1]

    y0 = (1.0 - f[1]) * x00 + f[1] * x10
    y1 = (1.0 - f[1]) * x01 + f[1] * x11
    return float((1.0 - f[2]) * y0 + f[2] * y1)


# 8-point Gauss-Legendre: exact for polynomials through degree 15.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)


def quadrature_tau(
    s: float,
    local_entry: np.ndarray,
    direction: np.ndarray,
    corners: np.ndarray,
    sigma_t: float = 1.0,
) -> float:
    """
    Independent optical-depth reference using 8-point Gauss-Legendre.
    Since direct density is degree <= 9 in s, this is exact in exact
    arithmetic and highly accurate in float64.
    """
    if s <= 0.0:
        return 0.0
    nodes = 0.5 * s * (_GL_X + 1.0)
    values = np.array(
        [direct_density(float(u), local_entry, direction, corners) for u in nodes]
    )
    return float(sigma_t * 0.5 * s * np.dot(_GL_W, values))


# ============================================================
# Monotone inversion
# ============================================================

@dataclass
class InversionResult:
    s: float
    iterations: int
    fallbacks: int
    converged: bool
    residual: float
    bracket_width: float


def safeguarded_newton(
    rho: np.ndarray,
    tau: np.ndarray,
    segment_length: float,
    target_tau: float,
    sigma_t: float = 1.0,
    max_iter: int = 20,
    tau_abs_tol: float = 2e-12,
    tau_rel_tol: float = 2e-11,
    s_tol: float = 2e-12,
    derivative_floor: float = 1e-14,
) -> InversionResult:
    """
    Left-bracketed safeguarded Newton for monotone tau(s).

    Invariants:
      tau(lo) <= target_tau <= tau(hi)
      0 <= lo <= hi <= segment_length
    """
    total = horner_ascending(tau, segment_length)
    if target_tau <= 0.0:
        return InversionResult(0.0, 0, 0, True, abs(target_tau), 0.0)
    if target_tau > total + max(tau_abs_tol, tau_rel_tol * max(1.0, abs(total))):
        raise ValueError("target_tau lies beyond this cell")

    lo, hi = 0.0, float(segment_length)
    # Optical-depth fraction is a good GPU-friendly initial guess.
    x = hi * min(max(target_tau / max(total, 1e-300), 0.0), 1.0)
    fallbacks = 0

    for it in range(1, max_iter + 1):
        tx = horner_ascending(tau, x)
        residual_signed = tx - target_tau
        residual = abs(residual_signed)
        tol = tau_abs_tol + tau_rel_tol * max(1.0, abs(target_tau))

        if residual <= tol or (hi - lo) <= s_tol * max(1.0, segment_length):
            return InversionResult(x, it, fallbacks, True, residual, hi - lo)

        if residual_signed < 0.0:
            lo = x
        else:
            hi = x

        density = max(horner_ascending(rho, x), 0.0)
        derivative = sigma_t * density

        use_newton = math.isfinite(derivative) and derivative > derivative_floor
        if use_newton:
            candidate = x - residual_signed / derivative
            use_newton = math.isfinite(candidate) and lo < candidate < hi
        else:
            candidate = math.nan

        if not use_newton:
            candidate = 0.5 * (lo + hi)
            fallbacks += 1

        x = candidate

    tx = horner_ascending(tau, x)
    return InversionResult(
        x,
        max_iter,
        fallbacks,
        False,
        abs(tx - target_tau),
        hi - lo,
    )


def bisection_reference(
    tau: np.ndarray,
    segment_length: float,
    target_tau: float,
    iterations: int = 90,
) -> float:
    """High-accuracy monotone inverse reference."""
    lo, hi = 0.0, float(segment_length)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if horner_ascending(tau, mid) < target_tau:
            lo = mid
        else:
            hi = mid
    return hi  # left generalized inverse approximation


# ============================================================
# Random valid ray segments inside a cell
# ============================================================

def random_segment(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate p(s)=p0+s*d, s in [0,L], guaranteed to remain in [0,1]^3.
    Direction is unit length so s has world-distance semantics.
    """
    p0 = rng.uniform(0.0, 1.0, size=3)
    d = rng.normal(size=3)
    norm = np.linalg.norm(d)
    if norm < 1e-15:
        d = np.array([1.0, 0.0, 0.0])
    else:
        d = d / norm

    exit_distances = []
    for axis in range(3):
        if d[axis] > 1e-14:
            exit_distances.append((1.0 - p0[axis]) / d[axis])
        elif d[axis] < -1e-14:
            exit_distances.append((0.0 - p0[axis]) / d[axis])
    max_len = min(exit_distances) if exit_distances else 0.0
    # Avoid exact endpoint for random tests; explicit boundary tests cover it.
    L = float(max_len * rng.uniform(1e-6, 0.999999))
    return p0, d, L


# ============================================================
# Validation
# ============================================================

@dataclass
class ValidationSummary:
    seed: int
    random_cases: int
    density_samples: int
    max_density_abs_error: float
    max_density_rel_error: float
    max_tau_abs_error: float
    max_tau_rel_error: float
    max_inverse_s_error: float
    max_inverse_tau_residual: float
    max_monotonicity_violation: float
    inversion_failures: int
    mean_newton_iterations: float
    p95_newton_iterations: float
    max_newton_iterations: int
    mean_fallbacks: float
    p95_fallbacks: float
    max_fallbacks: int
    plateau_cases_passed: bool
    constant_cases_passed: bool
    axis_aligned_cases_passed: bool


def rel_error(a: float, b: float) -> float:
    return abs(a - b) / max(1.0, abs(a), abs(b))


def run_validation(seed: int = 20260718, random_cases: int = 10000) -> ValidationSummary:
    rng = np.random.default_rng(seed)

    max_density_abs = 0.0
    max_density_rel = 0.0
    max_tau_abs = 0.0
    max_tau_rel = 0.0
    max_inv_s = 0.0
    max_inv_res = 0.0
    max_mono_violation = 0.0
    density_samples = 0
    failures = 0
    iterations: List[int] = []
    fallbacks: List[int] = []

    for _ in range(random_cases):
        corners = rng.uniform(0.0, 3.0, size=(2, 2, 2))
        p0, d, L = random_segment(rng)
        sigma_t = float(10.0 ** rng.uniform(-2.0, 1.0))

        rho = build_density_poly(p0, d, corners)
        tau = build_tau_poly(rho, sigma_t)

        # Density reconstruction at several points.
        for s in np.linspace(0.0, L, 7):
            direct = direct_density(float(s), p0, d, corners)
            poly = horner_ascending(rho, float(s))
            max_density_abs = max(max_density_abs, abs(direct - poly))
            max_density_rel = max(max_density_rel, rel_error(direct, poly))
            density_samples += 1

        # Optical-depth reconstruction.
        for s in np.linspace(0.0, L, 5):
            qref = quadrature_tau(float(s), p0, d, corners, sigma_t)
            qpoly = horner_ascending(tau, float(s))
            max_tau_abs = max(max_tau_abs, abs(qref - qpoly))
            max_tau_rel = max(max_tau_rel, rel_error(qref, qpoly))

        # Sampled monotonicity check.
        grid = np.linspace(0.0, L, 33)
        tau_grid = np.array([horner_ascending(tau, float(s)) for s in grid])
        diffs = np.diff(tau_grid)
        max_mono_violation = max(max_mono_violation, float(max(0.0, -np.min(diffs))))

        total_tau = horner_ascending(tau, L)
        if total_tau > 1e-13:
            # Exercise boundaries and interior targets.
            fractions = [1e-10, 1e-6, 0.01, 0.25, 0.5, 0.9, 1.0 - 1e-8]
            frac = float(fractions[rng.integers(0, len(fractions))])
            target = frac * total_tau

            result = safeguarded_newton(
                rho, tau, L, target, sigma_t=sigma_t, max_iter=20
            )
            ref_s = bisection_reference(tau, L, target)

            max_inv_s = max(max_inv_s, abs(result.s - ref_s))
            max_inv_res = max(max_inv_res, result.residual)
            iterations.append(result.iterations)
            fallbacks.append(result.fallbacks)
            if not result.converged:
                failures += 1

    # Explicit zero-density plateau.
    corners = np.zeros((2, 2, 2), dtype=np.float64)
    p0 = np.array([0.2, 0.3, 0.4])
    d = np.array([0.7, -0.2, 0.1])
    d = d / np.linalg.norm(d)
    rho = build_density_poly(p0, d, corners)
    tau = build_tau_poly(rho, 2.0)
    plateau_ok = bool(
        np.allclose(rho, 0.0, atol=0.0, rtol=0.0)
        and np.allclose(tau, 0.0, atol=0.0, rtol=0.0)
    )

    # Constant-density field.
    constant_ok = True
    for c in [0.0, 1e-9, 0.25, 2.0]:
        corners = np.full((2, 2, 2), c)
        p0, d, L = random_segment(rng)
        rho = build_density_poly(p0, d, corners)
        tau = build_tau_poly(rho, 1.7)
        expected_rho = np.zeros(10)
        expected_rho[0] = c
        expected_tau = np.zeros(11)
        expected_tau[1] = 1.7 * c
        constant_ok &= np.allclose(rho, expected_rho, atol=2e-14, rtol=2e-14)
        constant_ok &= np.allclose(tau, expected_tau, atol=2e-14, rtol=2e-14)

    # Axis-aligned rays.
    axis_ok = True
    for axis in range(3):
        for sign in [-1.0, 1.0]:
            corners = rng.uniform(0.0, 2.0, size=(2, 2, 2))
            p0 = rng.uniform(0.05, 0.95, size=3)
            d = np.zeros(3)
            d[axis] = sign
            L = (1.0 - p0[axis]) if sign > 0 else p0[axis]
            L *= 0.999999
            rho = build_density_poly(p0, d, corners)
            tau = build_tau_poly(rho, 0.8)
            for s in np.linspace(0.0, L, 11):
                axis_ok &= math.isclose(
                    horner_ascending(rho, float(s)),
                    direct_density(float(s), p0, d, corners),
                    rel_tol=2e-12,
                    abs_tol=2e-12,
                )
                axis_ok &= math.isclose(
                    horner_ascending(tau, float(s)),
                    quadrature_tau(float(s), p0, d, corners, 0.8),
                    rel_tol=2e-12,
                    abs_tol=2e-12,
                )

    iterations_arr = np.array(iterations, dtype=float)
    fallbacks_arr = np.array(fallbacks, dtype=float)

    return ValidationSummary(
        seed=seed,
        random_cases=random_cases,
        density_samples=density_samples,
        max_density_abs_error=max_density_abs,
        max_density_rel_error=max_density_rel,
        max_tau_abs_error=max_tau_abs,
        max_tau_rel_error=max_tau_rel,
        max_inverse_s_error=max_inv_s,
        max_inverse_tau_residual=max_inv_res,
        max_monotonicity_violation=max_mono_violation,
        inversion_failures=failures,
        mean_newton_iterations=float(np.mean(iterations_arr)),
        p95_newton_iterations=float(np.percentile(iterations_arr, 95)),
        max_newton_iterations=int(np.max(iterations_arr)),
        mean_fallbacks=float(np.mean(fallbacks_arr)),
        p95_fallbacks=float(np.percentile(fallbacks_arr, 95)),
        max_fallbacks=int(np.max(fallbacks_arr)),
        plateau_cases_passed=plateau_ok,
        constant_cases_passed=bool(constant_ok),
        axis_aligned_cases_passed=bool(axis_ok),
    )


def make_report(summary: ValidationSummary) -> str:
    status = (
        summary.inversion_failures == 0
        and summary.plateau_cases_passed
        and summary.constant_cases_passed
        and summary.axis_aligned_cases_passed
        and summary.max_density_abs_error < 1e-10
        and summary.max_tau_abs_error < 1e-10
        and summary.max_monotonicity_violation < 1e-11
    )
    return f"""# Cubic-Fade 3D Degree-10 Medium Validation

**Overall status:** {"PASS" if status else "NEEDS REVIEW"}

## Configuration

- Random seed: `{summary.seed}`
- Random valid in-cell ray segments: `{summary.random_cases}`
- Direct density comparison samples: `{summary.density_samples}`
- Corner densities: nonnegative random values in `[0, 3]`
- Extinction scale: log-uniform in `[10^-2, 10^1]`
- Optical-depth reference: independent 8-point Gauss-Legendre quadrature
- Inverse reference: 90-step monotone bisection
- Production candidate: bracketed safeguarded Newton

## Algebraic reconstruction

- Maximum absolute density error: `{summary.max_density_abs_error:.6e}`
- Maximum relative density error: `{summary.max_density_rel_error:.6e}`
- Maximum absolute optical-depth error: `{summary.max_tau_abs_error:.6e}`
- Maximum relative optical-depth error: `{summary.max_tau_rel_error:.6e}`
- Maximum sampled monotonicity violation: `{summary.max_monotonicity_violation:.6e}`

## Free-path inversion

- Inversion failures: `{summary.inversion_failures}`
- Maximum distance error versus reference: `{summary.max_inverse_s_error:.6e}`
- Maximum final optical-depth residual: `{summary.max_inverse_tau_residual:.6e}`
- Mean Newton iterations: `{summary.mean_newton_iterations:.3f}`
- P95 Newton iterations: `{summary.p95_newton_iterations:.3f}`
- Maximum Newton iterations: `{summary.max_newton_iterations}`
- Mean fallback count: `{summary.mean_fallbacks:.3f}`
- P95 fallback count: `{summary.p95_fallbacks:.3f}`
- Maximum fallback count: `{summary.max_fallbacks}`

## Explicit edge cases

- Zero-density plateau: `{"PASS" if summary.plateau_cases_passed else "FAIL"}`
- Constant-density cells: `{"PASS" if summary.constant_cases_passed else "FAIL"}`
- Axis-aligned rays: `{"PASS" if summary.axis_aligned_cases_passed else "FAIL"}`

## Interpretation

The test validates the central single-cell primitive:

1. Cubic-fade trivariate interpolation restricted to a straight ray is represented by a degree-at-most-9 polynomial.
2. Its accumulated optical depth is represented by a degree-at-most-10 polynomial.
3. For nonnegative corner densities, the direct interpolant is nonnegative and the optical-depth clock is monotone.
4. A bracketed safeguarded Newton solver recovers the left generalized inverse without general tenth-degree root isolation.
5. The coefficient layout and Horner evaluation are directly transferable to a GPU implementation with fixed-size arrays and fixed iteration caps.

This report validates the algebra and scalar inversion. It does not yet validate whole-volume DDA traversal, floating-point behavior in GLSL `float`, or GPU wall-clock performance.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--cases", type=int, default=10000)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    summary = run_validation(args.seed, args.cases)
    payload = asdict(summary)
    print(json.dumps(payload, indent=2))

    if args.json:
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.report:
        args.report.write_text(make_report(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
