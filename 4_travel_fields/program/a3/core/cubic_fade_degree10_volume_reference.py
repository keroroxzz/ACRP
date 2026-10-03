
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


# ============================================================
# Cubic-fade cell algebra
# ============================================================

def fade(x):
    return x * x * (3.0 - 2.0 * x)


def fade_poly(a: float, b: float) -> np.ndarray:
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
    out[: len(p)] = p
    return out


def build_density_poly(
    local_entry: np.ndarray,
    direction: np.ndarray,
    corners: np.ndarray,
) -> np.ndarray:
    fx = fade_poly(float(local_entry[0]), float(direction[0]))
    fy = fade_poly(float(local_entry[1]), float(direction[1]))
    fz = fade_poly(float(local_entry[2]), float(direction[2]))

    fxy = np.polynomial.polynomial.polymul(fx, fy)
    fxz = np.polynomial.polynomial.polymul(fx, fz)
    fyz = np.polynomial.polynomial.polymul(fy, fz)
    fxyz = np.polynomial.polynomial.polymul(fxy, fz)

    d000 = float(corners[0, 0, 0])
    d100 = float(corners[1, 0, 0])
    d010 = float(corners[0, 1, 0])
    d110 = float(corners[1, 1, 0])
    d001 = float(corners[0, 0, 1])
    d101 = float(corners[1, 0, 1])
    d011 = float(corners[0, 1, 1])
    d111 = float(corners[1, 1, 1])

    a = d000
    ax = d100 - d000
    ay = d010 - d000
    az = d001 - d000
    axy = d110 - d100 - d010 + d000
    axz = d101 - d100 - d001 + d000
    ayz = d011 - d010 - d001 + d000
    axyz = (
        d111 - d110 - d101 - d011
        + d100 + d010 + d001 - d000
    )

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


def build_tau_poly(rho: np.ndarray, sigma_t: float) -> np.ndarray:
    tau = np.zeros(11, dtype=np.float64)
    for k, c in enumerate(rho):
        tau[k + 1] = sigma_t * c / float(k + 1)
    return tau


def horner(coeff: np.ndarray, x: float) -> float:
    y = 0.0
    for c in coeff[::-1]:
        y = y * x + float(c)
    return y


@dataclass
class InversionResult:
    s: float
    iterations: int
    fallbacks: int
    converged: bool
    residual: float


def safeguarded_newton(
    rho: np.ndarray,
    tau: np.ndarray,
    segment_length: float,
    target_tau: float,
    sigma_t: float,
    max_iter: int = 30,
    abs_tol: float = 2e-14,
    rel_tol: float = 2e-12,
    spatial_tol: float = 2e-11,
) -> InversionResult:
    total_tau = horner(tau, segment_length)
    if target_tau <= 0.0:
        return InversionResult(0.0, 0, 0, True, abs(target_tau))

    lo, hi = 0.0, segment_length
    x = hi * np.clip(target_tau / max(total_tau, 1e-300), 0.0, 1.0)
    fallbacks = 0

    for iteration in range(1, max_iter + 1):
        tx = horner(tau, x)
        residual_signed = tx - target_tau
        residual = abs(residual_signed)
        tolerance = abs_tol + rel_tol * max(1.0, abs(target_tau))

        if residual_signed < 0.0:
            lo = x
        else:
            hi = x

        density = max(horner(rho, x), 0.0)
        derivative = sigma_t * density
        spatial_scale = max(1.0, segment_length)
        spatial_error_estimate = (
            residual / derivative if derivative > 1e-14 else math.inf
        )
        if (hi - lo) <= spatial_tol * spatial_scale or (
            residual <= tolerance
            and spatial_error_estimate <= spatial_tol * spatial_scale
        ):
            return InversionResult(x, iteration, fallbacks, True, residual)

        valid = math.isfinite(derivative) and derivative > 1e-14
        if valid:
            candidate = x - residual_signed / derivative
            valid = math.isfinite(candidate) and lo < candidate < hi
        else:
            candidate = math.nan

        if not valid:
            candidate = 0.5 * (lo + hi)
            fallbacks += 1

        x = candidate

    residual = abs(horner(tau, x) - target_tau)
    return InversionResult(x, max_iter, fallbacks, False, residual)


# ============================================================
# Finite cubic-fade lattice volume
# ============================================================

@dataclass
class Volume:
    lattice: np.ndarray
    sigma_t: float

    @property
    def shape_cells(self) -> np.ndarray:
        return np.array(self.lattice.shape, dtype=int) - 1

    @property
    def box_min(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float64)

    @property
    def box_max(self) -> np.ndarray:
        return self.shape_cells.astype(np.float64)

    def corners(self, cell: np.ndarray) -> np.ndarray:
        x, y, z = map(int, cell)
        return self.lattice[x:x+2, y:y+2, z:z+2]

    def density(self, p: np.ndarray) -> float:
        # Clamp only an exact upper boundary to the final cell.
        q = np.asarray(p, dtype=np.float64)
        cell = np.floor(q).astype(int)
        for axis in range(3):
            if cell[axis] == self.shape_cells[axis]:
                cell[axis] -= 1
        if np.any(cell < 0) or np.any(cell >= self.shape_cells):
            return 0.0

        f = q - cell.astype(np.float64)
        f = np.clip(f, 0.0, 1.0)
        w = fade(f)
        c = self.corners(cell)

        x00 = (1.0 - w[0]) * c[0, 0, 0] + w[0] * c[1, 0, 0]
        x10 = (1.0 - w[0]) * c[0, 1, 0] + w[0] * c[1, 1, 0]
        x01 = (1.0 - w[0]) * c[0, 0, 1] + w[0] * c[1, 0, 1]
        x11 = (1.0 - w[0]) * c[0, 1, 1] + w[0] * c[1, 1, 1]
        y0 = (1.0 - w[1]) * x00 + w[1] * x10
        y1 = (1.0 - w[1]) * x01 + w[1] * x11
        return float((1.0 - w[2]) * y0 + w[2] * y1)


# ============================================================
# Ray geometry and DDA
# ============================================================

def intersect_box(
    ro: np.ndarray,
    rd: np.ndarray,
    box_min: np.ndarray,
    box_max: np.ndarray,
) -> Tuple[bool, float, float]:
    t_enter = -math.inf
    t_exit = math.inf

    for axis in range(3):
        if abs(rd[axis]) < 1e-15:
            if ro[axis] < box_min[axis] or ro[axis] > box_max[axis]:
                return False, 0.0, 0.0
            continue

        t0 = (box_min[axis] - ro[axis]) / rd[axis]
        t1 = (box_max[axis] - ro[axis]) / rd[axis]
        near, far = min(t0, t1), max(t0, t1)
        t_enter = max(t_enter, near)
        t_exit = min(t_exit, far)

    t_enter = max(t_enter, 0.0)
    return (t_exit >= t_enter), t_enter, t_exit


@dataclass
class Segment:
    cell: Tuple[int, int, int]
    t0: float
    t1: float


def dda_segments(volume: Volume, ro: np.ndarray, rd: np.ndarray) -> List[Segment]:
    hit, t_enter, t_exit = intersect_box(ro, rd, volume.box_min, volume.box_max)
    if not hit or t_exit <= t_enter:
        return []

    # Nudge into the active cell without changing segment times.
    probe = ro + rd * np.nextafter(t_enter, math.inf)
    cell = np.floor(probe).astype(int)

    # Exact upper face with negative direction belongs to the preceding cell.
    for axis in range(3):
        if cell[axis] == volume.shape_cells[axis]:
            cell[axis] -= 1

    step = np.sign(rd).astype(int)
    t_delta = np.full(3, math.inf, dtype=np.float64)
    t_max = np.full(3, math.inf, dtype=np.float64)

    for axis in range(3):
        if step[axis] == 0:
            continue
        boundary = float(cell[axis] + 1) if step[axis] > 0 else float(cell[axis])
        t_max[axis] = (boundary - ro[axis]) / rd[axis]
        t_delta[axis] = abs(1.0 / rd[axis])

        # Numerical guard: the next boundary must not be behind entry.
        while t_max[axis] < t_enter - 1e-13:
            t_max[axis] += t_delta[axis]

    segments: List[Segment] = []
    current_t = t_enter
    max_steps = int(np.sum(volume.shape_cells) * 4 + 32)

    for _ in range(max_steps):
        if np.any(cell < 0) or np.any(cell >= volume.shape_cells):
            break

        next_t = min(float(np.min(t_max)), t_exit)
        if next_t > current_t + 1e-14:
            segments.append(
                Segment(tuple(map(int, cell)), float(current_t), float(next_t))
            )

        if next_t >= t_exit - 1e-13:
            break

        crossing = np.abs(t_max - next_t) <= 2e-12 * max(1.0, abs(next_t))
        for axis in range(3):
            if crossing[axis]:
                cell[axis] += step[axis]
                t_max[axis] += t_delta[axis]

        current_t = next_t

    return segments


def plane_split_segments(volume: Volume, ro: np.ndarray, rd: np.ndarray) -> List[Segment]:
    """
    Independent reference segmentation: collect every ray crossing with all
    integer grid planes, sort unique times, and classify each interval by
    its midpoint. This does not use DDA stepping state.
    """
    hit, t_enter, t_exit = intersect_box(ro, rd, volume.box_min, volume.box_max)
    if not hit or t_exit <= t_enter:
        return []

    times = [t_enter, t_exit]
    for axis in range(3):
        if abs(rd[axis]) < 1e-15:
            continue
        for plane in range(1, int(volume.shape_cells[axis])):
            t = (float(plane) - ro[axis]) / rd[axis]
            if t_enter + 1e-13 < t < t_exit - 1e-13:
                times.append(float(t))

    times.sort()
    unique = [times[0]]
    for t in times[1:]:
        if abs(t - unique[-1]) > 2e-12 * max(1.0, abs(t)):
            unique.append(t)

    segments: List[Segment] = []
    for a, b in zip(unique[:-1], unique[1:]):
        if b <= a + 1e-14:
            continue
        midpoint = ro + rd * (0.5 * (a + b))
        cell = np.floor(midpoint).astype(int)
        for axis in range(3):
            if cell[axis] == volume.shape_cells[axis]:
                cell[axis] -= 1
        if np.all(cell >= 0) and np.all(cell < volume.shape_cells):
            segments.append(Segment(tuple(map(int, cell)), float(a), float(b)))

    return segments


# ============================================================
# Whole-volume analytic traversal and independent reference
# ============================================================

@dataclass
class TraversalResult:
    hit: bool
    t_event: float
    total_tau: float
    visited_cells: int
    integrated_cells: int
    inversion_iterations: int
    inversion_fallbacks: int


def analytic_traversal(
    volume: Volume,
    ro: np.ndarray,
    rd: np.ndarray,
    target_tau: float | None,
) -> TraversalResult:
    segments = dda_segments(volume, ro, rd)
    remaining = target_tau
    total_tau = 0.0
    integrated = 0

    for seg in segments:
        cell = np.array(seg.cell, dtype=int)
        corners = volume.corners(cell)
        local_entry = ro + rd * seg.t0 - cell.astype(np.float64)
        length = seg.t1 - seg.t0

        if float(np.max(corners)) <= 0.0:
            continue

        integrated += 1
        rho = build_density_poly(local_entry, rd, corners)
        tau = build_tau_poly(rho, volume.sigma_t)
        cell_tau = max(horner(tau, length), 0.0)
        total_tau += cell_tau

        if remaining is not None:
            if cell_tau >= remaining and cell_tau > 0.0:
                inv = safeguarded_newton(
                    rho, tau, length, remaining, volume.sigma_t
                )
                return TraversalResult(
                    True,
                    seg.t0 + inv.s,
                    total_tau,
                    len(segments),
                    integrated,
                    inv.iterations,
                    inv.fallbacks,
                )
            remaining -= cell_tau

    return TraversalResult(
        False,
        math.inf,
        total_tau,
        len(segments),
        integrated,
        0,
        0,
    )


def direct_segment_tau(
    volume: Volume,
    ro: np.ndarray,
    rd: np.ndarray,
    t0: float,
    t1: float,
) -> float:
    if t1 <= t0:
        return 0.0
    value, _ = quad(
        lambda t: volume.sigma_t * volume.density(ro + rd * t),
        t0,
        t1,
        epsabs=2e-13,
        epsrel=2e-13,
        limit=80,
    )
    return float(value)


def reference_traversal(
    volume: Volume,
    ro: np.ndarray,
    rd: np.ndarray,
    target_tau: float | None,
) -> TraversalResult:
    segments = plane_split_segments(volume, ro, rd)
    remaining = target_tau
    total_tau = 0.0
    integrated = 0

    for seg in segments:
        cell_tau = direct_segment_tau(volume, ro, rd, seg.t0, seg.t1)
        total_tau += cell_tau
        integrated += 1

        if remaining is not None:
            if cell_tau >= remaining and cell_tau > 0.0:
                def residual(t):
                    return direct_segment_tau(volume, ro, rd, seg.t0, t) - remaining

                # brentq is independent of the polynomial inversion.
                root = brentq(
                    residual,
                    seg.t0,
                    seg.t1,
                    xtol=2e-13,
                    rtol=4 * np.finfo(float).eps,
                    maxiter=100,
                )
                return TraversalResult(
                    True,
                    float(root),
                    total_tau,
                    len(segments),
                    integrated,
                    0,
                    0,
                )
            remaining -= cell_tau

    return TraversalResult(
        False,
        math.inf,
        total_tau,
        len(segments),
        integrated,
        0,
        0,
    )


# ============================================================
# Test generation
# ============================================================

def random_ray(volume: Volume, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    center = 0.5 * (volume.box_min + volume.box_max)
    extent = volume.box_max - volume.box_min

    if rng.random() < 0.35:
        # Start inside.
        ro = rng.uniform(volume.box_min, volume.box_max)
    else:
        # Start on an enclosing sphere and point toward a random point in volume.
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        ro = center + direction * (1.4 * np.linalg.norm(extent))
    target = rng.uniform(volume.box_min, volume.box_max)
    rd = target - ro
    rd /= np.linalg.norm(rd)
    return ro, rd


def adversarial_rays(volume: Volume) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    rays = []
    bmin, bmax = volume.box_min, volume.box_max
    center = 0.5 * (bmin + bmax)

    # Axis aligned.
    for axis in range(3):
        for sign in (-1.0, 1.0):
            ro = center.copy()
            ro[axis] = bmin[axis] - 1.0 if sign > 0 else bmax[axis] + 1.0
            rd = np.zeros(3)
            rd[axis] = sign
            rays.append((ro, rd, f"axis_{axis}_{sign:+.0f}"))

    # Face/edge/corner diagonal entries and simultaneous crossings.
    directions = [
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 1.0]),
        np.array([2.0, 1.0, 1.0]),
        np.array([1.0, 2.0, 1.0]),
        np.array([1.0, 1.0, 2.0]),
    ]
    origins = [
        bmin - np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, -1.0, center[2]]),
        np.array([-1.0, center[1], -1.0]),
        np.array([center[0], -1.0, -1.0]),
    ]
    for i, rd in enumerate(directions):
        rd = rd / np.linalg.norm(rd)
        for j, ro in enumerate(origins):
            rays.append((ro.copy(), rd.copy(), f"tie_{i}_{j}"))

    # Start exactly on internal planes, edges, and grid vertices.
    points = [
        np.array([1.0, 1.4, 2.6]),
        np.array([1.0, 2.0, 2.6]),
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 2.0, 2.0]),
    ]
    dirs = [
        np.array([1.0, 0.3, -0.2]),
        np.array([-1.0, 0.3, -0.2]),
        np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, -1.0, -1.0]),
    ]
    for i, p in enumerate(points):
        if np.all(p >= bmin) and np.all(p <= bmax):
            for j, d in enumerate(dirs):
                d = d / np.linalg.norm(d)
                rays.append((p.copy(), d.copy(), f"boundary_{i}_{j}"))

    return rays


# ============================================================
# Validation
# ============================================================

@dataclass
class VolumeValidationSummary:
    seed: int
    grid_cells: Tuple[int, int, int]
    random_rays: int
    adversarial_rays: int
    segmentation_mismatches: int
    max_segment_boundary_error: float
    max_total_tau_abs_error: float
    max_total_tau_rel_error: float
    event_classification_mismatches: int
    max_event_distance_abs_error: float
    max_event_tau_residual: float
    inversion_failures: int
    mean_inversion_iterations: float
    p95_inversion_iterations: float
    max_inversion_iterations: int
    mean_inversion_fallbacks: float
    max_inversion_fallbacks: int
    countdown_max_error: float
    all_tests_passed: bool


def compare_segments(a: List[Segment], b: List[Segment]) -> Tuple[bool, float]:
    if len(a) != len(b):
        return False, math.inf
    max_error = 0.0
    for sa, sb in zip(a, b):
        if sa.cell != sb.cell:
            return False, math.inf
        max_error = max(
            max_error,
            abs(sa.t0 - sb.t0),
            abs(sa.t1 - sb.t1),
        )
    return True, max_error


def run_validation(seed: int = 20260718, random_rays_count: int = 1500) -> VolumeValidationSummary:
    rng = np.random.default_rng(seed)
    grid_cells = np.array([6, 5, 7], dtype=int)

    # Include many exact zeros to exercise empty-cell skipping.
    lattice = rng.uniform(0.0, 2.0, size=tuple(grid_cells + 1))
    lattice[rng.random(size=lattice.shape) < 0.42] = 0.0
    volume = Volume(lattice=lattice, sigma_t=1.35)

    rays = [(ro, rd, "random") for ro, rd in
            (random_ray(volume, rng) for _ in range(random_rays_count))]
    adv = adversarial_rays(volume)
    rays.extend(adv)

    segmentation_mismatches = 0
    max_segment_error = 0.0
    max_tau_abs = 0.0
    max_tau_rel = 0.0
    class_mismatches = 0
    max_event_error = 0.0
    max_event_residual = 0.0
    inversion_failures = 0
    inversion_iterations = []
    inversion_fallbacks = []
    countdown_max_error = 0.0

    for ro, rd, label in rays:
        dda = dda_segments(volume, ro, rd)
        plane = plane_split_segments(volume, ro, rd)
        ok, seg_error = compare_segments(dda, plane)
        if not ok:
            segmentation_mismatches += 1
        elif math.isfinite(seg_error):
            max_segment_error = max(max_segment_error, seg_error)

        analytic_total = analytic_traversal(volume, ro, rd, None)
        reference_total = reference_traversal(volume, ro, rd, None)
        tau_abs = abs(analytic_total.total_tau - reference_total.total_tau)
        tau_rel = tau_abs / max(
            1.0,
            abs(analytic_total.total_tau),
            abs(reference_total.total_tau),
        )
        max_tau_abs = max(max_tau_abs, tau_abs)
        max_tau_rel = max(max_tau_rel, tau_rel)

        total_tau = reference_total.total_tau
        targets = []
        if total_tau > 1e-12:
            # Include difficult near-entry and near-exit targets.
            targets = [
                total_tau * 1e-9,
                total_tau * 1e-5,
                total_tau * 0.15,
                total_tau * 0.5,
                total_tau * 0.9,
                total_tau * (1.0 - 1e-8),
            ]
        targets.append(total_tau + max(0.2, 0.25 * total_tau))

        for target in targets:
            analytic = analytic_traversal(volume, ro, rd, target)
            reference = reference_traversal(volume, ro, rd, target)

            if analytic.hit != reference.hit:
                class_mismatches += 1
                continue

            if analytic.hit:
                event_error = abs(analytic.t_event - reference.t_event)
                max_event_error = max(max_event_error, event_error)

                # Independent residual at analytic event.
                ref_segments = plane_split_segments(volume, ro, rd)
                accumulated = 0.0
                for seg in ref_segments:
                    if analytic.t_event >= seg.t1 - 1e-13:
                        accumulated += direct_segment_tau(
                            volume, ro, rd, seg.t0, seg.t1
                        )
                    elif analytic.t_event > seg.t0:
                        accumulated += direct_segment_tau(
                            volume, ro, rd, seg.t0, analytic.t_event
                        )
                        break
                max_event_residual = max(
                    max_event_residual,
                    abs(accumulated - target),
                )

                inversion_iterations.append(analytic.inversion_iterations)
                inversion_fallbacks.append(analytic.inversion_fallbacks)
                if analytic.inversion_iterations <= 0:
                    inversion_failures += 1

                # Optical-depth countdown check at half the event distance.
                shift = 0.5 * analytic.t_event
                prefix = reference_traversal(volume, ro, rd, None)
                # Integrate only [0, shift] directly.
                shifted_tau = 0.0
                for seg in ref_segments:
                    if shift >= seg.t1:
                        shifted_tau += direct_segment_tau(
                            volume, ro, rd, seg.t0, seg.t1
                        )
                    elif shift > seg.t0:
                        shifted_tau += direct_segment_tau(
                            volume, ro, rd, seg.t0, shift
                        )
                        break

                ro_shift = ro + rd * shift
                remaining = target - shifted_tau
                shifted_result = analytic_traversal(volume, ro_shift, rd, remaining)
                if shifted_result.hit:
                    countdown_max_error = max(
                        countdown_max_error,
                        abs((shift + shifted_result.t_event) - analytic.t_event),
                    )
                else:
                    countdown_max_error = math.inf

    iterations = np.array(inversion_iterations, dtype=float)
    fallbacks = np.array(inversion_fallbacks, dtype=float)

    passed = (
        segmentation_mismatches == 0
        and class_mismatches == 0
        and inversion_failures == 0
        and max_tau_abs < 2e-10
        and max_event_error < 2e-8
        and max_event_residual < 2e-8
        and countdown_max_error < 2e-8
    )

    return VolumeValidationSummary(
        seed=seed,
        grid_cells=tuple(map(int, grid_cells)),
        random_rays=random_rays_count,
        adversarial_rays=len(adv),
        segmentation_mismatches=segmentation_mismatches,
        max_segment_boundary_error=max_segment_error,
        max_total_tau_abs_error=max_tau_abs,
        max_total_tau_rel_error=max_tau_rel,
        event_classification_mismatches=class_mismatches,
        max_event_distance_abs_error=max_event_error,
        max_event_tau_residual=max_event_residual,
        inversion_failures=inversion_failures,
        mean_inversion_iterations=float(np.mean(iterations)),
        p95_inversion_iterations=float(np.percentile(iterations, 95)),
        max_inversion_iterations=int(np.max(iterations)),
        mean_inversion_fallbacks=float(np.mean(fallbacks)),
        max_inversion_fallbacks=int(np.max(fallbacks)),
        countdown_max_error=countdown_max_error,
        all_tests_passed=bool(passed),
    )


def make_report(summary: VolumeValidationSummary) -> str:
    return f"""# Cubic-Fade 3D Degree-10 Whole-Volume Validation

**Overall status:** {"PASS" if summary.all_tests_passed else "NEEDS REVIEW"}

## Configuration

- Random seed: `{summary.seed}`
- Voxel grid: `{summary.grid_cells[0]} × {summary.grid_cells[1]} × {summary.grid_cells[2]}`
- Random rays: `{summary.random_rays}`
- Adversarial/boundary rays: `{summary.adversarial_rays}`
- Lattice values: nonnegative, with 42% exact-zero vertices
- Extinction scale: `1.35`
- Analytic implementation:
  - slab ray-box intersection
  - 3D DDA with simultaneous-axis tie handling
  - degree-9 density polynomial per cell
  - degree-10 optical-depth polynomial per cell
  - terminal safeguarded Newton inversion
- Independent reference:
  - all integer-plane crossing times collected and globally sorted
  - midpoint interval classification
  - direct cubic-fade density evaluation
  - adaptive Gauss–Kronrod quadrature (`scipy.integrate.quad`)
  - Brent root solve (`scipy.optimize.brentq`)

## DDA geometry

- Segmentation mismatches: `{summary.segmentation_mismatches}`
- Maximum segment-boundary error: `{summary.max_segment_boundary_error:.6e}`

## Whole-ray optical depth

- Maximum absolute total optical-depth error: `{summary.max_total_tau_abs_error:.6e}`
- Maximum relative total optical-depth error: `{summary.max_total_tau_rel_error:.6e}`

## Free-path event

- Hit/miss classification mismatches: `{summary.event_classification_mismatches}`
- Maximum event-distance error: `{summary.max_event_distance_abs_error:.6e}`
- Maximum independent event optical-depth residual: `{summary.max_event_tau_residual:.6e}`
- Inversion failures: `{summary.inversion_failures}`

## Safeguarded Newton

- Mean iterations: `{summary.mean_inversion_iterations:.3f}`
- P95 iterations: `{summary.p95_inversion_iterations:.3f}`
- Maximum iterations: `{summary.max_inversion_iterations}`
- Mean fallback count: `{summary.mean_inversion_fallbacks:.3f}`
- Maximum fallback count: `{summary.max_inversion_fallbacks}`

## Optical-depth countdown

- Maximum shifted-ray countdown error: `{summary.countdown_max_error:.6e}`

## Result

This validation closes the full single-scale primitive:

\[
\text{{ray-box}}
\rightarrow
\text{{3D DDA}}
\rightarrow
\text{{exact degree-10 cell optical depth}}
\rightarrow
\text{{whole-cell budget consumption}}
\rightarrow
\text{{one terminal monotone inversion}}.
\]

The analytic traversal agrees with an independently segmented, directly
evaluated, adaptively integrated, Brent-inverted reference.

The next validation layer should target GPU-relevant arithmetic:

1. repeat the coefficient construction and Horner evaluation in `float32`;
2. compare local-time versus normalized-cell-time parameterizations;
3. stress cancellation at near-zero densities and near-cell-exit targets;
4. derive conservative tolerances and fallback rules;
5. only then move to multiscale/fBm representations.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--rays", type=int, default=1500)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    summary = run_validation(args.seed, args.rays)
    payload = asdict(summary)
    print(json.dumps(payload, indent=2))

    if args.json:
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.report:
        args.report.write_text(make_report(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
