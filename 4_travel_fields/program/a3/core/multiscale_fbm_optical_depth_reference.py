
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np
from scipy.integrate import quad


def fade(x):
    return x * x * (3.0 - 2.0 * x)


@dataclass
class Octave:
    lattice: np.ndarray
    frequency: int
    amplitude: float

    @property
    def cells(self):
        return np.array(self.lattice.shape, dtype=int) - 1

    def density(self, p_world: np.ndarray) -> float:
        q = p_world * float(self.frequency)
        cell = np.floor(q).astype(int)

        for a in range(3):
            if cell[a] == self.cells[a]:
                cell[a] -= 1

        if np.any(cell < 0) or np.any(cell >= self.cells):
            return 0.0

        f = np.clip(q - cell, 0.0, 1.0)
        w = fade(f)
        x, y, z = map(int, cell)
        c = self.lattice[x:x+2, y:y+2, z:z+2]

        x00 = (1-w[0])*c[0,0,0] + w[0]*c[1,0,0]
        x10 = (1-w[0])*c[0,1,0] + w[0]*c[1,1,0]
        x01 = (1-w[0])*c[0,0,1] + w[0]*c[1,0,1]
        x11 = (1-w[0])*c[0,1,1] + w[0]*c[1,1,1]
        y0 = (1-w[1])*x00 + w[1]*x10
        y1 = (1-w[1])*x01 + w[1]*x11
        return float(self.amplitude * ((1-w[2])*y0 + w[2]*y1))


@dataclass
class MultiScaleMedium:
    octaves: List[Octave]
    sigma_t: float

    def density(self, p: np.ndarray) -> float:
        return sum(o.density(p) for o in self.octaves)

    @property
    def box_min(self):
        return np.zeros(3)

    @property
    def box_max(self):
        # All octave lattices are constructed over the same world domain.
        base = self.octaves[0]
        return base.cells.astype(float) / float(base.frequency)


def intersect_box(ro, rd, bmin, bmax):
    t0 = -math.inf
    t1 = math.inf
    for a in range(3):
        if abs(rd[a]) < 1e-15:
            if ro[a] < bmin[a] or ro[a] > bmax[a]:
                return False, 0.0, 0.0
            continue
        s0 = (bmin[a] - ro[a]) / rd[a]
        s1 = (bmax[a] - ro[a]) / rd[a]
        t0 = max(t0, min(s0, s1))
        t1 = min(t1, max(s0, s1))
    t0 = max(t0, 0.0)
    return t1 >= t0, t0, t1


def octave_plane_times(octave: Octave, ro, rd, t_enter, t_exit):
    times = []
    freq = octave.frequency
    world_max = octave.cells.astype(float) / float(freq)

    for axis in range(3):
        if abs(rd[axis]) < 1e-15:
            continue
        # Internal octave cell planes in world coordinates.
        for k in range(1, int(octave.cells[axis])):
            plane = float(k) / float(freq)
            if plane <= 0.0 or plane >= world_max[axis]:
                continue
            t = (plane - ro[axis]) / rd[axis]
            if t_enter + 1e-13 < t < t_exit - 1e-13:
                times.append(float(t))
    return times


def union_segments(medium: MultiScaleMedium, ro, rd):
    hit, t_enter, t_exit = intersect_box(
        ro, rd, medium.box_min, medium.box_max
    )
    if not hit or t_exit <= t_enter:
        return []

    times = [t_enter, t_exit]
    for octave in medium.octaves:
        times.extend(octave_plane_times(octave, ro, rd, t_enter, t_exit))

    times.sort()
    unique = [times[0]]
    for t in times[1:]:
        if abs(t - unique[-1]) > 2e-12 * max(1.0, abs(t)):
            unique.append(t)

    return list(zip(unique[:-1], unique[1:]))


_GLX, _GLW = np.polynomial.legendre.leggauss(8)


def gl_integrate(func, a, b):
    if b <= a:
        return 0.0
    nodes = 0.5 * (b - a) * (_GLX + 1.0) + a
    values = np.array([func(float(t)) for t in nodes], dtype=np.float64)
    return float(0.5 * (b - a) * np.dot(_GLW, values))


def direct_tau(medium, ro, rd, a, b):
    # Split at the common refinement; on each segment every octave is a
    # degree<=9 polynomial along the ray, so 8-point GL is exact in exact arithmetic.
    total = 0.0
    for s0, s1 in union_segments(medium, ro, rd):
        total += gl_integrate(
            lambda t: medium.sigma_t * medium.density(ro + rd*t),
            s0, s1
        )
    return total


def separate_octave_tau(medium, ro, rd, a, b):
    total = 0.0
    for octave in medium.octaves:
        # Split only at this octave's own cell boundaries.
        times = [a, b]
        times.extend(octave_plane_times(octave, ro, rd, a, b))
        times = sorted(times)
        unique = [times[0]]
        for t in times[1:]:
            if abs(t - unique[-1]) > 2e-12 * max(1.0, abs(t)):
                unique.append(t)

        for s0, s1 in zip(unique[:-1], unique[1:]):
            total += gl_integrate(
                lambda t: medium.sigma_t * octave.density(ro + rd*t),
                s0, s1
            )
    return total


def hierarchical_tau(medium, ro, rd):
    total = 0.0
    segments = union_segments(medium, ro, rd)
    for a, b in segments:
        total += gl_integrate(
            lambda t: medium.sigma_t * medium.density(ro + rd*t),
            a, b
        )
    return total, len(segments)

def coarse_bound_for_octave(octave: Octave, pmin, pmax):
    """
    Conservative density bound over an axis-aligned world box:
    max of all lattice vertices whose interpolation support can overlap box.
    Cubic-fade value noise is a convex combination of its 8 corners.
    """
    qmin = np.floor(pmin * octave.frequency).astype(int)
    qmax = np.floor(pmax * octave.frequency).astype(int)
    qmin = np.maximum(qmin, 0)
    qmax = np.minimum(qmax + 1, octave.cells)

    sl = tuple(slice(int(qmin[a]), int(qmax[a]) + 1) for a in range(3))
    return float(octave.amplitude * np.max(octave.lattice[sl]))


def random_ray(medium, rng):
    center = 0.5 * (medium.box_min + medium.box_max)
    extent = medium.box_max - medium.box_min
    if rng.random() < 0.35:
        ro = rng.uniform(medium.box_min, medium.box_max)
    else:
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        ro = center + d * (1.5 * np.linalg.norm(extent))
    target = rng.uniform(medium.box_min, medium.box_max)
    rd = target - ro
    rd /= np.linalg.norm(rd)
    return ro, rd


@dataclass
class Summary:
    seed: int
    octaves: int
    frequencies: List[int]
    amplitudes: List[float]
    rays: int
    max_linearity_error: float
    max_hierarchy_error: float
    max_hierarchy_segments: int
    mean_hierarchy_segments: float
    max_density_bound_violation: float
    all_tests_passed: bool


def run(seed=20260718, rays=300):
    rng = np.random.default_rng(seed)
    world_cells = np.array([4, 4, 4])

    frequencies = [1, 2, 4]
    amplitudes = [1.0, 0.5, 0.25]
    octaves = []

    for f, a in zip(frequencies, amplitudes):
        cells = world_cells * f
        lattice = rng.uniform(0.0, 1.0, size=tuple(cells + 1))
        lattice[rng.random(size=lattice.shape) < 0.35] = 0.0
        octaves.append(Octave(lattice=lattice, frequency=f, amplitude=a))

    medium = MultiScaleMedium(octaves=octaves, sigma_t=1.3)

    linearity_errors = []
    hierarchy_errors = []
    segment_counts = []
    bound_violations = []

    for _ in range(rays):
        ro, rd = random_ray(medium, rng)
        hit, a, b = intersect_box(ro, rd, medium.box_min, medium.box_max)
        if not hit or b <= a:
            continue

        tau_direct = direct_tau(medium, ro, rd, a, b)
        tau_separate = separate_octave_tau(medium, ro, rd, a, b)
        tau_hierarchy, count = hierarchical_tau(medium, ro, rd)

        linearity_errors.append(abs(tau_direct - tau_separate))
        hierarchy_errors.append(abs(tau_direct - tau_hierarchy))
        segment_counts.append(count)

        # Test convex-hull density bounds on random subsegments.
        for _ in range(3):
            u0, u1 = sorted(rng.uniform(a, b, size=2))
            p0 = ro + rd*u0
            p1 = ro + rd*u1
            pmin = np.minimum(p0, p1)
            pmax = np.maximum(p0, p1)

            for octave in medium.octaves:
                bound = coarse_bound_for_octave(octave, pmin, pmax)
                ts = np.linspace(u0, u1, 25)
                actual = max(octave.density(ro + rd*t) for t in ts)
                bound_violations.append(max(0.0, actual - bound))

    max_lin = float(np.max(linearity_errors))
    max_hier = float(np.max(hierarchy_errors))
    max_bound = float(np.max(bound_violations))

    return Summary(
        seed=seed,
        octaves=len(octaves),
        frequencies=frequencies,
        amplitudes=amplitudes,
        rays=len(linearity_errors),
        max_linearity_error=max_lin,
        max_hierarchy_error=max_hier,
        max_hierarchy_segments=int(np.max(segment_counts)),
        mean_hierarchy_segments=float(np.mean(segment_counts)),
        max_density_bound_violation=max_bound,
        all_tests_passed=(
            max_lin < 2e-9 and
            max_hier < 2e-9 and
            max_bound < 1e-12
        ),
    )


def make_report(s: Summary) -> str:
    status = "PASS" if s.all_tests_passed else "NEEDS REVIEW"
    return (
        "# Multiscale Cubic-Fade / fBm Optical-Depth Pilot Study\n\n"
        f"**Status:** {status}\n\n"
        "## Configuration\n\n"
        f"- Octaves: `{s.octaves}`\n"
        f"- Frequencies: `{s.frequencies}`\n"
        f"- Amplitudes: `{s.amplitudes}`\n"
        f"- Random rays: `{s.rays}`\n\n"
        "## Exact additive decomposition\n\n"
        "For a nonnegative multiscale density formed as a weighted sum of octaves, "
        "optical depth is exactly the sum of the per-octave optical depths by "
        "linearity of integration.\n\n"
        f"- Maximum direct-vs-separate-octave error: `{s.max_linearity_error:.6e}`\n\n"
        "## Common-refinement hierarchy\n\n"
        "All cell-boundary crossing times from all octaves were merged into one "
        "sorted partition. Each common-refinement segment lies inside one cell "
        "of every octave.\n\n"
        f"- Maximum hierarchy error: `{s.max_hierarchy_error:.6e}`\n"
        f"- Mean hierarchy segments: `{s.mean_hierarchy_segments:.3f}`\n"
        f"- Maximum hierarchy segments: `{s.max_hierarchy_segments}`\n\n"
        "## Conservative octave bounds\n\n"
        "Cubic-fade value noise is a convex combination of its eight corner "
        "values, so the maximum overlapping lattice vertex provides a "
        "conservative density upper bound.\n\n"
        f"- Maximum sampled bound violation: `{s.max_density_bound_violation:.6e}`\n\n"
        "## Interpretation\n\n"
        "1. Optical depth is exactly additive across octaves.\n"
        "2. Common refinement is exact but tends toward finest-grid traversal cost.\n"
        "3. Conservative per-octave bounds can support hierarchical pruning.\n"
        "4. The next problem is avoiding evaluation of every octave on every "
        "finest-level segment.\n"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--rays", type=int, default=300)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    summary = run(args.seed, args.rays)
    print(json.dumps(asdict(summary), indent=2))

    if args.json:
        args.json.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    if args.report:
        args.report.write_text(make_report(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
