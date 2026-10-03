from __future__ import annotations

"""A3-GN exact reference kernel.

Architecture
------------
1. Streaming single finest-grid DDA.
2. Integer plane indices are authoritative.
3. Each crossing emits axis-specific octave masks.
4. Complete common segments are integrated exactly with GL5.
5. The terminal segment builds one combined degree-9 density polynomial.
6. A degree-10 optical-depth polynomial is inverted by safeguarded Newton.

The kernel is intentionally NumPy/CPU reference code. It is organized so a
GPU implementation can preserve the same state machine and invariants.
"""

from dataclasses import dataclass
import math
from typing import Iterator, Sequence

import numpy as np

GL5_X, GL5_W = np.polynomial.legendre.leggauss(5)


@dataclass(frozen=True)
class Octave:
    lattice: np.ndarray
    frequency: int
    amplitude: float

    def __post_init__(self) -> None:
        if self.lattice.ndim != 3:
            raise ValueError("lattice must be 3-D")
        if self.frequency <= 0:
            raise ValueError("frequency must be positive")
        if np.any(self.lattice < 0):
            raise ValueError("A3 monotone optical-depth reference expects nonnegative corners")

    @property
    def cells(self) -> np.ndarray:
        return np.asarray(self.lattice.shape, dtype=np.int64) - 1

    def corners(self, cell: np.ndarray) -> np.ndarray:
        x, y, z = map(int, cell)
        return self.amplitude * self.lattice[x:x+2, y:y+2, z:z+2]


@dataclass(frozen=True)
class Medium:
    octaves: tuple[Octave, ...]
    sigma_t: float
    world_cells: np.ndarray

    def __post_init__(self) -> None:
        if not self.octaves:
            raise ValueError("medium requires at least one octave")
        if self.sigma_t < 0:
            raise ValueError("sigma_t must be nonnegative")
        world = np.asarray(self.world_cells, dtype=np.int64)
        if world.shape != (3,) or np.any(world <= 0):
            raise ValueError("world_cells must be a positive length-3 vector")
        freqs = [o.frequency for o in self.octaves]
        F = max(freqs)
        if any(F % f != 0 or ((F // f) & ((F // f) - 1)) != 0 for f in freqs):
            raise ValueError("frequencies must form a dyadic hierarchy under the finest frequency")
        for o in self.octaves:
            expected = world * o.frequency
            if np.any(o.cells != expected):
                raise ValueError(
                    f"octave frequency {o.frequency} has cells {o.cells}, expected {expected}"
                )
        object.__setattr__(self, "world_cells", world)

    @property
    def bmin(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float64)

    @property
    def bmax(self) -> np.ndarray:
        return self.world_cells.astype(np.float64)

    @property
    def finest_frequency(self) -> int:
        return max(o.frequency for o in self.octaves)


@dataclass
class OctaveState:
    level: int
    cell: np.ndarray
    corners: np.ndarray


@dataclass(frozen=True)
class CrossingTransaction:
    t: float
    axis_masks: tuple[int, int, int]
    plane_indices: tuple[int | None, int | None, int | None]

    @property
    def combined_mask(self) -> int:
        return self.axis_masks[0] | self.axis_masks[1] | self.axis_masks[2]


@dataclass(frozen=True)
class TransportResult:
    hit: bool
    t_entry: float
    t_exit: float
    total_tau: float
    event_t: float | None
    segments: int
    octave_reloads: int
    terminal_newton_iterations: int
    terminal_newton_fallbacks: int


def intersect_box(
    ro: np.ndarray,
    rd: np.ndarray,
    bmin: np.ndarray,
    bmax: np.ndarray,
) -> tuple[bool, float, float]:
    te, tx = -math.inf, math.inf
    for axis in range(3):
        if abs(float(rd[axis])) < 1e-15:
            if ro[axis] < bmin[axis] or ro[axis] > bmax[axis]:
                return False, 0.0, 0.0
            continue
        t0 = (bmin[axis] - ro[axis]) / rd[axis]
        t1 = (bmax[axis] - ro[axis]) / rd[axis]
        te = max(te, float(min(t0, t1)))
        tx = min(tx, float(max(t0, t1)))
    te = max(te, 0.0)
    return tx >= te, te, tx


def _fade(u: np.ndarray) -> np.ndarray:
    return u * u * (3.0 - 2.0 * u)


def density_from_corners(local: np.ndarray, corners: np.ndarray) -> float:
    f = _fade(local)
    x00 = (1-f[0])*corners[0,0,0] + f[0]*corners[1,0,0]
    x10 = (1-f[0])*corners[0,1,0] + f[0]*corners[1,1,0]
    x01 = (1-f[0])*corners[0,0,1] + f[0]*corners[1,0,1]
    x11 = (1-f[0])*corners[0,1,1] + f[0]*corners[1,1,1]
    y0 = (1-f[1])*x00 + f[1]*x10
    y1 = (1-f[1])*x01 + f[1]*x11
    return float((1-f[2])*y0 + f[2]*y1)


def _direction_owned_cell(octave: Octave, ro: np.ndarray, rd: np.ndarray, t: float) -> np.ndarray:
    q = (ro + rd*t) * octave.frequency
    nearest = np.rint(q)
    close = np.abs(q-nearest) <= 2e-11*np.maximum(1.0, np.abs(q))
    q = np.where(close, q + np.sign(rd)*1e-10, q)
    cell = np.floor(q).astype(np.int64)
    return np.minimum(np.maximum(cell, 0), octave.cells-1)


def initialize_states(medium: Medium, ro: np.ndarray, rd: np.ndarray, t_entry: float) -> list[OctaveState]:
    states: list[OctaveState] = []
    for level, octave in enumerate(medium.octaves):
        cell = _direction_owned_cell(octave, ro, rd, t_entry)
        states.append(OctaveState(level, cell, octave.corners(cell)))
    return states


def octave_mask_for_plane(k: int, frequencies: Sequence[int], finest: int) -> int:
    mask = 0
    for level, frequency in enumerate(frequencies):
        stride = finest // frequency
        if k % stride == 0:
            mask |= 1 << level
    return mask


def streaming_transactions(
    medium: Medium,
    ro: np.ndarray,
    rd: np.ndarray,
    t_entry: float,
    t_exit: float,
) -> Iterator[CrossingTransaction]:
    """Yield entry, finest-grid crossings, and exit without materializing an event list.

    Invariant: integer next-plane indices are authoritative. Boundary times are
    recomputed from those indices; they are never advanced by cumulative addition.
    """

    frequencies = [o.frequency for o in medium.octaves]
    F = medium.finest_frequency
    world_fine = medium.world_cells * F

    q = (ro + rd*t_entry) * F
    nearest = np.rint(q)
    close = np.abs(q-nearest) <= 2e-11*np.maximum(1.0, np.abs(q))
    q = np.where(close, q + np.sign(rd)*1e-10, q)
    cell = np.floor(q).astype(np.int64)
    cell = np.minimum(np.maximum(cell, 0), world_fine-1)

    step = np.sign(rd).astype(np.int64)
    next_k = np.zeros(3, dtype=np.int64)
    t_max = np.full(3, math.inf, dtype=np.float64)

    for axis in range(3):
        if step[axis] == 0:
            continue
        next_k[axis] = cell[axis] + 1 if step[axis] > 0 else cell[axis]
        while True:
            plane = next_k[axis] / float(F)
            t_max[axis] = (plane - ro[axis]) / rd[axis]
            if t_max[axis] > t_entry + 1e-13:
                break
            next_k[axis] += step[axis]

    yield CrossingTransaction(t_entry, (0, 0, 0), (None, None, None))

    max_steps = int(np.sum(world_fine) * 4 + 64)
    for _ in range(max_steps):
        t_next = float(np.min(t_max))
        if not math.isfinite(t_next) or t_next >= t_exit - 1e-13:
            break

        tie_tol = 2e-12 * max(1.0, abs(t_next))
        axes = [axis for axis in range(3) if abs(t_max[axis]-t_next) <= tie_tol]
        axis_masks = [0, 0, 0]
        plane_indices: list[int | None] = [None, None, None]

        for axis in axes:
            k = int(next_k[axis])
            plane_indices[axis] = k
            if 0 < k < int(world_fine[axis]):
                axis_masks[axis] = octave_mask_for_plane(k, frequencies, F)

        yield CrossingTransaction(t_next, tuple(axis_masks), tuple(plane_indices))

        for axis in axes:
            cell[axis] += step[axis]
            next_k[axis] += step[axis]
            plane = next_k[axis] / float(F)
            t_max[axis] = (plane - ro[axis]) / rd[axis]
    else:
        raise RuntimeError("streaming DDA exceeded conservative step bound")

    yield CrossingTransaction(t_exit, (0, 0, 0), (None, None, None))


def commit_transaction(
    medium: Medium,
    states: list[OctaveState],
    transaction: CrossingTransaction,
    rd: np.ndarray,
) -> int:
    """Atomically commit all axis/octave cell changes for one crossing transaction."""

    step = np.sign(rd).astype(np.int64)
    reloads = 0
    for state in states:
        changed = False
        for axis in range(3):
            if transaction.axis_masks[axis] & (1 << state.level):
                state.cell[axis] += step[axis]
                changed = True
        if changed:
            octave = medium.octaves[state.level]
            state.cell = np.minimum(np.maximum(state.cell, 0), octave.cells-1)
            state.corners = octave.corners(state.cell)
            reloads += 1
    return reloads


def evaluate_cached_density(
    medium: Medium,
    states: Sequence[OctaveState],
    p: np.ndarray,
) -> float:
    total = 0.0
    for state in states:
        octave = medium.octaves[state.level]
        local = p*octave.frequency - state.cell
        total += density_from_corners(local, state.corners)
    return total


def integrate_segment_gl5(
    medium: Medium,
    states: Sequence[OctaveState],
    ro: np.ndarray,
    rd: np.ndarray,
    a: float,
    b: float,
) -> float:
    if b <= a:
        return 0.0
    nodes = 0.5*(b-a)*(GL5_X+1.0)+a
    values = np.array(
        [evaluate_cached_density(medium, states, ro+rd*t) for t in nodes],
        dtype=np.float64,
    )
    return medium.sigma_t * 0.5*(b-a)*float(np.dot(GL5_W, values))


def _fade_poly(a: float, b: float) -> np.ndarray:
    return np.array(
        [3*a*a-2*a*a*a, 6*a*b*(1-a), 3*b*b*(1-2*a), -2*b*b*b],
        dtype=np.float64,
    )


def _pad(poly: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.float64)
    out[:len(poly)] = poly
    return out


def density_polynomial(local_entry: np.ndarray, direction: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Ascending degree-9 density coefficients on a segment-local parameter s."""

    fx, fy, fz = [_fade_poly(local_entry[i], direction[i]) for i in range(3)]
    fxy = np.polynomial.polynomial.polymul(fx, fy)
    fxz = np.polynomial.polynomial.polymul(fx, fz)
    fyz = np.polynomial.polynomial.polymul(fy, fz)
    fxyz = np.polynomial.polynomial.polymul(fxy, fz)

    d000, d100, d010, d110, d001, d101, d011, d111 = (
        corners[0,0,0], corners[1,0,0], corners[0,1,0], corners[1,1,0],
        corners[0,0,1], corners[1,0,1], corners[0,1,1], corners[1,1,1],
    )
    ax = d100-d000
    ay = d010-d000
    az = d001-d000
    axy = d110-d100-d010+d000
    axz = d101-d100-d001+d000
    ayz = d011-d010-d001+d000
    axyz = d111-d110-d101-d011+d100+d010+d001-d000

    rho = np.zeros(10, dtype=np.float64)
    rho[0] = d000
    rho += ax*_pad(fx,10) + ay*_pad(fy,10) + az*_pad(fz,10)
    rho += axy*_pad(fxy,10) + axz*_pad(fxz,10) + ayz*_pad(fyz,10)
    rho += axyz*_pad(fxyz,10)
    return rho


def build_terminal_density_polynomial(
    medium: Medium,
    states: Sequence[OctaveState],
    ro: np.ndarray,
    rd: np.ndarray,
    segment_start: float,
) -> np.ndarray:
    p0 = ro + rd*segment_start
    combined = np.zeros(10, dtype=np.float64)
    for state in states:
        octave = medium.octaves[state.level]
        local = p0*octave.frequency - state.cell
        combined += density_polynomial(local, rd*octave.frequency, state.corners)
    return combined


def _integrated_coefficients(density_coeff: np.ndarray, sigma_t: float) -> np.ndarray:
    tau = np.zeros(len(density_coeff)+1, dtype=np.float64)
    for k, coeff in enumerate(density_coeff):
        tau[k+1] = sigma_t*coeff/(k+1)
    return tau


def _horner(coeff: np.ndarray, x: float) -> float:
    acc = 0.0
    for c in coeff[::-1]:
        acc = acc*x + float(c)
    return acc


def terminal_newton(
    density_coeff: np.ndarray,
    sigma_t: float,
    target_tau: float,
    segment_length: float,
    *,
    max_iterations: int = 12,
    spatial_tolerance: float = 1e-12,
) -> tuple[float, int, int]:
    """Invert the monotone degree-10 optical-depth clock on [0, segment_length]."""

    tau_coeff = _integrated_coefficients(density_coeff, sigma_t)
    tau_end = _horner(tau_coeff, segment_length)
    if target_tau <= 0:
        return 0.0, 0, 0
    if target_tau >= tau_end:
        return segment_length, 0, 0

    lo, hi = 0.0, segment_length
    x = segment_length * target_tau / max(tau_end, 1e-300)
    fallbacks = 0

    for iteration in range(1, max_iterations+1):
        tau_x = _horner(tau_coeff, x)
        derivative = sigma_t * max(_horner(density_coeff, x), 0.0)
        residual = tau_x-target_tau

        if residual < 0:
            lo = x
        else:
            hi = x

        spatial_residual = abs(residual)/max(derivative, 1e-14)
        if spatial_residual <= spatial_tolerance*max(1.0, segment_length):
            return x, iteration, fallbacks

        proposal = x-residual/derivative if derivative > 1e-14 else math.nan
        if not math.isfinite(proposal) or proposal <= lo or proposal >= hi:
            proposal = 0.5*(lo+hi)
            fallbacks += 1
        x = proposal

    return x, max_iterations, fallbacks


def trace(
    medium: Medium,
    ro: np.ndarray,
    rd: np.ndarray,
    *,
    target_tau: float | None = None,
) -> TransportResult:
    """Integrate total optical depth and optionally locate one free-path event."""

    ro = np.asarray(ro, dtype=np.float64)
    rd = np.asarray(rd, dtype=np.float64)
    norm = float(np.linalg.norm(rd))
    if norm == 0:
        raise ValueError("ray direction must be nonzero")
    rd = rd/norm

    hit, te, tx = intersect_box(ro, rd, medium.bmin, medium.bmax)
    if not hit or tx <= te:
        return TransportResult(False, te, tx, 0.0, None, 0, 0, 0, 0)

    transactions = streaming_transactions(medium, ro, rd, te, tx)
    current = next(transactions)
    states = initialize_states(medium, ro, rd, te)
    total_tau = 0.0
    remaining = target_tau
    segments = 0
    reloads = 0

    for nxt in transactions:
        a, b = current.t, nxt.t
        if b > a:
            segment_tau = integrate_segment_gl5(medium, states, ro, rd, a, b)
            segments += 1
            if remaining is not None and segment_tau >= remaining and segment_tau > 0:
                density_coeff = build_terminal_density_polynomial(medium, states, ro, rd, a)
                local_t, iterations, fallbacks = terminal_newton(
                    density_coeff,
                    medium.sigma_t,
                    remaining,
                    b-a,
                )
                return TransportResult(
                    True, te, tx, total_tau+remaining, a+local_t,
                    segments, reloads, iterations, fallbacks,
                )
            total_tau += segment_tau
            if remaining is not None:
                remaining -= segment_tau

        if nxt.t < tx:
            reloads += commit_transaction(medium, states, nxt, rd)
        current = nxt

    return TransportResult(True, te, tx, total_tau, None, segments, reloads, 0, 0)


def merged_stream_transactions(
    medium: Medium,
    ro: np.ndarray,
    rd: np.ndarray,
    t_entry: float,
    t_exit: float,
) -> list[CrossingTransaction]:
    """Independent multi-octave oracle used only by validation gates."""

    raw: list[tuple[float, int, int, int]] = []
    frequencies = [o.frequency for o in medium.octaves]
    F = medium.finest_frequency
    for level, octave in enumerate(medium.octaves):
        f = octave.frequency
        for axis in range(3):
            if abs(rd[axis]) < 1e-15:
                continue
            for k_local in range(1, int(medium.world_cells[axis]*f)):
                t = (k_local/f-ro[axis])/rd[axis]
                if t_entry+1e-13 < t < t_exit-1e-13:
                    k_fine = k_local*(F//f)
                    raw.append((float(t), axis, k_fine, 1 << level))
    raw.sort(key=lambda item: item[0])

    out = [CrossingTransaction(t_entry, (0,0,0), (None,None,None))]
    i = 0
    while i < len(raw):
        t0 = raw[i][0]
        axis_masks = [0,0,0]
        plane_indices: list[int | None] = [None,None,None]
        j = i
        while j < len(raw) and abs(raw[j][0]-t0) <= 2e-12*max(1.0, abs(t0)):
            _, axis, k_fine, mask = raw[j]
            axis_masks[axis] |= mask
            plane_indices[axis] = k_fine
            j += 1
        out.append(CrossingTransaction(t0, tuple(axis_masks), tuple(plane_indices)))
        i = j
    out.append(CrossingTransaction(t_exit, (0,0,0), (None,None,None)))
    return out


def make_demo_medium(seed: int = 20260718) -> Medium:
    rng = np.random.default_rng(seed)
    world = np.array([3,3,3], dtype=np.int64)
    frequencies = [1,2,4,8,16]
    amplitudes = [1.0,0.5,0.25,0.125,0.0625]
    octaves = []
    for f, a in zip(frequencies, amplitudes):
        lattice = rng.uniform(0.0, 1.0, size=tuple(world*f+1))
        lattice[rng.random(size=lattice.shape) < 0.35] = 0.0
        octaves.append(Octave(lattice, f, a))
    return Medium(tuple(octaves), 1.2, world)


__all__ = [
    "Octave", "Medium", "OctaveState", "CrossingTransaction", "TransportResult",
    "intersect_box", "streaming_transactions", "merged_stream_transactions",
    "commit_transaction", "integrate_segment_gl5", "build_terminal_density_polynomial",
    "terminal_newton", "trace", "make_demo_medium",
]
