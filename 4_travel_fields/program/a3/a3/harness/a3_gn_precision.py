"""Precision-parametrized A3-GN traversal for the float32 transport study (handoff Sec.18).

This mirrors the *state machine and invariants* of ``a3.kernel.a3_gn_kernel`` but exposes
the floating-point precision of the transport-sensitive arithmetic as a knob, while the
integer plane indices / octave masks / crossing transactions stay exact (int64).  It is
used only by the validation harness; the frozen kernel is never modified.

Faithfulness contract: with ``MODES["f64ref"]`` every quantity (total tau, event position,
and the ordered crossing stream) must reproduce ``kernel.trace`` bit-for-bit.  Only then are
the float32 numbers meaningful.

Precision modes (Sec.9, Sec.11 mixed-precision study):
  f64ref    : ray f64, math f64, division ordering        (== kernel; the reference)
  f32_div   : ray f32, math f32, division ordering
  f32_cross : ray f32, math f32, cross-multiplication ordering (no division in the compare)
  mixed64   : ray f32 (quantized input), math f64          (isolates ray-quantization loss)

CRITICAL: every arithmetic operand is a numpy scalar/array of the active dtype.  Python
floats are always 64-bit, so wrapping constants in ``dt(...)`` is what makes an f32 run
actually round in f32 (Invariant Sec.19.7: local coordinates formed before f32 conversion).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a3.kernel.a3_gn_kernel import Medium, octave_mask_for_plane  # noqa: E402

GL5_X64, GL5_W64 = np.polynomial.legendre.leggauss(5)


@dataclass(frozen=True)
class PrecisionConfig:
    """Three independent precision knobs, so the float32 error can be decomposed.

    ray_dtype   : precision the ray (origin+direction) is supplied/quantized at.
    clock_dtype : precision of the DDA boundary-time arithmetic / event ordering.
    arith_dtype : precision of GL5 integration, terminal polynomial and Newton.
    compare     : "div" (compare divided times) or "cross" (cross-multiplication ordering).

    Isolation logic (all vs the true-f64 reference unless noted):
      f64ref        : all f64 == kernel (faithfulness reference)
      mixed64       : ray f32, clock+arith f64  -> isolates RAY quantization alone
      clock32_*     : ray f32, clock f32, arith f64 -> + DDA-clock precision (frozen f32 style)
      full32_*      : ray f32, clock f32, arith f32 -> + full-f32 integration (Sec.18 new)
    Comparing clock32_*/full32_* against mixed64 (same f32 ray) isolates PIPELINE precision.
    """

    name: str
    ray_dtype: type
    clock_dtype: type       # DDA plane-time / face-ordering precision
    arith_dtype: type       # GL5 / terminal / density precision
    compare: str = "div"    # "div" | "cross"
    box_dtype: type = None  # ray-box clipping (entry/exit) precision; None -> clock_dtype


MODES = {
    "f64ref": PrecisionConfig("f64ref", np.float64, np.float64, np.float64, "div"),
    "mixed64": PrecisionConfig("mixed64", np.float32, np.float64, np.float64, "div"),
    "clock32_div": PrecisionConfig("clock32_div", np.float32, np.float32, np.float64, "div"),
    "clock32_cross": PrecisionConfig("clock32_cross", np.float32, np.float32, np.float64, "cross"),
    "full32_div": PrecisionConfig("full32_div", np.float32, np.float32, np.float32, "div"),
    "full32_cross": PrecisionConfig("full32_cross", np.float32, np.float32, np.float32, "cross"),
    # Formal recommended architecture: f64 ray + f64 clipping + f64 DDA decisions, f32 arith.
    "decision64": PrecisionConfig("decision64", np.float64, np.float64, np.float32, "div", np.float64),
}

# Precision-ablation modes for "minimal high-precision region" (collab priority 3).
# Baseline full-f32 is (ray f32, box f32, clock f32, arith f32).
ABLATION = {
    # everything f32 -- the tail is present here
    "full32": PrecisionConfig("full32", np.float32, np.float32, np.float32, "div", np.float32),
    # A: f64 ray, else f32
    "A_ray64": PrecisionConfig("A_ray64", np.float64, np.float32, np.float32, "div", np.float32),
    # B: f32 ray, f64 box clipping only
    "B_box64": PrecisionConfig("B_box64", np.float32, np.float32, np.float32, "div", np.float64),
    # C: f32 ray, f64 DDA plane-time/ordering only
    "C_dda64": PrecisionConfig("C_dda64", np.float32, np.float64, np.float32, "div", np.float32),
    # D: f64 box clipping + f64 DDA decision, but f32 RAY, f32 arithmetic (GL5/Newton)
    "D_decide64": PrecisionConfig("D_decide64", np.float32, np.float64, np.float32, "div", np.float64),
    # F: f64 ray + f64 box + f64 DDA (the whole discrete-decision path), f32 arith only
    "F_decidepath64": PrecisionConfig("F_decidepath64", np.float64, np.float64, np.float32, "div", np.float64),
    # G: f64 ray + f64 box entry/exit ONLY; f32 internal DDA + f32 arith. Tests whether the
    # (bit-identical) internal grid can stay f32 and only the amplified box clip needs f64.
    "G_boxonly64": PrecisionConfig("G_boxonly64", np.float64, np.float32, np.float32, "div", np.float64),
}


@dataclass
class PrecResult:
    hit: bool
    t_entry: float
    t_exit: float
    total_tau: float
    event_t: Optional[float]
    segments: int
    octave_reloads: int
    newton_iterations: int
    newton_fallbacks: int
    # ordered crossing stream token = tuple of plane_indices per transaction (topology)
    stream_tokens: Tuple[Tuple[Optional[int], Optional[int], Optional[int]], ...]
    stream_times: Tuple[float, ...]


# --------------------------------------------------------------------------------------
# dtype-faithful primitives
# --------------------------------------------------------------------------------------
def _fade(u: np.ndarray, dt) -> np.ndarray:
    three = dt(3.0)
    two = dt(2.0)
    return u * u * (three - two * u)


def _density_from_corners(local: np.ndarray, corners: np.ndarray, dt) -> np.ndarray:
    one = dt(1.0)
    f = _fade(local.astype(dt), dt)
    c = corners.astype(dt)
    x00 = (one - f[0]) * c[0, 0, 0] + f[0] * c[1, 0, 0]
    x10 = (one - f[0]) * c[0, 1, 0] + f[0] * c[1, 1, 0]
    x01 = (one - f[0]) * c[0, 0, 1] + f[0] * c[1, 0, 1]
    x11 = (one - f[0]) * c[0, 1, 1] + f[0] * c[1, 1, 1]
    y0 = (one - f[1]) * x00 + f[1] * x10
    y1 = (one - f[1]) * x01 + f[1] * x11
    return (one - f[2]) * y0 + f[2] * y1


def _cached_density(medium: Medium, states, p: np.ndarray, dt) -> np.ndarray:
    total = dt(0.0)
    for octave, cell, corners in states:
        local = p * dt(octave.frequency) - cell.astype(dt)
        total = total + _density_from_corners(local, corners, dt)
    return total


def _direction_owned_cell(octave, ro, rd, t, dt) -> np.ndarray:
    q = (ro + rd * t) * dt(octave.frequency)
    nearest = np.rint(q)
    close = np.abs(q - nearest) <= dt(2e-11) * np.maximum(dt(1.0), np.abs(q))
    q = np.where(close, q + np.sign(rd) * dt(1e-10), q)
    cell = np.floor(q).astype(np.int64)
    return np.minimum(np.maximum(cell, 0), octave.cells - 1)


def _intersect_box(ro, rd, bmin, bmax, dt):
    te, tx = dt(-np.inf), dt(np.inf)
    for a in range(3):
        if abs(float(rd[a])) < 1e-30:
            if ro[a] < bmin[a] or ro[a] > bmax[a]:
                return False, dt(0.0), dt(0.0)
            continue
        t0 = (bmin[a] - ro[a]) / rd[a]
        t1 = (bmax[a] - ro[a]) / rd[a]
        te = max(te, min(t0, t1))
        tx = min(tx, max(t0, t1))
    te = max(te, dt(0.0))
    return tx >= te, te, tx


# --------------------------------------------------------------------------------------
# streaming finest-grid DDA (dtype + compare aware)
# --------------------------------------------------------------------------------------
def _next_axes(t_max, numer, den, dt, compare: str, tie_scale: float):
    """Return (t_next, tied_axes). ``compare`` selects division vs cross-multiplication.

    Cross-multiplication compares positive ratios |numer|/|den| via |na|*|db| vs |nb|*|da|,
    avoiding forming the divided time for the ordering decision.
    """
    moving = [a for a in range(3) if math.isfinite(float(t_max[a]))]
    if not moving:
        return dt(np.inf), []
    if compare == "div":
        t_next = min(t_max[a] for a in moving)
        tol = dt(2e-12) * max(dt(1.0), abs(t_next))
        tied = [a for a in moving if abs(t_max[a] - t_next) <= tol]
        return t_next, tied
    # cross-multiplication ordering
    an = {a: abs(numer[a]) for a in moving}
    ad = {a: abs(den[a]) for a in moving}
    winner = moving[0]
    for a in moving[1:]:
        # a earlier than winner iff an[a]*ad[winner] < an[winner]*ad[a]
        if an[a] * ad[winner] < an[winner] * ad[a]:
            winner = a
    tied = []
    for a in moving:
        lhs = an[a] * ad[winner]
        rhs = an[winner] * ad[a]
        scale = max(abs(lhs), abs(rhs), dt(1.0))
        if abs(lhs - rhs) <= dt(2e-12) * scale:
            tied.append(a)
    t_next = t_max[winner]  # divided value only for the segment bound
    return t_next, tied


def _quantize_ray(ro_in, rd_in, ray_dtype):
    """Quantize the supplied ray to ray_dtype and normalize at that precision."""
    ro = np.asarray(ro_in, dtype=np.float64).astype(ray_dtype)
    rd = np.asarray(rd_in, dtype=np.float64).astype(ray_dtype)
    norm = ray_dtype(np.sqrt(np.sum(rd.astype(ray_dtype) ** 2)))
    if float(norm) == 0.0:
        raise ValueError("zero direction")
    return ro, (rd / norm)


def _trace(medium: Medium, ro_in, rd_in, cfg: PrecisionConfig,
           target_tau: Optional[float]) -> PrecResult:
    ct = cfg.clock_dtype    # DDA boundary-time / ordering precision
    at = cfg.arith_dtype    # GL5 / terminal / density precision
    bt = cfg.box_dtype or ct  # ray-box clipping (entry/exit) precision
    F = medium.finest_frequency
    freqs = [o.frequency for o in medium.octaves]
    world_fine = (medium.world_cells * F).astype(np.int64)

    ro_ray, rd_ray = _quantize_ray(ro_in, rd_in, cfg.ray_dtype)
    ro_c, rd_c = ro_ray.astype(ct), rd_ray.astype(ct)   # clock view
    ro_a, rd_a = ro_ray.astype(at), rd_ray.astype(at)   # arith view

    # Ray-box clipping at box precision (entry/exit fix the integration domain),
    # then cast the endpoints into the clock precision used by the DDA.
    ro_b, rd_b = ro_ray.astype(bt), rd_ray.astype(bt)
    hit, te_b, tx_b = _intersect_box(ro_b, rd_b, medium.bmin.astype(bt), medium.bmax.astype(bt), bt)
    if not hit or tx_b <= te_b:
        return PrecResult(False, float(te_b), float(tx_b), 0.0, None, 0, 0, 0, 0, (), ())
    te, tx = ct(float(te_b)), ct(float(tx_b))

    # initial finest cell (direction-owned), clock precision
    q = (ro_c + rd_c * te) * ct(F)
    nearest = np.rint(q)
    close = np.abs(q - nearest) <= ct(2e-11) * np.maximum(ct(1.0), np.abs(q))
    q = np.where(close, q + np.sign(rd_c) * ct(1e-10), q)
    cell = np.minimum(np.maximum(np.floor(q).astype(np.int64), 0), world_fine - 1)

    step = np.sign(rd_c).astype(np.int64)
    next_k = np.zeros(3, dtype=np.int64)
    t_max = np.full(3, np.inf, dtype=ct)
    numer = np.zeros(3, dtype=ct)
    den = rd_c.copy()

    for a in range(3):
        if step[a] == 0:
            continue
        next_k[a] = cell[a] + 1 if step[a] > 0 else cell[a]
        while True:
            plane = ct(next_k[a]) / ct(F)
            numer[a] = plane - ro_c[a]
            t_max[a] = numer[a] / rd_c[a]
            if float(t_max[a]) > float(te) + 1e-13:
                break
            next_k[a] += step[a]

    states = []
    for octave in medium.octaves:
        c = _direction_owned_cell(octave, ro_c, rd_c, te, ct)
        states.append([octave, c, octave.corners(c)])

    tokens: List[Tuple[Optional[int], Optional[int], Optional[int]]] = [(None, None, None)]
    times: List[float] = [float(te)]

    total_tau = at(0.0)
    remaining = None if target_tau is None else at(target_tau)
    segments = 0
    reloads = 0
    newton_iters = 0
    newton_fallbacks = 0
    event_t: Optional[float] = None

    cur_t = te
    max_steps = int(np.sum(world_fine) * 4 + 64)
    for _ in range(max_steps + 1):
        t_next, tied = _next_axes(t_max, numer, den, ct, cfg.compare, 1.0)
        seg_b = min(t_next, tx)

        if float(seg_b) > float(cur_t):
            seg_tau = _integrate_segment(medium, states, ro_a, rd_a,
                                         at(float(cur_t)), at(float(seg_b)), at)
            segments += 1
            if remaining is not None and float(seg_tau) >= float(remaining) and float(seg_tau) > 0:
                dcoeff = _terminal_density_poly(medium, states, ro_a, rd_a, at(float(cur_t)), at)
                # spatial tolerance must be reachable in the working precision: f64 -> 1e-12,
                # f32 -> ~1e-6 (eps_f32 ~ 1.2e-7), else Newton can never satisfy it and the
                # safeguard fires spuriously every iteration.
                spatial_tol = 1e-12 if at is np.float64 else 1e-6
                local_t, it, fb = _terminal_newton(dcoeff, at(medium.sigma_t), remaining,
                                                    at(float(seg_b) - float(cur_t)), at,
                                                    max_iterations=16, spatial_tolerance=spatial_tol)
                event_t = float(cur_t) + float(local_t)
                newton_iters, newton_fallbacks = it, fb
                total_tau = total_tau + remaining
                return PrecResult(True, float(te), float(tx), float(total_tau), event_t,
                                  segments, reloads, newton_iters, newton_fallbacks,
                                  tuple(tokens), tuple(times))
            total_tau = total_tau + seg_tau
            if remaining is not None:
                remaining = remaining - seg_tau

        if not math.isfinite(float(t_next)) or float(t_next) >= float(tx) - 1e-13:
            break

        plane_idx: List[Optional[int]] = [None, None, None]
        axis_masks = [0, 0, 0]
        for a in tied:
            k = int(next_k[a])
            plane_idx[a] = k
            if 0 < k < int(world_fine[a]):
                axis_masks[a] = octave_mask_for_plane(k, freqs, F)
        tokens.append((plane_idx[0], plane_idx[1], plane_idx[2]))
        times.append(float(t_next))

        # atomic commit across tied axes/octaves
        for level, st in enumerate(states):
            octave, c, _corners = st
            changed = False
            for a in tied:
                if axis_masks[a] & (1 << level):
                    c[a] += step[a]
                    changed = True
            if changed:
                c[:] = np.minimum(np.maximum(c, 0), octave.cells - 1)
                st[2] = octave.corners(c)
                reloads += 1

        for a in tied:
            cell[a] += step[a]
            next_k[a] += step[a]
            plane = ct(next_k[a]) / ct(F)
            numer[a] = plane - ro_c[a]
            t_max[a] = numer[a] / rd_c[a]
        cur_t = t_next

    tokens.append((None, None, None))
    times.append(float(tx))
    return PrecResult(True, float(te), float(tx), float(total_tau), None,
                      segments, reloads, 0, 0, tuple(tokens), tuple(times))


def _integrate_segment(medium, states, ro, rd, a, b, dt):
    if float(b) <= float(a):
        return dt(0.0)
    half = dt(0.5) * (b - a)
    mid = dt(0.5) * (b + a)
    nodes_x = GL5_X64.astype(dt)
    weights = GL5_W64.astype(dt)
    acc = dt(0.0)
    for i in range(len(nodes_x)):
        t = mid + half * nodes_x[i]
        acc = acc + weights[i] * _cached_density(medium, states, ro + rd * t, dt)
    return dt(medium.sigma_t) * half * acc


def _fade_poly(a, b, dt):
    three = dt(3.0)
    two = dt(2.0)
    six = dt(6.0)
    one = dt(1.0)
    return np.array([three * a * a - two * a * a * a,
                     six * a * b * (one - a),
                     three * b * b * (one - two * a),
                     -two * b * b * b], dtype=dt)


def _polymul(p, q, dt):
    out = np.zeros(len(p) + len(q) - 1, dtype=dt)
    for i in range(len(p)):
        for j in range(len(q)):
            out[i + j] = out[i + j] + p[i] * q[j]
    return out


def _pad(poly, size, dt):
    out = np.zeros(size, dtype=dt)
    out[:len(poly)] = poly
    return out


def _density_polynomial(local_entry, direction, corners, dt):
    fx = _fade_poly(local_entry[0], direction[0], dt)
    fy = _fade_poly(local_entry[1], direction[1], dt)
    fz = _fade_poly(local_entry[2], direction[2], dt)
    fxy = _polymul(fx, fy, dt)
    fxz = _polymul(fx, fz, dt)
    fyz = _polymul(fy, fz, dt)
    fxyz = _polymul(fxy, fz, dt)
    c = corners.astype(dt)
    d000, d100, d010, d110 = c[0, 0, 0], c[1, 0, 0], c[0, 1, 0], c[1, 1, 0]
    d001, d101, d011, d111 = c[0, 0, 1], c[1, 0, 1], c[0, 1, 1], c[1, 1, 1]
    ax = d100 - d000
    ay = d010 - d000
    az = d001 - d000
    axy = d110 - d100 - d010 + d000
    axz = d101 - d100 - d001 + d000
    ayz = d011 - d010 - d001 + d000
    axyz = d111 - d110 - d101 - d011 + d100 + d010 + d001 - d000
    rho = np.zeros(10, dtype=dt)
    rho[0] = d000
    rho = rho + ax * _pad(fx, 10, dt) + ay * _pad(fy, 10, dt) + az * _pad(fz, 10, dt)
    rho = rho + axy * _pad(fxy, 10, dt) + axz * _pad(fxz, 10, dt) + ayz * _pad(fyz, 10, dt)
    rho = rho + axyz * _pad(fxyz, 10, dt)
    return rho


def _terminal_density_poly(medium, states, ro, rd, seg_start, dt):
    p0 = ro + rd * seg_start
    combined = np.zeros(10, dtype=dt)
    for octave, cell, corners in states:
        local = p0 * dt(octave.frequency) - cell.astype(dt)
        combined = combined + _density_polynomial(local, rd * dt(octave.frequency), corners, dt)
    return combined


def _horner(coeff, x, dt):
    acc = dt(0.0)
    for c in coeff[::-1]:
        acc = acc * x + c
    return acc


def _terminal_newton(density_coeff, sigma_t, target_tau, segment_length, dt,
                     max_iterations=12, spatial_tolerance=1e-12):
    tau_coeff = np.zeros(len(density_coeff) + 1, dtype=dt)
    for k in range(len(density_coeff)):
        tau_coeff[k + 1] = sigma_t * density_coeff[k] / dt(k + 1)
    tau_end = _horner(tau_coeff, segment_length, dt)
    if float(target_tau) <= 0:
        return dt(0.0), 0, 0
    if float(target_tau) >= float(tau_end):
        return segment_length, 0, 0

    lo, hi = dt(0.0), segment_length
    x = segment_length * target_tau / max(tau_end, dt(1e-300))
    fallbacks = 0
    for iteration in range(1, max_iterations + 1):
        tau_x = _horner(tau_coeff, x, dt)
        deriv = sigma_t * max(_horner(density_coeff, x, dt), dt(0.0))
        residual = tau_x - target_tau
        if float(residual) < 0:
            lo = x
        else:
            hi = x
        spatial_residual = abs(residual) / max(deriv, dt(1e-14))
        if float(spatial_residual) <= spatial_tolerance * max(1.0, float(segment_length)):
            return x, iteration, fallbacks
        if float(deriv) > 1e-14:
            proposal = x - residual / deriv
        else:
            proposal = dt(np.nan)
        if not math.isfinite(float(proposal)) or float(proposal) <= float(lo) or float(proposal) >= float(hi):
            proposal = dt(0.5) * (lo + hi)
            fallbacks += 1
        x = proposal
    return x, max_iterations, fallbacks


def trace_precision(medium: Medium, ro, rd, cfg: PrecisionConfig,
                    target_tau: Optional[float] = None) -> PrecResult:
    return _trace(medium, ro, rd, cfg, target_tau)


# --------------------------------------------------------------------------------------
# Frozen production policy: A3-GBox64/F32
#   f64 ray normalize + f64 box entry/exit slab clip; internal transactional DDA, direct
#   normalized coefficient assembly, GL5 and terminal Newton all f32.  Established minimal
#   high-precision region for the CONTINUOUS shared-corner cubic-fade A3 primitive.
# --------------------------------------------------------------------------------------
PRODUCTION = PrecisionConfig("A3-GBox64/F32", np.float64, np.float32, np.float32, "div", np.float64)
MODES["production"] = PRODUCTION


def trace_production(medium: Medium, ro, rd, *,
                     target_tau: Optional[float] = None) -> PrecResult:
    """A3-GBox64/F32 production traversal (f64 ray+clip, everything else f32)."""
    return _trace(medium, ro, rd, PRODUCTION, target_tau)
