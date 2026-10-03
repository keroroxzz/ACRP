"""Real-time optimization loop: A3 vs classic raymarch, CPU-rendered, image + cost + quality.

Renders the SAME single-scattering procedural cloud three ways:
  * reference : very fine raymarch (ground truth)
  * raymarch  : classic fixed-step (baseline to beat)
  * a3        : analytic backbone (exact per-cell GL) + cheap residual + analytic sun

Cost metric = octave-point evaluations (the fundamental noise/hash workload, hardware-agnostic;
counted identically for both methods). Quality metric = RMSE vs the reference image.

Goal: an A3 config with cost <= raymarch_cost / 1.5 AND RMSE <= raymarch_RMSE.

Numba-jitted per-ray kernels; prange over pixels. Dumps PNGs to results/rt/.

Run:  python a3/research/rt_render_bench.py
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
from numba import njit, prange

# ---- scene ----------------------------------------------------------------
VOL = 4.0
SIGMA_T = 6.0
ALBEDO = 0.9
HG_G = 0.55
THRESH = 0.52
DENS = 2.4
N_OCT = 4                      # full detail octaves
SUN = np.array([0.55350, 0.75478, 0.35223])
SUNCOL = np.array([1.7, 1.5, 1.25]) * 3.0
SKYTOP = np.array([0.28, 0.45, 0.75])

# GL nodes/weights on [-1,1]
GL2X = np.array([-0.5773502691896257, 0.5773502691896257])
GL2W = np.array([1.0, 1.0])
GL3X = np.array([-0.7745966692414834, 0.0, 0.7745966692414834])
GL3W = np.array([0.5555555555555556, 0.8888888888888888, 0.5555555555555556])
GL5X = np.array([-0.906179845938664, -0.538469310105683, 0.0, 0.538469310105683, 0.906179845938664])
GL5W = np.array([0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189])


@njit(inline="always", cache=True)
def _fr(x):
    return x - math.floor(x)


@njit(inline="always", cache=True)
def hash13(x, y, z):
    px = _fr(x * 0.1031)
    py = _fr(y * 0.1030)
    pz = _fr(z * 0.0973)
    d = px * (py + 33.33) + py * (pz + 33.33) + pz * (px + 33.33)
    px += d
    py += d
    pz += d
    return _fr((px + py) * pz)


@njit(inline="always", cache=True)
def noise_cubic(x, y, z):
    ix = math.floor(x); iy = math.floor(y); iz = math.floor(z)
    fx = x - ix; fy = y - iy; fz = z - iz
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    uz = fz * fz * (3.0 - 2.0 * fz)
    n000 = hash13(ix, iy, iz); n100 = hash13(ix + 1, iy, iz)
    n010 = hash13(ix, iy + 1, iz); n110 = hash13(ix + 1, iy + 1, iz)
    n001 = hash13(ix, iy, iz + 1); n101 = hash13(ix + 1, iy, iz + 1)
    n011 = hash13(ix, iy + 1, iz + 1); n111 = hash13(ix + 1, iy + 1, iz + 1)
    x00 = n000 + (n100 - n000) * ux
    x10 = n010 + (n110 - n010) * ux
    x01 = n001 + (n101 - n001) * ux
    x11 = n011 + (n111 - n011) * ux
    y0 = x00 + (x10 - x00) * uy
    y1 = x01 + (x11 - x01) * uy
    return y0 + (y1 - y0) * uz


@njit(inline="always", cache=True)
def envelope(x, y, z):
    qx = (x - VOL * 0.5) / (VOL * 0.5)
    qy = (y - VOL * 0.5) / (VOL * 0.5)
    qz = (z - VOL * 0.5) / (VOL * 0.5)
    r = math.sqrt(qx * qx + qz * qz)
    # smoothstep(1.05,0.55,r) * smoothstep(1.0,0.35,|qy|)
    tr = (1.05 - r) / (1.05 - 0.55)
    tr = min(max(tr, 0.0), 1.0); tr = tr * tr * (3.0 - 2.0 * tr)
    ay = abs(qy)
    tv = (1.0 - ay) / (1.0 - 0.35)
    tv = min(max(tv, 0.0), 1.0); tv = tv * tv * (3.0 - 2.0 * tv)
    return tr * tv


@njit(inline="always", cache=True)
def raw_fbm(x, y, z, o_lo, o_hi):
    d = 0.0
    a = 0.5 * (2.0 ** o_lo)
    f = 2.0 ** o_lo
    for o in range(o_lo, o_hi):
        d += a * noise_cubic(x * f, y * f, z * f)
        a *= 0.5
        f *= 2.0
    return d


@njit(inline="always", cache=True)
def shape(d, x, y, z):
    # soft coverage threshold + envelope
    t = (d - THRESH) / 0.35
    t = min(max(t, 0.0), 1.0)
    t = t * t * (3.0 - 2.0 * t)
    return t * envelope(x, y, z) * DENS


@njit(inline="always", cache=True)
def density_full(x, y, z):
    return shape(raw_fbm(x, y, z, 0, N_OCT), x, y, z)


@njit(inline="always", cache=True)
def box_hit(ox, oy, oz, dx, dy, dz):
    # returns (hit, t0, t1) for box [0,VOL]^3
    t0 = -1e30; t1 = 1e30
    for i in range(3):
        o = ox if i == 0 else (oy if i == 1 else oz)
        dd = dx if i == 0 else (dy if i == 1 else dz)
        if abs(dd) < 1e-12:
            if o < 0.0 or o > VOL:
                return False, 0.0, 0.0
        else:
            a = (0.0 - o) / dd
            b = (VOL - o) / dd
            lo = min(a, b); hi = max(a, b)
            if lo > t0:
                t0 = lo
            if hi < t1:
                t1 = hi
    if t0 < 0.0:
        t0 = 0.0
    return t1 > t0, t0, t1


# ---- sun optical depth ----------------------------------------------------
@njit(inline="always", cache=True)
def sun_tau_march(x, y, z, steps):
    hit, t0, t1 = box_hit(x + SUN[0] * 1e-3, y + SUN[1] * 1e-3, z + SUN[2] * 1e-3,
                          SUN[0], SUN[1], SUN[2])
    if not hit:
        return 0.0, 0
    span = min(t1, VOL * 1.5)
    dt = span / steps
    tau = 0.0
    t = 0.5 * dt
    ev = 0
    for i in range(steps):
        if t > span:
            break
        tau += SIGMA_T * density_full(x + SUN[0] * t, y + SUN[1] * t, z + SUN[2] * t) * dt
        ev += N_OCT
        if tau > 8.0:
            break
        t += dt
    return tau, ev


# ---- classic raymarch -----------------------------------------------------
@njit(cache=True)
def render_raymarch(ox, oy, oz, dx, dy, dz, steps, sun_steps):
    hit, t0, t1 = box_hit(ox, oy, oz, dx, dy, dz)
    if not hit:
        return 0.0, 0.0, 0.0, 0.0, 0
    dt = (t1 - t0) / steps
    Tc = 1.0
    Lr = 0.0; Lg = 0.0; Lb = 0.0
    cph = dx * SUN[0] + dy * SUN[1] + dz * SUN[2]
    g2 = HG_G * HG_G
    ph = (1.0 - g2) / (4.0 * math.pi * (1.0 + g2 - 2.0 * HG_G * cph) ** 1.5)
    t = t0 + 0.5 * dt
    ev = 0
    for i in range(steps):
        px = ox + dx * t; py = oy + dy * t; pz = oz + dz * t
        rho = density_full(px, py, pz); ev += N_OCT
        dtau = SIGMA_T * rho * dt
        if dtau > 1e-5:
            st, se = sun_tau_march(px, py, pz, sun_steps); ev += se
            Ts = math.exp(-st)
            sc = 1.0 - math.exp(-dtau)
            w = Tc * ALBEDO * ph * Ts * sc
            Lr += w * SUNCOL[0] + Tc * ALBEDO * 0.12 * SKYTOP[0] * sc
            Lg += w * SUNCOL[1] + Tc * ALBEDO * 0.12 * SKYTOP[1] * sc
            Lb += w * SUNCOL[2] + Tc * ALBEDO * 0.12 * SKYTOP[2] * sc
            Tc *= math.exp(-dtau)
        if Tc < 0.004:
            break
        t += dt
    return Lr, Lg, Lb, 1.0 - Tc, ev


@njit(parallel=True, cache=True)
def render_image_raymarch(out, evout, eye, cw, res_w, res_h, steps, sun_steps):
    for j in prange(res_h):
        for i in range(res_w):
            u = (i + 0.5 - 0.5 * res_w) / res_h
            v = (j + 0.5 - 0.5 * res_h) / res_h
            # camera basis cw: rows r,u,f (3x3)
            dx = u * cw[0, 0] + v * cw[1, 0] + 1.3 * cw[2, 0]
            dy = u * cw[0, 1] + v * cw[1, 1] + 1.3 * cw[2, 1]
            dz = u * cw[0, 2] + v * cw[1, 2] + 1.3 * cw[2, 2]
            n = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= n; dy /= n; dz /= n
            r, g, b, a, ev = render_raymarch(eye[0], eye[1], eye[2], dx, dy, dz, steps, sun_steps)
            out[j, i, 0] = r; out[j, i, 1] = g; out[j, i, 2] = b; out[j, i, 3] = a
            evout[j, i] = ev


# ---- A3: DDA + empty-space skip + per-cell GL ------------------------------
@njit(inline="always", cache=True)
def _gl_cell_tau(ox, oy, oz, dx, dy, dz, ta, tb, order):
    # optical depth over [ta,tb] via GL(order) of full density. returns (tau, evals)
    h = 0.5 * (tb - ta); m = 0.5 * (tb + ta)
    s = 0.0
    if order == 2:
        for k in range(2):
            t = m + h * GL2X[k]
            s += GL2W[k] * density_full(ox + dx * t, oy + dy * t, oz + dz * t)
        return SIGMA_T * h * s, 2
    elif order == 3:
        for k in range(3):
            t = m + h * GL3X[k]
            s += GL3W[k] * density_full(ox + dx * t, oy + dy * t, oz + dz * t)
        return SIGMA_T * h * s, 3
    else:
        for k in range(5):
            t = m + h * GL5X[k]
            s += GL5W[k] * density_full(ox + dx * t, oy + dy * t, oz + dz * t)
        return SIGMA_T * h * s, 5


@njit(cache=True)
def sun_tau_a3(x, y, z, freq, order):
    # analytic sun optical depth: coarse DDA + empty-skip + GL. returns (tau, evals)
    ox = x + SUN[0] * 1e-3; oy = y + SUN[1] * 1e-3; oz = z + SUN[2] * 1e-3
    hit, t0, t1 = box_hit(ox, oy, oz, SUN[0], SUN[1], SUN[2])
    if not hit:
        return 0.0, 0
    t1 = min(t1, VOL * 1.5)
    cs = 1.0 / freq
    dx = SUN[0]; dy = SUN[1]; dz = SUN[2]
    ev = 0; tau = 0.0
    t = t0
    for _ in range(200):
        mx = ox + dx * t; my = oy + dy * t; mz = oz + dz * t
        # next cell boundary
        tnx = ((math.floor(mx / cs) + (1.0 if dx > 0 else 0.0)) * cs - mx) / dx if abs(dx) > 1e-9 else 1e30
        tny = ((math.floor(my / cs) + (1.0 if dy > 0 else 0.0)) * cs - my) / dy if abs(dy) > 1e-9 else 1e30
        tnz = ((math.floor(mz / cs) + (1.0 if dz > 0 else 0.0)) * cs - mz) / dz if abs(dz) > 1e-9 else 1e30
        tnext = t + min(tnx, min(tny, tnz)) + 1e-6
        tb = min(tnext, t1)
        if tb > t:
            cx = ox + dx * (0.5 * (t + tb)); cy = oy + dy * (0.5 * (t + tb)); cz = oz + dz * (0.5 * (t + tb))
            if envelope(cx, cy, cz) > 0.002:                 # empty-space skip
                ct, ce = _gl_cell_tau(ox, oy, oz, dx, dy, dz, t, tb, order)
                tau += ct; ev += ce
                if tau > 8.0:
                    return tau, ev
        if tnext >= t1:
            break
        t = tnext
    return tau, ev


@njit(cache=True)
def render_a3(ox, oy, oz, dx, dy, dz, cam_freq, cam_order, sun_freq, sun_order):
    hit, t0, t1 = box_hit(ox, oy, oz, dx, dy, dz)
    if not hit:
        return 0.0, 0.0, 0.0, 0.0, 0
    cs = 1.0 / cam_freq
    Tc = 1.0; Lr = 0.0; Lg = 0.0; Lb = 0.0; ev = 0
    cph = dx * SUN[0] + dy * SUN[1] + dz * SUN[2]
    g2 = HG_G * HG_G
    ph = (1.0 - g2) / (4.0 * math.pi * (1.0 + g2 - 2.0 * HG_G * cph) ** 1.5)
    t = t0
    for _ in range(400):
        mx = ox + dx * t; my = oy + dy * t; mz = oz + dz * t
        tnx = ((math.floor(mx / cs) + (1.0 if dx > 0 else 0.0)) * cs - mx) / dx if abs(dx) > 1e-9 else 1e30
        tny = ((math.floor(my / cs) + (1.0 if dy > 0 else 0.0)) * cs - my) / dy if abs(dy) > 1e-9 else 1e30
        tnz = ((math.floor(mz / cs) + (1.0 if dz > 0 else 0.0)) * cs - mz) / dz if abs(dz) > 1e-9 else 1e30
        tnext = t + min(tnx, min(tny, tnz)) + 1e-6
        tb = min(tnext, t1)
        if tb > t:
            cx = ox + dx * (0.5 * (t + tb)); cy = oy + dy * (0.5 * (t + tb)); cz = oz + dz * (0.5 * (t + tb))
            if envelope(cx, cy, cz) > 0.002:
                dtau, de = _gl_cell_tau(ox, oy, oz, dx, dy, dz, t, tb, cam_order); ev += de
                if dtau > 1e-5:
                    st, se = sun_tau_a3(cx, cy, cz, sun_freq, sun_order); ev += se
                    Ts = math.exp(-st)
                    sc = 1.0 - math.exp(-dtau)
                    w = Tc * ALBEDO * ph * Ts
                    Lr += w * sc * SUNCOL[0] + Tc * ALBEDO * 0.12 * SKYTOP[0] * sc
                    Lg += w * sc * SUNCOL[1] + Tc * ALBEDO * 0.12 * SKYTOP[1] * sc
                    Lb += w * sc * SUNCOL[2] + Tc * ALBEDO * 0.12 * SKYTOP[2] * sc
                    Tc *= math.exp(-dtau)
        if tnext >= t1 or Tc < 0.004:
            break
        t = tnext
    return Lr, Lg, Lb, 1.0 - Tc, ev


@njit(parallel=True, cache=True)
def render_image_a3(out, evout, eye, cw, res_w, res_h, cam_freq, cam_order, sun_freq, sun_order):
    for j in prange(res_h):
        for i in range(res_w):
            u = (i + 0.5 - 0.5 * res_w) / res_h
            v = (j + 0.5 - 0.5 * res_h) / res_h
            dx = u * cw[0, 0] + v * cw[1, 0] + 1.3 * cw[2, 0]
            dy = u * cw[0, 1] + v * cw[1, 1] + 1.3 * cw[2, 1]
            dz = u * cw[0, 2] + v * cw[1, 2] + 1.3 * cw[2, 2]
            n = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= n; dy /= n; dz /= n
            r, g, b, a, ev = render_a3(eye[0], eye[1], eye[2], dx, dy, dz,
                                       cam_freq, cam_order, sun_freq, sun_order)
            out[j, i, 0] = r; out[j, i, 1] = g; out[j, i, 2] = b; out[j, i, 3] = a
            evout[j, i] = ev
