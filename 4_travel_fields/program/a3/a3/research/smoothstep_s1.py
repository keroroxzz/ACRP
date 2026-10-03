"""Smoothstep S1: does root partition break the GL accuracy floor? (isolated, no sun/envelope)

Density along a segment is a cloud-shaping smoothstep of a degree-9 procedural field:

    rho(s) = smoothstep((f(s) - theta) / w),   f in P_9,  s in [0,1]

smoothstep(u) = 3u^2 - 2u^3 on the transition, so rho = smoothstep(P_9) is degree-27 there.
GL5 (exact to degree 9) therefore cannot integrate the whole segment exactly -> an error floor.

We compare, against a dense-quadrature reference optical depth:
  A) NO partition : GL5 over the whole [0,1] (the floored production method).
  B) PARTITION    : find the roots of f=theta and f=theta+w, split [0,1] into
                    empty / transition / saturated pieces, integrate the transition
                    pieces with GL14 (exact to degree 27) -- restoring piecewise-poly exactness.

Success = B's error is orders below A's (the floor is broken). smoothstep is C1 so this is a
BRANCH-PARTITION problem (no jump term); only a hard threshold would need a jump budget.

Run:  python a3/research/smoothstep_s1.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from a3.kernel.a3_gn_kernel import density_polynomial  # realistic cubic-fade degree-9

THETA = 0.5
W = 0.16

# Gauss-Legendre nodes/weights
_G5X, _G5W = np.polynomial.legendre.leggauss(5)
_G14X, _G14W = np.polynomial.legendre.leggauss(14)
_GREFX, _GREFW = np.polynomial.legendre.leggauss(64)  # dense reference per piece


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def rho_of(coef, s):
    f = np.polyval(coef, s)
    return smoothstep((f - THETA) / W)


def gl_integral(fn, a, b, X, Wt):
    h = 0.5 * (b - a); m = 0.5 * (b + a)
    return h * np.dot(Wt, fn(m + h * X))


def reference_tau(coef):
    # dense reference: subdivide at the true crossings, GL64 per piece (near-exact)
    br = _crossings(coef)
    pts = [0.0] + br + [1.0]
    tau = 0.0
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        if b > a:
            tau += gl_integral(lambda s: rho_of(coef, s), a, b, _GREFX, _GREFW)
    return tau


def _crossings(coef):
    br = []
    for target in (THETA, THETA + W):
        r = np.roots(np.array(coef, float) - np.array([0] * (len(coef) - 1) + [target]))
        for z in r:
            if abs(z.imag) < 1e-9 and -1e-9 < z.real < 1.0 + 1e-9:
                br.append(float(np.clip(z.real, 0.0, 1.0)))
    return sorted(set(br))


def tau_no_partition(coef):
    return gl_integral(lambda s: rho_of(coef, s), 0.0, 1.0, _G5X, _G5W)


def tau_partition(coef):
    br = _crossings(coef)
    pts = [0.0] + br + [1.0]
    tau = 0.0
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        if b <= a:
            continue
        fm = np.polyval(coef, 0.5 * (a + b))
        if fm <= THETA:
            continue                       # empty piece: rho = 0
        if fm >= THETA + W:
            tau += (b - a)                 # saturated piece: rho = 1
            continue
        tau += gl_integral(lambda s: rho_of(coef, s), a, b, _G14X, _G14W)  # transition: GL14
    return tau


def random_degree9(rng):
    # realistic cubic-fade cell field crossing the threshold: nonnegative corners, a ray
    # segment through one finest cell -> genuine degree-9 density along s in [0,1].
    corners = rng.uniform(0.2, 1.0, (2, 2, 2))
    local = rng.uniform(0.0, 1.0, 3)
    d = rng.normal(size=3); d /= np.linalg.norm(d)
    asc = density_polynomial(local, d, corners)    # ascending coeffs, len 10
    return np.asarray(asc[::-1], float)            # descending for np.polyval


def run(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    errs_A, errs_B, nroots = [], [], []
    for _ in range(n):
        coef = random_degree9(rng)
        ref = reference_tau(coef)
        errs_A.append(abs(tau_no_partition(coef) - ref))
        errs_B.append(abs(tau_partition(coef) - ref))
        nroots.append(len(_crossings(coef)))
    A = np.array(errs_A); B = np.array(errs_B)
    out = dict(
        n=n,
        no_partition=dict(p50=float(np.percentile(A, 50)), p99=float(np.percentile(A, 99)), max=float(A.max())),
        partition=dict(p50=float(np.percentile(B, 50)), p99=float(np.percentile(B, 99)), max=float(B.max())),
        floor_reduction_p99=float(np.percentile(A, 99) / max(np.percentile(B, 99), 1e-300)),
        mean_roots_per_segment=float(np.mean(nroots)),
    )
    print("smoothstep S1: %d degree-9 segments, theta=%.2f w=%.2f" % (n, THETA, W))
    print("  optical-depth error vs dense reference:")
    print("    NO partition (GL5 whole):  p50 %.2e  p99 %.2e  max %.2e"
          % (out["no_partition"]["p50"], out["no_partition"]["p99"], out["no_partition"]["max"]))
    print("    PARTITION (roots + GL14):  p50 %.2e  p99 %.2e  max %.2e"
          % (out["partition"]["p50"], out["partition"]["p99"], out["partition"]["max"]))
    print("  p99 floor reduction: %.0fx   mean roots/segment: %.2f"
          % (out["floor_reduction_p99"], out["mean_roots_per_segment"]))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", default=None); a = ap.parse_args()
    res = run()
    if a.json:
        open(a.json, "w").write(json.dumps(res, indent=2))
        print("wrote", a.json)


if __name__ == "__main__":
    main()
