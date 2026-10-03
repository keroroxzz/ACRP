"""Decisive experiment: A3 analytic free-flight vs delta / local-majorant tracking.

Hypothesis (the asymptotic-win claim): the cost of A3 analytic free-flight sampling --
measured as extinction (medium) evaluations per free flight -- is INDEPENDENT of density
contrast (sigma_max / sigma_mean), because A3 walks cells and inverts the monotone
optical-depth clock analytically. Woodcock delta tracking's cost instead grows with the
majorant looseness (~ contrast for a global majorant), because it rejects many null
collisions in high-contrast media. If A3 is a flat line while tracking rises, the
"contrast-independent free flight" advantage is nailed down.

Clean 1D setup (isolates the mechanism): a cubic-fade lattice extinction field on [0, L],
lattice spacing 1, so on each cell the field is a cubic and A3 integrates/inverts it in
closed form. Contrast is tuned by making the lattice corners peaky (corner = U^p); total
optical thickness is normalized to a fixed target so the free-flight distance (in cells) is
comparable across contrasts and only the contrast varies.

Cost metric: number of sigma(x) evaluations ("extinction evaluations"), the standard cost
unit. A3 reads each lattice corner once (cells+1); delta tracking evaluates sigma at every
trial step.

Run:  python a3/research/freeflight_benchmark.py [--flights N] [--json PATH] [--plot PATH]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

INF = float("inf")


# ---------------------------------------------------------------------------
# 1D cubic-fade lattice extinction field
# ---------------------------------------------------------------------------
def make_corners(L, p, seed):
    rng = np.random.default_rng(seed)
    c = rng.uniform(0.0, 1.0, size=L + 1) ** p     # peaky corners -> higher contrast
    return c


def _fade(u):
    return u * u * (3.0 - 2.0 * u)


def field(corners, C, t, L):
    if t < 0.0 or t >= L:
        return 0.0
    k = int(math.floor(t))
    u = t - k
    a = corners[k]
    b = corners[k + 1]
    return C * (a + (b - a) * _fade(u))


def cell_full_tau(corners, C, k):
    # exact integral over cell k of C*fade-interp = C*(a+b)/2  (since int_0^1 fade = 0.5)
    return C * 0.5 * (corners[k] + corners[k + 1])


def cell_partial_tau(corners, C, k, s):
    # exact integral over [0,s] within cell k: C*(a*s + (b-a)*(s^3 - s^4/2))
    a = corners[k]
    b = corners[k + 1]
    return C * (a * s + (b - a) * (s ** 3 - 0.5 * s ** 4))


def cell_sigma_local(corners, C, k, s):
    a = corners[k]
    b = corners[k + 1]
    return C * (a + (b - a) * _fade(s))


def total_tau(corners, C, L):
    return sum(cell_full_tau(corners, C, k) for k in range(L))


def normalize_C(corners, L, target_tau):
    raw = total_tau(corners, 1.0, L)
    return target_tau / raw if raw > 0 else 1.0


def sigma_max(corners, C):
    return C * float(np.max(corners))


def sigma_mean(corners, C, L):
    return total_tau(corners, C, L) / L


# ---------------------------------------------------------------------------
# A3 analytic free flight: walk cells, whole-cell consume, terminal Newton invert
# cost = distinct lattice corners read = terminal_cell + 2 (escape: L+1)
# ---------------------------------------------------------------------------
def a3_freeflight(corners, C, L, xi):
    cum = 0.0
    for k in range(L):
        dtk = cell_full_tau(corners, C, k)
        if cum + dtk >= xi:
            rem = xi - cum
            # solve cell_partial_tau(k, s) = rem, s in [0,1], monotone
            a = corners[k]
            b = corners[k + 1]
            lo, hi = 0.0, 1.0
            s = rem / max(dtk, 1e-30)              # good initial guess
            for _ in range(30):
                f = C * (a * s + (b - a) * (s ** 3 - 0.5 * s ** 4)) - rem
                fp = C * (a + (b - a) * _fade(s))  # derivative = C*sigma
                if f < 0.0:
                    lo = s
                else:
                    hi = s
                if abs(f) <= 1e-12 * max(1.0, rem) or (hi - lo) <= 1e-13:
                    break
                if fp > 1e-30:
                    sn = s - f / fp
                    if lo < sn < hi:
                        s = sn
                        continue
                s = 0.5 * (lo + hi)
            T = k + s
            evals = k + 2                          # corners 0..k+1 read
            return T, evals
        cum += dtk
    return INF, L + 1                              # escape


# ---------------------------------------------------------------------------
# Woodcock delta tracking with a GLOBAL majorant
# cost = number of sigma evaluations (trial steps)
# ---------------------------------------------------------------------------
def delta_global(corners, C, L, sig_bar, rng):
    t = 0.0
    n = 0
    while True:
        t += -math.log(1.0 - rng.random()) / sig_bar
        if t >= L:
            return INF, n
        s = field(corners, C, t, L)
        n += 1
        if rng.random() < s / sig_bar:
            return t, n


# ---------------------------------------------------------------------------
# Delta tracking with LOCAL (per-cell) majorants -- the production "super-voxel" style
# ---------------------------------------------------------------------------
def delta_local(corners, C, L, rng):
    """Returns (T, n_sigma, n_cells). n_sigma = sigma evals at trial hits (cached-grid cost).
    n_cells = cells entered, each of which needs a per-cell majorant = corner reads
    (the on-the-fly PROCEDURAL cost, normally amortized by a precomputed majorant grid)."""
    t = 0.0
    n = 0
    ncells = 0
    last_k = -1
    while t < L:
        k = int(math.floor(t))
        if k >= L:
            return INF, n, ncells
        if k != last_k:
            ncells += 1                             # new cell -> compute local majorant
            last_k = k
        sb = C * max(corners[k], corners[k + 1]) + 1e-30
        t2 = t - math.log(1.0 - rng.random()) / sb
        if t2 >= k + 1:
            t = float(k + 1)
            continue
        s = cell_sigma_local(corners, C, k, t2 - k)
        n += 1
        if rng.random() < s / sb:
            return t2, n, ncells
        t = t2
    return INF, n, ncells


# ---------------------------------------------------------------------------
# TRANSMITTANCE (next-event / shadow ray): A3 exact vs ratio tracking (noisy)
# ---------------------------------------------------------------------------
def a3_transmittance(corners, C, L):
    tau = total_tau(corners, C, L)                  # exact, cost = L+1 corner reads
    return math.exp(-tau), L + 1


def ratio_track_local(corners, C, L, rng):
    """Unbiased ratio-tracking transmittance estimate. Returns (T_est, n_sigma)."""
    t = 0.0
    Test = 1.0
    n = 0
    while t < L:
        k = int(math.floor(t))
        if k >= L:
            break
        sb = C * max(corners[k], corners[k + 1]) + 1e-30
        t2 = t - math.log(1.0 - rng.random()) / sb
        if t2 >= k + 1:
            t = float(k + 1)
            continue
        s = cell_sigma_local(corners, C, k, t2 - k)
        n += 1
        Test *= (1.0 - s / sb)
        t = t2
    return Test, n


# ---------------------------------------------------------------------------
def run(flights=20000, L=64, target_tau=4.0, p_list=(1, 2, 4, 8, 16, 32), seed=7):
    rows = []
    for p in p_list:
        corners = make_corners(L, p, seed + p)
        C = normalize_C(corners, L, target_tau)
        sM = sigma_max(corners, C)
        sMean = sigma_mean(corners, C, L)
        contrast = sM / sMean
        rng = np.random.default_rng(1000 + p)

        a3_evals, a3_T = [], []
        dg_evals, dg_T = [], []
        dl_sig, dl_proc, dl_T = [], [], []
        # transmittance: A3 exact vs ratio tracking variance
        Texact, a3_tr_evals = a3_transmittance(corners, C, L)
        rt_est, rt_evals = [], []
        for _ in range(flights):
            xi = -math.log(1.0 - rng.random())
            T, e = a3_freeflight(corners, C, L, xi)
            a3_evals.append(e)
            if math.isfinite(T):
                a3_T.append(T)
            T, e = delta_global(corners, C, L, sM, rng)
            dg_evals.append(e)
            if math.isfinite(T):
                dg_T.append(T)
            T, e, nc = delta_local(corners, C, L, rng)
            dl_sig.append(e)
            dl_proc.append(e + nc)                  # procedural: + majorant corner reads
            if math.isfinite(T):
                dl_T.append(T)
            Te, en = ratio_track_local(corners, C, L, rng)
            rt_est.append(Te)
            rt_evals.append(en)

        rt_std = float(np.std(rt_est))
        rt_ev = float(np.mean(rt_evals))
        # evals for ratio tracking to reach RMSE = 1% of transmittance (N ~ (std/target)^2)
        target = 0.01
        N_needed = (rt_std / target) ** 2 if rt_std > 0 else 0.0
        rt_evals_for_target = N_needed * rt_ev

        rows.append(dict(
            p=p, contrast=contrast, sigma_mean=sMean, sigma_max=sM,
            a3_mean_evals=float(np.mean(a3_evals)),
            delta_global_mean_evals=float(np.mean(dg_evals)),
            delta_local_sigma_evals=float(np.mean(dl_sig)),        # cached-grid cost
            delta_local_proc_evals=float(np.mean(dl_proc)),        # procedural cost (+majorants)
            a3_meanT=float(np.mean(a3_T)) if a3_T else INF,
            dl_meanT=float(np.mean(dl_T)) if dl_T else INF,
            collide_frac=len(a3_T) / flights,
            # transmittance
            T_exact=Texact,
            a3_tr_evals=a3_tr_evals,
            ratio_std=rt_std,
            ratio_evals_per_est=rt_ev,
            ratio_evals_for_1pct=rt_evals_for_target,
        ))
        r = rows[-1]
        print("p=%-2d contrast=%6.1f | FREE-FLIGHT evals A3=%6.2f dG=%7.2f "
              "dL(cached)=%5.2f dL(proc)=%6.2f | TRANSMITT A3(exact)=%d ev  "
              "ratio: std=%.3f, %.0f ev for 1%%RMSE"
              % (p, contrast, r["a3_mean_evals"], r["delta_global_mean_evals"],
                 r["delta_local_sigma_evals"], r["delta_local_proc_evals"],
                 r["a3_tr_evals"], r["ratio_std"], r["ratio_evals_for_1pct"]), flush=True)
    return rows


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = [r["contrast"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    ax[0].loglog(c, [r["delta_global_mean_evals"] for r in rows], "o-", label="delta (global majorant)")
    ax[0].loglog(c, [r["delta_local_sigma_evals"] for r in rows], "s-", label="delta local (cached grid)")
    ax[0].loglog(c, [r["delta_local_proc_evals"] for r in rows], "s--", label="delta local (procedural +majorants)")
    ax[0].loglog(c, [r["a3_mean_evals"] for r in rows], "^-", label="A3 analytic free-flight")
    ax[0].set_xlabel("density contrast  sigma_max/sigma_mean")
    ax[0].set_ylabel("extinction evals per free flight")
    ax[0].set_title("Free-flight SAMPLING cost\n(tracking wins with cached majorants)")
    ax[0].legend(); ax[0].grid(True, which="both", alpha=0.3)

    ax[1].loglog(c, [r["ratio_evals_for_1pct"] for r in rows], "o-", label="ratio tracking (for 1% RMSE)")
    ax[1].loglog(c, [r["a3_tr_evals"] for r in rows], "^-", label="A3 exact (zero error)")
    ax[1].set_xlabel("density contrast  sigma_max/sigma_mean")
    ax[1].set_ylabel("extinction evals for exact/1%-RMSE transmittance")
    ax[1].set_title("TRANSMITTANCE (shadow / NEE) cost\n(A3 exactness wins, grows with contrast)")
    ax[1].legend(); ax[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", type=int, default=20000)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--plot", type=Path)
    args = ap.parse_args()
    rows = run(flights=args.flights)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print("wrote", args.json)
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot(rows, str(args.plot))


if __name__ == "__main__":
    main()
