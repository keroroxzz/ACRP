"""Direct coefficient-assembly closure for the terminal degree-9 / degree-10 path.

Replaces an 11-sample Vandermonde surrogate with direct segment-normalized coefficient
assembly (already implemented in a3_gn_precision.density_polynomial). For terminal segments
(one finest cell, all octave cells constant), we compare assembly/eval variants against a
float64 normalized reference:

  coefficient error   : max_k |c_k^32 - c_k^64|, and a scale-normalized version (high-order
                        small coefficients have inflated relative error otherwise)
  transport error     : |dtau(1)| at segment end, |dT_xi| free-path event
  Newton              : iterations, fallbacks, hit mismatch

Variants:
  norm_direct_f32   : normalized s in [0,1], direct f32 assembly, ordinary Horner
  norm_direct_f32_comp : as above with compensated (2Sum) Horner
  norm_cast_f32     : build in f64 then cast coefficients to f32, normalized
  world_direct_f32  : world-distance parametrization (unnormalized s), direct f32

Run:  python a3/harness/direct_assembly.py [--json PATH]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a3.kernel import a3_gn_kernel as K  # noqa: E402
from a3.harness.a3_gn_precision import _density_polynomial as density_polynomial  # noqa: E402
from a3.harness.float32_transport_validation import make_medium  # noqa: E402


def octave_states_at(med, p):
    states = []
    for o in med.octaves:
        q = p * o.frequency
        cell = np.minimum(np.maximum(np.floor(q).astype(np.int64), 0), o.cells - 1)
        states.append((o, cell, o.corners(cell)))
    return states


def build_density(states, ro, rd, a, seg_len, dt, normalized):
    p0 = (ro + rd * a).astype(dt)
    combined = np.zeros(10, dtype=dt)
    scale = dt(seg_len) if normalized else dt(1.0)
    for o, cell, corners in states:
        local = p0 * dt(o.frequency) - cell.astype(dt)
        direction = rd.astype(dt) * dt(o.frequency) * scale
        combined = combined + density_polynomial(local, direction, corners, dt)
    return combined


def tau_coeffs(dens, sigma, seg_len, dt, normalized):
    # physical optical depth; normalized carries an extra seg_len Jacobian
    jac = dt(seg_len) if normalized else dt(1.0)
    tau = np.zeros(len(dens) + 1, dtype=dt)
    for k in range(len(dens)):
        tau[k + 1] = sigma * jac * dens[k] / dt(k + 1)
    return tau


def horner(coeff, x, dt):
    acc = dt(0.0)
    for c in coeff[::-1]:
        acc = acc * x + c
    return acc


def horner_comp(coeff, x, dt):
    # compensated (2Sum/2Prod) Horner for degree-n poly
    acc = dt(0.0); err = dt(0.0)
    for c in coeff[::-1]:
        p = acc * x
        e1 = math.fma(acc, float(x), -float(p)) if hasattr(math, "fma") else 0.0
        s = p + c
        # 2Sum
        z = s - p
        e2 = (p - (s - z)) + (c - z)
        acc = s
        err = err * x + (dt(e1) + dt(e2))
    return acc + err


def newton_invert(tau_coeff, dens_coeff, sigma_jac, target, seg_end, dt, evalfn,
                  max_it=16, tol=1e-6):
    tau_end = evalfn(tau_coeff, seg_end, dt)
    if float(target) <= 0:
        return dt(0.0), 0, 0
    if float(target) >= float(tau_end):
        return seg_end, 0, 0
    lo, hi = dt(0.0), seg_end
    x = seg_end * target / max(tau_end, dt(1e-300))
    fb = 0
    for it in range(1, max_it + 1):
        tx = evalfn(tau_coeff, x, dt)
        deriv = sigma_jac * max(evalfn(dens_coeff, x, dt), dt(0.0))
        res = tx - target
        if float(res) < 0:
            lo = x
        else:
            hi = x
        if float(abs(res) / max(deriv, dt(1e-14))) <= tol * max(1.0, float(seg_end)):
            return x, it, fb
        prop = x - res / deriv if float(deriv) > 1e-14 else dt(math.nan)
        if not math.isfinite(float(prop)) or float(prop) <= float(lo) or float(prop) >= float(hi):
            prop = dt(0.5) * (lo + hi); fb += 1
        x = prop
    return x, max_it, fb


def run(n=400, seed=20260718):
    variants = ["norm_direct_f32", "norm_direct_f32_comp", "norm_cast_f32", "world_direct_f32"]
    agg = {v: dict(coeff=[], coeff_rel=[], dtau=[], devent=[], it=[], fb=[], hit_mis=0) for v in variants}
    for F in [16, 64, 128]:
        med = make_medium(F)
        rng = np.random.default_rng(seed + F)
        got = 0
        while got < n:
            p = rng.uniform(med.bmin + 0.05, med.bmax - 0.05, 3)
            rd = rng.normal(size=3); rd /= np.linalg.norm(rd)
            # segment stays inside the current finest cell: distance to next finest plane
            q = p * F
            frac = q - np.floor(q)
            dist = np.array([(1 - frac[a]) / abs(rd[a] * F) if rd[a] > 0 else
                             (frac[a] / abs(rd[a] * F) if rd[a] < 0 else np.inf) for a in range(3)])
            seg_len = float(min(dist.min() * 0.9, 2.0 / F))
            if seg_len < 1e-4:
                continue
            states = octave_states_at(med, p)
            sigma = med.sigma_t
            # f64 normalized reference
            d64 = build_density(states, p, rd, 0.0, seg_len, np.float64, True)
            t64 = tau_coeffs(d64, sigma, seg_len, np.float64, True)
            tau_end64 = horner(t64, 1.0, np.float64)
            if tau_end64 < 1e-7:
                continue
            got += 1
            xi = 0.5 * tau_end64
            s_ref, _, _ = newton_invert(t64, d64, np.float64(sigma * seg_len), np.float64(xi),
                                        np.float64(1.0), np.float64, horner)
            T_ref = float(s_ref) * seg_len
            for v in variants:
                normalized = (v != "world_direct_f32")
                dt = np.float32
                if v == "norm_cast_f32":
                    dens = build_density(states, p, rd, 0.0, seg_len, np.float64, True).astype(np.float32)
                else:
                    dens = build_density(states, p, rd, 0.0, seg_len, np.float32, normalized)
                seg_end = np.float32(1.0) if normalized else np.float32(seg_len)
                tau = tau_coeffs(dens, np.float32(sigma), np.float32(seg_len), np.float32, normalized)
                evalfn = horner_comp if v.endswith("comp") else horner
                # coeff error only meaningful in the SAME (normalized) basis
                if normalized:
                    ce = float(np.max(np.abs(dens.astype(np.float64) - d64)))
                    scale = max(float(np.max(np.abs(d64))), 1e-12)
                    agg[v]["coeff"].append(ce)
                    agg[v]["coeff_rel"].append(ce / scale)
                tau_end = evalfn(tau, seg_end, np.float32)
                agg[v]["dtau"].append(abs(float(tau_end) - float(tau_end64)))
                xi32 = np.float32(xi)
                s32, it, fb = newton_invert(tau, dens, np.float32(sigma) * (np.float32(seg_len) if normalized else np.float32(1.0)),
                                            xi32, seg_end, np.float32, evalfn)
                T = float(s32) * (seg_len if normalized else 1.0)
                agg[v]["devent"].append(abs(T - T_ref))
                agg[v]["it"].append(it); agg[v]["fb"].append(fb)
    out = {"cases": {}}
    print("%-22s %-11s %-11s %-11s %-11s %-8s %s" % ("variant", "coeff_abs", "coeff_rel", "dtau(1)", "d_event", "it_mean", "fb"))
    for v in variants:
        a = agg[v]
        rec = dict(
            coeff_max=max(a["coeff"]) if a["coeff"] else None,
            coeff_rel_max=max(a["coeff_rel"]) if a["coeff_rel"] else None,
            dtau_max=max(a["dtau"]), devent_max=max(a["devent"]),
            it_mean=float(np.mean(a["it"])), fb_total=int(np.sum(a["fb"])))
        out["cases"][v] = rec
        print("%-22s %-11s %-11s %.3e   %.3e   %-8.2f %d" % (
            v, ("%.2e" % rec["coeff_max"]) if rec["coeff_max"] is not None else "-(world)",
            ("%.2e" % rec["coeff_rel_max"]) if rec["coeff_rel_max"] is not None else "-",
            rec["dtau_max"], rec["devent_max"], rec["it_mean"], rec["fb_total"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    res = run()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print("wrote", args.json)


if __name__ == "__main__":
    main()
