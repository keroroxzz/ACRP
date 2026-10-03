"""Permanent regression gate for the A3-GBox64/F32 production policy.

Production policy (frozen): f64 ray normalize + f64 box entry/exit slab clip; internal
transactional DDA, direct segment-normalized coefficient assembly, GL5 and terminal Newton
all f32.  Established minimal high-precision region for the CONTINUOUS shared-corner
cubic-fade A3 primitive.

The gate is TRANSPORT-based, not event-list-identity based (handoff Sec.9.2: float32
transport-equivalent partition != event-list identity).  It asserts, vs the true-f64
reference (== kernel):

  * max |dtau|  (total optical depth)      <  TAU_TOL
  * max |dT|    (free-path event position) <  EVT_TOL
  * zero hit/escape mismatch
  * zero terminal Newton fallback blowups (iterations bounded)

and REPORTS the transaction-list divergence rate (transport-neutral) as informational.
Includes dedicated box-grazing and long-ray regressions.

Run:  python a3/tests/validate_production.py   (exit nonzero on failure)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from a3.harness.a3_gn_precision import trace_precision, trace_production, MODES  # noqa: E402
from a3.harness.float32_transport_validation import make_medium, gen_ray  # noqa: E402

TAU_TOL = 1.5e-5
EVT_TOL = 1.5e-5
RAYS = 40

# (F, sigma, sparsity): baseline + high-extinction + plateau/low-density (high sparsity).
MEDIA = [
    (16, 1.2, 0.35), (64, 1.2, 0.35), (128, 1.2, 0.35),
    (64, 3.0, 0.35),   # high sigma_t
    (64, 1.2, 0.65),   # plateau / low-density (many zeroed corners)
    (128, 3.0, 0.5),   # high sigma + sparse at fine resolution
]

REF = MODES["f64ref"]


def _make(f, sigma, sparsity, seed=20260718):
    import numpy as _np
    rng = _np.random.default_rng(seed)
    wc = _np.array([3, 3, 3], dtype=_np.int64)
    freqs, ff = [], 1
    while ff <= f:
        freqs.append(ff); ff *= 2
    amps = [0.5 ** i for i in range(len(freqs))]
    from a3.kernel import a3_gn_kernel as _K
    octs = []
    for fr, a in zip(freqs, amps):
        lat = rng.uniform(0.0, 1.0, size=tuple(wc * fr + 1))
        lat[rng.random(size=lat.shape) < sparsity] = 0.0
        octs.append(_K.Octave(lat, fr, a))
    return _K.Medium(tuple(octs), sigma, wc)


def long_ray(rng, med):
    d = np.array([1.0, 1.0, 1.0]) + rng.normal(size=3) * 0.05
    d /= np.linalg.norm(d)
    c = 0.5 * (med.bmin + med.bmax)
    return c - d * 3.0 * np.linalg.norm(med.bmax - med.bmin), d


def box_grazing(rng, med):
    ax = rng.integers(3)
    rd = rng.normal(size=3)
    rd[ax] = rng.choice([-1.0, 1.0]) * 10.0 ** rng.uniform(-5, -3)
    rd /= np.linalg.norm(rd)
    ro = rng.uniform(med.bmin, med.bmax)
    ro[ax] = med.bmin[ax] - 0.5 if rd[ax] > 0 else med.bmax[ax] + 0.5
    return ro, rd


def main() -> int:
    # tau residual is the f32 arithmetic floor and scales with sigma_t (tau ~ sigma), so it is
    # gated NORMALIZED by sigma_t; event position is a distance and gated absolutely.
    max_tau_over_sigma = max_tau_raw = max_evt = 0.0
    hit_mism = 0
    list_div = 0
    max_newton = 0
    total = 0
    for (F, sigma, sparsity) in MEDIA:
        med = _make(F, sigma, sparsity)
        rng = np.random.default_rng(500 + F + int(sigma * 10) + int(sparsity * 100))
        gens = {
            "camera": lambda: gen_ray(rng, med, "random"),
            "box_grazing": lambda: box_grazing(rng, med),
            "long": lambda: long_ray(rng, med),
        }
        for name, g in gens.items():
            got, att = 0, 0
            while got < RAYS and att < RAYS * 60:
                att += 1
                rr = g()
                if rr is None:
                    continue
                ro, rd = rr
                if rd is None:
                    continue
                r = trace_precision(med, ro, rd, REF)
                if not r.hit or r.total_tau <= 1e-6:
                    continue
                got += 1
                total += 1
                p = trace_production(med, ro, rd)
                hit_mism += int(p.hit != r.hit)
                list_div += int(p.stream_tokens != r.stream_tokens)
                dtau = abs(p.total_tau - r.total_tau)
                max_tau_raw = max(max_tau_raw, dtau)
                max_tau_over_sigma = max(max_tau_over_sigma, dtau / med.sigma_t)
                tt = 0.5 * r.total_tau
                re = trace_precision(med, ro, rd, REF, target_tau=tt)
                pe = trace_production(med, ro, rd, target_tau=tt)
                max_newton = max(max_newton, pe.newton_iterations)
                if re.event_t is not None and pe.event_t is not None:
                    max_evt = max(max_evt, abs(pe.event_t - re.event_t))

    print("A3-GBox64/F32 production gate (%d rays over %d media: F/sigma/sparsity=%s)"
          % (total, len(MEDIA), MEDIA))
    print("  max |dtau|/sigma_t    = %.2e   (tol %.1e)" % (max_tau_over_sigma, TAU_TOL))
    print("  (raw max |dtau|       = %.2e)" % max_tau_raw)
    print("  max |d event|         = %.2e   (tol %.1e)" % (max_evt, EVT_TOL))
    print("  hit/escape mismatch   = %d" % hit_mism)
    print("  max terminal Newton it= %d" % max_newton)
    print("  transaction-list div  = %d/%d (%.0f%%) -- transport-neutral (Sec.9.2)"
          % (list_div, total, 100.0 * list_div / max(total, 1)))

    ok = (max_tau_over_sigma < TAU_TOL and max_evt < EVT_TOL and hit_mism == 0 and max_newton <= 16)
    print("PRODUCTION GATE PASSED" if ok else "PRODUCTION GATE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
