"""Faithfulness gates for the precision-parametrized traversal (harness self-validation).

Gate A: MODES['f64ref'] reproduces the frozen kernel (trace + streaming) to machine
        precision on random rays -- total tau, event position, and the crossing stream.
Gate B: the kernel's total optical depth matches an independent piecewise-GL12 oracle
        (finest-grid breakpoints enumerated without the kernel DDA) to ~1e-14.

Run:  python a3/tests/validate_float32_faithful.py
Exits nonzero if any gate fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from a3.kernel import a3_gn_kernel as K  # noqa: E402
from a3.harness.a3_gn_precision import trace_precision, MODES  # noqa: E402


def _indep_tau(med, ro, rd):
    F = med.finest_frequency
    ro = np.asarray(ro, float); rd = np.asarray(rd, float); rd = rd / np.linalg.norm(rd)
    hit, te, tx = K.intersect_box(ro, rd, med.bmin, med.bmax)
    if not hit or tx <= te:
        return None
    pts = []
    for a in range(3):
        if abs(rd[a]) < 1e-15:
            continue
        for k in range(1, int(med.world_cells[a] * F)):
            t = (k / F - ro[a]) / rd[a]
            if te < t < tx:
                pts.append(t)
    bnds = [te] + sorted(pts) + [tx]
    gx, gw = np.polynomial.legendre.leggauss(12)
    total = 0.0
    for i in range(len(bnds) - 1):
        a, b = bnds[i], bnds[i + 1]
        if b <= a:
            continue
        nodes = 0.5 * (b - a) * (gx + 1) + a
        vals = []
        for t in nodes:
            p = ro + rd * t
            s = 0.0
            for o in med.octaves:
                q = p * o.frequency
                cell = np.minimum(np.maximum(np.floor(q).astype(np.int64), 0), o.cells - 1)
                s += K.density_from_corners(q - cell, o.corners(cell))
            vals.append(s)
        total += 0.5 * (b - a) * float(np.dot(gw, vals))
    return med.sigma_t * total


def main() -> int:
    med = K.make_demo_medium()
    ref = MODES["f64ref"]
    rng = np.random.default_rng(20260718)
    max_tau = max_evt = max_oracle = 0.0
    stream_mism = seg_mism = reload_mism = 0
    n = 0
    for _ in range(300):
        center = 0.5 * (med.bmin + med.bmax)
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        ro = (rng.uniform(med.bmin, med.bmax) if rng.random() < 0.35
              else center + d * 1.5 * np.linalg.norm(med.bmax - med.bmin))
        rd = rng.uniform(med.bmin, med.bmax) - ro
        if np.linalg.norm(rd) < 1e-9:
            continue
        rd /= np.linalg.norm(rd)
        kr = K.trace(med, ro, rd)
        pr = trace_precision(med, ro, rd, ref)
        if kr.hit != pr.hit:
            print("HIT MISMATCH"); return 1
        if not kr.hit:
            continue
        n += 1
        max_tau = max(max_tau, abs(kr.total_tau - pr.total_tau))
        seg_mism += (kr.segments != pr.segments)
        reload_mism += (kr.octave_reloads != pr.octave_reloads)
        ktok = tuple(t.plane_indices for t in K.streaming_transactions(med, ro, rd, kr.t_entry, kr.t_exit))
        if ktok != pr.stream_tokens:
            stream_mism += 1
        oracle = _indep_tau(med, ro, rd)
        if oracle is not None:
            max_oracle = max(max_oracle, abs(kr.total_tau - oracle))
        if kr.total_tau > 1e-6:
            tt = 0.5 * kr.total_tau
            ke = K.trace(med, ro, rd, target_tau=tt)
            pe = trace_precision(med, ro, rd, ref, target_tau=tt)
            if ke.event_t is not None and pe.event_t is not None:
                max_evt = max(max_evt, abs(ke.event_t - pe.event_t))

    ok = (max_tau < 1e-12 and max_evt < 1e-11 and stream_mism == 0
          and seg_mism == 0 and reload_mism == 0 and max_oracle < 1e-12)
    print("Gate A (f64ref == kernel): rays=%d max|dtau|=%.2e max|d_evt|=%.2e "
          "stream_mism=%d seg_mism=%d reload_mism=%d"
          % (n, max_tau, max_evt, stream_mism, seg_mism, reload_mism))
    print("Gate B (kernel tau vs independent GL12 oracle): max|dtau|=%.2e" % max_oracle)
    print("ALL GATES PASSED" if ok else "GATE FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
