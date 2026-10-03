"""Generalization sweep: does the decision64 architecture cap the grazing tail to the
f32-arithmetic floor across F, world extent, ray length, sigma_t, and sparsity?

Architecture under test (collab-agreed):
    decision64 = f64 ray + f64 clipping + f64 DDA decisions + f32 density/GL5/Newton.

For each config we sample tail-bearing rays (grazing + near_diagonal + random) and report:
  full32_tail      = max|dtau| under all-f32 (the catastrophic tail),
  decision64_tail  = max|dtau| under decision64 (should collapse to the f32 arith floor),
  arith_floor      = max|dtau| of full32 on NON-grazing (random) rays (the f32 floor),
so the claim "decision64 caps tail to floor" is: decision64_tail ~ arith_floor << full32_tail.

Ground truth is always the true-f64 pipeline (== kernel).

Run:  python a3/harness/generalization_sweep.py [--rays N] [--json PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a3.kernel import a3_gn_kernel as K  # noqa: E402
from a3.harness.a3_gn_precision import trace_precision, MODES  # noqa: E402
from a3.harness.float32_transport_validation import gen_ray  # noqa: E402


def make_medium_ex(f_max, world=3, sigma=1.2, sparsity=0.35, seed=20260718):
    rng = np.random.default_rng(seed)
    wc = np.array([world] * 3, dtype=np.int64)
    freqs = []
    f = 1
    while f <= f_max:
        freqs.append(f)
        f *= 2
    amps = [0.5 ** i for i in range(len(freqs))]
    octaves = []
    for fr, a in zip(freqs, amps):
        lat = rng.uniform(0.0, 1.0, size=tuple(wc * fr + 1))
        lat[rng.random(size=lat.shape) < sparsity] = 0.0
        octaves.append(K.Octave(lat, fr, a))
    return K.Medium(tuple(octaves), sigma, wc)


def _tail_for_medium(med, rays, seed):
    ref = MODES["f64ref"]
    d64 = MODES["decision64"]
    f32 = MODES["full32_div"]
    rng = np.random.default_rng(seed)
    full_tail, dec_tail, floor = [], [], []
    for cat in ["grazing", "near_diagonal", "random"]:
        got, att = 0, 0
        while got < rays and att < rays * 60:
            att += 1
            ro, rd = gen_ray(rng, med, cat)
            if rd is None:
                continue
            r = trace_precision(med, ro, rd, ref)
            if not r.hit or r.total_tau <= 1e-6:
                continue
            got += 1
            ef = abs(trace_precision(med, ro, rd, f32).total_tau - r.total_tau)
            ed = abs(trace_precision(med, ro, rd, d64).total_tau - r.total_tau)
            full_tail.append(ef)
            dec_tail.append(ed)
            if cat == "random":
                floor.append(ef)
    return (max(full_tail) if full_tail else 0.0,
            max(dec_tail) if dec_tail else 0.0,
            max(floor) if floor else 0.0)


def run(rays=25, seed=20260718):
    configs = []
    # baseline then one-axis-at-a-time
    for F in [16, 32, 64, 128]:
        configs.append(dict(tag="F=%d" % F, F=F, world=3, sigma=1.2, sparsity=0.35))
    for w in [2, 4]:
        configs.append(dict(tag="world=%d" % w, F=64, world=w, sigma=1.2, sparsity=0.35))
    for s in [0.5, 3.0]:
        configs.append(dict(tag="sigma=%.1f" % s, F=64, world=3, sigma=s, sparsity=0.35))
    for sp in [0.2, 0.6]:
        configs.append(dict(tag="sparsity=%.2f" % sp, F=64, world=3, sigma=1.2, sparsity=sp))

    out = {"rays_per_category": rays, "configs": []}
    for c in configs:
        med = make_medium_ex(c["F"], c["world"], c["sigma"], c["sparsity"], seed=seed)
        ft, dt, fl = _tail_for_medium(med, rays, seed + c["F"] + c["world"])
        rec = dict(c, full32_tail=ft, decision64_tail=dt, arith_floor=fl,
                   tail_reduction=(ft / dt if dt > 0 else float("inf")))
        out["configs"].append(rec)
        print("%-14s full32_tail %.2e  decision64_tail %.2e  floor %.2e  reduction %.0fx"
              % (c["tag"], ft, dt, fl, rec["tail_reduction"]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rays", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260718)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    res = run(args.rays, args.seed)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print("wrote", args.json)


if __name__ == "__main__":
    main()
