"""Cost proxy for the A3-GBox64/F32 architecture: is the f64 box clip "near free"?

Architecture: f64 ray + f64 box entry/exit clip (once per ray) + everything else f32.
The only f64 work per ray is ray normalization + one slab box-clip; all per-segment work
(f32 DDA, per-octave coefficient build, GL5 density eval, terminal Newton) is f32.

We estimate C_f64_clip / C_total. Operation counts are transparent constants derived from
the frozen kernel (a3_gn_kernel.py); segment counts N_seg and octave count L are taken from
REAL traces so the ratio is grounded, not hand-waved. FP64 is charged a throughput penalty
r = (FP32 ops/s) / (FP64 ops/s): consumer GPUs ~32-64, datacenter ~2.

Run:  python a3/harness/cost_proxy.py [--json PATH]
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
from a3.harness.float32_transport_validation import make_medium, gen_ray  # noqa: E402

# --- transparent per-component FLOP constants (counted from the kernel) ----------------
# f64 work (done once per ray, in the box-only-f64 architecture):
RAY_NORMALIZE_F64 = 10     # 3 mul + 2 add + sqrt + 3 div + 2
BOX_CLIP_F64 = 26          # 3 axes x (2 sub + 2 div + 2 min/max) + running min/max + clamp
F64_PER_RAY = RAY_NORMALIZE_F64 + BOX_CLIP_F64

# f32 work per segment:
DDA_ADVANCE_F32 = 8        # recompute t_max (sub+div) + min-of-3 + tie tests
# GL5 density eval: 5 nodes; per node: point (6) + per-octave [local(6)+fade(12)+trilinear(21)]
GL5_PER_NODE_BASE = 6
GL5_PER_NODE_PER_OCTAVE = 39
GL5_NODES = 5
GL5_WEIGHTDOT = 9
# terminal (once per ray, only when a free-path event is located):
TERMINAL_PER_OCTAVE = 100  # fade_poly x3 + polymul cascade + assembly for degree-9 build
TERMINAL_FIXED = 160       # integrate + ~3 safeguarded-Newton iters (Horner deg-10/deg-9)


def f32_per_segment(L):
    gl5 = GL5_NODES * (GL5_PER_NODE_BASE + GL5_PER_NODE_PER_OCTAVE * L) + GL5_WEIGHTDOT
    return DDA_ADVANCE_F32 + gl5


def f32_per_ray(n_seg, L, event):
    seg = n_seg * f32_per_segment(L)
    term = (TERMINAL_PER_OCTAVE * L + TERMINAL_FIXED) if event else 0
    return seg + term


def measure_segments(f_max, n=40, seed=20260718):
    med = make_medium(f_max)
    rng = np.random.default_rng(seed + f_max)
    segs = []
    for cat in ["random", "random", "grazing"]:
        got, att = 0, 0
        while got < n and att < n * 30:
            att += 1
            ro, rd = gen_ray(rng, med, cat)
            if rd is None:
                continue
            r = K.trace(med, ro, rd)
            if not r.hit:
                continue
            got += 1
            segs.append(r.segments)
    L = len(med.octaves)
    return int(np.median(segs)), L


def run(seed=20260718):
    penalties = [2, 8, 32, 64]
    out = {"f64_per_ray_ops": F64_PER_RAY, "fp64_penalties": penalties, "cases": []}
    print("f64 ops/ray (normalize+clip) = %d" % F64_PER_RAY)
    print("%-7s %-4s %-8s %-10s  %s" % ("F", "L", "med_seg", "scenario", "  ".join("r=%d" % r for r in penalties)))
    for f_max in [16, 32, 64, 128]:
        med_seg, L = measure_segments(f_max, seed=seed)
        for scen, nseg, event in [("camera", med_seg, False),
                                  ("shadow", max(8, med_seg // 4), True)]:
            c32 = f32_per_ray(nseg, L, event)
            ratios = {}
            row = []
            for r in penalties:
                c64 = F64_PER_RAY * r
                ratio = c64 / (c64 + c32)
                ratios["r%d" % r] = ratio
                row.append("%.2f%%" % (100 * ratio))
            out["cases"].append(dict(F=f_max, L=L, med_seg=med_seg, scenario=scen,
                                     n_seg=nseg, f32_ops=c32, ratios=ratios))
            print("F=%-5d %-4d %-8d %-10s  %s" % (f_max, L, med_seg, scen, "  ".join(row)))
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
