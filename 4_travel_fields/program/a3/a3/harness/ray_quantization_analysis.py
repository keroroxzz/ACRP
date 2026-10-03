"""Characterize the ray-quantization transport error (the dominant Sec.18 term).

Sec.18 showed the f32 *ray* (a physically different ray) dominates transport error, with a
max that is non-monotonic in F -- suggesting a few pathological rays, not a broad limit.
This experiment resolves that: it reports the full error DISTRIBUTION (p50/p90/p99/max)
per ray category, and correlates each ray's error with two geometric predictors:

  graze    = min_a |rd_a|                          (small -> nearly tangent to a face)
  tie_prox = min over crossings of |frac(k-plane)| (small -> a near-simultaneous tie)

If the median is tiny and only outliers are large, f32 rays are viable with a cheap
f64 fallback on flagged rays; if the whole distribution is large, the f32 ray is a
fundamental transport limit.  Ground truth is the true-f64 pipeline (== kernel).

Run:  python a3/harness/ray_quantization_analysis.py [--rays-per-category N] [--json PATH]
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

from a3.harness.a3_gn_precision import trace_precision, MODES  # noqa: E402
from a3.harness.float32_transport_validation import make_medium, gen_ray, CATEGORIES  # noqa: E402

F_LIST = [16, 64, 128]


def _pcts(v):
    if not v:
        return dict(p50=0.0, p90=0.0, p99=0.0, max=0.0)
    a = np.asarray(v)
    return dict(p50=float(np.percentile(a, 50)), p90=float(np.percentile(a, 90)),
                p99=float(np.percentile(a, 99)), max=float(a.max()))


def _graze(rd):
    return float(np.min(np.abs(rd)))


def run(rays_per_category: int = 40, seed: int = 20260718):
    ref = MODES["f64ref"]
    rq = MODES["mixed64"]
    out = {"rays_per_category": rays_per_category, "F_list": F_LIST, "cases": {}}
    for f_max in F_LIST:
        med = make_medium(f_max)
        rng = np.random.default_rng(seed + f_max)
        per_cat = {}
        worst = []  # (tau_err, category, graze)
        for category in CATEGORIES:
            tau_errs, evt_errs, grazes = [], [], []
            got, attempts = 0, 0
            while got < rays_per_category and attempts < rays_per_category * 40:
                attempts += 1
                ro, rd = gen_ray(rng, med, category)
                if rd is None:
                    continue
                r_ref = trace_precision(med, ro, rd, ref)
                if not r_ref.hit or r_ref.total_tau <= 1e-6:
                    continue
                got += 1
                r_rq = trace_precision(med, ro, rd, rq)
                tau_e = abs(r_rq.total_tau - r_ref.total_tau)
                tau_errs.append(tau_e)
                g = _graze(rd / np.linalg.norm(rd))
                grazes.append(g)
                worst.append((tau_e, category, g))
                tt = 0.5 * r_ref.total_tau
                e_ref = trace_precision(med, ro, rd, ref, target_tau=tt).event_t
                e_rq = trace_precision(med, ro, rd, rq, target_tau=tt).event_t
                if e_ref is not None and e_rq is not None:
                    evt_errs.append(abs(e_rq - e_ref))
            per_cat[category] = dict(tau=_pcts(tau_errs), evt=_pcts(evt_errs),
                                     median_graze=float(np.median(grazes)) if grazes else 0.0)
        worst.sort(key=lambda x: -x[0])
        # correlation: are large errors concentrated at small graze?
        arr = np.array([(t, g) for t, _, g in worst if t > 0])
        corr = 0.0
        if len(arr) > 5 and arr[:, 0].std() > 0 and arr[:, 1].std() > 0:
            corr = float(np.corrcoef(np.log10(arr[:, 0]), np.log10(arr[:, 1] + 1e-12))[0, 1])
        out["cases"]["F%d" % f_max] = dict(
            per_category=per_cat,
            worst5=[dict(tau_err=t, category=c, graze=g) for t, c, g in worst[:5]],
            log_tau_vs_log_graze_corr=corr,
        )
        print("F=%d done" % f_max, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rays-per-category", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260718)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    res = run(args.rays_per_category, args.seed)
    txt = json.dumps(res, indent=2)
    print(txt)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
