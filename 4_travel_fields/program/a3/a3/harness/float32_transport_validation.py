"""Full-float32 transactional transport validation (handoff Sec.18).

Scales up the frozen mixed-precision pilot (A3_mixed_precision_event_ordering.md, 6-8 rays)
to hundreds of rays across F_max in {16,32,64,128} and six ray categories, and reports
event-TOPOLOGY mismatch separately from TRANSPORT error (|dtau|, event position, free-path
distance), per Invariant Sec.19.12.

Ground truth: the float64 reference path (== the frozen kernel, verified by the
faithfulness gate).  Modes compared: f32_div, f32_cross, mixed64.

Run:  python a3/harness/float32_transport_validation.py [--rays-per-category N] [--json PATH]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a3.kernel import a3_gn_kernel as K  # noqa: E402
from a3.harness.a3_gn_precision import trace_precision, MODES  # noqa: E402

F_MAX_LIST = [16, 32, 64, 128]
# Pipeline modes isolate DDA-clock precision (clock32_*) from full-f32 integration (full32_*).
PIPELINE_MODES = ["clock32_div", "clock32_cross", "full32_div", "full32_cross"]
TARGET_FRACTIONS = [0.2, 0.5, 0.8]
CATEGORIES = ["random", "axis_aligned", "near_axis", "exact_diagonal",
              "near_diagonal", "grazing"]


def make_medium(f_max: int, world: int = 3, seed: int = 20260718, sigma: float = 1.2):
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
        lat[rng.random(size=lat.shape) < 0.35] = 0.0
        octaves.append(K.Octave(lat, fr, a))
    return K.Medium(tuple(octaves), sigma, wc)


# --------------------------------------------------------------------------------------
# ray generators (local frame; origins fed directly per Sec.9)
# --------------------------------------------------------------------------------------
def _toward_box(rng, med, origin):
    target = rng.uniform(med.bmin, med.bmax)
    rd = target - origin
    n = np.linalg.norm(rd)
    return rd / n if n > 1e-9 else None


def gen_ray(rng, med, category):
    center = 0.5 * (med.bmin + med.bmax)
    diag = np.linalg.norm(med.bmax - med.bmin)
    if category == "random":
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        ro = center + d * 1.5 * diag
        rd = _toward_box(rng, med, ro)
    elif category == "axis_aligned":
        axis = rng.integers(3)
        rd = np.zeros(3)
        rd[axis] = 1.0 if rng.random() < 0.5 else -1.0
        ro = rng.uniform(med.bmin, med.bmax)
        ro[axis] = med.bmin[axis] - 1.0 if rd[axis] > 0 else med.bmax[axis] + 1.0
    elif category == "near_axis":
        axis = rng.integers(3)
        rd = rng.normal(size=3) * 1e-4
        rd[axis] = 1.0 if rng.random() < 0.5 else -1.0
        rd /= np.linalg.norm(rd)
        ro = center + (-rd) * 1.5 * diag
    elif category == "exact_diagonal":
        signs = np.array([1.0 if rng.random() < 0.5 else -1.0 for _ in range(3)])
        rd = signs / math.sqrt(3.0)
        ro = center - rd * 1.5 * diag
    elif category == "near_diagonal":
        signs = np.array([1.0 if rng.random() < 0.5 else -1.0 for _ in range(3)])
        rd = signs / math.sqrt(3.0) + rng.normal(size=3) * 1e-5
        rd /= np.linalg.norm(rd)
        ro = center - rd * 1.5 * diag
    elif category == "grazing":
        axis = rng.integers(3)
        rd = rng.normal(size=3)
        rd[axis] = rng.choice([-1.0, 1.0]) * 10.0 ** rng.uniform(-6, -3)
        rd /= np.linalg.norm(rd)
        # start near the grazed face
        ro = rng.uniform(med.bmin, med.bmax)
        ro[axis] = med.bmin[axis] + 1e-4 if rng.random() < 0.5 else med.bmax[axis] - 1e-4
        ro = ro - rd * 0.5 * diag
    else:
        raise ValueError(category)
    return ro, rd


def _percentile(vals, q):
    return float(np.percentile(vals, q)) if vals else 0.0


def _stats(tau_err, evt_err, topo, rays):
    return dict(
        rays=rays,
        topology_mismatches=topo,
        p99_tau_abs_error=_percentile(tau_err, 99),
        max_tau_abs_error=max(tau_err) if tau_err else 0.0,
        p99_event_distance_error=_percentile(evt_err, 99),
        max_event_distance_error=max(evt_err) if evt_err else 0.0,
    )


def run(rays_per_category: int = 20, seed: int = 20260718) -> Dict:
    """Decompose float32 transport error into ray-quantization, DDA-clock, and full-f32.

    Ground truth = full-f64 pipeline on the true f64 ray (== kernel).  For each ray we run
    the true-f64 reference, mixed64 (f32 ray, f64 pipeline), and the four pipeline modes.
    Pipeline modes are scored both end-to-end (vs true f64) and pipeline-only (vs the same
    f32 ray = mixed64, reproducing the frozen same-ray convention).
    """
    ref_cfg = MODES["f64ref"]
    rq_cfg = MODES["mixed64"]
    out = {"rays_per_category": rays_per_category, "categories": CATEGORIES,
           "ground_truth": "full-f64 on true f64 ray (== kernel)", "cases": {}}
    for f_max in F_MAX_LIST:
        med = make_medium(f_max)
        rng = np.random.default_rng(seed + f_max)
        rho_max = float(sum(o.amplitude for o in med.octaves))

        rq = dict(tau=[], evt=[], topo=0, rays=0)  # ray-quantization (mixed64 vs true)
        pipe = {m: dict(e2e_tau=[], e2e_evt=[], e2e_topo=0,
                        po_tau=[], po_evt=[], po_topo=0,
                        newton=[], fallbacks=0, rays=0) for m in PIPELINE_MODES}

        for category in CATEGORIES:
            got, attempts = 0, 0
            while got < rays_per_category and attempts < rays_per_category * 40:
                attempts += 1
                ro, rd = gen_ray(rng, med, category)
                if rd is None:
                    continue
                ref = trace_precision(med, ro, rd, ref_cfg)
                if not ref.hit or ref.total_tau <= 1e-6:
                    continue
                got += 1
                xi = -math.log(1.0 - rng.random())
                targets = [fr * ref.total_tau for fr in TARGET_FRACTIONS]
                if 0 < xi < ref.total_tau:
                    targets.append(xi)
                ref_ev = [trace_precision(med, ro, rd, ref_cfg, target_tau=t).event_t for t in targets]

                rqr = trace_precision(med, ro, rd, rq_cfg)
                rq_ev = [trace_precision(med, ro, rd, rq_cfg, target_tau=t).event_t for t in targets]
                rq["rays"] += 1
                rq["tau"].append(abs(rqr.total_tau - ref.total_tau))
                rq["topo"] += (rqr.stream_tokens != ref.stream_tokens)
                for e, er in zip(rq_ev, ref_ev):
                    if e is not None and er is not None:
                        rq["evt"].append(abs(e - er))

                for m in PIPELINE_MODES:
                    cfg = MODES[m]
                    mr = trace_precision(med, ro, rd, cfg)
                    m_ev = [trace_precision(med, ro, rd, cfg, target_tau=t) for t in targets]
                    P = pipe[m]
                    P["rays"] += 1
                    P["e2e_tau"].append(abs(mr.total_tau - ref.total_tau))
                    P["po_tau"].append(abs(mr.total_tau - rqr.total_tau))
                    P["e2e_topo"] += (mr.stream_tokens != ref.stream_tokens)
                    P["po_topo"] += (mr.stream_tokens != rqr.stream_tokens)
                    for pe, er, eq in zip(m_ev, ref_ev, rq_ev):
                        if pe.event_t is not None:
                            if er is not None:
                                P["e2e_evt"].append(abs(pe.event_t - er))
                            if eq is not None:
                                P["po_evt"].append(abs(pe.event_t - eq))
                        P["newton"].append(pe.newton_iterations)
                        P["fallbacks"] += pe.newton_fallbacks

        case = {
            "rho_max": rho_max,
            "ray_quantization": _stats(rq["tau"], rq["evt"], rq["topo"], rq["rays"]),
            "pipeline": {},
        }
        for m in PIPELINE_MODES:
            P = pipe[m]
            case["pipeline"][m] = {
                "rays": P["rays"],
                "end_to_end": _stats(P["e2e_tau"], P["e2e_evt"], P["e2e_topo"], P["rays"]),
                "pipeline_only": _stats(P["po_tau"], P["po_evt"], P["po_topo"], P["rays"]),
                "mean_newton_iterations": float(np.mean(P["newton"])) if P["newton"] else 0.0,
                "max_newton_iterations": int(max(P["newton"])) if P["newton"] else 0,
                "newton_fallbacks": P["fallbacks"],
            }
        out["cases"]["F%d" % f_max] = case
        print("F=%d done" % f_max, flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rays-per-category", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260718)
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    result = run(args.rays_per_category, args.seed)
    text = json.dumps(result, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
