"""Phase A validation driver for the Travel-Fields trilinear primitive.

Run with::

    python -m travel.validation

It executes the numerical counter-example search (analytic vs high-precision reference)
across all worst-case categories, audits the 3D DDA traversal for a clean partition of
the ray parameter, prints a report, and writes CSVs under ``results/travel/``.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from . import dda, harness

RESULTS_DIR = os.path.join("results", "travel")

# Phase-A acceptance gates (analytic vs reference, double precision).
GATES = dict(recon=1e-11, depth=1e-10, inv_t=1e-7, inv_A=1e-7)


# --------------------------------------------------------------------------------------
# DDA boundary audit
# --------------------------------------------------------------------------------------
def dda_boundary_audit(n: int = 5000, seed: int = 7) -> Dict[str, float]:
    """Check that DDA yields a gap-free, overlap-free partition with 1-voxel steps."""
    rng = np.random.RandomState(seed)
    max_gap = 0.0          # |t_exit(prev) - t_enter(next)|
    max_step = 0           # max L1 change of voxel index between consecutive segments
    min_len = np.inf       # shortest emitted segment (should stay > 0)
    bad_containment = 0    # segment midpoint not inside its reported voxel
    for _ in range(n):
        p = rng.uniform(-2.0, 2.0, 3)
        w = rng.uniform(-1.0, 1.0, 3)
        nrm = np.linalg.norm(w)
        if nrm < 1e-9:
            continue
        w = w / nrm
        t0 = rng.uniform(0.0, 1.0)
        t1 = t0 + rng.uniform(0.5, 5.0)
        segs = dda.traverse(p, w, t0, t1)
        if not segs:
            continue
        # endpoints span [t0, t1]
        max_gap = max(max_gap, abs(segs[0][1] - t0), abs(segs[-1][2] - t1))
        prev = None
        for (vox, te, tx) in segs:
            min_len = min(min_len, tx - te)
            mid = p + 0.5 * (te + tx) * w
            if (int(np.floor(mid[0])) != vox[0] or int(np.floor(mid[1])) != vox[1]
                    or int(np.floor(mid[2])) != vox[2]):
                bad_containment += 1
            if prev is not None:
                max_gap = max(max_gap, abs(te - prev[2]))
                l1 = abs(vox[0] - prev[0][0]) + abs(vox[1] - prev[0][1]) + abs(vox[2] - prev[0][2])
                max_step = max(max_step, l1)
            prev = (vox, te, tx)
    return dict(max_gap=max_gap, max_voxel_step=float(max_step),
                min_seg_len=float(min_len), bad_containment=float(bad_containment))


# --------------------------------------------------------------------------------------
# Report + CSV
# --------------------------------------------------------------------------------------
def _write_csv(summary: Dict, dda_stats: Dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "phaseA_summary.csv"), "w") as f:
        f.write("category,n,recon,depth,inv_t,inv_A\n")
        for name, s in summary.items():
            a = s["agg"]
            f.write("{},{},{:.6e},{:.6e},{:.6e},{:.6e}\n".format(
                name, s["n"], a["recon"], a["depth"], a["inv_t"], a["inv_A"]))
    with open(os.path.join(RESULTS_DIR, "phaseA_worst_cases.json"), "w") as f:
        payload = {name: [dict(score=w[0], metrics=w[1], case=w[2]) for w in s["worst"]]
                   for name, s in summary.items()}
        json.dump(payload, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "phaseA_dda.csv"), "w") as f:
        f.write("max_gap,max_voxel_step,min_seg_len,bad_containment\n")
        f.write("{max_gap:.3e},{max_voxel_step:.0f},{min_seg_len:.3e},"
                "{bad_containment:.0f}\n".format(**dda_stats))


def main(n_per_category: int = 500, seed: int = 12345) -> int:
    print("=" * 70)
    print("Travel Fields -- Phase A validation (trilinear analytic traversal)")
    print("=" * 70)

    summary = harness.run_suite(n_per_category=n_per_category, seed=seed)
    print("\n[Analytic vs high-precision reference]  (max abs error per category)\n")
    print(harness.format_summary(summary))

    print("\n[3D DDA boundary audit]\n")
    dda_stats = dda_boundary_audit()
    for k, v in dda_stats.items():
        print("  {:<16}: {:.3e}".format(k, v))

    _write_csv(summary, dda_stats)
    print("\nWrote CSV + worst-case JSON to {}/".format(RESULTS_DIR))

    # gate check
    failures: List[str] = []
    for name, s in summary.items():
        for metric, thresh in GATES.items():
            if s["agg"][metric] > thresh:
                failures.append("{}.{} = {:.2e} > {:.0e}".format(
                    name, metric, s["agg"][metric], thresh))
    if dda_stats["max_gap"] > 1e-9 or dda_stats["bad_containment"] > 0 or dda_stats["min_seg_len"] <= 0:
        failures.append("dda: gap={:.2e} bad_containment={:.0f} min_len={:.2e}".format(
            dda_stats["max_gap"], dda_stats["bad_containment"], dda_stats["min_seg_len"]))

    print("\n" + "=" * 70)
    if failures:
        print("GATE FAIL:")
        for fmsg in failures:
            print("  - " + fmsg)
        print("=" * 70)
        return 1
    print("GATE PASS: all categories within Phase-A tolerances.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
