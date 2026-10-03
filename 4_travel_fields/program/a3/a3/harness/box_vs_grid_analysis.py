"""Box-grazing vs internal-grid-grazing: where does the f32 tail error originate?

For each worst ray (largest |dtau| between full32 and the true-f64 reference), instrument
the divergence between the f32 and f64 traversals:

  entry/exit face   : which box face the ray clips through, and whether f32 changed it
  entry/exit t      : |t_entry|, |t_exit| difference (box-clipping shift)
  first crossing     : whether the first internal crossing token changed
  first diff index   : first position where the crossing streams diverge (+ its t, axis)
  n_crossings        : crossed-plane count difference

Classification:
  box-grazing  : entry/exit face changed OR entry/exit t shifted materially
  grid-grazing : crossing streams diverge internally with entry/exit unchanged

This tells whether raising box-clipping precision alone would help (box) or the internal
DDA decisions are the culprit (grid) -- complementing the ablation (which showed the WHOLE
decision path must be f64).

Run:  python a3/harness/box_vs_grid_analysis.py [--json PATH]
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

from a3.harness.a3_gn_precision import trace_precision, MODES  # noqa: E402
from a3.harness.float32_transport_validation import make_medium, gen_ray  # noqa: E402


def entry_exit_faces(ro, rd, bmin, bmax):
    """Return (entry_axis, exit_axis): the box faces the ray clips through."""
    ro = np.asarray(ro, float); rd = np.asarray(rd, float)
    t_near = np.full(3, -np.inf); t_far = np.full(3, np.inf)
    for a in range(3):
        if abs(rd[a]) < 1e-30:
            continue
        t0 = (bmin[a] - ro[a]) / rd[a]
        t1 = (bmax[a] - ro[a]) / rd[a]
        t_near[a] = min(t0, t1)
        t_far[a] = max(t0, t1)
    return int(np.argmax(t_near)), int(np.argmin(t_far))


def first_divergence(tok_a, tok_b, times_a):
    """First index where crossing streams differ; return (index, t, axis) or None."""
    n = min(len(tok_a), len(tok_b))
    for i in range(n):
        if tok_a[i] != tok_b[i]:
            axis = next((ax for ax in range(3) if (tok_a[i][ax] is not None) or
                         (tok_b[i][ax] is not None)), -1)
            t = times_a[i] if i < len(times_a) else float("nan")
            return i, float(t), axis
    if len(tok_a) != len(tok_b):
        return n, float(times_a[n]) if n < len(times_a) else float("nan"), -1
    return None


def run(n_worst=12, seed=20260718):
    ref = MODES["f64ref"]; f32 = MODES["full32_div"]
    rows = []
    for F in [64, 128]:
        med = make_medium(F)
        rng = np.random.default_rng(seed + F)
        for cat in ["grazing", "near_diagonal"]:
            got, att = 0, 0
            while got < 90 and att < 5000:
                att += 1
                ro, rd = gen_ray(rng, med, cat)
                if rd is None:
                    continue
                r = trace_precision(med, ro, rd, ref)
                if not r.hit or r.total_tau <= 1e-6:
                    continue
                got += 1
                m = trace_precision(med, ro, rd, f32)
                dtau = abs(m.total_tau - r.total_tau)
                if dtau < 1e-5:
                    continue
                rn = rd / np.linalg.norm(rd)
                ea_ref, xa_ref = entry_exit_faces(ro, rn, med.bmin, med.bmax)
                ea_f32, xa_f32 = entry_exit_faces((np.asarray(ro, np.float32)),
                                                  (np.asarray(rn, np.float32)),
                                                  med.bmin, med.bmax)
                face_changed = (ea_ref != ea_f32) or (xa_ref != xa_f32)
                dte = abs(r.t_entry - m.t_entry)
                dtx = abs(r.t_exit - m.t_exit)
                fd = first_divergence(r.stream_tokens, m.stream_tokens, r.stream_times)
                box = face_changed or dte > 1e-6 or dtx > 1e-6
                rows.append(dict(
                    F=F, cat=cat, dtau=dtau,
                    m0=float(np.min(np.abs(rn))), entry_axis=ea_ref, grazed_face=(ea_ref == int(np.argmin(np.abs(rn)))),
                    face_changed=bool(face_changed), d_t_entry=dte, d_t_exit=dtx,
                    n_cross_ref=len(r.stream_tokens), n_cross_f32=len(m.stream_tokens),
                    first_diff=fd, classify=("box" if box else "grid")))
    rows.sort(key=lambda x: -x["dtau"])
    rows = rows[:n_worst]
    n_box = sum(r["classify"] == "box" for r in rows)
    print("worst %d rays: box-grazing %d, grid-grazing %d" % (len(rows), n_box, len(rows) - n_box))
    for r in rows:
        fd = r["first_diff"]
        print("  F%3d %-13s dtau %.2e m0 %.1e face_changed %s dte %.1e dtx %.1e "
              "ncross %d/%d firstdiff %s -> %s"
              % (r["F"], r["cat"], r["dtau"], r["m0"], r["face_changed"],
                 r["d_t_entry"], r["d_t_exit"], r["n_cross_ref"], r["n_cross_f32"],
                 (fd[:1] + (round(fd[1], 4), fd[2]) if fd else None), r["classify"]))
    return dict(worst=rows, n_box=n_box, n_grid=len(rows) - n_box)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    res = run()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        # first_diff tuples -> lists for json
        for r in res["worst"]:
            if r["first_diff"] is not None:
                r["first_diff"] = list(r["first_diff"])
        args.json.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print("wrote", args.json)


if __name__ == "__main__":
    main()
