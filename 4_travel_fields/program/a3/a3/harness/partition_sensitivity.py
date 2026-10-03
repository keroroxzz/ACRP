"""Targeted operator-taxonomy validation (collab task b, reframed).

Three operator classes applied to the continuous cubic-fade density rho(t) along a ray:

  ReLU (value-continuous, 1-Lipschitz) : op(rho) = max(0, rho - theta)
  hard threshold (jump-sensitive)       : op(rho) = rho * [rho > theta]     (jump J ~ theta)
  brick / sparse activation (repr.-sens.): op(rho) = rho * active(cell)       (jump J ~ rho, at
                                            every cell boundary; branch is on discrete identity)

We inject a controlled ray-direction perturbation eps (models the representation error), and
for each operator measure the transport error against the unperturbed ray:

  E_tau = sigma_t * integral | op(rho_pert) - op(rho_orig) | dt .

Validation targets:
  * ReLU:  E_tau <= sigma_t * integral |rho_pert - rho_orig| dt   (Lipschitz; jump-independent)
  * jump ops:  E_tau ~ sigma_t * sum_i J_i |B_i|                   (jump x uncertain interval),
    where B_i are the branch-flip intervals; verify E_tau <= that bound and scales with the jump.
  * ReLU error << hard ~ brick, confirming value-continuous operators tolerate perturbation
    while jump / representation-sensitive operators do not (need branch-root partition, not bits).

Run:  python a3/harness/partition_sensitivity.py [--json PATH]
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

EPS_LIST = [1e-4, 1e-3, 1e-2]


def world_density(med, p):
    total = 0.0
    for o in med.octaves:
        q = p * o.frequency
        cell = np.minimum(np.maximum(np.floor(q).astype(np.int64), 0), o.cells - 1)
        total += K.density_from_corners(q - cell, o.corners(cell))
    return total


def brick_active(med, p):
    # per finest-cell binary activation (deterministic hash), models sparse/brick occupancy
    F = med.finest_frequency
    cell = np.floor(p * F).astype(np.int64)
    h = (np.sin(np.dot(cell, [12.9898, 78.233, 37.719])) * 43758.5453)
    return (h - np.floor(h)) > 0.5


def perturb_dir(rd, eps, rng):
    perp = rng.normal(size=3)
    perp -= perp.dot(rd) * rd
    perp /= np.linalg.norm(perp)
    return (rd + eps * perp) / np.linalg.norm(rd + eps * perp)


def run(n=60, seed=20260718, samples=800):
    med = make_medium(64)
    sigma = med.sigma_t
    rng = np.random.default_rng(seed)
    acc = {e: {op: dict(E=[], bound=[]) for op in ["relu", "hard", "brick"]} for e in EPS_LIST}
    got, att = 0, 0
    while got < n and att < n * 40:
        att += 1
        ro, rd = gen_ray(rng, med, "random")
        if rd is None:
            continue
        rd = rd / np.linalg.norm(rd)
        hit, te, tx = K.intersect_box(np.asarray(ro, float), rd, med.bmin, med.bmax)
        if not hit or tx - te < 1e-2:
            continue
        ts = np.linspace(te, tx, samples)
        dt = (tx - te) / (samples - 1)
        r0 = np.array([world_density(med, ro + rd * t) for t in ts])
        if r0.max() <= 1e-6:
            continue
        a0 = np.array([brick_active(med, ro + rd * t) for t in ts])
        got += 1
        theta = 0.5 * (r0.min() + r0.max())
        for eps in EPS_LIST:
            rp = perturb_dir(rd, eps, rng)
            rpp = np.array([world_density(med, ro + rp * t) for t in ts])
            ap = np.array([brick_active(med, ro + rp * t) for t in ts])
            # ReLU
            relu0 = np.maximum(0.0, r0 - theta); relup = np.maximum(0.0, rpp - theta)
            E_relu = sigma * np.abs(relup - relu0).sum() * dt
            bound_relu = sigma * np.abs(rpp - r0).sum() * dt  # Lipschitz (1-Lipschitz op)
            acc[eps]["relu"]["E"].append(E_relu); acc[eps]["relu"]["bound"].append(bound_relu)
            lipschitz = sigma * np.abs(rpp - r0).sum() * dt  # continuous part, shared by all
            # hard threshold: jump J ~ local value at flip; bound = Lipschitz + sigma*sum J|B|
            b0 = r0 > theta; bp = rpp > theta
            hard0 = r0 * b0; hardp = rpp * bp
            E_hard = sigma * np.abs(hardp - hard0).sum() * dt
            flip_h = (b0 != bp)
            jump_h = sigma * np.maximum(r0[flip_h], rpp[flip_h]).sum() * dt if flip_h.any() else 0.0
            acc[eps]["hard"]["E"].append(E_hard)
            acc[eps]["hard"]["bound"].append(lipschitz + jump_h)
            # brick / sparse: jump J ~ rho at every activated cell boundary (representation-sensitive)
            brick0 = r0 * a0; brickp = rpp * ap
            E_brick = sigma * np.abs(brickp - brick0).sum() * dt
            flip_b = (a0 != ap)
            jump_b = sigma * np.maximum(r0[flip_b], rpp[flip_b]).sum() * dt if flip_b.any() else 0.0
            acc[eps]["brick"]["E"].append(E_brick)
            acc[eps]["brick"]["bound"].append(lipschitz + jump_b)

    def stat(v):
        return (float(np.percentile(v, 50)), float(np.percentile(v, 99)), float(max(v))) if v else (0, 0, 0)

    out = {"n": got, "eps": {}}
    print("operator taxonomy: %d rays, sigma=%.2f" % (got, sigma))
    print("%-6s %-7s %-11s %-11s %-11s %s" % ("eps", "op", "E_p50", "E_max", "bound_max", "E<=bound"))
    for eps in EPS_LIST:
        out["eps"]["%.0e" % eps] = {}
        for op in ["relu", "hard", "brick"]:
            E = acc[eps][op]["E"]; B = acc[eps][op]["bound"]
            p50, _, emax = stat(E)
            _, _, bmax = stat(B)
            viol = sum(1 for e, b in zip(E, B) if e > b * 1.05 + 1e-12)
            out["eps"]["%.0e" % eps][op] = dict(E_p50=p50, E_max=emax, bound_max=bmax, violations=viol)
            print("%-6s %-7s %.3e   %.3e   %.3e   %s" % (
                "%.0e" % eps, op, p50, emax, bmax, "yes" if viol == 0 else "NO(%d)" % viol))
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
