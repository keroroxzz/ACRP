
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

_GLX, _GLW = np.polynomial.legendre.leggauss(8)


def fade(x):
    return x * x * (3.0 - 2.0 * x)


@dataclass
class Octave:
    lattice: np.ndarray
    frequency: int
    amplitude: float

    @property
    def cells(self):
        return np.array(self.lattice.shape, dtype=int) - 1

    def density(self, p_world):
        q = np.asarray(p_world, float) * self.frequency
        cell = np.floor(q).astype(int)
        for a in range(3):
            if cell[a] == self.cells[a]:
                cell[a] -= 1
        if np.any(cell < 0) or np.any(cell >= self.cells):
            return 0.0
        f = np.clip(q - cell, 0.0, 1.0)
        w = fade(f)
        x, y, z = map(int, cell)
        c = self.lattice[x:x+2, y:y+2, z:z+2]
        x00 = (1-w[0])*c[0,0,0] + w[0]*c[1,0,0]
        x10 = (1-w[0])*c[0,1,0] + w[0]*c[1,1,0]
        x01 = (1-w[0])*c[0,0,1] + w[0]*c[1,0,1]
        x11 = (1-w[0])*c[0,1,1] + w[0]*c[1,1,1]
        y0 = (1-w[1])*x00 + w[1]*x10
        y1 = (1-w[1])*x01 + w[1]*x11
        return float(self.amplitude * ((1-w[2])*y0 + w[2]*y1))


@dataclass
class Medium:
    octaves: List[Octave]
    sigma_t: float
    world_cells: np.ndarray

    @property
    def box_min(self):
        return np.zeros(3)

    @property
    def box_max(self):
        return self.world_cells.astype(float)


def intersect_box(ro, rd, bmin, bmax):
    te, tx = -math.inf, math.inf
    for a in range(3):
        if abs(rd[a]) < 1e-15:
            if ro[a] < bmin[a] or ro[a] > bmax[a]:
                return False, 0.0, 0.0
            continue
        t0 = (bmin[a]-ro[a])/rd[a]
        t1 = (bmax[a]-ro[a])/rd[a]
        te = max(te, min(t0,t1))
        tx = min(tx, max(t0,t1))
    te = max(te, 0.0)
    return tx >= te, te, tx


def boundary_times_for_level(medium, ro, rd, te, tx, level):
    f = medium.octaves[level].frequency
    times = []
    for axis in range(3):
        if abs(rd[axis]) < 1e-15:
            continue
        max_plane = int(medium.world_cells[axis] * f)
        for k in range(1, max_plane):
            plane = k / float(f)
            t = (plane - ro[axis]) / rd[axis]
            if te + 1e-13 < t < tx - 1e-13:
                times.append(float(t))
    return times


def unique_sorted(values):
    values = sorted(values)
    out = [values[0]]
    for v in values[1:]:
        if abs(v-out[-1]) > 2e-12*max(1.0,abs(v)):
            out.append(v)
    return out


def partition_for_levels(medium, ro, rd, te, tx, levels):
    times = [te, tx]
    for level in levels:
        times.extend(boundary_times_for_level(medium, ro, rd, te, tx, level))
    u = unique_sorted(times)
    return list(zip(u[:-1], u[1:]))


def gl_integral_octave(medium, octave_index, ro, rd, a, b):
    if b <= a:
        return 0.0
    o = medium.octaves[octave_index]
    nodes = 0.5*(b-a)*(_GLX+1.0)+a
    vals = np.array([medium.sigma_t*o.density(ro+rd*t) for t in nodes])
    return float(0.5*(b-a)*np.dot(_GLW, vals))


def exact_interval_tau(medium, ro, rd, a, b, levels):
    return sum(gl_integral_octave(medium, l, ro, rd, a, b) for l in levels)


def octave_bound_on_segment(octave, ro, rd, a, b):
    p0 = ro + rd*a
    p1 = ro + rd*b
    qmin = np.minimum(p0,p1)*octave.frequency
    qmax = np.maximum(p0,p1)*octave.frequency

    i0 = np.floor(qmin).astype(int)
    i1 = np.floor(qmax).astype(int) + 1
    i0 = np.maximum(i0, 0)
    i1 = np.minimum(i1, octave.cells)

    sl = tuple(slice(int(i0[k]), int(i1[k])+1) for k in range(3))
    return float(octave.amplitude*np.max(octave.lattice[sl]))


@dataclass
class Cost:
    segments: int = 0
    octave_integrals: int = 0
    bound_queries: int = 0


def method_A(medium, ro, rd, te, tx):
    levels = list(range(len(medium.octaves)))
    segs = partition_for_levels(medium, ro, rd, te, tx, levels)
    tau = 0.0
    cost = Cost()
    for a,b in segs:
        cost.segments += 1
        tau += exact_interval_tau(medium, ro, rd, a, b, levels)
        cost.octave_integrals += len(levels)
    return tau, cost


def method_B(medium, ro, rd, te, tx, cutoff):
    levels = list(range(cutoff))
    segs = partition_for_levels(medium, ro, rd, te, tx, levels)
    tau = 0.0
    cost = Cost()
    for a,b in segs:
        cost.segments += 1
        tau += exact_interval_tau(medium, ro, rd, a, b, levels)
        cost.octave_integrals += len(levels)
    return tau, cost


def method_C(medium, ro, rd, te, tx, backbone, eps_tau):
    """
    Exact low-frequency backbone plus recursively bounded high-frequency residual.

    For an interval:
      1. integrate currently active levels exactly;
      2. bound all omitted levels;
      3. if residual optical-depth bound <= eps_tau, stop;
      4. otherwise split by the next octave's boundaries and activate it.

    The total per-ray error is controlled by distributing eps_tau proportionally
    to interval length, so accepted residual bounds sum to <= eps_tau.
    """
    total_length = max(tx-te, 1e-30)
    active = list(range(backbone))
    root_segments = partition_for_levels(medium, ro, rd, te, tx, active)
    cost = Cost()

    def recurse(a, b, next_level):
        cost.segments += 1
        tau_local = exact_interval_tau(medium, ro, rd, a, b, list(range(next_level)))
        cost.octave_integrals += next_level

        omitted_bound = 0.0
        for l in range(next_level, len(medium.octaves)):
            cost.bound_queries += 1
            omitted_bound += octave_bound_on_segment(medium.octaves[l], ro, rd, a, b)

        tau_bound = medium.sigma_t*(b-a)*omitted_bound
        local_budget = eps_tau*((b-a)/total_length)

        if next_level >= len(medium.octaves) or tau_bound <= local_budget:
            return tau_local

        # Refine only by the next octave's cell boundaries.
        times = [a,b]
        times.extend(boundary_times_for_level(medium, ro, rd, a, b, next_level))
        times = unique_sorted(times)
        return sum(recurse(x,y,next_level+1) for x,y in zip(times[:-1],times[1:]))

    tau = sum(recurse(a,b,backbone) for a,b in root_segments)
    return tau, cost


def random_ray(medium, rng):
    center = 0.5*(medium.box_min+medium.box_max)
    extent = medium.box_max-medium.box_min
    if rng.random() < 0.35:
        ro = rng.uniform(medium.box_min, medium.box_max)
    else:
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        ro = center + d*1.5*np.linalg.norm(extent)
    target = rng.uniform(medium.box_min, medium.box_max)
    rd = target-ro; rd /= np.linalg.norm(rd)
    return ro,rd


@dataclass
class Summary:
    seed: int
    rays: int
    octaves: int
    frequencies: List[int]
    amplitudes: List[float]
    method_A_mean_segments: float
    method_A_mean_octave_integrals: float
    method_B_results: dict
    method_C_results: dict
    all_bounds_respected: bool


def stats(errors, trans_errors, segs, evals, bounds, ref_segs, ref_evals):
    return {
        "mean_abs_tau_error": float(np.mean(errors)),
        "p95_abs_tau_error": float(np.percentile(errors,95)),
        "max_abs_tau_error": float(np.max(errors)),
        "p95_transmittance_error": float(np.percentile(trans_errors,95)),
        "max_transmittance_error": float(np.max(trans_errors)),
        "mean_segments": float(np.mean(segs)),
        "mean_octave_integrals": float(np.mean(evals)),
        "mean_bound_queries": float(np.mean(bounds)),
        "segment_reduction_vs_A": float(1.0 - np.mean(segs)/np.mean(ref_segs)),
        "integral_reduction_vs_A": float(1.0 - np.mean(evals)/np.mean(ref_evals)),
    }


def run(seed=20260718, rays=220):
    rng = np.random.default_rng(seed)
    world_cells = np.array([4,4,4])
    freqs = [1,2,4,8]
    amps = [1.0,0.5,0.25,0.125]
    octaves = []
    for f,a in zip(freqs,amps):
        lattice = rng.uniform(0.0,1.0,size=tuple(world_cells*f+1))
        lattice[rng.random(size=lattice.shape)<0.40] = 0.0
        octaves.append(Octave(lattice,f,a))
    medium = Medium(octaves,1.25,world_cells)

    A_tau=[]; A_seg=[]; A_eval=[]
    rays_data=[]
    for _ in range(rays):
        ro,rd=random_ray(medium,rng)
        hit,te,tx=intersect_box(ro,rd,medium.box_min,medium.box_max)
        if not hit or tx<=te: continue
        tau,cost=method_A(medium,ro,rd,te,tx)
        A_tau.append(tau); A_seg.append(cost.segments); A_eval.append(cost.octave_integrals)
        rays_data.append((ro,rd,te,tx,tau))

    B_results={}
    for cutoff in [1,2,3]:
        errs=[]; terr=[]; seg=[]; eva=[]; bq=[]
        for ro,rd,te,tx,ref in rays_data:
            tau,cost=method_B(medium,ro,rd,te,tx,cutoff)
            errs.append(abs(tau-ref))
            terr.append(abs(math.exp(-tau)-math.exp(-ref)))
            seg.append(cost.segments); eva.append(cost.octave_integrals); bq.append(0)
        B_results[f"K{cutoff}"]=stats(errs,terr,seg,eva,bq,A_seg,A_eval)

    C_results={}
    all_bounds=True
    for backbone in [1,2]:
        for eps in [1e-1,3e-2,1e-2,3e-3]:
            errs=[]; terr=[]; seg=[]; eva=[]; bq=[]
            for ro,rd,te,tx,ref in rays_data:
                tau,cost=method_C(medium,ro,rd,te,tx,backbone,eps)
                err=abs(tau-ref)
                errs.append(err)
                terr.append(abs(math.exp(-tau)-math.exp(-ref)))
                seg.append(cost.segments); eva.append(cost.octave_integrals); bq.append(cost.bound_queries)
                if err > eps + 2e-10:
                    all_bounds=False
            C_results[f"B{backbone}_eps{eps:g}"]=stats(errs,terr,seg,eva,bq,A_seg,A_eval)
            C_results[f"B{backbone}_eps{eps:g}"]["declared_tau_bound"]=eps
            C_results[f"B{backbone}_eps{eps:g}"]["bound_respected"]=bool(max(errs)<=eps+2e-10)

    return Summary(
        seed=seed,
        rays=len(rays_data),
        octaves=len(octaves),
        frequencies=freqs,
        amplitudes=amps,
        method_A_mean_segments=float(np.mean(A_seg)),
        method_A_mean_octave_integrals=float(np.mean(A_eval)),
        method_B_results=B_results,
        method_C_results=C_results,
        all_bounds_respected=all_bounds,
    )


def make_report(s: Summary):
    lines = [
        "# Multiscale Cloud Method Comparison",
        "",
        f"- Rays: `{s.rays}`",
        f"- Octaves: `{s.octaves}`",
        f"- Frequencies: `{s.frequencies}`",
        f"- Amplitudes: `{s.amplitudes}`",
        "",
        "## Method A — exact common refinement",
        "",
        f"- Mean segments: `{s.method_A_mean_segments:.3f}`",
        f"- Mean octave integrals: `{s.method_A_mean_octave_integrals:.3f}`",
        "",
        "## Method B — fixed octave cutoff",
        "",
        "| Config | P95 tau error | Max tau error | P95 T error | Segment reduction | Integral reduction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k,v in s.method_B_results.items():
        lines.append(
            f"| `{k}` | {v['p95_abs_tau_error']:.3e} | {v['max_abs_tau_error']:.3e} | "
            f"{v['p95_transmittance_error']:.3e} | {100*v['segment_reduction_vs_A']:.1f}% | "
            f"{100*v['integral_reduction_vs_A']:.1f}% |"
        )

    lines += [
        "",
        "## Method C — exact backbone + bounded adaptive residual",
        "",
        "| Config | Bound | Bound ok | P95 tau error | Max tau error | P95 T error | Segment reduction | Integral reduction | Mean bound queries |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k,v in s.method_C_results.items():
        lines.append(
            f"| `{k}` | {v['declared_tau_bound']:.1e} | {'yes' if v['bound_respected'] else 'no'} | "
            f"{v['p95_abs_tau_error']:.3e} | {v['max_abs_tau_error']:.3e} | "
            f"{v['p95_transmittance_error']:.3e} | {100*v['segment_reduction_vs_A']:.1f}% | "
            f"{100*v['integral_reduction_vs_A']:.1f}% | {v['mean_bound_queries']:.1f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- A is the exact multiscale ground truth, but its cost follows the finest partition.",
        "- B is cheap and simple, but has no per-ray error guarantee.",
        "- C is the production-oriented candidate because it trades work for a declared optical-depth error budget.",
        "- Since `|exp(-tau)-exp(-(tau+delta))| <= 1-exp(-|delta|) <= |delta|`, the same optical-depth budget also bounds transmittance error.",
        f"- All declared C bounds respected in this run: `{s.all_bounds_respected}`.",
    ]
    return "\n".join(lines)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=20260718)
    p.add_argument("--rays",type=int,default=220)
    p.add_argument("--json",type=Path)
    p.add_argument("--report",type=Path)
    a=p.parse_args()
    s=run(a.seed,a.rays)
    print(json.dumps(asdict(s),indent=2))
    if a.json: a.json.write_text(json.dumps(asdict(s),indent=2),encoding="utf-8")
    if a.report: a.report.write_text(make_report(s),encoding="utf-8")


if __name__=="__main__":
    main()
