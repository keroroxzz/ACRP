"""Self-validation loop for the trilinear analytic-traversal primitive.

Generates random and adversarial single-cell cases, runs the analytic method against
the independent reference, and reports the worst offenders.  This is the local
closed-loop the research route asks for: "large-scale numerical counter-example search
is better suited to a local closed loop than manual shader observation".

A *case* is one unit cell [0,1]^3 with corner densities, a ray (p, w) clipped to the
cell, an iso-threshold tau and an extinction scale sigma_t.  For each case we check:

  recon : ray-cubic reconstruction   max_t |V(p+tw) - poly(t)|
  depth : optical depth              max_t |A_analytic(t) - A_ref(t)|  (incl. total)
  inv_t : free-path position         max_xi |T_analytic(xi) - T_ref(xi)|
  inv_A : free-path consistency      max_xi |A_analytic(T_analytic) - xi| / total

Worst-case *categories* stress the failure modes named in the route: grazing traversal,
near-zero density, zero-density plateau, extreme coefficient cancellation, and
cell-boundary hits.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import trilinear, opticaldepth, reference, inversion, poly


# --------------------------------------------------------------------------------------
# Case representation
# --------------------------------------------------------------------------------------
class Case:
    __slots__ = ("coefs", "p", "w", "tau", "sigma_t", "t_a", "t_b", "corners", "tag")

    def __init__(self, corners, p, w, tau, sigma_t, t_a, t_b, tag=""):
        self.corners = np.asarray(corners, dtype=float)
        self.coefs = trilinear.trilinear_coefs(self.corners)
        self.p = np.asarray(p, dtype=float)
        self.w = np.asarray(w, dtype=float)
        self.tau = float(tau)
        self.sigma_t = float(sigma_t)
        self.t_a = float(t_a)
        self.t_b = float(t_b)
        self.tag = tag

    def repro(self) -> Dict:
        """A JSON-friendly dict that reproduces this case exactly."""
        return dict(corners=self.corners.tolist(), p=self.p.tolist(),
                    w=self.w.tolist(), tau=self.tau, sigma_t=self.sigma_t,
                    t_a=self.t_a, t_b=self.t_b, tag=self.tag)


def _clip_unit_cell(p, w) -> Optional[Tuple[float, float]]:
    """Entry/exit t of ray p + t w through the unit cell [0,1]^3, or None."""
    lo, hi = -np.inf, np.inf
    for i in range(3):
        if abs(w[i]) < 1e-15:
            if p[i] < 0.0 or p[i] > 1.0:
                return None
        else:
            t1 = (0.0 - p[i]) / w[i]
            t2 = (1.0 - p[i]) / w[i]
            lo = max(lo, min(t1, t2))
            hi = min(hi, max(t1, t2))
    if hi - lo < 1e-4:
        return None
    return lo, hi


def _random_ray(rng) -> Tuple[np.ndarray, np.ndarray]:
    p = rng.uniform(-0.3, 0.3, 3)
    w = rng.uniform(-1.0, 1.0, 3)
    n = np.linalg.norm(w)
    if n < 1e-9:
        w = np.array([1.0, 0.0, 0.0])
        n = 1.0
    return p, w / n


# --------------------------------------------------------------------------------------
# Generators (return a Case or None)
# --------------------------------------------------------------------------------------
def gen_random(rng) -> Optional[Case]:
    corners = rng.uniform(-1.0, 2.0, (2, 2, 2))
    p, w = _random_ray(rng)
    clip = _clip_unit_cell(p, w)
    if clip is None:
        return None
    tau = rng.uniform(-0.5, 1.0)
    sigma_t = rng.uniform(0.3, 4.0)
    return Case(corners, p, w, tau, sigma_t, clip[0], clip[1], "random")


def gen_grazing(rng) -> Optional[Case]:
    """Ray nearly tangent to a cell face (one direction component ~ 0)."""
    p, w = _random_ray(rng)
    axis = rng.randint(3)
    w[axis] = rng.choice([-1.0, 1.0]) * 10.0 ** rng.uniform(-9, -4)
    w = w / np.linalg.norm(w)
    p[axis] = rng.choice([1e-7, 1.0 - 1e-7, 0.5])
    clip = _clip_unit_cell(p, w)
    if clip is None:
        return None
    corners = rng.uniform(-1.0, 2.0, (2, 2, 2))
    tau = rng.uniform(-0.5, 1.0)
    sigma_t = rng.uniform(0.3, 4.0)
    return Case(corners, p, w, tau, sigma_t, clip[0], clip[1], "grazing")


def gen_near_zero_density(rng) -> Optional[Case]:
    """Corners clustered just around tau so d = max(0, V-tau) ~ 0 everywhere."""
    base = rng.uniform(-1.0, 1.0)
    corners = base + rng.uniform(-1e-4, 1e-4, (2, 2, 2))
    p, w = _random_ray(rng)
    clip = _clip_unit_cell(p, w)
    if clip is None:
        return None
    tau = base + rng.uniform(-1e-4, 1e-4)
    sigma_t = rng.uniform(0.3, 4.0)
    return Case(corners, p, w, tau, sigma_t, clip[0], clip[1], "near_zero")


def gen_plateau(rng) -> Optional[Case]:
    """Half the corners well below tau -> a genuine zero-density plateau in the cell."""
    corners = rng.uniform(-1.0, 2.0, (2, 2, 2))
    # force one z-slab far below threshold
    corners[:, :, 0] = rng.uniform(-3.0, -2.0)
    p, w = _random_ray(rng)
    clip = _clip_unit_cell(p, w)
    if clip is None:
        return None
    tau = rng.uniform(0.0, 1.0)
    sigma_t = rng.uniform(0.3, 4.0)
    return Case(corners, p, w, tau, sigma_t, clip[0], clip[1], "plateau")


def gen_cancellation(rng) -> Optional[Case]:
    """Corners with alternating large signs -> large c111 with heavy cancellation."""
    big = 10.0 ** rng.uniform(3, 6)
    signs = np.array([[[1, -1], [-1, 1]], [[-1, 1], [1, -1]]], dtype=float)
    corners = big * signs + rng.uniform(-1.0, 1.0, (2, 2, 2))
    p, w = _random_ray(rng)
    clip = _clip_unit_cell(p, w)
    if clip is None:
        return None
    tau = rng.uniform(-1.0, 1.0)
    sigma_t = rng.uniform(0.3, 4.0)
    return Case(corners, p, w, tau, sigma_t, clip[0], clip[1], "cancellation")


def gen_boundary_hit(rng) -> Optional[Case]:
    """tau chosen so an iso-crossing sits (almost) exactly on the cell entry/exit."""
    corners = rng.uniform(-1.0, 2.0, (2, 2, 2))
    coefs = trilinear.trilinear_coefs(corners)
    p, w = _random_ray(rng)
    clip = _clip_unit_cell(p, w)
    if clip is None:
        return None
    t_a, t_b = clip
    endpoint = t_a if rng.random() < 0.5 else t_b
    x = p + endpoint * w
    tau = trilinear.evaluate(coefs, x[0], x[1], x[2])  # root exactly at endpoint
    sigma_t = rng.uniform(0.3, 4.0)
    return Case(corners, p, w, tau, sigma_t, t_a, t_b, "boundary_hit")


GENERATORS: Dict[str, Callable] = {
    "random": gen_random,
    "grazing": gen_grazing,
    "near_zero": gen_near_zero_density,
    "plateau": gen_plateau,
    "cancellation": gen_cancellation,
    "boundary_hit": gen_boundary_hit,
}


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
# Below this total optical depth the free path is ill-conditioned (A is nearly flat),
# so inversion-position error is not a meaningful accuracy signal.
WELL_POSED_TOTAL = 1e-3


def evaluate(case: Case, n_samples: int = 9,
             xi_fracs=(0.05, 0.2, 0.4, 0.6, 0.8, 0.95)) -> Dict[str, float]:
    """Return *scale-aware* error metrics for one case (analytic vs reference).

    recon / depth are normalized by the field / optical-depth magnitude so that
    large-amplitude (cancellation) cells are judged on relative error.  Inversion
    metrics are only reported for well-posed cases (total optical depth not tiny).
    """
    rc = trilinear.ray_density_poly(case.p, case.w, case.coefs)
    prof = opticaldepth.OpticalDepthProfile(rc, case.tau, case.sigma_t, case.t_a, case.t_b)
    ref = reference.ReferenceProfile(case.coefs, case.p, case.w, case.tau,
                                     case.sigma_t, case.t_a, case.t_b)

    field_scale = float(np.max(np.abs(case.corners)))
    seg_len = case.t_b - case.t_a

    ts = np.linspace(case.t_a, case.t_b, n_samples)
    recon_abs = 0.0
    depth_abs = abs(prof.total - ref.total)
    for t in ts:
        x = case.p + t * case.w
        direct = trilinear.evaluate(case.coefs, x[0], x[1], x[2])
        recon_abs = max(recon_abs, abs(direct - poly.eval_horner(rc, t)))
        depth_abs = max(depth_abs, abs(prof.A(t) - ref.A(t)))

    inv_t = inv_A = 0.0
    well_posed = prof.total > WELL_POSED_TOTAL
    if well_posed:
        for f in xi_fracs:
            xi = f * prof.total
            T_an = inversion.free_path(prof, xi)
            T_ref = ref.free_path(xi)
            if np.isfinite(T_an) and np.isfinite(T_ref):
                inv_t = max(inv_t, abs(T_an - T_ref) / (1.0 + seg_len))
                inv_A = max(inv_A, abs(prof.A(T_an) - xi) / prof.total)

    return dict(
        recon=recon_abs / (1.0 + field_scale),
        depth=depth_abs / (1.0 + prof.total),
        inv_t=inv_t,
        inv_A=inv_A,
        recon_abs=recon_abs,
        depth_abs=depth_abs,
        total=prof.total,
        well_posed=float(well_posed),
    )


# --------------------------------------------------------------------------------------
# Property suite + worst-case search
# --------------------------------------------------------------------------------------
def run_suite(n_per_category: int = 500, seed: int = 12345,
              keep_worst: int = 3) -> Dict:
    """Run every generator category; return aggregate max errors + worst offenders."""
    rng = np.random.RandomState(seed)
    summary = {}
    for name, gen in GENERATORS.items():
        agg = dict(recon=0.0, depth=0.0, inv_t=0.0, inv_A=0.0)
        worst: List[Tuple[float, Dict, Dict]] = []
        n_ok = 0
        attempts = 0
        while n_ok < n_per_category and attempts < n_per_category * 20:
            attempts += 1
            case = gen(rng)
            if case is None:
                continue
            n_ok += 1
            m = evaluate(case)
            for k in agg:
                agg[k] = max(agg[k], m[k])
            score = max(m["recon"], m["depth"], m["inv_A"])
            worst.append((score, m, case.repro()))
        worst.sort(key=lambda x: -x[0])
        summary[name] = dict(n=n_ok, agg=agg, worst=worst[:keep_worst])
    return summary


def format_summary(summary: Dict) -> str:
    lines = ["category      n      recon       depth       inv_t       inv_A"]
    for name, s in summary.items():
        a = s["agg"]
        lines.append("{:<12} {:>4}  {:>10.2e}  {:>10.2e}  {:>10.2e}  {:>10.2e}".format(
            name, s["n"], a["recon"], a["depth"], a["inv_t"], a["inv_A"]))
    return "\n".join(lines)
