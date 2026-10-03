
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ============================================================
# Cubic-fade algebra
# ============================================================

def fade(x):
    return x * x * (3.0 - 2.0 * x)


def fade_poly(a, b, dtype):
    a = dtype(a)
    b = dtype(b)
    return np.array(
        [
            dtype(3.0) * a * a - dtype(2.0) * a * a * a,
            dtype(6.0) * a * b * (dtype(1.0) - a),
            dtype(3.0) * b * b * (dtype(1.0) - dtype(2.0) * a),
            -dtype(2.0) * b * b * b,
        ],
        dtype=dtype,
    )


def pad(p, n, dtype):
    out = np.zeros(n, dtype=dtype)
    out[:len(p)] = p
    return out


def build_density_poly(local_entry, direction, corners, dtype=np.float64):
    local_entry = np.asarray(local_entry, dtype=dtype)
    direction = np.asarray(direction, dtype=dtype)
    corners = np.asarray(corners, dtype=dtype)

    fx = fade_poly(local_entry[0], direction[0], dtype)
    fy = fade_poly(local_entry[1], direction[1], dtype)
    fz = fade_poly(local_entry[2], direction[2], dtype)

    fxy = np.polynomial.polynomial.polymul(fx, fy).astype(dtype)
    fxz = np.polynomial.polynomial.polymul(fx, fz).astype(dtype)
    fyz = np.polynomial.polynomial.polymul(fy, fz).astype(dtype)
    fxyz = np.polynomial.polynomial.polymul(fxy, fz).astype(dtype)

    d000 = corners[0, 0, 0]
    d100 = corners[1, 0, 0]
    d010 = corners[0, 1, 0]
    d110 = corners[1, 1, 0]
    d001 = corners[0, 0, 1]
    d101 = corners[1, 0, 1]
    d011 = corners[0, 1, 1]
    d111 = corners[1, 1, 1]

    a = d000
    ax = d100 - d000
    ay = d010 - d000
    az = d001 - d000
    axy = d110 - d100 - d010 + d000
    axz = d101 - d100 - d001 + d000
    ayz = d011 - d010 - d001 + d000
    axyz = d111 - d110 - d101 - d011 + d100 + d010 + d001 - d000

    rho = np.zeros(10, dtype=dtype)
    rho[0] = a
    rho += ax * pad(fx, 10, dtype)
    rho += ay * pad(fy, 10, dtype)
    rho += az * pad(fz, 10, dtype)
    rho += axy * pad(fxy, 10, dtype)
    rho += axz * pad(fxz, 10, dtype)
    rho += ayz * pad(fyz, 10, dtype)
    rho += axyz * pad(fxyz, 10, dtype)
    return rho


def build_tau_poly(rho, sigma_t, dtype=np.float64):
    rho = np.asarray(rho, dtype=dtype)
    tau = np.zeros(11, dtype=dtype)
    sigma_t = dtype(sigma_t)
    for k in range(10):
        tau[k + 1] = sigma_t * rho[k] / dtype(k + 1)
    return tau


def horner(coeff, x, dtype=None):
    if dtype is None:
        dtype = coeff.dtype
    x = dtype(x)
    y = dtype(0.0)
    for c in coeff[::-1]:
        y = y * x + dtype(c)
    return float(y)


def compensated_horner_float32(coeff, x):
    """
    Lightweight error-compensated Horner using float32 operations plus
    an accumulated first-order correction. This is not a true EFT on GPU,
    but approximates the benefit of one compensation accumulator.
    """
    coeff = np.asarray(coeff, dtype=np.float32)
    x = np.float32(x)
    y = np.float32(coeff[-1])
    e = np.float32(0.0)
    for c in coeff[-2::-1]:
        prod64 = float(y) * float(x)
        prod = np.float32(prod64)
        prod_err = np.float32(prod64 - float(prod))
        sum64 = float(prod) + float(c)
        new_y = np.float32(sum64)
        sum_err = np.float32(sum64 - float(new_y))
        e = np.float32(e * x + prod_err + sum_err)
        y = new_y
    return float(np.float32(y + e))


def direct_density(s, local_entry, direction, corners):
    q = np.asarray(local_entry, dtype=np.float64) + float(s) * np.asarray(direction, dtype=np.float64)
    f = fade(q)
    c = np.asarray(corners, dtype=np.float64)
    x00 = (1.0 - f[0]) * c[0,0,0] + f[0] * c[1,0,0]
    x10 = (1.0 - f[0]) * c[0,1,0] + f[0] * c[1,1,0]
    x01 = (1.0 - f[0]) * c[0,0,1] + f[0] * c[1,0,1]
    x11 = (1.0 - f[0]) * c[0,1,1] + f[0] * c[1,1,1]
    y0 = (1.0 - f[1]) * x00 + f[1] * x10
    y1 = (1.0 - f[1]) * x01 + f[1] * x11
    return float((1.0 - f[2]) * y0 + f[2] * y1)


def bisection_ref(tau64, L, target, iters=100):
    lo, hi = 0.0, float(L)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if horner(tau64, mid, np.float64) < target:
            lo = mid
        else:
            hi = mid
    return hi


# ============================================================
# Parameterizations
# ============================================================

def to_normalized_parameterization(rho_s, L, sigma_t, dtype=np.float32):
    """
    rho(s) = sum c_k s^k, s=L u.
    rho_u(u) = sum (c_k L^k) u^k.
    tau_u(u) = sigma_t * L * integral rho_u(u) du.
    """
    rho_s = np.asarray(rho_s, dtype=np.float64)
    rho_u = np.zeros(10, dtype=dtype)
    for k in range(10):
        rho_u[k] = dtype(rho_s[k] * (L ** k))
    tau_u = np.zeros(11, dtype=dtype)
    for k in range(10):
        tau_u[k + 1] = dtype(sigma_t * L * float(rho_u[k]) / float(k + 1))
    return rho_u, tau_u


# ============================================================
# GPU-style safeguarded solver
# ============================================================

@dataclass
class SolveResult:
    s: float
    iterations: int
    fallbacks: int
    residual: float
    bracket: float


def solve_float32(
    rho32,
    tau32,
    domain_length,
    target_tau,
    sigma_derivative_scale,
    max_iter,
    eval_mode="horner",
    normalized=False,
    derivative_floor=1e-7,
):
    """
    Fixed-iteration GPU-style safeguarded Newton.
    If normalized=True, variable is u in [0,1], and returned s=u*domain_length.
    sigma_derivative_scale is:
      - sigma_t for world-s parameterization
      - sigma_t * L for normalized u parameterization
    """
    target = np.float32(target_tau)
    hi0 = np.float32(1.0 if normalized else domain_length)
    lo = np.float32(0.0)
    hi = hi0

    evaluator = compensated_horner_float32 if eval_mode == "compensated" else lambda c, x: horner(c, x, np.float32)

    total = evaluator(tau32, hi)
    x = np.float32(float(hi) * np.clip(float(target) / max(total, 1e-30), 0.0, 1.0))
    fallbacks = 0

    for _ in range(max_iter):
        tx = evaluator(tau32, x)
        r = np.float32(tx - float(target))

        if r < 0.0:
            lo = x
        else:
            hi = x

        density = max(evaluator(rho32, x), 0.0)
        derivative = np.float32(sigma_derivative_scale * density)

        valid = math.isfinite(float(derivative)) and float(derivative) > derivative_floor
        if valid:
            candidate = np.float32(float(x) - float(r) / float(derivative))
            valid = math.isfinite(float(candidate)) and float(lo) < float(candidate) < float(hi)
        else:
            candidate = np.float32(np.nan)

        if not valid:
            candidate = np.float32(0.5) * (lo + hi)
            fallbacks += 1

        x = candidate

    # Return bracket midpoint, not the final Newton iterate.
    x_final = np.float32(0.5) * (lo + hi)
    tau_final = evaluator(tau32, x_final)
    if normalized:
        s_final = float(x_final) * float(domain_length)
        bracket = float(hi - lo) * float(domain_length)
    else:
        s_final = float(x_final)
        bracket = float(hi - lo)

    return SolveResult(
        s=s_final,
        iterations=max_iter,
        fallbacks=fallbacks,
        residual=abs(tau_final - float(target)),
        bracket=bracket,
    )


# ============================================================
# Test case generation
# ============================================================

def random_segment(rng):
    p0 = rng.uniform(0.0, 1.0, size=3)
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    exits = []
    for a in range(3):
        if d[a] > 1e-14:
            exits.append((1.0 - p0[a]) / d[a])
        elif d[a] < -1e-14:
            exits.append((0.0 - p0[a]) / d[a])
    max_len = min(exits)
    # Include very short and near-full-cell segments.
    if rng.random() < 0.25:
        fraction = 10.0 ** rng.uniform(-7.0, -2.0)
    elif rng.random() < 0.25:
        fraction = 1.0 - 10.0 ** rng.uniform(-7.0, -2.0)
    else:
        fraction = rng.uniform(1e-4, 0.9999)
    return p0, d, float(max_len * fraction)


def make_case(rng):
    mode = rng.integers(0, 6)
    if mode == 0:
        corners = rng.uniform(0.0, 3.0, size=(2,2,2))
    elif mode == 1:
        # Sparse / near plateau.
        corners = rng.uniform(0.0, 3.0, size=(2,2,2))
        corners[rng.random(size=(2,2,2)) < 0.75] = 0.0
    elif mode == 2:
        # Tiny densities.
        corners = 10.0 ** rng.uniform(-12.0, -4.0, size=(2,2,2))
    elif mode == 3:
        # Large dynamic range.
        corners = 10.0 ** rng.uniform(-8.0, 3.0, size=(2,2,2))
    elif mode == 4:
        # Nearly constant field to stress cancellation.
        base = 10.0 ** rng.uniform(-5.0, 2.0)
        corners = base * (1.0 + rng.normal(0.0, 1e-6, size=(2,2,2)))
        corners = np.maximum(corners, 0.0)
    else:
        # Binary corners.
        corners = (rng.random(size=(2,2,2)) > 0.5).astype(np.float64) * rng.uniform(0.1, 3.0)

    p0, d, L = random_segment(rng)
    sigma_t = 10.0 ** rng.uniform(-3.0, 2.0)
    rho64 = build_density_poly(p0, d, corners, np.float64)
    tau64 = build_tau_poly(rho64, sigma_t, np.float64)
    total_tau = horner(tau64, L, np.float64)

    # Ensure an invertible positive target.
    if total_tau <= 1e-22:
        return None

    target_modes = [
        1e-8, 1e-6, 1e-4, 0.01, 0.25, 0.5, 0.9, 1.0 - 1e-6, 1.0 - 1e-8
    ]
    frac = float(target_modes[rng.integers(0, len(target_modes))])
    target = total_tau * frac
    ref_s = bisection_ref(tau64, L, target)

    return {
        "corners": corners,
        "p0": p0,
        "d": d,
        "L": L,
        "sigma_t": sigma_t,
        "rho64": rho64,
        "tau64": tau64,
        "total_tau": total_tau,
        "target": target,
        "ref_s": ref_s,
    }


# ============================================================
# Study
# ============================================================

@dataclass
class StudySummary:
    seed: int
    cases: int
    max_density_eval_abs_error_world: float
    max_density_eval_abs_error_normalized: float
    max_tau_eval_abs_error_world: float
    max_tau_eval_abs_error_normalized: float
    world_wins_tau_eval: int
    normalized_wins_tau_eval: int
    compensated_wins_tau_eval: int
    median_world_tau_eval_error: float
    median_normalized_tau_eval_error: float
    median_compensated_tau_eval_error: float
    solver_stats: Dict[str, Dict[str, float]]
    recommended_parameterization: str
    recommended_iterations: int
    recommended_eval_mode: str
    derivative_floor_recommendation: float


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def run_study(seed=20260718, cases=20000):
    rng = np.random.default_rng(seed)

    density_err_world = []
    density_err_norm = []
    tau_err_world = []
    tau_err_norm = []
    tau_err_comp = []

    world_wins = 0
    norm_wins = 0
    comp_wins = 0

    iteration_options = [6, 8, 10, 12, 16]
    configurations = []
    for param in ["world", "normalized"]:
        for mode in ["horner", "compensated"]:
            for iters in iteration_options:
                configurations.append((param, mode, iters))

    stats = {
        cfg: {
            "s_error": [],
            "tau_residual": [],
            "fallbacks": [],
            "bracket": [],
            "nonfinite": 0,
        }
        for cfg in configurations
    }

    accepted = 0
    while accepted < cases:
        case = make_case(rng)
        if case is None:
            continue
        accepted += 1

        p0 = case["p0"]
        d = case["d"]
        L = case["L"]
        corners = case["corners"]
        sigma_t = case["sigma_t"]
        rho64 = case["rho64"]
        tau64 = case["tau64"]
        target = case["target"]
        ref_s = case["ref_s"]

        rho32_world = build_density_poly(p0, d, corners, np.float32)
        tau32_world = build_tau_poly(rho32_world, sigma_t, np.float32)
        rho32_norm, tau32_norm = to_normalized_parameterization(rho64, L, sigma_t, np.float32)

        # Evaluation at random point.
        s = float(rng.uniform(0.0, L))
        u = s / L if L > 0.0 else 0.0
        rho_ref = direct_density(s, p0, d, corners)
        tau_ref = horner(tau64, s, np.float64)

        rho_world = horner(rho32_world, s, np.float32)
        rho_norm = horner(rho32_norm, u, np.float32)
        tw = horner(tau32_world, s, np.float32)
        tn = horner(tau32_norm, u, np.float32)
        tc = compensated_horner_float32(tau32_norm, u)

        density_err_world.append(abs(rho_world - rho_ref))
        density_err_norm.append(abs(rho_norm - rho_ref))
        tau_err_world.append(abs(tw - tau_ref))
        tau_err_norm.append(abs(tn - tau_ref))
        tau_err_comp.append(abs(tc - tau_ref))

        errors = [abs(tw - tau_ref), abs(tn - tau_ref), abs(tc - tau_ref)]
        winner = int(np.argmin(errors))
        if winner == 0:
            world_wins += 1
        elif winner == 1:
            norm_wins += 1
        else:
            comp_wins += 1

        for param, mode, iters in configurations:
            if param == "world":
                rho = rho32_world
                tau = tau32_world
                domain = L
                deriv_scale = sigma_t
                normalized = False
            else:
                rho = rho32_norm
                tau = tau32_norm
                domain = L
                deriv_scale = sigma_t * L
                normalized = True

            result = solve_float32(
                rho,
                tau,
                domain,
                target,
                deriv_scale,
                iters,
                eval_mode=mode,
                normalized=normalized,
                derivative_floor=1e-7,
            )

            key = (param, mode, iters)
            if not math.isfinite(result.s):
                stats[key]["nonfinite"] += 1
                continue

            stats[key]["s_error"].append(abs(result.s - ref_s))
            stats[key]["tau_residual"].append(result.residual)
            stats[key]["fallbacks"].append(result.fallbacks)
            stats[key]["bracket"].append(result.bracket)

    solver_stats = {}
    for cfg, values in stats.items():
        param, mode, iters = cfg
        label = f"{param}_{mode}_{iters}"
        s_errors = values["s_error"]
        tau_res = values["tau_residual"]
        fallbacks = values["fallbacks"]
        brackets = values["bracket"]

        solver_stats[label] = {
            "mean_s_error": float(np.mean(s_errors)),
            "p95_s_error": percentile(s_errors, 95),
            "p99_s_error": percentile(s_errors, 99),
            "max_s_error": float(np.max(s_errors)),
            "mean_tau_residual": float(np.mean(tau_res)),
            "p99_tau_residual": percentile(tau_res, 99),
            "max_tau_residual": float(np.max(tau_res)),
            "mean_fallbacks": float(np.mean(fallbacks)),
            "p95_fallbacks": percentile(fallbacks, 95),
            "max_fallbacks": int(np.max(fallbacks)),
            "mean_bracket": float(np.mean(brackets)),
            "p99_bracket": percentile(brackets, 99),
            "nonfinite": int(values["nonfinite"]),
        }

    # Select practical recommendation:
    # minimize p99 distance error, then iterations, penalize nonfinite.
    candidates = []
    for label, s in solver_stats.items():
        if s["nonfinite"] == 0:
            param, mode, iters_s = label.rsplit("_", 2)
            iters = int(iters_s)
            score = (s["p99_s_error"], iters, s["mean_fallbacks"])
            candidates.append((score, label))
    candidates.sort()
    best_label = candidates[0][1]
    best_param, best_mode, best_iters = best_label.rsplit("_", 2)

    return StudySummary(
        seed=seed,
        cases=cases,
        max_density_eval_abs_error_world=float(np.max(density_err_world)),
        max_density_eval_abs_error_normalized=float(np.max(density_err_norm)),
        max_tau_eval_abs_error_world=float(np.max(tau_err_world)),
        max_tau_eval_abs_error_normalized=float(np.max(tau_err_norm)),
        world_wins_tau_eval=world_wins,
        normalized_wins_tau_eval=norm_wins,
        compensated_wins_tau_eval=comp_wins,
        median_world_tau_eval_error=float(np.median(tau_err_world)),
        median_normalized_tau_eval_error=float(np.median(tau_err_norm)),
        median_compensated_tau_eval_error=float(np.median(tau_err_comp)),
        solver_stats=solver_stats,
        recommended_parameterization=best_param,
        recommended_iterations=int(best_iters),
        recommended_eval_mode=best_mode,
        derivative_floor_recommendation=1e-7,
    )


def make_report(summary: StudySummary) -> str:
    rows = []
    for label, s in summary.solver_stats.items():
        if label.endswith(("_6", "_8", "_10", "_12", "_16")):
            rows.append((
                label,
                s["p95_s_error"],
                s["p99_s_error"],
                s["max_s_error"],
                s["p99_tau_residual"],
                s["mean_fallbacks"],
                s["p95_fallbacks"],
            ))
    rows.sort(key=lambda r: (r[0].split("_")[0], r[0].split("_")[1], int(r[0].split("_")[2])))

    table = [
        "| Configuration | P95 distance error | P99 distance error | Max distance error | P99 tau residual | Mean fallback | P95 fallback |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        table.append(
            f"| `{r[0]}` | {r[1]:.3e} | {r[2]:.3e} | {r[3]:.3e} | {r[4]:.3e} | {r[5]:.3f} | {r[6]:.1f} |"
        )

    return f"""# Cubic-Fade Degree-10 Float32 / GPU-Oriented Study

## Configuration

- Random seed: `{summary.seed}`
- Adversarial single-cell cases: `{summary.cases}`
- Arithmetic under test: IEEE-like `float32`
- Reference: float64 degree-10 optical-depth polynomial and 100-step bisection
- Tested parameterizations:
  - world distance `s`
  - normalized cell coordinate `u=s/L`
- Tested evaluators:
  - ordinary Horner
  - one-accumulator compensated Horner approximation
- Fixed solver iteration caps: 6, 8, 10, 12, 16
- Safeguard derivative floor: `1e-7`

## Polynomial evaluation

- Maximum density error, world-distance coefficients: `{summary.max_density_eval_abs_error_world:.6e}`
- Maximum density error, normalized coefficients: `{summary.max_density_eval_abs_error_normalized:.6e}`
- Maximum optical-depth error, world-distance coefficients: `{summary.max_tau_eval_abs_error_world:.6e}`
- Maximum optical-depth error, normalized coefficients: `{summary.max_tau_eval_abs_error_normalized:.6e}`

Median optical-depth evaluation error:

- world-distance Horner: `{summary.median_world_tau_eval_error:.6e}`
- normalized Horner: `{summary.median_normalized_tau_eval_error:.6e}`
- normalized compensated Horner: `{summary.median_compensated_tau_eval_error:.6e}`

Per-case best optical-depth evaluator:

- world-distance Horner: `{summary.world_wins_tau_eval}`
- normalized Horner: `{summary.normalized_wins_tau_eval}`
- normalized compensated Horner: `{summary.compensated_wins_tau_eval}`

## Fixed-iteration free-path inversion

{chr(10).join(table)}

## Current recommendation

- Parameterization: **{summary.recommended_parameterization}**
- Evaluator: **{summary.recommended_eval_mode}**
- Fixed iteration cap: **{summary.recommended_iterations}**
- Derivative floor: **{summary.derivative_floor_recommendation:.1e}**

The recommendation is selected from the tested configurations by lowest P99
distance error, then lower fixed iteration count, then lower fallback rate.

## Interpretation

1. Cell-normalized time keeps the solve domain fixed at `[0,1]`, which is
   attractive for GPU branch behavior and tolerance design.
2. World-distance coefficients can be better for some short segments because
   high-order terms naturally shrink with powers of `L`; normalized coefficients
   can instead expose coefficient cancellation directly.
3. A compensated evaluator can improve some cancellation-heavy cases, but its
   extra operations must be justified against a plain Horner baseline.
4. Distance accuracy and optical-depth residual are not equivalent near
   zero-density regions. The production solver should retain a bracket and
   return its midpoint after a fixed iteration cap.
5. Fixed iteration counts are preferable for GPU coherence. The table above
   quantifies the accuracy cost of 6/8/10/12/16 iterations.

This study is limited to isolated cells. The next step is to apply the selected
float32 configuration to complete multi-cell DDA rays and compare accumulated
optical depth and free-path events against the float64 whole-volume reference.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--cases", type=int, default=20000)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    summary = run_study(args.seed, args.cases)
    payload = asdict(summary)
    print(json.dumps(payload, indent=2))

    if args.json:
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.report:
        args.report.write_text(make_report(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
