# travel: iso-surface / nonlinear-operator cell traversal reference

> **Scope note (2026-07-18).** This package models the **DPDF/Gemini signed-lattice
> iso-surface** medium `d(t) = max(0, V(t) − τ)` — **not** the authoritative **A3-GN**
> volume algorithm (nonnegative density, raw quartic optical depth, no root-finding; see
> `a3/` and the A3 handoff). Within the A3 project it corresponds to the **§13.6 "B5
> nonlinear operator"** line `ρ = max(0, B − λR)`, whose operator algebra is exactly
> "`max(0, f)` is piecewise polynomial, partitioned by the roots of `f = 0`" — which is
> what the cubic root partition here implements. The grazing-root robustness finding
> below belongs to this line, not to A3.

CPU high-precision reference and self-validation loop for that cellwise event
computation, independent of the `psrt/` package (Papers 1–3).

    fixed-step ray marching  ->  DDA cell traversal + analytic integration / root inversion

For a volume, the event is the generalized inverse of a monotone optical-depth clock
`A(t) = sigma_t * integral_0^t max(0, V - tau) ds`, where `V` is the (cubic) trilinear
density along the ray and `tau` an iso-threshold.

## Modules

| module | role |
|---|---|
| `poly.py` | monomial-basis polynomial utilities (Horner, derivative, antiderivative, definite integral) |
| `trilinear.py` | trilinear cell: corner densities -> coefficients -> cubic ray density polynomial |
| `opticaldepth.py` | analytic `A(t)`: robust real roots + sign-constant partition + exact quartic antiderivative |
| `reference.py` | **independent** high-precision reference: numeric kink scan + 64-node Gauss-Legendre panels |
| `inversion.py` | monotone generalized inverse `T_xi` (safeguarded Newton + bisection bracket) |
| `dda.py` | 3D voxel traversal (Amanatides-Woo) with corner/edge tie-breaking |
| `harness.py` | random + adversarial case generators, scale-aware metrics, worst-case search |
| `validation.py` | Phase-A driver: `python -m travel.validation` |

## Running

```bash
python -m pytest tests/test_travel_*.py     # fast gates
python -m travel.validation                 # full worst-case sweep -> results/travel/
```

`validation` writes `results/travel/phaseA_summary.csv`,
`phaseA_worst_cases.json` (reproducible worst offenders per category), and
`phaseA_dda.csv`.

## Metrics (scale-aware, analytic vs reference)

- `recon`  — ray-cubic reconstruction error, relative to field magnitude
- `depth`  — optical depth `|A_analytic - A_ref|`, relative to total
- `inv_t`  — free-path position error, relative to segment length (well-posed cases only)
- `inv_A`  — free-path consistency `|A(T) - xi| / total` (well-posed cases only)

Worst-case categories: `random`, `grazing`, `near_zero`, `plateau`, `cancellation`,
`boundary_hit`.

## Phase-A findings

1. **Naive Cardano fails on grazing rays.** When a ray grazes a voxel
   (`w_z -> 0`), the ray-cubic leading coefficient `a3 = c111 * w_x * w_y * w_z -> 0`.
   The a3-normalized closed-form (Cardano) solve then loses all precision and can drop a
   genuine in-segment root, integrating the *unclamped* density and producing errors up
   to ~85% of the optical depth. The hot path uses the companion-matrix solver
   (`opticaldepth.real_roots`, via `numpy.roots`), which stays accurate; the naive
   variant is retained as `cubic_real_roots_cardano` only for this documented
   comparison. **Recommendation for the GPU shader: relative degree-reduction
   (treat as quadratic when `|a3|` is negligible vs. the other coefficients) plus a
   stable quadratic solve — do not use bare Cardano on the ray-cubic.**

2. **Multiple roots are conditioned to `eps**(1/m)`.** A root of multiplicity `m`
   cannot be located tighter than ~`eps**(1/m)` in double precision (double ~1e-8,
   triple ~6e-6). This is harmless for the optical-depth integral (a grazing/tangent
   touch contributes ~zero clamped area, and an odd-multiplicity crossing has negligible
   position sensitivity), but it matters for **surface** first-hit and must be stated as
   a conditioning limit, not chased with a tighter tolerance.

3. **Free path is ill-conditioned when total optical depth is tiny.** With
   `A` nearly flat (`total < 1e-3`), the inverse position is not a meaningful accuracy
   signal; the harness gates inversion only on well-posed cases.
