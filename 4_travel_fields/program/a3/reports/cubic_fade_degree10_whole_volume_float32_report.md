# Cubic-Fade Degree-10 Whole-Volume Float32 Validation

**Overall status:** NEEDS REVIEW

## Configuration

- Grid: `10 × 8 × 11` cells
- Random rays: `500`
- Adversarial/boundary rays: `50`
- Tested free-path targets: `3796`
- Exact-zero lattice probability: `0.48`
- Float32 arithmetic:
  - normalized cell coordinate `u∈[0,1]`
  - coefficients built directly in float32
  - ordinary float32 Horner
  - 12-step safeguarded Newton
  - convergence freeze
  - bracket-midpoint return
- DDA geometry remains float64 to isolate the medium arithmetic.
- Ground truth: validated float64 analytic DDA traversal.

## Whole-ray optical depth

- P95 absolute error: `1.199266e-06`
- P99 absolute error: `2.046550e-06`
- Maximum absolute error: `1.027459e-05`
- Maximum relative error: `2.065553e-06`

## Free-path events

- Hit/miss mismatches: `0`
- P95 distance error: `3.570005e-01`
- P99 distance error: `5.733536e-01`
- Maximum distance error: `8.547758e-01`
- P99 independent optical-depth residual: `5.251658e-01`
- Maximum independent optical-depth residual: `1.048976e+00`

## Solver behavior

- Mean genuine fallback count: `0.046`
- P95 fallback count: `0.000`
- Maximum fallback count: `1`
- Mean convergence-freeze iteration: `3.035`
- P95 convergence-freeze iteration: `5.000`
- Maximum convergence-freeze iteration: `12`

## Interpretation

This validates cross-cell float32 optical-depth accumulation and terminal
degree-10 inversion independently of DDA geometry precision. A pass closes
the arithmetic layer of the single-scale cubic-fade medium. The only remaining
GPU-specific item is float32 DDA tie and boundary handling.
