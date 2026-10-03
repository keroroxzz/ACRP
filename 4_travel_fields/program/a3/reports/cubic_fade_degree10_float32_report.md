# Cubic-Fade Degree-10 Float32 / GPU-Oriented Study

## Configuration

- Random seed: `20260718`
- Adversarial single-cell cases: `1200`
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

- Maximum density error, world-distance coefficients: `9.552030e-05`
- Maximum density error, normalized coefficients: `3.443304e-05`
- Maximum optical-depth error, world-distance coefficients: `9.274263e-04`
- Maximum optical-depth error, normalized coefficients: `4.391450e-04`

Median optical-depth evaluation error:

- world-distance Horner: `1.610430e-11`
- normalized Horner: `5.769657e-12`
- normalized compensated Horner: `5.190211e-12`

Per-case best optical-depth evaluator:

- world-distance Horner: `362`
- normalized Horner: `685`
- normalized compensated Horner: `153`

## Fixed-iteration free-path inversion

| Configuration | P95 distance error | P99 distance error | Max distance error | P99 tau residual | Mean fallback | P95 fallback |
|---|---:|---:|---:|---:|---:|---:|
| `normalized_compensated_6` | 3.534e-03 | 1.821e-02 | 1.588e-01 | 1.877e-01 | 4.556 | 6.0 |
| `normalized_compensated_8` | 7.821e-04 | 2.881e-03 | 3.969e-02 | 4.292e-02 | 6.504 | 8.0 |
| `normalized_compensated_10` | 1.979e-04 | 7.197e-04 | 9.923e-03 | 2.436e-03 | 8.473 | 10.0 |
| `normalized_compensated_12` | 4.406e-05 | 1.967e-04 | 2.481e-03 | 6.090e-04 | 10.457 | 12.0 |
| `normalized_compensated_16` | 2.741e-06 | 1.070e-05 | 1.117e-03 | 2.043e-04 | 14.432 | 16.0 |
| `normalized_horner_6` | 2.908e-03 | 1.311e-02 | 1.588e-01 | 4.621e-02 | 4.522 | 6.0 |
| `normalized_horner_8` | 7.139e-04 | 3.147e-03 | 3.969e-02 | 1.145e-02 | 6.468 | 8.0 |
| `normalized_horner_10` | 1.643e-04 | 7.869e-04 | 9.923e-03 | 1.726e-03 | 8.425 | 10.0 |
| `normalized_horner_12` | 3.746e-05 | 2.015e-04 | 2.481e-03 | 4.890e-04 | 10.402 | 12.0 |
| `normalized_horner_16` | 2.147e-06 | 1.028e-05 | 1.664e-03 | 3.696e-05 | 14.368 | 16.0 |
| `world_compensated_6` | 2.897e-03 | 1.764e-02 | 1.588e-01 | 9.217e-02 | 4.445 | 6.0 |
| `world_compensated_8` | 7.038e-04 | 3.458e-03 | 3.969e-02 | 1.806e-02 | 6.404 | 8.0 |
| `world_compensated_10` | 1.763e-04 | 9.309e-04 | 9.924e-03 | 4.510e-03 | 8.384 | 10.0 |
| `world_compensated_12` | 4.141e-05 | 2.331e-04 | 2.481e-03 | 1.129e-03 | 10.369 | 12.0 |
| `world_compensated_16` | 2.705e-06 | 2.066e-05 | 9.885e-04 | 7.360e-05 | 14.347 | 16.0 |
| `world_horner_6` | 2.501e-03 | 1.260e-02 | 4.303e-02 | 6.575e-02 | 4.382 | 6.0 |
| `world_horner_8` | 6.252e-04 | 2.882e-03 | 1.076e-02 | 1.647e-02 | 6.344 | 8.0 |
| `world_horner_10` | 1.510e-04 | 7.875e-04 | 2.689e-03 | 4.120e-03 | 8.321 | 10.0 |
| `world_horner_12` | 3.581e-05 | 2.013e-04 | 9.885e-04 | 1.030e-03 | 10.303 | 12.0 |
| `world_horner_16` | 2.235e-06 | 1.455e-05 | 9.885e-04 | 7.052e-05 | 14.272 | 16.0 |

## Current recommendation

- Parameterization: **normalized**
- Evaluator: **horner**
- Fixed iteration cap: **16**
- Derivative floor: **1.0e-07**

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
