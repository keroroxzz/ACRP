# Cubic-Fade 3D Degree-10 Medium Validation

**Overall status:** PASS

## Configuration

- Random seed: `20260718`
- Random valid in-cell ray segments: `10000`
- Direct density comparison samples: `70000`
- Corner densities: nonnegative random values in `[0, 3]`
- Extinction scale: log-uniform in `[10^-2, 10^1]`
- Optical-depth reference: independent 8-point Gauss-Legendre quadrature
- Inverse reference: 90-step monotone bisection
- Production candidate: bracketed safeguarded Newton

## Algebraic reconstruction

- Maximum absolute density error: `3.774758e-15`
- Maximum relative density error: `3.729730e-15`
- Maximum absolute optical-depth error: `7.105427e-15`
- Maximum relative optical-depth error: `2.656983e-15`
- Maximum sampled monotonicity violation: `0.000000e+00`

## Free-path inversion

- Inversion failures: `0`
- Maximum distance error versus reference: `3.007575e-09`
- Maximum final optical-depth residual: `1.468639e-10`
- Mean Newton iterations: `2.294`
- P95 Newton iterations: `4.000`
- Maximum Newton iterations: `6`
- Mean fallback count: `0.000`
- P95 fallback count: `0.000`
- Maximum fallback count: `0`

## Explicit edge cases

- Zero-density plateau: `PASS`
- Constant-density cells: `PASS`
- Axis-aligned rays: `PASS`

## Interpretation

The test validates the central single-cell primitive:

1. Cubic-fade trivariate interpolation restricted to a straight ray is represented by a degree-at-most-9 polynomial.
2. Its accumulated optical depth is represented by a degree-at-most-10 polynomial.
3. For nonnegative corner densities, the direct interpolant is nonnegative and the optical-depth clock is monotone.
4. A bracketed safeguarded Newton solver recovers the left generalized inverse without general tenth-degree root isolation.
5. The coefficient layout and Horner evaluation are directly transferable to a GPU implementation with fixed-size arrays and fixed iteration caps.

This report validates the algebra and scalar inversion. It does not yet validate whole-volume DDA traversal, floating-point behavior in GLSL `float`, or GPU wall-clock performance.
