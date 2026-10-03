# Cubic-Fade 3D Degree-10 Whole-Volume Validation

**Overall status:** PASS

## Configuration

- Random seed: `20260718`
- Voxel grid: `6 × 5 × 7`
- Random rays: `180`
- Adversarial/boundary rays: `50`
- Lattice values: nonnegative, with 42% exact-zero vertices
- Extinction scale: `1.35`
- Analytic implementation:
  - slab ray-box intersection
  - 3D DDA with simultaneous-axis tie handling
  - degree-9 density polynomial per cell
  - degree-10 optical-depth polynomial per cell
  - terminal safeguarded Newton inversion
- Independent reference:
  - all integer-plane crossing times collected and globally sorted
  - midpoint interval classification
  - direct cubic-fade density evaluation
  - adaptive Gauss–Kronrod quadrature (`scipy.integrate.quad`)
  - Brent root solve (`scipy.optimize.brentq`)

## DDA geometry

- Segmentation mismatches: `0`
- Maximum segment-boundary error: `3.552714e-15`

## Whole-ray optical depth

- Maximum absolute total optical-depth error: `1.354472e-14`
- Maximum relative total optical-depth error: `1.178560e-14`

## Free-path event

- Hit/miss classification mismatches: `0`
- Maximum event-distance error: `6.841354e-09`
- Maximum independent event optical-depth residual: `1.970230e-12`
- Inversion failures: `0`

## Safeguarded Newton

- Mean iterations: `4.118`
- P95 iterations: `7.000`
- Maximum iterations: `22`
- Mean fallback count: `0.072`
- Maximum fallback count: `2`

## Optical-depth countdown

- Maximum shifted-ray countdown error: `8.993297e-09`

## Result

This validation closes the full single-scale primitive:

\[
	ext{ray-box}
ightarrow
	ext{3D DDA}
ightarrow
	ext{exact degree-10 cell optical depth}
ightarrow
	ext{whole-cell budget consumption}
ightarrow
	ext{one terminal monotone inversion}.
\]

The analytic traversal agrees with an independently segmented, directly
evaluated, adaptively integrated, Brent-inverted reference.

The next validation layer should target GPU-relevant arithmetic:

1. repeat the coefficient construction and Horner evaluation in `float32`;
2. compare local-time versus normalized-cell-time parameterizations;
3. stress cancellation at near-zero densities and near-cell-exit targets;
4. derive conservative tolerances and fallback rules;
5. only then move to multiscale/fBm representations.
