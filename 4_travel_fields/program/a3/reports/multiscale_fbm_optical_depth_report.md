# Multiscale Cubic-Fade / fBm Optical-Depth Pilot Study

**Status:** PASS

## Configuration

- Octaves: `3`
- Frequencies: `[1, 2, 4]`
- Amplitudes: `[1.0, 0.5, 0.25]`
- Random rays: `160`

## Exact additive decomposition

For a nonnegative multiscale density formed as a weighted sum of octaves, optical depth is exactly the sum of the per-octave optical depths by linearity of integration.

- Maximum direct-vs-separate-octave error: `2.664535e-15`

## Common-refinement hierarchy

All cell-boundary crossing times from all octaves were merged into one sorted partition. Each common-refinement segment lies inside one cell of every octave.

- Maximum hierarchy error: `0.000000e+00`
- Mean hierarchy segments: `22.256`
- Maximum hierarchy segments: `41`

## Conservative octave bounds

Cubic-fade value noise is a convex combination of its eight corner values, so the maximum overlapping lattice vertex provides a conservative density upper bound.

- Maximum sampled bound violation: `0.000000e+00`

## Interpretation

1. Optical depth is exactly additive across octaves.
2. Common refinement is exact but tends toward finest-grid traversal cost.
3. Conservative per-octave bounds can support hierarchical pruning.
4. The next problem is avoiding evaluation of every octave on every finest-level segment.
