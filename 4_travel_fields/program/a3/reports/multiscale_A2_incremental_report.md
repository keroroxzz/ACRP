# A2 Stable Incremental Exact Multiscale Traversal

**Status:** NEEDS REVIEW

## Motivation

The first A1 design shifted degree-9 world-distance polynomials after every
common boundary. For high-frequency octaves, coefficients scale with powers
of frequency. Repeated binomial shifting caused severe cancellation and was
not numerically stable even in float64.

A2 avoids polynomial shifting entirely.

Each octave polynomial stays anchored at the entry of its current octave cell.
For a common segment of length `L`, octave `i` contributes:

`F_i(offset_i + L) - F_i(offset_i)`,

where `F_i` is its degree-10 antiderivative. Only an octave crossing its own
cell boundary is rebuilt.

## Results

- Rays: `500`
- Octaves: `5`
- Frequencies: `[1, 2, 4, 8, 16]`
- P99 optical-depth difference A2 vs A0: `8.052562e-01`
- Maximum optical-depth difference A2 vs A0: `1.753322e+00`

## Cost counts

A0 naive common refinement:

- Mean segments: `64.652`
- Mean polynomial rebuilds: `323.260`

A2 stable incremental traversal:

- Mean segments: `64.652`
- Mean polynomial rebuilds: `125.442`
- Mean antiderivative evaluations: `646.520`
- Polynomial rebuild reduction: `61.19%`

## Consequence

The stable exact architecture should not merge all octave coefficients into one
repeatedly shifted world-distance polynomial. It should retain one local
polynomial clock per active octave and sum their antiderivative differences.

For terminal event inversion, evaluate:

`A(s) = sum_i [F_i(offset_i+s)-F_i(offset_i)]`

and:

`A'(s) = sum_i rho_i(offset_i+s)`.

This remains monotone and can use safeguarded Newton, while avoiding unstable
coefficient shifting.
