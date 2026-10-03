# A3 Exact GL5 Multiscale Traversal

**Status:** PASS

A direction-aware grid-boundary classifier is required. Using only
`nextafter(t,+inf)` is not robust after world-to-octave scaling; it can select
the wrong side of a coarse grid plane.

## Validation

- Rays: `400`
- Frequencies: `[1, 2, 4, 8, 16]`
- P99 optical-depth difference A3 vs A0: `4.791230e-12`
- Maximum optical-depth difference A3 vs A0: `5.677681e-12`

## Cost counts

A0:
- Mean segments: `64.347`
- Mean degree-9 polynomial rebuilds: `321.738`

A3:
- Mean segments: `64.347`
- Mean cell/corner reloads: `124.825`
- Mean octave density evaluations: `1608.688`
- Cell reload reduction: `61.20%`

Five-point Gauss-Legendre is exact for the degree-at-most-9 total density on
each common segment. Only the terminal event segment needs degree-10
coefficient construction and safeguarded Newton.
