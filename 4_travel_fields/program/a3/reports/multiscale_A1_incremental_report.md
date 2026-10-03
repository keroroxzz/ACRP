# A1 Incremental Exact Multiscale Traversal

**Status:** NEEDS REVIEW

- Rays: `300`
- Octaves: `5`
- Frequencies: `[1, 2, 4, 8, 16]`

## Exactness

- P99 optical-depth difference: `7.986893e-01`
- Maximum optical-depth difference: `1.753322e+00`

## A0 naive exact common refinement

- Mean segments: `63.880`
- Mean octave polynomial rebuilds: `319.400`

## A1 incremental exact common refinement

- Mean segments: `63.880`
- Mean octave polynomial rebuilds: `123.913`
- Mean degree-9 polynomial shifts: `195.487`
- Raw rebuild reduction: `61.20%`

If one polynomial shift costs 0.25 of a rebuild, estimated equivalent reduction:
`45.90%`.

If one polynomial shift costs 0.50 of a rebuild, estimated equivalent reduction:
`30.60%`.

A1 preserves the exact common-refinement partition and exact degree-10
optical depth. Only octaves crossing a boundary are rebuilt; the rest are
reparameterized to the next segment origin.
