# A3 Streaming Incremental Finest-Grid DDA

**Status:** PASS

## Validation

- Rays: `2027`
- Adversarial rays: `27`
- Event-stream mismatches: `0`
- Mask mismatches: `0`
- Event-count mismatch: `0`
- Maximum boundary-time difference: `0.000000e+00`

The streaming iterator exactly reproduces the precomputed finest-grid event
stream while storing no event array.

## Critical numerical result

The initial implementation advanced each axis with:

`t_max += t_delta`.

An exact diagonal adversarial ray exposed accumulated floating-point drift:
simultaneous crossings could separate by a few `1e-12`, producing duplicate
events and an incorrect octave-update sequence.

The corrected rule stores the integer next-plane index `k` and recomputes:

`t_max = (k/F - ray_origin_axis) / ray_direction_axis`

after every advance.

This preserves exact-grid tie coherence in float64 and makes the integer plane
index the authoritative traversal state.

## Streaming kernel state

- three current finest-cell indices;
- three integer next-plane indices;
- three `t_max` values;
- three direction signs;
- current octave cell/corner states.

Memory is `O(1)` for traversal plus `O(L)` for octave density state.

## Final A3-GN traversal

1. stream the finest dyadic grid;
2. merge tied axes;
3. derive the octave mask from integer plane indices;
4. consume the segment with exact GL5;
5. reload masked octaves;
6. build one terminal degree-9 polynomial;
7. solve the degree-10 optical-depth clock with safeguarded Newton.
