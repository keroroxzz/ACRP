# A3 Nested Dyadic Traversal: Single Finest-Grid DDA

**Status:** PASS

## Configuration

- Rays tested: `1028`
- Adversarial rays: `28`
- Octave frequencies: `[1, 2, 4, 8, 16]`

The baseline constructs the common-refinement event stream by merging one
boundary stream per octave.  The candidate traverses only the finest grid and
derives an octave-crossing bitmask from the finest integer plane index.

For finest frequency `F` and octave frequency `f`, octave `f` crosses at a
finest-plane index `k` exactly when:

`k mod (F/f) = 0`.

Because all ratios are powers of two, this condition can be implemented with
bit tests.

## Exact equivalence

- Event-list mismatches: `0`
- Maximum boundary-time difference: `0.000000e+00`
- Maximum reload-count difference: `0`
- All event streams match: `True`

## Traversal statistics

- Mean events per ray, including entry and exit:
  `64.232`
- P95 events per ray: `100.0`
- Mean octave reloads per ray:
  `117.687`
- Maximum simultaneous octave crossings:
  `5`

## State reduction

A direct multi-octave DDA conceptually stores one next-boundary clock for each
axis of each octave:

`3 * L = 15` clocks.

The nested dyadic traversal stores only the finest-grid clocks:

`3` clocks.

This is a clock-state reduction of
`80.0%`.

Octave-local cell state is still required for density evaluation, but its
updates are driven by a compact bitmask rather than by independent DDA clocks.

## Architecture

At each finest-grid crossing:

1. advance the single 3D DDA;
2. obtain the crossed finest plane index `k`;
3. compute the octave mask from divisibility or trailing-zero tests;
4. rebuild only the octave states whose mask bits are set;
5. continue GL5 exact common-segment consumption.

This preserves the exact A3 partition while substantially simplifying traversal
state and making the nested-grid structure explicit.
