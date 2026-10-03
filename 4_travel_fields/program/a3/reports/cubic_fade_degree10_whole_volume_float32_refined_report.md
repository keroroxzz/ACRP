# Refined Whole-Volume Float32 Validation

**Status:** NEEDS REVIEW

The first run exposed a false convergence rule: an absolute optical-depth
tolerance scaled by `max(1, tau)` was far too loose for small remaining
optical-depth targets. This run replaces it with:

- normalized spatial error estimate `|tau-target| / tau'(u)`;
- normalized bracket-width convergence;
- derivative threshold relative to cell optical depth, without a unit floor;
- fixed 16-step loop with convergence freeze;
- bracket-midpoint return.

## Results

- Hit/miss mismatches: `0`
- P99 event-distance error: `5.730958e-01`
- Maximum event-distance error: `7.078905e-01`
- P99 independent optical-depth residual: `5.251658e-01`
- Maximum independent optical-depth residual: `1.048976e+00`
- P99 whole-ray optical-depth absolute error: `2.046550e-06`
- Maximum whole-ray relative error: `2.065553e-06`
- Mean genuine fallback count: `0.053`
- P95 genuine fallback count: `0.000`
- Mean convergence-freeze iteration: `3.128`
- P95 convergence-freeze iteration: `6.000`
