# Final Whole-Volume Float32 Arithmetic Validation

**Status:** PASS

## Critical solver correction

A bracket midpoint is safe only after the bracket itself has contracted.
When Newton reaches a small spatial residual while the opposite bracket end
remains distant, returning the midpoint introduces a large error. The final
rule is:

- return the converged Newton point when `|tau-target|/tau'(u)` is small;
- return the bracket midpoint only when the bracket is small or the fixed
  iteration budget expires without point convergence.

## Results

- Hit/miss mismatches: `0`
- P99 whole-ray optical-depth absolute error: `2.046550e-06`
- Maximum whole-ray relative error: `2.065553e-06`
- P95 event-distance error: `3.786793e-06`
- P99 event-distance error: `7.607942e-05`
- Maximum event-distance error: `7.441661e-04`
- P99 independent event optical-depth residual: `4.807531e-06`
- Maximum event optical-depth residual: `1.112594e-05`
- Mean genuine fallbacks: `0.053`
- P95 genuine fallbacks: `0.000`
- Mean convergence iteration: `3.128`
- P95 convergence iteration: `6.000`
