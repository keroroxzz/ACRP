# Transport-Aware LOD Pilot: Conditional Mean under Low-Frequency Modulation

- Rays: `650`
- Full model: `rho(x) = B(x) + m(x) R(x)`
- Low frequencies: `[1, 2, 4]`
- Residual frequencies: `[8, 16]`
- Modulation range: `[0.20, 1.50]`

The omitted residual is independent and stationary, while its amplitude is
controlled by a retained low-frequency modulation field.  Three LOD models
are compared:

1. raw cutoff: `B`;
2. unconditional global mean: `B + E[m] E[R]`;
3. conditional mean: `B + m(x) E[R]`.

## Polynomial closure

Within a common-refinement segment, `B`, `m`, and `R` each have degree at
most 9 along the ray.  Therefore `mR` has degree at most 18 and the full
optical depth has degree at most 19.  GL10 is exact for full-segment
consumption.  Both compensated LOD models remain degree at most 9 and use
exact GL5 consumption.

## Results

| Method | Mean signed tau error | Mean abs tau error | P95 abs tau error | P99 T error |
|---|---:|---:|---:|---:|
| `raw` | -2.123e-01 | 2.123e-01 | 3.252e-01 | 9.361e-02 |
| `global_mean` | 1.776e-03 | 2.700e-02 | 6.818e-02 | 2.393e-02 |
| `conditional_mean` | 2.031e-03 | 1.966e-02 | 4.866e-02 | 1.675e-02 |

## Relative improvement

- Global mean versus raw, P95 tau-error reduction: `79.04%`.
- Conditional mean versus raw, P95 tau-error reduction: `85.04%`.
- Conditional mean versus global mean, P95 tau-error reduction: `28.64%`.

## Interpretation

A global residual mean cannot follow spatial changes in residual amplitude.
The conditional model reuses the retained low-frequency modulation and adds
no residual texture lookup.  This is the first LOD correction in the study
that exploits procedural-graph structure rather than only global statistics.