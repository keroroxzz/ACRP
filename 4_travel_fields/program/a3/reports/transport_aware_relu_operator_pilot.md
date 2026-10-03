# Transport-Aware LOD Pilot: Distribution-Aware ReLU/Erosion Operator

- Rays: `420`
- Full density: `rho=max(0,B-lambda R)` with `lambda=1.35`
- Residual mean: `1.082381e-01`
- Residual standard deviation: `5.249822e-02`

Three retained-backbone approximations are compared:

1. raw cutoff: `max(0,B)`;
2. mean substitution: `max(0,B-lambda E[R])`;
3. distribution-aware conditional mean: `h(B)=E[(B-lambda R)_+]`.

## Results

| Method | Mean signed tau error | Mean abs tau error | P95 abs tau error | P99 T error |
|---|---:|---:|---:|---:|
| `raw` | 4.155e-01 | 4.155e-01 | 6.263e-01 | 2.753e-01 |
| `mean_substitution` | -7.089e-03 | 3.813e-02 | 9.528e-02 | 6.432e-02 |
| `distribution_aware` | -2.079e-03 | 3.663e-02 | 9.271e-02 | 6.107e-02 |

## Relative improvement

- Mean substitution vs raw, P95 tau-error reduction: `84.79%`.
- Distribution-aware vs raw, P95 tau-error reduction: `85.20%`.
- Distribution-aware vs mean substitution, P95 tau-error reduction: `2.70%`.

## Operator-algebra consequence

`max(0,f)` converts a polynomial field into a piecewise-polynomial field whose
partition additionally contains the roots of `f`.  For the full model,
`B-lambda R` remains degree at most 9 on a common cell segment, but exact
integration requires isolating all roots where the sign changes.

The distribution-aware LOD replaces the unresolved nonlinear subgraph by a
one-dimensional transfer function of the retained backbone.  It therefore
avoids high-frequency traversal, but the transfer function is no longer a
polynomial unless it is approximated by a piecewise polynomial or rational
surrogate.