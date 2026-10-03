# Transport-Aware LOD Pilot: Mean Compensation

- Rays: `300`
- Frequencies: `[1, 2, 4, 8, 16]`
- Exact octave spatial means: `[0.331923, 0.156378, 0.078129, 0.041107, 0.0204]`

For cutoff K, the pilot adds the exact global spatial mean of all omitted
octaves as a constant residual density along the ray.

| K | Method | Mean signed tau error | P95 abs tau error | P99 T error |
|---:|---|---:|---:|---:|
| 3 | raw | -1.964e-01 | 2.935e-01 | 6.839e-02 |
| 3 | mean | -2.434e-04 | 4.693e-02 | 1.120e-02 |
| 4 | raw | -6.510e-02 | 9.524e-02 | 2.513e-02 |
| 4 | mean | -2.694e-05 | 1.324e-02 | 4.221e-03 |

This is only a first statistical correction. It preserves global mean
extinction but not local variance, covariance along the ray, or the free-path
distribution. Its value is diagnostic: it tests how much of raw cutoff error
is systematic mean loss versus unresolved spatial fluctuation.