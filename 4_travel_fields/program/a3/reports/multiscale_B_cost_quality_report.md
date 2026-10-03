# A3 versus Production LOD Cost–Quality Study

- Rays: `240`
- Frequencies: `[1, 2, 4, 8, 16]`

## Exact A3 reference

- Mean segments: `65.392`
- Mean density evaluations: `1634.792`
- Mean cell reloads: `126.412`

## Fixed cutoff

| K | P95 tau error | P99 transmittance error | Event mismatches | P95 event distance error | Density-eval reduction |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.571e+00 | 4.870e-01 | 56 | 1.523e+00 | 98.8% |
| 2 | 7.010e-01 | 1.684e-01 | 22 | 7.048e-01 | 95.0% |
| 3 | 2.952e-01 | 7.682e-02 | 12 | 3.168e-01 | 85.0% |
| 4 | 1.008e-01 | 2.405e-02 | 5 | 1.063e-01 | 60.1% |
| 5 | 0.000e+00 | 0.000e+00 | 0 | 0.000e+00 | 0.0% |

## Pixel-footprint LOD

| Alpha | Mean K | P95 tau error | P99 transmittance error | Event mismatches | Density-eval reduction |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 5.00 | 0.000e+00 | 0.000e+00 | 0 | 0.0% |
| 1 | 5.00 | 0.000e+00 | 0.000e+00 | 0 | 0.0% |
| 2 | 5.00 | 0.000e+00 | 0.000e+00 | 0 | 0.0% |
| 4 | 4.68 | 8.408e-02 | 2.019e-02 | 4 | 17.2% |
| 8 | 3.85 | 2.525e-01 | 6.217e-02 | 8 | 57.1% |
| 16 | 2.85 | 5.604e-01 | 1.566e-01 | 10 | 83.7% |