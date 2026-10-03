# Single-Scattering Image Benchmark

- Resolution: `24 x 16`
- Target optical depth: `0.68`

| Method | Mean abs L error | P95 abs L error | Hit mismatches | Cost ratio vs A3 |
|---|---:|---:|---:|---:|
| `A3` | 0.000e+00 | 0.000e+00 | 0 | 1.000 |
| `K4mean` | 2.948e-03 | 5.577e-03 | 2 | 0.472 |
| `RM24` | 7.876e-05 | 3.533e-04 | 0 | 0.034 |
| `RM48` | 9.096e-06 | 4.515e-05 | 0 | 0.067 |
| `RM96` | 2.378e-06 | 9.667e-06 | 0 | 0.133 |

The image uses one deterministic free-path event per pixel and exact or approximate sun transmittance. Cost is a scalar density-evaluation proxy, not GPU wall time.