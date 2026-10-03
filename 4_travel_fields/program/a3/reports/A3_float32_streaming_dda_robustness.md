# A3 Float32 Streaming-DDA Robustness

- Finest frequencies: `16, 32, 64, 128`
- World offsets: `0, 1e3, 1e5, 1e7`

Three coordinate pipelines are tested:

1. `world32`: large world-space origins are used directly;
2. `local32_from_world32`: two float32 world values are subtracted;
3. `local32_direct`: host precision forms the local origin before float32 traversal.

## Aggregate topology mismatches

- `world32`: `1772`
- `local32_from_world32`: `1772`
- `local32_direct`: `759`

## Per-case results

| Case | Rays | Topology mismatches | P99 time error | Max time error |
|---|---:|---:|---:|---:|
| `F16_offset0_world32` | 180 | 6 | 4.600e-06 | 1.014e-05 |
| `F16_offset0_local32_from_world32` | 180 | 6 | 4.600e-06 | 1.014e-05 |
| `F16_offset0_local32_direct` | 180 | 6 | 4.600e-06 | 1.014e-05 |
| `F16_offset1000_world32` | 180 | 21 | 8.332e-04 | 1.069e-03 |
| `F16_offset1000_local32_from_world32` | 180 | 21 | 8.332e-04 | 1.069e-03 |
| `F16_offset1000_local32_direct` | 180 | 5 | 2.833e-06 | 3.865e-06 |
| `F16_offset100000_world32` | 180 | 132 | 6.941e-02 | 7.696e-02 |
| `F16_offset100000_local32_from_world32` | 180 | 132 | 6.941e-02 | 7.696e-02 |
| `F16_offset100000_local32_direct` | 180 | 5 | 2.283e-06 | 3.537e-06 |
| `F16_offset1e+07_world32` | 180 | 180 | 1.306e-01 | 1.439e-01 |
| `F16_offset1e+07_local32_from_world32` | 180 | 180 | 1.306e-01 | 1.439e-01 |
| `F16_offset1e+07_local32_direct` | 180 | 6 | 2.459e-06 | 3.289e-06 |
| `F32_offset0_world32` | 180 | 19 | 3.234e-06 | 4.160e-06 |
| `F32_offset0_local32_from_world32` | 180 | 19 | 3.234e-06 | 4.160e-06 |
| `F32_offset0_local32_direct` | 180 | 19 | 3.234e-06 | 4.160e-06 |
| `F32_offset1000_world32` | 180 | 35 | 1.930e-02 | 4.440e-02 |
| `F32_offset1000_local32_from_world32` | 180 | 35 | 1.930e-02 | 4.440e-02 |
| `F32_offset1000_local32_direct` | 180 | 16 | 3.492e-06 | 9.658e-06 |
| `F32_offset100000_world32` | 180 | 161 | 4.354e-02 | 4.606e-02 |
| `F32_offset100000_local32_from_world32` | 180 | 161 | 4.354e-02 | 4.606e-02 |
| `F32_offset100000_local32_direct` | 180 | 14 | 4.156e-06 | 1.494e-05 |
| `F32_offset1e+07_world32` | 180 | 180 | 5.962e-02 | 6.324e-02 |
| `F32_offset1e+07_local32_from_world32` | 180 | 180 | 5.962e-02 | 6.324e-02 |
| `F32_offset1e+07_local32_direct` | 180 | 10 | 3.446e-06 | 5.906e-06 |
| `F64_offset0_world32` | 180 | 46 | 6.929e-06 | 2.108e-05 |
| `F64_offset0_local32_from_world32` | 180 | 46 | 6.929e-06 | 2.108e-05 |
| `F64_offset0_local32_direct` | 180 | 46 | 6.929e-06 | 2.108e-05 |
| `F64_offset1000_world32` | 180 | 68 | 2.092e-02 | 2.153e-02 |
| `F64_offset1000_local32_from_world32` | 180 | 68 | 2.092e-02 | 2.153e-02 |
| `F64_offset1000_local32_direct` | 180 | 56 | 3.672e-06 | 1.481e-05 |
| `F64_offset100000_world32` | 180 | 155 | 2.183e-02 | 2.490e-02 |
| `F64_offset100000_local32_from_world32` | 180 | 155 | 2.183e-02 | 2.490e-02 |
| `F64_offset100000_local32_direct` | 180 | 52 | 5.054e-06 | 1.783e-05 |
| `F64_offset1e+07_world32` | 180 | 180 | 3.102e-02 | 3.146e-02 |
| `F64_offset1e+07_local32_from_world32` | 180 | 180 | 3.102e-02 | 3.146e-02 |
| `F64_offset1e+07_local32_direct` | 180 | 53 | 4.924e-06 | 1.718e-05 |
| `F128_offset0_world32` | 180 | 107 | 5.080e-06 | 5.726e-06 |
| `F128_offset0_local32_from_world32` | 180 | 107 | 5.080e-06 | 5.726e-06 |
| `F128_offset0_local32_direct` | 180 | 107 | 5.080e-06 | 5.726e-06 |
| `F128_offset1000_world32` | 180 | 143 | 9.862e-03 | 1.086e-02 |
| `F128_offset1000_local32_from_world32` | 180 | 143 | 9.862e-03 | 1.086e-02 |
| `F128_offset1000_local32_direct` | 180 | 134 | 2.022e-05 | 3.192e-05 |
| `F128_offset100000_world32` | 180 | 159 | 1.512e-02 | 1.822e-02 |
| `F128_offset100000_local32_from_world32` | 180 | 159 | 1.512e-02 | 1.822e-02 |
| `F128_offset100000_local32_direct` | 180 | 120 | 1.817e-05 | 2.827e-05 |
| `F128_offset1e+07_world32` | 180 | 180 | 1.567e-02 | 1.604e-02 |
| `F128_offset1e+07_local32_from_world32` | 180 | 180 | 1.567e-02 | 1.604e-02 |
| `F128_offset1e+07_local32_direct` | 180 | 110 | 5.398e-06 | 1.269e-05 |

## Architectural rule

The GPU kernel should receive a volume-local or camera-relative ray origin
already formed in higher precision. Subtracting two large float32 world
coordinates is not equivalent and can erase finest-cell information before
the DDA begins.

Integer next-plane indices remain authoritative. Boundary times are always
recomputed from those indices rather than accumulated.