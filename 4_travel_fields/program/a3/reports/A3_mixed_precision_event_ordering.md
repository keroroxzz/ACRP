# A3 Mixed-Precision Event Ordering

| Case | Topology mismatches | P99 time error | P99 tau error | Max tau error |
|---|---:|---:|---:|---:|
| `F16_f32_div` | 4 | 1.202e-06 | 8.544e-15 | 9.326e-15 |
| `F16_f32_cross` | 5 | 1.203e-06 | 1.091e-14 | 1.110e-14 |
| `F16_mixed64` | 5 | 6.587e-07 | 1.776e-15 | 1.776e-15 |
| `F64_f32_div` | 8 | 2.230e-02 | 3.738e-14 | 4.041e-14 |
| `F64_f32_cross` | 8 | 2.230e-02 | 3.738e-14 | 4.041e-14 |
| `F64_mixed64` | 10 | 1.656e-06 | 3.109e-15 | 3.109e-15 |
| `F128_f32_div` | 11 | 1.057e-02 | 5.324e-13 | 5.520e-13 |
| `F128_f32_cross` | 11 | 1.051e-02 | 2.250e-11 | 2.517e-11 |
| `F128_mixed64` | 9 | 6.699e-07 | 6.120e-15 | 6.217e-15 |

`mixed64` uses float32 local ray state but float64 event clocks. Cross multiplication removes division comparison error but cannot recover information already lost in float32 numerators and directions. Topology and optical-depth errors are reported separately.