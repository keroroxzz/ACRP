# Multiscale Cloud Method Comparison

- Rays: `25`
- Octaves: `4`
- Frequencies: `[1, 2, 4, 8]`
- Amplitudes: `[1.0, 0.5, 0.25, 0.125]`

## Method A — exact common refinement

- Mean segments: `40.640`
- Mean octave integrals: `162.560`

## Method B — fixed octave cutoff

| Config | P95 tau error | Max tau error | P95 T error | Segment reduction | Integral reduction |
|---|---:|---:|---:|---:|---:|
| `K1` | 1.871e+00 | 2.036e+00 | 4.745e-01 | 87.8% | 96.9% |
| `K2` | 7.816e-01 | 8.297e-01 | 1.262e-01 | 75.1% | 87.5% |
| `K3` | 2.805e-01 | 2.995e-01 | 4.005e-02 | 49.7% | 62.3% |

## Method C — exact backbone + bounded adaptive residual

| Config | Bound | Bound ok | P95 tau error | Max tau error | P95 T error | Segment reduction | Integral reduction | Mean bound queries |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|
| `B1_eps0.1` | 1.0e-01 | yes | 1.155e-15 | 1.332e-15 | 5.551e-17 | -87.4% | -53.2% | 55.6 |
| `B1_eps0.03` | 3.0e-02 | yes | 1.155e-15 | 1.332e-15 | 5.551e-17 | -87.4% | -53.2% | 55.6 |
| `B1_eps0.01` | 1.0e-02 | yes | 1.155e-15 | 1.332e-15 | 5.551e-17 | -87.4% | -53.2% | 55.6 |
| `B1_eps0.003` | 3.0e-03 | yes | 1.155e-15 | 1.332e-15 | 5.551e-17 | -87.4% | -53.2% | 55.6 |
| `B2_eps0.1` | 1.0e-01 | yes | 1.155e-15 | 1.332e-15 | 5.551e-17 | -75.2% | -50.2% | 40.7 |
| `B2_eps0.03` | 3.0e-02 | yes | 1.155e-15 | 1.332e-15 | 5.551e-17 | -75.2% | -50.2% | 40.7 |
| `B2_eps0.01` | 1.0e-02 | yes | 1.155e-15 | 1.332e-15 | 5.551e-17 | -75.2% | -50.2% | 40.7 |
| `B2_eps0.003` | 3.0e-03 | yes | 1.155e-15 | 1.332e-15 | 5.551e-17 | -75.2% | -50.2% | 40.7 |

## Interpretation

- A is the exact multiscale ground truth, but its cost follows the finest partition.
- B is cheap and simple, but has no per-ray error guarantee.
- C is the production-oriented candidate because it trades work for a declared optical-depth error budget.
- Since `|exp(-tau)-exp(-(tau+delta))| <= 1-exp(-|delta|) <= |delta|`, the same optical-depth budget also bounds transmittance error.
- All declared C bounds respected in this run: `True`.