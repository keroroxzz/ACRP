# Transport-Aware LOD Pilot: Second-Cumulant Correction

- Training rays: `720`
- Held-out test rays: `480`
- Frequencies: `[1, 2, 4, 8, 16]`

The residual optical depth is modeled as having mean `mu_tau` and
conditional variance `v_tau(L) = a L + b L^2`.  The second-cumulant
transmittance approximation uses an effective optical depth
`tau_eff = tau_low + mu_tau - v_tau/2`.

## K=3

- Omitted mean density: `6.150665e-02`
- Variance fit: `a=1.024534e-04`, `b=3.667607e-05`

### Pointwise held-out errors

| Method | Mean abs tau error | P95 abs tau error | Mean abs T error | P99 T error |
|---|---:|---:|---:|---:|
| `raw_cutoff` | 1.943e-01 | 2.869e-01 | 3.088e-02 | 8.225e-02 |
| `mean_compensated` | 1.678e-02 | 4.343e-02 | 2.795e-03 | 1.323e-02 |
| `second_cumulant` | 1.678e-02 | 4.308e-02 | 2.795e-03 | 1.327e-02 |

### Length-binned expected-transmittance errors

| Method | Weighted mean bin error | Maximum bin error |
|---|---:|---:|
| `raw_cutoff` | 3.088e-02 | 4.146e-02 |
| `mean_compensated` | 3.441e-04 | 8.887e-04 |
| `second_cumulant` | 3.439e-04 | 9.236e-04 |

## K=4

- Omitted mean density: `2.039978e-02`
- Variance fit: `a=1.990275e-05`, `b=8.005644e-07`

### Pointwise held-out errors

| Method | Mean abs tau error | P95 abs tau error | Mean abs T error | P99 T error |
|---|---:|---:|---:|---:|
| `raw_cutoff` | 6.492e-02 | 9.856e-02 | 9.671e-03 | 2.528e-02 |
| `mean_compensated` | 5.761e-03 | 1.458e-02 | 9.218e-04 | 4.107e-03 |
| `second_cumulant` | 5.761e-03 | 1.462e-02 | 9.220e-04 | 4.114e-03 |

### Length-binned expected-transmittance errors

| Method | Weighted mean bin error | Maximum bin error |
|---|---:|---:|
| `raw_cutoff` | 9.671e-03 | 1.311e-02 |
| `mean_compensated` | 8.590e-05 | 2.519e-04 |
| `second_cumulant` | 8.846e-05 | 2.566e-04 |

## Interpretation

- Mean compensation targets individual-ray optical-depth bias.
- The second cumulant targets ensemble expected transmittance, not the
  exact transmittance of each realization.
- Therefore it should be judged primarily by binned or ensemble
  transmittance statistics; pointwise error can remain unchanged or worsen.
- A useful next model must include directional correlation or a local
  residual statistic rather than only path length.