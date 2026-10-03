# A3 Transport-Safe Tie Envelope

- Optical-depth merge budget: `1.0e-06`

| Case | Topology mismatches | P99 tau error | Max tau error | Mean merged groups | Bound coverage |
|---|---:|---:|---:|---:|---:|
| `F16_immediate` | 0 | 1.022e+00 | 1.087e+00 | 0.00 | 0.0% |
| `F16_ulp` | 0 | 1.022e+00 | 1.087e+00 | 0.00 | 0.0% |
| `F16_transport` | 0 | 1.022e+00 | 1.087e+00 | 0.00 | 0.0% |
| `F128_immediate` | 2 | 8.444e-01 | 8.811e-01 | 0.06 | 0.0% |
| `F128_ulp` | 16 | 8.445e-01 | 8.812e-01 | 2.94 | 0.0% |
| `F128_transport` | 2 | 8.444e-01 | 8.811e-01 | 0.06 | 0.0% |

The cached-state evaluator reloads only octaves indicated by each emitted mask. Coalesced crossings are committed as one transaction before the next nonzero segment. The local bound is `delta_tau <= sigma_t rho_max delta_t`; it bounds omitted collapsed length, while transactional mask commit prevents a tiny tie split from causing a much longer stale-state interval.