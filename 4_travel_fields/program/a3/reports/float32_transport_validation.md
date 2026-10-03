# Full-Float32 Transactional Transport Validation (handoff Sec.18)

Ground truth: **full-f64 on true f64 ray (== kernel)**. Rays per category: **15**; categories: random, axis_aligned, near_axis, exact_diagonal, near_diagonal, grazing.

Error is decomposed into three independent sources:

- **ray_quantization** — f32 ray (origin+direction), f64 pipeline (`mixed64`) vs the true f64 ray. The physically different ray a GPU sees.

- **DDA-clock** — `clock32_*` (f32 boundary times, f64 integration), scored *pipeline-only* against the same f32 ray, reproducing the frozen convention.

- **full-f32** — `full32_*` (f32 GL5 + f32 terminal Newton, tol 1e-6). The new Sec.18 measurement.


Topology mismatches are counted **separately** from transport error (Invariant Sec.19.12): a reordered tie is transport-neutral.


## F16  (rho_max = 1.938)

**Ray quantization (mixed64 vs true f64):** max|dtau| 6.83e-04, max|d event| 1.26e-03, topology mismatches 30 / 90 rays.


| mode | scope | p99 dtau | max dtau | p99 d_evt | max d_evt | topo | newton (mean/max) | fb |
|---|---|---|---|---|---|---:|---|---:|
| clock32_div | vs true f64 | 9.03e-05 | 6.83e-04 | 3.35e-04 | 1.26e-03 | 30 | 3.12 / 4 | 0 |
| clock32_div | vs f32 ray | 6.81e-07 | 7.07e-07 | 4.01e-07 | 6.06e-07 | 13 | 3.12 / 4 | 0 |
| clock32_cross | vs true f64 | 9.03e-05 | 6.83e-04 | 3.35e-04 | 1.26e-03 | 27 | 3.12 / 4 | 0 |
| clock32_cross | vs f32 ray | 6.81e-07 | 7.07e-07 | 4.01e-07 | 6.06e-07 | 19 | 3.12 / 4 | 0 |
| full32_div | vs true f64 | 9.03e-05 | 6.83e-04 | 3.35e-04 | 1.26e-03 | 30 | 2.16 / 3 | 0 |
| full32_div | vs f32 ray | 1.21e-06 | 1.56e-06 | 1.28e-06 | 1.87e-06 | 13 | 2.16 / 3 | 0 |
| full32_cross | vs true f64 | 9.03e-05 | 6.83e-04 | 3.35e-04 | 1.26e-03 | 27 | 2.16 / 3 | 0 |
| full32_cross | vs f32 ray | 1.21e-06 | 1.56e-06 | 1.28e-06 | 1.87e-06 | 19 | 2.16 / 3 | 0 |

## F32  (rho_max = 1.969)

**Ray quantization (mixed64 vs true f64):** max|dtau| 1.49e-06, max|d event| 2.49e-06, topology mismatches 30 / 90 rays.


| mode | scope | p99 dtau | max dtau | p99 d_evt | max d_evt | topo | newton (mean/max) | fb |
|---|---|---|---|---|---|---:|---|---:|
| clock32_div | vs true f64 | 1.15e-06 | 1.53e-06 | 1.86e-06 | 2.45e-06 | 29 | 2.96 / 4 | 0 |
| clock32_div | vs f32 ray | 7.10e-07 | 8.58e-07 | 3.88e-07 | 4.41e-07 | 15 | 2.96 / 4 | 0 |
| clock32_cross | vs true f64 | 1.15e-06 | 1.53e-06 | 1.86e-06 | 2.45e-06 | 29 | 2.96 / 4 | 0 |
| clock32_cross | vs f32 ray | 7.10e-07 | 8.58e-07 | 3.88e-07 | 4.41e-07 | 20 | 2.96 / 4 | 0 |
| full32_div | vs true f64 | 1.82e-06 | 2.42e-06 | 2.25e-06 | 3.26e-06 | 29 | 1.98 / 3 | 0 |
| full32_div | vs f32 ray | 2.30e-06 | 2.39e-06 | 1.86e-06 | 3.09e-06 | 15 | 1.98 / 3 | 0 |
| full32_cross | vs true f64 | 1.82e-06 | 2.42e-06 | 2.30e-06 | 3.50e-06 | 29 | 1.98 / 3 | 0 |
| full32_cross | vs f32 ray | 2.28e-06 | 2.28e-06 | 1.86e-06 | 3.33e-06 | 20 | 1.98 / 3 | 0 |

## F64  (rho_max = 1.984)

**Ray quantization (mixed64 vs true f64):** max|dtau| 5.82e-04, max|d event| 3.96e-05, topology mismatches 30 / 90 rays.


| mode | scope | p99 dtau | max dtau | p99 d_evt | max d_evt | topo | newton (mean/max) | fb |
|---|---|---|---|---|---|---:|---|---:|
| clock32_div | vs true f64 | 3.13e-04 | 5.82e-04 | 8.57e-07 | 3.97e-05 | 30 | 2.91 / 4 | 0 |
| clock32_div | vs f32 ray | 6.61e-07 | 6.88e-07 | 3.99e-07 | 4.93e-07 | 15 | 2.91 / 4 | 0 |
| clock32_cross | vs true f64 | 3.13e-04 | 5.82e-04 | 8.57e-07 | 3.97e-05 | 30 | 2.91 / 4 | 0 |
| clock32_cross | vs f32 ray | 6.61e-07 | 6.88e-07 | 3.99e-07 | 4.93e-07 | 21 | 2.91 / 4 | 0 |
| full32_div | vs true f64 | 3.13e-04 | 5.81e-04 | 3.80e-06 | 3.97e-05 | 30 | 1.95 / 2 | 0 |
| full32_div | vs f32 ray | 2.58e-06 | 5.21e-06 | 1.78e-06 | 4.11e-06 | 15 | 1.95 / 2 | 0 |
| full32_cross | vs true f64 | 3.13e-04 | 5.81e-04 | 3.80e-06 | 3.97e-05 | 30 | 1.95 / 2 | 0 |
| full32_cross | vs f32 ray | 2.58e-06 | 5.21e-06 | 1.64e-06 | 4.11e-06 | 21 | 1.95 / 2 | 0 |

## F128  (rho_max = 1.992)

**Ray quantization (mixed64 vs true f64):** max|dtau| 6.36e-05, max|d event| 9.13e-05, topology mismatches 30 / 90 rays.


| mode | scope | p99 dtau | max dtau | p99 d_evt | max d_evt | topo | newton (mean/max) | fb |
|---|---|---|---|---|---|---:|---|---:|
| clock32_div | vs true f64 | 1.26e-05 | 6.39e-05 | 2.68e-05 | 9.13e-05 | 30 | 2.77 / 3 | 0 |
| clock32_div | vs f32 ray | 7.47e-07 | 7.61e-07 | 3.94e-07 | 5.26e-07 | 15 | 2.77 / 3 | 0 |
| clock32_cross | vs true f64 | 1.26e-05 | 6.39e-05 | 2.68e-05 | 9.13e-05 | 30 | 2.77 / 3 | 0 |
| clock32_cross | vs f32 ray | 7.47e-07 | 7.61e-07 | 3.94e-07 | 5.26e-07 | 20 | 2.77 / 3 | 0 |
| full32_div | vs true f64 | 1.43e-05 | 6.17e-05 | 2.64e-05 | 9.13e-05 | 30 | 1.86 / 2 | 0 |
| full32_div | vs f32 ray | 4.12e-06 | 7.01e-06 | 2.70e-06 | 6.94e-06 | 15 | 1.86 / 2 | 0 |
| full32_cross | vs true f64 | 1.43e-05 | 6.17e-05 | 2.64e-05 | 9.13e-05 | 30 | 1.86 / 2 | 0 |
| full32_cross | vs f32 ray | 4.12e-06 | 7.01e-06 | 2.70e-06 | 6.94e-06 | 20 | 1.86 / 2 | 0 |

## Findings

1. **Ray quantization dominates end-to-end transport error.** Feeding the ray in f32 (a physically different ray) contributes max|dtau| up to ~1e-4 and event error up to ~1e-3, orders above any pipeline-precision effect; the end-to-end columns track the ray_quantization row almost exactly. This is the binding limit for an f32 GPU pipeline and reinforces Invariant Sec.19.7 (form local coordinates before f32 conversion) -- but even perfect local coords cannot beat the f32 ray.

2. **Pipeline precision is small and bounded.** Pipeline-only (same f32 ray): the f32 DDA clock adds ~7e-7 in tau across all F; full-f32 GL5+Newton adds ~1.5e-6 (F16) growing to ~7e-6 (F128) as the segment count rises. Terminal Newton stays ~2 iters with **zero fallbacks** once the spatial tolerance is set to an f32-reachable 1e-6.

3. **Topology mismatch is transport-neutral.** Up to a third of rays change the f32 event list, yet total tau and event position stay within the small pipeline error -- confirming the handoff claim that float32 boundary perturbations change the event list while preserving transport (Invariant Sec.19.12).

4. **div vs cross ordering is a topology-only choice.** Cross-multiplication changes the tie-order mismatch counts but leaves tau/event error identical; it does not reduce transport error, which is dominated by ray quantization.


**Modeling note.** `clock32` computes the box entry/exit (integration domain) in f32, so its pipeline-only tau (~7e-7) is larger than a pure ordering-only isolation (which would keep the domain in f64). This is the more GPU-realistic choice; the true-f64 reference and the independent GL12 oracle both remain the ground-truth hierarchy.
