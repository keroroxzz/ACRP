# A3 analytic transport vs delta/ratio tracking — where does A3 actually win?

Decisive 1D benchmark (cubic-fade lattice extinction, fixed optical thickness, contrast tuned
by peaky corners). Cost metric: extinction (medium) evaluations. Code:
`a3/a3/research/freeflight_benchmark.py`; data `a3/results/freeflight_benchmark.json`; plot
`a3/results/freeflight_benchmark.png`. All methods verified UNBIASED (mean free-flight T agrees
to MC error across A3 / delta-global / delta-local).

## Result 1 (refutes the naive claim): free-flight SAMPLING — A3 does NOT win

| contrast | A3 | delta global | delta local (cached grid) | delta local (procedural, +majorants) |
|---|---|---|---|---|
| 2.3 | 16.3 | 2.1 | **1.3** | 16.5 |
| 6.9 | 18.5 | 7.4 | **1.8** | 19.2 |
| 18.1 | 30.4 | 32.9 | **1.9** | 31.2 |

- A3 free-flight cost is **path-sensitive** (~cells to collision), NOT contrast-independent-cheap.
- **delta_local with a precomputed majorant grid wins ~10x** (output-sensitive: ~2 evals to
  collision regardless of contrast). This is what production voxel clouds do.
- Only delta_global (loose majorant) rises with contrast and eventually loses to A3.
- In a PROCEDURAL setting (no cache; the per-cell majorant must be read from the field on the
  fly), delta_local costs ~cells too, so it TIES A3.

Conclusion: for stochastic collision sampling, A3 is not a speed win against tracking with a
cached majorant. The "contrast-independent -> far exceed" hypothesis is false for free flight.

## Result 2 (the real win): TRANSMITTANCE / next-event estimation — A3 wins

Transmittance T = exp(-tau) along a shadow / NEE ray. A3 computes it EXACTLY (one cell walk,
zero variance). Ratio tracking is unbiased but NOISY.

| contrast | A3 exact | ratio tracking: std/est | ratio evals for 1% RMSE |
|---|---|---|---|
| 2.3 | **65 ev, 0 error** | 0.097 | ~495 |
| 6.9 | **65 ev, 0 error** | 0.078 | ~446 |
| 18.1 | **65 ev, 0 error** | 0.076 | ~449 |

- A3 exact transmittance is **~7x cheaper** than ratio tracking for 1% RMSE (65 vs ~450-510
  evals), and it is **exact** (ratio tracking at 65 evals ~ one estimate has ~8% error).
- The advantage **grows with the accuracy demand**: ratio tracking scales as 1/eps^2 (0.1% RMSE
  ~ 48000 evals), while A3 stays fixed at 65.
- Shared work: A3's single cell walk yields BOTH the exact transmittance AND supports the
  free-flight inversion; tracking needs SEPARATE delta (collision) + ratio (transmittance) passes.

## Honest verdict on "can A3 far exceed AAA?"

A3's edge is **exactness / zero variance**, realized on **deterministic transport quantities**
(transmittance, next-event estimation) and in **procedural, high-accuracy, or loose-majorant**
regimes -- NOT on stochastic free-flight sampling, and NOT against cached voxel grids with tight
precomputed majorants.

Where a net path-tracer win is plausible:
- Heavy next-event estimation (every scatter shoots a shadow ray): exact A3 transmittance
  replaces noisy ratio tracking -> ~7x fewer evals at 1% and no shadow noise to denoise.
- High accuracy targets (film): A3 fixed vs tracking 1/eps^2.
- Procedural media without a giant precomputed majorant grid (A3's native setting).

Not a win:
- Real-time raster clouds with temporal amortization (production wins).
- Free-flight collision sampling with a cached majorant grid (tracking wins ~10x).

## Next experiments
1. Ratio tracking with a LOOSE (global) majorant: variance should rise with contrast -> A3's
   transmittance win should GROW with contrast (the loose-majorant regime).
2. Full volumetric path tracer with NEE: measure the NET speedup / noise reduction from exact
   A3 transmittance vs ratio-tracking NEE, at equal image quality.
3. 3D + multiscale: confirm the transmittance win survives with the real A3 kernel.
