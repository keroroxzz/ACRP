# CloudWorks baseline study

Source studied: `articles/0_cloudworks/shader.glsl` (2021-05-30).

## Why it allocates work better than the current H0

CloudWorks separates each deck into a cheap macro `Chunk` field and conditional
detail. Two differently scaled noises create coverage and large cloud bodies;
distorted detail is evaluated afterward. The height profile supplies both a
soft density threshold and vertical solidness, instead of multiplying a generic
FBM by a slab mask.

Its camera marcher is adaptive in four useful ways:

1. step length follows the density-field slope;
2. step length grows with distance;
3. trapezoidal density integration reduces bias at changing steps;
4. shadow marching stops after cumulative optical depth reaches 2.3.

The shadow path is also cheaper than the camera path. Low density immediately
uses an approximation, while nontrivial shadow samples use macro chunk plus one
detail octave rather than the complete camera density. At most ten shadow steps
are taken.

Cloud layers are composited front-to-back as independent profiles. Camera
altitude changes their evaluation order, which improves early termination.
Atmospheric scattering is applied to the representative cloud depth after the
cloud integral instead of inside every cloud sample.

## Parts not to copy unchanged

- The original hash uses `sin`, which is fast but has weak cross-platform
  reproducibility and visible lattice risks.
- Several values are artistic rather than energy conserving; its `MiePhase`
  and light scaling are not normalized physical phase functions.
- Density and step parameters are expressed in GTA-world metres and depend on
  the original camera convention.
- The atmosphere predates the validated Paper 3 Chapman and twilight hybrid.
- Dynamic stepping is not conservative and needs a clean-reference quality
  gate before becoming the new H0.

## Research 7 migration order

1. Optical-depth shadow budget, without changing density or silhouette.
2. CloudWorks macro/detail split for a new deck profile implementation.
3. Slope- and distance-adaptive camera stepping with trapezoidal extinction.
4. Altitude-ordered independent deck compositing.
5. Paper 3 Chapman solar transmittance and sky airlight, evaluated outside the
   inner cloud-density loop.

Every slice remains behind a compile-time define until it passes both timing
and clean-reference image gates.

## Exact macro/detail dispatch result

The first structural migration gates the existing tower, feeder, anvil, and
crown detail stacks by their already-computed analytic macro support. It does
not change any nonzero density formula, octave count, camera step, or lighting
parameter.

An in-process comparison compiles the shader from commit `82f569d` beside the
new source. Across all six 1280 x 720 views, RGB RMSE is exactly zero in the
float framebuffer. Median speedups are:

| Scene | `82f569d` H0 | macro/detail H0-v2 | speedup |
|---|---:|---:|---:|
| sparse_wide | 152.189 ms | 44.873 ms | 3.39x |
| low_up | 285.260 ms | 79.946 ms | 3.57x |
| near_tower | 169.989 ms | 51.710 ms | 3.29x |
| inside_proxy | 88.439 ms | 41.391 ms | 2.14x |
| high_overcast_proxy | 124.638 ms | 36.401 ms | 3.42x |
| grazing | 140.177 ms | 39.090 ms | 3.59x |

This becomes the H0-v2 baseline candidate. It also changes the interpretation
of earlier H1 speedups: the former H0 paid detail costs outside their support,
so H1 must be rebenchmarked against H0-v2 before any hybrid speedup claim.

The separate optical-depth shadow cutoff is not accepted. Threshold sweeps
produce small and inconsistent timing changes, including regressions from
divergent branching, while introducing nonzero image error. It remains off.
