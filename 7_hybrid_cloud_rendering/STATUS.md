# Research 7 Status

Last updated: 2026-07-25

## Active phase

H0 — strong production raymarch baseline.

## Initial decision

Bootstrap H0 from the live real-sky shader at `a3/gpu/sky_frag.glsl`. It already
provides:

- layered low/mid/high clouds plus a vertically developed cumulonimbus;
- a production raymarch with coarse presence skipping, fine cloud steps,
  transmittance early termination, jitter, short sun queries, and cheap
  multiple scattering;
- a clean, higher-sample reference configuration;
- workload instrumentation hooks.

The Research 7 harness records the shader SHA-256 so results remain attributable
while the live shader evolves. A self-contained snapshot will be made once the
H0 baseline configuration passes its stability and quality gates.

## H0 gates

- [x] Reproducible offscreen rendering on the RTX 4070.
- [x] Continuous 1.5 s GPU warmup before timing.
- [ ] Multiple stress views, including sparse, near/inside, overcast-control,
      multi-cell proxy, sunset proxy, and grazing.
- [x] GPU median, p95, min/max, and per-view results.
- [x] Quality error against a clean high-sample reference.
- [x] Raw result JSON and labeled comparison image.
- [ ] Stable baseline suitable for later H1–H3 comparisons.

## Measured results

### Initial inherited-baseline characterization

Command:

```text
python articles/7_hybrid_cloud_rendering/program/bench_h0.py
```

Environment: Python 3.7 OpenGL environment, NVIDIA GeForce RTX 4070,
1280x720, 21 timed frames per scene after 1.5 s continuous warmup per scene.
Live shader SHA-256:
`4c19c2771c52bacf94ec100ea2e911ede9f148b7c87216c1157c6a1777958813`.

| View | Median GPU ms | Per-view p95 ms | RMSE vs clean reference |
| --- | ---: | ---: | ---: |
| sparse_wide | 17.720 | 22.856 | 0.01107 |
| low_up | 20.127 | 22.359 | 0.01409 |
| near_tower | 18.919 | 22.877 | 0.01117 |
| inside_proxy | 2.030 | 2.418 | 0.01161 |
| high_overcast_proxy | 18.334 | 22.114 | 0.00932 |
| grazing | 20.912 | 21.308 | 0.01333 |

Raw outputs:

- `results/h0/baseline_1280x720.json`
- `results/h0/baseline_1280x720.png`

These numbers characterize the inherited shader; they do **not** pass H0 yet.
Visual inspection found:

- `inside_proxy` is not a valid camera-inside-cloud stress case despite its
  name; its anomalously low cost must not be used as an inside-cloud result.
- `high_overcast_proxy` renders nearly blank and is not a valid full-screen
  overcast control.
- The scene lacks the required multi-cell storm, sunset-lighting, and temporal
  sequences.
- The inherited density has visibly simple/blocky structures. It is useful for
  transport benchmarking but is not yet the roadmap's production-quality
  visual baseline.

Next H0 work is therefore to create Research 7-owned stress-scene controls and
upgrade the baseline morphology while retaining the measured configuration as
an immutable provenance point.

### Cumulonimbus morphology correction

The inherited cumulonimbus was analytically a cylinder plus a rotationally
symmetric flat disk, and visually read as a giant mushroom. That construction
has been replaced in the live Research 7 source path with:

- multiple leaning thermal turrets whose dominance changes with height;
- widening and merging cauliflower lobes rather than a single stalk;
- high-frequency boundary erosion;
- a one-sided, wind-sheared anvil with a compact upwind side;
- porous anvil texture and upper turrets protruding through the shoulder;
- a separate conservative support bound broad enough not to amputate density.

The first correction still produced a smooth, dominant cap and was rejected
after visual inspection. The second correction removes that cap and produces a
credible developing convective tower in the low-up and near-tower views.

Preview command:

```text
python articles/7_hybrid_cloud_rendering/program/bench_h0.py \
  --width 640 --height 360 --frames 3
```

Accepted preview shader SHA-256:
`b948f8fa260dbe9cb2899425e910df984e74a420e083a5fe5432adcbebfd09d7`.

Preview output: `results/h0/baseline_640x360.png`.

The current shape represents a developing/towering cumulonimbus. A mature
storm's broad anvil should become an explicit scene parameter or morphology
variant, not an unavoidable disk attached to every tower. The wider
conservative support also raises traversal cost; H1 must recover that cost with
a tighter hierarchy without compromising support safety.

### Multiscale detail correction

The corrected macro shape still used essentially one isotropic FBM family and
looked too uniformly soft. It now uses scale-separated, decorrelated fields:

- low-frequency vector coordinate distortion for large billow displacement;
- a second, smaller distortion for detail decorrelation;
- medium-scale normalized billow FBM for convex cauliflower cells;
- finer ridged FBM for inter-cell erosion;
- a micro-billow layer restricted primarily to optically thin boundaries;
- separate horizontally anisotropic detail coordinates for the anvil.

Visual inspection of `results/h0/baseline_640x360.png` confirms substantially
more realistic cauliflower structure in the `low_up` and `near_tower` views.
The corresponding shader SHA-256 is
`93cba008c549f763ffb5cc386d89da6d368c884062617846ca571828464a074f`.

This is deliberately not recorded as a performance win. The naive implementation
evaluates all detail bands at every full-density query and costs approximately
25–33 ms at only 640x360 in the preview views. This result makes the H1/H2
research target concrete: preserve this full-detail density as the quality
reference, then use support classification, octave pruning, and boundary-only
fine evaluation to avoid paying for every scale throughout the volume.

### Paper 6 shader transfer study

The density and lighting paths in `cumulus_beauty.glsl` and
`cumulus_multilayer.glsl` were compared directly with the H0 shader.

Accepted design transfers:

- successive remap/erosion is preferable to one additive FBM sum;
- low water clouds, granular mid-level clouds, and fibrous ice clouds require
  different noise anisotropy and vertical profiles;
- wind shear should be applied before detail sampling;
- fine erosion should weaken near rounded cauliflower tops;
- sun shadows should use the same macro extinction but may omit expensive fine
  erosion and fibres;
- each altitude deck needs its own slab intersection, sampling rate, optical
  thickness, and water/ice lighting model.

A direct density-only transplant was tested and rejected. In the old single
0.5–8 km slab marcher it produced severe horizontal streaking and repeated
layers because one fixed world-space step was being used for all deck
thicknesses and frequencies. The stable background density was restored.

The coarse `shadowDensity` transfer was retained and the shader smoke gate
passes with SHA-256
`cd2189f99f9eda42fb2adfabb353b84a988bdd3c27f8df6ee6962d91b0a8d995`.

Required H0 architecture change: build a Research 7-owned deck compositor with
separate low/mid/high interval marches, carrying front-to-back transmittance.
Only then should the richer Paper 6 deck densities be enabled. The central
cumulonimbus remains the full multiscale quality reference.

### Single-photo ViT GA morphology fit

Research 7 now has a single-photo DINOv2 GA:
`program/optimize_h0_vit.py`.

Target:
`reference_photos/IMG_20200531_141120.jpg`.

Fixed geometry from Paper 6 `fixed_geo.json`:

- pitch: 0.5117994;
- sun azimuth: -1.3989905;
- sun elevation: 0.6178097;
- focal: 0.665.

The initial target had no curated mask. A Research 7-local top-band connected
skyline mask was generated. The first mask was rejected because its power-line
heuristic misclassified shadowed cloud edges and leaked the target silhouette
into the composite. The accepted `mask_v2` fills all sky-component holes and
retains only the land foreground.

The shader now exposes camera focal length, sun direction, cumulonimbus X/Z
position, lean, radii, warp scales, erosion strengths, anvil extent/density,
and crown strength as host-controlled uniforms. General H0 defaults remain
explicit in `bench_h0.py`; the photo-specific fit is stored separately.

An 8-generation, 24-candidate DINOv2 search improved fitness monotonically:

```text
0.80199 -> 0.80346 -> 0.80994 -> 0.81116
        -> 0.81354 -> 0.81527 -> 0.82139
```

Outputs:

- `results/h0_vit/vit_best.json`;
- `results/h0_vit/vit_vs_photo.png`;
- `results/h0_vit/vit_render.png`;
- `results/h0_vit/IMG_20200531_141120_mask_v2.png`.

The result is an initial fit, not a final appearance match. It successfully
moves and reshapes the cumulonimbus toward the reference, but the inherited
background decks, atmosphere, and lighting remain visibly less realistic than
Paper 6. The next fit should occur only after the separate-deck compositor and
Research 7-owned lighting path exist; otherwise the GA spends morphology
capacity compensating for unrelated renderer deficiencies.

#### Curated-mask refit

The user added the high-quality curated mask:
`reference_photos/IMG_20200531_141120_mask.png`.

It has replaced the derived Research 7 mask as the optimizer's preferred input.
The automatic mask remains only as a fallback. Visual inspection confirms that
the curated mask includes the complete sky and cloud field while excluding the
terrain and vegetation silhouette.

An 8-generation warm-start refit under the curated mask improved its new-mask
fitness from 0.81243 to 0.81681. This score is not directly comparable to the
previous 0.82139 because the evaluated sky region changed. `vit_best.json`
records the exact curated mask path, and `vit_vs_photo.png` contains the updated
comparison.

### Claude handoff audit and local procedural-cloud skill

The handoff was audited against the current shader and harnesses. Research 7
now owns a self-contained `program/sky_frag.glsl`; both the benchmark and ViT
optimizer use that snapshot. The cumulonimbus position is pinned for the
single-photo fit, its radius bounds were expanded, and the optimizer warm-start
now reconstructs normalized genes by parameter name so bound changes do not
silently corrupt prior results.

The user-provided, untracked `procedural-clouds-threejs` skill was reviewed as
reference material only and must not be committed. Useful transfers are:

- separated low-frequency Perlin/FBM shape and higher-frequency Worley-style
  erosion;
- dual-lobe Henyey-Greenstein water-cloud phase;
- darker vertical base lighting;
- precomputed 3-D noise and temporal low-resolution reconstruction as future
  real-time options.

The current shader adopted the dual HG phase and vertical base-light response.
A one-generation fixed-geometry warm-start retained fitness at 0.82003. This
small numerical change is not claimed as a morphology improvement.

Runtime evaluation of 27-neighbour Worley noise was rejected for H0 because the
existing full-detail reference is already too slow. At 320x180, the latest
smoke benchmark measures 20.861 ms (`sparse_wide`) and 27.750 ms (`low_up`).
A precomputed noise texture may be evaluated later as a production raymarch
variant or negative control, but it is not an A3-exact density path.

An explicit stack of overlapping ellipsoidal thermal bubbles was also tested.
It broke the column silhouette but exposed obvious spherical blobs and reduced
ViT fitness first to 0.80477 and then 0.79466 after support distortion. The
experiment was reverted. This is a useful negative result: the next morphology
iteration should use a continuous coverage/type field or precomputed 3-D
Perlin-Worley support, not visible analytic spheres.

The current 0.82003 render is still visibly column-like. Whole-sky DINO style
similarity is too insensitive to the target cumulonimbus outline, so future GA
work needs a cloud-region/silhouette term before more generations are spent.

### Mid/low-level composition study

The user judged the developing crown as plausible and identified the empty
middle and lower cloud field as the more important visual defect. The crown
topology was therefore held fixed while the lower composition was studied.

Accepted changes:

- low-cumulus coverage uses a modest storm-local inflow boost in the existing
  continuous weather field;
- two subtle feeder-cloud envelopes connect the tower to its lower environment;
- feeder detail is evaluated only inside its support and uses 3+2+2 noise
  octaves rather than an unconditional 4+3+3 evaluation;
- the low deck remains at 0.72--2.18 km with 0.24 coverage frequency, avoiding
  the former ceiling artifact.

Rejected ablations:

- increasing background extinction from 2.0 to 2.7 and 4.0 made distant cloud
  marks brighter but did not fix their scale; the strongest test scored
  0.81137 and was reverted;
- large analytic feeder lobes exposed spherical side bubbles and scored
  0.80929;
- a 1.3--4.5 km radial shoulder formed a horizontal ring and scored 0.81962;
- a single 0.12-frequency low weather field underwent a coverage phase change:
  low coverage removed the deck, while 0.35--0.42 formed a fog wall
  (0.81466 and 0.80178);
- a low-frequency-region times mid-frequency-cell coverage field avoided the
  wall but was not visible enough for its extra FBM cost and was removed;
- raising the low deck to 0.85--2.75 km moved sparse marks upward without
  building a useful cloud field and scored 0.81403.

The retained version scored 0.82326 in its first warm-start generation. An
8-generation, 24-candidate confirmation improved it monotonically:

```text
0.82326 -> 0.82423 -> 0.82677
```

The accepted result is stored in `results/h0_vit`. It preserves the crown,
adds small-scale mid-tower variation, and reduces the visual separation between
the base and feeder clouds. It remains a developing tower rather than a final
photorealistic match.

The optimized feeder version's smoke measurements at 320x180 were 22.287 ms
for `sparse_wide` and 28.826 ms for `low_up`. This is still above the earlier
20.861/27.750 ms measurements, so feeder support/detail classification remains
an H1 optimization target rather than a free H0 quality change.

The formal 1280x720 H0 run (shader SHA-256
`84d0d133a125afa9bcf97188e599b7f4ca0c4117202405b2ce913ccf6914b3fe`)
confirmed that this full-detail baseline is far outside the real-time budget:

| Scene | median ms | p95 ms | RMSE vs clean reference |
|---|---:|---:|---:|
| sparse_wide | 171.864 | 180.511 | 0.003828 |
| low_up | 293.779 | 296.906 | 0.005733 |
| near_tower | 163.358 | 169.817 | 0.006595 |
| inside_proxy | 87.314 | 89.361 | 0.008289 |
| high_overcast_proxy | 121.911 | 131.270 | 0.002028 |
| grazing | 137.265 | 150.092 | 0.002708 |

These are measured baseline costs, not hybrid results. They reinforce the
Roadmap decision to preserve H0 as a quality reference and move expensive
support/detail work into interval-local H1 processing.

### Wide-base and higher-coverage follow-up

After checkpoint commit `20e892f`, the user requested a storm base three to
four times wider and higher coverage in the surrounding low deck.

The current experiment uses:

- a noise-distorted base support with horizontal radii
  `3.7 * uCbBaseR` and `2.8 * uCbBaseR`;
- a conservative matching presence bound extending to
  `4.2 * uCbBaseR`;
- low-deck coverage increased from 0.39 to 0.46;
- background extinction increased from 2.0 to 2.8 and distance fade reduced
  from 0.060 to 0.050.

A smooth wide ellipse produced a rectangular slab and scored 0.81251. A hard
billow gate removed nearly the entire base. The accepted intermediate uses
continuous billow modulation with a nonzero floor, leaving a broad connected
base with broken edges and separate feeder cells.

Base edge/cell noise initially ran at every density query and raised smoke
cost to 25.773/33.112 ms. Restricting those seven octaves to the base height
and ellipse reduced smoke cost to:

```text
sparse_wide  23.656 ms
low_up       29.904 ms
```

An eight-generation warm-start with the wide base and higher low-cloud
coverage improved fitness from 0.81777 to 0.81982. This remains below the
narrower 0.82677 checkpoint because whole-sky DINO prefers less cloud, but the
wider composition is an explicit user-directed visual target rather than a
metric regression to be optimized away.

## Phase H1: active-interval extraction prototype

H1 now has a dedicated `program/bench_h1.py` harness. It produces:

- march/empty class maps;
- active-length heatmaps;
- full and total evaluation heatmaps;
- missed-support masks;
- H0/H1 RGB comparisons;
- raw JSON with interval, evaluation, timing, support, and RMSE metrics.

The original METHOD=2 implementation was rejected immediately. It performed
GL5 integration in every fine DDA cell and measured 53.3/74.3 ms at 320x180,
with 393/666 mixed procedural evaluations per ray and nonzero missed support.

Replacing per-cell GL5 with one-density short marching reduced the first
tight-support prototype to 10.9/12.4 ms. Splitting the counter established that
it reduced full density evaluations by 2.47x/2.51x. However, its point-sampled
macro support missed thin edges.

A three-class experiment then separated:

- tight support: 2.5x H0 short march, with a narrow boundary fallback;
- conservative-only support: coarse density confirmation;
- empty support: direct skip.

Inactive gaps up to 4 km are merged in interval-stream accounting while still
being skipped during integration. At 320x180 this working point measured:

```text
scene          median ms   density reduction   interval p95   missed pixels
sparse_wide       11.918          2.07x              4             79
low_up            13.553          2.30x              3            101
```

The full 1280x720 six-scene run is stored in
`results/h1/h1_1280x720.json`. Median times range from 36.451 ms
(`inside_proxy`) to 137.951 ms (`low_up`). Four scenes exceed 2x full-density
reduction; `grazing` reaches only 1.92x. `high_overcast_proxy` has interval
p95 5. Therefore the workload gates do not pass across the full suite.

More importantly, H0/H1 RGB RMSE ranges from 0.0046 to 0.0437. Visual
comparison shows that the widened H1-only procedural threshold renders large
false-positive cloud regions that H0's tight presence skips. This is not a
valid conservative bound: it changes the support definition rather than
bounding the existing density.

H1 is therefore **not complete**. The current prototype establishes a useful
performance upper bound and diagnostics, but fails:

- zero missed cloud support;
- full-suite 2x density reduction;
- full-suite p95 active intervals <= 4;
- acceptable H0/H1 visual agreement.

The next H1 implementation must attach a density upper bound and threshold
margin to macro cells (derived from the actual field), then dispatch
empty/boundary/interior from that metadata. Further tuning of ray probe spacing
or a second widened procedural support function is not considered a valid
path to the H1 gate.

### H1.1 sampled density-max hierarchy

H1.1 replaces the invalid widened procedural support with a GPU-built 3-D
texture derived from the actual `cloudDensity()` function.

Configuration:

- world domain: x `[-48,48]`, y `[0.5,8.0]`, z `[-32,96]` km;
- grid: 96 x 16 x 128 cells;
- cell size: 1.0 x 0.46875 x 1.0 km;
- build samples: 4 x 4 x 4 = 64 density samples per cell;
- R32F stores the sampled cell maximum;
- runtime lookup reads the current and ray-backward cell;
- cells with sampled maximum above 0.001 become active;
- gaps up to 4 km are merged in interval-stream accounting but remain skipped
  during integration.

Several validation errors were found and corrected:

- 3-D texture axes were initially sampled as x/y/z although storage layers are
  x/z/y;
- the original 0.02 presence cutoff incorrectly discarded low but cumulative
  density cells;
- the initial x/z domain excluded distant horizon rays;
- 3^3 cell sampling missed small density islands in sparse and grazing views;
- six-neighbour and full 3^3 runtime dilation achieved coverage but expanded
  the active volume enough to fail the density-reduction gate.

The accepted 4^3 build plus current/ray-backward lookup produces zero
macro-support misses across all six 1280x720 scenes. Formal 21-frame results:

| Scene | median ms | full-density reduction | interval p95 | support misses |
|---|---:|---:|---:|---:|
| sparse_wide | 71.093 | 2.04x | 4 | 0 |
| low_up | 62.492 | 5.08x | 4 | 0 |
| near_tower | 64.572 | 2.24x | 4 | 0 |
| inside_proxy | 25.274 | 3.51x | 1 | 0 |
| high_overcast_proxy | 43.931 | 2.61x | 4 | 0 |
| grazing | 38.339 | 3.64x | 3 | 0 |

Thus the H1 **traversal/extraction gates pass**:

- zero missed macro support;
- grazing stability;
- p95 active intervals <= 4;
- at least 2x full-density reduction in every formal scene.

The result is not yet production-ready:

- the 96x16x128, 64-sample hierarchy build costs 29.030 ms GPU time, versus
  the Roadmap target of 0.5--1.5 ms;
- the uniform 0.18 km interval march still has RGB RMSE approximately
  0.005--0.053 versus H0 production;
- macro extraction has zero misses, but interval sampling still misses thin
  alpha contributions;
- traversal alone remains above the final cloud budget at 1280x720.

The 29 ms build can be amortized for the current static benchmark, but animated
production clouds require incremental dirty-region updates, lower sample
counts, or a coarse-to-refined hierarchy. The next rendering task is to use
cell maximum/threshold margin to spend H0-sized steps only at boundaries while
using coarse integration or later B1/B2 closure in interiors.

## Phase H2: cell-level transmittance closure

The H1.1 macro texture now stores four statistics from the same 64 density
samples per cell:

- R: sampled maximum density;
- G: mean density;
- B: density variance;
- A: empirical `E[exp(-k rho)]` for a selected optical scale `k`.

`program/bench_h2.py` compares three cell-distribution closures against the A
channel reference: B0 uses maximum density, B1 uses mean density, and B2 adds
the second cumulant,
`exp(-k mean + 0.5 k^2 variance)`.
An additional offline statistics pass stores mean, variance, third cumulant,
and fourth cumulant so that B3/B4 truncations can be evaluated without
expanding the current runtime texture.

The deterministic classifier marks a cell as interior only when maximum
density exceeds 0.02, mean/maximum is at least 0.25, and normalized variance
does not exceed 2. The B2 analytic path additionally requires the second-order
cumulant risk `0.5 k^2 variance <= 0.1`; all other active cells fall back.

The current 96 x 16 x 128 experiment contains 23,133 active cells and 942
interior cells. On the cells accepted by the analytic risk gate:

| k | analytic cells | active coverage | B1 p95 rel. error | B2 p95 rel. error |
|---:|---:|---:|---:|---:|
| 0.5 | 942 | 4.07% | 2.863% | 0.159% |
| 1.0 | 849 | 3.67% | 7.699% | 0.832% |
| 2.0 | 242 | 1.05% | 8.259% | 1.185% |
| 4.0 | 17 | 0.07% | 7.634% | 1.120% |
| 8.0 | 0 | 0.00% | fallback | fallback |

For the same accepted cells, adding the third and fourth cumulants reduces p95
relative error further:

| k | B2 p95 | B3 p95 | B4 p95 |
|---:|---:|---:|---:|
| 0.5 | 0.159% | 0.0138% | 0.0022% |
| 1.0 | 0.832% | 0.102% | 0.0288% |
| 2.0 | 1.185% | 0.111% | 0.0436% |
| 4.0 | 1.120% | 0.0904% | 0.0561% |
| 8.0 | fallback | fallback | fallback |

This establishes two useful local facts:

- B1 mean-density closure is not accurate enough for these nonlinear cells;
- B2 is below 2% p95 wherever the conservative risk gate accepts it and is
  substantially better than B1 at every nonempty tested optical scale;
- κ3/κ4 are well behaved inside this same conservative gate and B4 is better
  than B2 at every nonempty scale in the cell-distribution experiment.

It also exposes the main limitation: analytic coverage collapses as optical
scale increases. At `k=8` every cell falls back, so the error gate is safe but
provides no acceleration. Boundary cells are always fallback by construction.

These measurements do **not** pass the H2 phase gate yet. The empirical target
is an expectation over 64 samples inside one cell, not transmittance integrated
along a ray or interval. Third and fourth cumulants require a separate
28.664 ms offline pass and are not retained by the current runtime texture.
Full-ray interior transmittance, boundary dispatch, and the cost/coverage
trade-off must be measured before promoting any cumulant closure into the
renderer.

### H2 full-camera-ray validation

`program/bench_h2_ray.py` now traces the six H0 camera views through the macro
grid and compares a 30 m midpoint reference against B1, B2, and B4. To isolate
transport closure from lighting, this first test uses the synthetic uniform
extinction `cloudDensity * SIGMA_T * DENS`; boundary and out-of-grid segments
automatically use the fine reference path.

This test rejects the apparently successful cell result as a practical
renderer path. A one-kilometre horizontal cell has an optical scale near
`7 * 2.4 * 1 = 16.8`, while the cell experiment already fell back completely
at `k=8`. With the existing `0.5 k^2 variance <= 0.1` risk gate, mean analytic
path-length coverage is:

| Scene | analytic path length |
|---|---:|
| sparse_wide | 0.01% |
| low_up | 0.01% |
| near_tower | <0.01% |
| inside_proxy | 0.11% |
| high_overcast_proxy | 0.01% |
| grazing | 0.01% |

B1/B2/B4 consequently show near-zero ray error only because essentially the
entire ray falls back to the 30 m reference march. This is a safe classifier,
but not an accelerator. The ray benchmark therefore includes a separate 10%
minimum analytic-length gate, which fails all scenes; error gates must not be
interpreted without this coverage gate.

The immediate H2 conclusion is that cumulants of density over the current
one-kilometre cells are too coarse for dense-cloud camera transport. The next
experiment should either construct statistics for much shorter path-aligned
subcells or preserve statistics of integrated optical depth rather than the
marginal density distribution. Simply relaxing the variance gate would turn a
coverage failure into uncontrolled high-optical-depth expansion error.

A physical segment-length sweep using the current interior variance estimates
quantifies the required scale:

| segment length | accepted interior cells |
|---:|---:|
| 30 m | 100.0% |
| 60 m | 89.8% |
| 125 m | 22.8% |
| 250 m | 1.7% |
| 500 m | 0.0% |
| 1 km | 0.0% |

This points to approximately 30--60 m transport segments for the dense-cloud
extinction used by H0. Such a flat 3-D grid would erase much of H1's memory and
build advantage, so the promising implementation direction is hierarchical:
retain kilometre cells for support exclusion, but create short path-aligned
statistics only inside accepted interior intervals.

### Rejected H3 whole-cell boundary refinement

An initial H3 dispatch experiment used the macro maximum, fill ratio, and
variance to switch boundary cells from the H1 180 m step back to H0's 60 m
step. The statistically conservative classifier marked nearly every active
cell as boundary: at 320 x 180, `sparse_wide` performed 1.38 times as many
density evaluations as H0 rather than reducing them.

Restricting refinement to threshold-margin cells (`max < 0.02` or
`mean/max < 0.03`) still produced only 1.20x evaluation reduction in
`sparse_wide`, versus the H1.1 extraction target of at least 2x. Its RGB RMSE
was 0.0528 and it increased alpha misses. `low_up` retained 3.16x reduction,
showing that the failure is view/workload dependent rather than a universal
step-size issue.

This shader behavior was rejected and reverted to the accepted uniform H1.1
short march. A kilometre cell may contain a thin boundary without its entire
ray chord requiring fine steps. H3 therefore needs entry/exit localization or
a short path-aligned subcell cache; classifying the whole macro cell as a fine
march interval is too coarse.

### H3 transition-local refinement experiment

`program/bench_h3.py` tests a narrower alternative. H1 continues marching at
180 m, but when the center density changes between empty and nonempty, only
that one segment is evaluated at three 60 m substeps. The feature is isolated
behind `H3_BOUNDARY_REFINE=1`; the default renderer remains H1.1.

The first implementation evaluated lighting independently at each substep. At
320 x 180 it increased mean density evaluations by only 9%, but raised GPU
time from 8.733 to 22.555 ms in `sparse_wide` and from 7.423 to 19.727 ms in
`low_up`. RMSE improved by only 0.0004 and 0.0003. The repeated sun marches,
not camera-density work, dominated the cost.

A second implementation integrated three extinction samples but reused one
center lighting query. It still measured 19.943/17.313 ms versus H1's
8.661/7.388 ms and produced no four-decimal RMSE improvement. Divergent extra
procedural density work at sparse transitions is itself expensive enough to
make this dispatch unattractive.

The experiment is therefore rejected as an H3 rendering path. It establishes
that boundary refinement must not invoke full procedural density octaves on
demand. A useful subcell cache must contain cheap precomputed extinction or a
low-octave boundary locator; otherwise H1's saved camera evaluations are spent
again exactly at the most divergent pixels.

### H3.1 presence-guided, budgeted entry refinement

A cheaper locator uses two stratified `presence()` probes only when the center
density first changes from empty to nonempty. It estimates the occupied
fraction of that 180 m segment, reuses the center density when possible, and
performs one lighting query. A hard budget permits at most two refined entries
per camera ray.

The 1280 x 720, 21-frame six-view benchmark measured:

| Scene | H1 ms | H3.1 ms | overhead | H1 RMSE | H3.1 RMSE | improvement |
|---|---:|---:|---:|---:|---:|---:|
| sparse_wide | 68.725 | 77.007 | 12.1% | 0.0528 | 0.0503 | 4.7% |
| low_up | 57.069 | 67.028 | 17.5% | 0.0388 | 0.0363 | 6.5% |
| near_tower | 60.916 | 71.625 | 17.6% | 0.0341 | 0.0323 | 5.5% |
| inside_proxy | 23.426 | 27.746 | 18.4% | 0.0054 | 0.0054 | 0.0% |
| high_overcast_proxy | 43.417 | 48.501 | 11.7% | 0.0273 | 0.0250 | 8.4% |
| grazing | 38.201 | 45.552 | 19.2% | 0.0434 | 0.0415 | 4.4% |

The hard budget keeps GPU overhead below 20% across the suite, adds
effectively no full-density evaluation count at image scale, improves five
views, and does not regress `inside_proxy`. The inside view has no initial
empty-to-dense transition available to refine.

H3.1 is retained as a bounded prototype, not declared the H3 phase result.
The absolute RMSE remains too high for the Roadmap's production-quality gate,
and H3.1 spends 12--19% GPU time for a modest edge improvement. It does,
however, validate the required dispatch shape: cheap locator, one shared
lighting evaluation, and a strict per-ray refinement budget.

## H0-v2: CloudWorks-derived baseline restructuring

The user identified `articles/0_cloudworks/shader.glsl` as the strongest prior
hand-optimized renderer. Its central performance feature is a strict
macro/detail split: `Chunk` establishes support before distorted detail and
shadow work are evaluated. Research 7's H0 violated this principle for the
cumulonimbus, evaluating tower, anvil, and crown FBM stacks even when their
analytic shape multiplier was zero.

H0-v2 gates the existing detail stacks by their existing macro support. This
does not reduce octaves, change steps, or alter any nonzero density equation.
`program/bench_h0_cloudworks.py` compiles the committed `82f569d` shader and the
working H0-v2 source in the same process. Formal 1280 x 720 results:

| Scene | old H0 ms | H0-v2 ms | speedup | RGB RMSE |
|---|---:|---:|---:|---:|
| sparse_wide | 152.189 | 44.873 | 3.39x | 0.0 |
| low_up | 285.260 | 79.946 | 3.57x | 0.0 |
| near_tower | 169.989 | 51.710 | 3.29x | 0.0 |
| inside_proxy | 88.439 | 41.391 | 2.14x | 0.0 |
| high_overcast_proxy | 124.638 | 36.401 | 3.42x | 0.0 |
| grazing | 140.177 | 39.090 | 3.59x | 0.0 |

This is the first H0-v2 baseline candidate and invalidates performance ratios
that use the pre-dispatch H0 as denominator. H1/H2/H3 ideas remain informative,
but all claimed renderer speedups must be rerun against H0-v2.

A second CloudWorks feature, stopping cloud shadow marches after camera
optical depth becomes large, was tested at transmittance thresholds 0.10,
0.20, and 0.35. It produced only small, inconsistent gains and occasional
regressions from divergent branching, with nonzero image error. It remains
disabled. Paper 3 atmospheric scattering is the next independent H0-v2 slice.
