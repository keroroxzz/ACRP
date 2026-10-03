# CloudWorks "Next" — status

Adaptive band-limited ray marching for the CloudWorks cloud shader.

| file | what it is |
|---|---|
| `shader.glsl` | 2021 CloudWorks, pre-HSE. Archive snapshot — different camera convention and cloud time, **not** a comparable baseline. |
| `shader_HSE.glsl` | + Large-Scale Hashed Spatiotemporal Low-Cloud Top. The baseline everything is measured against. |
| `shader_HSE_Next.glsl` | the adaptive marcher. Every change is behind a `NEXT_*` define and can be switched off. |
| `shader_HSE_PSMS_Next.glsl` | **generated** — the Next marcher + Paper 3's atmosphere. Do not hand-edit; run `program/build_psms.py`. |
| `program/` | reproduction harness (`cw_gl.py`, `hse_field.py`, `scene_gate.py`, `bench_2k.py`, `bench_ab.py`) and the PSMS generator (`build_psms.py`, `psms_atmosphere.glsl`) |
| `results/` | gate contact sheets + `perf_2k.json` ledger |

`NubisEvolved-NoVideos.pdf` (SIGGRAPH 2022 course, third-party) is gitignored — keep it locally.

## Where it stands

2560×1440, six configurations, versus `shader_HSE.glsl`: **3.7–6.3× faster**, and
**6–37× closer** to a converged reference. All six gate configurations pass.

Perf ledger (`results/perf_2k.json`), each entry an interleaved A/B against the
pinned `be143db` baseline in the same run:

| commit | change | 2K total | cumulative |
|---|---|---|---|
| `be143db` | baseline | 620 ms | — |
| `3094da3` | 2-segment light-ray roof model | 393 ms | −32% |
| `5be8540` | jitter 0.10 + light-march stride 2 | 313 ms | −46% |
| `51154e1` | jitter off, coarse light march 25→20, roof span 240 | 271 ms | −54% |

## The core idea

`DensityMix()` is a *clamped* remap: the rendered density is exactly 0 below the
erosion band and exactly its ceiling above it. Only inside the band does the
noise carry visible signal. So the quantity that should drive the step length is
not density but **distance from that band**, measured in band widths:

```
q    = (df - bandCentre)/bandWidth      // ±0.5 at the band edges
outQ = |q| - 0.5                        // > 0 ⇒ the field is clamped
h    = safety · outQ / lipschitz(dq/ds)
```

with the Lipschitz bound taken from the free backward difference along the ray.
That is sphere tracing on the *rendered* field, so empty space **and saturated
cloud cores** both get long steps — the old rule ("denser ⇒ shorter") only
handled the empty side and burned steps inside solid bodies. Using `q` rather
than `df` also absorbs the band's own motion as the vertical profile changes
with height.

The surface step budget is the safety net that makes an aggressive planner safe:
the step that lands inside a cloud is walked back in fixed Nyquist substeps, the
substep densities are kept in a small array and replayed forward. At
`NEXT_REACH_SAFETY 3.0` it cuts error by ~39%.

## Three step-count couplings that had to go first

None of this can be validated until the marcher is invariant to step length, and
it was coupled in three separate places:

1. **Opacity** — optical depth was weighted by the constant `march.x` instead of
   the real step length, so it tracked the step *count*, not distance travelled.
2. **Brightness** — in-scatter attenuated each segment by the transmittance
   *after* it, and used `τ` rather than `1-exp(-τ)`. Halving the step brightened
   the clouds. Nubis Evolved p.61 uses the pre-update transmittance.
3. **Depth** — `d.y = d.y*(1-T) + T*d.x` is an exponential moving average, so the
   value feeding the atmospheric blend depended on how many times it ran. Now an
   opacity-weighted mean (Nubis Evolved p.187). This was the single largest
   source of non-convergence in the top-down views.

`NEXT_ENERGY_CONSERVING` and `NEXT_WEIGHTED_DEPTH` make the shader darker and
higher-contrast than the 2021 original. That is a correction toward the
converged reference, not a style change; both are switchable.

## Measured and rejected

Kept in the source so they are not tried again.

- **Per-cell HSE event cache.** Bit-exact, removes most of the arithmetic — and
  **19% slower**. 45 extra VGPRs cost more occupancy than the instructions
  saved. Same lesson as A3's per-cell adaptive branching.
- **Importance-culling light-march refreshes** (`NEXT_LIGHT_CULL`). Neutral to
  slower; the branch costs more than the marches it skips. Left at 0 because it
  bounds the error of a more aggressive stride.
- **Light-march stride 3.** Gate ratio 1.14, fails.
- **Trading the roof-model win back for view-march accuracy.** Does not help —
  the residual is the roof model, not the planner.

## Where the remaining error is

`B-g` (ground looking up at cumulonimbus) is the configuration that binds the
gate. **60% of its squared error sits in one 50-pixel band above the horizon** —
the far field, where the profile's own `march.z = 8` lets the step reach 480 m
against the reference's uniform 2.5 m. That is an authored LOD parameter, not a
planner defect, and it is why several later changes moved the gate ratio not at
all: they do not touch the far field.

## Measurement gotchas

These cost real time to find; do not re-derive them.

- **Cross-run absolute times are not comparable.** The same shader measured
  536 ms and 620 ms in different sessions. Only the ratio inside a single
  *interleaved* run means anything — `bench_2k.py` interleaves per configuration
  and records the gain against a pinned baseline shader. Self-A/B of the
  baseline reports −0.5%, which is the method's noise floor.
- **The reference must reach `a.fade` (40 km).** A reference that stops at 25 km
  is comparing different clouds; this showed up as ~5% "error" concentrated in
  the level and top-down views.
- **The reference must pin the lighting model.** `ShadowMarching()`'s
  low-density early-out is scaled by `march.x`, and the light-march step counts
  are shipping defaults that get tuned — if the reference inherits either, the
  gate measures against a target that drifts with the thing under test.
- **A converged reference is not free of error.** `|ref@5m − ref@2.5m|` is
  0.0005–0.0124 depending on the view. An absolute gate below that is measuring
  the reference's own noise, so the gate is `max(0.005, 2 × floor)`.
- **GPU clocks:** check `nvidia-smi --query-gpu=clocks_event_reasons.active`
  under load before saying "thermal". This GPU held 2805 MHz / 52 °C with no
  throttle flags throughout.

## Can this reach the industry's ~2 ms? — measured budget

Nubis quotes 1.3–4.2 ms for Skies and ≤2–3 ms for the VFX superstorms, **on PS5
at 1920×1080** (PS4: 960×540). Our numbers at the same resolutions,
`shader_HSE_PSMS_Next_2`, median of three representative configs:

| resolution | | ms/frame |
|---|---|---|
| 2560×1440 | our benchmark | 89.1 |
| 1920×1080 | Nubis PS5 max | **54.1** |
| 1280×720 | | 26.5 |
| 960×540 | Nubis PS4 max | 16.4 |

So the honest gap is **54 ms vs 2 ms at 1080p — 27×**. The RTX 4070 is not the
excuse; a PS5 GPU is slower. The difference is three techniques we do not have,
none of which is shader micro-optimisation:

| lever | factor | evidence | where the work is |
|---|---|---|---|
| noise from a **texture**, not `frac(sin())` | **1.6×** measured, likely 2–2.5× with a real 3D texture | swapping the sin hash for an ALU-only one: 54.2 → 34.2 ms @1080p | shader + a bake pass; **changes the look, needs a refit** |
| **low-resolution** cloud buffer + upscale | **3.3×** | 54.1 → 16.4 ms at 960×540, which is exactly what Nubis PS4 ships with a 2 px blur | engine: separate render target |
| **temporal reprojection** | 4–8× typical | Nubis p.155–157; cannot be measured here | engine: history buffer + motion vectors |

`hash()` is `frac(sin(n/1873)*seed)` — 8 sins per `noise3d`, ~5 `noise3d` per
density sample, so roughly 40 sin instructions per sample against one filtered
texture fetch for Nubis. That is the single biggest *shader-side* lever left.

Multiplying the two measurable levers: 54.1 / (1.6 × 3.3) ≈ **10 ms**. Temporal
reprojection at 4–8× closes the rest. So 2 ms is reachable, but it is an
**engine-integration project, not a shader project** — and two of the three
levers need RenderHook to provide a history buffer and a separate render target.

Already-spent lever, for context: Nubis' NVDF cone-step empty-space skip is
worth 4.2 → 1.3 ms in their own slides. We have an analytic equivalent (the
band-limited planner + cone-step roof skip), which is where the 3.76× over naive
HSE came from — so that one is largely banked already.

Free-ish items worth doing regardless: the three ambient probes are
view-independent and recomputed per pixel (~3%); they belong in a 1×1 pre-pass
the moment there is one, exactly as Paper 6 does it.

## Still missing versus Nubis

Ordered by expected value.

1. **Multiple-scattering approximation** (p.54) — CloudWorks has single scatter
   only, so cloud interiors read flat and dark.
2. **Dual-lobe phase function** — `MiePhase` is forward-only, no back lobe, so
   no silver lining.
3. **Ambient term** `pow(1 - dimensional_profile, 0.5) · height_fraction`
   (p.59/113) — currently a flat constant, so shadow sides have no depth.
4. **Height-fraction noise-type blend**, wispy at the base → billowy at the top
   (p.99).
5. **`pow(…, 0.27)` density shaping** (p.105) — the knob that makes Nubis clouds
   read as solid.
6. **Temporal reprojection + low resolution + blur** (p.155–157, p.183). Biggest
   remaining perf item, but it needs a history buffer in RenderHook.
7. **Long-distance shadow map** for the far field (p.112). `ShadowOnGround` is
   already half of it.

## LOD on the light march — implemented, measured, shipped off

Both axes are in `shader_HSE_Next.glsl` behind `NEXT_LIGHT_LOD` (default 0):
distance (`NEXT_LIGHT_LOD_MAXLOD`) and accumulated opacity
(`NEXT_LIGHT_LOD_THI/TLO`), driving both the step counts and the fbm octaves.

**It does not pay here.** At any setting that holds the gate it is worth 2.7% at
most, barely above the benchmark's ~1% noise floor, and it consumes nearly all
the remaining gate margin to get it.

| setting | gate | gain |
|---|---|---|
| distance only, `MAXLOD 12`, floors 4/12 | 0.984 | −1.4% |
| opacity only, `T 0.45→0.25`, floors 4/14 | 0.989 | −1.6% |
| opacity only, `T 0.40→0.24`, floors 3/10 | 0.997 | −2.7% |
| steps + octaves, aggressive | 1.95 | −12% (fails) |

Three things were learned getting there, all worth keeping:

- **The light march is already adaptive.** It breaks as soon as `d` passes
  `2/shadow.w/shadow.x`, which in dense cloud is about two steps. The full 6+20
  steps only run where the path is *thin* — exactly where the early-out has
  already decided they are needed. LOD is trying to cut what is not being spent.
- **Fewer steps must cover the same distance.** The first implementation
  truncated the light path; that alone is a +0.23% shift in mean luminance,
  already half the gate budget. `kFine`/`kCoarse` stretch the steps and weight
  each by the stretch instead. Fixing this moved the gate ratio from 1.99 → 1.20.
- **Octave LOD is worse than step LOD** (1.95 vs 1.20 at equal aggressiveness).
  Replacing an octave by its mean is unbiased in *density*, but the density then
  goes through `DensityMix()`, a clamped nonlinear remap, so the mean does not
  survive it.
- The opacity axis must key on the **transmittance in front of the sample**, not
  on the sample's own radiance weight. Individually thin samples contribute
  little but collectively carry the look; keying on per-sample weight degrades
  all of them at once, so the errors add instead of cancelling.

Turn it back on if the early-out threshold is raised or the quality target is
relaxed.

## Paper 3 atmosphere — `shader_HSE_PSMS_Next.glsl`

The hand-rolled two-term Rayleigh/Mie sky replaced by Paper 3's pseudo-spherical
multiple-scattering model, with the clouds wired into it. **Baseline: correctness
first, performance not considered** — it costs ~2× the Next marcher (4.5→10.2 ms
at 800×450 for A-g). Gate contact sheet: `results/psms_vs_next.png`.

- **Sun colour on a cloud** = Paper 3 solar transmittance at the cloud's own
  altitude and slant path, so cloud light reddens with the sun as the sky does.
- **Ambient on a cloud** = the real sky radiance it can see (zenith/horizon
  probes from the same integral, MS included) instead of a hand-keyed palette.
  `CloudColorVarying`'s day/sunset constants are gone.
- **Aerial perspective** = `PSMSIntegrate()` over the eye→cloud segment, which
  returns the path transmittance *and* the airlight scattered into it. As the
  cloud recedes, transmittance → 0 and airlight → the sky radiance for that
  direction, so cloud and sky converge on their own. The same function with
  `tMax = ∞` **is** the sky, so the two cannot disagree — no distance-fog lerp.

### Regenerate, don't fork

`shader_HSE_PSMS_Next.glsl` contains a copy of the marcher, so it would silently
rot as `shader_HSE_Next.glsl` advances. It is therefore **generated**: run
`program/build_psms.py` after any marcher change. Every substitution in that
script asserts, so a shape change in the base fails loudly rather than producing
a half-patched shader.

### Four things that had to be fixed to make it work here

1. **Ozone.** Paper 3's shader uses `ozA = 0.30`; Paper 6 independently found
   that turns clouds purple. Confirmed — scene B's cumulonimbus rendered frankly
   magenta, because `betaO` absorbs green ~2.4× harder than red. The model's
   vertical column is `ozA·ozW`, so 0.30 gives a green Chappuis optical depth of
   0.075 against a real atmosphere's ~0.025. Now `0.10`, which reproduces the
   real column.
2. **Planet occlusion at the surface.** CloudWorks puts the eye exactly *on*
   `planetRadius`, where `raySphere`'s `c = dot(ro,ro) − R²` is zero to within
   float noise. The strict test (`pl.x > tS`) let downward tangent rays march
   underground until `exp(−h/H)` overflowed — NaN across the lower frame; the
   loose test let float noise turn *rising* rays into spurious grazing hits —
   isolated black specks (118 of them). Now decided geometrically: from the
   surface only a downward ray is occluded, and immediately.
3. **Ambient is a dome, not a direction.** Mixing zenith↔horizon by height in
   the deck conflates "which way is the sky brighter" with "how much sky can
   this sample see". It left thin distant wisps scattering dim zenith light
   while blocking bright horizon sky — dark speckle. Height now controls only
   the dome's *visibility*; the dome average itself is horizon-heavy.
4. **Exposure.** Paper 3's scale is written against its own `1−exp(−L)` tonemap
   with a low sun; at daylight elevations it saturates. One `PSMS_EXPOSURE`
   constant is applied to the *light*, so sky, probes, aerial perspective and
   cloud sunlight all move together and cannot drift apart.

Deviations from the paper, both switchable: a grey aerosol layer
(`PSMS_AEROSOL`, on — without a boundary-layer haze the horizon stays
implausibly clear from ground level) and the ozone value above. The ground below
the horizon is black: this is an atmosphere-only model with no ground albedo,
and the target engine draws its own terrain there.

### Optimised: `shader_HSE_PSMS_Next_2.glsl` — 2K −34%

Generated alongside the baseline by `build_psms.py`; the baseline keeps every
switch off and stays the quality target. Ledger: `results/perf_2k_psms.json`.

| | 2K total, 12 configs | |
|---|---|---|
| `shader_HSE_PSMS_Next` (baseline) | 1518.7 ms | — |
| `shader_HSE_PSMS_Next_2` | 997.0 ms | **−34.4%** |

One change did all of it. The diffuse gather is 6 Chapman beams and it ran at
every march step of every integral — the sky (40), the eye→cloud segment (24)
and three ambient probes (12 each), ~600 Chapman evaluations per pixel, 40% of
the shader. But J is the *diffuse* source: smooth by construction, which is the
whole reason it exists. `PSMS_GATHER_LERP 12` samples it every 12 steps and
interpolates; with it amortised the aerial segment no longer needs 24 steps.

Error came out **lower** than the baseline on 11 of 12 configurations
(0.94–1.00×) — interpolating a smooth quantity also removes coarse-step noise.

### The wall, and the measurement that proves it

After that one change the atmosphere is no longer the cost. Short-circuiting
every `PSMSIntegrate` before its march loop:

> **atmosphere integrals 13.6% of the frame, cloud marcher 86.4%**

So deleting the atmosphere outright would only take the total to ~43%. And the
marcher is already at its own gate-bound limit from the Next line — every knob
retested here (`NEXT_SHADOW_STRIDE 3`, light-march LOD at two settings,
`NEXT_REACH_SAFETY 2.5`, surface budget off) either fails the gate or is
*slower*, the same verdict as before.

A hypothesis worth recording as **not supported**: physically-correct aerial
perspective washes out the far field, and far-field error is exactly what bound
the Next gate — so the far field ought to tolerate a coarser march here. It does
not. The same knobs fail for the same reasons.

Further squeezing of the remaining 13.6% (`GATHER_LERP` 20/40, `AERIAL_STEPS` 8,
`PROBE_STEPS` 8) all measured *slower* than round 1 or failed the gate.

### Sun matrix — `psms_gate.py`

Twelve configurations: the six cameras at the default sun, plus a sun matrix
shot from scene B at ground level — elevation either side of Paper 3's
day/twilight switch (solar zenith 88° = +2° elevation) crossed with back-lit /
side-lit / front-lit.

It earned its keep immediately. `PSMS_SUNTRANS_LERP 4` fails at 2.93× and fails
**specifically on the two side-lit configurations** — a midday-only test set
would have accepted it.

Post-sunset elevation is **−1°, not −4°**. At −4° the frames came back nearly
black and their RMSE collapsed to 0.002, because CloudWorks' own `LightSource`
fades the sun out over z ∈ [−0.075, −0.025] — the configuration would have been
exercising the twilight branch on an image with nothing in it. At −1° the sun is
below the horizon (Paper 3 on its twilight branch) while CloudWorks still gives
full light, and the frames render the purple Chappuis twilight properly.

The gate is **relative**: the reference is finer than the baseline in both the
marcher and the atmosphere, so the baseline already sits 0.012–0.020 from it by
construction. That distance is the shipping shader's accepted quality, not a
budget to spend, so no metric may exceed `max(0.005, 1.15 × the baseline's own
distance)`.

### Not gate-checked

`scene_gate.py` compares against references built from the *Next* atmosphere, so
it does not apply to PSMS as-is — the sky is a different image.
`get_reference()` now takes a `tag_prefix` so a PSMS gate gets its own cache
namespace rather than silently reusing the old sky's ground truth. The marcher
itself is unchanged, so the existing gate and perf ledger still describe it.

### Where to go instead

The remaining light-march cost is the marches that *do* run to completion, in
thin regions. Making each step cheaper fails on the remap nonlinearity above, so
the way forward is to avoid the march: Nubis' **long-distance shadow map**
(p.112), a cheap 2D lookup for the far field. `ShadowOnGround()` is already half
of that. Beyond it, temporal reprojection (p.155–157) is the largest remaining
item but needs a history buffer in RenderHook.
