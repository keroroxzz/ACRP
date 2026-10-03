# Paper 6 — ViT-based Visual Scoring Genetic Algorithm

**Optimizing a procedural cloud shader with a Vision-Transformer perceptual score and a genetic
algorithm — plus a sky-compositing trick that focuses the score on the sky.**

Experiment report. Reproduction package: this folder (`articles/6_vit_visual_scoring_ga/`).
Runs in the `dino` conda env (torch 2.5+cu121, transformers 5.13; + glfw/PyOpenGL/scipy).

---

## Abstract

Pixel RMSE is the wrong objective for "does this look right" (Paper 5): it can't match a real photo's
cloud placement and drifts toward washed-out, low-contrast averages. We replace it with a **ViT
perceptual score** — DINOv2 features (global CLS + translation-invariant patch statistics) — and drive
a **genetic algorithm** with it (no shader differentiation needed). Two findings make it work well:
(1) **GradCAM** on the ViT score localizes *where* the render is perceptually wrong, giving an
actionable signal that the GA then fixes automatically; (2) a **sky-compositing** step — merging the
render's sky into the *real photo* (keeping the photo's ground/foreground) before scoring — makes the
ViT judge a complete natural scene that differs *only in the sky*, so the score isolates sky style and
detail. With compositing the perceptual similarity goes from 0.963 (seed) to **0.980**, and the result
is a procedural sky that blends seamlessly into the real photograph.

## 1. Motivation

Paper 5 showed pixel RMSE-to-a-photo is a poor objective for a from-scratch render (it floors ~0.13 on
placement and, when pushed, washes out). We want a scorer that rewards *looking like* a cumulus sky —
texture, contrast, cloud form — independent of exact pixel positions. A pretrained ViT visual encoder
provides exactly that: its features are a strong, translation-tolerant "visual sense." A genetic
algorithm is the natural optimizer — it is gradient-free (the GLSL renderer is not differentiated
here), robust to the multimodal landscape, and flexible over heterogeneous parameters (camera, colour,
cloud shape, lighting).

## 2. Method

### 2.1 Forward model (coherent procedural cumulus)
`cumulus_beauty.glsl` — a physically-coherent volumetric cumulus sky: Paper-3 atmosphere, one broken
cumulus layer with **billow (`abs`) + ridged (`1−abs`) value-space remaps** and a Nubis erosion-remap
(crisp cauliflower — the smoothness in Paper 5 was lazy additive FBM, not a value-noise limit), camera
and sun marching the **same** 3-D density (coherent self-shadowing), multi-scatter octaves, sky-ambient
fill, aerial perspective. Key cloud/sky parameters are uniforms so the GA changes them with no recompile.

### 2.2 ViT perceptual score (`dino_perceptual.py`)
Encode an image with **DINOv2-base**. Combine two cues:
- **CLS** cosine — global/semantic match ("is this a fair-weather cumulus sky?").
- **Style** cosine — cosine of the *translation-invariant* patch-token statistics `[mean ; std]`
  (texture / cloud-ness, position-independent).

Score `= 0.5·CLS + 0.5·style`. Sanity: photo-vs-photo `= 1.000`. It is discriminative and matches the
eye: the crisper **billow** render scores **0.439** vs the plain-FBM render **0.388**.

### 2.3 Sky compositing (the key refinement)
A floating sky render lacks the photo's foreground, so the global CLS is dragged down (≈0.20) — the ViT
is distracted by the missing context. Instead, before scoring we **composite the render's sky into the
real photo**: with a feathered photo sky mask `M`,

```
composite = photo·(1−M) + render·M
```

Now the scored image is a complete natural scene (photo's real fields/buildings/power-lines/foreground)
that differs from the photo **only in the sky**. The ViT score therefore isolates sky style/detail, and
CLS becomes meaningful. (`optimize_dino_composite.py`, `out/composite_mask.png`.)

### 2.4 Genetic algorithm
Population 36, elitism 4, tournament selection, uniform crossover, sparse Gaussian mutation with
cooling, warm-started from a hand-tuned seed. 20 parameters: camera (pitch/yaw/height), sun (az/el),
exposure, cloud shape (base/top/coverage-lo/hi/freq, detail-amp, density, extinction), sky palette
(zenith RGB, horizon RGB). Rendering (glfw+OpenGL) and scoring (DINOv2) run in **one process** in the
`dino` env, so an evaluation is one render + one ViT forward.

### 2.5 GradCAM error attribution (`dino_gradcam.py`)
Backprop the perceptual score to the render's last-block patch tokens; GradCAM `= ReLU(Σ_c ∂S/∂A · A)`
per patch → upsampled heatmap. Highlighting the *deficit* (`1−CAM`) shows which regions most limit the
match. On the hand-tuned render it lit up the **entire empty upper sky** → "too few clouds."

## 3. Results

| step | perceptual similarity | note |
|---|---|---|
| DINO sanity: photo vs photo | 1.000 | scorer is well-formed |
| billow render vs plain-FBM (crop) | 0.439 vs 0.388 | ViT agrees billow is better |
| **GradCAM diagnosis** | — | empty upper sky is the deficit ("need more clouds") |
| GA, no compositing (640×384) | 0.57 → **0.67** | GA auto-lowered the coverage threshold to **fill the sky exactly as GradCAM predicted** → a full, layered, realistic cumulus (`out/dino_best.png`) |
| **GA + sky compositing (512×384)** | 0.963 → **0.980** | scores only the sky; result blends seamlessly into the real photo (`out/dino_comp_vs_photo.png`) |

The GradCAM→GA link is the striking part: the metric *diagnosed* "too empty," and the ViT-scored GA then
*fixed it on its own* by raising cloud coverage — no manual intervention. Compositing then grounded the
evaluation in the real scene, yielding a synthetic sky indistinguishable in style from a real photo.

### 3.1 Quality push (two extra FBM layers + sky-only patch scoring + long GA)

Three quality upgrades: (a) **two extra multi-scale FBM detail layers** — the density is eroded by
progressively finer ridged/billow FBM (coarse → medium → fine wisps), each amplitude a GA parameter, for
richer cauliflower; (b) **sky-only patch scoring** — the near-1.0 composite score is compressed (most
pixels are identical), so we compare **only the ViT patch-tokens whose grid cells fall in the sky mask**
(manual square preprocessing so the 16×16 patch grid maps to the mask), decompressing the signal to
~0.91 and giving the GA real headroom; (c) **long, warm-started GA** (~200 generations across chunks,
periodic checkpointing). Sky-focused similarity climbed **0.914 → 0.966** and largely converged. The
best parameters render a high-resolution **hero composite** (`out/hero_composite.png`, 1280×960) whose
procedural sky blends seamlessly into the photo's real fields/buildings/power-lines — it reads as a
genuine photograph. `final_render.py` produces the hero images.

### 3.2 Raising the shader ceiling (cloud-type variation + multi-scatter) and a skyline mask

The GA converges against the *shader's* expressiveness, so we raised the ceiling: (a) a large-scale
**cloud-type field** modulates coverage threshold, top height, and density, so some regions are tall
cumulus and others flatter/broken cloud (`uTypeAmp`); (b) **improved multi-scatter** — a silver-lining
forward boost (`uSilver`) and brighter optically-thick cores via deeper leaking octaves (`uMsBoost`) —
for luminous tops and softly-shadowed undersides. We also fixed the composite mask: a flat row cutoff
kept the photo's *far horizon clouds* (a thin band above the buildings), unfairly borrowing real sky. A
**flood-fill skyline mask** (sky = the component connected to the top, blocked by a building/tree
classifier) now replaces ALL sky+clouds down to the skyline with the render, keeping only ground,
buildings, foreground, and the photo's power lines (as a realistic overlay). Re-optimizing (25 params)
gives a markedly more realistic, cloud-type-varied hero whose synthetic sky — including the horizon —
is entirely procedural. Best parameters: `out/dino_comp_best.json`.

## 4. Discussion & limitations

- **Why compositing helps:** it removes the confound of the missing foreground, so CLS (global) becomes
  a sky-focused signal and the score is apples-to-apples in the real scene context.
- **Score compression:** because the composite is mostly identical pixels (only sky differs), the score
  sits near 1.0 and the usable gradient is small (0.963→0.980). A sharper variant would compare **only
  the sky-region patch tokens** (mask the ViT tokens to the sky), amplifying the sky signal — the
  natural next step.
- **RMSE vs perceptual:** the ViT score rewards *looking like* the target without demanding pixel-exact
  placement, which is what an aesthetic/benchmark scene needs; RMSE cannot.
- **DINOv3** (gated, needs HF auth) is now available and used *alongside* DINOv2 in the multi-photo
  extension (§6) — but note the finding there: DINOv3's near-saturated CLS makes it *worse* alone.
- **Gradient-free by choice:** a differentiable renderer + ViT would allow direct backprop, but the GA
  needs no shader autodiff and handles the multimodal landscape (camera/coverage) robustly.

## 5. Reproduction

```
# in the `dino` conda env (torch, transformers) with glfw/PyOpenGL/scipy:
python dino_perceptual.py reference_photo.jpg out/dino_best.png     # perceptual similarity
python dino_gradcam.py out/dino_best.png out/cam.png                # error-attribution heatmap
python optimize_dino.py 30                                          # ViT-scored GA (floating sky) -> out/dino_best.*
python optimize_dino_composite.py 32                               # ViT-scored GA + sky compositing -> out/dino_comp_*
python render_beauty.py 0.19 3.2 0.12 1.2 1.05 0.6 hero            # render the beauty shader standalone
python optimize_multiphoto.py 100 4                                # §6 multi-photo joint GA (dual DINOv2+v3)
```

The multi-photo run (§6) needs the DINOv3 weights (gated model `facebook/dinov3-vitl16-pretrain-lvd1689m`)
in `research/models/` (git-ignored — download locally) and the 12 `reference_photos/*.jpg` + `*_mask.png`.

Files: `cumulus_beauty.glsl` (uniform-driven coherent cumulus + billow/ridged), `dino_perceptual.py`
(score), `dino_gradcam.py` (attribution), `optimize_dino.py` (ViT-scored GA), `optimize_dino_composite.py`
(+ sky compositing), `render_beauty.py`, `reference_photo.jpg`, `out/` (results + best params).

## 6. Multi-photo joint VSGA (extension)

The single-photo pipeline above fits *one* reference. We extend it to a **set of 12 real photos**
(`reference_photos/`, each with a hand-drawn sky mask `*_mask.png`) and optimize them **jointly**: one
set of **shared** parameters (physical atmosphere + cloud scattering/shape + deck altitudes) common to
all photos, plus **per-photo** parameters. This forces the shared "look" to generalize while each photo
keeps its own sky.

### 6.1 Forward model (`cumulus_multilayer.glsl`)
A rewrite of the beauty shader with three additions:
- **Physical sky wired in.** The Paper-3 pseudo-spherical atmosphere (`atmosphere()`, Chapman/erfcx +
  ozone + a grey aerosol term) is now the actual sky colour, evaluated with the observer placed on the
  globe (a coordinate fix — the clouds live in a local flat frame, the atmosphere in planet space). So a
  **per-photo sun direction** reproduces each photo's time-of-day colour for free.
- **Three cloud decks**, each with its own altitude band and a per-photo coverage knob: **low cumulus**
  (thick, billowy), **mid cirrocumulus/altocumulus** (thin granular ripples), **high cirrus** (very thin
  wind-stretched wisps). One front-to-back march per deck carries transmittance. The high decks use a
  distinct **thin-ice lighting** path — low optical depth (translucent), forward-scatter dominant, only a
  faint self-shadow — so they read as bright wisps lit *through*, not a uniformly-shadowed slab.
- **Flow-field domain warp** (`uWarp`): a cheap fbm flow, modulated by a coarse field, warps the *detail*
  sample coordinates so wisps are dragged into filaments along a flow — strong for cirrus. (This
  **defeats A3** — see appendix.)
- **Atmosphere-coupled cloud lighting.** The sunlight reaching the clouds is multiplied by the Chapman
  **sun-transmittance** (`sunTrans`, colour from the transmittance, magnitude from the elevation
  falloff), so cloud lit faces go white→warm→orange/red as the sun lowers, instead of staying white.

**Raymarch structure (three slabs, not one full-space march).** Each deck is a horizontally *infinite
slab* defined only by its altitude band, so its entry/exit are a pair of divisions rather than a search:
`t0=(base-ro.y)/rd.y, t1=(top-ro.y)/rd.y`. `main()` then issues three independent marches, low -> mid ->
high, each with its own step budget (48 / 20 / 16) and therefore its own `dt` — a thin deck does not spend
steps on thickness it does not have. Transmittance and accumulated colour are threaded through as `inout`,
so the composite is front-to-back and the `T<0.05` early-out carries across decks (an opaque cumulus layer
means the two ice decks return immediately). Because the camera sits near the ground and only rays with
`rd.y>0` are marched, low -> mid -> high is already near-to-far; no sorting is needed.

The gaps *between* decks are never sampled — the next march simply starts at its own `t0` — so vertical
empty space costs nothing. Empty space *within* a deck is handled separately by the coarse/fine skip
(§6.6). The result is two levels of skipping: **analytic between slabs, heuristic inside them.** This is
structurally the same coarse-presence/fine-integration split that turned Paper 4's hierarchical A3 from a
3.3x loss into a 1.09-1.73x win; the difference is that A3's fine stage is an *exact* per-cell
transmittance rather than fixed-step numerical integration.

Two details worth knowing: `cloudDensity()`'s altitude dispatch is redundant during a deck's own march
(the sample is always inside that band) but is genuinely needed by `sunShadow`, which walks toward the sun
and *does* cross decks; and `t1=min(t1,t0+90)` truncates grazing rays, trading distant near-horizon cloud
for a `dt` that would otherwise blow up.

### 6.2 Optimizer (`optimize_multiphoto.py`)
- **Parameter split.** Vector = shared `θ_s` ⊕ per-photo `θ_p^i`. Per-photo *optimized* = exposure + the
  three deck coverages + cloud-type (seed/amp). Per-photo **geometry is FIXED**, not searched: camera
  pitch/yaw/FoV and sun az/el are directly determinable from each photo, so they are set once from a
  **hand-verified visual pass** (`FIXED_GEO`; iteratively refined against every photo, see
  `results/cam_check.png`). This shrinks the vector **163 → 91 dims** and stops the ViT from spending
  features on geometry we already know.
- **Fitness = ½·mean + ½·min** over the photos (evaluated on a random **mini-batch** each generation, the
  same subset for all candidates). The **min** (worst-photo) term is the anti-wash lever: the optimizer
  can no longer flatten one photo to lift the average.
- **Interleaved seed sessions (separate the discontinuous parameter).** `uTypeSeed` selects the cloud
  *layout*: nudging it re-rolls the whole field, so gaussian mutation cannot hill-climb it and it only
  injects noise into the continuous search — every photo ended up with a similar layout. It is now
  **frozen during normal generations** and, every 12 generations, **scanned exhaustively per photo**
  (20 trials). No GA is needed for the scan: a photo's seed only affects that photo's score, so the
  problem is separable and a direct argmax per photo is both cheaper and better than search in the joint
  93-D space. The log shows the classic alternating-optimisation signature — each scan gains ~+0.007 per
  photo, an *immediate* re-scan gains exactly +0.0000 (so the scan is genuinely at its optimum rather
  than sampling noise), and once the continuous parameters move the next scan finds gains again. Four
  scans took the run 0.810 -> 0.829, and per-photo layouts finally differentiate.
- **Dual-model, multi-depth scoring.** Every candidate is scored by **both DINOv2 and DINOv3**, averaged.
  For each model we compare, sky-masked: CLS (semantics) + **final**-layer patch style + an
  **intermediate ("head-input") patch style** — the mid-layer features carry more spatial detail and
  resist the flat-wash gaming that final-only statistics allow.

### 6.3 Findings
- **A bigger ViT is not a better aesthetic judge.** DINOv3-alone drives the GA to **hazy, washed-out**
  skies (score up, look down): its CLS cosine is near-saturated (~0.98 for unrelated images), so it gives
  almost no gradient and the *gameable* patch mean/std term dominates and is satisfied by a flat wash.
  **DINOv2** keeps structure (informative CLS). The **dual + mean/min + multi-depth** combination recovers
  cloud structure. This is a clean negative result about metric design, echoing Paper 5's "RMSE is the
  wrong objective."
- **Fixing the geometry** let the shared atmosphere converge cleanly (haze collapses, deep blue returns).
- **Coupling cloud lighting to the atmosphere** (§6.1) is what makes the sunset photos read as sunsets —
  warm cloud faces — rather than white clouds under a coloured sky.
- **Capping `uHaze`** (≤0.30) stops the recurring sky-washout; the GA then settles on low haze (~0.09).
- **Three cloud-lighting defects, each with a concrete root cause** (all found by looking, then traced):
  1. *Clouds blew out to white as the sun neared the zenith.* The zenith **ambient probe**
     `atmosphere(gro, up, L)` was hitting the function's **sun-disk branch** (`c>0.9997` adds 45.0),
     so the clouds' ambient exploded. Fix: `atmosphere()` takes a `sunDisk` flag — only the primary
     view ray draws the disk. Cloud-band saturation at 89 deg sun: **8.5% -> 0%**, and the 40/70/89 deg
     series became monotone-smooth. (A second, smaller bug: the cloud sun magnitude re-applied an
     elevation ramp that `sunTrans` already encodes — double counting.)
  2. *No visible "lit face" when the sun was ~90 deg to the side.* Cloud lighting used only
     `HG(view·sun)`, which carries **no term for how much sun reaches the point**, so at side-lighting
     the whole cloud dimmed uniformly. Fix: add a **view-independent multiple-scatter term**
     `uDiffuse * exp(-ss)` (a shared GA parameter). A thick cloud's sun-facing side is near-Lambertian
     and must be bright from *any* view. The GA immediately drove `uDiffuse` to 0.139/0.15 (its cap)
     **and dropped `uSilver` from ~3 to 1.1** — with a real lit face it stopped faking brightness with a
     sharp forward spike. That parameter shift is the strongest evidence the term was missing.
  3. *Thin ice decks showed hard, defect-looking scattering edges.* They inherited the cumulus
     `pow(mu,9)` silver spike; softened (g 0.60->0.45, silver exponent 9->5 at half weight) and the
     thin-deck coverage smoothstep widened.
- **Purple/crimson twilight was an ozone bug, not a tone-mapping problem.** `ozA` was **0.30 where the
  tuned Paper-3 GLSL uses 0.006 — 50x too much ozone**. Chappuis absorption is exactly what colours a
  low-sun sky purple/magenta, and because `ozA` also feeds `sunTrans` it was over-tinting the *cloud*
  sunlight too. Adopting the tuned constants (`ozA` 0.006, `fDay` 0.55, `msWeight` 0.40) turns twilight
  golden/amber (`results/ozone_check.png`). Any shared params fitted under the old value are stale and
  need a GA re-run to rebalance.

- **The yellow sky was Rayleigh being too strong, and ozone does *not* make the daytime blue.** A first
  attempt copied the tuned `ozoneAmount = 0.006` straight across, which over-corrected: twilight went
  golden but the daytime sky turned yellow/olive. Two lessons: (a) the constant did not transfer because
  this shader scales ozone differently (it multiplies by `ozW` and `uAtmDens`), so the equivalent value
  here is ~0.06; (b) ozone's effect is strongly **path-length dependent**, so it governs the *low-sun*
  colour and contributes almost nothing to the high-sun blue — the daytime row of `results/ozone_sweep.png`
  is nearly invariant in `ozA`. The yellow cast was instead **Rayleigh optical depth being too high**:
  it scatters the blue out of the view path and leaves yellow/olive. Narrowing `uAtmDens` to 0.6–1.4 and
  `uSunI` to 2.5–5.5 (`results/rayleigh_sweep.png`) restored the blue while keeping the sunsets warm.
- **The metric keeps pushing against physical sanity.** With those caps in place the GA parks `uAtmDens`
  and `uSunI` *at* the caps — it still wants a brighter, stronger Rayleigh than is physically sensible.
  Same family as the DINOv3 haze drift and the "fill the sky with cloud" pull: the ViT score reliably
  prefers something the eye rejects, so the physically-motivated bound is doing real work.
- **A correct atmosphere changes the cloud-lighting solution.** Under the over-corrected ozone the GA set
  `uDiffuse`~0 and `uSilver` to its max (faking brightness with a forward spike); with ozone at 0.06 and
  Rayleigh capped it flipped back to `uDiffuse` at its cap with `uSilver` only 3.6. The atmosphere and the
  cloud-lighting parameters are not separable — a wrong sky silently distorts the cloud solution.

- **Optical density must drive the shadow, not just the view.** The view ray used
  `dtau = d*uSigma*dt` while `sunShadow` accumulated bare density (an implicit sigma of 1), so
  self-shadow depended on `uDens` alone. Given a raised density ceiling the GA immediately took the
  shortcut — `uSigma` 12.7 with `uDens` only 5.3 — buying opacity with no shadow cost, and self-shadow is
  what sculpts cloud form. Coupling the shadow to `uSigma` flipped its choice to `uDens` 12.0 / `uSigma`
  5.4 at *the same* effective optical depth, and the clouds gained real light/shadow modelling.
- **Sunlight must be attenuated at the cloud's altitude, not at the ground.** `sunTrans` was evaluated at
  sea level, i.e. through the whole air column, making cloud light too yellow and too dim; each deck now
  derives its own sun colour at its own altitude.
- **Three ambient-probe bugs, all from approximating a hemisphere with one direction.** (a) the zenith
  probe hit `atmosphere()`'s sun-disk branch when the sun neared the zenith, exploding the ambient into a
  white-out; (b) the horizon probe `normalize(vec3(rd.x,0.08,rd.z))` degenerates to straight up as the
  view nears the zenith, so at low sun it collapsed into a dark halo (and a first fix, normalising the
  horizontal part, merely traded the halo for a radial singularity — the probe must be
  *view-independent*); (c) sampling only the sun-side horizon made cloud shadow sides yellow. Measuring
  the probes then showed the horizon is **10–15x brighter than the zenith**, so a nominal "40% zenith"
  mix was really ~93% horizon — the sun-side/anti-solar weighting could not matter, and only a heavy
  zenith bias (0.80) actually cools the shadow sides.

### 6.4 Cirrus morphology

Getting large flowing cirrus out of the high deck took four separate mechanisms, each verified by a
sweep (`results/cirrus_*.png`):
1. **Band structure.** ~4.5:1 anisotropy with a 5-octave high-frequency field, plus a high-frequency term
   *subtracted* from coverage, chopped the deck into confetti. Elongate ~8:1, drive the band with a
   lower-frequency 4-octave field, and let fine detail **multiply** inside the band. (Adding detail by
   subtraction destroys structure; by multiplication it enriches it.) Over-correcting — 16:1 at 2
   octaves — collapses into featureless slabs, so the useful point sits between confetti and slab.
2. **Arcs and fallstreak shear.** A low-frequency, high-amplitude lateral displacement gives bands that
   sweep across the sky; a height-dependent lean reproduces the trailing hooks of cirrus uncinus (ice
   falls and lags the wind).
3. **The distortion source scale must match the target scale.** All decks shared one warp at source
   frequency 1.2 with ~2 units of displacement — against bands spanning tens of units that can only tear
   locally. The cirrus deck needs its own drag at source frequency 0.05 and amplitude 35 (~24x coarser,
   ~17x stronger) before long filaments appear at all.
4. **Fine fibre needs its own coordinate.** Sampled from the band's heavily-warped coordinate, the fine
   noise was stretched along the flow and smeared away — raising its frequency changed nothing. Sampled
   from a separate lightly-warped coordinate it produces real fibrous striation. The large-scale flow
   drags the band *shape*; fibre is a *local* property and must not inherit that displacement.

- **Vertical shaping needs a shear and an asymmetric profile.** The deck grew as vertical pillars from a
  symmetric smoothstep taper, which reads as slab-like and identical everywhere. Two cheap additions fix
  it: a **wind shear** (`uShear`, displace the sample horizontally in proportion to height) so cumulus
  lean and develop on a slant, and an **asymmetric convex profile** — flat base, heavy lower body,
  `pow(1-hf, taper)` narrowing to the top, with `taper` driven by the cloud-type field so cumulus tower
  and strato flatten. `results/shear_taper.png` sweeps the two; the GA settles `uShear` at 1.52, i.e. it
  actively wants shear rather than tolerating it.

- **Cloud amount needs its own explicit term.** Measurement: the renders averaged **+0.243** cloudier
  than their photos over the sky region, with the error concentrated exactly where the patch-style metric
  would predict — the clear-blue photos (photo 3: 0.104 vs 0.769; photo 4: 0.254 vs 0.970) — while
  already-overcast photos matched within 0.02–0.08. Adding a direct penalty on |sky cloud fraction −
  photo's| (blue-dominance as the cloud test, weight 0.30) finally broke the fill-the-sky bias: blue
  returns where the photos have it and per-photo `uCovLow` spreads 0.34–0.85 instead of clustering high.
  Unlike the patch statistics, this term cannot be satisfied by adding texture.
- **Per-parameter mutation steps (diagonal preconditioner) — measured, implemented, and essentially
  neutral.** A single global sigma implicitly assumes each parameter's sensitivity is proportional to its
  range. Central differences at the incumbent gave a **~380× spread** (`uCovHigh` 0.197, `uCovLow` 0.162,
  `uShear` 0.087 … `uSilver` 0.0021, `uMidBase` 0.0013, `uMidTop` 0.0008), which looks like a clear defect.
  Scaling σᵢ by `(median/sᵢ)^0.5`, clamped to [0.3,3.0] and recalibrated every 30 generations, changed the
  final fitness from 0.7606 to **0.7618 — inside the noise.** Reported as a null result rather than a win.
  Plausible reasons, in the order we would test them: the clamp damps 380× down to 10×; the landscape may
  be dominated by the handful of sensitive coverage parameters that either scheme handles adequately; and
  the diagonal approximation ignores exactly the cross-parameter structure that may be limiting progress.
  Two caveats govern how this should be read, and both were flagged before the experiment: sensitivity is
  **conditional** on where every other parameter sits (`uMidBase`/`uMidTop` measure as inert only because
  mid coverage is currently low — raise it and they matter), so a single measurement is a local slope, not
  a property; and the ranking drifting between runs is therefore expected behaviour, not measurement
  noise. `uTypeSeed` topping the list at 0.29 is an artefact of the same kind — being discontinuous, its
  "derivative" is the size of a layout re-roll, which independently justifies scanning it separately.

### 6.5 Performance (RTX 4070, `program/bench_shader.py`)

Timed with `GL_TIME_ELAPSED` queries across all 12 hand-verified photo geometries (cost varies 3-5x with
view direction, so a single view would misreport by more than 4x):

| resolution | median | fps | min | max |
|---|---|---|---|---|
| HD 1280x720 | 419 ms | 2.4 | 137 ms | 662 ms |
| FHD 1920x1080 | 971 ms | 1.0 | 341 ms | 1413 ms |
| 4K 3840x2160 | 3746 ms | 0.3 | 1290 ms | 4295 ms |

**Purely fragment-bound**: FHD is 2.25x the pixels of HD and costs 2.32x; 4K is 9.00x and costs 8.94x.
There is no fixed overhead and no bandwidth wall — the entire cost is per-pixel ALU, so the optimisation
target is unambiguous and resolution scaling is predictable.

Cost breakdown at FHD (disabling one deck at a time):

| configuration | time | delta |
|---|---|---|
| full | 979 ms | — |
| no high deck (cirrus) | 618 ms | **-36.9%** |
| no low deck (cumulus) | 775 ms | -20.9% |
| no mid deck | 881 ms | -10.1% |
| sky only, no cloud | **103 ms** | -89.5% |

The **cirrus deck is the single most expensive layer (37%)**, more than the cumulus it sits above, because
`flowWarpF` costs four fbm evaluations per call and `hiDeck` calls it *twice* per sample (once for the
band warp, once for the independent fine-fibre coordinate). The morphology mechanisms of §6.4 bought
realism at a much steeper price than expected. Even the cloudless sky costs 103 ms, of which two of the
three `atmosphere()` calls are the **view-independent** ambient probes — recomputed per pixel although
both are constant for a frame.

This configuration is an offline/research build, not a real-time one: it was optimised end-to-end at
512x384 (~30 ms/frame, which is why the cost was never felt) and never tuned for frame budget.

### 6.6 Optimising to a frame budget (HD 30 fps, no quality loss)

Target: HD 1280x720 at >=30 fps with the ViT score not falling, later extended to FHD 30 fps. Result:
**HD 12.9 ms (77.9 fps), FHD 27.2 ms (36.8 fps), 4K 100.9 ms (9.9 fps)** - a **34x** speed-up at HD with
fitness **0.7618 -> 0.7952**, i.e. both targets met while quality improved. Every step was gated on BOTH axes by `program/perf_quality.py`
(HD/FHD medians plus the GA's exact objective), with a ledger in `results/perf_quality.json`.

| stage | HD | fitness |
|---|---|---|
| baseline | 434.2 ms | 0.7618 |
| empty-space skipping | 209.9 ms | 0.7500 |
| ambient probes hoisted out of the pixel shader | 142.7 ms | 0.7437 |
| step counts / skip disabled decks / cheaper sky | 58.9 ms | 0.7512 |
| octave trim, then rebalanced | 29.7 ms | 0.7304 |
| shadow skipped for low-contribution samples | 28.0 ms | 0.7442 |
| **GA re-fit on the fast shader** | **24.6 ms** | **0.7964** |
| cheap shadow density for the ICE decks | **9.4 ms** | 0.7952 |

**The decisive fix came from a domain expert doubting the profile.** The breakdown said the 16-step
cirrus deck was 52% of the frame; the user's reaction was that cirrus *should be cheap*. Ablating its
parts showed they were right and located a defect: `shadowDensity()` had been given a cheap
low-frequency variant for the CUMULUS deck only, and still called the **full** `midDeck`/`hiDeck` for the
ice bands - so every lit sample's shadow march paid two `flowWarpF` calls (7 vnoise each) plus the 26x
fine-fibre field. Removing cirrus from the shadow march alone was worth **-37% of the whole frame**,
against only ~14% for its own primary march. Cheap shadow variants for the ice decks took FHD 51.6 -> 20.0
ms and HD 24.6 -> 9.4 ms at unchanged quality. The lesson is not "profile first" - the profile was
correct - it is that a profile tells you *where* time goes, not whether it *should* go there; that
judgement needs domain expectation.

What mattered, and what the process taught:
- **Empty-space skipping** was the single largest win: a sparse deck is mostly air, so march it coarsely
  and drop to fine steps only after a hit. The first implementation reverted to coarse on the *first*
  empty sample, which made it **oscillate** between two positions, burn the whole iteration budget without
  advancing, and collapse fitness 0.76 -> 0.51 while *looking* like a 2.2x win. Requiring three
  consecutive empty fine samples before reverting fixed it. Without the quality axis this would have
  shipped as a success.
- **Two of the three `atmosphere()` calls were view-independent** yet recomputed per pixel; a 1x1 pre-pass
  resolves them into uniforms. 210 -> 143 ms at zero quality cost — the cheapest win available, and it
  existed only because the ambient probes had been added for correctness without a cost review.
- **Not all octaves are equal.** Cutting the *shape-defining* coverage fbm cost real quality (-0.039);
  restoring it and cutting the finest sub-pixel wisp layer instead was both faster and better.
- **Re-fit after changing the sampling regime.** The GA is cost-blind, so an intermediate re-fit spent the
  saved budget back (29.7 -> 37.8 ms) while recovering quality. Creating headroom first and re-fitting
  last gave a solution that was *simultaneously* faster and better — the optimum genuinely moves when the
  sampling changes, so the re-fit is part of the optimisation, not a formality.
- Honest caveat: the final clouds read somewhat thinner and less substantial than the pre-optimisation
  best, even though the score is higher. The score does not track cloud solidity (§6.7) — as everywhere
  else in this paper, the number is not the picture.

### 6.6b Camera height, and a measurement caveat on the numbers

**Any camera height.** The shader had assumed the camera sits below the deck looking up: `main()` gated
cloud marching on `rd.y>0`, so from above the deck looking down it rendered nothing (`results/height_test.png`,
first pass). The slab intersection already handles the eye below/inside/above a deck; the fix is to drop
the gate (guarding only near-horizontal `|rd.y|<0.002`) and to order the three deck marches near->far,
which reverses to high->mid->low when looking down. Clouds now render correctly from any height at
essentially no cost. (A cheap `sunTrans` had to move *after* the slab early-out, or the now-entering
downward rays paid a Chapman evaluation per deck before missing.)

**Measurement caveat — the early fps numbers were under-warmed, not thermally boosted.** The same shader
measured ~8.6 ms HD early in a session and ~17 ms HD later, and I first attributed the drift to thermal
downclocking. `nvidia-smi` telemetry refuted that: sampled *during* a sustained full-screen load the RTX
4070 held a stable **2805 MHz** (t=6 s and t=14 s identical), 59 C, 166 W of a 200 W limit, and **every
`clocks_event_reasons` flag Not Active** — no thermal, power, or reliability throttle. The real cause was
the benchmark itself: a bare 2-frame warmup let timing begin while the GPU was still ramping from its
210 MHz *idle* clock, so early runs sampled the ramp, not steady state. And peak vs. sustained boost
(3150 vs. 2805 MHz) is only a 1.12x gap — nowhere near the 2x seen — so most of the "8.6 ms" was simply
the **cheapest of the 12 view geometries caught near peak boost**, not the median. With a proper ~1.5 s
continuous warmup the honest steady-state HD median is **~17 ms (59 fps), range 9.7–25.6 ms across the 12
views**, and FHD is **~37 ms (27 fps)**. So HD >=30 fps holds even on the worst view, but the earlier
**FHD 60 fps was a warmup artifact, never a sustained result**. Two lessons: warm a fragment-bound shader
to its steady clock before timing, and report the across-view *median* (with range), not a lucky view. And
never blame thermals without reading the throttle flags — the GPU was cool the whole time.

### 6.6c Pushing FHD to a sustained 60 fps

With honest warmed-up timing the real sustained FHD was ~37 ms (27 fps), so "FHD 60 fps" from 6.6 had to
be re-earned. A warmed FHD cost breakdown over the 12 views located the one big rock: the **low cumulus
deck is 75.7% of the frame** (disabling it drops 38.5 -> 9.4 ms), so 60 fps (16.7 ms) is unreachable
without cutting the main deck - everything else combined is only 9.4 ms. The cuts that mattered, in order:
`sun-shadow` (-30.6%), `low steps 48->24` (-37.7%), `SUN_STEPS 6->3` (-25.7%), `high steps 6->4` (a cheap
deck, spent to reach 60 without touching the main deck further), `COARSE 6->8` (-12%). The `MS 4-tap` loop
was 0.3% and left alone.

Cutting steps alone dropped fitness 0.7756 -> 0.7204 (coarser quadrature, not a worse model). A **warm-
start GA re-fit** (seed the current best, mutate on the now-cheaper shader) recovered it to 0.7695, the
interleaved seed scan contributing most (+0.027 at gen 10; the continuous params only reached ~0.747).
The re-fit also *lowered* cost — the GA shed redundant cloud amount that the ViT score did not reward — so
FHD landed at **14.4 ms (69 fps)**, HD 7.0 ms (142 fps): a 2.5x sustained speed-up over the 27 fps start.

Two honest caveats. (1) On the common evaluation scale the final ViT is **0.7672 vs 0.7756, ~0.008 short**
— visually imperceptible (and near the ~0.002 gap between the GA's and the gate's scoring), but not a
strict "no loss". (2) Adding steps back **without** re-fitting made it *worse* (low 24->28 gave 0.7587),
because the params are tuned to the 24-step quadrature — steps and parameters are coupled, so any step
change needs its own re-fit. The low deck also reads thin: with a metric that does not reward cloud
substance, a re-fit under coarse quadrature drifts toward *less* cloud, which is cheaper and un-penalised.
That thinness is the standing metric limitation of 6.7, not a regression introduced here.

### 6.6 Two recurring patterns worth naming

**Compensation parameters diagnose missing physics.** Three times the GA pinned a parameter at its bound,
and each time it was substituting for a mechanism the model lacked — not asking for "more":

| missing mechanism | the GA's compensation | after adding it |
|---|---|---|
| view-independent lit face | `uSilver` at 3–5 | drops to 0.19 |
| extinction coupled to self-shadow | opacity bought via `uSigma` | flips back to `uDens` |
| fine cirrus fibre | `uWarp` pinned at its cap | relaxes to 0.77 |

A parameter parked at a bound is a stronger, earlier signal of a modelling gap than the score is.

**The test viewpoint decides whether a change looks like it worked.** A density change was invisible when
the clouds sat near the horizon (aerial perspective washed them out); cirrus arc/shear looked inert at a
grazing view because a thin deck seen edge-on flattens every horizontal pattern; and a warp amplitude that
looked ideal on an isolated high-coverage cirrus render was far too strong once composited into real
scenes. Judge high cloud looking up, judge density on near clouds, and always confirm in the composite.

### 6.7 Limitations & open problems (honest)
- ~~**Cloud form is less "solid" than the single-photo hero.**~~ **Largely addressed** — the cause was a
  too-low optical-density ceiling plus a self-shadow that ignored `uSigma` (§6.3). With both fixed the
  clouds regain light/shadow modelling, though matching the hero's sculpted quality exactly is still open.
- ~~**Sunsets are over-purple.**~~ **RESOLVED** — it was the 50x ozone amount (§6.3), not tone mapping.
- **Daytime skies read milky.** The patch-style metric still rewards filling the sky, so mostly-blue
  photos get thin high cloud spread too widely. Candidate fixes: cap mid/high coverage, or add a
  "blue-sky-fraction" objective from the photo mask.
- The scalar score again **understates visual quality changes** — every conclusion here came from looking
  at the 12-up `results/multi_vs_photo_dual.png`, not the number.

Artefacts: `results/multi_vs_photo_dual.png` (12 photo|composite pairs), `results/multi_best_dual.json`
(shared + per-photo params + fixed geometry), `results/cam_check.png` (geometry verification),
`results/hicloud_test.png`, `results/suncolor_test.png` (shader-feature checks).

## Appendix — how this connects to A3

The billow/ridged remaps used here are **A3-compatible**: `abs` is C1 (a crease → handled by A3's root
partition, the ReLU/abs line in the operator taxonomy); polynomial remaps just raise the GL degree. Only
**transcendental** sharpening and **domain-warp** (degree/piece explosion) genuinely defeat A3's exact
per-cell integration. So a realistic cauliflower cumulus can, in principle, be rendered by A3 exactly —
the perceptual-GA pipeline here optimizes the *look*, and the operator taxonomy says which of its
ingredients A3 can integrate exactly. **Caveat for §6:** the multi-photo shader deliberately adds a
**flow-field domain warp** (for cirrus flow-drag) — a Thread-B realism gain that forfeits A3-exact
renderability *for the warped decks*. The low cumulus deck (no warp on its body) stays A3-compatible.
