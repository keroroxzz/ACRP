# Paper 5 — Photo-Guided Procedural Cloud Fitting

**Inverse-fitting a procedural volumetric cumulus scene to a single real photograph, by genetic
optimization, with a physically-based (Paper-3) atmosphere.**

Author's experiment report. Reproduction package: this folder (`articles/5_photofit/`).
Result: masked sky/cloud RMSE **0.141 → 0.051** vs the reference photo.

---

## Abstract

We want a *reference benchmark sky* that looks like a specific real photograph (a fair-weather
cumulus field), so that a from-scratch volumetric renderer (e.g. A3 vs classic raymarch) can be
evaluated on a realistic, reproducible scene. We treat this as inverse rendering: the photo's
sky/cloud region is the ground truth, and a genetic algorithm searches the parameters of a
procedural cumulus scene to minimize the masked RMSE against the photo. A pure-procedural model
*floors at ≈0.13* because a single global noise offset cannot place individual clouds where the
photo has them. We break that floor with **photo-guided screen-space coverage** — the photo's
per-pixel cloudiness gates the cloud density, so the silhouette/distribution matches this view while
procedural 3-D billow and high-frequency detail shade the interior — reaching **0.051**, which we
show is the hard floor for a *smooth-atmosphere + procedural-cloud* render (the residual is the
photo's unmatchable spatial texture). The sky uses the Paper-3 hybrid single+multiple-scattering+
ozone atmosphere.

---

## 1. Problem & motivation

A good real-time-cloud benchmark needs a scene that (a) is procedural (renderable from scratch by any
method under test) and (b) resembles a real sky, so speed/quality numbers transfer to production.
Rather than hand-tuning, we *fit* the scene to a real photo. The metric — masked RMSE of our render
vs the photo over the sky/cloud region — doubles as a "how real does it look" score.

Reference photo: `reference_photo.jpg` (a fair-weather cumulus field over farmland; broken cumulus
with flat bases, blue gaps, denser toward the horizon).

## 2. Forward model (what we render)

Shader: `cumulus_fit.glsl` (GLSL 330, offscreen). All scene parameters are **uniforms** so the
optimizer changes them with zero recompile (thousands of evaluations/minute).

- **Sky — Paper 3 atmosphere.** Analytic Chapman single scattering + a coarse directional-gather
  multiple-scattering term + ozone (Chappuis) absorption. The observer is placed on the globe at
  `(0, planetRadius+h, 0)` so local up = +y and our scene rays map directly; `lightDirection` = the
  fitted sun direction.
- **Clouds — one broken cumulus layer.** Flat condensation base `cbase`, rounded top whose height
  grows with coverage, procedural coverage from cubic-fade value-noise fbm, billowy 3-D erosion, and
  a **high-frequency detail** erosion term (`detAmp/detFreq`) so the clouds are not too smooth.
  Density is `max(mask·profile·(0.22+billow·bAmp) − erode − detail, 0)·dens`.
- **Lighting.** A cheap multi-scatter approximation (octave beer-powder) so cloud cores are bright,
  not black; tunable cloud ambient; HG phase; a few-step sun march for self-shadow.
- **Empty-space skip.** A coarse `presence()` gate lets the camera march big-step through clear air.
- **Tonemap.** ACES + gamma, with a final white-balance/exposure `colGain`.

## 3. Method

### 3.1 Target extraction (`fit_photo_guided.py::build_target`)

The photo is resized to 480×360 and a **sky/cloud mask** is built: upper region (rows < 0.535),
bright (> 0.34), not green (drops the fields), minus the dark window-frame corner and the dark thin
**power lines** (dark relative to a blurred copy, over blue). From the masked photo we compute a
per-pixel **cloudiness** map: `clip((brightness−0.55)/0.30) · (saturation<0.22)` — bright *and*
whitish → cloud; blue sky → 0. Saved to `out/photo_skymask.png`, `out/cloudiness.png`.

### 3.2 Photo-guided screen-space coverage (the key idea)

Pure procedural placement fails (§4.2). Instead we upload the cloudiness map as a texture and, in the
shader, sample it at `gl_FragCoord.xy/uRes` to get a per-pixel value `gScr` (constant along that
camera ray). We then **gate the cloud density by `gScr`**:

```
gs   = pow(gScr, uCovGamma);            // opacity curve
skip cell if presence(p)*gs <= 0.02;    // gScr≈0 (blue sky) -> ray stays empty
dens = cumulus(p) * gs;                 // gScr>0 (photo cloud) -> procedural 3-D cloud appears
L   += T * S * mix(uCloudLo,uCloudHi,gScr) * (1-exp(-dtau));  // brighter where the photo cloud is brighter
```

So **the photo decides *where* clouds are and *how opaque/bright*; procedural noise decides the *3-D
shape and detail*.** It is still a real volumetric render, just *placed* by the photo — clouds have
correct silhouettes for this view, procedural billow/erosion interiors, and Paper-3-lit shading.
(The cloudiness texture is uploaded vertically flipped, because `glReadPixels` is bottom-origin and
the render is flipped back on read; getting this wrong renders the sky upside-down — see §4.4.)

### 3.3 Optimizer (genetic algorithm)

39 parameters (camera 4, sun+atmosphere 5, cloud shape 11, detail 2, lighting 6, colour gain 3,
cloud-brightness curve 2, opacity gamma 1, horizon haze 2, sky-ambient 3). GA: population 44,
elitism 5, tournament selection from the better half, uniform crossover, sparse Gaussian mutation
with cooling, warm-started from its own best across runs. **Loss** = masked RMSE (full + a lightly
blurred term, so large-scale distribution is matched robustly). Fit resolution 480×360 for speed.

## 4. Pitfalls (what went wrong on the way)

### 4.1 Paper-3 atmosphere white-out
The Paper-3 radiance is HDR on a large scale (`sunIntensity·betaR·…`, with `sunIntensity≈22` for its
own `1−e^{−L}` tonemap). Fed into our ACES tonemap it saturated to pure white (`mean rgb ≈ 0.9`).
**Fix:** treat `sunIntensity`/`atmDens`/`exposure` as free params and *calibrate to the photo's blue*
— a small sweep matched the photo's blue-sky mean `(0.44,0.54,0.65)` at `sunInt≈12, expo≈0.12,
atmDens≈2.2`. Lesson: a physically-based sky still needs an exposure calibration against the target.

### 4.2 Pure procedural floors at ≈0.13
With procedural coverage (a global domain offset `offX/offZ`), the GA can slide the whole noise field
but cannot independently place clouds. It reproduces the *statistics* (coverage fraction, size, colour)
but not the *positions*, so the per-pixel RMSE floors around 0.13 (see `fit_procedural.py`). This is
the core reason photo guidance is needed.

### 4.3 World-space coverage projection failed
First attempt at guidance: project the photo's cloudiness onto the cloud-base plane (`y=cbase`) via
the fitted camera, bake a world-space coverage texture, render from that. The **horizon perspective
compression** (rays near the horizon hit the plane at huge `t`) crammed distant clouds into a tiny
corner and pushed near clouds outside the mapped extent — the clouds bunched at the top of frame and
the horizon band vanished (`out/guided_vs_photo.png`). **Fix:** abandon world-space; use
**screen-space** coverage (§3.2), robust for a single fixed view.

### 4.4 Vertical-flip bug
`glReadPixels` returns rows bottom-to-top; saving without flipping renders the whole sky (and the
cumulonimbus in the sibling scene) upside-down. Fixed by flipping the readback and uploading the
cloudiness texture pre-flipped so it aligns.

### 4.5 Over-aggressive power-line masking
A first line-removal (dark-relative-to-blur, threshold 0.045) removed **13.8%** of pixels — it also
caught cloud edges and shadows, biasing the fit. Tightened to `blur−bright>0.10 & bright<0.55 & b>r`
(dark thin over *blue* only) → **1.86%** removed, just the lines.

### 4.6 The 0.051 floor is real (two no-op fixes)
To beat 0.051 we added (a) an opacity gamma on the cloudiness and (b) a tunable aerosol horizon-haze
term (Mie-like, since Paper 3 is Rayleigh-only). **The GA drove both to no-ops** — RMSE stayed 0.051.
The residual is the photo's *spatial texture* (real cloud micro-detail + patchy sky haze/thin cirrus),
which no smooth model or *uncorrelated* procedural noise can reduce (uncorrelated detail only raises
RMSE). Below ≈0.05 requires compositing the photo's actual texture — no longer a procedural render.

## 5. Results

### 5.1 RMSE progression (masked sky/cloud region vs the photo)

| step | RMSE | change |
|---|---|---|
| initial procedural, linear-gradient sky | 0.141 | baseline |
| + Paper-3 atmosphere, calibrated | 0.131 | physical blue matches |
| pure-procedural GA floor | ~0.13 | global offset can't place clouds (§4.2) |
| + photo-guided screen coverage | 0.088 | **distribution matches** |
| + extended/feathered sky mask (horizon band) | 0.067 | capture the low cumulus band |
| + final colour-balance gain | 0.059 | white-balance/exposure match |
| + power-line masking | 0.056 | drop unmodelled dark lines |
| + cloud-brightness vs cloudiness | **0.051** | brighter where photo cloud is brighter |
| + opacity gamma, + aerosol haze | 0.051 | **no-ops → floor confirmed** |

Diagnostic split at 0.051: cloud-region RMSE ≈ 0.050, sky-region RMSE ≈ 0.060. Figures:
`out/FINAL_photo_vs_fit.png` (photo vs fit), `out/fit_vs_photo.png`, `out/procedural_vs_photo.png`
(the 0.13 baseline), `out/cloudiness.png`, `out/guided_vs_photo.png` (the failed world-space attempt).

### 5.2 Final optimized parameters (masked RMSE 0.0512)

Camera: `pitch 0.489, yaw −0.219, height 1.182, focal 1.250`.
Sun/atmosphere: `sunAz −3.140, sunEl 1.212, sunInt 3.458, atmDens 2.484, expo 0.546`.
Clouds: `cbase 1.000, ctop 3.509, covFreq 0.783, covLo 0.434, covW 0.120, dens 2.000, sigma 7.133,
bAmp 1.180, bFreq 1.600, erode 0.310, offX 0.0, offZ 28.28`.
Detail: `detAmp 0.391, detFreq 2.855`.
Lighting: `sunScale 7.000, amb 1.258; skyTop(ambient) (0.251,0.100,0.787), skyHor(ambient)
(0.795,0.929,1.000)`.
Photo-guidance: `cloudLo 0.806, cloudHi 1.261, covGamma 0.966`.
Colour gain: `(1.220,1.117,1.016)`.  Horizon haze: `hazeAmp 1.535, hazeH 0.234`.
(Machine-readable in `out/best_params.json`; `x` is the normalized vector.)

## 6. Discussion & limitations

- **Screen-space guidance is 2.5-D.** Clouds have the photo's silhouette for this view; 3-D parallax
  is approximate. For a single benchmark viewpoint this is exactly what we want; for a free-flying
  camera one would bake a proper world/volume coverage (the world-space attempt, done right with a
  bounded extent and horizon clamp).
- **Floor ≈ 0.05.** Honest limit of *procedural* matching. The fitted *style* (sky, lighting, cloud
  shape/detail params) can also drive a fully-procedural coverage scene (no photo) as a
  photo-*realistic* — not photo-*identical* — benchmark; that is the intended downstream use.
- **The optimizer is gradient-free** (GLSL is not autodiff'd here). A GA is robust to the multimodal
  landscape (the noise offset especially) and needs no shader differentiation.

## 7. Reproduction

```
pip install numpy scipy glfw PyOpenGL Pillow          # + an OpenGL 3.3 GPU
python fit_photo_guided.py 200     # builds target from reference_photo.jpg, GA-fits, -> out/
python fit_procedural.py 200       # the no-guidance baseline (floors ~0.13)
python render_final.py             # labelled photo-vs-fit comparison from out/best_params.json
```
`fit_photo_guided.py` warm-starts from `out/best_params.json`, so re-running keeps refining. Files:
`cumulus_fit.glsl` (shader), `fit_photo_guided.py` (main), `fit_procedural.py` (baseline),
`render_final.py` (figure), `reference_photo.jpg` (target), `out/` (params + images).
