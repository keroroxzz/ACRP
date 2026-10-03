# Paper 5 — Photo-Guided Procedural Cloud Fitting (reproduction package)

Fit a procedural volumetric cumulus scene (Paper-3 atmosphere) to a real photo by genetic
optimization. Full write-up: **[Paper5_photo_guided_cloud_fitting.md](Paper5_photo_guided_cloud_fitting.md)**.

Result: masked sky/cloud RMSE **0.141 → 0.051** vs `reference_photo.jpg`.

## Layout
- `program/` — all code + the target photo (`reference_photo.jpg`); scripts write to `../results`.
- `results/` — generated params (`best_params.json`) + comparison images.
- `Paper5_photo_guided_cloud_fitting.md` — the experiment report (method, pitfalls, results, final params).

| `program/` file | what |
|---|---|
| `cumulus_fit.glsl` | the uniform-driven shader (Paper-3 sky + volumetric cumulus + photo-guided coverage) |
| `fit_photo_guided.py` | **main**: builds the target from the photo, GA-fits, saves best params + comparison |
| `fit_procedural.py` | baseline with no photo guidance (floors ~0.13) |
| `render_final.py` | render the fitted scene + labelled photo-vs-fit figure |

## Requirements
- An OpenGL 3.3 capable GPU (uses an offscreen FBO via GLFW).
- `pip install numpy scipy glfw PyOpenGL Pillow`

## Run
All scripts live in `program/` and write results to `../results/`:
```bash
cd program
python fit_photo_guided.py 200     # ~few minutes; builds target, GA-fits -> ../results/best_params.json, fit_vs_photo.png
python fit_procedural.py 200       # the no-guidance baseline for comparison (~0.13)
python render_final.py             # -> ../results/FINAL_photo_vs_fit.png
```
`fit_photo_guided.py` **warm-starts** from `../results/best_params.json` if present, so re-running keeps
refining. A pre-fitted `results/best_params.json` (RMSE 0.0512) ships with the package — delete it to fit
from scratch.

## One-paragraph method
The photo's per-pixel *cloudiness* (bright + whitish, in the sky region) is uploaded as a texture and,
in the shader, gates the cloud density per screen pixel: where the photo is blue sky the ray stays
empty; where the photo has cloud, a procedural 3-D volumetric cumulus (flat base, billowy top,
high-frequency detail, Paper-3-lit) is rendered and its opacity/brightness scaled by the cloudiness.
So the **photo places the clouds**, the **procedural model gives their 3-D shape/detail**, and a
**genetic algorithm** tunes 39 params (camera, sun, atmosphere, cloud shape/detail, colour) to
minimize the masked RMSE. Pure procedural (no photo) can only match statistics, not positions, so it
floors ~0.13; photo guidance reaches 0.051 (the floor for a smooth-sky + procedural-cloud render vs a
real photo — the rest is unmatchable fine texture).
```
