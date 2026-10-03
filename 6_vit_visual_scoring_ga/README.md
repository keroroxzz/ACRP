# Paper 6 — ViT-based Visual Scoring Genetic Algorithm (reproduction package)

Optimize a procedural cloud shader with a **Vision-Transformer perceptual score** (DINOv2) + a
**genetic algorithm**, with a **sky-compositing** step that focuses the score on the sky.
Full write-up: **[Paper6_vit_visual_scoring_ga.md](Paper6_vit_visual_scoring_ga.md)**.

Key result: compositing the render's sky into the real photo and scoring with DINOv2 drives the GA to a
procedural sky that blends seamlessly into the photo — perceptual similarity 0.963 → **0.980**
(`results/dino_comp_vs_photo.png`).

**§6 extension (multi-photo):** the pipeline is extended to a **12-photo set** with a rewritten shader
(physical sky + 3 cloud decks + domain-warp + atmosphere-coupled cloud lighting), fixed per-photo
geometry, mean+min fitness, and **dual DINOv2+DINOv3 multi-depth** scoring. Headline finding: a bigger
ViT is *not* a better aesthetic judge — DINOv3 alone drifts to hazy washes (saturated CLS). See
`results/multi_vs_photo_dual.png` and report §6.

## Layout
- `program/` — all code + the target photo (`reference_photo.jpg`); scripts write to `../results`.
- `results/` — generated figures, composites, best params, gradcam heatmaps.
- `Paper6_vit_visual_scoring_ga.md` — the report (method, results, discussion).

| `program/` file | what |
|---|---|
| `cumulus_beauty.glsl` | uniform-driven coherent procedural cumulus (Paper-3 sky + billow/ridged) |
| `dino_perceptual.py` | DINOv2 perceptual similarity between two images |
| `dino_gradcam.py` | GradCAM error-attribution heatmap of the perceptual score |
| `optimize_dino.py` | ViT-scored GA (floating sky) |
| `optimize_dino_composite.py` | **ViT-scored GA + sky compositing** (the method) |
| `render_beauty.py` | render the beauty shader standalone |
| `final_render.py` | high-res hero render from the best params |
| `cumulus_multilayer.glsl` | **§6** multi-photo shader: physical sky + 3 decks (cumulus/cirrocumulus/cirrus) + domain-warp + atmosphere-coupled cloud lighting |
| `optimize_multiphoto.py` | **§6** multi-photo joint GA: fixed per-photo geometry, mean+min fitness, dual DINOv2+v3 multi-depth scoring |

## Requirements
Run in a conda env with **torch + transformers** (DINOv2) **and** an OpenGL 3.3 GPU:
```
pip install glfw PyOpenGL scipy pillow            # into your torch env
# DINOv2-base downloads from HF on first use (or is cached). DINOv3 is gated (needs HF auth).
```

## Run
All scripts live in `program/` and write results to `../results/`:
```
cd program
python optimize_dino_composite.py 32              # the main method -> ../results/dino_comp_vs_photo.png, dino_comp_best.json
python optimize_dino.py 30                        # variant without compositing (floating sky) -> ../results/dino_best.png
python dino_gradcam.py ../results/dino_best.png   # error-attribution heatmap
python render_beauty.py 0.19 3.2 0.12 1.2 1.05 0.6 hero   # standalone render of the shader
python optimize_multiphoto.py 100 4                       # §6 multi-photo joint GA (12 photos, dual DINOv2+v3)
```
The §6 multi-photo run additionally needs the gated DINOv3 weights
(`facebook/dinov3-vitl16-pretrain-lvd1689m`) in `research/models/` (git-ignored) and the 12
`research/reference_photos/*.jpg` + `*_mask.png`. See report §6 for the method and honest limitations
(cloud form less solid than the single-photo hero; sunsets over-purple; daytime skies milky).

## One-paragraph method
RMSE is the wrong objective for "looks right." Score renders with a pretrained ViT (DINOv2: global CLS +
translation-invariant patch statistics) and let a genetic algorithm maximize it. GradCAM on the score
localizes the deficit (it flagged "empty sky"), which the GA then fixes on its own (raising cloud
coverage). To focus the ViT on the sky, composite the render's sky into the *real photo* (keep the
photo's ground) before scoring — the images then differ only in the sky, so the score isolates sky style
and detail, and the result blends into the real scene.
