"""Single-photo ViT GA for the Research-7 cumulonimbus morphology.

Geometry is fixed from Paper 6's fixed_geo.json. Only Research-7 morphology
uniforms are optimized. Run with the dino environment:

  python optimize_h0_vit.py 20
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import glfw
import numpy as np
import torch
import torch.nn.functional as F
from OpenGL.GL import *
from PIL import Image
from scipy.ndimage import (
    binary_closing, binary_fill_holes, binary_opening, gaussian_filter, label
)
from transformers import AutoModel


HERE = os.path.dirname(os.path.abspath(__file__))
ARTICLE = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(ARTICLE))
OUT = os.path.join(ARTICLE, "results", "h0_vit")
SHADER = os.path.join(HERE, "sky_frag.glsl")   # self-contained Research-7 snapshot (migrated from a3/gpu/sky_frag.glsl)
PHOTO = os.path.join(REPO, "reference_photos", "IMG_20200531_141120.jpg")
CURATED_MASK = os.path.join(
    REPO, "reference_photos", "IMG_20200531_141120_mask.png"
)
DERIVED_MASK = os.path.join(OUT, "IMG_20200531_141120_mask_v2.png")
GEO_FILE = os.path.join(
    REPO, "articles", "6_vit_visual_scoring_ga", "program", "fixed_geo.json"
)

W, H = 512, 384
HEIGHT = 0.06
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VS = (
    "#version 330 core\n"
    "void main(){vec2 p=vec2(float((gl_VertexID<<1)&2),"
    "float(gl_VertexID&2));gl_Position=vec4(p*2.0-1.0,0.0,1.0);}\n"
)
DEFINES = {
    "METHOD": 1, "SUN_STEPS": 4, "SMALL_STEP": 0.075,
    "BIG_STEP": 0.34, "MAXI": 1200, "JITTER_ON": 1,
    "SUN_JITTER": 0.0, "INSTRUMENT": 0,
    "A3_CAMF": 6, "A3_CAMGL": 5, "A3_SUNF": 4, "A3_SUNGL": 4,
}

# Bounds were chosen to preserve a plausible convective tower while allowing
# the GA to discover tower/anvil stage and multiscale surface character.
PARAMS = [
    # uCbX / uCbZ are FIXED (pinned in Renderer.__init__), NOT searched: the GA kept moving the whole
    # storm out of frame to score a clean blue sky (uCbX hit its -7.86 bound). The GA now only shapes it.
    ("uCbLean", 0.02, 0.34),
    ("uCbBaseR", 0.60, 2.60),
    ("uCbTopR", 1.00, 3.80),
    ("uCbWarp", 0.10, 1.15),
    ("uCbDetailWarp", 0.04, 0.48),
    ("uCbMacroScale", 0.30, 0.78),
    ("uCbBillowScale", 0.72, 1.85),
    ("uCbRidge", 0.02, 0.20),
    ("uCbMicro", 0.005, 0.095),
    ("uCbAnvil", 1.8, 5.6),
    ("uCbAnvilDens", 0.15, 1.45),
    ("uCbCrown", 0.65, 3.20),
]
NAMES = [p[0] for p in PARAMS]
LO = np.asarray([p[1] for p in PARAMS], np.float32)
HI = np.asarray([p[2] for p in PARAMS], np.float32)
D = len(PARAMS)
DEFAULTS = np.asarray(
    [0.16, 1.0, 1.6, 0.72, 0.23, 0.48, 1.18,
     0.11, 0.045, 4.5, 1.05, 2.2],
    np.float32,
)
DEFAULT_X = (DEFAULTS - LO) / (HI - LO)


def inject(source):
    lines = source.split("\n")
    defs = "\n".join("#define %s %s" % item for item in DEFINES.items())
    return lines[0] + "\n" + defs + "\n" + "\n".join(lines[1:])


def shader(source, kind):
    obj = glCreateShader(kind)
    glShaderSource(obj, source)
    glCompileShader(obj)
    if glGetShaderiv(obj, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(obj).decode("utf-8", "replace"))
    return obj


def program(fragment):
    obj = glCreateProgram()
    glAttachShader(obj, shader(VS, GL_VERTEX_SHADER))
    glAttachShader(obj, shader(inject(fragment), GL_FRAGMENT_SHADER))
    glLinkProgram(obj)
    if glGetProgramiv(obj, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(obj).decode("utf-8", "replace"))
    return obj


def basis(eye, direction):
    f = direction / np.linalg.norm(direction)
    r = np.cross([0.0, 1.0, 0.0], f)
    r /= np.linalg.norm(r)
    u = np.cross(f, r)
    return np.stack([r, u, f], axis=1).astype(np.float32)


def decode(x):
    return LO + np.clip(x, 0.0, 1.0) * (HI - LO)


def load_inputs():
    photo = np.asarray(
        Image.open(PHOTO).convert("RGB").resize((W, H), Image.LANCZOS),
        np.float32,
    ) / 255.0
    mask_path = CURATED_MASK if os.path.exists(CURATED_MASK) else DERIVED_MASK
    if os.path.exists(mask_path):
        mask = np.asarray(
            Image.open(mask_path).convert("L").resize((W, H), Image.LANCZOS),
            np.float32,
        ) / 255.0
    else:
        # Paper-6 top-band connected-component skyline extraction. This target
        # predates the curated mask set, so keep its derived mask local to R7.
        r, g, b = photo[..., 0], photo[..., 1], photo[..., 2]
        bright = photo.mean(2)
        sat = photo.max(2) - photo.min(2)
        rows = np.arange(H)[:, None] * np.ones((1, W))
        cols = np.ones((H, 1)) * np.arange(W)[None, :]
        green = (g > r + 0.03) & (g > b + 0.03)
        building = (
            (rows > 0.54 * H) & (sat < 0.16)
            & (bright > 0.38) & (bright < 0.78)
        )
        candidate = (
            (bright > 0.34) & (~green) & (~building) & (rows < 0.63 * H)
        )
        candidate = binary_closing(
            binary_opening(candidate, np.ones((2, 2))), np.ones((3, 3))
        )
        components, _ = label(candidate)
        band = components[:max(3, int(0.05 * H))]
        top_labels = set(band[band > 0].tolist())
        sky = binary_fill_holes(binary_closing(
            np.isin(components, list(top_labels)), np.ones((5, 5))
        ))
        mask = gaussian_filter(sky.astype(np.float32), 2.0)
        os.makedirs(OUT, exist_ok=True)
        Image.fromarray((mask * 255).astype(np.uint8)).save(DERIVED_MASK)
    geo = json.load(open(GEO_FILE, encoding="utf-8"))["IMG_20200531_141120"]
    return photo, mask[:, :, None], {k: float(v) for k, v in geo.items()}


class Renderer:
    def __init__(self, fragment, geo):
        if not glfw.init():
            raise RuntimeError("glfw.init failed")
        for k, v in (
            (glfw.VISIBLE, glfw.FALSE),
            (glfw.CONTEXT_VERSION_MAJOR, 3),
            (glfw.CONTEXT_VERSION_MINOR, 3),
            (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE),
        ):
            glfw.window_hint(k, v)
        self.window = glfw.create_window(256, 256, "r7-vit", None, None)
        if self.window is None:
            raise RuntimeError("OpenGL context creation failed")
        glfw.make_context_current(self.window)
        self.program = program(fragment)
        self.vao = glGenVertexArrays(1)
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0,
            GL_RGBA, GL_FLOAT, None,
        )
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0
        )
        glViewport(0, 0, W, H)
        glUseProgram(self.program)
        uniform = lambda n: glGetUniformLocation(self.program, n)
        self.locations = {n: uniform(n) for n in NAMES}
        eye = np.asarray([0.0, HEIGHT, 0.0], np.float32)
        direction = np.asarray(
            [0.0, math.sin(geo["pitch"]), math.cos(geo["pitch"])],
            np.float32,
        )
        camera = basis(eye, direction)
        sun = np.asarray(
            [
                math.cos(geo["sunEl"]) * math.sin(geo["sunAz"]),
                math.sin(geo["sunEl"]),
                math.cos(geo["sunEl"]) * math.cos(geo["sunAz"]),
            ],
            np.float32,
        )
        glUniform2f(uniform("uRes"), float(W), float(H))
        glUniform3f(uniform("uEye"), *[float(v) for v in eye])
        glUniformMatrix3fv(uniform("uCam"), 1, GL_FALSE, camera.T.copy())
        focal = float(geo["focal"])
        glUniform1f(uniform("uFocal"), focal)
        glUniform3f(uniform("uSunDir"), *[float(v) for v in sun])
        # Fixed from the last non-gaming placement fit. Shape remains optimized,
        # but the storm cannot escape the frame to buy a clean-sky score.
        glUniform1f(uniform("uCbX"), -1.3)
        glUniform1f(uniform("uCbZ"), 5.4)
        self.geo = geo
        self.sun = sun.tolist()

    def render(self, x):
        glUseProgram(self.program)
        for name, value in zip(NAMES, decode(x)):
            glUniform1f(self.locations[name], float(value))
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glFinish()
        raw = glReadPixels(0, 0, W, H, GL_RGB, GL_FLOAT)
        return np.frombuffer(raw, np.float32).reshape(H, W, 3)[::-1].copy()


class DinoScore:
    def __init__(self, photo, mask):
        self.photo = photo
        self.mask = mask
        self.model = AutoModel.from_pretrained(
            "facebook/dinov2-base", local_files_only=True
        ).eval().to(DEVICE)
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406], device=DEVICE
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229, 0.224, 0.225], device=DEVICE
        ).view(1, 3, 1, 1)
        self.target = self.embed(photo)
        patches = self.target[1]
        grid = int(round(patches.shape[1] ** 0.5))
        w = torch.tensor(mask[:, :, 0], device=DEVICE)[None, None]
        w = F.interpolate(w, size=(grid, grid), mode="area")
        self.weights = w.reshape(1, -1, 1)
        self.weights /= self.weights.sum() + 1e-6
        self.target_style = self.style(patches)
        self.target_cloud = self.cloudiness(photo)

    @torch.no_grad()
    def embed(self, image):
        t = torch.tensor(
            np.ascontiguousarray(image), device=DEVICE, dtype=torch.float32
        ).permute(2, 0, 1)[None]
        t = F.interpolate(t, size=(224, 224), mode="bilinear",
                          align_corners=False)
        out = self.model(pixel_values=(t - self.mean) / self.std)
        return out.pooler_output, out.last_hidden_state[:, 1:]

    def style(self, patches):
        w = self.weights
        mean = (patches * w).sum(1)
        var = ((patches - mean[:, None]) ** 2 * w).sum(1)
        return torch.cat([mean, torch.sqrt(var + 1e-8)], dim=-1)

    def cloudiness(self, image):
        blue = np.clip((image[..., 2] - image[..., 0]) * 3.0, 0.0, 1.0)
        w = self.mask[..., 0]
        return float(((1.0 - blue) * w).sum() / max(w.sum(), 1e-6))

    @torch.no_grad()
    def score(self, render):
        comp = self.photo * (1.0 - self.mask) + render * self.mask
        cls, patches = self.embed(comp)
        sim = (
            0.40 * F.cosine_similarity(cls, self.target[0])
            + 0.60 * F.cosine_similarity(
                self.style(patches), self.target_style
            )
        ).item()
        amount_penalty = 0.25 * abs(
            self.cloudiness(comp) - self.target_cloud
        )
        return sim - amount_penalty, comp


def main():
    generations = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    os.makedirs(OUT, exist_ok=True)
    photo, mask, geo = load_inputs()
    fragment = open(SHADER, encoding="utf-8").read()
    renderer = Renderer(fragment, geo)
    scorer = DinoScore(photo, mask)
    rng = np.random.default_rng(7)
    population_size, elites = 24, 4
    seed = DEFAULT_X.copy()
    prior_path = os.path.join(OUT, "vit_best.json")
    if os.path.exists(prior_path):
        try:
            prior = json.load(open(prior_path, encoding="utf-8"))
            prior_params = prior.get("params", {})
            if all(name in prior_params for name in NAMES):
                values = np.asarray(
                    [prior_params[name] for name in NAMES], np.float32
                )
                seed = np.clip((values - LO) / (HI - LO), 0.0, 1.0)
                print("warm-start:", prior_path)
        except Exception as error:
            print("warm-start ignored:", error)
    population = np.clip(
        np.concatenate(
            [seed[None], seed[None] +
             0.16 * rng.standard_normal((population_size - 1, D))]
        ),
        0.0, 1.0,
    )
    best, best_score = DEFAULT_X.copy(), -1e9

    def evaluate(x):
        score, comp = scorer.score(renderer.render(x))
        return score, comp

    def save(x, score, comp):
        render = renderer.render(x)
        Image.fromarray(
            (np.clip(np.concatenate([photo, comp], axis=1), 0, 1) * 255)
            .astype(np.uint8)
        ).save(os.path.join(OUT, "vit_vs_photo.png"))
        Image.fromarray(
            (np.clip(render, 0, 1) * 255).astype(np.uint8)
        ).save(os.path.join(OUT, "vit_render.png"))
        values = decode(x)
        result = {
            "fitness": float(score),
            "photo": os.path.relpath(PHOTO, REPO).replace("\\", "/"),
            "mask": os.path.relpath(
                CURATED_MASK if os.path.exists(CURATED_MASK) else DERIVED_MASK,
                REPO,
            ).replace("\\", "/"),
            "geometry": geo,
            "sun_direction_recorded": renderer.sun,
            "shader_sha256": hashlib.sha256(
                fragment.encode("utf-8")
            ).hexdigest(),
            "params": {n: float(v) for n, v in zip(NAMES, values)},
            "x": x.tolist(),
            "note": "Camera and sun geometry fixed from Paper 6 fixed_geo.json.",
        }
        with open(os.path.join(OUT, "vit_best.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

    for generation in range(generations):
        fitness = []
        composites = []
        for candidate in population:
            score, comp = evaluate(candidate)
            fitness.append(score)
            composites.append(comp)
        order = np.argsort(-np.asarray(fitness))
        population = population[order]
        fitness = np.asarray(fitness)[order]
        composites = [composites[i] for i in order]
        if fitness[0] > best_score:
            best_score = float(fitness[0])
            best = population[0].copy()
            save(best, best_score, composites[0])
        print("generation %3d  best %.5f  global %.5f" %
              (generation, fitness[0], best_score))
        sigma = 0.12 * (1.0 - 0.65 * generation /
                        max(generations - 1, 1))
        next_population = [population[i].copy() for i in range(elites)]
        while len(next_population) < population_size:
            a, b = rng.integers(0, population_size // 2, 2)
            parent = population[a] if fitness[a] > fitness[b] else population[b]
            child = np.clip(
                parent + (rng.random(D) < 0.65) * sigma *
                rng.standard_normal(D),
                0.0, 1.0,
            )
            next_population.append(child)
        population = np.asarray(next_population)
    score, comp = evaluate(best)
    save(best, score, comp)
    print("FINAL %.5f -> %s" % (score, os.path.join(OUT, "vit_best.json")))
    glfw.terminate()


if __name__ == "__main__":
    main()
