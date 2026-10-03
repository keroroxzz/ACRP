"""Phase-H2 full-camera-ray transmittance closure validation.

This transport-only test compares a 30 m midpoint reference with B1/B2/B4
macro-cell closures. Boundary cells always use the fine reference path.
"""
from __future__ import print_function

import argparse
import hashlib
import json
import os
import time

import glfw
import numpy as np
from OpenGL.GL import *

import bench_h0 as h0
import bench_h1 as h1


HERE = os.path.dirname(os.path.abspath(__file__))
ARTICLE = os.path.dirname(HERE)
OUT = os.path.join(ARTICLE, "results", "h2")
SHADER = os.path.join(HERE, "sky_frag.glsl")


def definitions(mode):
    base = dict(h0.REFERENCE)
    base.update(
        METHOD=2, INSTRUMENT=4, USE_MACRO_TEX=1, BUILD_BOUNDS=0,
        BUILD_CUMULANTS=0, H2_MODE=mode, MAXI=3000,
    )
    return base


def bind_textures(program, bounds, cumulants):
    h1.set_macro_uniforms(program, bounds)
    glUseProgram(program)
    glUniform1i(glGetUniformLocation(program, "uMacroCumulants"), 1)
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_3D, cumulants)
    glActiveTexture(GL_TEXTURE0)


def stats(estimate, reference, mask):
    absolute = np.abs(estimate-reference)[mask]
    relative = (np.abs(estimate-reference) / np.maximum(reference, 0.02))[mask]
    return {
        "pixels": int(mask.sum()),
        "mean_absolute_error": float(absolute.mean()) if absolute.size else None,
        "p95_absolute_error": (
            float(np.percentile(absolute, 95)) if absolute.size else None
        ),
        "mean_relative_error": float(relative.mean()) if relative.size else None,
        "p95_relative_error": (
            float(np.percentile(relative, 95)) if relative.size else None
        ),
        "max_relative_error": float(relative.max()) if relative.size else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args()
    if args.smoke:
        args.width, args.height = 160, 90

    fragment = open(SHADER, encoding="utf-8").read()
    shader_hash = hashlib.sha256(fragment.encode("utf-8")).hexdigest()
    if not glfw.init():
        raise RuntimeError("glfw.init failed")
    for key, value in (
        (glfw.VISIBLE, glfw.FALSE),
        (glfw.CONTEXT_VERSION_MAJOR, 3),
        (glfw.CONTEXT_VERSION_MINOR, 3),
        (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE),
    ):
        glfw.window_hint(key, value)
    window = glfw.create_window(256, 256, "research7-h2-ray", None, None)
    if window is None:
        raise RuntimeError("OpenGL context creation failed")
    glfw.make_context_current(window)
    vao = glGenVertexArrays(1)
    bounds, bounds_ms = h1.build_macro_texture(fragment, vao)
    cumulants, cumulants_ms = h1.build_macro_texture(
        fragment, vao, cumulants=True
    )
    programs = {mode: h0.make_program(fragment, definitions(mode))
                for mode in (0, 1, 2, 4)}
    target = h0.Target(args.width, args.height)
    os.makedirs(OUT, exist_ok=True)
    records = []
    selected = h0.scenes()[:2] if args.smoke else h0.scenes()
    for name, eye_values, target_values in selected:
        eye = np.asarray(eye_values, np.float32)
        look = np.asarray(target_values, np.float32)
        camera = h0.camera_basis(eye, look)
        images = {}
        for mode, program in programs.items():
            h0.set_view(program, args.width, args.height, eye, camera)
            bind_textures(program, bounds, cumulants)
            h0.draw_once(program, target, vao)
            glFinish()
            images[mode] = h0.read_image(target)
        reference = images[0][..., 0]
        cloud = reference < 0.99
        substantial = reference < 0.95
        analytic_length = images[2][..., 1]
        total_length = images[2][..., 2]
        record = {
            "scene": name,
            "cloud_pixels": int(cloud.sum()),
            "substantial_pixels": int(substantial.sum()),
            "mean_analytic_length_fraction": float(
                (analytic_length/np.maximum(total_length, 1e-6))[cloud].mean()
            ) if cloud.any() else 0.0,
            "b1": stats(images[1][..., 0], reference, substantial),
            "b2": stats(images[2][..., 0], reference, substantial),
            "b4": stats(images[4][..., 0], reference, substantial),
        }
        records.append(record)
        print(
            "%-22s pixels=%6d analytic=%6.2f%%  p95 rel B1=%7.4f "
            "B2=%7.4f B4=%7.4f" % (
                name, record["substantial_pixels"],
                100.0*record["mean_analytic_length_fraction"],
                record["b1"]["p95_relative_error"] or 0.0,
                record["b2"]["p95_relative_error"] or 0.0,
                record["b4"]["p95_relative_error"] or 0.0,
            )
        )
    result = {
        "phase": "H2",
        "experiment": "full-camera-ray synthetic-extinction closure",
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": glGetString(GL_RENDERER).decode("utf-8", "replace"),
        "resolution": [args.width, args.height],
        "shader_sha256": shader_hash,
        "reference_step_km": 0.03,
        "synthetic_extinction": "cloudDensity * SIGMA_T * DENS",
        "macro_build_ms": bounds_ms,
        "cumulant_build_ms": cumulants_ms,
        "records": records,
        "gates": {
            "analytic_length_at_least_10pct_all_scenes": all(
                r["mean_analytic_length_fraction"] >= 0.10 for r in records
            ),
            "b2_p95_under_2pct_all_scenes": all(
                r["b2"]["p95_relative_error"] is not None
                and r["b2"]["p95_relative_error"] < 0.02 for r in records
            ),
            "b4_p95_under_2pct_all_scenes": all(
                r["b4"]["p95_relative_error"] is not None
                and r["b4"]["p95_relative_error"] < 0.02 for r in records
            ),
            "b2_better_than_b1_all_scenes": all(
                r["b2"]["p95_relative_error"]
                < r["b1"]["p95_relative_error"] for r in records
            ),
        },
        "limitations": [
            "Uses synthetic uniform extinction, not deck-specific production optics.",
            "Validates camera-ray transmittance only, not in-scattering.",
            "Boundary and out-of-grid segments use the fine reference path.",
            "An error gate is not meaningful unless analytic coverage also passes.",
        ],
    }
    with open(os.path.join(OUT, "h2_ray_closure.json"), "w",
              encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    glfw.terminate()


if __name__ == "__main__":
    main()
