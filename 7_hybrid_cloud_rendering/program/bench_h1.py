"""Phase-H1 active-interval benchmark and debug-view generator.

This evaluates the existing METHOD=2 coarse-to-fine traversal against the
frozen H0 production/reference definitions. It is deliberately diagnostic:
failure of the H1 gates is recorded rather than hidden by favorable scenes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time

import glfw
import numpy as np
from OpenGL.GL import *
from PIL import Image

import bench_h0 as h0


HERE = os.path.dirname(os.path.abspath(__file__))
ARTICLE = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(ARTICLE))
OUT = os.path.join(ARTICLE, "results", "h1")
SHADER = os.path.join(HERE, "sky_frag.glsl")
GRID_DIMS = (96, 16, 128)  # x, y, z
GRID_ORIGIN = (-48.0, 0.5, -32.0)
GRID_CELL = (1.0, 0.46875, 1.0)
MACRO_MARGIN = 0.002


def definitions(method, instrument=0, reference=False):
    base = dict(h0.REFERENCE if reference else h0.PRODUCTION)
    base.update(
        METHOD=method, INSTRUMENT=instrument,
        USE_MACRO_TEX=1 if method == 2 else 0,
        BUILD_BOUNDS=0,
    )
    return base


def build_definitions(cumulants=False):
    base = dict(h0.PRODUCTION)
    base.update(
        METHOD=2, INSTRUMENT=0, USE_MACRO_TEX=0, BUILD_BOUNDS=1,
        BUILD_CUMULANTS=1 if cumulants else 0,
    )
    return base


def set_cloud_defaults(program):
    glUseProgram(program)
    for name, value in h0.CB_DEFAULTS.items():
        glUniform1f(glGetUniformLocation(program, name), float(value))


def set_macro_uniforms(program, texture):
    glUseProgram(program)
    glUniform3f(
        glGetUniformLocation(program, "uMacroOrigin"), *GRID_ORIGIN
    )
    glUniform3f(
        glGetUniformLocation(program, "uMacroCellSize"), *GRID_CELL
    )
    glUniform3f(
        glGetUniformLocation(program, "uMacroDims"),
        *[float(v) for v in GRID_DIMS],
    )
    glUniform1f(
        glGetUniformLocation(program, "uMacroMargin"), MACRO_MARGIN
    )
    glUniform1i(glGetUniformLocation(program, "uMacroBounds"), 0)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_3D, texture)


def build_macro_texture(fragment, vao, closure_k=1.0, cumulants=False):
    nx, ny, nz = GRID_DIMS
    texture = glGenTextures(1)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_3D, texture)
    glTexImage3D(
        GL_TEXTURE_3D, 0, GL_RGBA32F, nx, nz, ny, 0,
        GL_RGBA, GL_FLOAT, None,
    )
    for pname in (
        GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER,
    ):
        glTexParameteri(GL_TEXTURE_3D, pname, GL_NEAREST)
    for pname in (
        GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, GL_TEXTURE_WRAP_R,
    ):
        glTexParameteri(GL_TEXTURE_3D, pname, GL_CLAMP_TO_EDGE)

    builder = h0.make_program(fragment, build_definitions(cumulants))
    set_cloud_defaults(builder)
    glUniform3f(
        glGetUniformLocation(builder, "uMacroOrigin"), *GRID_ORIGIN
    )
    glUniform3f(
        glGetUniformLocation(builder, "uMacroCellSize"), *GRID_CELL
    )
    glUniform3f(
        glGetUniformLocation(builder, "uMacroDims"),
        *[float(v) for v in GRID_DIMS],
    )
    glUniform1f(
        glGetUniformLocation(builder, "uClosureK"), float(closure_k)
    )
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glUseProgram(builder)
    glBindVertexArray(vao)
    glViewport(0, 0, nx, nz)
    query = int(glGenQueries(1)[0])
    gpu_ms = 0.0
    for layer in range(ny):
        glFramebufferTextureLayer(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture, 0, layer
        )
        if glCheckFramebufferStatus(
            GL_FRAMEBUFFER
        ) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("macro-bound framebuffer incomplete")
        glUniform1f(
            glGetUniformLocation(builder, "uBoundLayer"), float(layer)
        )
        glBeginQuery(GL_TIME_ELAPSED, query)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glEndQuery(GL_TIME_ELAPSED)
        while glGetQueryObjectiv(
            query, GL_QUERY_RESULT_AVAILABLE
        ) == GL_FALSE:
            pass
        gpu_ms += float(glGetQueryObjectuiv(
            query, GL_QUERY_RESULT
        )) / 1.0e6
    return texture, gpu_ms


def draw(program, target, vao):
    glUseProgram(program)
    glBindFramebuffer(GL_FRAMEBUFFER, target.fbo)
    glViewport(0, 0, target.width, target.height)
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, 3)


def read_rgba(target):
    glBindFramebuffer(GL_FRAMEBUFFER, target.fbo)
    raw = glReadPixels(
        0, 0, target.width, target.height, GL_RGBA, GL_FLOAT
    )
    return np.frombuffer(raw, np.float32).reshape(
        target.height, target.width, 4
    )[::-1].copy()


def percentile(values, q):
    return float(np.percentile(np.asarray(values, np.float64), q))


def heat(values, scale=None):
    values = np.maximum(values, 0.0)
    if scale is None:
        positive = values[values > 0.0]
        scale = percentile(positive, 95) if positive.size else 1.0
    x = np.clip(values / max(scale, 1e-6), 0.0, 1.0)
    rgb = np.stack(
        [
            np.clip(1.8 * x, 0.0, 1.0),
            np.clip(1.8 - np.abs(4.0 * x - 2.0), 0.0, 1.0),
            np.clip(1.6 * (1.0 - x), 0.0, 1.0),
        ],
        axis=-1,
    )
    return (rgb * 255.0).astype(np.uint8)


def class_map(intervals):
    image = np.zeros(intervals.shape + (3,), np.uint8)
    image[intervals > 0.0] = (230, 70, 45)  # march
    image[intervals > 4.0] = (255, 210, 40)  # over-budget
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=21)
    args = parser.parse_args()
    if args.smoke:
        args.width, args.height, args.frames = 320, 180, 3

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
    window = glfw.create_window(256, 256, "research7-h1", None, None)
    if window is None:
        raise RuntimeError("OpenGL context creation failed")
    glfw.make_context_current(window)

    vao = glGenVertexArrays(1)
    macro_texture, macro_build_ms = build_macro_texture(fragment, vao)
    print("macro bound build: %.3f ms" % macro_build_ms)
    render = h0.make_program(fragment, definitions(2))
    render_h0 = h0.make_program(fragment, definitions(1))
    stats = h0.make_program(fragment, definitions(2, instrument=1))
    stats_h0 = h0.make_program(fragment, definitions(1, instrument=1))
    alpha_h1 = h0.make_program(fragment, definitions(2, instrument=2))
    support_h1 = h0.make_program(fragment, definitions(2, instrument=3))
    alpha_ref = h0.make_program(
        fragment, definitions(1, instrument=2, reference=True)
    )
    target = h0.Target(args.width, args.height)
    query = int(glGenQueries(1)[0])
    os.makedirs(OUT, exist_ok=True)

    records = []
    selected = h0.scenes()[:2] if args.smoke else h0.scenes()
    for name, eye_values, target_values in selected:
        eye = np.asarray(eye_values, np.float32)
        look = np.asarray(target_values, np.float32)
        camera = h0.camera_basis(eye, look)
        for program in (
            render, render_h0, stats, stats_h0, alpha_h1, alpha_ref
            , support_h1
        ):
            h0.set_view(program, args.width, args.height, eye, camera)
        for program in (render, stats, alpha_h1, support_h1):
            set_macro_uniforms(program, macro_texture)

        h0.warm_gpu(render, target, vao, 0.15 if args.smoke else 1.5)
        samples = [
            h0.timed_draw(render, target, vao, query)
            for _ in range(args.frames)
        ]
        draw(render, target, vao)
        glFinish()
        h1_image = read_rgba(target)
        draw(render_h0, target, vao)
        glFinish()
        h0_image = read_rgba(target)
        draw(stats, target, vao)
        glFinish()
        counters = read_rgba(target)
        draw(stats_h0, target, vao)
        glFinish()
        h0_counters = read_rgba(target)
        draw(alpha_h1, target, vao)
        glFinish()
        h1_a = read_rgba(target)[..., 0]
        draw(alpha_ref, target, vao)
        glFinish()
        ref_a = read_rgba(target)[..., 0]
        draw(support_h1, target, vao)
        glFinish()
        macro_hit = read_rgba(target)[..., 0] > 0.5

        cloud = ref_a > 0.01
        alpha_missed = cloud & (h1_a <= 0.001)
        missed = cloud & (~macro_hit)
        intervals = counters[..., 1]
        active_length = counters[..., 2]
        evaluations = counters[..., 0]
        total_evaluations = counters[..., 3]
        h0_evaluations = h0_counters[..., 0]
        h0_total_evaluations = h0_counters[..., 3]
        active_values = intervals[intervals > 0.0]
        record = {
            "scene": name,
            "gpu_ms_samples": samples,
            "median_ms": float(statistics.median(samples)),
            "p95_ms": percentile(samples, 95),
            "rmse_vs_h0_production": h0.rmse(h1_image, h0_image),
            "mean_evaluations_per_ray": float(evaluations.mean()),
            "p95_evaluations_per_ray": percentile(evaluations, 95),
            "mean_total_procedural_evaluations_per_ray": float(
                total_evaluations.mean()
            ),
            "p95_total_procedural_evaluations_per_ray": percentile(
                total_evaluations, 95
            ),
            "h0_mean_evaluations_per_ray": float(h0_evaluations.mean()),
            "h0_p95_evaluations_per_ray": percentile(h0_evaluations, 95),
            "h0_mean_total_procedural_evaluations_per_ray": float(
                h0_total_evaluations.mean()
            ),
            "evaluation_reduction_vs_h0": float(
                h0_evaluations.mean() / max(evaluations.mean(), 1e-6)
            ),
            "mean_active_intervals_per_ray": float(intervals.mean()),
            "p95_active_intervals_per_ray": percentile(intervals, 95),
            "mean_active_length": float(active_length.mean()),
            "p95_active_length": percentile(active_length, 95),
            "active_ray_fraction": float((intervals > 0.0).mean()),
            "missed_support_pixels": int(missed.sum()),
            "alpha_missed_pixels": int(alpha_missed.sum()),
            "reference_support_pixels": int(cloud.sum()),
            "false_empty_rate": float(
                missed.sum() / max(int(cloud.sum()), 1)
            ),
            "max_active_intervals": float(
                active_values.max() if active_values.size else 0.0
            ),
        }
        records.append(record)
        Image.fromarray(class_map(intervals)).save(
            os.path.join(OUT, name + "_classes.png")
        )
        Image.fromarray(heat(active_length)).save(
            os.path.join(OUT, name + "_active_length.png")
        )
        Image.fromarray(heat(evaluations)).save(
            os.path.join(OUT, name + "_evaluations.png")
        )
        Image.fromarray((missed.astype(np.uint8) * 255)).save(
            os.path.join(OUT, name + "_missed_support.png")
        )
        Image.fromarray((alpha_missed.astype(np.uint8) * 255)).save(
            os.path.join(OUT, name + "_alpha_missed.png")
        )
        comparison = np.concatenate(
            [
                h0.label(h0_image, name + " H0 production"),
                h0.label(
                    h1_image,
                    name + " H1  RMSE %.4f" %
                    record["rmse_vs_h0_production"],
                ),
            ],
            axis=1,
        )
        Image.fromarray(comparison).save(
            os.path.join(OUT, name + "_comparison.png")
        )
        print(
            "%-22s %8.3f ms  eval %.1f/p95 %.1f (x%.2f)  "
            "interval %.2f/p95 %.1f  missed %d" %
            (
                name, record["median_ms"],
                record["mean_evaluations_per_ray"],
                record["p95_evaluations_per_ray"],
                record["evaluation_reduction_vs_h0"],
                record["mean_active_intervals_per_ray"],
                record["p95_active_intervals_per_ray"],
                record["missed_support_pixels"],
            )
        )

    result = {
        "phase": "H1",
        "prototype": "METHOD=2 coarse-presence to fine-DDA",
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": glGetString(GL_RENDERER).decode("utf-8", "replace"),
        "resolution": [args.width, args.height],
        "frames_per_scene": args.frames,
        "shader_sha256": shader_hash,
        "macro_grid": {
            "dimensions_xyz": list(GRID_DIMS),
            "origin_xyz": list(GRID_ORIGIN),
            "cell_size_xyz": list(GRID_CELL),
            "samples_per_cell": 64,
            "gpu_build_ms": macro_build_ms,
        },
        "h1_defines": definitions(2),
        "gates": {
            "zero_missed_support": all(
                r["missed_support_pixels"] == 0 for r in records
            ),
            "p95_intervals_at_most_4": all(
                r["p95_active_intervals_per_ray"] <= 4.0 for r in records
            ),
            "density_evaluation_reduction_2x": all(
                r["evaluation_reduction_vs_h0"] >= 2.0 for r in records
            ),
            "grazing_stable": (
                next(
                    (
                        r["missed_support_pixels"] == 0
                        and r["p95_active_intervals_per_ray"] <= 4.0
                        for r in records if r["scene"] == "grazing"
                    ),
                    None,
                )
            ),
            "macro_build_under_1_5_ms": macro_build_ms <= 1.5,
        },
        "scenes": records,
    }
    suffix = "smoke" if args.smoke else "%dx%d" % (
        args.width, args.height
    )
    path = os.path.join(OUT, "h1_%s.json" % suffix)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print("Results:", path)
    glfw.terminate()


if __name__ == "__main__":
    main()
