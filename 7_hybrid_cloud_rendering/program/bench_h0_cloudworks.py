"""CloudWorks-derived H0 optimization slices against the unchanged H0."""
from __future__ import print_function

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time

import glfw
import numpy as np
from OpenGL.GL import *
from PIL import Image

import bench_h0 as h0


HERE = os.path.dirname(os.path.abspath(__file__))
ARTICLE = os.path.dirname(HERE)
OUT = os.path.join(ARTICLE, "results", "h0_cloudworks")
SHADER = os.path.join(HERE, "sky_frag.glsl")


def definitions(enabled, shadow_t=0.10):
    result = dict(h0.PRODUCTION)
    result["CLOUDWORKS_SHADOW_BUDGET"] = 1 if enabled else 0
    result["CLOUDWORKS_SHADOW_T"] = shadow_t
    return result


def percentile(values, q):
    return float(np.percentile(np.asarray(values, np.float64), q))


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
    legacy_fragment = subprocess.check_output(
        ["git", "show", "82f569d:articles/7_hybrid_cloud_rendering/program/sky_frag.glsl"],
        cwd=os.path.dirname(os.path.dirname(ARTICLE)),
    ).decode("utf-8")
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
    window = glfw.create_window(256, 256, "research7-cloudworks", None, None)
    if window is None:
        raise RuntimeError("OpenGL context creation failed")
    glfw.make_context_current(window)
    vao = glGenVertexArrays(1)
    query = int(glGenQueries(1)[0])
    target = h0.Target(args.width, args.height)
    programs = {
        "legacy": h0.make_program(legacy_fragment, h0.PRODUCTION),
        "h0": h0.make_program(fragment, definitions(False)),
    }
    for threshold in (0.10, 0.20, 0.35):
        programs["cw%.2f" % threshold] = h0.make_program(
            fragment, definitions(True, threshold)
        )
    selected = h0.scenes()[:2] if args.smoke else h0.scenes()
    records = []
    h0_images = []
    for name, eye_values, target_values in selected:
        eye = np.asarray(eye_values, np.float32)
        look = np.asarray(target_values, np.float32)
        camera = h0.camera_basis(eye, look)
        for program in programs.values():
            h0.set_view(program, args.width, args.height, eye, camera)
        h0.warm_gpu(programs["cw0.20"], target, vao,
                    0.15 if args.smoke else 1.5)
        images, timings = {}, {}
        for key, program in programs.items():
            timings[key] = [h0.timed_draw(program, target, vao, query)
                            for _ in range(args.frames)]
            h0.draw_once(program, target, vao)
            glFinish()
            images[key] = h0.read_image(target)
        variants = {}
        for key in sorted(k for k in programs if k.startswith("cw")):
            variants[key] = {
                "median_ms": float(statistics.median(timings[key])),
                "p95_ms": percentile(timings[key], 95),
                "speedup": float(statistics.median(timings["h0"]) /
                                 statistics.median(timings[key])),
                "rmse_vs_h0": h0.rmse(images[key], images["h0"]),
            }
        record = {
            "scene": name,
            "legacy_median_ms": float(statistics.median(timings["legacy"])),
            "h0_median_ms": float(statistics.median(timings["h0"])),
            "macro_dispatch_speedup_vs_82f569d": float(
                statistics.median(timings["legacy"])/
                statistics.median(timings["h0"])
            ),
            "macro_dispatch_rmse_vs_82f569d": h0.rmse(
                images["h0"], images["legacy"]
            ),
            "variants": variants,
        }
        h0_images.append(images["h0"])
        records.append(record)
        print("%-22s legacy %7.3f -> H0v2 %7.3f ms x%.2f RMSE %.7f" % (
            name, record["legacy_median_ms"], record["h0_median_ms"],
            record["macro_dispatch_speedup_vs_82f569d"],
            record["macro_dispatch_rmse_vs_82f569d"]))
        for key, value in variants.items():
            print("  %-8s %7.3f ms x%.2f RMSE %.6f" % (
                key, value["median_ms"], value["speedup"],
                value["rmse_vs_h0"]))
    result = {
        "phase": "H0-v2",
        "experiment": "CloudWorks optical-depth shadow budget",
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": glGetString(GL_RENDERER).decode("utf-8", "replace"),
        "resolution": [args.width, args.height],
        "frames_per_scene": args.frames,
        "shader_sha256": shader_hash,
        "records": records,
        "thresholds": [0.10, 0.20, 0.35],
    }
    os.makedirs(OUT, exist_ok=True)
    if h0_images:
        montage = np.concatenate([
            (np.clip(image[..., :3], 0.0, 1.0)*255.0).astype(np.uint8)
            for image in h0_images
        ], axis=0)
        Image.fromarray(montage).save(os.path.join(OUT, "h0_v2_montage.png"))
    filename = "smoke.json" if args.smoke else "formal_1280x720.json"
    with open(os.path.join(OUT, filename), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    glfw.terminate()


if __name__ == "__main__":
    main()
