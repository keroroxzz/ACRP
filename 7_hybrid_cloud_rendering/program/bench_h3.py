"""Phase-H3 transition-local boundary refinement benchmark."""
from __future__ import print_function

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
import bench_h1 as h1


HERE = os.path.dirname(os.path.abspath(__file__))
ARTICLE = os.path.dirname(HERE)
OUT = os.path.join(ARTICLE, "results", "h3")
SHADER = os.path.join(HERE, "sky_frag.glsl")


def definitions(refine, instrument=0):
    result = h1.definitions(2, instrument=instrument)
    result["H3_BOUNDARY_REFINE"] = int(refine)
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
    window = glfw.create_window(256, 256, "research7-h3", None, None)
    if window is None:
        raise RuntimeError("OpenGL context creation failed")
    glfw.make_context_current(window)
    vao = glGenVertexArrays(1)
    macro, build_ms = h1.build_macro_texture(fragment, vao)
    programs = {
        "h1": h0.make_program(fragment, definitions(0)),
        "h3": h0.make_program(fragment, definitions(2)),
        "h0": h0.make_program(fragment, h1.definitions(1)),
        "h1_stats": h0.make_program(fragment, definitions(0, 1)),
        "h3_stats": h0.make_program(fragment, definitions(2, 1)),
    }
    target = h0.Target(args.width, args.height)
    query = int(glGenQueries(1)[0])
    selected = h0.scenes()[:2] if args.smoke else h0.scenes()
    records = []
    for name, eye_values, target_values in selected:
        eye = np.asarray(eye_values, np.float32)
        look = np.asarray(target_values, np.float32)
        camera = h0.camera_basis(eye, look)
        for program in programs.values():
            h0.set_view(program, args.width, args.height, eye, camera)
        for key in ("h1", "h3", "h1_stats", "h3_stats"):
            h1.set_macro_uniforms(programs[key], macro)
        h0.warm_gpu(programs["h3"], target, vao, 0.15 if args.smoke else 1.5)
        timing = {}
        images = {}
        for key in ("h1", "h3"):
            timing[key] = [
                h0.timed_draw(programs[key], target, vao, query)
                for _ in range(args.frames)
            ]
            h0.draw_once(programs[key], target, vao)
            glFinish()
            images[key] = h0.read_image(target)
        h0.draw_once(programs["h0"], target, vao)
        glFinish()
        images["h0"] = h0.read_image(target)
        counters = {}
        for key in ("h1_stats", "h3_stats"):
            h0.draw_once(programs[key], target, vao)
            glFinish()
            counters[key] = h0.read_image(target)
        h1_evals = float(counters["h1_stats"][..., 0].mean())
        h3_evals = float(counters["h3_stats"][..., 0].mean())
        record = {
            "scene": name,
            "h1_median_ms": float(statistics.median(timing["h1"])),
            "h3_median_ms": float(statistics.median(timing["h3"])),
            "h3_p95_ms": percentile(timing["h3"], 95),
            "h1_rmse_vs_h0": h0.rmse(images["h1"], images["h0"]),
            "h3_rmse_vs_h0": h0.rmse(images["h3"], images["h0"]),
            "h1_mean_density_evals": h1_evals,
            "h3_mean_density_evals": h3_evals,
            "h3_eval_multiplier_vs_h1": h3_evals/max(h1_evals, 1e-6),
        }
        records.append(record)
        comparison = np.concatenate(
            [
                h0.label(images["h0"], name + " H0 production"),
                h0.label(images["h1"], name + " H1 uniform 180 m"),
                h0.label(
                    images["h3"],
                    name + " H3 entry budget 2  RMSE %.4f" %
                    record["h3_rmse_vs_h0"],
                ),
            ],
            axis=1,
        )
        os.makedirs(OUT, exist_ok=True)
        Image.fromarray(comparison).save(
            os.path.join(OUT, name + "_comparison.png")
        )
        print(
            "%-22s H1/H3 %7.3f/%7.3f ms eval x%.2f  RMSE %.4f -> %.4f" %
            (name, record["h1_median_ms"], record["h3_median_ms"],
             record["h3_eval_multiplier_vs_h1"], record["h1_rmse_vs_h0"],
             record["h3_rmse_vs_h0"])
        )
    result = {
        "phase": "H3",
        "experiment": "presence-guided single-segment entry-exit refinement",
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": glGetString(GL_RENDERER).decode("utf-8", "replace"),
        "resolution": [args.width, args.height],
        "frames_per_scene": args.frames,
        "shader_sha256": shader_hash,
        "macro_build_ms": build_ms,
        "records": records,
        "gates": {
            "rmse_nonregression_all_scenes": all(
                r["h3_rmse_vs_h0"] <= r["h1_rmse_vs_h0"]+1e-7
                for r in records
            ),
            "rmse_improves_at_least_five_scenes": sum(
                r["h3_rmse_vs_h0"] < r["h1_rmse_vs_h0"]-1e-7
                for r in records
            ) >= min(5, len(records)),
            "density_eval_overhead_under_25pct": all(
                r["h3_eval_multiplier_vs_h1"] <= 1.25 for r in records
            ),
            "gpu_time_overhead_under_25pct": all(
                r["h3_median_ms"] <= 1.25*r["h1_median_ms"]
                for r in records
            ),
        },
    }
    os.makedirs(OUT, exist_ok=True)
    filename = "h3_smoke.json" if args.smoke else "h3_1280x720.json"
    with open(os.path.join(OUT, filename), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    glfw.terminate()


if __name__ == "__main__":
    main()
