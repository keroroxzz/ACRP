"""Research 7 Phase H0: production raymarch baseline benchmark.

This first harness deliberately reuses the live real-sky shader rather than
forking it prematurely. Every result records the shader SHA-256. Once H0 is
stable, the accepted baseline will be snapshotted into this article.

Run from the repository root with the Python 3.7 OpenGL environment:

    python articles/7_hybrid_cloud_rendering/program/bench_h0.py

Use --smoke for a fast compile/render/timing validation.
"""
from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import statistics
import time

import glfw
import numpy as np
from OpenGL.GL import *
from PIL import Image, ImageDraw


HERE = os.path.dirname(os.path.abspath(__file__))
ARTICLE = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(ARTICLE))
SHADER = os.path.join(HERE, "sky_frag.glsl")   # self-contained Research-7 snapshot (migrated from a3/gpu/sky_frag.glsl)
OUT = os.path.join(ARTICLE, "results", "h0")

VS = (
    "#version 330 core\n"
    "void main(){vec2 p=vec2(float((gl_VertexID<<1)&2),"
    "float(gl_VertexID&2));gl_Position=vec4(p*2.0-1.0,0.0,1.0);}\n"
)

# Production baseline and clean reference follow the established real-sky gate.
COMMON_A3 = dict(A3_CAMF=6, A3_CAMGL=5, A3_SUNF=4, A3_SUNGL=4)
REFERENCE = dict(
    METHOD=1, SUN_STEPS=12, SMALL_STEP=0.03, BIG_STEP=0.25,
    MAXI=2400, JITTER_ON=0, SUN_JITTER=0.0, INSTRUMENT=0, **COMMON_A3
)
PRODUCTION = dict(
    METHOD=1, SUN_STEPS=6, SMALL_STEP=0.06, BIG_STEP=0.30,
    MAXI=1400, JITTER_ON=1, SUN_JITTER=0.0, INSTRUMENT=0, **COMMON_A3
)
CB_DEFAULTS = {
    "uCbX": 6.0, "uCbZ": 9.0,
    "uCbLean": 0.16, "uCbBaseR": 0.62, "uCbTopR": 1.18,
    "uCbWarp": 0.72, "uCbDetailWarp": 0.23,
    "uCbMacroScale": 0.48, "uCbBillowScale": 1.18,
    "uCbRidge": 0.11, "uCbMicro": 0.045,
    "uCbAnvil": 4.5, "uCbAnvilDens": 1.05, "uCbCrown": 2.2,
}
SUN_DEFAULT = (0.60, 0.62, 0.50)


def percentile(values, q):
    a = np.asarray(values, dtype=np.float64)
    return float(np.percentile(a, q))


def inject(source, defines):
    lines = source.split("\n")
    header = "\n".join("#define %s %s" % item for item in defines.items())
    return lines[0] + "\n" + header + "\n" + "\n".join(lines[1:])


def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader).decode("utf-8", "replace"))
    return shader


def make_program(fragment_source, defines):
    program = glCreateProgram()
    glAttachShader(program, compile_shader(VS, GL_VERTEX_SHADER))
    glAttachShader(
        program,
        compile_shader(inject(fragment_source, defines), GL_FRAGMENT_SHADER),
    )
    glLinkProgram(program)
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(program).decode("utf-8", "replace"))
    return program


def camera_basis(eye, target):
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross([0.0, 1.0, 0.0], forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return np.stack([right, up, forward], axis=1).astype(np.float32)


def scenes():
    """H0 stress proxies supported by the current upstream scene."""
    return [
        ("sparse_wide", (-10.0, 3.2, -7.0), (6.0, 4.3, 9.0)),
        ("low_up", (-2.0, 1.0, -2.0), (6.0, 5.0, 9.0)),
        ("near_tower", (2.0, 3.5, 2.0), (6.0, 5.5, 9.0)),
        ("inside_proxy", (5.8, 3.5, 8.7), (6.0, 5.2, 9.2)),
        ("high_overcast_proxy", (-8.0, 6.5, -6.0), (6.0, 4.0, 9.0)),
        ("grazing", (-14.0, 1.1, -10.0), (6.0, 1.5, 9.0)),
    ]


class Target(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0,
            GL_RGBA, GL_FLOAT, None,
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
            self.texture, 0,
        )
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("incomplete framebuffer")

    def bind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)


def set_view(program, width, height, eye, camera):
    glUseProgram(program)
    glUniform2f(glGetUniformLocation(program, "uRes"), float(width), float(height))
    glUniform3f(
        glGetUniformLocation(program, "uEye"),
        float(eye[0]), float(eye[1]), float(eye[2]),
    )
    glUniform3f(
        glGetUniformLocation(program, "uSunDir"),
        float(SUN_DEFAULT[0]), float(SUN_DEFAULT[1]), float(SUN_DEFAULT[2]),
    )
    glUniform1f(glGetUniformLocation(program, "uFocal"), 1.25)
    glUniformMatrix3fv(
        glGetUniformLocation(program, "uCam"),
        1, GL_FALSE, camera.T.copy(),
    )
    for name, value in CB_DEFAULTS.items():
        glUniform1f(glGetUniformLocation(program, name), float(value))


def draw_once(program, target, vao, query=None):
    target.bind()
    glUseProgram(program)
    glBindVertexArray(vao)
    if query is not None:
        glBeginQuery(GL_TIME_ELAPSED, query)
    glDrawArrays(GL_TRIANGLES, 0, 3)
    if query is not None:
        glEndQuery(GL_TIME_ELAPSED)


def warm_gpu(program, target, vao, seconds):
    start = time.time()
    while time.time() - start < seconds:
        for _ in range(20):
            draw_once(program, target, vao)
        glFinish()


def timed_draw(program, target, vao, query):
    draw_once(program, target, vao, query)
    while glGetQueryObjectiv(query, GL_QUERY_RESULT_AVAILABLE) == GL_FALSE:
        pass
    return float(glGetQueryObjectuiv(query, GL_QUERY_RESULT)) / 1.0e6


def read_image(target):
    target.bind()
    raw = glReadPixels(
        0, 0, target.width, target.height, GL_RGBA, GL_FLOAT
    )
    image = np.frombuffer(raw, np.float32).reshape(
        target.height, target.width, 4
    )
    return image[::-1].copy()


def rmse(a, b):
    return float(np.sqrt(np.mean((a[:, :, :3] - b[:, :, :3]) ** 2)))


def label(image, text):
    rgb = (np.clip(image[:, :, :3], 0.0, 1.0) * 255.0).astype(np.uint8)
    result = Image.fromarray(rgb)
    painter = ImageDraw.Draw(result)
    painter.rectangle((0, 0, min(result.width, 720), 27), fill=(0, 0, 0))
    painter.text((8, 7), text, fill=(255, 255, 120))
    return np.asarray(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=21)
    args = parser.parse_args()
    if args.smoke:
        args.width, args.height, args.frames = 320, 180, 3

    with open(SHADER, "r", encoding="utf-8") as handle:
        fragment_source = handle.read()
    shader_sha256 = hashlib.sha256(
        fragment_source.encode("utf-8")
    ).hexdigest()

    if not glfw.init():
        raise RuntimeError("glfw.init() failed")
    for key, value in (
        (glfw.VISIBLE, glfw.FALSE),
        (glfw.CONTEXT_VERSION_MAJOR, 3),
        (glfw.CONTEXT_VERSION_MINOR, 3),
        (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE),
    ):
        glfw.window_hint(key, value)
    window = glfw.create_window(256, 256, "research7-h0", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("failed to create OpenGL 3.3 context")
    glfw.make_context_current(window)

    gpu = glGetString(GL_RENDERER).decode("utf-8", "replace")
    driver = glGetString(GL_VERSION).decode("utf-8", "replace")
    production = make_program(fragment_source, PRODUCTION)
    reference = make_program(fragment_source, REFERENCE)
    target = Target(args.width, args.height)
    vao = glGenVertexArrays(1)
    query = int(glGenQueries(1)[0])
    os.makedirs(OUT, exist_ok=True)

    records = []
    comparison = []
    active_scenes = scenes()[:2] if args.smoke else scenes()
    for name, eye_values, target_values in active_scenes:
        eye = np.asarray(eye_values, np.float32)
        look = np.asarray(target_values, np.float32)
        camera = camera_basis(eye, look)

        set_view(reference, args.width, args.height, eye, camera)
        for _ in range(3):
            draw_once(reference, target, vao)
        glFinish()
        reference_image = read_image(target)

        set_view(production, args.width, args.height, eye, camera)
        warm_gpu(production, target, vao, 0.15 if args.smoke else 1.5)
        samples = [
            timed_draw(production, target, vao, query)
            for _ in range(args.frames)
        ]
        production_image = read_image(target)
        error = rmse(production_image, reference_image)
        record = {
            "scene": name,
            "gpu_ms_samples": samples,
            "median_ms": float(statistics.median(samples)),
            "p95_ms": percentile(samples, 95),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "rmse_vs_reference": error,
        }
        records.append(record)
        comparison.append(label(
            production_image,
            "%s  %.2f ms  RMSE %.3e" %
            (name, record["median_ms"], error),
        ))
        print(
            "%-22s median %8.3f ms  p95 %8.3f ms  RMSE %.3e" %
            (name, record["median_ms"], record["p95_ms"], error)
        )

    medians = [item["median_ms"] for item in records]
    result = {
        "phase": "H0",
        "smoke": bool(args.smoke),
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": gpu,
        "gl_version": driver,
        "resolution": [args.width, args.height],
        "frames_per_scene": args.frames,
        "warmup_seconds_per_scene": 0.15 if args.smoke else 1.5,
        "shader_path": os.path.relpath(SHADER, REPO).replace("\\", "/"),
        "shader_sha256": shader_sha256,
        "production_defines": PRODUCTION,
        "reference_defines": REFERENCE,
        "cumulonimbus_defaults": CB_DEFAULTS,
        "scenes": records,
        "summary": {
            "median_of_scene_medians_ms": float(statistics.median(medians)),
            "p95_of_scene_medians_ms": percentile(medians, 95),
            "min_scene_median_ms": min(medians),
            "max_scene_median_ms": max(medians),
        },
    }
    suffix = "smoke" if args.smoke else "%dx%d" % (args.width, args.height)
    json_path = os.path.join(OUT, "baseline_%s.json" % suffix)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    if comparison:
        Image.fromarray(np.concatenate(comparison, axis=0)).save(
            os.path.join(OUT, "baseline_%s.png" % suffix)
        )
    print("GPU:", gpu)
    print("Shader SHA-256:", shader_sha256)
    print("Results:", json_path)
    glfw.terminate()


if __name__ == "__main__":
    main()
