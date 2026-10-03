"""GPU benchmark harness: A3 vs classic raymarch, real GLSL on the GPU.

Compiles a3/gpu/cloud_frag.glsl with injected #defines, renders a fullscreen pass to an
FBO at real-time resolution, times it with GL_TIME_ELAPSED (GPU time), reads back the
image (PNG dump), and computes RMSE vs a high-quality reference render.

Goal of the loop: an A3 config with GPU_ms <= raymarch_ms / 1.5 AND RMSE <= raymarch_RMSE.

Run (Python 3.7 with glfw + PyOpenGL):  python a3/gpu/bench.py
"""
from __future__ import annotations

import os
import sys
import numpy as np

import glfw
from OpenGL.GL import *

HERE = os.path.dirname(os.path.abspath(__file__))
FRAG_PATH = os.path.join(HERE, "cloud_frag.glsl")
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "results", "rt")

VS = """#version 330 core
void main(){ vec2 p = vec2(float((gl_VertexID<<1)&2), float(gl_VertexID&2));
    gl_Position = vec4(p*2.0-1.0, 0.0, 1.0); }
"""

W, H = 1280, 720
_frag_src = open(FRAG_PATH).read()


def make_frag(defines):
    lines = _frag_src.split("\n")
    inject = "\n".join("#define %s %s" % (k, v) for k, v in defines.items())
    # insert after the #version line
    return lines[0] + "\n" + inject + "\n" + "\n".join(lines[1:])


def compile_shader(src, stage):
    s = glCreateShader(stage)
    glShaderSource(s, src)
    glCompileShader(s)
    if glGetShaderiv(s, GL_COMPILE_STATUS) != GL_TRUE:
        log = glGetShaderInfoLog(s).decode()
        raise RuntimeError("shader compile error:\n" + log)
    return s


def make_program(defines):
    vs = compile_shader(VS, GL_VERTEX_SHADER)
    fs = compile_shader(make_frag(defines), GL_FRAGMENT_SHADER)
    p = glCreateProgram()
    glAttachShader(p, vs); glAttachShader(p, fs)
    glLinkProgram(p)
    if glGetProgramiv(p, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError("link error:\n" + glGetProgramInfoLog(p).decode())
    glDeleteShader(vs); glDeleteShader(fs)
    return p


def camera(yaw=0.9, pitch=0.35, rad=9.0):
    center = np.array([2.0, 2.0, 2.0], np.float32)
    eye = center + rad * np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch),
                                   np.cos(pitch) * np.cos(yaw)], np.float32)
    f = center - eye; f /= np.linalg.norm(f)
    r = np.cross(np.array([0, 1, 0], np.float32), f); r /= np.linalg.norm(r)
    u = np.cross(f, r)
    cam = np.stack([r, u, f], axis=1).astype(np.float32)  # columns r,u,f
    return eye, cam


def render(prog, fbo, vao, eye, cam, frames=60):
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glViewport(0, 0, W, H)
    glUseProgram(prog)
    glUniform2f(glGetUniformLocation(prog, "uRes"), float(W), float(H))
    glUniform3f(glGetUniformLocation(prog, "uEye"), *[float(x) for x in eye])
    glUniformMatrix3fv(glGetUniformLocation(prog, "uCam"), 1, GL_FALSE, cam.T.copy())
    glBindVertexArray(vao)
    # warmup
    for _ in range(8):
        glDrawArrays(GL_TRIANGLES, 0, 3)
    glFinish()
    q = glGenQueries(1)
    times = []
    for _ in range(frames):
        glBeginQuery(GL_TIME_ELAPSED, int(q[0]))
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glEndQuery(GL_TIME_ELAPSED)
        glFinish()
        ns = glGetQueryObjectuiv(int(q[0]), GL_QUERY_RESULT)
        times.append(float(ns) / 1e6)  # ms
    img = glReadPixels(0, 0, W, H, GL_RGBA, GL_FLOAT)
    img = np.frombuffer(img, np.float32).reshape(H, W, 4)[:, :, :3].copy()
    return float(np.median(times)), img


def save_png(img, name):
    from PIL import Image
    a = np.clip(img[::-1] * 255.0, 0, 255).astype(np.uint8)  # flip Y
    os.makedirs(OUTDIR, exist_ok=True)
    Image.fromarray(a).save(os.path.join(OUTDIR, name))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def edge_mask(ref):
    # high-gradient (cloud-edge) pixels of the reference luminance
    lum = ref.mean(axis=2)
    gy, gx = np.gradient(lum)
    g = np.sqrt(gx * gx + gy * gy)
    thr = np.percentile(g, 88.0)
    return g >= thr


def edge_rmse(a, b, mask):
    d = ((a - b) ** 2).mean(axis=2)
    return float(np.sqrt(d[mask].mean())) if mask.any() else 0.0


def main():
    if not glfw.init():
        print("glfw init failed"); return 1
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    win = glfw.create_window(W, H, "bench", None, None)
    glfw.make_context_current(win)

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0, GL_RGBA, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
    vao = glGenVertexArrays(1)

    eye, cam = camera()
    print("GPU:", glGetString(GL_RENDERER).decode(), "| res %dx%d" % (W, H))

    BASE = dict(NOCT=4, NOCC=2, DETAIL_AMP=0.1875, NOCC_SUN=4, SUN_BIAS=0.0, SHAPED=1,
                SUN_INTERVAL=0.0, A3_ADAPT=0, CLASS_MARG=0.08, A3_TRGL=5, CAM_STEPS=64, SUN_STEPS=16,
                A3_CAMF=8, A3_CAMGL=5, A3_SUNF=4, A3_SUNGL=3)

    def cfg(**kw):
        d = dict(BASE); d.update(kw); return d

    # reference (very fine raymarch, smoothstep density)
    ref_prog = make_program(cfg(METHOD=0, CAM_STEPS=512, SUN_STEPS=64))
    _, ref_img = render(ref_prog, fbo, vao, eye, cam, frames=8)
    save_png(ref_img, "reference.png")
    emask = edge_mask(ref_img)

    configs = []
    for cs, ss in [(48, 14), (64, 16), (96, 24), (128, 32), (192, 48)]:
        configs.append(("march_c%d" % cs, cfg(METHOD=0, CAM_STEPS=cs, SUN_STEPS=ss)))
    # A3 smoothstep, uniform GL5 (no classification) -- the ~1.0x baseline
    for cg, sf in [(5, 4), (3, 4), (3, 2)]:
        configs.append(("a3ss_uniform_g%d_s%d" % (cg, sf),
                        cfg(METHOD=1, A3_ADAPT=0, A3_CAMGL=cg, A3_SUNF=sf)))
    # A3 smoothstep, adaptive classification: vary cell freq (f8/f16), transition GL order (5/8)
    for cfr, trg, mg, sf in [(8, 5, 0.05, 4), (8, 8, 0.05, 4), (16, 5, 0.04, 4),
                             (16, 8, 0.04, 4), (16, 5, 0.04, 2), (16, 8, 0.04, 2),
                             (16, 8, 0.03, 4)]:
        configs.append(("a3ss_adapt_f%d_tr%d_m%.2f_s%d" % (cfr, trg, mg, sf),
                        cfg(METHOD=1, A3_ADAPT=1, A3_CAMF=cfr, A3_TRGL=trg, CLASS_MARG=mg,
                            A3_SUNF=sf)))

    print("%-26s %9s %10s %10s" % ("config", "GPU_ms", "RMSE", "edgeRMSE"))
    results = []
    for name, defs in configs:
        try:
            prog = make_program(defs)
        except RuntimeError as e:
            print(name, "COMPILE FAIL:", str(e)[:200]); continue
        ms, img = render(prog, fbo, vao, eye, cam)
        e = rmse(img, ref_img)
        ee = edge_rmse(img, ref_img, emask)
        save_png(img, name + ".png")
        results.append((name, ms, e, defs["METHOD"], ee))
        print("%-26s %9.3f %10.5f %10.5f" % (name, ms, e, ee))
        glDeleteProgram(prog)

    glfw.terminate()

    march = [r for r in results if r[3] == 0]
    a3 = [r for r in results if r[3] == 1]

    def interp_frontier(qidx):
        mm = sorted(march, key=lambda r: r[qidx])
        mq = np.array([r[qidx] for r in mm]); ms = np.array([r[1] for r in mm])
        def f(q):
            if q <= mq[0]:
                return ms[0]
            if q >= mq[-1]:
                return ms[-1]
            return float(np.interp(q, mq, ms))
        return f

    for label, qidx in [("full-image RMSE", 2), ("EDGE-region RMSE", 4)]:
        f = interp_frontier(qidx)
        print("\n-- equal-quality speedup by %s (march ms interpolated at A3's quality) --" % label)
        best = 0.0
        for r in sorted(a3, key=lambda r: r[1]):
            name, ms, q = r[0], r[1], r[qidx]
            mm = f(q); sp = mm / ms; best = max(best, sp)
            print("A3 %-24s %.3f ms @ %.5f  vs march %.3f ms  -> %.2fx" % (name, ms, q, mm, sp))
        print("BEST equal-quality speedup by %s: %.2fx" % (label, best))
    return 0


if __name__ == "__main__":
    sys.exit(main())
