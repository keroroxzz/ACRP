"""FROZEN production benchmark: A3-GBox64/F32 ReLU baseline vs classic raymarch.

Produces the frozen 3-preset results table (low / medium / converged quality) for the
validated 1.58x ReLU configuration. This is the baseline for all later smoothstep (S1/S2)
experiments and should not change.

Per preset: GPU ms (RTX 4070, 1280x720), full-image RMSE and edge-region RMSE vs a 512-step
reference, plus instrumented per-pixel density evaluations, sun-ray count, and skipped-segment
ratio. Also reports the equal-quality raymarch cost (interpolated) and the speedup.

Run (Python 3.7 + glfw + PyOpenGL):  python a3/gpu/bench_production.py
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np

import glfw
from OpenGL.GL import *

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(os.path.dirname(HERE))
OUTDIR = os.path.join(RESEARCH, "results", "rt_production")
A3_SRC = open(os.path.join(HERE, "a3_production.glsl")).read()
MARCH_SRC = open(os.path.join(HERE, "cloud_frag.glsl")).read()

VS = ("#version 330 core\nvoid main(){ vec2 p=vec2(float((gl_VertexID<<1)&2),"
      "float(gl_VertexID&2)); gl_Position=vec4(p*2.0-1.0,0.0,1.0); }\n")
W, H = 1280, 720


def inject(src, defines):
    lines = src.split("\n")
    inj = "\n".join("#define %s %s" % (k, v) for k, v in defines.items())
    return lines[0] + "\n" + inj + "\n" + "\n".join(lines[1:])


def sh(src, stage):
    s = glCreateShader(stage); glShaderSource(s, src); glCompileShader(s)
    if glGetShaderiv(s, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(s).decode())
    return s


def prog(src, defines):
    vs = sh(VS, GL_VERTEX_SHADER); fs = sh(inject(src, defines), GL_FRAGMENT_SHADER)
    p = glCreateProgram(); glAttachShader(p, vs); glAttachShader(p, fs); glLinkProgram(p)
    if glGetProgramiv(p, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(p).decode())
    glDeleteShader(vs); glDeleteShader(fs); return p


def camera(yaw=0.9, pitch=0.35, rad=9.0):
    c = np.array([2.0, 2.0, 2.0], np.float32)
    eye = c + rad*np.array([np.cos(pitch)*np.sin(yaw), np.sin(pitch), np.cos(pitch)*np.cos(yaw)], np.float32)
    f = c-eye; f/=np.linalg.norm(f); r = np.cross([0,1,0], f); r/=np.linalg.norm(r); u = np.cross(f, r)
    return eye, np.stack([r, u, f], 1).astype(np.float32)


def draw(p, fbo, vao, eye, cam, frames):
    glBindFramebuffer(GL_FRAMEBUFFER, fbo); glViewport(0, 0, W, H); glUseProgram(p)
    glUniform2f(glGetUniformLocation(p, "uRes"), float(W), float(H))
    glUniform3f(glGetUniformLocation(p, "uEye"), *[float(x) for x in eye])
    glUniformMatrix3fv(glGetUniformLocation(p, "uCam"), 1, GL_FALSE, cam.T.copy())
    glBindVertexArray(vao)
    for _ in range(8):
        glDrawArrays(GL_TRIANGLES, 0, 3)
    glFinish()
    q = glGenQueries(1); times = []
    for _ in range(frames):
        glBeginQuery(GL_TIME_ELAPSED, int(q[0])); glDrawArrays(GL_TRIANGLES, 0, 3); glEndQuery(GL_TIME_ELAPSED)
        glFinish(); times.append(float(glGetQueryObjectuiv(int(q[0]), GL_QUERY_RESULT))/1e6)
    img = np.frombuffer(glReadPixels(0, 0, W, H, GL_RGBA, GL_FLOAT), np.float32).reshape(H, W, 4).copy()
    return float(np.median(times)), img


def save_png(img, name):
    from PIL import Image
    os.makedirs(OUTDIR, exist_ok=True)
    Image.fromarray(np.clip(img[::-1, :, :3]*255, 0, 255).astype(np.uint8)).save(os.path.join(OUTDIR, name))


def rmse(a, b): return float(np.sqrt(np.mean((a[:, :, :3]-b[:, :, :3])**2)))
def emask(ref):
    lum = ref[:, :, :3].mean(2); gy, gx = np.gradient(lum); g = np.sqrt(gx*gx+gy*gy)
    return g >= np.percentile(g, 88.0)
def ermse(a, b, m):
    d = ((a[:, :, :3]-b[:, :, :3])**2).mean(2); return float(np.sqrt(d[m].mean()))


def main():
    glfw.init()
    for k, v in [(glfw.VISIBLE, glfw.FALSE), (glfw.CONTEXT_VERSION_MAJOR, 3),
                 (glfw.CONTEXT_VERSION_MINOR, 3), (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)]:
        glfw.window_hint(k, v)
    win = glfw.create_window(W, H, "prod", None, None); glfw.make_context_current(win)
    tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0, GL_RGBA, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    fbo = glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
    vao = glGenVertexArrays(1)
    eye, cam = camera()
    print("GPU:", glGetString(GL_RENDERER).decode(), "| %dx%d\n" % (W, H))

    MB = dict(NOCT=4, NOCC=2, DETAIL_AMP=0.1875, NOCC_SUN=4, SUN_BIAS=0.0, SHAPED=0,
              SUN_INTERVAL=0.0, A3_ADAPT=0, CLASS_MARG=0.08, A3_TRGL=5, CAM_STEPS=64, SUN_STEPS=16,
              A3_CAMF=8, A3_CAMGL=5, A3_SUNF=4, A3_SUNGL=3, METHOD=0)

    def mcfg(**kw): d = dict(MB); d.update(kw); return d

    # reference (fine march)
    _, ref = draw(prog(MARCH_SRC, mcfg(CAM_STEPS=512, SUN_STEPS=64)), fbo, vao, eye, cam, 8)
    save_png(ref, "reference.png"); em = emask(ref)

    # march frontier for equal-quality interpolation
    march = []
    for cs, ss in [(40, 12), (48, 14), (56, 16), (64, 16), (80, 20), (96, 24), (128, 32), (192, 48)]:
        ms, img = draw(prog(MARCH_SRC, mcfg(CAM_STEPS=cs, SUN_STEPS=ss)), fbo, vao, eye, cam, 40)
        march.append((ms, rmse(img, ref), ermse(img, ref, em)))
    march.sort(key=lambda r: r[1])
    mq = np.array([r[1] for r in march]); mms = np.array([r[0] for r in march])

    def march_ms_at(q): return float(np.interp(q, mq, mms)) if mq[0] < q < mq[-1] else (mms[0] if q <= mq[0] else mms[-1])

    presets = [("low", dict(A3_CAMF=8, A3_CAMGL=2, A3_SUNF=2, A3_SUNGL=2, NOCC_SUN=4)),
               ("medium", dict(A3_CAMF=8, A3_CAMGL=3, A3_SUNF=4, A3_SUNGL=2, NOCC_SUN=4)),
               ("converged", dict(A3_CAMF=8, A3_CAMGL=5, A3_SUNF=4, A3_SUNGL=2, NOCC_SUN=4))]

    print("%-10s %8s %9s %9s %9s %9s %9s %8s" %
          ("preset", "GPU_ms", "RMSE", "edgeRMSE", "evals/px", "sunray/px", "skip%", "speedup"))
    out = {"gpu": glGetString(GL_RENDERER).decode(), "res": [W, H], "presets": {}}
    for name, kw in presets:
        d = dict(kw); d.update(dict(INSTRUMENT=0))
        ms, img = draw(prog(A3_SRC, d), fbo, vao, eye, cam, 60)
        e = rmse(img, ref); ee = ermse(img, ref, em)
        save_png(img, "a3_%s.png" % name)
        di = dict(kw); di.update(dict(INSTRUMENT=1))
        _, ci = draw(prog(A3_SRC, di), fbo, vao, eye, cam, 2)
        evals = float(ci[:, :, 0].mean()); sunr = float(ci[:, :, 1].mean())
        skip = float(ci[:, :, 2].mean()); segs = float(ci[:, :, 3].mean())
        skip_ratio = skip / max(segs, 1e-9)
        sp = march_ms_at(e) / ms
        out["presets"][name] = dict(gpu_ms=ms, rmse=e, edge_rmse=ee, evals_per_px=evals,
                                    sunrays_per_px=sunr, skipped_ratio=skip_ratio,
                                    equal_quality_march_ms=march_ms_at(e), speedup=sp)
        print("%-10s %8.3f %9.5f %9.5f %9.1f %9.2f %8.1f %7.2fx" %
              (name, ms, e, ee, evals, sunr, 100*skip_ratio, sp))

    glfw.terminate()
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(out, open(os.path.join(OUTDIR, "results.json"), "w"), indent=2)
    print("\nfrozen results -> %s/results.json + PNGs" % OUTDIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
