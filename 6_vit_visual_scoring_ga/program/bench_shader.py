"""GPU benchmark for cumulus_multilayer.glsl at HD / FHD / 4K.

Timing uses GL_TIME_ELAPSED queries (actual GPU time, not wall clock) with warm-up frames
discarded. Cost varies strongly with view direction -- looking up through the decks is far more
expensive than grazing the horizon -- so every resolution is measured across ALL 12 hand-verified
photo geometries and reported as a MEDIAN with the full spread, rather than one flattering view.

Also breaks the cost down by disabling decks, to show where the time actually goes.

Run with the `dino` env (or any env with glfw/PyOpenGL):  python bench_shader.py
"""
from __future__ import annotations
import os, json, math, time, statistics as st
import numpy as np, glfw
from OpenGL.GL import *

HERE   = os.path.dirname(os.path.abspath(__file__))
SHADER = os.path.join(HERE, "cumulus_multilayer.glsl")
BEST   = os.path.join(HERE, "..", "results", "multi_best_dual.json")
VS = ("#version 330 core\nvoid main(){ vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));"
      " gl_Position=vec4(p*2.0-1.0,0.0,1.0);}\n")
HEIGHT = 0.06
RESOLUTIONS = [("HD  1280x720", 1280, 720), ("FHD 1920x1080", 1920, 1080), ("4K  3840x2160", 3840, 2160)]
WARMUP, FRAMES = 3, 7

def sh(src, st_):
    s = glCreateShader(st_); glShaderSource(s, src); glCompileShader(s)
    if glGetShaderiv(s, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(s).decode())
    return s

def main():
    best = json.load(open(BEST, encoding="utf-8"))
    shared, per_photo = best["shared"], best["per_photo"]
    names = sorted(per_photo.keys())

    glfw.init()
    for k, v in [(glfw.VISIBLE, glfw.FALSE), (glfw.CONTEXT_VERSION_MAJOR, 3),
                 (glfw.CONTEXT_VERSION_MINOR, 3), (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)]:
        glfw.window_hint(k, v)
    win = glfw.create_window(256, 256, "bench", None, None); glfw.make_context_current(win)
    print("GPU:", glGetString(GL_RENDERER).decode())

    src = open(SHADER, encoding="utf-8").read()
    prog = glCreateProgram()
    glAttachShader(prog, sh(VS, GL_VERTEX_SHADER)); glAttachShader(prog, sh(src, GL_FRAGMENT_SHADER))
    glLinkProgram(prog)
    if glGetProgramiv(prog, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(prog).decode())
    vao = glGenVertexArrays(1); q = int(glGenQueries(1)[0])

    def _spin_warmup(sec=1.5):
        # GPU boost ramps from the 210 MHz idle clock; 2 frames is nowhere near steady 2805 MHz.
        # Draw continuously for ~1.5 s so timings reflect the sustained clock, not the ramp.
        glBindVertexArray(vao); t=time.time()
        while time.time()-t < sec:
            for _ in range(30): glDrawArrays(GL_TRIANGLES,0,3)
            glFinish()

    def make_target(w, h):
        tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, None)
        fbo = glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
        glViewport(0, 0, w, h); return fbo, tex

    def frame_ms(w, h, pp, override=None):
        glUseProgram(prog); U = lambda n: glGetUniformLocation(prog, n)
        eye = np.array([0, HEIGHT, 0], np.float32); yaw = 0.0
        d = np.array([math.sin(yaw)*math.cos(pp['pitch']), math.sin(pp['pitch']),
                      math.cos(yaw)*math.cos(pp['pitch'])], np.float32)
        f = d/np.linalg.norm(d); r = np.cross([0, 1, 0], f); r /= np.linalg.norm(r); u = np.cross(f, r)
        cam = np.stack([r, u, f], 1).astype(np.float32)
        sd = np.array([math.cos(pp['sunEl'])*math.sin(pp['sunAz']), math.sin(pp['sunEl']),
                       math.cos(pp['sunEl'])*math.cos(pp['sunAz'])], np.float32)
        glUniform2f(U("uRes"), float(w), float(h)); glUniform3f(U("uEye"), *[float(z) for z in eye])
        glUniformMatrix3fv(U("uCam"), 1, GL_FALSE, cam.T.copy()); glUniform1f(U("uFocal"), float(pp['focal']))
        glUniform3f(U("uSunDir"), *[float(z) for z in sd]); glUniform1f(U("uExposure"), float(pp['expo']))
        for k, v in shared.items(): glUniform1f(U(k), float(v))
        cov = {k: float(pp[k]) for k in ("uCovLow", "uCovMid", "uCovHigh", "uTypeAmp", "uTypeSeed")}
        if override: cov.update(override)
        for k, v in cov.items(): glUniform1f(U(k), float(v))
        glBindVertexArray(vao)
        glBeginQuery(GL_TIME_ELAPSED, q); glDrawArrays(GL_TRIANGLES, 0, 3); glEndQuery(GL_TIME_ELAPSED)
        done = GL_FALSE
        while done == GL_FALSE: done = glGetQueryObjectiv(q, GL_QUERY_RESULT_AVAILABLE)
        return glGetQueryObjectuiv(q, GL_QUERY_RESULT) / 1e6

    def measure(w, h, override=None):
        per_view = []
        for n in names:
            pp = per_photo[n]
            for _ in range(WARMUP): frame_ms(w, h, pp, override)
            per_view.append(st.median([frame_ms(w, h, pp, override) for _ in range(FRAMES)]))
        return per_view

    print("\n%-14s %9s %9s %9s %9s %9s" % ("resolution", "median", "fps", "min", "max", "max/min"))
    print("-"*66)
    results = {}
    for label, w, h in RESOLUTIONS:
        fbo, tex = make_target(w, h)
        v = measure(w, h); results[label] = v
        med = st.median(v)
        print("%-14s %8.2fms %8.1f %8.2fms %8.2fms %8.1fx"
              % (label, med, 1000.0/med, min(v), max(v), max(v)/max(min(v), 1e-6)))
        glDeleteFramebuffers(1, [fbo]); glDeleteTextures(1, [tex])

    # scaling check: cost should be ~linear in pixel count for a pure fragment workload
    hd, fhd, k4 = (st.median(results[l]) for l, _, _ in RESOLUTIONS)
    print("\nscaling vs pixel count (HD = 1.00x):")
    print("  FHD  pixels 2.25x   time %.2fx" % (fhd/hd))
    print("  4K   pixels 9.00x   time %.2fx" % (k4/hd))

    # where does the time go? disable decks one at a time at FHD
    print("\ncost breakdown at FHD (median across the 12 views):")
    fbo, tex = make_target(1920, 1080)
    full = st.median(measure(1920, 1080))
    for tag, ov in (("no high (cirrus)", {"uCovHigh": 0.0}),
                    ("no mid  (cirrocu)", {"uCovMid": 0.0}),
                    ("no low  (cumulus)", {"uCovLow": 0.0}),
                    ("sky only (no cloud)", {"uCovLow": 0.0, "uCovMid": 0.0, "uCovHigh": 0.0})):
        m = st.median(measure(1920, 1080, ov))
        print("  %-20s %7.2fms   (%+.1f%% vs full %.2fms)" % (tag, m, 100.0*(m-full)/full, full))
    glDeleteFramebuffers(1, [fbo]); glDeleteTextures(1, [tex])

    print("\nper-view spread at FHD (why a single-view number would mislead):")
    v = results["FHD 1920x1080"]
    for n, t in sorted(zip(names, v), key=lambda z: -z[1]):
        print("  %-26s %7.2fms" % (n, t))
    glfw.terminate()

if __name__ == "__main__":
    main()
