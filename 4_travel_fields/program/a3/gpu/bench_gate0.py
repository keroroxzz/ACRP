"""Gate 0: conservative, multi-camera credibility audit of the 1.60x ReLU speedup.

Addresses the audit gaps:
  * multi-camera (12 viewpoints incl. inside-volume and grazing), not one scene;
  * FULL raymarch frontier saved raw (steps, sun_steps, ms, rmse, edge_rmse);
  * CONSERVATIVE speedup: cheapest ACTUAL march point no worse than A3 on BOTH full-image
    and edge-region RMSE (no interpolation); if none, the match is beyond the frontier;
  * reference convergence: 512/64 vs 1024/128 self-RMSE must be << A3 RMSE.

Reports per-preset median / P10 / P90 speedup across cameras. Saves everything to
results/rt_gate0/gate0.json.

Run (Py3.7 + glfw + PyOpenGL):  python a3/gpu/bench_gate0.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import glfw
from OpenGL.GL import *

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(os.path.dirname(HERE))
OUTDIR = os.path.join(RESEARCH, "results", "rt_gate0")
A3_SRC = open(os.path.join(HERE, "a3_production.glsl")).read()
MARCH_SRC = open(os.path.join(HERE, "cloud_frag.glsl")).read()
VS = ("#version 330 core\nvoid main(){ vec2 p=vec2(float((gl_VertexID<<1)&2),"
      "float(gl_VertexID&2)); gl_Position=vec4(p*2.0-1.0,0.0,1.0); }\n")
W, H = 1280, 720
VOL = 4.0


def inject(src, d):
    L = src.split("\n"); return L[0] + "\n" + "\n".join("#define %s %s" % kv for kv in d.items()) + "\n" + "\n".join(L[1:])
def sh(src, st):
    s = glCreateShader(st); glShaderSource(s, src); glCompileShader(s)
    if glGetShaderiv(s, GL_COMPILE_STATUS) != GL_TRUE: raise RuntimeError(glGetShaderInfoLog(s).decode())
    return s
def prog(src, d):
    p = glCreateProgram(); glAttachShader(p, sh(VS, GL_VERTEX_SHADER)); glAttachShader(p, sh(inject(src, d), GL_FRAGMENT_SHADER))
    glLinkProgram(p)
    if glGetProgramiv(p, GL_LINK_STATUS) != GL_TRUE: raise RuntimeError(glGetProgramInfoLog(p).decode())
    return p


def cam_basis(eye, ctr):
    f = ctr - eye; f /= np.linalg.norm(f)
    r = np.cross([0, 1, 0], f)
    if np.linalg.norm(r) < 1e-4: r = np.array([1.0, 0, 0])
    r /= np.linalg.norm(r); u = np.cross(f, r)
    return np.stack([r, u, f], 1).astype(np.float32)


def cameras():
    ctr = np.array([2.0, 2.0, 2.0], np.float32); out = []
    rng = np.random.default_rng(3)
    for i in range(10):  # orbit viewpoints
        yaw = 2 * np.pi * i / 10.0 + 0.1; pitch = 0.15 + 0.5 * rng.random(); rad = 8.0 + 3.0 * rng.random()
        eye = ctr + rad * np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)], np.float32)
        out.append(("orbit%d" % i, eye.astype(np.float32), cam_basis(eye, ctr)))
    # inside-volume camera
    eye = np.array([2.0, 2.0, 3.6], np.float32); out.append(("inside", eye, cam_basis(eye, np.array([2, 2, 0], np.float32))))
    # grazing: look nearly tangent to the top face
    eye = np.array([-4.0, 3.9, 2.0], np.float32); out.append(("grazing", eye, cam_basis(eye, np.array([6, 4.05, 2], np.float32))))
    return out


def draw(p, fbo, vao, eye, cam, frames):
    glBindFramebuffer(GL_FRAMEBUFFER, fbo); glViewport(0, 0, W, H); glUseProgram(p)
    glUniform2f(glGetUniformLocation(p, "uRes"), float(W), float(H))
    glUniform3f(glGetUniformLocation(p, "uEye"), *[float(x) for x in eye])
    glUniformMatrix3fv(glGetUniformLocation(p, "uCam"), 1, GL_FALSE, cam.T.copy()); glBindVertexArray(vao)
    for _ in range(6): glDrawArrays(GL_TRIANGLES, 0, 3)
    glFinish(); q = glGenQueries(1); ts = []
    for _ in range(frames):
        glBeginQuery(GL_TIME_ELAPSED, int(q[0])); glDrawArrays(GL_TRIANGLES, 0, 3); glEndQuery(GL_TIME_ELAPSED)
        glFinish(); ts.append(float(glGetQueryObjectuiv(int(q[0]), GL_QUERY_RESULT)) / 1e6)
    img = np.frombuffer(glReadPixels(0, 0, W, H, GL_RGBA, GL_FLOAT), np.float32).reshape(H, W, 4).copy()
    return float(np.median(ts)), img


def rmse(a, b): return float(np.sqrt(np.mean((a[:, :, :3] - b[:, :, :3]) ** 2)))
def emask(r):
    lum = r[:, :, :3].mean(2); gy, gx = np.gradient(lum); g = np.hypot(gx, gy); return g >= np.percentile(g, 88.0)
def ermse(a, b, m):
    d = ((a[:, :, :3] - b[:, :, :3]) ** 2).mean(2); return float(np.sqrt(d[m].mean()))


def main():
    glfw.init()
    for k, v in [(glfw.VISIBLE, glfw.FALSE), (glfw.CONTEXT_VERSION_MAJOR, 3), (glfw.CONTEXT_VERSION_MINOR, 3),
                 (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)]:
        glfw.window_hint(k, v)
    win = glfw.create_window(W, H, "g0", None, None); glfw.make_context_current(win)
    tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0, GL_RGBA, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    fbo = glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0); vao = glGenVertexArrays(1)
    print("GPU:", glGetString(GL_RENDERER).decode(), "| %dx%d | conservative multi-camera Gate 0\n" % (W, H))

    MB = dict(NOCT=4, NOCC=2, DETAIL_AMP=0.1875, NOCC_SUN=4, SUN_BIAS=0.0, SHAPED=0, SUN_INTERVAL=0.0,
              A3_ADAPT=0, CLASS_MARG=0.08, A3_TRGL=5, CAM_STEPS=64, SUN_STEPS=16,
              A3_CAMF=8, A3_CAMGL=5, A3_SUNF=4, A3_SUNGL=3, METHOD=0)
    def mc(**k): d = dict(MB); d.update(k); return d
    march_pts = [(40, 12), (48, 14), (56, 16), (64, 16), (80, 20), (96, 24), (128, 32), (192, 48)]
    presets = [("low", dict(A3_CAMF=8, A3_CAMGL=2, A3_SUNF=2, A3_SUNGL=2, NOCC_SUN=4)),
               ("medium", dict(A3_CAMF=8, A3_CAMGL=3, A3_SUNF=4, A3_SUNGL=2, NOCC_SUN=4)),
               ("converged", dict(A3_CAMF=8, A3_CAMGL=5, A3_SUNF=4, A3_SUNGL=2, NOCC_SUN=4))]
    # precompile
    P_ref = prog(MARCH_SRC, mc(CAM_STEPS=512, SUN_STEPS=64))
    P_ref2 = prog(MARCH_SRC, mc(CAM_STEPS=1024, SUN_STEPS=128))
    P_march = [(cs, ss, prog(MARCH_SRC, mc(CAM_STEPS=cs, SUN_STEPS=ss))) for cs, ss in march_pts]
    P_a3 = [(n, prog(A3_SRC, dict(kw, INSTRUMENT=0))) for n, kw in presets]

    data = {"gpu": glGetString(GL_RENDERER).decode(), "cameras": {}}
    ref_conv = []
    speedups = {n: [] for n, _ in presets}
    for cname, eye, cam in cameras():
        _, ref = draw(P_ref, fbo, vao, eye, cam, 4)
        _, ref2 = draw(P_ref2, fbo, vao, eye, cam, 4)
        rc = rmse(ref, ref2); ref_conv.append(rc); em = emask(ref)
        frontier = []
        for cs, ss, p in P_march:
            ms, img = draw(p, fbo, vao, eye, cam, 25)
            frontier.append(dict(steps=cs, sun=ss, ms=ms, rmse=rmse(img, ref), edge=ermse(img, ref, em)))
        cam_rec = {"ref_conv_rmse": rc, "march": frontier, "a3": {}}
        for n, p in P_a3:
            ms, img = draw(p, fbo, vao, eye, cam, 40)
            e = rmse(img, ref); ee = ermse(img, ref, em)
            # conservative: cheapest march no worse on BOTH metrics
            ok = [m for m in frontier if m["rmse"] <= e and m["edge"] <= ee]
            if ok:
                mm = min(m["ms"] for m in ok); sp = mm / ms; beyond = False
            else:
                mm = max(m["ms"] for m in frontier); sp = mm / ms; beyond = True  # lower bound
            speedups[n].append(sp)
            cam_rec["a3"][n] = dict(ms=ms, rmse=e, edge=ee, cons_march_ms=mm, speedup=sp, match_beyond_frontier=beyond)
        data["cameras"][cname] = cam_rec
        print("%-8s refConv %.1e | " % (cname, rc) + "  ".join(
            "%s %.2fx%s" % (n, cam_rec["a3"][n]["speedup"], "*" if cam_rec["a3"][n]["match_beyond_frontier"] else "")
            for n, _ in presets))

    print("\n-- reference convergence (512/64 vs 1024/128) max self-RMSE: %.2e --" % max(ref_conv))
    print("-- conservative speedup across %d cameras (P10 / median / P90; * = match beyond frontier) --" % len(speedups["low"]))
    data["summary"] = {}
    for n, _ in presets:
        s = np.array(speedups[n])
        data["summary"][n] = dict(p10=float(np.percentile(s, 10)), median=float(np.median(s)), p90=float(np.percentile(s, 90)))
        print("  %-10s  P10 %.2fx  median %.2fx  P90 %.2fx" % (n, np.percentile(s, 10), np.median(s), np.percentile(s, 90)))
    data["ref_conv_max"] = max(ref_conv)
    glfw.terminate(); os.makedirs(OUTDIR, exist_ok=True)
    json.dump(data, open(os.path.join(OUTDIR, "gate0.json"), "w"), indent=2)
    print("\nfull raw frontier + per-camera data -> %s/gate0.json" % OUTDIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
