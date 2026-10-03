"""Benchmark the fair-weather cumulus scene. RMSE is measured over the SKY/CLOUD region only
(pixels whose primary ray points above the horizon), per the requirement that the sky/cloud RMSE
be low. Compares AAA raymarch vs A3 hierarchical against a clean HQ reference, across cameras."""
from __future__ import annotations
import numpy as np, glfw, os, json
from OpenGL.GL import *
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(RESEARCH, "results", "cumulus")
SRC = open(os.path.join(HERE, "cumulus_frag.glsl")).read()
VS = ("#version 330 core\nvoid main(){ vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));"
      " gl_Position=vec4(p*2.0-1.0,0.0,1.0); }\n")
W, H = 1280, 720
FOCAL = 0.95


def inject(src, d): L = src.split("\n"); return L[0] + "\n" + "\n".join("#define %s %s" % kv for kv in d.items()) + "\n" + "\n".join(L[1:])
def sh(src, st):
    s = glCreateShader(st); glShaderSource(s, src); glCompileShader(s)
    if glGetShaderiv(s, GL_COMPILE_STATUS) != GL_TRUE: raise RuntimeError(glGetShaderInfoLog(s).decode())
    return s
def prog(d):
    p = glCreateProgram(); glAttachShader(p, sh(VS, GL_VERTEX_SHADER)); glAttachShader(p, sh(inject(SRC, d), GL_FRAGMENT_SHADER)); glLinkProgram(p)
    if glGetProgramiv(p, GL_LINK_STATUS) != GL_TRUE: raise RuntimeError(glGetProgramInfoLog(p).decode())
    return p
def cam_basis(eye, ctr):
    f = ctr - eye; f /= np.linalg.norm(f); r = np.cross([0, 1, 0], f); r /= np.linalg.norm(r); u = np.cross(f, r)
    return np.stack([r, u, f], 1).astype(np.float32)
def draw(p, fbo, vao, eye, cam, frames):
    glBindFramebuffer(GL_FRAMEBUFFER, fbo); glViewport(0, 0, W, H); glUseProgram(p)
    glUniform2f(glGetUniformLocation(p, "uRes"), float(W), float(H)); glUniform3f(glGetUniformLocation(p, "uEye"), *[float(x) for x in eye])
    glUniformMatrix3fv(glGetUniformLocation(p, "uCam"), 1, GL_FALSE, cam.T.copy()); glBindVertexArray(vao)
    for _ in range(4): glDrawArrays(GL_TRIANGLES, 0, 3)
    glFinish(); q = glGenQueries(1); ts = []
    for _ in range(frames):
        glBeginQuery(GL_TIME_ELAPSED, int(q[0])); glDrawArrays(GL_TRIANGLES, 0, 3); glEndQuery(GL_TIME_ELAPSED)
        glFinish(); ts.append(float(glGetQueryObjectuiv(int(q[0]), GL_QUERY_RESULT)) / 1e6)
    img = np.frombuffer(glReadPixels(0, 0, W, H, GL_RGBA, GL_FLOAT), np.float32).reshape(H, W, 4)
    return float(np.median(ts)), img[::-1].copy()


def sky_mask(eye, cam):  # True where the primary ray points above the horizon (sky/cloud pixels)
    ys, xs = np.mgrid[0:H, 0:W]
    uvx = (xs + 0.5 - 0.5 * W) / H; uvy = ((H - 1 - ys) + 0.5 - 0.5 * H) / H
    d = np.stack([uvx, uvy, np.full_like(uvx, FOCAL)], -1); d /= np.linalg.norm(d, axis=-1, keepdims=True)
    rd = d @ cam.T
    return rd[:, :, 1] > 0.0
def masked_rmse(a, b, m): d = ((a[:, :, :3] - b[:, :, :3]) ** 2).mean(2); return float(np.sqrt(d[m].mean()))


def cameras():
    return [("main", np.array([0.0, 0.5, 0.0], np.float32), np.array([0.0, 2.6, 6.0], np.float32)),
            ("left", np.array([0.0, 0.5, 0.0], np.float32), np.array([-4.0, 2.4, 6.0], np.float32)),
            ("higher", np.array([0.0, 0.7, 0.0], np.float32), np.array([0.0, 3.4, 6.0], np.float32)),
            ("shallow", np.array([0.0, 0.4, 0.0], np.float32), np.array([2.0, 1.7, 7.0], np.float32))]


def main():
    os.makedirs(OUT, exist_ok=True); glfw.init()
    for k, v in [(glfw.VISIBLE, glfw.FALSE), (glfw.CONTEXT_VERSION_MAJOR, 3), (glfw.CONTEXT_VERSION_MINOR, 3),
                 (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)]:
        glfw.window_hint(k, v)
    win = glfw.create_window(W, H, "b", None, None); glfw.make_context_current(win)
    tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, tex); glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0, GL_RGBA, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    fbo = glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER, fbo); glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0); vao = glGenVertexArrays(1)
    print("GPU:", glGetString(GL_RENDERER).decode(), "| cumulus scene: sky-region RMSE (masked)\n")

    A3 = dict(A3_SUNF=4, A3_SUNGL=4)
    REF = dict(METHOD=1, SUN_STEPS=12, SMALL_STEP=0.03, BIG_STEP=0.22, MAXI=3000, JITTER_ON=0, SUN_JITTER=0.0, A3_CAMF=6, A3_CAMGL=5, **A3)
    variants = [
        ("march",     dict(METHOD=1, SUN_STEPS=6, SMALL_STEP=0.06, BIG_STEP=0.28, MAXI=1600, JITTER_ON=1, SUN_JITTER=0.0, A3_CAMF=6, A3_CAMGL=5, **A3)),
        ("a3_hier",   dict(METHOD=2, SUN_STEPS=6, SMALL_STEP=0.06, BIG_STEP=0.28, MAXI=1600, JITTER_ON=1, SUN_JITTER=1.0, A3_CAMF=6, A3_CAMGL=2, **A3)),
        ("a3_hier_clean", dict(METHOD=2, SUN_STEPS=8, SMALL_STEP=0.06, BIG_STEP=0.28, MAXI=1600, JITTER_ON=1, SUN_JITTER=0.35, A3_CAMF=8, A3_CAMGL=3, **A3)),
    ]
    Pref = prog(REF); Pv = [(n, prog(d)) for n, d in variants]
    data = {"gpu": glGetString(GL_RENDERER).decode(), "cameras": {}}
    agg = {n: {"ms": [], "rmse": [], "sp": []} for n, _ in variants}
    print("%-9s %-15s %-8s %-10s %-8s" % ("view", "variant", "ms", "skyRMSE", "vs march"))
    for cname, eye, ctr in cameras():
        cam = cam_basis(eye, ctr); m = sky_mask(eye, cam); _, ref = draw(Pref, fbo, vao, eye, cam, 4)
        rec = {}; march_ms = None
        for n, P in Pv:
            ms, img = draw(P, fbo, vao, eye, cam, 30); e = masked_rmse(img, ref, m)
            if n == "march": march_ms = ms
            rec[n] = dict(ms=ms, sky_rmse=e); agg[n]["ms"].append(ms); agg[n]["rmse"].append(e)
            if cname == "main": Image.fromarray((np.clip(img[:, :, :3], 0, 1) * 255).astype(np.uint8)).save(os.path.join(OUT, "bench_%s.png" % n))
        for n, _ in variants:
            sp = march_ms / rec[n]["ms"]; rec[n]["speedup"] = sp; agg[n]["sp"].append(sp)
            print("%-9s %-15s %-8.2f %-10.2e %-8.2f" % (cname, n, rec[n]["ms"], rec[n]["sky_rmse"], sp))
        data["cameras"][cname] = rec; print()
    print("-- median across %d cameras --" % len(agg["march"]["ms"]))
    print("%-15s %-8s %-10s %-8s" % ("variant", "medMs", "medSkyRMSE", "medSpeedup"))
    data["summary"] = {}
    for n, _ in variants:
        mm = float(np.median(agg[n]["ms"])); mr = float(np.median(agg[n]["rmse"])); sp = float(np.median(agg[n]["sp"]))
        data["summary"][n] = dict(med_ms=mm, med_sky_rmse=mr, med_speedup=sp)
        print("%-15s %-8.2f %-10.2e %-8.2f" % (n, mm, mr, sp))
    json.dump(data, open(os.path.join(OUT, "bench_cumulus.json"), "w"), indent=2)
    glfw.terminate(); print("\n-> %s/bench_cumulus.json" % OUT)


if __name__ == "__main__":
    main()
