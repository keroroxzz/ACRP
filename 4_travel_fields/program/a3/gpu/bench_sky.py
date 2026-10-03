"""Benchmark the real sky scene: AAA raymarch vs A3-uniform vs A3-hierarchical (two-level),
across several cameras, against a clean no-jitter high-quality reference. Reports GPU ms, RMSE,
and speedup vs the AAA march. Also dumps a labeled comparison image."""
from __future__ import annotations
import numpy as np, glfw, os, json
from OpenGL.GL import *
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(RESEARCH, "results", "sky")
SRC = open(os.path.join(HERE, "sky_frag.glsl")).read()
VS = ("#version 330 core\nvoid main(){ vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));"
      " gl_Position=vec4(p*2.0-1.0,0.0,1.0); }\n")
W, H = 1280, 720


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
def rmse(a, b): return float(np.sqrt(np.mean((a[:, :, :3] - b[:, :, :3]) ** 2)))
def lab(img, txt):
    im = Image.fromarray((np.clip(img[:, :, :3], 0, 1) * 255).astype(np.uint8)); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 620, 26], fill=(0, 0, 0)); d.text((8, 7), txt, fill=(255, 255, 120)); return np.asarray(im)


def cameras():
    out = []
    out.append(("wide", np.array([-10.0, 3.2, -7.0], np.float32), np.array([6.0, 4.3, 9.0], np.float32)))
    out.append(("low_up", np.array([-2.0, 1.0, -2.0], np.float32), np.array([6.0, 5.0, 9.0], np.float32)))
    out.append(("near_cb", np.array([2.0, 3.5, 2.0], np.float32), np.array([6.0, 5.5, 9.0], np.float32)))
    out.append(("high", np.array([-8.0, 6.5, -6.0], np.float32), np.array([6.0, 4.0, 9.0], np.float32)))
    return out


def main():
    os.makedirs(OUT, exist_ok=True); glfw.init()
    for k, v in [(glfw.VISIBLE, glfw.FALSE), (glfw.CONTEXT_VERSION_MAJOR, 3), (glfw.CONTEXT_VERSION_MINOR, 3),
                 (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)]:
        glfw.window_hint(k, v)
    win = glfw.create_window(W, H, "b", None, None); glfw.make_context_current(win)
    tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, tex); glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0, GL_RGBA, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    fbo = glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER, fbo); glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0); vao = glGenVertexArrays(1)
    print("GPU:", glGetString(GL_RENDERER).decode(), "| real sky scene: march vs A3-uniform vs A3-hier\n")

    A3 = dict(A3_CAMF=6, A3_CAMGL=5, A3_SUNF=4, A3_SUNGL=4)
    REF = dict(METHOD=1, SUN_STEPS=12, SMALL_STEP=0.03, BIG_STEP=0.25, MAXI=2400, JITTER_ON=0, SUN_JITTER=0.0, **A3)
    base = dict(SUN_STEPS=6, SMALL_STEP=0.06, BIG_STEP=0.30, MAXI=1400, JITTER_ON=1, A3_SUNF=4, A3_SUNGL=4)
    variants = [
        ("march",        dict(base, METHOD=1, SUN_JITTER=0.0, A3_CAMF=6, A3_CAMGL=5)),
        ("a3_uniform",   dict(base, METHOD=0, SUN_JITTER=1.0, A3_CAMF=6, A3_CAMGL=5)),
        ("a3_hier_eq",   dict(base, METHOD=2, SUN_JITTER=1.0, A3_CAMF=6, A3_CAMGL=2)),  # equal quality
        ("a3_hier_fast", dict(base, METHOD=2, SUN_JITTER=1.0, A3_CAMF=4, A3_CAMGL=2)),  # faster, ~equal
        ("a3_hier_max",  dict(base, METHOD=2, SUN_JITTER=1.0, A3_CAMF=3, A3_CAMGL=2)),  # fastest
    ]
    Pref = prog(REF); Pv = [(n, prog(d)) for n, d in variants]
    data = {"gpu": glGetString(GL_RENDERER).decode(), "cameras": {}}
    agg = {n: {"ms": [], "rmse": [], "sp": []} for n, _ in variants}
    print("%-9s %-11s %-8s %-9s %-8s" % ("view", "variant", "ms", "rmse", "vs march"))
    for cname, eye, ctr in cameras():
        cam = cam_basis(eye, ctr); _, ref = draw(Pref, fbo, vao, eye, cam, 4)
        march_ms = None; rec = {}
        strip = []
        for n, P in Pv:
            ms, img = draw(P, fbo, vao, eye, cam, 30); e = rmse(img, ref)
            if n == "march": march_ms = ms
            rec[n] = dict(ms=ms, rmse=e); agg[n]["ms"].append(ms); agg[n]["rmse"].append(e)
            if cname == "wide": strip.append(lab(img, "%s  %.1fms  rmse %.1e" % (n, ms, e)))
        for n, _ in variants:
            sp = march_ms / rec[n]["ms"]; rec[n]["speedup_vs_march"] = sp; agg[n]["sp"].append(sp)
            print("%-9s %-11s %-8.2f %-9.2e %-8.2f" % (cname, n, rec[n]["ms"], rec[n]["rmse"], sp))
        data["cameras"][cname] = rec
        if strip: Image.fromarray(np.concatenate(strip, 0)).save(os.path.join(OUT, "bench_compare.png"))
        print()
    print("-- median across %d cameras --" % len(agg["march"]["ms"]))
    print("%-11s %-8s %-9s %-9s" % ("variant", "medMs", "medRMSE", "medSpeedup(vs march)"))
    data["summary"] = {}
    for n, _ in variants:
        mm = float(np.median(agg[n]["ms"])); mr = float(np.median(agg[n]["rmse"])); ms_ = float(np.median(agg[n]["sp"]))
        data["summary"][n] = dict(med_ms=mm, med_rmse=mr, med_speedup=ms_)
        print("%-11s %-8.2f %-9.2e %-9.2f" % (n, mm, mr, ms_))
    json.dump(data, open(os.path.join(OUT, "bench_sky.json"), "w"), indent=2)
    glfw.terminate(); print("\n-> %s/bench_sky.json + bench_compare.png" % OUT)


if __name__ == "__main__":
    main()
