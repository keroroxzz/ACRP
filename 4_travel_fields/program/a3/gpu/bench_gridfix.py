"""Grid-artifact fix verification: baseline (midpoint skip) vs gridfix (whole-segment
certificate), on the views where the f8 cell lattice was visible (inside, grazing).

Renders both, saves the tonemapped image and the |A3 - raymarch reference| x40 diff for
each, and reports GPU ms, RMSE vs reference, and skipped-segment ratio. If the grid is an
occupancy-decision artifact, it should vanish in gridfix while RMSE drops and ms rises modestly.

Run:  python a3/gpu/bench_gridfix.py
"""
from __future__ import annotations
import numpy as np, glfw, os
from OpenGL.GL import *
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(RESEARCH, "results", "rt_gridfix")
# reuse helpers (prog/draw/cameras/inject/rmse/emask/ermse, W/H, MARCH_SRC path logic)
_src = open(os.path.join(HERE, "bench_gate0.py")).read().split("def main()")[0]
_src = _src.replace('A3_SRC = open(os.path.join(HERE, "a3_production.glsl")).read()',
                    'A3_SRC = open(os.path.join(HERE, "a3_production.glsl")).read()\n'
                    'FIX_SRC = open(os.path.join(HERE, "a3_gridfix.glsl")).read()')
exec(_src)


def main():
    os.makedirs(OUT, exist_ok=True)
    glfw.init()
    for k, v in [(glfw.VISIBLE, glfw.FALSE), (glfw.CONTEXT_VERSION_MAJOR, 3), (glfw.CONTEXT_VERSION_MINOR, 3),
                 (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)]:
        glfw.window_hint(k, v)
    win = glfw.create_window(W, H, "gf", None, None); glfw.make_context_current(win)
    tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0, GL_RGBA, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    fbo = glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0); vao = glGenVertexArrays(1)

    MB = dict(NOCT=4, NOCC=2, DETAIL_AMP=0.1875, NOCC_SUN=4, SUN_BIAS=0.0, SHAPED=0, SUN_INTERVAL=0.0,
              A3_ADAPT=0, CLASS_MARG=0.08, A3_TRGL=5, CAM_STEPS=64, SUN_STEPS=16,
              A3_CAMF=8, A3_CAMGL=5, A3_SUNF=4, A3_SUNGL=3, METHOD=0)
    def mc(**k): d = dict(MB); d.update(k); return d
    A3KW = dict(A3_CAMF=8, A3_CAMGL=2, A3_SUNF=2, A3_SUNGL=2, NOCC_SUN=4)
    P_ref = prog(MARCH_SRC, mc(CAM_STEPS=512, SUN_STEPS=64))
    P_base = prog(A3_SRC, dict(A3KW, INSTRUMENT=0)); P_baseI = prog(A3_SRC, dict(A3KW, INSTRUMENT=1))
    P_fix = prog(FIX_SRC, dict(A3KW, INSTRUMENT=0)); P_fixI = prog(FIX_SRC, dict(A3KW, INSTRUMENT=1))
    cams = {c[0]: c for c in cameras()}

    print("GPU:", glGetString(GL_RENDERER).decode(), "| baseline vs gridfix on grid-prone views\n")
    print("%-8s %-8s  %-9s %-9s %-8s %-8s" % ("view", "variant", "ms", "rmse", "skip%", "evals/px"))
    for cname in ["inside", "grazing", "orbit1"]:
        _, eye, cam = cams[cname]
        _, ref = draw(P_ref, fbo, vao, eye, cam, 4); em = emask(ref)
        for label, P, PI in [("baseline", P_base, P_baseI), ("gridfix", P_fix, P_fixI)]:
            ms, img = draw(P, fbo, vao, eye, cam, 40)
            _, inst = draw(PI, fbo, vao, eye, cam, 2)
            segs = inst[:, :, 3]; skipped = inst[:, :, 2]
            skip_ratio = float(skipped.sum() / max(segs.sum(), 1.0)); evals = float(inst[:, :, 0].mean())
            e = rmse(img, ref)
            Image.fromarray((np.clip(img[:, :, :3], 0, 1) * 255).astype(np.uint8)).save(os.path.join(OUT, "%s_%s.png" % (cname, label)))
            diff = np.abs(img[:, :, :3] - ref[:, :, :3]).mean(2)
            Image.fromarray(np.clip(diff * 40 * 255, 0, 255).astype(np.uint8)).save(os.path.join(OUT, "%s_%s_diffx40.png" % (cname, label)))
            print("%-8s %-8s  %-9.3f %-9.2e %-8.1f %-8.1f" % (cname, label, ms, e, skip_ratio * 100, evals))
        print()
    glfw.terminate()
    print("images -> %s (compare *_baseline_diffx40 vs *_gridfix_diffx40)" % OUT)


if __name__ == "__main__":
    main()
