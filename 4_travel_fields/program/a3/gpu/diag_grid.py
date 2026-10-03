"""Diagnose the true cause of the cell-lattice artifact on the inside view.

Grid-strength metric: high-pass the |A3 - ref| diff (subtract a box-blurred copy), then take
the column-profile autocorrelation peak in the cell-pitch band (30-60px). Higher = stronger grid.

Sweep the knobs that isolate causes:
  * A3_CAMGL (camera quadrature order): if the grid fades as CAMGL rises -> camera per-cell quadrature.
  * A3_SUNF  (sun cell fineness) and A3_SUNGL: sun-shadow granularity.
  * A3_CAMF  (camera cell fineness): if finer camera cells move the pitch / fade it -> cell structure.
Baseline shader (midpoint skip) is fine here; the artifact is independent of the skip test.
"""
from __future__ import annotations
import numpy as np, glfw, os
from OpenGL.GL import *
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(RESEARCH, "results", "rt_gridfix")
exec(open(os.path.join(HERE, "bench_gate0.py")).read().split("def main()")[0])


def grid_score(diff):
    sub = diff[80:640, 380:1000]
    # high-pass: remove low-freq cloud envelope
    from scipy.ndimage import uniform_filter
    hp = sub - uniform_filter(sub, 9)
    col = hp.mean(0); col = col - col.mean()
    ac = np.correlate(col, col, 'full')[len(col) - 1:]
    ac = ac / max(ac[0], 1e-12)
    band = ac[30:60]
    return float(band.max())


def main():
    os.makedirs(OUT, exist_ok=True)
    glfw.init()
    for k, v in [(glfw.VISIBLE, glfw.FALSE), (glfw.CONTEXT_VERSION_MAJOR, 3), (glfw.CONTEXT_VERSION_MINOR, 3),
                 (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)]:
        glfw.window_hint(k, v)
    win = glfw.create_window(W, H, "d", None, None); glfw.make_context_current(win)
    tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0, GL_RGBA, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    fbo = glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0); vao = glGenVertexArrays(1)
    MB = dict(NOCT=4, NOCC=2, DETAIL_AMP=0.1875, NOCC_SUN=4, SUN_BIAS=0.0, SHAPED=0, SUN_INTERVAL=0.0,
              A3_ADAPT=0, CLASS_MARG=0.08, A3_TRGL=5, CAM_STEPS=64, SUN_STEPS=16,
              A3_CAMF=8, A3_CAMGL=5, A3_SUNF=4, A3_SUNGL=3, METHOD=0)
    def mc(**k): d = dict(MB); d.update(k); return d
    cams = {c[0]: c for c in cameras()}
    _, eye, cam = cams["inside"]
    P_ref = prog(MARCH_SRC, mc(CAM_STEPS=512, SUN_STEPS=64))
    _, ref = draw(P_ref, fbo, vao, eye, cam, 4)

    configs = [
        ("base CAMGL2 SUNF4", dict(A3_CAMF=8, A3_CAMGL=2, A3_SUNF=4, A3_SUNGL=2, NOCC_SUN=4)),
        ("CAMGL5",            dict(A3_CAMF=8, A3_CAMGL=5, A3_SUNF=4, A3_SUNGL=2, NOCC_SUN=4)),
        ("CAMGL8",            dict(A3_CAMF=8, A3_CAMGL=8, A3_SUNF=4, A3_SUNGL=2, NOCC_SUN=4)),
        ("CAMGL5 SUNF8",      dict(A3_CAMF=8, A3_CAMGL=5, A3_SUNF=8, A3_SUNGL=3, NOCC_SUN=4)),
        ("CAMGL5 SUNF16",     dict(A3_CAMF=8, A3_CAMGL=5, A3_SUNF=16, A3_SUNGL=3, NOCC_SUN=4)),
        ("CAMF16 CAMGL5",     dict(A3_CAMF=16, A3_CAMGL=5, A3_SUNF=8, A3_SUNGL=3, NOCC_SUN=4)),
    ]
    print("inside view -- grid-strength (autocorr peak in 30-60px band; higher=stronger grid)\n")
    print("%-20s %-8s %-10s" % ("config", "rmse", "gridScore"))
    for label, kw in configs:
        _, img = draw(prog(A3_SRC, dict(kw, INSTRUMENT=0)), fbo, vao, eye, cam, 3)
        diff = np.abs(img[:, :, :3] - ref[:, :, :3]).mean(2)
        gs = grid_score(diff); e = rmse(img, ref)
        Image.fromarray(np.clip(diff * 40 * 255, 0, 255).astype(np.uint8)).save(
            os.path.join(OUT, "diag_inside_%s.png" % label.replace(" ", "_")))
        print("%-20s %-8.2e %-10.3f" % (label, e, gs))
    glfw.terminate()
    print("\ndiff images -> %s/diag_inside_*.png" % OUT)


if __name__ == "__main__":
    main()
