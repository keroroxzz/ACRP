"""Fair net-performance test: does the grid fix (always-on two-anchor sun) hurt the
equal-quality speedup vs raymarch?

For each of the 12 cameras, at the converged preset, compare:
  * A3 baseline   (SUN_MODE=0, midpoint sun -- has the grid)
  * A3 two-anchor (SUN_MODE=2, TA_WEIGHT=0 -- grid halved)
Conservative equal-quality speedup = cheapest ACTUAL march point no worse than that A3
variant on BOTH full-image and edge RMSE (no interpolation). Also report each variant's
grid score (autocorr peak in the cell-pitch band) so the quality change is visible, not
hidden by RMSE. Reference convergence checked (512/64 vs 1024/128).

Reports P10/median/P90 speedup for each variant + median grid score. -> results/rt_gate0/gate0_gridfix.json
"""
from __future__ import annotations
import json, os, numpy as np, glfw
from OpenGL.GL import *
from scipy.ndimage import uniform_filter

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(RESEARCH, "results", "rt_gate0")
_src = open(os.path.join(HERE, "bench_gate0.py")).read().split("def main()")[0]
_src = _src.replace('A3_SRC = open(os.path.join(HERE, "a3_production.glsl")).read()',
                    'A3_SRC = open(os.path.join(HERE, "a3_production.glsl")).read()\n'
                    'FIX_SRC = open(os.path.join(HERE, "a3_gridfix.glsl")).read()')
exec(_src)


def grid_score(diff):
    sub = diff[80:640, 380:1000]; hp = sub - uniform_filter(sub, 9)
    col = hp.mean(0) - hp.mean(0).mean()
    ac = np.correlate(col, col, 'full')[len(col) - 1:]; ac /= max(ac[0], 1e-12)
    return float(ac[30:60].max())


def conservative_speedup(frontier, e, ee):
    ok = [m for m in frontier if m["rmse"] <= e and m["edge"] <= ee]
    if ok:
        return min(m["ms"] for m in ok), False
    return max(m["ms"] for m in frontier), True   # match beyond frontier (lower bound)


def main():
    os.makedirs(OUT, exist_ok=True); glfw.init()
    for k, v in [(glfw.VISIBLE, glfw.FALSE), (glfw.CONTEXT_VERSION_MAJOR, 3), (glfw.CONTEXT_VERSION_MINOR, 3),
                 (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)]:
        glfw.window_hint(k, v)
    win = glfw.create_window(W, H, "gg", None, None); glfw.make_context_current(win)
    tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0, GL_RGBA, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    fbo = glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0); vao = glGenVertexArrays(1)
    MB = dict(NOCT=4, NOCC=2, DETAIL_AMP=0.1875, NOCC_SUN=4, SUN_BIAS=0.0, SHAPED=0, SUN_INTERVAL=0.0,
              A3_ADAPT=0, CLASS_MARG=0.08, A3_TRGL=5, CAM_STEPS=64, SUN_STEPS=16,
              A3_CAMF=8, A3_CAMGL=5, A3_SUNF=4, A3_SUNGL=3, METHOD=0)
    def mc(**k): d = dict(MB); d.update(k); return d
    march_pts = [(40, 12), (48, 14), (56, 16), (64, 16), (80, 20), (96, 24), (128, 32), (192, 48)]
    conv = dict(A3_CAMF=8, A3_CAMGL=5, A3_SUNF=4, A3_SUNGL=2, NOCC_SUN=4)
    P_ref = prog(MARCH_SRC, mc(CAM_STEPS=512, SUN_STEPS=64)); P_ref2 = prog(MARCH_SRC, mc(CAM_STEPS=1024, SUN_STEPS=128))
    P_march = [(cs, ss, prog(MARCH_SRC, mc(CAM_STEPS=cs, SUN_STEPS=ss))) for cs, ss in march_pts]
    P_base = prog(A3_SRC, dict(conv, INSTRUMENT=0))                       # true frozen production baseline
    P_fix = prog(FIX_SRC, dict(conv, SUN_MODE=2, TA_WEIGHT=0.0, INSTRUMENT=0))  # production + two-anchor sun

    variants = [("baseline", P_base), ("two_anchor", P_fix)]
    sp = {n: [] for n, _ in variants}; gs = {n: [] for n, _ in variants}; ms_all = {n: [] for n, _ in variants}
    ref_conv = []; data = {"gpu": glGetString(GL_RENDERER).decode(), "cameras": {}}
    print("GPU:", data["gpu"], "| converged preset | baseline vs two-anchor grid fix\n")
    print("%-8s %-11s %-8s %-9s %-9s %-8s %s" % ("view", "variant", "ms", "rmse", "grid", "speedup", ""))
    for cname, eye, cam in cameras():
        _, ref = draw(P_ref, fbo, vao, eye, cam, 4); _, ref2 = draw(P_ref2, fbo, vao, eye, cam, 4)
        rc = rmse(ref, ref2); ref_conv.append(rc); em = emask(ref)
        frontier = []
        for cs, ss, p in P_march:
            ms, img = draw(p, fbo, vao, eye, cam, 25)
            frontier.append(dict(steps=cs, sun=ss, ms=ms, rmse=rmse(img, ref), edge=ermse(img, ref, em)))
        rec = {"ref_conv": rc, "variants": {}}
        for n, P in variants:
            ms, img = draw(P, fbo, vao, eye, cam, 40)
            e = rmse(img, ref); ee = ermse(img, ref, em); g = grid_score(np.abs(img[:, :, :3] - ref[:, :, :3]).mean(2))
            mm, beyond = conservative_speedup(frontier, e, ee); s = mm / ms
            sp[n].append(s); gs[n].append(g); ms_all[n].append(ms)
            rec["variants"][n] = dict(ms=ms, rmse=e, edge=ee, grid=g, speedup=s, beyond=beyond)
            print("%-8s %-11s %-8.2f %-9.2e %-9.3f %-8.2f%s" % (cname, n, ms, e, g, s, " *" if beyond else ""))
        data["cameras"][cname] = rec
        print()

    print("-- reference convergence max self-RMSE: %.2e --" % max(ref_conv))
    print("%-11s %-9s %-9s %-9s %-10s %-10s" % ("variant", "P10", "median", "P90", "medGrid", "medMs"))
    data["summary"] = {}
    for n, _ in variants:
        a = np.array(sp[n])
        data["summary"][n] = dict(p10=float(np.percentile(a, 10)), median=float(np.median(a)), p90=float(np.percentile(a, 90)),
                                  med_grid=float(np.median(gs[n])), med_ms=float(np.median(ms_all[n])))
        print("%-11s %-9.2f %-9.2f %-9.2f %-10.3f %-10.2f" % (
            n, np.percentile(a, 10), np.median(a), np.percentile(a, 90), np.median(gs[n]), np.median(ms_all[n])))
    data["ref_conv_max"] = max(ref_conv)
    glfw.terminate(); json.dump(data, open(os.path.join(OUT, "gate0_gridfix.json"), "w"), indent=2)
    print("\n-> %s/gate0_gridfix.json" % OUT)


if __name__ == "__main__":
    main()
