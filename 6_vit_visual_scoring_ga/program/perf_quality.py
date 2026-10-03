"""Joint performance + quality gate for shader optimisation.

Every optimisation must be verified on BOTH axes: HD frame time must fall, and the ViT score must not.
Reports HD/FHD medians (GL_TIME_ELAPSED, across all 12 view geometries) alongside the exact dual
DINOv2+DINOv3 score the GA optimises, using the current best parameters.

  python perf_quality.py                 # measure and print
  python perf_quality.py --save NAME     # also record to results/perf_quality.json under NAME
"""
from __future__ import annotations
import os, sys, json, math, time, statistics as st
import numpy as np, glfw
from OpenGL.GL import *

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); os.chdir(HERE)
import optimize_multiphoto as O          # module scope loads photos, masks, models, targets

BEST = os.path.join(HERE, "..", "results", "multi_best_dual.json")
LEDGER = os.path.join(HERE, "..", "results", "perf_quality.json")
WARMUP, FRAMES = 2, 5
RES = [("HD", 1280, 720), ("FHD", 1920, 1080)]

def main():
    best = json.load(open(BEST, encoding="utf-8"))
    shared, per_photo = best["shared"], best["per_photo"]
    names = [b for b, _, _ in O.PHOTOS]

    glfw.init()
    for k, v in [(glfw.VISIBLE, glfw.FALSE), (glfw.CONTEXT_VERSION_MAJOR, 3),
                 (glfw.CONTEXT_VERSION_MINOR, 3), (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)]:
        glfw.window_hint(k, v)
    win = glfw.create_window(256, 256, "pq", None, None); glfw.make_context_current(win)
    prog = O.progc(); vao = glGenVertexArrays(1); q = int(glGenQueries(1)[0])

    def _spin_warmup(sec=1.5):
        # GPU boost ramps from the 210 MHz idle clock; 2 frames is nowhere near steady 2805 MHz.
        # Draw continuously for ~1.5 s so timings reflect the sustained clock, not the ramp.
        glBindVertexArray(vao); t=time.time()
        while time.time()-t < sec:
            for _ in range(30): glDrawArrays(GL_TRIANGLES,0,3)
            glFinish()

    def target(w, h):
        tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, None)
        fbo = glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
        glViewport(0, 0, w, h); return fbo, tex

    def setup(w, h, geo, cloud):
        glUseProgram(prog); U = lambda n: glGetUniformLocation(prog, n)
        eye = np.array([0, O.HEIGHT, 0], np.float32)
        d = np.array([0.0, math.sin(geo['pitch']), math.cos(geo['pitch'])], np.float32)
        f = d/np.linalg.norm(d); r = np.cross([0, 1, 0], f); r /= np.linalg.norm(r); u = np.cross(f, r)
        cam = np.stack([r, u, f], 1).astype(np.float32)
        sd = np.array([math.cos(geo['sunEl'])*math.sin(geo['sunAz']), math.sin(geo['sunEl']),
                       math.cos(geo['sunEl'])*math.cos(geo['sunAz'])], np.float32)
        glUniform2f(U("uRes"), float(w), float(h)); glUniform3f(U("uEye"), *[float(z) for z in eye])
        glUniformMatrix3fv(U("uCam"), 1, GL_FALSE, cam.T.copy()); glUniform1f(U("uFocal"), float(geo['focal']))
        glUniform3f(U("uSunDir"), *[float(z) for z in sd]); glUniform1f(U("uExposure"), float(cloud['expo']))
        for k, v in shared.items(): glUniform1f(U(k), float(v))
        for k in ("uCovLow", "uCovMid", "uCovHigh", "uTypeAmp", "uTypeSeed"):
            glUniform1f(U(k), float(cloud[k]))
        # 1x1 pre-pass: the ambient probes are constant for the frame, so resolve them once
        glUniform1f(U("uAmbMode"), 0.0)
        amb = []
        for mode in (1.0, 2.0):
            glUniform1f(U("uProbe"), mode); glViewport(0, 0, 1, 1)
            glBindVertexArray(vao); glDrawArrays(GL_TRIANGLES, 0, 3)
            amb.append(np.frombuffer(glReadPixels(0, 0, 1, 1, GL_RGB, GL_FLOAT), np.float32))
        glUniform1f(U("uProbe"), 0.0); glUniform1f(U("uAmbMode"), 1.0)
        glUniform3f(U("uAmbZen"), *[float(z) for z in amb[0]])
        glUniform3f(U("uAmbHor"), *[float(z) for z in amb[1]])
        glViewport(0, 0, w, h)

    def timed(w, h, geo, cloud):
        setup(w, h, geo, cloud); glBindVertexArray(vao)
        glBeginQuery(GL_TIME_ELAPSED, q); glDrawArrays(GL_TRIANGLES, 0, 3); glEndQuery(GL_TIME_ELAPSED)
        avail = GL_FALSE
        while avail == GL_FALSE: avail = glGetQueryObjectiv(q, GL_QUERY_RESULT_AVAILABLE)
        return glGetQueryObjectuiv(q, GL_QUERY_RESULT)/1e6

    fbo0, tex0 = target(1920, 1080); _spin_warmup(1.5)   # bring the GPU to its sustained boost clock
    glDeleteFramebuffers(1, [fbo0]); glDeleteTextures(1, [tex0])
    print("=== performance (median over %d views, GL_TIME_ELAPSED) ===" % len(names))
    perf = {}
    for label, w, h in RES:
        fbo, tex = target(w, h)
        v = []
        for n in names:
            g, c = per_photo[n], per_photo[n]
            for _ in range(WARMUP): timed(w, h, g, c)
            v.append(st.median([timed(w, h, g, c) for _ in range(FRAMES)]))
        med = st.median(v); perf[label] = med
        print("  %-4s %8.2f ms  %6.1f fps   (min %.1f  max %.1f)" % (label, med, 1000/med, min(v), max(v)))
        glDeleteFramebuffers(1, [fbo]); glDeleteTextures(1, [tex])

    # quality: the exact objective the GA uses, at the optimiser's own resolution
    print("=== quality (dual DINOv2+v3, the GA's own objective) ===")
    fbo, tex = target(O.W, O.H)
    O._g["prog"] = prog; O._g["vao"] = vao
    x = np.array(best["x"], float)
    fit = O.full_fitness(x)
    per = []
    xs = O.un_shared(x[:O.DS]); ps = O.split(x)[1]
    for i in range(O.NP): per.append(O.eval_photo(xs, ps[i], i))
    print("  fitness (0.5*mean+0.5*min) = %.4f" % fit)
    print("  per-photo mean = %.4f   min = %.4f" % (float(np.mean(per)), float(np.min(per))))

    if "--save" in sys.argv:
        tag = sys.argv[sys.argv.index("--save")+1]
        led = json.load(open(LEDGER, encoding="utf-8")) if os.path.exists(LEDGER) else {}
        led[tag] = {"hd_ms": perf["HD"], "fhd_ms": perf["FHD"], "fitness": fit,
                    "mean": float(np.mean(per)), "min": float(np.min(per))}
        json.dump(led, open(LEDGER, "w"), indent=2)
        print("\nrecorded as '%s' in results/perf_quality.json" % tag)
        base = led.get("baseline")
        if base and tag != "baseline":
            print("  vs baseline: HD %.2f -> %.2f ms (%.2fx faster)   fitness %.4f -> %.4f (%+.4f)"
                  % (base["hd_ms"], perf["HD"], base["hd_ms"]/perf["HD"], base["fitness"], fit, fit-base["fitness"]))
    glfw.terminate()

if __name__ == "__main__":
    main()
