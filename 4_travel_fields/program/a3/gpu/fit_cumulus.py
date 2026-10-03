"""Inverse-fit the procedural cumulus scene (Paper-3 atmosphere + detail noise) to the photo.

Genetic algorithm over scene params (camera, sun, atmosphere sunIntensity/density, cloud
coverage/distribution/noise-offset, billow + high-freq detail, density, exposure, cloud ambient).
Loss = multi-scale masked RMSE (full + blurred) + a coverage-profile term (row/col masked brightness)
so cloud DISTRIBUTION is matched, not just colour. Saves best render + side-by-side + RMSE curve.

Run:  python a3/gpu/fit_cumulus.py [generations]
"""
from __future__ import annotations
import numpy as np, glfw, os, sys, json
from OpenGL.GL import *
from PIL import Image
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__)); RESEARCH = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(RESEARCH, "results", "cumulus")
SRC = open(os.path.join(HERE, "cumulus_fit.glsl")).read()
VS = ("#version 330 core\nvoid main(){ vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));"
      " gl_Position=vec4(p*2.0-1.0,0.0,1.0); }\n")
TGT = np.load(os.path.join(OUT, "target_img.npy")); MASK = np.load(os.path.join(OUT, "target_mask.npy"))
H, W = TGT.shape[:2]; TGT_B = gaussian_filter(TGT, (3, 3, 0))
_tb = np.where(MASK[:, :, None], TGT, np.nan).mean(2)
TROW = np.nanmean(_tb, axis=1); TCOL = np.nanmean(_tb, axis=0)

P = [
    ("pitch", 0.06, 0.50), ("yaw", -0.30, 0.30), ("height", 0.2, 1.2), ("focal", 0.75, 1.25),
    ("stR", 0.05, 0.40), ("stG", 0.10, 0.50), ("stB", 0.30, 0.90),
    ("shR", 0.40, 0.90), ("shG", 0.50, 0.95), ("shB", 0.60, 1.00),
    ("sunScale", 2.5, 7.0), ("sunAz", -3.14, 3.14), ("sunEl", 0.35, 1.35),
    ("cbase", 1.0, 1.9), ("ctop", 2.4, 4.4), ("covFreq", 0.20, 0.80),
    ("covLo", 0.40, 0.62), ("covW", 0.05, 0.22), ("dens", 2.0, 5.5), ("amb", 0.4, 1.3),
    ("expo", 0.06, 0.60), ("offX", 0.0, 40.0), ("offZ", 0.0, 40.0),
    ("bAmp", 0.8, 1.9), ("bFreq", 1.6, 3.2), ("erode", 0.10, 0.32), ("sigma", 6.0, 11.0),
    ("sunInt", 2.0, 25.0), ("atmDens", 0.6, 2.5), ("detAmp", 0.0, 0.40), ("detFreq", 2.0, 6.0),
]
D = len(P); LO = np.array([p[1] for p in P]); HI = np.array([p[2] for p in P])
def unpack(x): return LO + np.clip(x, 0, 1) * (HI - LO)


def sh(src, st):
    s = glCreateShader(st); glShaderSource(s, src); glCompileShader(s)
    if glGetShaderiv(s, GL_COMPILE_STATUS) != GL_TRUE: raise RuntimeError(glGetShaderInfoLog(s).decode())
    return s
def prog():
    p = glCreateProgram(); glAttachShader(p, sh(VS, GL_VERTEX_SHADER)); glAttachShader(p, sh(SRC, GL_FRAGMENT_SHADER)); glLinkProgram(p)
    if glGetProgramiv(p, GL_LINK_STATUS) != GL_TRUE: raise RuntimeError(glGetProgramInfoLog(p).decode())
    return p
def cam_basis(eye, ctr):
    f = ctr - eye; f /= np.linalg.norm(f); r = np.cross([0, 1, 0], f); r /= np.linalg.norm(r); u = np.cross(f, r)
    return np.stack([r, u, f], 1).astype(np.float32)
_gl = {}
def render(x):
    v = unpack(x); pr = _gl["prog"]; glUseProgram(pr)
    (pitch, yaw, height, focal, stR, stG, stB, shR, shG, shB, sunScale, sunAz, sunEl,
     cbase, ctop, covFreq, covLo, covW, dens, amb, expo, offX, offZ, bAmp, bFreq, erode, sigma,
     sunInt, atmDens, detAmp, detFreq) = v
    eye = np.array([0, height, 0], np.float32)
    d = np.array([np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)], np.float32)
    cam = cam_basis(eye, eye + d)
    sd = np.array([np.cos(sunEl)*np.sin(sunAz), np.sin(sunEl), np.cos(sunEl)*np.cos(sunAz)], np.float32)
    U = lambda n: glGetUniformLocation(pr, n)
    glUniform2f(U("uRes"), float(W), float(H)); glUniform3f(U("uEye"), *[float(z) for z in eye])
    glUniformMatrix3fv(U("uCam"), 1, GL_FALSE, cam.T.copy()); glUniform1f(U("uFocal"), float(focal))
    glUniform3f(U("uSkyTop"), stR, stG, stB); glUniform3f(U("uSkyHor"), shR, shG, shB)
    glUniform3f(U("uSunColV"), 1.75*sunScale, 1.6*sunScale, 1.35*sunScale)
    glUniform3f(U("uGround"), 0.30, 0.38, 0.20); glUniform3f(U("uSunDir"), *[float(z) for z in sd])
    glUniform1f(U("uCBase"), cbase); glUniform1f(U("uCTop"), max(ctop, cbase+0.6)); glUniform1f(U("uCovFreq"), covFreq)
    glUniform1f(U("uCovLo"), covLo); glUniform1f(U("uCovHi"), covLo+covW); glUniform1f(U("uDens"), dens)
    glUniform1f(U("uAmb"), amb); glUniform1f(U("uExposure"), expo); glUniform2f(U("uCovOff"), offX, offZ)
    glUniform1f(U("uBillowAmp"), bAmp); glUniform1f(U("uBillowFreq"), bFreq); glUniform1f(U("uErode"), erode)
    glUniform1f(U("uSigma"), sigma); glUniform1f(U("uSunIntensity"), sunInt); glUniform1f(U("uAtmDens"), atmDens)
    glUniform1f(U("uDetailAmp"), detAmp); glUniform1f(U("uDetailFreq"), detFreq)
    glBindVertexArray(_gl["vao"]); glDrawArrays(GL_TRIANGLES, 0, 3); glFinish()
    img = np.frombuffer(glReadPixels(0, 0, W, H, GL_RGB, GL_FLOAT), np.float32).reshape(H, W, 3)
    return img[::-1].copy()

def loss(x):
    img = render(x)
    d = (img - TGT)[MASK]; full = np.sqrt((d*d).mean())
    db = (gaussian_filter(img, (3, 3, 0)) - TGT_B)[MASK]; blur = np.sqrt((db*db).mean())
    bm = np.where(MASK[:, :, None], img, np.nan).mean(2)
    prow = np.nanmean(bm, axis=1); pcol = np.nanmean(bm, axis=0)
    pr = np.nanmean((prow - TROW)**2); pc = np.nanmean((pcol - TCOL)**2)
    return 0.45*blur + 0.35*full + 0.20*np.sqrt(pr + pc)
def full_rmse(x):
    img = render(x); d = (img - TGT)[MASK]; return float(np.sqrt((d*d).mean())), img


def main():
    G = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    glfw.init()
    for k, v in [(glfw.VISIBLE, glfw.FALSE), (glfw.CONTEXT_VERSION_MAJOR, 3), (glfw.CONTEXT_VERSION_MINOR, 3),
                 (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)]:
        glfw.window_hint(k, v)
    win = glfw.create_window(W, H, "fit", None, None); glfw.make_context_current(win)
    tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, tex); glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0, GL_RGBA, GL_FLOAT, None)
    fbo = glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER, fbo); glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
    glViewport(0, 0, W, H); _gl["prog"] = prog(); _gl["vao"] = glGenVertexArrays(1)
    rng = np.random.default_rng(0)

    # --- genetic algorithm ---
    N, E = 48, 5
    pop = rng.random((N, D))
    seedf = os.path.join(OUT, "seed_x.npy")
    if os.path.exists(seedf):                       # inject the calibrated seed + mutated copies
        s = np.load(seedf); pop[0] = s
        for j in range(1, 6): pop[j] = np.clip(s + 0.06*rng.standard_normal(D), 0, 1)
    bestf = os.path.join(OUT, "fit_result.json")    # warm-start from a previous best if present
    if os.path.exists(bestf):
        try:
            xb = np.array(json.load(open(bestf))["x"]); pop[6] = xb
            for j in range(7, 11): pop[j] = np.clip(xb + 0.05*rng.standard_normal(D), 0, 1)
        except Exception: pass
    fit = np.array([loss(x) for x in pop])
    curve = []
    for gen in range(G):
        order = np.argsort(fit); pop, fit = pop[order], fit[order]
        if gen % 5 == 0 or gen == G-1:
            fr, _ = full_rmse(pop[0]); curve.append([gen, float(fit[0]), fr])
            print("gen %3d  loss %.4f  fullRMSE %.4f" % (gen, fit[0], fr))
        newpop = [pop[i] for i in range(E)]              # elitism
        newfit = [fit[i] for i in range(E)]
        sig = 0.12 * (1.0 - 0.6*gen/G)                   # mutation cooling
        while len(newpop) < N:
            # tournament selection from the better half
            def pick():
                a, b = rng.integers(0, N//2, 2); return pop[a] if fit[a] < fit[b] else pop[b]
            p1, p2 = pick(), pick()
            mask = rng.random(D) < 0.5
            child = np.where(mask, p1, p2)               # uniform crossover
            child = child + (rng.random(D) < 0.5) * sig * rng.standard_normal(D)  # sparse Gaussian mutation
            child = np.clip(child, 0, 1)
            newpop.append(child); newfit.append(loss(child))
        pop = np.array(newpop); fit = np.array(newfit)

    order = np.argsort(fit); best = pop[order[0]]
    fr, img = full_rmse(best)
    Image.fromarray((np.clip(img, 0, 1)*255).astype(np.uint8)).save(os.path.join(OUT, "fit_best.png"))
    comp = np.concatenate([TGT, img], 1)
    Image.fromarray((np.clip(comp, 0, 1)*255).astype(np.uint8)).save(os.path.join(OUT, "fit_vs_photo.png"))
    json.dump({"params": {P[i][0]: float(unpack(best)[i]) for i in range(D)}, "full_masked_rmse": fr,
               "curve": curve, "x": best.tolist()}, open(os.path.join(OUT, "fit_result.json"), "w"), indent=2)
    print("\nBEST full masked sky RMSE vs photo: %.4f -> fit_best.png / fit_vs_photo.png" % fr)
    glfw.terminate()


if __name__ == "__main__":
    main()
