"""Screen-guided cumulus fit. The photo's per-pixel cloudiness gates the cloud density (silhouette
matches the photo exactly for this fixed view); procedural 3D billow + detail noise shade the interior;
Paper-3 atmosphere renders the sky. GA over sky + lighting + profile + detail to drive masked RMSE low.

Run:  python a3/gpu/fit_cumulus_screen.py [generations]
"""
from __future__ import annotations
import numpy as np, glfw, os, sys, json
from OpenGL.GL import *
from PIL import Image
from scipy.ndimage import gaussian_filter

RESEARCH = "."
OUT = os.path.join(RESEARCH, "results", "cumulus")
SRC = open(os.path.join(RESEARCH, "a3", "gpu", "cumulus_fit.glsl")).read()
VS = ("#version 330 core\nvoid main(){ vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));"
      " gl_Position=vec4(p*2.0-1.0,0.0,1.0); }\n")
TGT = np.load(os.path.join(OUT, "target_img.npy")); MASK = np.load(os.path.join(OUT, "target_mask.npy"))
H, W = TGT.shape[:2]; TGT_B = gaussian_filter(TGT, (3, 3, 0))

P = [("pitch",0.06,0.50),("yaw",-0.30,0.30),("height",0.2,1.2),("focal",0.75,1.25),
     ("stR",0.05,0.40),("stG",0.10,0.50),("stB",0.30,0.90),("shR",0.40,0.90),("shG",0.50,0.95),("shB",0.60,1.00),
     ("sunScale",2.5,7.0),("sunAz",-3.14,3.14),("sunEl",0.35,1.35),("cbase",1.0,1.9),("ctop",2.4,4.4),
     ("covFreq",0.20,0.80),("covLo",0.40,0.62),("covW",0.05,0.22),("dens",2.0,6.5),("amb",0.4,1.3),
     ("expo",0.06,0.60),("offX",0.0,40.0),("offZ",0.0,40.0),("bAmp",0.8,1.9),("bFreq",1.6,3.2),
     ("erode",0.10,0.32),("sigma",6.0,12.0),("sunInt",2.0,25.0),("atmDens",0.6,2.5),("detAmp",0.0,0.40),("detFreq",2.0,6.0),
     ("colR",0.70,1.30),("colG",0.70,1.30),("colB",0.70,1.30),("cloudLo",0.55,1.15),("cloudHi",0.85,1.45),
     ("covGamma",0.60,1.80),("hazeAmp",0.0,3.0),("hazeH",0.10,0.55)]
D = len(P); NAMES = [p[0] for p in P]; LO = np.array([p[1] for p in P]); HI = np.array([p[2] for p in P])
def unpack(x): return LO + np.clip(x, 0, 1)*(HI-LO)
IDX = {n: i for i, (n, _, _) in enumerate(P)}


def sh(src, st):
    s = glCreateShader(st); glShaderSource(s, src); glCompileShader(s)
    if glGetShaderiv(s, GL_COMPILE_STATUS) != GL_TRUE: raise RuntimeError(glGetShaderInfoLog(s).decode())
    return s
def prog():
    p = glCreateProgram(); glAttachShader(p, sh(VS, GL_VERTEX_SHADER)); glAttachShader(p, sh(SRC, GL_FRAGMENT_SHADER)); glLinkProgram(p)
    if glGetProgramiv(p, GL_LINK_STATUS) != GL_TRUE: raise RuntimeError(glGetProgramInfoLog(p).decode())
    return p
def cam_basis(eye, ctr):
    f = ctr-eye; f /= np.linalg.norm(f); r = np.cross([0, 1, 0], f); r /= np.linalg.norm(r); u = np.cross(f, r)
    return np.stack([r, u, f], 1).astype(np.float32)
_gl = {}
def render(x):
    v = unpack(x); pr = _gl["prog"]; glUseProgram(pr)
    (pitch, yaw, height, focal, stR, stG, stB, shR, shG, shB, sunScale, sunAz, sunEl, cbase, ctop,
     covFreq, covLo, covW, dens, amb, expo, offX, offZ, bAmp, bFreq, erode, sigma, sunInt, atmDens, detAmp, detFreq,
     colR, colG, colB, cloudLo, cloudHi, covGamma, hazeAmp, hazeH) = v
    eye = np.array([0, height, 0], np.float32)
    d = np.array([np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)], np.float32)
    cam = cam_basis(eye, eye+d)
    sd = np.array([np.cos(sunEl)*np.sin(sunAz), np.sin(sunEl), np.cos(sunEl)*np.cos(sunAz)], np.float32)
    U = lambda n: glGetUniformLocation(pr, n)
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, _gl["scrtex"]); glUniform1i(U("uScrCov"), 1)
    glUniform1f(U("uUseScr"), 1.0); glUniform1f(U("uUseTex"), 0.0)
    glUniform2f(U("uRes"), float(W), float(H)); glUniform3f(U("uEye"), *[float(z) for z in eye])
    glUniformMatrix3fv(U("uCam"), 1, GL_FALSE, cam.T.copy()); glUniform1f(U("uFocal"), float(focal))
    glUniform3f(U("uSkyTop"), stR, stG, stB); glUniform3f(U("uSkyHor"), shR, shG, shB)
    glUniform3f(U("uSunColV"), 1.75*sunScale, 1.6*sunScale, 1.35*sunScale)
    glUniform3f(U("uGround"), 0.30, 0.38, 0.20); glUniform3f(U("uSunDir"), *[float(z) for z in sd])
    glUniform1f(U("uCBase"), cbase); glUniform1f(U("uCTop"), max(ctop, cbase+0.8)); glUniform1f(U("uCovFreq"), covFreq)
    glUniform1f(U("uCovLo"), covLo); glUniform1f(U("uCovHi"), covLo+covW); glUniform1f(U("uDens"), dens)
    glUniform1f(U("uAmb"), amb); glUniform1f(U("uExposure"), expo); glUniform2f(U("uCovOff"), offX, offZ)
    glUniform1f(U("uBillowAmp"), bAmp); glUniform1f(U("uBillowFreq"), bFreq); glUniform1f(U("uErode"), erode)
    glUniform1f(U("uSigma"), sigma); glUniform1f(U("uSunIntensity"), sunInt); glUniform1f(U("uAtmDens"), atmDens)
    glUniform1f(U("uDetailAmp"), detAmp); glUniform1f(U("uDetailFreq"), detFreq)
    glUniform3f(U("uColGain"), colR, colG, colB)
    glUniform1f(U("uCloudLo"), cloudLo); glUniform1f(U("uCloudHi"), cloudHi); glUniform1f(U("uCovGamma"), covGamma)
    glUniform1f(U("uHazeAmp"), hazeAmp); glUniform1f(U("uHazeH"), hazeH)
    glBindVertexArray(_gl["vao"]); glDrawArrays(GL_TRIANGLES, 0, 3); glFinish()
    img = np.frombuffer(glReadPixels(0, 0, W, H, GL_RGB, GL_FLOAT), np.float32).reshape(H, W, 3)
    return img[::-1].copy()

def loss(x):
    img = render(x)
    d = (img-TGT)[MASK]; full = np.sqrt((d*d).mean())
    db = (gaussian_filter(img, (2.0, 2.0, 0))-gaussian_filter(TGT, (2.0, 2.0, 0)))[MASK]; blur = np.sqrt((db*db).mean())
    return 0.7*full + 0.3*blur
def full_rmse(x):
    img = render(x); d = (img-TGT)[MASK]; return float(np.sqrt((d*d).mean())), img


def main():
    G = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    glfw.init()
    for k, val in [(glfw.VISIBLE, glfw.FALSE), (glfw.CONTEXT_VERSION_MAJOR, 3), (glfw.CONTEXT_VERSION_MINOR, 3),
                   (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)]:
        glfw.window_hint(k, val)
    win = glfw.create_window(W, H, "s", None, None); glfw.make_context_current(win)
    tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, tex); glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0, GL_RGBA, GL_FLOAT, None)
    fbo = glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER, fbo); glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
    glViewport(0, 0, W, H); _gl["prog"] = prog(); _gl["vao"] = glGenVertexArrays(1)
    rng = np.random.default_rng(2)

    # screen cloudiness from the photo (continuous), uploaded flipped for GL bottom-origin sampling
    bright = TGT.mean(2); sat = TGT.max(2)-TGT.min(2)
    cloud = np.clip((bright-0.55)/0.30, 0, 1) * (sat < 0.22) * MASK
    cloud = gaussian_filter(cloud.astype(np.float32), 0.6)
    st = glGenTextures(1); _gl["scrtex"] = st
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, st)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, W, H, 0, GL_RED, GL_FLOAT, np.ascontiguousarray(cloud[::-1]))
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    Image.fromarray((cloud*255).astype(np.uint8)).save(os.path.join(OUT, "screen_cloudiness.png"))

    scr = os.path.join(OUT, "screen_result.json")
    if os.path.exists(scr):                                                        # warm-start from own best
        base = np.array(json.load(open(scr))["x"]); print("warm-start from screen_result.json")
    else:
        base = np.array(json.load(open(os.path.join(OUT, "fit_result.json")))["x"])
        base[IDX["cbase"]] = (1.15-1.0)/(1.9-1.0); base[IDX["ctop"]] = (4.0-2.4)/(4.4-2.4)
    if base.shape[0] < D: base = np.concatenate([base, 0.5*np.ones(D-base.shape[0])])  # pad new params (colGain->1.0)
    fr0, img0 = full_rmse(base); Image.fromarray((np.clip(img0, 0, 1)*255).astype(np.uint8)).save(os.path.join(OUT, "screen_init.png"))
    print("screen mask + init render. init full RMSE %.4f" % fr0)

    N, E = 44, 5
    pop = np.tile(base, (N, 1))
    for i in range(1, N): pop[i] = np.clip(base + 0.12*rng.standard_normal(D), 0, 1)
    fit = np.array([loss(x) for x in pop]); curve = []; best_x = pop[0].copy(); best_l = fit.min()
    for gen in range(G):
        order = np.argsort(fit); pop, fit = pop[order], fit[order]
        if fit[0] < best_l: best_l, best_x = fit[0], pop[0].copy()
        if gen % 5 == 0 or gen == G-1:
            fr, _ = full_rmse(pop[0]); curve.append([gen, float(fit[0]), fr]); print("gen %3d loss %.4f fullRMSE %.4f" % (gen, fit[0], fr))
        newpop = [pop[i].copy() for i in range(E)]; newfit = [fit[i] for i in range(E)]
        sig = 0.10*(1.0-0.6*gen/G)
        while len(newpop) < N:
            def pick():
                a, b = rng.integers(0, N//2, 2); return pop[a] if fit[a] < fit[b] else pop[b]
            p1, p2 = pick(), pick(); m = rng.random(D) < 0.5
            child = np.where(m, p1, p2) + (rng.random(D) < 0.5)*sig*rng.standard_normal(D)
            child = np.clip(child, 0, 1); newpop.append(child); newfit.append(loss(child))
        pop = np.array(newpop); fit = np.array(newfit)

    fr, img = full_rmse(best_x)
    Image.fromarray((np.clip(img, 0, 1)*255).astype(np.uint8)).save(os.path.join(OUT, "screen_best.png"))
    Image.fromarray((np.clip(np.concatenate([TGT, img], 1), 0, 1)*255).astype(np.uint8)).save(os.path.join(OUT, "screen_vs_photo.png"))
    json.dump({"params": {NAMES[i]: float(unpack(best_x)[i]) for i in range(D)}, "full_masked_rmse": fr,
               "curve": curve, "x": best_x.tolist()}, open(os.path.join(OUT, "screen_result.json"), "w"), indent=2)
    print("\nSCREEN-GUIDED best full masked RMSE vs photo: %.4f -> screen_vs_photo.png" % fr)
    glfw.terminate()


if __name__ == "__main__":
    main()
