"""Photo-guided cumulus fit. Stage 2: bake a world-space coverage map from the photo (project the
photo's cloud brightness onto the cloud-base plane using the stage-1 fitted camera), so clouds land
where the photo has them; keep procedural 3D billow + high-freq detail for texture; then GA the
lighting / colour / detail / density (camera + cbase fixed so the bake stays aligned).

Run:  python a3/gpu/fit_cumulus_guided.py [generations]
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
_tb = np.where(MASK[:, :, None], TGT, np.nan).mean(2)
TROW = np.nanmean(_tb, axis=1); TCOL = np.nanmean(_tb, axis=0)

P = [("pitch",0.06,0.50),("yaw",-0.30,0.30),("height",0.2,1.2),("focal",0.75,1.25),
     ("stR",0.05,0.40),("stG",0.10,0.50),("stB",0.30,0.90),("shR",0.40,0.90),("shG",0.50,0.95),("shB",0.60,1.00),
     ("sunScale",2.5,7.0),("sunAz",-3.14,3.14),("sunEl",0.35,1.35),("cbase",1.0,1.9),("ctop",2.4,4.4),
     ("covFreq",0.20,0.80),("covLo",0.40,0.62),("covW",0.05,0.22),("dens",2.0,5.5),("amb",0.4,1.3),
     ("expo",0.06,0.60),("offX",0.0,40.0),("offZ",0.0,40.0),("bAmp",0.8,1.9),("bFreq",1.6,3.2),
     ("erode",0.10,0.32),("sigma",6.0,11.0),("sunInt",2.0,25.0),("atmDens",0.6,2.5),("detAmp",0.0,0.40),("detFreq",2.0,6.0)]
D = len(P); NAMES = [p[0] for p in P]; LO = np.array([p[1] for p in P]); HI = np.array([p[2] for p in P])
def unpack(x): return LO + np.clip(x, 0, 1)*(HI-LO)
IDX = {n: i for i, (n, _, _) in enumerate(P)}
# fixed during stage 2 (camera + cloud base, needed for bake alignment); covLo/covW retuned; offsets keep break-up
FIXED = [IDX[n] for n in ("pitch", "yaw", "height", "focal", "cbase")]
FREE = [i for i in range(D) if i not in FIXED]


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


def bake_coverage(v):
    pitch, yaw, height, focal, cbase = v[0], v[1], v[2], v[3], v[13]
    eye = np.array([0, height, 0], np.float32)
    fwd = np.array([np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)], np.float32)
    cam = cam_basis(eye, eye+fwd)
    bright = TGT.mean(2); sat = TGT.max(2)-TGT.min(2)
    cloud = np.clip((bright-0.56)/0.26, 0, 1) * (sat < 0.20) * MASK
    ys, xs = np.mgrid[0:H, 0:W]
    fx = (xs+0.5-0.5*W)/H; fy = ((H-1-ys)+0.5-0.5*H)/H
    dirs = np.stack([fx, fy, np.full_like(fx, focal)], -1); dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    rd = dirs @ cam.T; rdy = rd[:, :, 1]
    valid = rdy > 0.02
    t = (cbase-height)/np.where(valid, rdy, 1.0); valid &= (t > 0) & (t < 60)
    wx = t*rd[:, :, 0]; wz = t*rd[:, :, 2]
    vv = valid & MASK
    xmin, xmax = np.percentile(wx[vv], [1, 99]); zmin, zmax = np.percentile(wz[vv], [1, 99])
    R = 256; cov = np.zeros((R, R)); cnt = np.zeros((R, R))
    ui = ((wx-xmin)/(xmax-xmin)*(R-1)); vi = ((wz-zmin)/(zmax-zmin)*(R-1))
    good = vv & (ui >= 0) & (ui < R) & (vi >= 0) & (vi < R)
    uii = ui[good].astype(int); vii = vi[good].astype(int)
    np.add.at(cov, (vii, uii), cloud[good]); np.add.at(cnt, (vii, uii), 1.0)
    cov = cov/np.maximum(cnt, 1.0)
    w = (cnt > 0).astype(float)
    num = gaussian_filter(cov*w, 3.0); den = gaussian_filter(w, 3.0)
    filled = np.where(cnt > 0, cov, num/np.maximum(den, 1e-3))
    filled = gaussian_filter(filled, 1.0).astype(np.float32)
    return filled, (float(xmin), float(zmin)), (float(xmax-xmin), float(zmax-zmin))


def upload_tex(cov):
    tid = _gl.get("covtex")
    if tid is None: tid = glGenTextures(1); _gl["covtex"] = tid
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, tid)
    R = cov.shape[0]
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, R, R, 0, GL_RED, GL_FLOAT, np.ascontiguousarray(cov))
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)


def render(x):
    v = unpack(x); pr = _gl["prog"]; glUseProgram(pr)
    (pitch, yaw, height, focal, stR, stG, stB, shR, shG, shB, sunScale, sunAz, sunEl, cbase, ctop,
     covFreq, covLo, covW, dens, amb, expo, offX, offZ, bAmp, bFreq, erode, sigma, sunInt, atmDens, detAmp, detFreq) = v
    eye = np.array([0, height, 0], np.float32)
    d = np.array([np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)], np.float32)
    cam = cam_basis(eye, eye+d)
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
    glUniform1f(U("uUseTex"), _gl.get("usetex", 0.0))
    glUniform2f(U("uCovMin"), *_gl.get("covmin", (0.0, 0.0))); glUniform2f(U("uCovSize"), *_gl.get("covsize", (1.0, 1.0)))
    glUniform1i(U("uCovTex"), 0)
    glBindVertexArray(_gl["vao"]); glDrawArrays(GL_TRIANGLES, 0, 3); glFinish()
    img = np.frombuffer(glReadPixels(0, 0, W, H, GL_RGB, GL_FLOAT), np.float32).reshape(H, W, 3)
    return img[::-1].copy()

def loss(x):
    img = render(x)
    d = (img-TGT)[MASK]; full = np.sqrt((d*d).mean())
    db = (gaussian_filter(img, (3, 3, 0))-TGT_B)[MASK]; blur = np.sqrt((db*db).mean())
    bm = np.where(MASK[:, :, None], img, np.nan).mean(2)
    prow = np.nanmean(bm, axis=1); pcol = np.nanmean(bm, axis=0)
    return 0.5*full + 0.35*blur + 0.15*np.sqrt(np.nanmean((prow-TROW)**2)+np.nanmean((pcol-TCOL)**2))
def full_rmse(x):
    img = render(x); d = (img-TGT)[MASK]; return float(np.sqrt((d*d).mean())), img


def main():
    G = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    glfw.init()
    for k, val in [(glfw.VISIBLE, glfw.FALSE), (glfw.CONTEXT_VERSION_MAJOR, 3), (glfw.CONTEXT_VERSION_MINOR, 3),
                   (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)]:
        glfw.window_hint(k, val)
    win = glfw.create_window(W, H, "g", None, None); glfw.make_context_current(win)
    tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, tex); glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0, GL_RGBA, GL_FLOAT, None)
    fbo = glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER, fbo); glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
    glViewport(0, 0, W, H); _gl["prog"] = prog(); _gl["vao"] = glGenVertexArrays(1)
    rng = np.random.default_rng(1)

    base = np.array(json.load(open(os.path.join(OUT, "fit_result.json")))["x"])   # stage-1 best
    cov, cmin, csize = bake_coverage(unpack(base))
    Image.fromarray((np.clip(cov, 0, 1)*255).astype(np.uint8)).save(os.path.join(OUT, "bake_covmap.png"))
    upload_tex(cov); _gl["usetex"] = 1.0; _gl["covmin"] = cmin; _gl["covsize"] = csize
    # tune covLo/covW for the texture-domain values (map is cloudiness ~[0,1])
    base[IDX["covLo"]] = (0.18-0.40)/(0.62-0.40); base[IDX["covW"]] = (0.14-0.05)/(0.22-0.05)
    fr0, img0 = full_rmse(base)
    Image.fromarray((np.clip(img0, 0, 1)*255).astype(np.uint8)).save(os.path.join(OUT, "guided_init.png"))
    print("baked map + init render. init full RMSE %.4f" % fr0)

    # GA over FREE params only (camera + cbase fixed from base)
    N, E = 44, 5
    pop = np.tile(base, (N, 1))
    for i in range(N):
        pop[i, FREE] = np.clip(base[FREE] + (0.10 if i else 0.0)*rng.standard_normal(len(FREE)), 0, 1)
    fit = np.array([loss(x) for x in pop]); curve = []
    best_x = pop[0].copy(); best_l = fit.min()
    for gen in range(G):
        order = np.argsort(fit); pop, fit = pop[order], fit[order]
        if fit[0] < best_l: best_l, best_x = fit[0], pop[0].copy()
        if gen % 5 == 0 or gen == G-1:
            fr, _ = full_rmse(pop[0]); curve.append([gen, float(fit[0]), fr]); print("gen %3d loss %.4f fullRMSE %.4f" % (gen, fit[0], fr))
        newpop = [pop[i].copy() for i in range(E)]; newfit = [fit[i] for i in range(E)]
        sig = 0.11*(1.0-0.6*gen/G)
        while len(newpop) < N:
            def pick():
                a, b = rng.integers(0, N//2, 2); return pop[a] if fit[a] < fit[b] else pop[b]
            p1, p2 = pick(), pick(); child = base.copy()
            m = rng.random(len(FREE)) < 0.5
            child[FREE] = np.where(m, p1[FREE], p2[FREE])
            mut = (rng.random(len(FREE)) < 0.5)*sig*rng.standard_normal(len(FREE))
            child[FREE] = np.clip(child[FREE]+mut, 0, 1)
            newpop.append(child); newfit.append(loss(child))
        pop = np.array(newpop); fit = np.array(newfit)

    fr, img = full_rmse(best_x)
    Image.fromarray((np.clip(img, 0, 1)*255).astype(np.uint8)).save(os.path.join(OUT, "guided_best.png"))
    Image.fromarray((np.clip(np.concatenate([TGT, img], 1), 0, 1)*255).astype(np.uint8)).save(os.path.join(OUT, "guided_vs_photo.png"))
    json.dump({"params": {NAMES[i]: float(unpack(best_x)[i]) for i in range(D)}, "full_masked_rmse": fr,
               "curve": curve, "x": best_x.tolist(), "covmin": cmin, "covsize": csize},
              open(os.path.join(OUT, "guided_result.json"), "w"), indent=2)
    print("\nGUIDED best full masked RMSE vs photo: %.4f -> guided_vs_photo.png" % fr)
    glfw.terminate()


if __name__ == "__main__":
    main()
