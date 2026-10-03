"""Paper 5 -- PURE PROCEDURAL baseline (no photo guidance).

Identical forward model to fit_photo_guided.py but the cloud placement comes from the procedural
coverage noise (a global domain offset), NOT the photo. Shows the floor: a single global noise offset
cannot place individual clouds where the photo has them, so the masked RMSE floors around ~0.13.

Run:  python fit_procedural.py [generations]   (deps: numpy, scipy, glfw, PyOpenGL, Pillow)
"""
from __future__ import annotations
import numpy as np, glfw, os, sys, json
from OpenGL.GL import *
from PIL import Image
from scipy.ndimage import gaussian_filter

# reuse the target-build, params, GL, and render from the main script (same folder)
import importlib.util
_spec = importlib.util.spec_from_file_location("_g", os.path.join(os.path.dirname(os.path.abspath(__file__)), "fit_photo_guided.py"))
_g = importlib.util.module_from_spec(_spec)
# prevent the main() from running on import
import types
_src = open(_spec.origin).read().split("def main()")[0]
exec(_src, _g.__dict__)
W, H, D, P, NAMES, LO, HI, TGT, MASK, TGT_B, OUT, SHADER, VS = (
    _g.W, _g.H, _g.D, _g.P, _g.NAMES, _g.LO, _g.HI, _g.TGT, _g.MASK, _g.TGT_B, _g.OUT, _g.SHADER, _g.VS)
unpack, cam_basis, sh, prog = _g.unpack, _g.cam_basis, _g.sh, _g.prog
_gl = _g._gl


def render(x):  # same as main but uUseScr=0 (procedural coverage places the clouds)
    v = unpack(x); pr = _gl["prog"]; glUseProgram(pr)
    (pitch, yaw, height, focal, stR, stG, stB, shR, shG, shB, sunScale, sunAz, sunEl, cbase, ctop,
     covFreq, covLo, covW, dens, amb, expo, offX, offZ, bAmp, bFreq, erode, sigma, sunInt, atmDens, detAmp, detFreq,
     colR, colG, colB, cloudLo, cloudHi, covGamma, hazeAmp, hazeH) = v
    eye = np.array([0, height, 0], np.float32)
    d = np.array([np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)], np.float32)
    cam = cam_basis(eye, eye + d)
    sd = np.array([np.cos(sunEl)*np.sin(sunAz), np.sin(sunEl), np.cos(sunEl)*np.cos(sunAz)], np.float32)
    U = lambda n: glGetUniformLocation(pr, n)
    glUniform1f(U("uUseScr"), 0.0); glUniform1f(U("uUseTex"), 0.0)
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
    img = render(x); dd = (img - TGT)[MASK]; full = np.sqrt((dd*dd).mean())
    db = (gaussian_filter(img, (2.0, 2.0, 0)) - TGT_B)[MASK]; return 0.7*full + 0.3*np.sqrt((db*db).mean())
def full_rmse(x):
    img = render(x); dd = (img - TGT)[MASK]; return float(np.sqrt((dd*dd).mean())), img


def main():
    G = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    glfw.init()
    for k, val in [(glfw.VISIBLE, glfw.FALSE), (glfw.CONTEXT_VERSION_MAJOR, 3), (glfw.CONTEXT_VERSION_MINOR, 3),
                   (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)]:
        glfw.window_hint(k, val)
    win = glfw.create_window(W, H, "proc", None, None); glfw.make_context_current(win)
    tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, tex); glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0, GL_RGBA, GL_FLOAT, None)
    fbo = glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER, fbo); glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
    glViewport(0, 0, W, H); _gl["prog"] = prog(); _gl["vao"] = glGenVertexArrays(1)
    rng = np.random.default_rng(0)
    pop = rng.random((48, D)); fit = np.array([loss(x) for x in pop]); best_x, best_l = pop[0].copy(), fit.min()
    for gen in range(G):
        order = np.argsort(fit); pop, fit = pop[order], fit[order]
        if fit[0] < best_l: best_l, best_x = fit[0], pop[0].copy()
        if gen % 10 == 0 or gen == G-1:
            fr, _ = full_rmse(pop[0]); print("gen %3d  loss %.4f  fullRMSE %.4f" % (gen, fit[0], fr))
        newpop = [pop[i].copy() for i in range(5)]; newfit = [fit[i] for i in range(5)]
        sig = 0.12*(1.0 - 0.6*gen/G)
        while len(newpop) < 48:
            def pick():
                a, b = rng.integers(0, 24, 2); return pop[a] if fit[a] < fit[b] else pop[b]
            p1, p2 = pick(), pick(); m = rng.random(D) < 0.5
            child = np.clip(np.where(m, p1, p2) + (rng.random(D) < 0.5)*sig*rng.standard_normal(D), 0, 1)
            newpop.append(child); newfit.append(loss(child))
        pop = np.array(newpop); fit = np.array(newfit)
    fr, img = full_rmse(best_x)
    Image.fromarray((np.clip(np.concatenate([TGT, img], 1), 0, 1)*255).astype(np.uint8)).save(os.path.join(OUT, "procedural_vs_photo.png"))
    print("\nPROCEDURAL (no photo guidance) best masked RMSE: %.4f  (floors ~0.13) -> out/procedural_vs_photo.png" % fr)
    glfw.terminate()


if __name__ == "__main__":
    main()
