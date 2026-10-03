"""Paper 5 -- render the fitted scene from out/best_params.json and save a labelled
photo-vs-fit comparison (sky region) with the masked RMSE.  Run:  python render_final.py
"""
from __future__ import annotations
import numpy as np, glfw, os, json, importlib.util
from OpenGL.GL import *
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("_g", os.path.join(HERE, "fit_photo_guided.py"))
_g = importlib.util.module_from_spec(_spec)
exec(open(_spec.origin).read().split("def main()")[0], _g.__dict__)
W, H, D, OUT, TGT, MASK, CLOUD, render, unpack = _g.W, _g.H, _g.D, _g.OUT, _g.TGT, _g.MASK, _g.CLOUD, _g.render, _g.unpack
_gl = _g._gl


def main():
    glfw.init()
    for k, val in [(glfw.VISIBLE, glfw.FALSE), (glfw.CONTEXT_VERSION_MAJOR, 3), (glfw.CONTEXT_VERSION_MINOR, 3),
                   (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)]:
        glfw.window_hint(k, val)
    win = glfw.create_window(W, H, "r", None, None); glfw.make_context_current(win)
    tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, tex); glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0, GL_RGBA, GL_FLOAT, None)
    fbo = glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER, fbo); glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
    glViewport(0, 0, W, H); _gl["prog"] = _g.prog(); _gl["vao"] = glGenVertexArrays(1)
    st = glGenTextures(1); _gl["scrtex"] = st; glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, st)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, W, H, 0, GL_RED, GL_FLOAT, np.ascontiguousarray(CLOUD[::-1]))
    for pn in (GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER): glTexParameteri(GL_TEXTURE_2D, pn, GL_LINEAR)
    for pn in (GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T): glTexParameteri(GL_TEXTURE_2D, pn, GL_CLAMP_TO_EDGE)

    x = np.array(json.load(open(os.path.join(OUT, "best_params.json")))["x"])
    if x.shape[0] < D: x = np.concatenate([x, 0.5*np.ones(D - x.shape[0])])
    img = render(x); rmse = float(np.sqrt(((img - TGT)[MASK]**2).mean()))
    h2 = int(0.54*H)
    def lab(a, t):
        im = Image.fromarray((np.clip(a, 0, 1)*255).astype(np.uint8)).convert("RGB"); d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 470, 20], fill=(0, 0, 0)); d.text((6, 5), t, fill=(255, 255, 120)); return np.asarray(im)
    out = np.concatenate([lab(TGT[:h2], "REFERENCE PHOTO (sky region)"),
                          lab(img[:h2], "FITTED PROCEDURAL RENDER  masked RMSE %.3f" % rmse)], 0)
    Image.fromarray(out).save(os.path.join(OUT, "FINAL_photo_vs_fit.png"))
    print("masked RMSE %.4f -> out/FINAL_photo_vs_fit.png" % rmse)
    glfw.terminate()


if __name__ == "__main__":
    main()
