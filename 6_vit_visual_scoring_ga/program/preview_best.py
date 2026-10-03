"""Render the photo-vs-render comparison grid for the CURRENT shader and the CURRENT best params,
without running the GA. Use this to judge a shader change visually before spending a fit.

  python preview_best.py                 # all photos of the 12-photo set, -> ../results/preview_best.png
  python preview_best.py 1 3 11          # only those indices
  BEST=../results/multi_best_dual_p0.json python preview_best.py     # a different params file

Run with the py3.7 GL interpreter or the dino env (no torch needed).
"""
import os, sys, json, math
import numpy as np, glfw
from OpenGL.GL import *
from PIL import Image, ImageDraw

HERE   = os.path.dirname(os.path.abspath(__file__))
PDIR   = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reference_photos"))
OUT    = os.path.join(HERE, "..", "results")
SHADER = os.path.join(HERE, "cumulus_multilayer.glsl")
BEST   = os.environ.get("BEST", os.path.join(OUT, "multi_best_dual.json"))
VS = ("#version 330 core\nvoid main(){ vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));"
      " gl_Position=vec4(p*2.0-1.0,0.0,1.0);}\n")
W, H = 420, 315

def _sh(src, kind):
    o = glCreateShader(kind); glShaderSource(o, src); glCompileShader(o)
    if glGetShaderiv(o, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(o).decode("utf-8", "replace"))
    return o

def main():
    best = json.load(open(BEST, encoding="utf-8"))
    names = sorted(best["per_photo"].keys())
    want = [int(a) for a in sys.argv[1:]]
    if want: names = [names[i] for i in want]
    glfw.init()
    for k, v in ((glfw.VISIBLE, glfw.FALSE), (glfw.CONTEXT_VERSION_MAJOR, 3),
                 (glfw.CONTEXT_VERSION_MINOR, 3), (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)):
        glfw.window_hint(k, v)
    win = glfw.create_window(64, 64, "prev", None, None); glfw.make_context_current(win)
    prog = glCreateProgram()
    glAttachShader(prog, _sh(VS, GL_VERTEX_SHADER))
    glAttachShader(prog, _sh(open(SHADER, encoding="utf-8").read(), GL_FRAGMENT_SHADER))
    glLinkProgram(prog)
    if glGetProgramiv(prog, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(prog).decode("utf-8", "replace"))
    tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0, GL_RGBA, GL_FLOAT, None)
    fbo = glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
    glViewport(0, 0, W, H); vao = glGenVertexArrays(1); glUseProgram(prog)
    U = lambda n: glGetUniformLocation(prog, n)

    rows = []
    for name in names:
        pp = best["per_photo"][name]
        d = np.array([0.0, math.sin(pp["pitch"]), math.cos(pp["pitch"])], np.float32)
        f = d/np.linalg.norm(d); r = np.cross([0, 1, 0], f); r /= np.linalg.norm(r)
        cam = np.stack([r, np.cross(f, r), f], 1).astype(np.float32)
        sd = np.array([math.cos(pp["sunEl"])*math.sin(pp["sunAz"]), math.sin(pp["sunEl"]),
                       math.cos(pp["sunEl"])*math.cos(pp["sunAz"])], np.float32)
        glUniform2f(U("uRes"), float(W), float(H)); glUniform3f(U("uEye"), 0.0, 0.06, 0.0)
        glUniformMatrix3fv(U("uCam"), 1, GL_FALSE, cam.T.copy())
        glUniform1f(U("uFocal"), float(pp["focal"]))
        glUniform3f(U("uSunDir"), *[float(z) for z in sd])
        glUniform1f(U("uExposure"), float(pp["expo"]))
        for k, v in best["shared"].items(): glUniform1f(U(k), float(v))
        for k in ("uCovLow", "uCovMid", "uCovHigh", "uTypeAmp", "uTypeSeed"):
            glUniform1f(U(k), float(pp[k]))
        glBindVertexArray(vao)
        glUniform1f(U("uAmbMode"), 0.0)                        # 1x1 ambient pre-pass
        glUniform1f(U("uProbe"), 1.0); glDrawArrays(GL_TRIANGLES, 0, 3)
        zen = np.frombuffer(glReadPixels(0, 0, 1, 1, GL_RGB, GL_FLOAT), np.float32)[:3].copy()
        glUniform1f(U("uProbe"), 2.0); glDrawArrays(GL_TRIANGLES, 0, 3)
        hor = np.frombuffer(glReadPixels(0, 0, 1, 1, GL_RGB, GL_FLOAT), np.float32)[:3].copy()
        glUniform1f(U("uProbe"), 0.0); glUniform1f(U("uAmbMode"), 1.0)
        glUniform3f(U("uAmbZen"), *[float(z) for z in zen])
        glUniform3f(U("uAmbHor"), *[float(z) for z in hor])
        glDrawArrays(GL_TRIANGLES, 0, 3); glFinish()
        rnd = np.clip(np.frombuffer(glReadPixels(0, 0, W, H, GL_RGB, GL_FLOAT),
                                    np.float32).reshape(H, W, 3)[::-1], 0, 1)
        jpg = os.path.join(PDIR, name + ".jpg")
        ph = (np.asarray(Image.open(jpg).convert("RGB").resize((W, H), Image.LANCZOS), np.float32)/255.0
              if os.path.exists(jpg) else np.zeros_like(rnd))
        mp = os.path.join(PDIR, name + "_mask.png")
        if os.path.exists(mp):
            mk = np.asarray(Image.open(mp).convert("L").resize((W, H), Image.LANCZOS), np.float32)[:, :, None]/255.0
            comp = ph*(1.0-mk) + rnd*mk
        else:
            comp = rnd
        rows.append((name, np.concatenate([ph, comp], 1)))
    grid = np.concatenate([r[1] for r in rows], 0)
    im = Image.fromarray((np.clip(grid, 0, 1)*255).astype(np.uint8)); dr = ImageDraw.Draw(im)
    for i, (name, _) in enumerate(rows):
        dr.rectangle([0, i*H, 240, i*H+13], fill=(0, 0, 0))
        dr.text((4, i*H+2), "%d %s" % (i, name), fill=(255, 255, 0))
    path = os.path.join(OUT, "preview_best.png"); im.save(path)
    print("saved", os.path.relpath(path, HERE), " (%d photos, params %s)" % (len(rows), os.path.basename(BEST)))
    glfw.terminate()

if __name__ == "__main__":
    main()
