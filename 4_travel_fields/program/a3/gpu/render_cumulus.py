"""Render the fair-weather cumulus-field scene (a3/gpu/cumulus_frag.glsl)."""
from __future__ import annotations
import numpy as np, glfw, os
from OpenGL.GL import *
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(RESEARCH, "results", "cumulus")
SRC = open(os.path.join(HERE, "cumulus_frag.glsl")).read()
VS = ("#version 330 core\nvoid main(){ vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));"
      " gl_Position=vec4(p*2.0-1.0,0.0,1.0); }\n")
W, H = 1280, 720


def inject(src, d): L = src.split("\n"); return L[0] + "\n" + "\n".join("#define %s %s" % kv for kv in d.items()) + "\n" + "\n".join(L[1:])
def sh(src, st):
    s = glCreateShader(st); glShaderSource(s, src); glCompileShader(s)
    if glGetShaderiv(s, GL_COMPILE_STATUS) != GL_TRUE: raise RuntimeError(glGetShaderInfoLog(s).decode())
    return s
def prog(d):
    p = glCreateProgram(); glAttachShader(p, sh(VS, GL_VERTEX_SHADER)); glAttachShader(p, sh(inject(SRC, d), GL_FRAGMENT_SHADER)); glLinkProgram(p)
    if glGetProgramiv(p, GL_LINK_STATUS) != GL_TRUE: raise RuntimeError(glGetProgramInfoLog(p).decode())
    return p
def cam_basis(eye, ctr):
    f = ctr - eye; f /= np.linalg.norm(f); r = np.cross([0, 1, 0], f); r /= np.linalg.norm(r); u = np.cross(f, r)
    return np.stack([r, u, f], 1).astype(np.float32)
def draw(p, fbo, vao, eye, cam, frames):
    glBindFramebuffer(GL_FRAMEBUFFER, fbo); glViewport(0, 0, W, H); glUseProgram(p)
    glUniform2f(glGetUniformLocation(p, "uRes"), float(W), float(H)); glUniform3f(glGetUniformLocation(p, "uEye"), *[float(x) for x in eye])
    glUniformMatrix3fv(glGetUniformLocation(p, "uCam"), 1, GL_FALSE, cam.T.copy()); glBindVertexArray(vao)
    for _ in range(4): glDrawArrays(GL_TRIANGLES, 0, 3)
    glFinish(); q = glGenQueries(1); ts = []
    for _ in range(frames):
        glBeginQuery(GL_TIME_ELAPSED, int(q[0])); glDrawArrays(GL_TRIANGLES, 0, 3); glEndQuery(GL_TIME_ELAPSED)
        glFinish(); ts.append(float(glGetQueryObjectuiv(int(q[0]), GL_QUERY_RESULT)) / 1e6)
    img = np.frombuffer(glReadPixels(0, 0, W, H, GL_RGBA, GL_FLOAT), np.float32).reshape(H, W, 4)
    return float(np.median(ts)), img[::-1].copy()


def main():
    os.makedirs(OUT, exist_ok=True); glfw.init()
    for k, v in [(glfw.VISIBLE, glfw.FALSE), (glfw.CONTEXT_VERSION_MAJOR, 3), (glfw.CONTEXT_VERSION_MINOR, 3),
                 (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)]:
        glfw.window_hint(k, v)
    win = glfw.create_window(W, H, "cu", None, None); glfw.make_context_current(win)
    tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, tex); glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0, GL_RGBA, GL_FLOAT, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    fbo = glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER, fbo); glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0); vao = glGenVertexArrays(1)
    print("GPU:", glGetString(GL_RENDERER).decode())

    eye = np.array([0.0, 0.5, 0.0], np.float32); ctr = np.array([0.0, 2.6, 6.0], np.float32); cam = cam_basis(eye, ctr)
    A3 = dict(A3_SUNF=4, A3_SUNGL=4)
    jobs = [
        ("ref_HQ", dict(METHOD=1, SUN_STEPS=12, SMALL_STEP=0.03, BIG_STEP=0.22, MAXI=3000, JITTER_ON=0, SUN_JITTER=0.0, A3_CAMF=6, A3_CAMGL=5, **A3)),
        ("march_rt", dict(METHOD=1, SUN_STEPS=6, SMALL_STEP=0.06, BIG_STEP=0.28, MAXI=1600, JITTER_ON=1, SUN_JITTER=0.0, A3_CAMF=6, A3_CAMGL=5, **A3)),
        ("a3_hier", dict(METHOD=2, SUN_STEPS=6, SMALL_STEP=0.06, BIG_STEP=0.28, MAXI=1600, JITTER_ON=1, SUN_JITTER=1.0, A3_CAMF=6, A3_CAMGL=2, **A3)),
    ]
    for name, d in jobs:
        ms, img = draw(prog(d), fbo, vao, eye, cam, 16)
        Image.fromarray((np.clip(img[:, :, :3], 0, 1) * 255).astype(np.uint8)).save(os.path.join(OUT, name + ".png"))
        print("%-10s %.2f ms" % (name, ms))
    glfw.terminate(); print("-> %s" % OUT)


if __name__ == "__main__":
    main()
