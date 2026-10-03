"""Render the aesthetic cumulus sky (cumulus_beauty.glsl). Standalone.
Usage:  python render_beauty.py [pitch] [yaw] [height] [sunAz] [sunEl] [expo] [name]
"""
from __future__ import annotations
import numpy as np, glfw, os, sys
from OpenGL.GL import *
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "results"); os.makedirs(OUT, exist_ok=True)
SHADER = os.path.join(HERE, "cumulus_beauty.glsl")
VS = ("#version 330 core\nvoid main(){ vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));"
      " gl_Position=vec4(p*2.0-1.0,0.0,1.0); }\n")
W, H = 1280, 720


def sh(src, st):
    s = glCreateShader(st); glShaderSource(s, src); glCompileShader(s)
    if glGetShaderiv(s, GL_COMPILE_STATUS) != GL_TRUE: raise RuntimeError(glGetShaderInfoLog(s).decode())
    return s
def prog():
    p = glCreateProgram(); glAttachShader(p, sh(VS, GL_VERTEX_SHADER)); glAttachShader(p, sh(open(SHADER).read(), GL_FRAGMENT_SHADER)); glLinkProgram(p)
    if glGetProgramiv(p, GL_LINK_STATUS) != GL_TRUE: raise RuntimeError(glGetProgramInfoLog(p).decode())
    return p
def cam_basis(eye, ctr):
    f = ctr - eye; f /= np.linalg.norm(f); r = np.cross([0, 1, 0], f); r /= np.linalg.norm(r); u = np.cross(f, r)
    return np.stack([r, u, f], 1).astype(np.float32)


def main():
    a = sys.argv
    pitch = float(a[1]) if len(a) > 1 else 0.22
    yaw = float(a[2]) if len(a) > 2 else 0.30
    height = float(a[3]) if len(a) > 3 else 0.5
    sunAz = float(a[4]) if len(a) > 4 else 0.55
    sunEl = float(a[5]) if len(a) > 5 else 0.80
    expo = float(a[6]) if len(a) > 6 else 0.55
    name = a[7] if len(a) > 7 else "beauty"
    glfw.init()
    for k, v in [(glfw.VISIBLE, glfw.FALSE), (glfw.CONTEXT_VERSION_MAJOR, 3), (glfw.CONTEXT_VERSION_MINOR, 3),
                 (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)]:
        glfw.window_hint(k, v)
    win = glfw.create_window(W, H, "b", None, None); glfw.make_context_current(win)
    tex = glGenTextures(1); glBindTexture(GL_TEXTURE_2D, tex); glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, W, H, 0, GL_RGBA, GL_FLOAT, None)
    fbo = glGenFramebuffers(1); glBindFramebuffer(GL_FRAMEBUFFER, fbo); glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
    glViewport(0, 0, W, H); pr = prog(); vao = glGenVertexArrays(1); glUseProgram(pr)
    eye = np.array([0, height, 0], np.float32)
    d = np.array([np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)], np.float32)
    cam = cam_basis(eye, eye + d)
    sd = np.array([np.cos(sunEl)*np.sin(sunAz), np.sin(sunEl), np.cos(sunEl)*np.cos(sunAz)], np.float32)
    U = lambda n: glGetUniformLocation(pr, n)
    glUniform2f(U("uRes"), float(W), float(H)); glUniform3f(U("uEye"), *[float(z) for z in eye])
    glUniformMatrix3fv(U("uCam"), 1, GL_FALSE, cam.T.copy()); glUniform1f(U("uFocal"), 1.0)
    glUniform3f(U("uSunDir"), *[float(z) for z in sd]); glUniform1f(U("uExposure"), expo)
    # cloud/sky params (defaults = current tuned look)
    for n, val in [("uCbase", 0.85), ("uCtop", 3.35), ("uCovLo", 0.485), ("uCovHi", 0.71),
                   ("uCovFreq", 0.15), ("uDetAmp", 0.50), ("uDens", 5.4), ("uSigma", 4.2)]:
        glUniform1f(U(n), val)
    glUniform3f(U("uZen"), 0.045, 0.18, 0.60); glUniform3f(U("uHor"), 0.50, 0.65, 0.83)
    glBindVertexArray(vao)
    for _ in range(2): glDrawArrays(GL_TRIANGLES, 0, 3)
    glFinish()
    img = np.frombuffer(glReadPixels(0, 0, W, H, GL_RGB, GL_FLOAT), np.float32).reshape(H, W, 3)[::-1]
    Image.fromarray((np.clip(img, 0, 1)*255).astype(np.uint8)).save(os.path.join(OUT, name + ".png"))
    print("wrote out/%s.png" % name)
    glfw.terminate()


if __name__ == "__main__":
    main()
