"""Shared offscreen-GL harness for the CloudWorks shaders.

The shaders are Shadertoy-style (they define mainImage() and expect
iResolution / iTime / iMouse), so this wraps them in a GLSL 330 core program
that draws one fullscreen triangle into an RGBA32F FBO.

Run with the py3.7 GL interpreter (glfw + PyOpenGL + PIL) - see HANDOVER.md.
"""
from __future__ import annotations
import math
import os
import re
import time

import glfw
import numpy as np
from OpenGL.GL import *

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)                       # articles/0_cloudworks
OUT = os.path.join(ART, "results")
CACHE = os.path.join(OUT, "ref_cache")            # gitignored: ~6 MB per reference

W, H = 800, 450
ITIME = 10.0                     # -> cloudTime 6000; fix this and the cloud seed is fixed

_VS = ("#version 330 core\n"
       "void main(){ vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));"
       " gl_Position=vec4(p*2.0-1.0,0.0,1.0); }\n")
_HEAD = ("#version 330 core\n"
         "uniform vec3 iResolution;\nuniform float iTime;\nuniform vec4 iMouse;\n"
         "out vec4 _fragOut;\n")
_TAIL = "\nvoid main(){ vec4 c; mainImage(c, gl_FragCoord.xy); _fragOut = c; }\n"


# --------------------------------------------------------------------------- GL

def _shader(src, stage):
    s = glCreateShader(stage)
    glShaderSource(s, src)
    glCompileShader(s)
    if glGetShaderiv(s, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(s).decode())
    return s


def program(body):
    """Compile a Shadertoy-style shader body into a drawable program."""
    p = glCreateProgram()
    glAttachShader(p, _shader(_VS, GL_VERTEX_SHADER))
    glAttachShader(p, _shader(_HEAD + body + _TAIL, GL_FRAGMENT_SHADER))
    glLinkProgram(p)
    if glGetProgramiv(p, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(p).decode())
    return p


class Target(object):
    """Hidden GL context plus an RGBA32F render target."""

    def __init__(self, width=W, height=H):
        self.w, self.h = width, height
        glfw.init()
        for k, v in [(glfw.VISIBLE, glfw.FALSE),
                     (glfw.CONTEXT_VERSION_MAJOR, 3), (glfw.CONTEXT_VERSION_MINOR, 3),
                     (glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)]:
            glfw.window_hint(k, v)
        self.win = glfw.create_window(self.w, self.h, "cloudworks", None, None)
        glfw.make_context_current(self.win)
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self.w, self.h, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
        glBindVertexArray(glGenVertexArrays(1))
        self.renderer = glGetString(GL_RENDERER).decode()

    def draw(self, p, frames=15, warm=0.8, tiles=1):
        """Render once and return (median ms, HDR image).

        GL_TIME_ELAPSED off a cold GPU samples the boost-clock ramp, not steady
        state, so spin for `warm` seconds of continuous draws first and report
        the median (HANDOVER pitfall).

        `tiles > 1` splits the frame with glScissor.  A converged reference can
        take seconds per frame, which would trip the driver watchdog; timing is
        skipped in that mode.
        """
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.w, self.h)
        glUseProgram(p)
        glUniform3f(glGetUniformLocation(p, "iResolution"), float(self.w), float(self.h), 1.0)
        glUniform1f(glGetUniformLocation(p, "iTime"), ITIME)
        glUniform4f(glGetUniformLocation(p, "iMouse"), 0.0, 0.0, 0.0, 0.0)
        ms = 0.0
        if tiles > 1:
            glEnable(GL_SCISSOR_TEST)
            for i in range(tiles):
                y0, y1 = self.h * i // tiles, self.h * (i + 1) // tiles
                glScissor(0, y0, self.w, y1 - y0)
                glDrawArrays(GL_TRIANGLES, 0, 3)
                glFinish()
            glDisable(GL_SCISSOR_TEST)
        else:
            t0 = time.time()
            while time.time() - t0 < warm:
                for _ in range(4):
                    glDrawArrays(GL_TRIANGLES, 0, 3)
                glFinish()
            q = glGenQueries(1)
            ts = []
            for _ in range(frames):
                glBeginQuery(GL_TIME_ELAPSED, int(q[0]))
                glDrawArrays(GL_TRIANGLES, 0, 3)
                glEndQuery(GL_TIME_ELAPSED)
                glFinish()
                ts.append(float(glGetQueryObjectuiv(int(q[0]), GL_QUERY_RESULT)) / 1e6)
            ms = float(np.median(ts))
        buf = glReadPixels(0, 0, self.w, self.h, GL_RGBA, GL_FLOAT)
        img = np.frombuffer(buf, np.float32).reshape(self.h, self.w, 4)
        return ms, img[::-1].copy()               # glReadPixels is bottom-origin


# ------------------------------------------------------------------ shader text

def load(name):
    """Read one of the CloudWorks shaders. utf-8 is mandatory (cp950 default)."""
    return open(os.path.join(ART, name), encoding="utf-8").read()


def setdef(src, name, value):
    """Override a #define in the shader source."""
    return re.sub(r"(#define %s\s+)\S+" % name,
                  lambda m: m.group(1) + str(value), src, count=1)


def place(src, cam, ray_bias):
    """Move the test camera and tilt the view.

    ray_bias shifts the z of the eye ray: +0.40 looks up, 0.0 is level,
    -0.35 looks down.

    Each substitution is checked, because a silently unmatched pattern means
    the shader renders a DIFFERENT camera and the comparison is meaningless.
    shader.glsl (the pre-HSE snapshot) declares its camera at the origin and
    then passes a literal to RenderClouds instead of using it, so it needs an
    extra rewrite - hence the alternatives below.  Note it is rewritten onto the
    +Y-forward convention of the HSE shaders, but its gameTime offset differs
    (-2000 vs +5000), so it renders a DIFFERENT cloud time: treat it as an
    archive snapshot, not as a comparable baseline.
    """
    def sub(text, alternatives, new, what):
        for old in alternatives:
            if old in text:
                return text.replace(old, new)
        raise ValueError("place(): no %s pattern matched; the shader's mainImage "
                         "has changed and the camera would be wrong" % what)

    src = sub(src, ["vec3 CameraPosition = vec3(25000.0,0.0,0.0);",
                    "vec3 CameraPosition = vec3(0.0,0.0,0.0);"],
              "vec3 CameraPosition = vec3(%.1f,%.1f,%.1f);" % tuple(cam), "camera")
    # the pre-HSE shader ignores CameraPosition when it calls RenderClouds
    src = src.replace("RenderClouds(direction, float3(0.0),",
                      "RenderClouds(direction, CameraPosition,")
    src = sub(src, ["normalize(vec3(-uv.x, 1.0/fov, uv.y+0.4))",
                    # pre-HSE snapshot: forward is +X, not +Y
                    "normalize(vec3(1.0/fov, -uv.x, uv.y))"],
              "normalize(vec3(-uv.x, 1.0/fov, uv.y+(%.3f)))" % ray_bias, "eye ray")
    # Follow the camera height, otherwise the high views get a ground-level sky.
    # shader_HSE_PSMS_Next already derives its sky origin from CameraPosition
    # (it needs the same point for the ambient probes), so there it is a no-op -
    # but it still has to be *checked*, not assumed.
    if "Sky(OutterSpace(direction, vSunLightDir), probeCam" in src:
        return src
    return sub(src, ["Sky(c, Game2Atm_Alt(float3(0.0))"],
               "Sky(c, Game2Atm_Alt(CameraPosition)", "sky origin")


def sun(src, elev_deg, azim_deg):
    """Pin the sun to an explicit elevation/azimuth instead of the iMouse hook.

    The test cameras look along +Y and screen-right is -X (getViewRay returns
    vec3(-uv.x, 1/fov, uv.y+bias)), so azimuth is measured from the view
    direction:

        azim   0 deg -> sun straight ahead, camera faces it   (back-lit)
        azim 180 deg -> sun behind the camera                 (front-lit)
        azim  90 deg -> sun to screen-LEFT                    (side-lit)

    Elevation is above the horizon; negative is below it, which is what puts
    Paper 3 on its twilight branch (the switch is at a solar zenith of 88 deg,
    i.e. +2 deg elevation).
    """
    e = math.radians(elev_deg)
    a = math.radians(azim_deg)
    v = (math.cos(e) * math.sin(a),      # +X is screen-left
         math.cos(e) * math.cos(a),      # +Y is forward
         math.sin(e))
    old = "    vSunLightDir = vec3(cos(mouse.x)*cos(mouse.y), sin(mouse.x)*cos(mouse.y), sin(mouse.y));"
    if old not in src:
        raise ValueError("sun(): the vSunLightDir hook is gone; the sun would be wrong")
    return src.replace(old, "    vSunLightDir = normalize(vec3(%.6f,%.6f,%.6f));" % v)


def reference(src):
    """Turn shader_HSE_Next.glsl into a converged ground truth.

    Uniform 2.5 m steps with no distance coarsening, every adaptive mechanism
    disabled, and a step budget large enough to reach a.fade (40 km) - if the
    reference stops short of the shader it is comparing different clouds, which
    is worth several percent RMSE on the level and top-down views.

    ShadowMarching()'s low-density early-out is scaled by march.x, so pin it to
    the shipping value or the reference changes the LIGHTING model too.

    The light-march step counts are pinned as well.  They are shipping defaults
    that get tuned for speed, and if the reference inherits them it moves every
    time they change - the gate would then be measuring against a target that
    drifts with the thing under test.
    """
    src = src.replace("float4(5.0, 60.0, 8.0, 350.0)", "float4(2.5, 2.5, 1.0, 16100.0)")
    src = src.replace("ShadowMarching(dens*a.march.x", "ShadowMarching(dens*5.0")
    for name, val in (("SHADOW_FINE_MARCH_STEP", 6), ("SHADOW_COARSE_MARCH_STEP", 25)):
        src = re.sub(r"(#define %s\s+)\S+" % name, lambda m, v=val: m.group(1) + str(v),
                     src, count=1)

    # Same argument for the Paper 3 atmosphere: its step counts, gather size and
    # any LOD switch are all things that get tuned for speed, so the ground truth
    # pins them at (or above) the baseline instead of inheriting them.  Only
    # applied if the shader actually has them, so the Next-only path is unaffected.
    if "PSMS_SKY_STEPS" in src:
        for name, val in (("PSMS_SKY_STEPS", 64), ("PSMS_AERIAL_STEPS", 48),
                          ("PSMS_PROBE_STEPS", 32), ("PSMS_GATHER_ELEV", 3),
                          ("PSMS_GATHER_AZIM", 2), ("PSMS_GATHER_LERP", 0),
                          ("PSMS_SUNTRANS_LERP", 0), ("PSMS_AERIAL_LOD", 0)):
            src = re.sub(r"(#define %s\s+)\S+" % name, lambda m, v=val: m.group(1) + str(v),
                         src, count=1)
    for k, v in dict(NEXT_JITTER=0.0, NEXT_REACH_SAFETY=0.0, NEXT_TAU_BUDGET=0.001,
                     NEXT_ROOF_SPAN_XY=25.0, NEXT_ROOF_SPAN_XY_SHADOW=100.0,
                     NEXT_SHADOW_ROOF=3,          # densest roof model available
                     NEXT_LIGHT_LOD=0,            # reference never uses light LOD
                     NEXT_SURFACE_BUDGET=0, NEXT_ROOF_SKIP=0).items():
        src = setdef(src, k, v)
    return src


# ---------------------------------------------------------------------- metrics

def tonemap(hdr):
    """Exponential tonemap + sRGB. All metrics live in this space, because a
    linear-HDR RMSE is dominated by a handful of bright pixels."""
    c = np.clip(hdr[:, :, :3], 0.0, None)
    return np.clip(1.0 - np.exp(-c * 1.2), 0, 1) ** (1 / 2.2)


def _grad(x):
    return np.stack([np.diff(x, axis=0)[:, :-1], np.diff(x, axis=1)[:-1, :]], -1)


def _lap(x):
    l = x.mean(2)
    return l[1:-1, 1:-1] * 4 - l[:-2, 1:-1] - l[2:, 1:-1] - l[1:-1, :-2] - l[1:-1, 2:]


def compare(a, ref):
    """(rmse, edge, speckle) of tonemapped image `a` against tonemapped `ref`.

    edge    - RMSE of the first difference; penalises smeared silhouettes.
    speckle - RMSE of the Laplacian; penalises per-pixel sampling noise.
    """
    return (float(np.sqrt(((a - ref) ** 2).mean())),
            float(np.sqrt(((_grad(a.mean(2)) - _grad(ref.mean(2))) ** 2).mean())),
            float(np.sqrt(((_lap(a) - _lap(ref)) ** 2).mean())))
