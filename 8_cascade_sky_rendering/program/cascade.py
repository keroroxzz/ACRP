"""Actual multipass GPU cascade. No CPU mask readback in the render path."""
from pathlib import Path
import math
import re
import time

import glfw
import numpy as np
from OpenGL.GL import *

HERE = Path(__file__).resolve().parent
VERTEX = '''#version 430 core
void main() {
    vec2 p=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));
    gl_Position=vec4(p*2.0-1.0,0.0,1.0);
}
'''


def compile_program(fragment):
    shaders = []
    p = glCreateProgram()
    for kind, src in ((GL_VERTEX_SHADER, VERTEX), (GL_FRAGMENT_SHADER, fragment)):
        s = glCreateShader(kind)
        glShaderSource(s, src)
        glCompileShader(s)
        if not glGetShaderiv(s, GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(s).decode())
        glAttachShader(p, s)
        shaders.append(s)
    glLinkProgram(p)
    if not glGetProgramiv(p, GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(p).decode())
    for s in shaders:
        glDetachShader(p, s)
        glDeleteShader(s)
    return p


def replace_once(src, old, new):
    if src.count(old) != 1:
        raise ValueError('Source anchor changed: '+old)
    return src.replace(old, new)


def sky_source():
    src = (HERE/'baseline_psms.glsl').read_text()
    # GLSL 4.x reserves "active"; the GLSL 330 archive used it as a local/member.
    src = re.sub(r'\bactive\b', 'hseActive', src)
    src = replace_once(src, 'vec3 CameraPosition = vec3(25000.0,0.0,0.0);',
                       'vec3 CameraPosition = uCamera;')
    src = replace_once(src, 'normalize(vec3(-uv.x, 1.0/fov, uv.y+0.4))',
                       'normalize(vec3(-uv.x, 1.0/fov, uv.y+uRayBias))')
    src = replace_once(src,
        'vSunLightDir = vec3(cos(mouse.x)*cos(mouse.y), sin(mouse.x)*cos(mouse.y), sin(mouse.y));',
        'vSunLightDir = uSun;')
    src = replace_once(src,
        'vec4 clouds = RenderClouds(direction, CameraPosition, vSunLightDir, 9999999999.0, time);',
        'vec4 clouds = uSkyOnly ? vec4(0.0,0.0,0.0,1.0) : '
        'RenderClouds(direction, CameraPosition, vSunLightDir, 9999999999.0, time);')
    return '''#version 430 core
layout(early_fragment_tests) in;
uniform vec3 iResolution;
uniform float iTime;
uniform vec4 iMouse;
uniform vec3 uCamera, uSun;
uniform float uRayBias;
uniform bool uSkyOnly;
uniform vec2 uTargetSize;
layout(location=0) out vec4 outColor;
''' + src + '''
void main() {
    // Always use FINAL-resolution camera coordinates, including odd dimensions.
    mainImage(outColor, gl_FragCoord.xy*iResolution.xy/uTargetSize);
}
'''


def display_color(image):
    return np.clip(1.0-np.exp(-np.maximum(image[..., :3], 0)*1.2), 0, 1)**(1/2.2)


def level_sizes(width, height, levels, factor):
    if factor < 2 or levels < 1 or min(width, height) < 2:
        raise ValueError('factor >= 2, levels >= 1, and width/height >= 2 required')
    sizes = [(int(math.ceil(width/factor**k)), int(math.ceil(height/factor**k)))
             for k in range(levels-1, -1, -1)]
    if len(set(sizes)) != levels:
        raise ValueError('Too many levels for this resolution')
    return sizes


class Surface:
    def __init__(self, width, height):
        self.w, self.h = width, height
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        self.textures = []
        for index, internal, fmt, filtering in (
                (0, GL_RGBA32F, GL_RGBA, GL_LINEAR), (1, GL_R8, GL_RED, GL_NEAREST)):
            t = glGenTextures(1)
            self.textures.append(t)
            glBindTexture(GL_TEXTURE_2D, t)
            glTexImage2D(GL_TEXTURE_2D, 0, internal, width, height, 0, fmt, GL_FLOAT, None)
            for axis in (GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T):
                glTexParameteri(GL_TEXTURE_2D, axis, GL_CLAMP_TO_EDGE)
            for param in (GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER):
                glTexParameteri(GL_TEXTURE_2D, param, filtering)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+index,
                                   GL_TEXTURE_2D, t, 0)
        self.depth_stencil = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_stencil)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                                  GL_RENDERBUFFER, self.depth_stencil)
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Incomplete framebuffer')
        # The mask texture must not remain attached to the active draw FBO while
        # it is sampled. A second FBO shares only the depth/stencil storage.
        self.stencil_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.stencil_fbo)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                                  GL_RENDERBUFFER, self.depth_stencil)
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Incomplete stencil framebuffer')

    def bind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.w, self.h)

    def read(self, mask=False):
        self.bind()
        glReadBuffer(GL_COLOR_ATTACHMENT1 if mask else GL_COLOR_ATTACHMENT0)
        fmt, channels = (GL_RED, 1) if mask else (GL_RGBA, 4)
        a = np.frombuffer(glReadPixels(0, 0, self.w, self.h, fmt, GL_FLOAT), np.float32)
        return a.reshape(self.h, self.w, channels)[::-1].copy()

    def close(self):
        glDeleteFramebuffers(2, [self.fbo, self.stencil_fbo])
        glDeleteTextures(self.textures)
        glDeleteRenderbuffers(1, [self.depth_stencil])


class Renderer:
    def __init__(self, width, height):
        if not glfw.init():
            raise RuntimeError('GLFW initialization failed; a GPU/display session is required')
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        self.window = glfw.create_window(64, 64, 'cascade-sky-research', None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError('OpenGL 4.3 context creation failed')
        glfw.make_context_current(self.window)
        self.gpu = glGetString(GL_RENDERER).decode()
        self.version = glGetString(GL_VERSION).decode()
        if 'llvmpipe' in self.gpu.lower() or 'softpipe' in self.gpu.lower():
            raise RuntimeError('Refusing to report software rendering as GPU performance')
        self.w, self.h = width, height
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glDisable(GL_DITHER)
        self.sky = compile_program(sky_source())
        self.reconstruct = compile_program((HERE/'reconstruct.frag').read_text())
        self.stencil = compile_program((HERE/'stencil.frag').read_text())
        self.locations = {}
        self.surfaces = {}
        self.query = int(np.asarray(glGenQueries(1)).reshape(-1)[0])
        self.profile_pending = None
        self.stage = ''

    def loc(self, p, name):
        key = (p, name)
        if key not in self.locations:
            self.locations[key] = glGetUniformLocation(p, name)
        return self.locations[key]

    def surface(self, size):
        if size not in self.surfaces:
            # Allocation must not overwrite a bound scene lookup on the active
            # texture unit. This matters on the very first isolated image.
            previous_texture=int(glGetIntegerv(GL_TEXTURE_BINDING_2D))
            self.surfaces[size] = Surface(*size)
            glBindTexture(GL_TEXTURE_2D,previous_texture)
        return self.surfaces[size]

    def scene(self, camera, bias, sun, sky_only=False, itime=10.0):
        p = self.sky
        glUseProgram(p)
        glUniform3f(self.loc(p, 'iResolution'), self.w, self.h, 1)
        glUniform1f(self.loc(p, 'iTime'), itime)
        glUniform4f(self.loc(p, 'iMouse'), 0, 0, 0, 0)
        glUniform3f(self.loc(p, 'uCamera'), *camera)
        glUniform3f(self.loc(p, 'uSun'), *sun)
        glUniform1f(self.loc(p, 'uRayBias'), bias)
        glUniform1i(self.loc(p, 'uSkyOnly'), sky_only)

    def triangle(self):
        if self.profile_pending is not None:
            q = self.profile_pool[len(self.profile_pending)]
            glBeginQuery(GL_TIME_ELAPSED, q)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        if self.profile_pending is not None:
            glEndQuery(GL_TIME_ELAPSED)
            self.profile_pending.append((self.stage,q))

    def shade(self, surface):
        glUseProgram(self.sky)
        glUniform2f(self.loc(self.sky, 'uTargetSize'), surface.w, surface.h)
        self.triangle()

    def before_reconstruct(self, previous, current, threshold, radius, force):
        pass

    def render(self, levels=1, factor=2, threshold=0.01, radius=1, force=0):
        sizes = level_sizes(self.w, self.h, levels, factor)
        glColorMask(True, True, True, True)
        glDisable(GL_STENCIL_TEST)
        first = self.surface(sizes[0])
        first.bind()
        glDrawBuffers(1, [GL_COLOR_ATTACHMENT0])
        self.stage = 'L0-shade'
        self.shade(first)
        previous = first
        for level,size in enumerate(sizes[1:],1):
            current = self.surface(size)
            self.before_reconstruct(previous,current,threshold,radius,force)
            current.bind()
            glDisable(GL_STENCIL_TEST)
            glDrawBuffers(2, [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1])
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, previous.textures[0])
            p = self.reconstruct
            glUseProgram(p)
            glUniform1i(self.loc(p, 'uPrevious'), 0)
            glUniform2f(self.loc(p, 'uSize'), *size)
            glUniform1f(self.loc(p, 'uThreshold'), threshold)
            glUniform1i(self.loc(p, 'uRadius'), radius)
            glUniform1i(self.loc(p, 'uForce'), force)
            self.stage = 'L%d-reconstruct-classify'%level
            self.triangle()
            # Turn the GPU-resident decision texture into a stencil. Disable all
            # color outputs so reading the mask is not a framebuffer feedback loop.
            glBindFramebuffer(GL_FRAMEBUFFER, current.stencil_fbo)
            glEnable(GL_STENCIL_TEST)
            glStencilMask(0xff)
            glClearStencil(0)
            glClear(GL_STENCIL_BUFFER_BIT)
            glStencilFunc(GL_ALWAYS, 1, 0xff)
            glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE)
            glUseProgram(self.stencil)
            glBindTexture(GL_TEXTURE_2D, current.textures[1])
            glUniform1i(self.loc(self.stencil, 'uMask'), 0)
            self.stage = 'L%d-stencil'%level
            self.triangle()
            glBindTexture(GL_TEXTURE_2D, 0)
            current.bind()
            glDrawBuffers(1, [GL_COLOR_ATTACHMENT0])
            glStencilMask(0)
            glStencilFunc(GL_EQUAL, 1, 0xff)
            glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP)
            self.stage = 'L%d-shade'%level
            self.shade(current)  # early_fragment_tests skips expensive cloud/sky code
            glDisable(GL_STENCIL_TEST)
            previous = current
        return previous

    def timed(self, **variant):
        glBeginQuery(GL_TIME_ELAPSED, self.query)
        self.render(**variant)
        glEndQuery(GL_TIME_ELAPSED)
        # Synchronization is OUTSIDE the GPU interval; no image/mask reads inside it.
        glFinish()
        return float(glGetQueryObjectuiv(self.query, GL_QUERY_RESULT))/1e6

    def profile(self, **variant):
        count = 1+3*(variant.get('levels',1)-1)
        self.profile_pool = [int(q) for q in np.asarray(glGenQueries(count)).reshape(-1)]
        self.profile_pending = []
        self.render(**variant)
        glFinish()
        samples = {name:float(glGetQueryObjectuiv(q,GL_QUERY_RESULT))/1e6
                   for name,q in self.profile_pending}
        self.profile_pending = None
        glDeleteQueries(count,self.profile_pool)
        return samples

    def warm(self, seconds=1.0):
        start = time.perf_counter()
        while time.perf_counter()-start < seconds:
            self.render()
            glFinish()

    def capture(self, **variant):
        image = self.render(**variant).read()
        if not np.isfinite(image).all():
            raise RuntimeError('Non-finite output')
        sizes = level_sizes(self.w, self.h, variant.get('levels', 1), variant.get('factor', 2))
        work = sizes[0][0]*sizes[0][1]
        stats = [dict(size=list(sizes[0]), shaded_fraction=1.0)]
        for size in sizes[1:]:
            mask = self.surface(size).read(mask=True)
            fraction = float((mask > 0.5).mean())
            work += fraction*size[0]*size[1]
            stats.append(dict(size=list(size), shaded_fraction=fraction))
        return image, dict(levels=stats, full_resolution_equivalent_shading=work/(self.w*self.h))

    def close(self):
        for surface in self.surfaces.values():
            surface.close()
        for p in (self.sky, self.reconstruct, self.stencil):
            glDeleteProgram(p)
        glDeleteQueries(1, [self.query])
        glDeleteVertexArrays(1, [self.vao])
        glfw.destroy_window(self.window)
        glfw.terminate()
