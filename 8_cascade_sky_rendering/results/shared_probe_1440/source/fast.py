"""Single-image shared-work experiments. All prepasses run inside render/timing."""
from cascade import *


def shared_sources():
    src = sky_source()
    start = src.index('    float3 probeCam = Game2Atm_Alt(CameraPosition);', src.index('void mainImage'))
    end = src.index('\n    vec3 c = Sky(', start)
    probes = src[start:end]
    header = src[:src.index('void mainImage')]
    probe = header + '''
void main() {
    vec3 CameraPosition = uCamera;
    vSunLightDir = uSun;
''' + probes + '''
    outColor = vec4(gl_FragCoord.x < 1.0 ? gAmbZenith : gAmbHorizon, 1.0);
}
'''
    src = src[:start] + '''
    float3 probeCam = Game2Atm_Alt(CameraPosition);
    gAmbZenith = texelFetch(uProbes, ivec2(0,0), 0).rgb;
    gAmbHorizon = texelFetch(uProbes, ivec2(1,0), 0).rgb;
''' + src[end:]
    src = src.replace('uniform vec2 uTargetSize;', 'uniform vec2 uTargetSize;\nuniform sampler2D uProbes;')
    return src, probe


class FastRenderer(Renderer):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.original_sky = self.sky
        fast, probe = shared_sources()
        self.fast_sky = compile_program(fast)
        self.probe_program = compile_program(probe)
        self.probes = Surface(2, 1)

    def scene(self, **kwargs):
        for p in (self.original_sky, self.fast_sky, self.probe_program):
            self.sky = p
            super().scene(**kwargs)
        self.sky = self.original_sky

    def render(self, shared=False, **kwargs):
        if shared:
            self.probes.bind()
            glDisable(GL_STENCIL_TEST)
            glDrawBuffers(1, [GL_COLOR_ATTACHMENT0])
            glUseProgram(self.probe_program)
            self.stage = 'shared-probes'
            self.triangle()
            glActiveTexture(GL_TEXTURE3)
            glBindTexture(GL_TEXTURE_2D, self.probes.textures[0])
            self.sky = self.fast_sky
            glUseProgram(self.sky)
            glUniform1i(self.loc(self.sky, 'uProbes'), 3)
        else:
            self.sky = self.original_sky
        return super().render(**kwargs)

    def close(self):
        self.sky = self.original_sky
        self.probes.close()
        glDeleteProgram(self.fast_sky)
        glDeleteProgram(self.probe_program)
        super().close()
