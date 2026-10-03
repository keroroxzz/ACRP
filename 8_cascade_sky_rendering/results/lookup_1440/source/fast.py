"""Single-image shared-work experiments. All prepasses run inside render/timing."""
from cascade import *


def replace_function(src, signature, body):
    start = src.index(signature)
    left = src.index('{', start)
    depth = 1
    right = left + 1
    while depth:
        depth += (src[right] == '{') - (src[right] == '}')
        right += 1
    return src[:left] + '{\n' + body + '\n}' + src[right:]


def lookup_sources(src):
    # Spherical atmosphere: diffuse source depends only on radius and solar zenith.
    header = src[:src.index('void mainImage')]
    lut = header + '''
void main() {
    float h = (gl_FragCoord.x - 0.5) / 511.0 * 0.1;
    float mu = (gl_FragCoord.y - 0.5) / 255.0 * 2.0 - 1.0;
    float r = planetRadius + h;
    vec3 L = vec3(sqrt(max(0.0,1.0-mu*mu)),0.0,mu);
    outColor = vec4(PSMSGather(vec3(0.0,0.0,r),vec3(0.0,0.0,1.0),L),1.0);
}
'''
    # A 32x32 cell table rebuilt for this camera/time; no per-ray register array.
    event = header + '''
void main() {
    ivec2 pix = ivec2(gl_FragCoord.xy);
    vec2 cell = floor(uCamera.xy / HSE_CELL_SIZE) - 16.0 + vec2(pix.x/2,pix.y);
    float cloudTime = iTime*100.0+5000.0;
    vec2 r0 = HSEHash22(cell+7.13), r1 = HSEHash22(cell+23.91), r2 = HSEHash22(cell+51.37);
    float u = fract(cloudTime*HSE_TIME_SCALE+HSEHash12(cell+37.11));
    float life = HSELife(u);
    vec2 center = (cell+mix(vec2(0.25),vec2(0.75),r0))*HSE_CELL_SIZE;
    center += (r1*2.0-1.0)*(650.0*(u*2.0-1.0));
    float radius = mix(HSE_MIN_RADIUS,HSE_MAX_RADIUS,r1.y)*HSERadiusLife(u);
    float aspect = mix(0.68,1.0,r2.y);
    vec2 radii = r0.x < 0.5 ? vec2(radius,radius*aspect) : vec2(radius*aspect,radius);
    float amp = (HSEHash12(cell+19.71) < 0.38 || life <= 0.0001) ? 0.0 : mix(700.0,HSE_MAX_RISE,r2.x)*life;
    outColor = pix.x % 2 == 0 ? vec4(center,max(radii,vec2(1.0))) : vec4(amp,0.0,0.0,1.0);
}
'''
    # Keep exact evaluation for table-domain excursions.
    src = src.replace('float HSEEventRise(', 'float HSEEventRiseExact(', 1)
    at = src.index('//================ NEXT: per-cell event cache')
    src = src[:at] + '''
float HSEEventRise(vec2 p, float cloudTime, vec2 cell) {
    ivec2 q = ivec2(cell - (floor(uCamera.xy/HSE_CELL_SIZE)-16.0));
    if (any(lessThan(q,ivec2(0))) || any(greaterThanEqual(q,ivec2(32))))
        return HSEEventRiseExact(p,cloudTime,cell);
    float amp = texelFetch(uEvents,ivec2(q.x*2+1,q.y),0).x;
    if (amp == 0.0) return 0.0;
    vec4 e = texelFetch(uEvents,ivec2(q.x*2,q.y),0);
    vec2 v = (p-e.xy)/e.zw;
    float rSq = dot(v,v);
    if (rSq >= 1.0) return 0.0;
    float s = 1.0-rSq;
    return amp*(s*s*s);
}
''' + src[at:]
    src = replace_function(src, 'float3 PSMSGather(', '''
    vec2 f = vec2(clamp((length(P)-planetRadius)/0.1,0.0,1.0),clamp(dot(up,L)*0.5+0.5,0.0,1.0));
    return texture(uGather,(f*vec2(511.0,255.0)+0.5)/vec2(512.0,256.0)).rgb;
''')
    src = src.replace('uniform sampler2D uProbes;', 'uniform sampler2D uProbes, uGather, uEvents;')
    return src, lut, event


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
        lookups, gather, events = lookup_sources(fast)
        self.lookup_sky = compile_program(lookups)
        self.gather_program = compile_program(gather)
        self.event_program = compile_program(events)
        self.gather = Surface(512, 256)
        self.events = Surface(64, 32)

    def scene(self, **kwargs):
        for p in (self.original_sky, self.fast_sky, self.lookup_sky, self.probe_program,
                  self.gather_program, self.event_program):
            self.sky = p
            super().scene(**kwargs)
        self.sky = self.original_sky

    def render(self, shared=False, lookup=False, **kwargs):
        shared = shared or lookup
        if shared:
            self.probes.bind()
            glDisable(GL_STENCIL_TEST)
            glDrawBuffers(1, [GL_COLOR_ATTACHMENT0])
            glUseProgram(self.probe_program)
            self.stage = 'shared-probes'
            self.triangle()
            glActiveTexture(GL_TEXTURE3)
            glBindTexture(GL_TEXTURE_2D, self.probes.textures[0])
            self.sky = self.lookup_sky if lookup else self.fast_sky
            glUseProgram(self.sky)
            glUniform1i(self.loc(self.sky, 'uProbes'), 3)
            if lookup:
                for surface, p, unit, name in ((self.gather,self.gather_program,4,'uGather'),
                                              (self.events,self.event_program,5,'uEvents')):
                    surface.bind()
                    glUseProgram(p)
                    glDrawBuffers(1,[GL_COLOR_ATTACHMENT0])
                    self.stage = name
                    self.triangle()
                    glActiveTexture(GL_TEXTURE0+unit)
                    glBindTexture(GL_TEXTURE_2D,surface.textures[0])
                    glUseProgram(self.sky)
                    glUniform1i(self.loc(self.sky,name),unit)
        else:
            self.sky = self.original_sky
        return super().render(**kwargs)

    def close(self):
        self.sky = self.original_sky
        self.probes.close()
        self.gather.close()
        self.events.close()
        for p in (self.lookup_sky,self.gather_program,self.event_program):
            glDeleteProgram(p)
        glDeleteProgram(self.fast_sky)
        glDeleteProgram(self.probe_program)
        super().close()
