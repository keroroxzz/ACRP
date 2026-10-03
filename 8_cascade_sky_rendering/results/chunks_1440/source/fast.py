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


def hash_sources(src):
    # Integer lattice hashes are immutable shader data. Regenerate inside every
    # frame here so even a cold, isolated image pays the complete construction cost.
    prepass = '''#version 430 core
layout(location=0) out vec4 outColor;
void main() {
    int idx = int(gl_FragCoord.y)*8192 + int(gl_FragCoord.x);
    float n = float(idx-16777216);
    outColor=vec4(fract(sin(n/1873.1873)*1618.03398875),0.0,0.0,1.0);
}
'''
    at = src.index('float noise3d(')
    src = src[:at] + '''
float latticeHash(float n) {
    int idx=int(n)+16777216;
    if(idx<0 || idx>=33554432) return hash(n);
    return texelFetch(uHash,ivec2(idx%8192,idx/8192),0).x;
}
''' + src[at:]
    start = src.index('float noise3d(')
    end = src.index('float noise3d_timedep',start)
    src = src[:start] + src[start:end].replace('hash(', 'latticeHash(') + src[end:]
    src = src.replace('uniform sampler2D uProbes, uGather, uEvents;',
                      'uniform sampler2D uProbes, uGather, uEvents, uHash;')
    return src, prepass


class HashSurface:
    def __init__(self, width=8192, height=4096):
        self.w, self.h = width, height
        self.fbo=glGenFramebuffers(1)
        self.textures=[glGenTextures(1)]
        glBindTexture(GL_TEXTURE_2D,self.textures[0])
        glTexStorage2D(GL_TEXTURE_2D,1,GL_R32F,self.w,self.h)
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST)
        glBindFramebuffer(GL_FRAMEBUFFER,self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,self.textures[0],0)
        if glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Hash framebuffer incomplete')

    bind=Surface.bind

    def close(self):
        glDeleteFramebuffers(1,[self.fbo])
        glDeleteTextures(self.textures)


def roof_sources(src):
    header = src[:src.index('void mainImage')]
    prepass = header + '''
void main() {
    vec2 p = uCamera.xy + ((gl_FragCoord.xy-0.5)/(uTargetSize-1.0)*2.0-1.0)*50000.0;
    outColor = vec4(HSELowCloudTop(p,iTime*100.0+5000.0,0.0),0.0,0.0,1.0);
}
'''
    src = src.replace('float HSELowCloudTop(', 'float HSELowCloudTopExact(',1)
    at = src.index('// Apply an already-evaluated local roof')
    src = src[:at] + '''
float HSELowCloudTop(vec2 p, float cloudTime, float baseTop) {
    vec2 f = (p-uCamera.xy)/100000.0+0.5;
    if(any(lessThan(f,vec2(0.0))) || any(greaterThan(f,vec2(1.0))))
        return HSELowCloudTopExact(p,cloudTime,baseTop);
    vec2 size = vec2(textureSize(uRoof,0));
    return baseTop+texture(uRoof,(f*(size-1.0)+0.5)/size).x;
}
''' + src[at:]
    src = src.replace('uniform sampler2D uProbes, uGather, uEvents;',
                      'uniform sampler2D uProbes, uGather, uEvents, uRoof;')
    return src, prepass


def candidate_sources(src):
    header = src[:src.index('void mainImage')]
    prepass = header + '''
void main() {
    vec2 origin = (floor(uCamera.xy/HSE_CELL_SIZE)-16.0)*HSE_CELL_SIZE;
    vec2 lo = origin + floor(gl_FragCoord.xy)*500.0;
    vec2 hi = lo+500.0;
    vec2 cell = floor((lo+250.0)/HSE_CELL_SIZE);
    float upper[9];
    float bestLower=0.0;
    for(int k=0;k<9;k++) {
        ivec2 q=ivec2(cell+vec2(k%3-1,k/3-1)-(floor(uCamera.xy/HSE_CELL_SIZE)-16.0));
        upper[k]=0.0;
        if(any(lessThan(q,ivec2(0)))||any(greaterThanEqual(q,ivec2(32)))) {
            // Outside the event table: mark this tile for exact fallback.
            outColor=vec4(-1.0); return;
        }
        float amp=texelFetch(uEvents,ivec2(q.x*2+1,q.y),0).x;
        vec4 e=texelFetch(uEvents,ivec2(q.x*2,q.y),0);
        vec2 nearD=(clamp(e.xy,lo,hi)-e.xy)/e.zw;
        vec2 farD=max(abs(lo-e.xy),abs(hi-e.xy))/e.zw;
        float u=max(1.0-dot(nearD,nearD),0.0);
        float l=max(1.0-dot(farD,farD),0.0);
        upper[k]=amp*u*u*u;
        bestLower=max(bestLower,amp*l*l*l);
    }
    int mask=0;
    for(int k=0;k<9;k++)
        if(upper[k]>0.0 && upper[k]+0.01>=bestLower) mask|=1<<k;
    outColor=vec4(float(mask),0.0,0.0,1.0);
}
'''
    src = src.replace('float HSELowCloudTop(', 'float HSELowCloudTopExact(',1)
    at = src.index('// Apply an already-evaluated local roof')
    src = src[:at] + '''
float HSELowCloudTop(vec2 p, float cloudTime, float baseTop) {
    vec2 origin=(floor(uCamera.xy/HSE_CELL_SIZE)-16.0)*HSE_CELL_SIZE;
    ivec2 q=ivec2(floor((p-origin)/500.0));
    if(any(lessThan(q,ivec2(0)))||any(greaterThanEqual(q,ivec2(512))))
        return HSELowCloudTopExact(p,cloudTime,baseTop);
    int mask=int(texelFetch(uCandidates,q,0).x);
    if(mask<0) return HSELowCloudTopExact(p,cloudTime,baseTop);
    float rise=0.0;
    vec2 baseCell=floor(p/HSE_CELL_SIZE);
    while(mask!=0) {
        int k=findLSB(mask);
        mask &= mask-1;
        rise=max(rise,HSEEventRise(p,cloudTime,baseCell+vec2(k%3-1,k/3-1)));
    }
    float broad=(noise2d(vec3(p*0.000055,cloudTime*0.00002))-0.5)*90.0;
    return baseTop+max(0.0,rise+broad);
}
''' + src[at:]
    src = src.replace('uniform sampler2D uProbes, uGather, uEvents;',
                      'uniform sampler2D uProbes, uGather, uEvents, uCandidates;')
    return src, prepass


class FastRenderer(Renderer):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.original_sky = self.sky
        self.original_reconstruct = self.reconstruct
        self.curvature = compile_program((HERE/'curvature.frag').read_text())
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
        hashed, hash_prepass = hash_sources(lookups)
        self.hash_sky = compile_program(hashed)
        self.hash_program = compile_program(hash_prepass)
        self.hash_surface = HashSurface()
        roof, roof_prepass = roof_sources(lookups)
        self.roof_sky = compile_program(roof)
        self.roof_program = compile_program(roof_prepass)
        self.roof_surfaces = {}
        candidate, candidate_prepass = candidate_sources(lookups)
        self.candidate_sky=compile_program(candidate)
        self.candidate_program=compile_program(candidate_prepass)
        self.candidates=HashSurface(512,512)
        from chunk_cache import ChunkCache, sources
        self.chunk_cache=ChunkCache()
        self.chunk_sky=compile_program(sources(candidate))

    def scene(self, **kwargs):
        for p in (self.original_sky, self.fast_sky, self.lookup_sky, self.probe_program,
                  self.gather_program, self.event_program, self.hash_sky,
                  self.roof_sky, self.roof_program, self.candidate_sky,self.candidate_program,self.chunk_sky):
            self.sky = p
            super().scene(**kwargs)
        self.sky = self.original_sky
        self.chunk_cache.scene(kwargs['camera'],kwargs.get('itime',10.0))

    def render(self, shared=False, lookup=False, curvature=False, hashed=False, roof=0, candidates=False, chunks=False, **kwargs):
        candidates = candidates or chunks
        lookup = lookup or hashed or roof>0 or candidates
        self.reconstruct = self.curvature if curvature else self.original_reconstruct
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
            self.sky = self.chunk_sky if chunks else self.candidate_sky if candidates else self.roof_sky if roof else self.hash_sky if hashed else self.lookup_sky if lookup else self.fast_sky
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
            if hashed:
                self.hash_surface.bind()
                glUseProgram(self.hash_program)
                self.stage='integer-hashes'
                self.triangle()
                glActiveTexture(GL_TEXTURE6)
                glBindTexture(GL_TEXTURE_2D,self.hash_surface.textures[0])
                glUseProgram(self.sky)
                glUniform1i(self.loc(self.sky,'uHash'),6)
            if roof:
                glActiveTexture(GL_TEXTURE7)
                if roof not in self.roof_surfaces:
                    self.roof_surfaces[roof] = HashSurface(roof,roof)
                    glBindTexture(GL_TEXTURE_2D,self.roof_surfaces[roof].textures[0])
                    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR)
                surface = self.roof_surfaces[roof]
                surface.bind()
                glUseProgram(self.roof_program)
                glUniform2f(self.loc(self.roof_program,'uTargetSize'),roof,roof)
                glUniform1i(self.loc(self.roof_program,'uEvents'),5)
                self.stage='roof-field'
                self.triangle()
                glActiveTexture(GL_TEXTURE7)
                glBindTexture(GL_TEXTURE_2D,surface.textures[0])
                glUseProgram(self.sky)
                glUniform1i(self.loc(self.sky,'uRoof'),7)
            if candidates:
                self.candidates.bind()
                glUseProgram(self.candidate_program)
                glUniform1i(self.loc(self.candidate_program,'uEvents'),5)
                self.stage='roof-candidates'
                self.triangle()
                glActiveTexture(GL_TEXTURE7)
                glBindTexture(GL_TEXTURE_2D,self.candidates.textures[0])
                glUseProgram(self.sky)
                glUniform1i(self.loc(self.sky,'uCandidates'),7)
            if chunks:
                self.chunk_cache.render(self)
        else:
            self.sky = self.original_sky
        return super().render(**kwargs)

    def close(self):
        self.sky = self.original_sky
        self.reconstruct = self.original_reconstruct
        glDeleteProgram(self.curvature)
        self.probes.close()
        self.gather.close()
        self.events.close()
        self.hash_surface.close()
        self.candidates.close()
        self.chunk_cache.close()
        glDeleteProgram(self.chunk_sky)
        for surface in self.roof_surfaces.values():
            surface.close()
        for p in (self.lookup_sky,self.gather_program,self.event_program,self.hash_sky,self.hash_program,
                  self.roof_sky,self.roof_program,self.candidate_sky,self.candidate_program):
            glDeleteProgram(p)
        glDeleteProgram(self.fast_sky)
        glDeleteProgram(self.probe_program)
        super().close()
