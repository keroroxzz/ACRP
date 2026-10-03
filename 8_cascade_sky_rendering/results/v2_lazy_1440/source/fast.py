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


def bounded_source(src):
    # Cloud profile 0 fades at 40 km. View anchors extend < 240 m in XY;
    # shadow anchors < 4 km. This is inside the table's > 120 km margin.
    src=replace_once(src,'if (d.x >= 0.0) {','if (d.x >= 0.0 && d.x < a.fade) {')
    src=replace_once(src,'''    if (any(lessThan(q,ivec2(0))) || any(greaterThanEqual(q,ivec2(32))))
        return HSEEventRiseExact(p,cloudTime,cell);''','')
    src=replace_once(src,'''    if(any(lessThan(q,ivec2(0)))||any(greaterThanEqual(q,ivec2(512))))
        return HSELowCloudTopExact(p,cloudTime,baseTop);''','')
    src=replace_once(src,'''    if(mask<0) return HSELowCloudTopExact(p,cloudTime,baseTop);''','')
    return src


def lighting_source(src, mode):
    if mode in (3,4):
        src=re.sub(r'(#define NEXT_SHADOW_STRIDE\s+)2\b',r'\g<1>'+str(mode),src)
        src=re.sub(r'(#define NEXT_SURF_SHADOW_STRIDE\s+)2\b',r'\g<1>'+str(mode),src)
    elif mode==10:
        src=replace_once(src,'int coarseSteps = SHADOW_COARSE_MARCH_STEP;','int coarseSteps = 10;')
        src=replace_once(src,'float kCoarse = 1.0;','float kCoarse = 2.0;')
    return src


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
        self.bounded_sky=compile_program(bounded_source(candidate))
        self.lighting_skies={mode:compile_program(lighting_source(bounded_source(candidate),mode))
                             for mode in (3,4,10)}
        self.parent_score=compile_program((HERE/'parent_score.frag').read_text())
        self.scored_reconstruct=compile_program((HERE/'scored_reconstruct.frag').read_text())
        self.scores={}
        self.use_scores=False
        self.axis_reconstruct=compile_program((HERE/'axis_reconstruct.frag').read_text())
        from transport import sources as transport_sources
        transport, reuse=transport_sources(bounded_source(candidate))
        self.transport_sky=compile_program(transport)
        self.reuse_sky=compile_program(reuse)
        reuse=re.sub(r'gReuseShadow=mn.y>0.05[^;]+;', 'gReuseShadow=true;',reuse)
        self.pure_reuse_sky=compile_program(reuse)
        self.transport_reconstruct=compile_program((HERE/'transport_reconstruct.frag').read_text())
        self.transport_stencil=compile_program((HERE/'transport_stencil.frag').read_text())
        self.transport_textures={}
        self.reuse_light=0.0

    def scene(self, **kwargs):
        for p in (self.original_sky, self.fast_sky, self.lookup_sky, self.probe_program,
                  self.gather_program, self.event_program, self.hash_sky,
                  self.roof_sky, self.roof_program, self.candidate_sky,self.candidate_program,self.chunk_sky,
                  self.bounded_sky,*self.lighting_skies.values(),self.transport_sky,self.reuse_sky,self.pure_reuse_sky):
            self.sky = p
            super().scene(**kwargs)
        self.sky = self.original_sky
        self.chunk_cache.scene(kwargs['camera'],kwargs.get('itime',10.0))

    def before_reconstruct(self, previous, current, threshold, radius, force):
        if not self.use_scores:
            return
        size=(previous.w+1,previous.h+1)
        glActiveTexture(GL_TEXTURE10)
        if size not in self.scores:
            self.scores[size]=HashSurface(*size)
        surface=self.scores[size]
        surface.bind()
        glDisable(GL_STENCIL_TEST)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,previous.textures[0])
        glUseProgram(self.parent_score)
        glUniform1i(self.loc(self.parent_score,'uPrevious'),0)
        glUniform1i(self.loc(self.parent_score,'uRadius'),radius)
        self.stage='parent-score-%dx%d'%size
        self.triangle()
        glActiveTexture(GL_TEXTURE10)
        glBindTexture(GL_TEXTURE_2D,surface.textures[0])
        glUseProgram(self.scored_reconstruct)
        glUniform1i(self.loc(self.scored_reconstruct,'uScores'),10)

    def render(self, shared=False, lookup=False, curvature=False, hashed=False, roof=0, candidates=False, chunks=False, bounded=False, scores=False, lighting=0, separable=0, reuse_light=0.0, split_light=0.0, **kwargs):
        reuse_light = max(reuse_light,split_light)
        self.reuse_light=reuse_light
        self.previous_transport=None
        bounded = bounded or separable>0 or reuse_light>0
        bounded = bounded or lighting>0
        candidates = candidates or chunks or bounded
        lookup = lookup or hashed or roof>0 or candidates
        self.use_scores=scores
        self.reconstruct = self.scored_reconstruct if scores else self.curvature if curvature else self.original_reconstruct
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
            self.sky = self.bounded_sky if bounded else self.chunk_sky if chunks else self.candidate_sky if candidates else self.roof_sky if roof else self.hash_sky if hashed else self.lookup_sky if lookup else self.fast_sky
            if lighting:
                self.sky=self.lighting_skies[lighting]
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
        if split_light:
            return self.render_split(kwargs.get('factor',4),kwargs.get('threshold',.035),split_light)
        if separable:
            return self.render_separable(separable,kwargs.get('threshold',.04))
        return super().render(**kwargs)

    def render_split(self,factor,threshold,shadow_limit):
        sizes=[(math.ceil(self.w/factor),math.ceil(self.h/factor)),(self.w,self.h)]
        glDisable(GL_STENCIL_TEST)
        glColorMask(True,True,True,True)
        first=self.surface(sizes[0])
        first.bind()
        glDrawBuffers(1,[GL_COLOR_ATTACHMENT0])
        self.stage='transport-base'
        self.shade(first)
        current=self.surface(sizes[1])
        self.use_scores=True
        self.before_reconstruct(first,current,threshold,1,0)
        current.bind()
        glDrawBuffers(2,[GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1])
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,first.textures[0])
        glActiveTexture(GL_TEXTURE11)
        glBindTexture(GL_TEXTURE_2D,self.previous_transport)
        p=self.transport_reconstruct
        glUseProgram(p)
        for name,unit in (('uPrevious',0),('uScores',10),('uTransport',11)):
            glUniform1i(self.loc(p,name),unit)
        glUniform2f(self.loc(p,'uSize'),self.w,self.h)
        glUniform1f(self.loc(p,'uThreshold'),threshold)
        glUniform1f(self.loc(p,'uShadowLimit'),shadow_limit)
        self.stage='transport-classify'
        self.triangle()
        for kind,p in ((1,self.bounded_sky),(2,self.pure_reuse_sky)):
            glBindFramebuffer(GL_FRAMEBUFFER,current.stencil_fbo)
            glEnable(GL_STENCIL_TEST)
            glStencilMask(0xff)
            if kind==1:
                glClearStencil(0)
                glClear(GL_STENCIL_BUFFER_BIT)
            glStencilFunc(GL_ALWAYS,kind,0xff)
            glStencilOp(GL_KEEP,GL_KEEP,GL_REPLACE)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D,current.textures[1])
            glUseProgram(self.transport_stencil)
            glUniform1i(self.loc(self.transport_stencil,'uMask'),0)
            glUniform1i(self.loc(self.transport_stencil,'uKind'),kind)
            self.stage='transport%d-stencil'%kind
            self.triangle()
            glBindTexture(GL_TEXTURE_2D,0)
            current.bind()
            glDrawBuffers(1,[GL_COLOR_ATTACHMENT0])
            glStencilMask(0)
            glStencilFunc(GL_EQUAL,kind,0xff)
            glStencilOp(GL_KEEP,GL_KEEP,GL_KEEP)
            self.sky=p
            glUseProgram(p)
            for name,unit in (('uProbes',3),('uGather',4),('uEvents',5),('uCandidates',7),('uTransport',11)):
                glUniform1i(self.loc(p,name),unit)
            self.stage='transport%d-shade'%kind
            Renderer.shade(self,current)
        glDisable(GL_STENCIL_TEST)
        return current

    def shade(self,surface):
        if self.reuse_light:
            first=self.previous_transport is None
            self.sky=self.transport_sky if first else self.reuse_sky
            glUseProgram(self.sky)
            for name,unit in (('uProbes',3),('uGather',4),('uEvents',5),('uCandidates',7)):
                glUniform1i(self.loc(self.sky,name),unit)
            glActiveTexture(GL_TEXTURE11)
            if first:
                size=(surface.w,surface.h)
                if size not in self.transport_textures:
                    t=glGenTextures(1)
                    self.transport_textures[size]=t
                    glBindTexture(GL_TEXTURE_2D,t)
                    glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA32F,*size)
                    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST)
                    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST)
                self.previous_transport=self.transport_textures[size]
                glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT2,GL_TEXTURE_2D,self.previous_transport,0)
                glDrawBuffers(3,[GL_COLOR_ATTACHMENT0,GL_NONE,GL_COLOR_ATTACHMENT2])
            else:
                glBindTexture(GL_TEXTURE_2D,self.previous_transport)
                glUniform1i(self.loc(self.sky,'uTransport'),11)
                glUniform1f(self.loc(self.sky,'uShadowLimit'),self.reuse_light)
        super().shade(surface)

    def render_separable(self,factor,threshold):
        sizes=[(math.ceil(self.w/factor),math.ceil(self.h/factor)),
               (math.ceil(self.w/factor),self.h),(self.w,self.h)]
        self.last_sizes=sizes
        glColorMask(True,True,True,True)
        glDisable(GL_STENCIL_TEST)
        first=self.surface(sizes[0])
        first.bind()
        glDrawBuffers(1,[GL_COLOR_ATTACHMENT0])
        self.stage='base'
        self.shade(first)
        previous=first
        for size,axis in zip(sizes[1:],(1,0)):
            current=self.surface(size)
            current.bind()
            glDisable(GL_STENCIL_TEST)
            glDrawBuffers(2,[GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1])
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D,previous.textures[0])
            p=self.axis_reconstruct
            glUseProgram(p)
            glUniform1i(self.loc(p,'uPrevious'),0)
            glUniform1i(self.loc(p,'uAxis'),axis)
            glUniform1f(self.loc(p,'uThreshold'),threshold)
            glUniform2f(self.loc(p,'uSize'),*size)
            self.stage='axis%d-reconstruct'%axis
            self.triangle()
            glBindFramebuffer(GL_FRAMEBUFFER,current.stencil_fbo)
            glEnable(GL_STENCIL_TEST)
            glStencilMask(0xff)
            glClearStencil(0)
            glClear(GL_STENCIL_BUFFER_BIT)
            glStencilFunc(GL_ALWAYS,1,0xff)
            glStencilOp(GL_KEEP,GL_KEEP,GL_REPLACE)
            glUseProgram(self.stencil)
            glBindTexture(GL_TEXTURE_2D,current.textures[1])
            glUniform1i(self.loc(self.stencil,'uMask'),0)
            self.stage='axis%d-stencil'%axis
            self.triangle()
            glBindTexture(GL_TEXTURE_2D,0)
            current.bind()
            glDrawBuffers(1,[GL_COLOR_ATTACHMENT0])
            glStencilMask(0)
            glStencilFunc(GL_EQUAL,1,0xff)
            glStencilOp(GL_KEEP,GL_KEEP,GL_KEEP)
            self.stage='axis%d-shade'%axis
            self.shade(current)
            glDisable(GL_STENCIL_TEST)
            previous=current
        return previous

    def capture(self,**variant):
        if not variant.get('separable'):
            return super().capture(**variant)
        image=self.render(**variant).read()
        if not np.isfinite(image).all():
            raise RuntimeError('Non-finite separable output')
        work=0
        stats=[]
        for i,size in enumerate(self.last_sizes):
            fraction=1.0 if i==0 else float((self.surface(size).read(mask=True)>.5).mean())
            work+=fraction*size[0]*size[1]
            stats.append(dict(size=list(size),shaded_fraction=fraction))
        return image,dict(levels=stats,full_resolution_equivalent_shading=work/(self.w*self.h))

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
        for p in (self.bounded_sky,self.parent_score,self.scored_reconstruct):
            glDeleteProgram(p)
        glDeleteProgram(self.axis_reconstruct)
        glDeleteProgram(self.transport_sky)
        glDeleteProgram(self.reuse_sky)
        for p in (self.pure_reuse_sky,self.transport_reconstruct,self.transport_stencil):
            glDeleteProgram(p)
        if self.transport_textures:
            glDeleteTextures(list(self.transport_textures.values()))
        for p in self.lighting_skies.values():
            glDeleteProgram(p)
        for surface in self.scores.values():
            surface.close()
        for surface in self.roof_surfaces.values():
            surface.close()
        for p in (self.lookup_sky,self.gather_program,self.event_program,self.hash_sky,self.hash_program,
                  self.roof_sky,self.roof_program,self.candidate_sky,self.candidate_program):
            glDeleteProgram(p)
        glDeleteProgram(self.fast_sky)
        glDeleteProgram(self.probe_program)
        super().close()

    def profile(self,**variant):
        count=128
        self.profile_pool=[int(q) for q in np.asarray(glGenQueries(count)).reshape(-1)]
        self.profile_pending=[]
        self.render(**variant)
        glFinish()
        samples={name:float(glGetQueryObjectuiv(q,GL_QUERY_RESULT))/1e6
                 for name,q in self.profile_pending}
        self.profile_pending=None
        glDeleteQueries(count,self.profile_pool)
        return samples
