"""V2 experiments: GPU-only tiled ray compaction with unchanged V1 decisions."""
from single_frame import *
from v2_sources import shadow_bound_source, opaque_source


def compile_compute(source):
    shader=glCreateShader(GL_COMPUTE_SHADER)
    glShaderSource(shader,source)
    glCompileShader(shader)
    if not glGetShaderiv(shader,GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    program=glCreateProgram()
    glAttachShader(program,shader)
    glLinkProgram(program)
    if not glGetProgramiv(program,GL_LINK_STATUS):
        raise RuntimeError(glGetProgramInfoLog(program).decode())
    glDetachShader(program,shader)
    glDeleteShader(shader)
    return program


QUEUE='''#version 430 core
layout(local_size_x=8,local_size_y=8) in;
layout(std430,binding=0) buffer Queue { uint pixels[]; };
layout(std430,binding=1) buffer Count { uint count; };
uniform sampler2D uMask;
shared uint n, start;
shared uint localPixels[64];
void main() {
    uint lane=gl_LocalInvocationIndex;
    if(lane==0u) n=0u;
    barrier();
    ivec2 p=ivec2(gl_GlobalInvocationID.xy), size=textureSize(uMask,0);
    bool activePixel=all(lessThan(p,size)) && texelFetch(uMask,p,0).r>0.5;
    // Stable order within a tile preserves spatial coherence independently of
    // atomic scheduling. Only one global atomic per tile.
    localPixels[lane]=activePixel?1u:0u;
    barrier();
    uint rank=0u;
    for(uint i=0u;i<lane;++i) rank+=localPixels[i];
    if(lane==63u) n=rank+localPixels[lane];
    barrier();
    if(lane==0u) start=atomicAdd(count,n);
    barrier();
    if(activePixel) pixels[start+rank]=uint(p.y*size.x+p.x);
}
'''
DISPATCH='''#version 430 core
layout(local_size_x=1) in;
layout(std430,binding=1) readonly buffer Count { uint count; };
layout(std430,binding=2) buffer Dispatch { uint x,y,z; };
uniform uint uGroup;
void main() { uint groups=(count+uGroup-1u)/uGroup;
    x=min(groups,65535u); y=max(1u,(groups+65534u)/65535u); z=1u;
}
'''


def compute_source(fragment,group,direct=False):
    src=fragment.replace('layout(early_fragment_tests) in;','')
    src=src.replace('layout(location=0) out vec4 outColor;','')
    src=src[:src.rfind('\nvoid main()')]
    # The gather texture has one level; explicitly select it in compute.
    src=src.replace('texture(uGather,(f*vec2(511.0,255.0)+0.5)/vec2(512.0,256.0))',
                    'textureLod(uGather,(f*vec2(511.0,255.0)+0.5)/vec2(512.0,256.0),0.0)')
    if direct:
        header='layout(local_size_x=8,local_size_y=8) in;\nuniform sampler2D uMask;'
        index='ivec2 p=ivec2(gl_GlobalInvocationID.xy); if(any(greaterThanEqual(p,ivec2(iResolution.xy))) || texelFetch(uMask,p,0).r<0.5) return;'
    else:
        header=f'''layout(local_size_x={group}) in;
layout(std430,binding=0) readonly buffer Queue {{ uint pixels[]; }};
layout(std430,binding=1) readonly buffer Count {{ uint count; }};'''
        index=f'''uint i=(gl_WorkGroupID.y*65535u+gl_WorkGroupID.x)*{group}u+gl_LocalInvocationIndex;
    if(i>=count) return;
    uint k=pixels[i]; ivec2 p=ivec2(k%uint(iResolution.x),k/uint(iResolution.x));'''
    return src+'\n'+header+'''
layout(rgba32f,binding=0) writeonly uniform image2D uOutput;
void main() { '''+index+'''
    vec4 color; mainImage(color,vec2(p)+0.5); imageStore(uOutput,p,color);
}
'''


class V2Renderer(SingleFrameRenderer):
    def __init__(self,width,height):
        super().__init__(width,height)
        self.compute={n:compile_compute(compute_source(self.optimized_source,n,n==0)) for n in (0,32,64,128)}
        self.queue_program=compile_compute(QUEUE)
        self.dispatch_program=compile_compute(DISPATCH)
        self.buffers=[int(x) for x in glGenBuffers(3)]
        for buffer,size in zip(self.buffers,(width*height*4,4,12)):
            glBindBuffer(GL_SHADER_STORAGE_BUFFER,buffer)
            glBufferData(GL_SHADER_STORAGE_BUFFER,size,None,GL_DYNAMIC_DRAW)
        self.kernels={'bounds':compile_program(shadow_bound_source(self.optimized_source)),
                      'opaque':compile_program(opaque_source(self.optimized_source)),
                      'combined':compile_program(opaque_source(shadow_bound_source(self.optimized_source)))}
        self.queue_mode=None
        self.cubic_program=compile_program((HERE/'cubic_reconstruct.frag').read_text())

    def scene(self,**kwargs):
        super().scene(**kwargs)
        for p in list(self.compute.values())+list(self.kernels.values()):
            self.sky=p
            Renderer.scene(self,**kwargs)
            for _,_,_,unit,name in self.prepasses:
                glUniform1i(self.loc(p,name),unit)
        self.sky=self.original_sky

    def render(self,queue=None,cubic=False,kernel=None,**variant):
        self.queue_mode=queue
        original_sky=self.optimized_sky
        if kernel:
            self.optimized_sky=self.kernels[kernel]
        original=self.scored_reconstruct
        if cubic:
            self.scored_reconstruct=self.cubic_program
        try:
            return super().render(**variant)
        finally:
            self.scored_reconstruct=original
            self.optimized_sky=original_sky

    def dispatch(self,name,x=None,y=1):
        if self.profile_pending is not None:
            q=self.profile_pool[len(self.profile_pending)]
            glBeginQuery(GL_TIME_ELAPSED,q)
        if x is None:
            glDispatchComputeIndirect(0)
        else:
            glDispatchCompute(x,y,1)
        if self.profile_pending is not None:
            glEndQuery(GL_TIME_ELAPSED)
            self.profile_pending.append((name,q))

    def render_levels(self,levels=1,factor=2,threshold=.035,radius=1,force=0):
        if self.queue_mode is None or not self.optimizing or levels==1:
            return super().render_levels(levels=levels,factor=factor,threshold=threshold,radius=radius,force=force)
        if levels!=2:
            raise ValueError('V2 currently uses exactly two levels')
        sizes=level_sizes(self.w,self.h,levels,factor)
        previous=self.surface(sizes[0]); current=self.surface(sizes[1])
        previous.bind(); glDrawBuffers(1,[GL_COLOR_ATTACHMENT0])
        self.stage='L0-shade'; self.shade(previous)
        self.before_reconstruct(previous,current,threshold,radius,force)
        current.bind(); glDrawBuffers(2,[GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1])
        glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,previous.textures[0])
        p=self.scored_reconstruct; glUseProgram(p)
        glUniform1i(self.loc(p,'uPrevious'),0)
        glUniform2f(self.loc(p,'uSize'),self.w,self.h)
        glUniform1f(self.loc(p,'uThreshold'),threshold)
        glUniform1i(self.loc(p,'uForce'),force)
        self.stage='L1-reconstruct-classify'; self.triangle()
        glBindFramebuffer(GL_FRAMEBUFFER,0)
        glBindTexture(GL_TEXTURE_2D,current.textures[1])
        glBindImageTexture(0,current.textures[0],0,False,0,GL_WRITE_ONLY,GL_RGBA32F)
        mode=self.queue_mode
        if mode:
            for binding,buffer in enumerate(self.buffers):
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER,binding,buffer)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER,self.buffers[1])
            glBufferSubData(GL_SHADER_STORAGE_BUFFER,0,4,np.zeros(1,np.uint32))
            glUseProgram(self.queue_program)
            glUniform1i(self.loc(self.queue_program,'uMask'),0)
            self.dispatch('queue',math.ceil(self.w/8),math.ceil(self.h/8))
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
            glUseProgram(self.dispatch_program)
            glUniform1ui(self.loc(self.dispatch_program,'uGroup'),mode)
            self.dispatch('dispatch-command',1)
            glMemoryBarrier(GL_COMMAND_BARRIER_BIT|GL_SHADER_STORAGE_BARRIER_BIT)
            glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER,self.buffers[2])
        p=self.compute[mode]; glUseProgram(p)
        if mode:
            self.dispatch('L1-compute-shade')
        else:
            glUniform1i(self.loc(p,'uMask'),0)
            self.dispatch('L1-compute-shade',math.ceil(self.w/8),math.ceil(self.h/8))
        glMemoryBarrier(GL_FRAMEBUFFER_BARRIER_BIT|GL_TEXTURE_FETCH_BARRIER_BIT|GL_SHADER_IMAGE_ACCESS_BARRIER_BIT|GL_BUFFER_UPDATE_BARRIER_BIT)
        glBindImageTexture(0,0,0,False,0,GL_WRITE_ONLY,GL_RGBA32F)
        return current

    def close(self):
        glDeleteBuffers(3,self.buffers)
        for p in list(self.compute.values())+list(self.kernels.values())+[self.queue_program,self.dispatch_program,self.cubic_program]:
            glDeleteProgram(p)
        super().close()
