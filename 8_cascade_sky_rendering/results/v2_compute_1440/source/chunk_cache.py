"""Per-image exact-lattice cache for the two low-frequency cloud octaves."""
from cascade import *

COMPUTE = '''#version 430 core
layout(local_size_x=8,local_size_y=8,local_size_z=4) in;
layout(r32f,binding=0) writeonly uniform image2DArray uOutput;
uniform ivec3 uOrigin;
void main() {
    ivec3 p=ivec3(gl_GlobalInvocationID);
    if(any(greaterThanEqual(p,imageSize(uOutput)))) return;
    vec3 fr=vec3(p+uOrigin);
    float n=1153.0*fr.x+2381.0*fr.y+fr.z;
    imageStore(uOutput,p,vec4(fract(sin(n/1873.1873)*1618.03398875)));
}
'''

SAMPLE = '''
float chunkNoise(vec3 p, bool fine) {
    vec3 fr=floor(p), ft=fract(p);
    ivec3 q=ivec3(fr)-(fine?uChunkOriginB:uChunkOriginA);
    ivec3 size=fine?textureSize(uChunkB,0):textureSize(uChunkA,0);
    if(any(lessThan(q,ivec3(0)))||any(greaterThanEqual(q+1,size))) return noise3d(p);
    vec2 uv=(vec2(q.xy)+1.0)/vec2(size.xy);
    vec4 lo,hi;
    if(fine) {
        lo=textureGather(uChunkB,vec3(uv,q.z),0);
        hi=textureGather(uChunkB,vec3(uv,q.z+1),0);
    } else {
        lo=textureGather(uChunkA,vec3(uv,q.z),0);
        hi=textureGather(uChunkA,vec3(uv,q.z+1),0);
    }
    // Gather ordering: (x,y+1), (x+1,y+1), (x+1,y), (x,y).
    vec4 v=mix(lo,hi,ft.z);
    return mix(mix(v.w,v.z,ft.x),mix(v.x,v.y,ft.x),ft.y);
}
'''


def sources(src):
    at=src.index('//large chunk for the distribution')
    src=src[:at]+SAMPLE+src[at:]
    src=src.replace('noise3d(pA) * (noise3d(pB)*density.y+density.x)*cs',
                    'chunkNoise(pA,false) * (chunkNoise(pB,true)*density.y+density.x)*cs')
    src=src.replace('noise3d((pos + offsetB)*scale.y)','chunkNoise((pos + offsetB)*scale.y,true)')
    src=src.replace('noise3d((pos + offsetA)*scale.x)','chunkNoise((pos + offsetA)*scale.x,false)')
    src=src.replace('uniform vec2 uTargetSize;', '''uniform vec2 uTargetSize;
uniform sampler2DArray uChunkA, uChunkB;
uniform ivec3 uChunkOriginA,uChunkOriginB;''')
    return src


class ChunkCache:
    def __init__(self):
        s=glCreateShader(GL_COMPUTE_SHADER)
        glShaderSource(s,COMPUTE)
        glCompileShader(s)
        if not glGetShaderiv(s,GL_COMPILE_STATUS):
            raise RuntimeError(glGetShaderInfoLog(s).decode())
        self.program=glCreateProgram()
        glAttachShader(self.program,s)
        glLinkProgram(self.program)
        glDeleteShader(s)
        if not glGetProgramiv(self.program,GL_LINK_STATUS):
            raise RuntimeError(glGetProgramInfoLog(self.program).decode())
        self.textures=[]
        self.sizes=[(128,128,8),(512,512,32)]
        for size in self.sizes:
            t=glGenTextures(1)
            self.textures.append(t)
            glBindTexture(GL_TEXTURE_2D_ARRAY,t)
            glTexStorage3D(GL_TEXTURE_2D_ARRAY,1,GL_R32F,*size)
            glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MIN_FILTER,GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D_ARRAY,GL_TEXTURE_MAG_FILTER,GL_NEAREST)
        self.origins=[]

    def scene(self,camera,itime):
        t=itime*100.0+5000.0
        self.origins=[]
        for scale,velocity in ((.0008,(-1.8,1.0,0)),(.005,(-2.0,-.2,0))):
            low=np.array(camera)*[1,1,0]+[-48000,-46000,0]+np.array(velocity)*t
            self.origins.append(np.floor(low*scale).astype(np.int32))

    def render(self,renderer):
        for i,(size,t,origin) in enumerate(zip(self.sizes,self.textures,self.origins)):
            glUseProgram(self.program)
            glUniform3i(renderer.loc(self.program,'uOrigin'),*origin)
            glBindImageTexture(0,t,0,True,0,GL_WRITE_ONLY,GL_R32F)
            glDispatchCompute((size[0]+7)//8,(size[1]+7)//8,(size[2]+3)//4)
        glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT)
        glUseProgram(renderer.sky)
        for i,(t,origin) in enumerate(zip(self.textures,self.origins)):
            glActiveTexture(GL_TEXTURE8+i)
            glBindTexture(GL_TEXTURE_2D_ARRAY,t)
            glUniform1i(renderer.loc(renderer.sky,'uChunk'+('A','B')[i]),8+i)
            glUniform3i(renderer.loc(renderer.sky,'uChunkOrigin'+('A','B')[i]),*origin)

    def close(self):
        glDeleteTextures(self.textures)
        glDeleteProgram(self.program)
