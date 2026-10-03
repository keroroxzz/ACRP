"""V2 shadow bounds: omit noise octaves only when density is clamped."""
from cascade import replace_once


def shadow_bound_source(src):
    helper='''
float ShadowChunkBound(vec3 pos, vec2 density, vec3 scale, vec3 shift,
                      vec3 offsetA, vec3 offsetB, float cs,
                      float gain, float bandLo, float bandHi) {
    pos.z/=scale.z;
    pos+=shift*pos.z;
    float nA=noise3d((pos+offsetA)*scale.x);
    // Noise values are in [0,1]. Keep a roundoff margin on both bounds.
    float upper=min(1.0,nA*(density.y+density.x)*cs);
    if(upper*gain+0.00001<bandLo) return 0.0;
    float lower=min(1.0,nA*density.x*cs);
    if(lower>bandHi+0.00001) return bandHi+0.00001;
    float nB=noise3d((pos+offsetB)*scale.y);
    return saturate(nA*(nB*density.y+density.x)*cs);
}
'''
    idx=src.index('inline float ShadowMarching(')
    src=src[:idx]+helper+src[idx:]
    old='float d1 = ChunkLod(p, densityChunk, scaleChunk, shift, offsetA, offsetB, cs.x, octChunk);'
    assert src.count(old)==2
    src=src.replace(old,'float d1 = ShadowChunkBound(p,densityChunk,scaleChunk,shift,offsetA,offsetB,cs.x,1.0+fineGain,bandLo,cs.y);',1)
    src=src.replace(old,'float d1 = ShadowChunkBound(p,densityChunk,scaleChunk,shift,offsetA,offsetB,cs.x,1.5,bandLo,cs.y);',1)
    # This specialization is valid for the fixed reference NEXT_LIGHT_LOD=0.
    assert '#define NEXT_LIGHT_LOD          0' in src or '#define NEXT_LIGHT_LOD 0' in src or __import__('re').search(r'#define\s+NEXT_LIGHT_LOD\s+0\b',src)
    return src


def opaque_source(src):
    line='    vec3 c = Sky(OutterSpace(direction, vSunLightDir), probeCam, direction, vSunLightDir);'
    src=replace_once(src,line,'')
    return replace_once(src,'    c = clouds.xyz+c*clouds.w;', '''    vec3 c=clouds.xyz;
    if(clouds.w>0.0) c+=Sky(OutterSpace(direction,vSunLightDir),probeCam,direction,vSunLightDir)*clouds.w;''')
