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


def deferred_sources(src):
    from optimized_sources import replace_function
    start=src.index('            float suncross = dot(lightDirection.xyz, cam_dir);')
    end=src.index('            return float4(C, fx.z);',start)+len('            return float4(C, fx.z);')
    lighting=src[start:end]
    cloud=src[:start]+'            return vec4(fx.x,fx.z,d.y,1.0);'+src[end:]
    cloud=replace_once(cloud,'    return float4(0.0, 0.0, 0.0, 1.0);\n}\n\n//=========================settings',
                       '    return vec4(0.0,1.0,0.0,0.0);\n}\n\n//=========================settings')
    cloud=replace_function(cloud,'void mainImage(','''
    vSunLightDir=uSun;
    float time=gameTime(iTime*100.0)+5000.0;
    timeEffect(time,cloudProfiles[0]);
    CloudRandomness(time);
    float angle; vec3 direction=getViewRay(fragCoord,fov,angle);
    vec3 lightDir; vec3 light=LightSource(uSun,lightDir);
    float depth=9999999999.0;
    fragColor=CloudAtRay(cloudProfiles[0],baseColor,direction,uCamera,light,lightDir,time,0,depth);
''')
    compose=opaque_source(src).replace('uniform vec2 uTargetSize;',
                                     'uniform vec2 uTargetSize;\nuniform sampler2D uTransport;')
    compose=replace_function(compose,'float4 CloudAtRay(','''
    vec4 saved=texelFetch(uTransport,ivec2(gl_FragCoord.xy),0);
    if(saved.w<0.5) return vec4(0.0,0.0,0.0,1.0);
    vec3 fx=vec3(saved.x,0.0,saved.y);
    vec4 d=vec4(0.0,saved.z,0.0,0.0);
'''+lighting)
    return cloud,compose
