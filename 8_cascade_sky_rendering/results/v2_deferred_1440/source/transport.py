"""Share only smooth incident cloud lighting; keep fine-ray density integration."""
from cascade import replace_once


def sources(src):
    src=src.replace('layout(location=0) out vec4 outColor;', '''layout(location=0) out vec4 outColor;
layout(location=2) out vec4 outTransport;
vec4 gTransport=vec4(0.0,0.0,0.0,1.0);
bool gReuseShadow=false;
float gShadowMean=0.0;
uniform sampler2D uTransport;
uniform float uShadowLimit;''')
    src=replace_once(src,'if (fx.y < 2.3 && dens > 0.0)',
                     'if (fx.y < 2.3 && dens > 0.0 && !gReuseShadow)')
    src=replace_once(src,'            float suncross = dot(lightDirection.xyz, cam_dir);', '''
            if (gReuseShadow) fx.x=gShadowMean*wacc.y;
            gTransport=vec4(fx.x/max(wacc.y,1e-8),wacc.y,d.y,fx.z);
            float suncross = dot(lightDirection.xyz, cam_dir);''')
    src=replace_once(src,'mainImage(outColor, gl_FragCoord.xy*iResolution.xy/uTargetSize);', '''
    mainImage(outColor, gl_FragCoord.xy*iResolution.xy/uTargetSize);
    outTransport=gTransport;''')
    coarse=src
    marker='    // Ambient probes for the clouds:'
    at=src.index(marker)
    src=src[:at]+'''
    vec2 tc=fragCoord/iResolution.xy*vec2(textureSize(uTransport,0))-0.5;
    ivec2 tp=ivec2(floor(tc));
    vec2 tf=fract(tc);
    vec4 v[4];
    vec4 mn=vec4(1e30),mx=vec4(-1e30);
    for(int j=0;j<4;j++) {
        v[j]=texelFetch(uTransport,clamp(tp+ivec2(j%2,j/2),ivec2(0),textureSize(uTransport,0)-1),0);
        mn=min(mn,v[j]); mx=max(mx,v[j]);
    }
    gReuseShadow=mn.y>0.05 && mx.x-mn.x<uShadowLimit && mx.z-mn.z<max(100.0,mn.z*0.2);
    gShadowMean=mix(mix(v[0].x,v[1].x,tf.x),mix(v[2].x,v[3].x,tf.x),tf.y);
''' +src[at:]
    return coarse,src
