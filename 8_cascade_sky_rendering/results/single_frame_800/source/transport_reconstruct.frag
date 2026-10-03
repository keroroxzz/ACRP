#version 430 core
uniform sampler2D uPrevious,uScores,uTransport;
uniform vec2 uSize;
uniform float uThreshold,uShadowLimit;
layout(location=0) out vec4 outColor;
layout(location=1) out float outMask;
vec4 hdr(ivec2 p) {
    return texelFetch(uPrevious,clamp(p,ivec2(0),textureSize(uPrevious,0)-1),0);
}
void main() {
    vec2 coord=gl_FragCoord.xy/uSize*vec2(textureSize(uPrevious,0))-0.5;
    ivec2 p=ivec2(floor(coord));
    vec2 f=fract(coord);
    outColor=mix(mix(hdr(p),hdr(p+ivec2(1,0)),f.x),mix(hdr(p+ivec2(0,1)),hdr(p+1),f.x),f.y);
    outMask=0.0;
    if(texelFetch(uScores,p+1,0).x<=uThreshold) return;
    vec4 mn=vec4(1e30),mx=vec4(-1e30);
    for(int j=0;j<4;j++) {
        vec4 v=texelFetch(uTransport,clamp(p+ivec2(j%2,j/2),ivec2(0),textureSize(uTransport,0)-1),0);
        mn=min(mn,v); mx=max(mx,v);
    }
    bool reuse=mn.y>0.05 && mx.x-mn.x<uShadowLimit && mx.z-mn.z<max(100.0,mn.z*0.2);
    outMask=reuse?1.0:0.6;
}
