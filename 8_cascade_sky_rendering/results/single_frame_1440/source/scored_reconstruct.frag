#version 430 core
uniform sampler2D uPrevious, uScores;
uniform vec2 uSize;
uniform float uThreshold;
uniform int uForce;
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
    outMask=texelFetch(uScores,p+1,0).x>uThreshold?1.0:0.0;
    if(uForce!=0) outMask=uForce>0?1.0:0.0;
}
