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
vec4 cubic(vec4 a,vec4 b,vec4 c,vec4 d,float t) {
    return b+0.5*t*(c-a+t*(2.0*a-5.0*b+4.0*c-d+t*(3.0*(b-c)+d-a)));
}
void main() {
    vec2 coord=gl_FragCoord.xy/uSize*vec2(textureSize(uPrevious,0))-0.5;
    ivec2 p=ivec2(floor(coord));
    vec2 f=fract(coord);
    vec4 rows[4];
    for(int y=-1;y<=2;y++)
        rows[y+1]=cubic(hdr(p+ivec2(-1,y)),hdr(p+ivec2(0,y)),hdr(p+ivec2(1,y)),hdr(p+ivec2(2,y)),f.x);
    vec4 a=hdr(p),b=hdr(p+ivec2(1,0)),c=hdr(p+ivec2(0,1)),d=hdr(p+1);
    outColor=clamp(cubic(rows[0],rows[1],rows[2],rows[3],f.y),min(min(a,b),min(c,d)),max(max(a,b),max(c,d)));
    outMask=texelFetch(uScores,p+1,0).x>uThreshold?1.0:0.0;
    if(uForce!=0) outMask=uForce>0?1.0:0.0;
}
