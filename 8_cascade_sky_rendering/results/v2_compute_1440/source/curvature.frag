#version 430 core
uniform sampler2D uPrevious;
uniform vec2 uSize;
uniform float uThreshold;
uniform int uRadius, uForce;
layout(location=0) out vec4 outColor;
layout(location=1) out float outMask;
vec4 hdr(ivec2 p) {
    return texelFetch(uPrevious,clamp(p,ivec2(0),textureSize(uPrevious,0)-1),0);
}
vec3 disp(ivec2 p) {
    return pow(clamp(1.0-exp(-max(hdr(p).rgb,vec3(0.0))*1.2),0.0,1.0),vec3(1.0/2.2));
}
void main() {
    vec2 coord=gl_FragCoord.xy/uSize*vec2(textureSize(uPrevious,0))-0.5;
    ivec2 p=ivec2(floor(coord));
    vec2 f=fract(coord);
    outColor=mix(mix(hdr(p),hdr(p+ivec2(1,0)),f.x),mix(hdr(p+ivec2(0,1)),hdr(p+1),f.x),f.y);
    vec3 g=vec3(0.0);
    for(int y=0;y<2;y++) for(int x=0;x<2;x++) {
        ivec2 q=p+ivec2(x,y);
        vec3 c=disp(q);
        g=max(g,abs(disp(q+ivec2(1,0))+disp(q-ivec2(1,0))-2.0*c));
        g=max(g,abs(disp(q+ivec2(0,1))+disp(q-ivec2(0,1))-2.0*c));
    }
    outMask=max(g.x,max(g.y,g.z))*0.375>uThreshold?1.0:0.0;
    if(uForce!=0) outMask=uForce>0?1.0:0.0;
}
