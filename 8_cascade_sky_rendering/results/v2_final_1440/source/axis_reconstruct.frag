#version 430 core
uniform sampler2D uPrevious;
uniform vec2 uSize;
uniform float uThreshold;
uniform int uAxis;
layout(location=0) out vec4 outColor;
layout(location=1) out float outMask;
vec4 hdr(ivec2 p) {
    return texelFetch(uPrevious,clamp(p,ivec2(0),textureSize(uPrevious,0)-1),0);
}
vec3 color(ivec2 p) {
    return pow(clamp(1.0-exp(-max(hdr(p).rgb,vec3(0.0))*1.2),0.0,1.0),vec3(1.0/2.2));
}
void main() {
    vec2 coord=gl_FragCoord.xy/uSize*vec2(textureSize(uPrevious,0))-0.5;
    if(uAxis==0) coord.y=gl_FragCoord.y-0.5;
    else coord.x=gl_FragCoord.x-0.5;
    ivec2 p=ivec2(floor(coord));
    ivec2 axis=uAxis==0?ivec2(1,0):ivec2(0,1);
    outColor=mix(hdr(p),hdr(p+axis),fract(coord)[uAxis]);
    vec3 g=vec3(0.0);
    for(int i=-1;i<=1;i++) g=max(g,abs(color(p+axis*i)-color(p+axis*(i+1))));
    outMask=max(g.x,max(g.y,g.z))>uThreshold?1.0:0.0;
}
