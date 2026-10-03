#version 430 core
uniform sampler2D uPrevious;
uniform int uRadius;
layout(location=0) out vec4 outColor;
vec3 color(ivec2 p) {
    vec3 c=texelFetch(uPrevious,clamp(p,ivec2(0),textureSize(uPrevious,0)-1),0).rgb;
    return pow(clamp(1.0-exp(-max(c,vec3(0.0))*1.2),0.0,1.0),vec3(1.0/2.2));
}
void main() {
    ivec2 p=ivec2(gl_FragCoord.xy)-1;
    vec3 g=vec3(0.0);
    for(int y=-uRadius;y<=1+uRadius;y++) for(int x=-uRadius;x<=1+uRadius;x++) {
        vec3 a=color(p+ivec2(x,y));
        if(x<1+uRadius) g=max(g,abs(a-color(p+ivec2(x+1,y))));
        if(y<1+uRadius) g=max(g,abs(a-color(p+ivec2(x,y+1))));
    }
    outColor=vec4(max(g.x,max(g.y,g.z)),0.0,0.0,1.0);
}
