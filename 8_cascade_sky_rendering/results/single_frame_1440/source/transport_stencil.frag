#version 430 core
uniform sampler2D uMask;
uniform int uKind;
void main() {
    float m=texelFetch(uMask,ivec2(gl_FragCoord.xy),0).x;
    if(uKind==1 ? (m<0.5 || m>0.8) : m<0.8) discard;
}
