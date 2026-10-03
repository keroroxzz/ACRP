#version 430 core
uniform sampler2D uMask;
void main() {
    if (texelFetch(uMask, ivec2(gl_FragCoord.xy), 0).r < 0.5) discard;
}
