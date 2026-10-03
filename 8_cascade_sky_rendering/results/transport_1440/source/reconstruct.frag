#version 430 core
// Linear HDR reconstruction; gradients use the same display transform as evaluation.
uniform sampler2D uPrevious;
uniform vec2 uSize;
uniform float uThreshold;
uniform int uRadius;
uniform int uForce; // -1: interpolate everything, 0: gradient, +1: shade everything
layout(location=0) out vec4 outColor;
layout(location=1) out float outMask;

vec3 displayColor(vec3 c) {
    return pow(clamp(1.0-exp(-max(c, vec3(0.0))*1.2), 0.0, 1.0), vec3(1.0/2.2));
}
vec3 parentColor(ivec2 p) {
    return displayColor(texelFetch(uPrevious,
        clamp(p, ivec2(0), textureSize(uPrevious, 0)-1), 0).rgb);
}
vec4 parentHDR(ivec2 p) {
    return texelFetch(uPrevious, clamp(p, ivec2(0), textureSize(uPrevious, 0)-1), 0);
}
void main() {
    vec2 uv = gl_FragCoord.xy/uSize;
    vec2 coord = uv*vec2(textureSize(uPrevious, 0))-0.5;
    ivec2 p = ivec2(floor(coord));
    vec2 f = fract(coord);
    // Explicit float interpolation avoids hardware sub-texel weight quantization
    // at sharp HDR transitions, and makes arbitrary-size CPU parity meaningful.
    outColor = mix(mix(parentHDR(p), parentHDR(p+ivec2(1,0)), f.x),
                   mix(parentHDR(p+ivec2(0,1)), parentHDR(p+ivec2(1,1)), f.x), f.y);
    if (uForce != 0) {
        outMask = uForce > 0 ? 1.0 : 0.0;
        return;
    }
    // Compare adjacent samples covering the interpolation footprint, with an
    // optional parent-pixel halo. This is NOT an upper bound on unseen detail.
    float g = 0.0;
    for (int y = -uRadius; y <= 1+uRadius; ++y) {
        for (int x = -uRadius; x <= 1+uRadius; ++x) {
            vec3 a = parentColor(p+ivec2(x,y));
            if (x < 1+uRadius) {
                vec3 d = abs(a-parentColor(p+ivec2(x+1,y)));
                g = max(g, max(d.x, max(d.y,d.z)));
            }
            if (y < 1+uRadius) {
                vec3 d = abs(a-parentColor(p+ivec2(x,y+1)));
                g = max(g, max(d.x, max(d.y,d.z)));
            }
        }
    }
    outMask = g > uThreshold ? 1.0 : 0.0;
}
