#version 450
#extension GL_GOOGLE_include_directive : require
// Pass 1: what the CPU frame holds before the first polygon (background + text at priority 4).
#include "s22_common.glsl"
#include "s22_mix.glsl"
layout(location = 0) out vec4 oColor;
layout(location = 1) out vec4 oPri;
void main() {
    ivec2 p;
    bool hasText = textTexel(p);
    uvec3 c = Bg.xyz;
    if (hasText && (priAt(p) & 4u) != 0u) c = mixText(p, c, 4);
    oColor = vec4(vec3(c) / 255.0, 1.0);
    oPri = vec4(0.0);
}
