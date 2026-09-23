#version 450
#extension GL_GOOGLE_include_directive : require
// Pass 3 as the SECOND SUBPASS of the scene render pass (full-resolution immersive): the resolved scene
// colour and priority are read on-tile as input attachments, never stored to memory nor sampled back.
// Same logic as s22_comp_frag (text over polygons at priority 6, then the gamma tables).
#include "s22_common.glsl"
#include "s22_mix.glsl"
layout(input_attachment_index = 0, set = 0, binding = 17) uniform subpassInput InColor;
layout(input_attachment_index = 1, set = 0, binding = 18) uniform subpassInput InPri;
layout(location = 0) out vec4 oColor;
void main() {
    ivec2 p;
    bool hasText = textTexel(p);
    vec2 q; bool onPlane = textCoord(q);
    vec2 tpp = abs(dFdx(q)) + abs(dFdy(q));
    uvec3 c = uvec3(subpassLoad(InColor).rgb * 255.0 + 0.5);
    uint scenePri = uint(subpassLoad(InPri).r * 255.0 + 0.5);
    if (Mix2.z != 0u && onPlane) c = uvec3(mixTextSharp(q, tpp, c, 6, scenePri, 6u) + 0.5);
    else {
        uint pri = (hasText ? (priAt(p) & 4u) : 0u) | scenePri;
        if (pri == 6u) c = mixText(p, c, 6);
    }
    uvec3 g = uvec3(gammaAt(c.r), gammaAt(256u + c.g), gammaAt(512u + c.b));
    oColor = vec4(outColor(vec3(g) / 255.0), 1.0);
}
