#version 450
#extension GL_GOOGLE_include_directive : require
// Pass 3: text over polygons at priority 6, then the gamma tables (port of kCompositeFs), into the eye image.
#include "s22_common.glsl"
#include "s22_mix.glsl"
layout(location = 0) out vec4 oColor;
void main() {
    vec2 uv = gl_FragCoord.xy / OutText.xy;
    ivec2 p;
    bool hasText = textTexel(p);
    uvec3 c = uvec3(textureLod(SceneColor, uv, 0.0).rgb * 255.0 + 0.5);
    uint pri = (hasText ? (priAt(p) & 4u) : 0u) | uint(texelFetch(ScenePri, ivec2(uv * vec2(textureSize(ScenePri, 0))), 0).r * 255.0 + 0.5);
    if (pri == 6u) c = mixText(p, c, 6);
    uvec3 g = uvec3(gammaAt(c.r), gammaAt(256u + c.g), gammaAt(512u + c.b));
    oColor = vec4(vec3(g) / 255.0, 1.0);
}
