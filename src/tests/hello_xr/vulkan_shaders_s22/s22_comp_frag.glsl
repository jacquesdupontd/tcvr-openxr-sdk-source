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
    vec3 cur = textureLod(SceneColor, vec3(uv, float(Sprite.z)), 0.0).rgb;
    if (Sprite.w != 0) {
        // Port of the GL dark-flicker pass: where a pixel is dark in one of the two last arcade frames and
        // changes a lot, show their average (what a CRT's phosphor integrated), e.g. Dirt Dash car shadows.
        vec3 prev = textureLod(SceneColor, vec3(uv, float(1 - Sprite.z)), 0.0).rgb;
        float darkest = min(dot(cur, vec3(0.299, 0.587, 0.114)), dot(prev, vec3(0.299, 0.587, 0.114)));
        vec3 d = abs(cur - prev);
        float change = max(max(d.r, d.g), d.b);
        float integrate = (1.0 - smoothstep(0.08, 0.18, darkest)) * smoothstep(0.12, 0.28, change);
        cur = mix(cur, (cur + prev) * 0.5, integrate);
    }
    uvec3 c = uvec3(cur * 255.0 + 0.5);
    uint pri = (hasText ? (priAt(p) & 4u) : 0u) | uint(texelFetch(ScenePri, ivec2(uv * vec2(textureSize(ScenePri, 0))), 0).r * 255.0 + 0.5);
    if (pri == 6u) c = mixText(p, c, 6);
    uvec3 g = uvec3(gammaAt(c.r), gammaAt(256u + c.g), gammaAt(512u + c.b));
    oColor = vec4(vec3(g) / 255.0, 1.0);
}
