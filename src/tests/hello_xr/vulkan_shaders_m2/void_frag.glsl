#version 450
layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 oColor;

layout(push_constant) uniform VoidPushConstants {
    mat4 uInvMvp;
    vec4 uSky;
    vec4 uGround;
    vec2 uOutSize;
};

void main() {
    vec2 ndc = gl_FragCoord.xy / uOutSize * 2.0 - 1.0;
    vec4 a = uInvMvp * vec4(ndc, -1.0, 1.0);
    vec4 b = uInvMvp * vec4(ndc,  1.0, 1.0);
    vec3 dir = b.xyz / b.w - a.xyz / a.w;
    float up = dir.y / max(length(dir), 1e-6);
    oColor = vec4(mix(uGround.xyz, uSky.xyz, smoothstep(-0.03, 0.03, up)), 1.0);
}
